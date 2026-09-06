"""L36a: the software-pipelined prefill scan, proved on CPU.

The kernel change this file guards cannot be checked by running it here -- it is
Metal, and this box has no GPU in the desk configuration -- so what is checked
instead is the two things a GPU run would only *sample*:

  1. VALUE IDENTITY, symbolically.  Both schedules are re-executed in Python
     over a shared expression DAG: every arithmetic step becomes a node keyed by
     (op, operand-node-ids), interned in one global table shared by the two
     runs.  Two values are the same node if and only if they were produced by
     the same operation applied to the same operands in the same order.  So
     asserting that the pipelined run's y / state / conv-window nodes are the
     SAME INTEGERS as the baseline's is a proof of bit-identity that is
     insensitive to nothing: any reassociation, any reordering of a reduction's
     operands, any change of which lane accumulates what, mints a new node and
     the assertion fails.  (A negative control below perturbs one operand order
     and shows the assertion does fire.)

  2. SCHEDULE SAFETY.  The same re-execution runs through a threadgroup-memory
     model that tags every cell with the epoch (barrier count) and the tid that
     last wrote it, and rejects a read whose value was written in the SAME epoch
     by a DIFFERENT thread (read-before-write across the double buffer) and a
     write to a cell read in the same epoch by a different thread (clobber).
     A negative control collapses the double buffer to a single one and shows
     the model reports the hazard rather than passing.

Everything else here is a pin: the baseline generated source is hashed so the
"the reduction bodies are the same text" claim cannot rot, and the shared
snippets are asserted to occur verbatim in both sources.
"""
import hashlib

import mlx.core as mx
import pytest

import mlx_vlm.models.glm5_next.fused_kda_prefill as F

# ---------------------------------------------------------------- source pins
# sha256 of the generated Metal source at a6634a75, the revision whose
# bit-exactness is already gated (L23).  The pipelined variant is only allowed
# to claim "same reduction bodies" for as long as these do not move.
_BASELINE_SHA = {
    True: "718cf60f48d2cc408b33dae25f52494ea69379d62d3aaab9206fbddd95687f7f",
    False: "1750583b8d51114ce14c428f85e559c70c64cb9c33a64d0f5fe39285daba4bc7",
}


def _norm(text):
    """Whitespace-insensitive, so indentation may differ but tokens may not."""
    return " ".join(text.split())


@pytest.mark.parametrize("fuse_norm", [True, False])
def test_the_baseline_kernel_source_is_untouched(fuse_norm):
    got = hashlib.sha256(F._make_scan_source(fuse_norm).encode()).hexdigest()
    assert got == _BASELINE_SHA[fuse_norm], (
        "the non-pipelined scan changed; L36a's bit-identity argument is stated "
        "against a6634a75's source and has to be re-derived"
    )


def test_every_reduction_body_is_the_same_text_in_both_variants():
    """The q/k L2 norm, the rescale and the gated RMSNorm are shared strings.

    This is the load-bearing half of "the arithmetic order is unchanged": not an
    argument that two hand-written bodies agree, but the observation that there
    is one body.  The RMSNorm differs by exactly one character sequence --
    ``shr[0]`` becomes ``shr[3]``, because in the pipeline the norm scalar of
    token t-2 has to coexist with the L2 scalars of token t in the same parity
    slot -- and that is a destination, not an operand.
    """
    base_f = _norm(F._make_scan_source(True))
    base_s = _norm(F._make_scan_source(False))
    pipe_f = _norm(F._make_scan_source_pipelined(True))
    pipe_s = _norm(F._make_scan_source_pipelined(False))

    l2 = _norm(F._L2_BODY)
    rescale = _norm(F._RESCALE_BODY)
    rms = _norm(F._RMS_BODY)
    for src in (base_f, base_s, pipe_f, pipe_s):
        assert l2 in src
        assert rescale in src
    assert rms in base_f
    assert _norm(F._RMS_BODY.replace("shr[0]", "shr[3]")) in pipe_f
    # ... and the split variants have no in-kernel norm at all, in both.
    assert rms not in base_s and rms not in pipe_s


def test_the_pipelined_variant_removes_five_of_the_seven_barriers():
    """7 threadgroup barriers per token -> 2, and the seed barrier stays."""
    for fuse_norm in (True, False):
        base = F._make_scan_source(fuse_norm)
        pipe = F._make_scan_source_pipelined(fuse_norm)
        n_base = base.count("threadgroup_barrier")
        n_pipe = pipe.count("threadgroup_barrier")
        # the baseline's per-token barriers, plus the window-seed one; the fused
        # variant carries the two the in-kernel RMSNorm needs.
        assert n_base == (7 if fuse_norm else 5) + 1
        # the pipelined body is emitted twice (the token loop is unrolled by two
        # so the buffer parity is a compile-time constant), so 2 per token shows
        # up as 4 in the source, plus the same window-seed barrier.
        assert n_pipe == 2 * 2 + 1


def test_the_pipeline_is_off_by_default_and_the_env_flag_turns_it_on(monkeypatch):
    monkeypatch.setattr(F, "_PIPELINE_ENV", None)
    monkeypatch.delenv("MLX_VLM_GLM5_FUSED_KDA_PREFILL_PIPELINE", raising=False)
    assert F._pipeline_enabled() is False
    monkeypatch.setattr(F, "_PIPELINE_ENV", None)
    monkeypatch.setenv("MLX_VLM_GLM5_FUSED_KDA_PREFILL_PIPELINE", "1")
    assert F._pipeline_enabled() is True


# --------------------------------------------------------------------------- #
# Dispatch: which kernel the flag actually selects.
# --------------------------------------------------------------------------- #
H_, D_, K_ = 4, 128, 4


class _FakeKernel:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __call__(self, *, inputs, template, grid, threadgroup, output_shapes,
                 output_dtypes):
        self.log.append(dict(name=self.name, template=dict(template), grid=grid))
        return [mx.zeros(s, dtype=d) for s, d in zip(output_shapes, output_dtypes)]


@pytest.fixture
def fake(monkeypatch):
    log = []
    monkeypatch.setattr(F, "_KERNEL_TRIED", True)
    monkeypatch.setattr(
        F, "_KERNELS",
        {k: _FakeKernel(k, log) for k in
         ("fused", "split", "norm", "fused_pipe", "split_pipe",
          "fused_bar", "split_bar")},
    )
    return log


def _call(nv, ty=32, pipeline=None):
    r = lambda *sh: mx.zeros(sh, mx.bfloat16)  # noqa: E731
    return F.fused_kda_prefill(
        r(1, 7, H_ * D_), r(1, 7, H_ * D_), r(1, 7, H_ * D_),
        r(1, K_ - 1, 3 * H_ * D_), r(3 * H_ * D_, K_, 1), r(1, 7, H_ * D_),
        r(1, 7, H_), mx.zeros((H_,), mx.float32), mx.zeros((H_ * D_,), mx.float32),
        mx.zeros((1, H_, D_, D_), mx.float32), r(1, 7, H_ * D_), r(D_),
        num_heads=H_, head_dim=D_, conv_kernel_size=K_, lower_bound=-5.0,
        norm_eps=1e-5, nv=nv, ty=ty, pipeline=pipeline,
    )


@pytest.mark.parametrize("nv,names", [(1, ["fused"]), (4, ["split", "norm"])])
def test_pipeline_off_keeps_the_shipped_kernels(nv, names, fake):
    _call(nv, pipeline=0)
    assert [c["name"] for c in fake] == names


@pytest.mark.parametrize("nv,names", [(1, ["fused_pipe"]), (4, ["split_pipe", "norm"])])
def test_pipeline_on_selects_the_pipelined_scan(nv, names, fake):
    _call(nv, pipeline=1)
    assert [c["name"] for c in fake] == names


@pytest.mark.parametrize("nv,names", [(1, ["fused_bar"]), (4, ["split_bar", "norm"])])
def test_mode_two_selects_the_barrier_only_ablation(nv, names, fake):
    _call(nv, pipeline=2)
    assert [c["name"] for c in fake] == names


def test_the_env_flag_distinguishes_the_two_pipelined_modes(monkeypatch):
    for v, want in (("0", 0), ("1", 1), ("2", 2), ("on", 1), ("off", 0)):
        monkeypatch.setattr(F, "_PIPELINE_ENV", None)
        monkeypatch.setenv("MLX_VLM_GLM5_FUSED_KDA_PREFILL_PIPELINE", v)
        assert F._pipeline_mode() == want, v


def test_ty1_falls_back_because_there_is_no_worker_simdgroup_left(fake):
    """Simdgroup 0 services the reductions; at TY=1 it is the only simdgroup, so
    the pipeline has no one to overlap with.  The probe degrades TY on devices
    whose threadgroup limit is low, so this path is reachable in production and
    must fall back rather than launch a kernel with NWG == 0."""
    _call(1, ty=1, pipeline=1)
    assert [c["name"] for c in fake] == ["fused"]


# --------------------------------------------------------------------------- #
# The symbolic re-execution.
# --------------------------------------------------------------------------- #
class Hazard(AssertionError):
    pass


class Terms:
    """Interned expression DAG.  One table, both schedules."""

    def __init__(self):
        self.t = {}

    def __call__(self, *parts):
        return self.t.setdefault(parts, len(self.t))


class Mem:
    """Threadgroup memory with epoch/owner tags."""

    def __init__(self):
        self.cell, self.reads, self.epoch = {}, {}, 0

    def barrier(self):
        self.epoch += 1

    def write(self, addr, val, tid):
        for e, th in self.reads.get(addr, ()):
            if e == self.epoch and th != tid:
                raise Hazard(f"WAR on {addr}: tid {tid} clobbers tid {th}'s epoch")
        self.cell[addr] = (val, self.epoch, tid)
        self.reads[addr] = []
        return val

    def read(self, addr, tid):
        if addr not in self.cell:
            raise Hazard(f"read before write: {addr}")
        val, e, th = self.cell[addr]
        if e == self.epoch and th != tid:
            raise Hazard(f"RAW inside epoch {e} on {addr}: tid {tid} vs writer {th}")
        self.reads.setdefault(addr, []).append((self.epoch, tid))
        return val


D, K, TY, NV = 128, 4, 32, 1
NT, NDK, KM1 = 32 * TY, D // 32, K - 1
RBLK, REXTRA = D // 128, D - (D // 128) * 128


def _stage(T, m, t, S, par, *, single_buffer=False):
    """conv -> silu -> gates -> beta, for token t, into parity `par`."""
    p = 0 if single_buffer else par
    for idx in range(3 * D):
        tid = idx % NT
        part, d = idx // D, idx % D
        acc = T("zero")
        for j in range(K - 1):
            slot = (t + j) % KM1
            acc = T("addmul", acc, m.read(("twin", slot, idx), tid), T("wc", idx, j))
        xnew = T("xnew", t, part, d)
        acc = T("addmul", acc, xnew, T("wc", idx, K - 1))
        m.write(("twin", t % KM1, idx), xnew, tid)
        xb = T("castT", acc)
        sl = T("mul", xb, T("sigfast", xb))
        m.write((("sq", "sk", "sv")[part], p, d), T("float", sl), tid)
    for d in range(D):
        m.write(("sg", p, d), T("safegate", t, d), d % NT)
    m.write(("shr", p, 2), T("beta", t), 0)


def _l2(T, m, par, *, single_buffer=False):
    """MLX's row_reduce partition, one simdgroup, verbatim operand order."""
    p = 0 if single_buffer else par
    for lane in range(32):
        pq = pk = T("zero")
        for blk in range(RBLK):
            base = blk * 128 + 4 * lane
            for i in range(4):
                pq = T("sq_acc", pq, m.read(("sq", p, base + i), lane))
                pk = T("sq_acc", pk, m.read(("sk", p, base + i), lane))
        base = RBLK * 128 + 4 * lane
        rng = range(4) if 4 * lane + 4 <= REXTRA else range(max(0, REXTRA - 4 * lane))
        for i in rng:
            pq = T("sq_acc", pq, m.read(("sq", p, base + i), lane))
            pk = T("sq_acc", pk, m.read(("sk", p, base + i), lane))
        if lane == 0:
            sq_sum, sk_sum = T("simd_sum", pq), T("simd_sum", pk)
    m.write(("shr", p, 0), T("rsqrt_eps", sq_sum), 0)
    m.write(("shr", p, 1), T("rsqrt_eps", sk_sum), 0)


def _rescale(T, m, par, *, single_buffer=False):
    p = 0 if single_buffer else par
    for d in range(D):
        tid = d % NT
        rq, rk = m.read(("shr", p, 0), tid), m.read(("shr", p, 1), tid)
        m.write(("sq", p, d),
                T("castT_qscale", m.read(("sq", p, d), tid), rq), tid)
        m.write(("sk", p, d), T("castT", m.read(("sk", p, d), tid), rk), tid)


def _phase1(T, m, rows, par, st, *, single_buffer=False, swap_operands=False):
    """rows: {simdgroup -> [value rows, in the order that simdgroup walks them]}"""
    p = 0 if single_buffer else par
    for sg, dvs in rows.items():
        for dv in dvs:
            tid0 = 32 * sg
            beta = m.read(("shr", p, 2), tid0)
            kv = {}
            for lane in range(32):
                acc = T("zero")
                for i in range(NDK):
                    s = NDK * lane + i
                    st[dv][s] = T("mul", st[dv][s], m.read(("sg", p, s), 32 * sg + lane))
                    a, b = st[dv][s], m.read(("sk", p, s), 32 * sg + lane)
                    acc = T("addmul", acc, *((b, a) if swap_operands else (a, b)))
                kv[lane] = acc
            kvs = T("simd_sum", *(kv[l] for l in range(32)))
            delta = T("mul", T("sub", m.read(("sv", p, dv), tid0), kvs), beta)
            o = {}
            for lane in range(32):
                acc = T("zero")
                for i in range(NDK):
                    s = NDK * lane + i
                    st[dv][s] = T("addmul", st[dv][s],
                                  m.read(("sk", p, s), 32 * sg + lane), delta)
                    acc = T("addmul", acc, st[dv][s],
                            m.read(("sq", p, s), 32 * sg + lane))
                o[lane] = acc
            m.write(("sy", p, dv), T("floatcastT", T("simd_sum", *(o[l] for l in range(32)))), tid0)


def _rms(T, m, par, slot, *, single_buffer=False):
    p = 0 if single_buffer else par
    for lane in range(32):
        po = T("zero")
        for blk in range(RBLK):
            base = blk * 128 + 4 * lane
            for i in range(4):
                po = T("sq_acc", po, m.read(("sy", p, base + i), lane))
        base = RBLK * 128 + 4 * lane
        rng = range(4) if 4 * lane + 4 <= REXTRA else range(max(0, REXTRA - 4 * lane))
        for i in rng:
            po = T("sq_acc", po, m.read(("sy", p, base + i), lane))
        if lane == 0:
            tot = T("simd_sum", po)
    m.write(("shr", p, slot), T("rsqrt_mean", tot), 0)


def _writeout(T, m, t, par, slot, out, *, single_buffer=False):
    p = 0 if single_buffer else par
    for d in range(D):
        tid = d % NT
        rn = m.read(("shr", p, slot), tid)
        x = T("mul", m.read(("sy", p, d), tid), rn)
        x = T("mul", T("o_w", d), x)
        out[(t, d)] = T("castT", T("mul", x, T("sigprecise", T("gate", t, d))))


def _seed(T, m):
    for slot in range(KM1):
        for idx in range(3 * D):
            m.write(("twin", slot, idx), T("cachetap", slot, idx), idx % NT)
    m.barrier()


def run_baseline(T, S):
    m, out = Mem(), {}
    st = {dv: {s: T("state0", dv, s) for s in range(D)} for dv in range(D)}
    _seed(T, m)
    rows = {sg: [sg + TY * j for j in range(D // TY)] for sg in range(TY)}
    for t in range(S):
        _stage(T, m, t, S, 0)          # the baseline's conv/window barrier is
        m.barrier()                    # inside _stage's span; keep it explicit
        m.barrier()
        _l2(T, m, 0)
        m.barrier()
        _rescale(T, m, 0)
        m.barrier()
        _phase1(T, m, rows, 0, st)
        m.barrier()
        _rms(T, m, 0, 0)
        m.barrier()
        _writeout(T, m, t, 0, 0, out)
        m.barrier()
    return out, st, m


def run_pipelined(T, S, *, single_buffer=False, swap_operands=False):
    m, out = Mem(), {}
    st = {dv: {s: T("state0", dv, s) for s in range(D)} for dv in range(D)}
    _seed(T, m)
    nwg = TY - 1
    ndv = -(-D // nwg)
    rows = {sg: [(sg - 1) + nwg * j for j in range(ndv) if (sg - 1) + nwg * j < D]
            for sg in range(1, TY)}
    kw = dict(single_buffer=single_buffer)
    for t in range(S + 2):
        pp = t & 1
        if t < S:
            _stage(T, m, t, S, pp, **kw)
        m.barrier()
        if t < S:
            _l2(T, m, pp, **kw)
        if t >= 2 and t - 2 < S:
            _rms(T, m, pp, 3, **kw)
        if 1 <= t <= S:
            _phase1(T, m, rows, pp ^ 1, st, swap_operands=swap_operands, **kw)
        m.barrier()
        if t < S:
            _rescale(T, m, pp, **kw)
        if t >= 2 and t - 2 < S:
            _writeout(T, m, t - 2, pp, 3, out, **kw)
    return out, st, m


_S = 6


def test_pipelined_schedule_is_value_identical_to_the_baseline():
    """Same DAG nodes for every output element, every state element.

    Because the table is shared, equal integers mean the two schedules built the
    same expression out of the same leaves with the same operand order -- which
    is exactly the property `atol = rtol = 0` needs and the property a
    reassociated reduction would break.
    """
    T = Terms()
    out_b, st_b, mb = run_baseline(T, _S)
    out_p, st_p, mp = run_pipelined(T, _S)
    assert out_p == out_b, "write-out expression differs"
    assert st_p == st_b, "recurrent state expression differs"
    # and the conv window the epilogue stores
    for slot in range(KM1):
        for idx in range(3 * D):
            a = mb.cell[("twin", (_S + slot) % KM1, idx)][0]
            b = mp.cell[("twin", (_S + slot) % KM1, idx)][0]
            assert a == b, "conv window differs"


def test_the_pipelined_schedule_has_no_cross_thread_hazard():
    """No read-before-write and no clobber across the double buffers."""
    run_pipelined(Terms(), _S)          # raises Hazard if either occurs


def test_the_baseline_schedule_also_passes_the_hazard_model():
    """Non-vacuity, half one: the model does not merely reject everything."""
    run_baseline(Terms(), _S)


def test_a_single_buffered_pipeline_is_rejected_by_the_hazard_model():
    """Non-vacuity, half two: collapse the double buffer and the model fires.

    This is the failure the whole design exists to avoid -- token t's stage
    overwriting the q/k/v the workers are still reading for token t-1 -- so a
    model that did not catch it would be checking nothing.
    """
    with pytest.raises(Hazard):
        run_pipelined(Terms(), _S, single_buffer=True)


def test_swapping_one_reduction_operand_order_is_caught():
    """Non-vacuity, half three: the DAG comparison is sensitive to operand order.

    `kv += st * sk` written as `sk * st` is the same real number and (on this
    hardware) the same float, but it is NOT the same text, and the claim being
    made is textual identity of the arithmetic.  If the comparison did not fire
    here it could not detect a genuine reassociation either.
    """
    T = Terms()
    out_b, st_b, _ = run_baseline(T, _S)
    out_p, st_p, _ = run_pipelined(T, _S, swap_operands=True)
    assert (out_p, st_p) != (out_b, st_b)


def test_every_value_row_is_owned_by_exactly_one_worker_simdgroup():
    """The redistribution over TY-1 workers must be a partition of [0, D): a
    dropped row is a silently wrong state, a duplicated one is a double update.
    Rows are independent -- `kv` and `o` are simd_sum-ed within ONE simdgroup
    over the 32 key lanes -- so which worker owns a row cannot change any bit,
    but whether it is owned exactly once can change every bit.
    """
    nwg = TY - 1
    ndv = -(-D // nwg)
    owned = [(sg - 1) + nwg * j for sg in range(1, TY) for j in range(ndv)
             if (sg - 1) + nwg * j < D]
    assert sorted(owned) == list(range(D))
    assert ndv * nwg >= D and (ndv - 1) * nwg < D, "NDV is not the tight ceiling"


def test_the_key_axis_partition_is_untouched():
    """`lane` still owns key elements [NDK*lane, NDK*lane+NDK) -- the invariant
    the simd_sum inherits from gated_delta_kernel."""
    seen = [s for lane in range(32) for s in
            (NDK * lane + i for i in range(NDK))]
    assert seen == list(range(D))
