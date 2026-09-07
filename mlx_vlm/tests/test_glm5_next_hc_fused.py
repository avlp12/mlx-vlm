"""L34: hand-written Metal kernels for the two element-wise ends of the
hyper-connection cycle -- the pre-norm in front of the mixing matmul and the
residual inject (``hc_expand``) behind the block.

Both live behind ``MLX_VLM_GLM5_HC_FUSED`` (the pre-norm additionally behind
``MLX_VLM_GLM5_HC_PRENORM_FUSED``).  Both are DEFAULT ON since I1450 -- unset
or empty means ON, and an explicit "0"/"false"/"no"/"off" is the opt-OUT that
restores the eager/compiled path of 5cd9d8b5.  Both are still skipped below
``MLX_VLM_GLM5_HC_FUSED_MIN_ROWS`` (512) rows, so decode keeps the path it was
tuned on no matter what the flags say.

Everything here runs on CPU.  ``mx.fast.metal_kernel`` cannot execute without
Metal, so what CPU can cover is: (a) the flag/eligibility logic, (b) that the
fallback is the untouched eager path, and (c) the NUMERICS, by comparing the
eager mx reference against a numpy re-implementation of the exact arithmetic
the kernel source performs (element order, accumulation order, fp32 widening).
Kernel *compilation* is exercised only when Metal is present.
"""

import numpy as np
import mlx.core as mx
import pytest

from mlx_vlm.models.deepseek_v4 import hyper_connection as hc


# --------------------------------------------------------------------- utils
# The reset hook lives in the module under test (production owns the contract
# that these two caches are the whole of the flag state); this is the alias the
# tests below have always called.
_reset_flag_cache = hc._reset_flag_cache


@pytest.fixture(autouse=True)
def _cpu_and_clean_flags(monkeypatch):
    # Three independent guarantees, all needed:
    #  - `mx.default_device` is patched so the module's own `_metal_ok()` gate
    #    reads CPU regardless of how pytest was invoked;
    #  - the two kernel handles are forced to None so that NOTHING in this file
    #    can dispatch a Metal kernel even on a box where Metal is present and
    #    MLX_DEFAULT_DEVICE was not exported.  Tests that want to exercise the
    #    eligibility predicate re-patch a sentinel in place of the handle;
    #  - both flags are DELETED from the environment.  Since I1450 unset means
    #    ON, so an operator who exported ``MLX_VLM_GLM5_HC_FUSED=0`` in the
    #    shell that launched pytest would otherwise silently invert every
    #    default-path assertion in this file.
    monkeypatch.setattr(mx, "default_device", lambda: mx.cpu)
    monkeypatch.setattr(hc, "_hc_expand_kernel", None)
    monkeypatch.setattr(hc, "_hc_prenorm_kernel", None)
    monkeypatch.delenv("MLX_VLM_GLM5_HC_FUSED", raising=False)
    monkeypatch.delenv("MLX_VLM_GLM5_HC_PRENORM_FUSED", raising=False)
    _reset_flag_cache()
    yield
    _reset_flag_cache()


class _PathSpy:
    """Records which of the two ``hc_expand`` implementations actually ran.

    Both are looked up as module globals inside ``hc_expand``, so replacing the
    module attributes intercepts the real dispatch rather than a copy of it.
    """

    def __init__(self, monkeypatch):
        self.calls = []
        # Captured BEFORE the patch, so a test can still reach the genuine
        # 5cd9d8b5 implementation to compare against.
        self.real_op = real_op = hc._hc_expand_op

        def op(*a, **kw):
            self.calls.append("eager")
            return real_op(*a, **kw)

        def fused(*a, **kw):
            self.calls.append("fused")
            # Never actually dispatched on CPU: the sentinel kernel handle the
            # eligibility check accepted is not callable.  Shape-correct stand-in.
            x, residual, post, comb = a
            return mx.zeros(residual.shape, dtype=x.dtype)

        monkeypatch.setattr(hc, "_hc_expand_op", op)
        monkeypatch.setattr(hc, "_hc_expand_fused", fused)

    @property
    def which(self):
        assert len(self.calls) == 1, f"expected one dispatch, got {self.calls}"
        return self.calls[0]


def np_expand_kernel(x, residual, post, comb):
    """Exact arithmetic of ``_HC_EXPAND_SOURCE``.

    out[h] = post[h] * x + (((c[0][h]*r[0] + c[1][h]*r[1]) + ...) ), the j sum
    accumulating in ascending j and the post*x term added last, all in fp32
    with the inputs widened from the storage dtype.
    """
    B, L, D = x.shape
    H = comb.shape[-1]
    out = np.empty((B, L, H, D), np.float32)
    for h in range(H):
        acc = comb[:, :, 0, h, None] * residual[:, :, 0, :]
        for j in range(1, H):
            acc = acc + comb[:, :, j, h, None] * residual[:, :, j, :]
        out[:, :, h, :] = post[:, :, h, None] * x + acc
    return out


def _simd_sum(vals):
    """Butterfly reduction over 32 lanes, the shape ``simd_sum`` lowers to."""
    v = np.array(vals, np.float32)
    off = 1
    while off < 32:
        v = v + v[np.arange(32) ^ off]
        off *= 2
    return v[0]


def np_prenorm_kernel(x, eps, nthreads=256):
    """Exact arithmetic of ``_HC_PRENORM_SOURCE`` for one row."""
    B, L, H, D = x.shape
    N = H * D
    rows = x.reshape(-1, N).astype(np.float32)
    out = np.empty_like(rows)
    per_thread = np.zeros((rows.shape[0], nthreads), np.float32)
    for lid in range(nthreads):
        acc = np.zeros(rows.shape[0], np.float32)
        for r in range(0, N, nthreads * 4):
            o = r + lid * 4
            for i in range(4):
                xi = rows[:, o + i]
                acc = acc + xi * xi
        per_thread[:, lid] = acc
    sg = nthreads // 32
    for row in range(rows.shape[0]):
        parts = [
            _simd_sum(per_thread[row, g * 32 : (g + 1) * 32]) for g in range(sg)
        ]
        total = np.float32(0.0)
        for pval in parts:
            total = np.float32(total + pval)
        inv = np.float32(1.0) / np.sqrt(np.float32(total / np.float32(N)) + np.float32(eps))
        out[row] = rows[row] * inv
    return out.reshape(B, L, N)


def _mk(B, L, H, D, dtype=mx.float32, seed=0):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((B, L, D)).astype(np.float32)).astype(dtype)
    residual = mx.array(rng.standard_normal((B, L, H, D)).astype(np.float32)).astype(dtype)
    post = mx.array(rng.random((B, L, H)).astype(np.float32) * 2)
    comb = mx.array(rng.random((B, L, H, H)).astype(np.float32))
    return x, residual, post, comb


# ------------------------------------------------------------------- flags
def test_hc_fused_default_on(monkeypatch):
    """I1450 promotion, inverted from the 5cd9d8b5 assertion: UNSET is ON."""
    monkeypatch.delenv("MLX_VLM_GLM5_HC_FUSED", raising=False)
    monkeypatch.delenv("MLX_VLM_GLM5_HC_PRENORM_FUSED", raising=False)
    _reset_flag_cache()
    assert hc._hc_fused_enabled() is True
    assert hc._hc_prenorm_fused_enabled() is True


@pytest.mark.parametrize("value", ["", " ", "1", "true", "TRUE", "yes", "on", "junk"])
def test_hc_fused_env_enables(monkeypatch, value):
    """Everything that is not an explicit off-word is ON -- the empty string in
    particular, which is what a launcher emits for a variable it did not set."""
    monkeypatch.setenv("MLX_VLM_GLM5_HC_FUSED", value)
    monkeypatch.setenv("MLX_VLM_GLM5_HC_PRENORM_FUSED", value)
    _reset_flag_cache()
    assert hc._hc_fused_enabled() is True
    assert hc._hc_prenorm_fused_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " 0 "])
def test_hc_fused_env_opts_out(monkeypatch, value):
    monkeypatch.setenv("MLX_VLM_GLM5_HC_FUSED", value)
    _reset_flag_cache()
    assert hc._hc_fused_enabled() is False
    # The master flag drags the pre-norm down with it even though the pre-norm
    # flag itself is unset (= ON).
    assert hc._hc_prenorm_fused_enabled() is False


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_prenorm_opts_out_alone(monkeypatch, value):
    """The pre-norm can be dropped on its own, leaving the expand kernel on --
    that is the whole reason it has a second flag."""
    monkeypatch.setenv("MLX_VLM_GLM5_HC_PRENORM_FUSED", value)
    _reset_flag_cache()
    assert hc._hc_fused_enabled() is True
    assert hc._hc_prenorm_fused_enabled() is False


def test_prenorm_still_gated_by_master_flag(monkeypatch):
    """Replaces ``test_prenorm_requires_both_flags``: the AND is unchanged, only
    the default each side of it starts from."""
    monkeypatch.setenv("MLX_VLM_GLM5_HC_PRENORM_FUSED", "1")
    monkeypatch.setenv("MLX_VLM_GLM5_HC_FUSED", "0")
    _reset_flag_cache()
    assert hc._hc_prenorm_fused_enabled() is False
    monkeypatch.setenv("MLX_VLM_GLM5_HC_FUSED", "1")
    _reset_flag_cache()
    assert hc._hc_prenorm_fused_enabled() is True


def test_flag_cache_is_read_once(monkeypatch):
    """Serving reads os.environ once per process; the reset hook is the only
    way back.  Pins the hook the tests above depend on."""
    _reset_flag_cache()
    assert hc._hc_fused_enabled() is True
    monkeypatch.setenv("MLX_VLM_GLM5_HC_FUSED", "0")
    assert hc._hc_fused_enabled() is True      # cached
    _reset_flag_cache()
    assert hc._hc_fused_enabled() is False     # re-read


# ------------------------------------------------------------- eligibility
def test_expand_eligibility_is_false_without_metal():
    # No kernel object -> every call falls back, whatever the shape.
    x, residual, post, comb = _mk(1, 1024, 4, 64)
    assert hc._hc_expand_kernel is None
    assert hc.hc_expand_fused_eligible(x, residual, post, comb) is False


@pytest.mark.parametrize("rows", [1, 8, 511])
def test_expand_row_floor_rejects_decode_shapes(monkeypatch, rows):
    monkeypatch.setattr(hc, "_hc_expand_kernel", object())
    x, residual, post, comb = _mk(1, rows, 4, 64)
    assert hc.hc_expand_fused_eligible(x, residual, post, comb) is False


def test_expand_eligibility_accepts_prefill_shape(monkeypatch):
    monkeypatch.setattr(hc, "_hc_expand_kernel", object())
    x, residual, post, comb = _mk(1, 1024, 4, 64)
    assert hc.hc_expand_fused_eligible(x, residual, post, comb) is True


def test_expand_eligibility_rejects_bad_dtypes(monkeypatch):
    monkeypatch.setattr(hc, "_hc_expand_kernel", object())
    x, residual, post, comb = _mk(1, 1024, 4, 64)
    assert hc.hc_expand_fused_eligible(x, residual, post.astype(mx.bfloat16), comb) is False
    assert hc.hc_expand_fused_eligible(x, residual.astype(mx.bfloat16), post, comb) is False


def test_expand_eligibility_rejects_unaligned_D(monkeypatch):
    monkeypatch.setattr(hc, "_hc_expand_kernel", object())
    x, residual, post, comb = _mk(1, 1024, 4, 66)
    assert hc.hc_expand_fused_eligible(x, residual, post, comb) is False


def test_prenorm_eligibility_row_floor_and_width(monkeypatch):
    monkeypatch.setattr(hc, "_hc_prenorm_kernel", object())
    rng = np.random.default_rng(1)
    big = mx.array(rng.standard_normal((1, 1024, 4, 1024)).astype(np.float32))
    small = mx.array(rng.standard_normal((1, 4, 4, 1024)).astype(np.float32))
    odd = mx.array(rng.standard_normal((1, 1024, 4, 100)).astype(np.float32))
    assert hc.hc_prenorm_fused_eligible(big) is True
    assert hc.hc_prenorm_fused_eligible(small) is False
    assert hc.hc_prenorm_fused_eligible(odd) is False


# ----------------------------------------------------------------- fallback
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_hc_expand_flag_on_is_a_noop_without_metal(monkeypatch, dtype):
    monkeypatch.setenv("MLX_VLM_GLM5_HC_FUSED", "1")
    _reset_flag_cache()
    x, residual, post, comb = _mk(1, 1024, 4, 64, dtype=dtype)
    got = hc.hc_expand(x, residual, post, comb)
    ref = hc._hc_expand_op(x, residual, post, comb)
    assert got.dtype == ref.dtype
    assert mx.array_equal(got, ref).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_hc_expand_default_on_is_a_noop_without_metal(monkeypatch, dtype):
    """The CPU / no-Metal fallback that the promotion rests on: with the flags
    at their new defaults the kernel handle is None, so ``hc_expand`` returns
    the untouched eager result, bit for bit."""
    _reset_flag_cache()
    assert hc._hc_fused_enabled() is True
    assert hc._hc_expand_kernel is None
    x, residual, post, comb = _mk(1, 1024, 4, 64, dtype=dtype)
    got = hc.hc_expand(x, residual, post, comb)
    ref = hc._hc_expand_op(x, residual, post, comb)
    assert got.dtype == ref.dtype
    assert mx.array_equal(got, ref).item()


# ------------------------------------------------------------ dispatch choice
def test_default_on_dispatches_the_fused_path(monkeypatch):
    """Env gating, direction 1: unset flags + an eligible prefill shape reach
    ``_hc_expand_fused``."""
    monkeypatch.setattr(hc, "_hc_expand_kernel", object())
    spy = _PathSpy(monkeypatch)
    _reset_flag_cache()
    x, residual, post, comb = _mk(1, 1024, 4, 64)
    hc.hc_expand(x, residual, post, comb)
    assert spy.which == "fused"


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_explicit_off_restores_the_eager_path(monkeypatch, value):
    """Env gating, direction 2 -- the opt-out contract.

    Asserts on the FUNCTION THAT RAN, not on the returned values: the eager and
    fused results agree to a few ulp, so a value comparison would not notice a
    flag that stopped working.
    """
    monkeypatch.setenv("MLX_VLM_GLM5_HC_FUSED", value)
    monkeypatch.setattr(hc, "_hc_expand_kernel", object())
    spy = _PathSpy(monkeypatch)
    _reset_flag_cache()
    x, residual, post, comb = _mk(1, 1024, 4, 64)
    got = hc.hc_expand(x, residual, post, comb)
    assert spy.which == "eager"
    # ...and what it returned is the genuine eager result, bit for bit.
    ref = spy.real_op(x, residual, post, comb)
    assert got.dtype == ref.dtype
    assert mx.array_equal(got, ref).item()


@pytest.mark.parametrize("rows", [1, 8, 511])
def test_row_floor_holds_with_flags_defaulted_on(monkeypatch, rows):
    """DECODE PATH UNCHANGED by the I1450 promotion.

    1 row (greedy decode) and 511 rows (any speculative width we run, and one
    below the floor) still take the eager path with the flags at their new
    default-ON values and a Metal kernel handle present.
    """
    monkeypatch.setattr(hc, "_hc_expand_kernel", object())
    spy = _PathSpy(monkeypatch)
    _reset_flag_cache()
    assert hc._hc_fused_enabled() is True
    x, residual, post, comb = _mk(1, rows, 4, 64)
    hc.hc_expand(x, residual, post, comb)
    assert spy.which == "eager"


def test_row_floor_boundary_is_512(monkeypatch):
    """511 eager / 512 fused, with the flags defaulted on: the floor constant is
    what separates decode from prefill, not the flag."""
    assert hc._HC_FUSED_MIN_ROWS == 512
    monkeypatch.setattr(hc, "_hc_expand_kernel", object())
    _reset_flag_cache()
    x, residual, post, comb = _mk(1, 511, 4, 64)
    assert hc.hc_expand_fused_eligible(x, residual, post, comb) is False
    x, residual, post, comb = _mk(1, 512, 4, 64)
    assert hc.hc_expand_fused_eligible(x, residual, post, comb) is True


def test_prenorm_row_floor_holds_with_flags_defaulted_on(monkeypatch):
    """Same floor on the other kernel: the pre-norm is not taken at decode
    width even though its flag now defaults ON."""
    monkeypatch.setattr(hc, "_hc_prenorm_kernel", object())
    _reset_flag_cache()
    assert hc._hc_prenorm_fused_enabled() is True
    rng = np.random.default_rng(11)
    for rows, expected in ((1, False), (511, False), (512, True)):
        z = mx.array(rng.standard_normal((1, rows, 4, 1024)).astype(np.float32))
        assert hc.hc_prenorm_fused_eligible(z) is expected


# ----------------------------------------------------------------- numerics
def test_expand_kernel_arithmetic_matches_eager_fp32():
    """fp32 activations: the kernel's order and mx's must agree to fp32 noise."""
    x, residual, post, comb = _mk(1, 64, 4, 128, dtype=mx.float32, seed=3)
    ref = np.array(hc._hc_expand_op(x, residual, post, comb), copy=False)
    got = np_expand_kernel(
        np.array(x, copy=False),
        np.array(residual, copy=False),
        np.array(post, copy=False),
        np.array(comb, copy=False),
    )
    # NOT zero.  The kernel sums the hyper axis in ascending j with plain
    # mul/add; mx.matmul does not, so the two differ by about one fp32 ulp of
    # the row scale even at fp32.  Recorded here as the reason the L34 gate is
    # the natural-panel text sha + speculative-rail parity (B0 rule 13) and
    # NOT a bit-identical logit fingerprint.
    scale = float(np.max(np.abs(ref)))
    ulps = np.max(np.abs(got - ref)) / (np.finfo(np.float32).eps * scale)
    assert ulps <= 8.0, f"expand kernel drifted {ulps:.2f} ulp from mx.matmul"
    assert np.max(np.abs(got - ref)) > 0.0


def test_expand_kernel_arithmetic_matches_eager_bf16():
    x, residual, post, comb = _mk(1, 64, 4, 128, dtype=mx.bfloat16, seed=4)
    ref = np.array(
        hc._hc_expand_op(x, residual, post, comb).astype(mx.float32), copy=False
    )
    got = np_expand_kernel(
        np.array(x.astype(mx.float32), copy=False),
        np.array(residual.astype(mx.float32), copy=False),
        np.array(post, copy=False),
        np.array(comb, copy=False),
    )
    # Round the kernel result the way the kernel's store does.
    got = np.array(mx.array(got).astype(mx.bfloat16).astype(mx.float32), copy=False)
    denom = np.maximum(np.abs(ref), 1e-2)
    assert np.max(np.abs(got - ref) / denom) < 1e-2


def test_prenorm_kernel_arithmetic_matches_mx_rms_norm():
    """NOT an identity assertion.

    ``mx.fast.rms_norm``'s reduction tree is an implementation detail of the
    installed MLX; this pins the kernel's own arithmetic and reports that the
    two agree to fp32 reduction noise, which is what the L23 fingerprint gate
    then has to confirm (or refute) on GPU.
    """
    eps = 1e-5
    rng = np.random.default_rng(7)
    x = mx.array(rng.standard_normal((1, 2, 4, 512)).astype(np.float32)).astype(
        mx.bfloat16
    )
    ref = np.array(
        mx.fast.rms_norm(x.astype(mx.float32).flatten(-2), None, eps), copy=False
    )
    got = np_prenorm_kernel(np.array(x.astype(mx.float32), copy=False), eps)
    denom = np.maximum(np.abs(ref), 1e-3)
    assert np.max(np.abs(got - ref) / denom) < 1e-5


# ------------------------------------------------------ compile-only (Metal)
@pytest.mark.skipif(
    not mx.metal.is_available(), reason="metal_kernel needs Metal to compile"
)
def test_kernel_sources_build(monkeypatch):
    # Builds the kernel OBJECTS only.  mx.fast.metal_kernel does not compile or
    # dispatch anything until it is called, so this touches no GPU.
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert hc._make_hc_expand_kernel() is not None
    assert hc._make_hc_prenorm_kernel() is not None
