"""v3 row #2 -- identity rails for the fused GLM-5.3-Flash decode router.

CPU ONLY.  ``mx.fast.metal_kernel`` is GPU-only
(``mlx/backend/common/metal_kernel.cpp:40-50``), so the kernel itself cannot be
executed here and ``fused_router_supported`` returns False on this device -- which
is exactly why the flag is byte-identical on CPU.  What IS testable, and what
these rails test, is the *algorithm* the kernel implements, transliterated in
``fused_router.reference_select``, against
``fused_router.metal_semantics_select`` -- the eager router rewritten with the
two substitutions MLX's own Metal backend makes (argpartition -> argsort,
sum -> sequential row_reduce_small).  See the ``fused_router`` module docstring
for the line-by-line source citations.

NOT covered here (stated, not hidden): the equality of ``metal::exp`` between
MLX's unary/compiled kernel libraries and a ``mx.fast.metal_kernel`` library, and
the kernel's actual execution.  Those need the GPU arm in
``bench/ops/queues/gesicht_v2a_router_20260907.sh``.
"""

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

import mlx_vlm.models.glm5_next.language as glm5  # noqa: E402
from mlx_vlm.models.deepseek_v32.language import group_expert_select  # noqa: E402
from mlx_vlm.models.glm5_next import fused_router as FR  # noqa: E402
from mlx_vlm.models.glm5_next.config import TextConfig  # noqa: E402

E = 288
TOP_K = 8
RSF = 2.5
N_ROWS = 2000


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _logit_bank(seed: int = 1234):
    """2,000 rows: 1,400 plain random + 600 constructed ties / near-ties."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((N_ROWS, E)) * 3.0).astype(np.float32)
    # exact ties, one pair, inside the top-k band
    x[:200, 5] = x[:200, 7]
    # exact three-way tie, one of them far down the row
    x[200:400, 3] = x[200:400, 11] = x[200:400, 200]
    # 1-ulp near-ties
    blk = x[400:600].copy()
    blk[:, 20] = np.nextafter(blk[:, 21], np.float32(np.inf)).astype(np.float32)
    x[400:600] = blk
    # a whole row of the same value: every tie-break decided by index alone
    x[600:610, :] = np.float32(0.25)
    bias = (rng.standard_normal(E) * 0.1).astype(np.float32)
    return mx.array(x), mx.array(bias)


@pytest.fixture(scope="module")
def bank():
    logits, bias = _logit_bank()
    scores = mx.sigmoid(logits)
    mx.eval(logits, bias, scores)
    return logits, bias, scores


# --------------------------------------------------------------------------- #
# 1. the flag
# --------------------------------------------------------------------------- #

def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(FR.ROUTER_FUSED_ENV, raising=False)
    assert FR.router_fused_enabled() is False


def test_flag_env_on(monkeypatch):
    monkeypatch.setenv(FR.ROUTER_FUSED_ENV, "1")
    assert FR.router_fused_enabled() is True


def test_config_overrides_env(monkeypatch):
    monkeypatch.setenv(FR.ROUTER_FUSED_ENV, "1")

    class C:
        router_fused = False

    assert FR.router_fused_enabled(C()) is False


def test_unsupported_on_cpu():
    """The whole point of the CPU byte-identity guarantee."""
    assert mx.default_device() == mx.cpu
    assert FR.fused_router_supported(1, E, TOP_K, 1, True) is False


@pytest.mark.parametrize(
    "rows,n_experts,top_k,n_group,norm",
    [
        (0, E, TOP_K, 1, True),                 # no rows
        (FR.MAX_FUSED_ROWS + 1, E, TOP_K, 1, True),   # prefill width
        (1, E, TOP_K, 8, True),                 # group-limited branch
        (1, E, TOP_K, 1, False),                # norm_topk_prob off
        (1, 4, TOP_K, 1, True),                 # fewer experts than top_k
        (1, FR.MAX_FUSED_EXPERTS + 1, TOP_K, 1, True),   # threadgroup memory
    ],
)
def test_support_gates_reject(monkeypatch, rows, n_experts, top_k, n_group, norm):
    """Each guard must reject on its own, independently of the device check."""
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert FR.fused_router_supported(rows, n_experts, top_k, n_group, norm) is False


def test_support_gate_accepts_decode_shape(monkeypatch):
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert FR.fused_router_supported(1, E, TOP_K, 1, True) is True
    assert FR.fused_router_supported(FR.MAX_FUSED_ROWS, E, TOP_K, 1, True) is True


# --------------------------------------------------------------------------- #
# 2. THE identity rail: kernel algorithm == Metal eager, bit for bit
# --------------------------------------------------------------------------- #

def test_reference_matches_metal_semantics_indices_and_weights(bank):
    _, bias, scores = bank
    i_ref, g_ref = FR.reference_select(scores, bias, TOP_K, RSF, True)
    i_ms, g_ms = FR.metal_semantics_select(scores, bias, TOP_K, RSF, True)
    mx.eval(i_ref, g_ref, i_ms, g_ms)
    assert i_ref.shape == (N_ROWS, TOP_K) and i_ref.dtype == mx.uint32
    assert g_ref.dtype == mx.float32
    assert bool(mx.array_equal(i_ref, i_ms))
    assert bool(mx.array_equal(g_ref, g_ms))


@pytest.mark.parametrize("top_k", [1, 2, 4, 8, 16])
def test_reference_matches_metal_semantics_across_top_k(bank, top_k):
    _, bias, scores = bank
    s = scores[:256]
    i_ref, g_ref = FR.reference_select(s, bias, top_k, RSF, True)
    i_ms, g_ms = FR.metal_semantics_select(s, bias, top_k, RSF, True)
    mx.eval(i_ref, g_ref, i_ms, g_ms)
    assert bool(mx.array_equal(i_ref, i_ms))
    assert bool(mx.array_equal(g_ref, g_ms))


def test_reference_ties_break_to_the_lowest_expert_index():
    """A row that is constant everywhere: the answer must be 0,1,...,K-1."""
    scores = mx.full((1, E), 0.5, dtype=mx.float32)
    bias = mx.zeros((E,), dtype=mx.float32)
    inds, _ = FR.reference_select(scores, bias, TOP_K, RSF, True)
    mx.eval(inds)
    assert np.array(inds).tolist() == [list(range(TOP_K))]
    i_ms, _ = FR.metal_semantics_select(scores, bias, TOP_K, RSF, True)
    mx.eval(i_ms)
    assert bool(mx.array_equal(inds, i_ms))


def test_reference_orders_descending_by_biased_score(bank):
    _, bias, scores = bank
    inds, _ = FR.reference_select(scores, bias, TOP_K, RSF, True)
    biased = np.array(scores + bias)
    picked = np.take_along_axis(biased, np.array(inds).astype(np.int64), axis=-1)
    assert bool((np.diff(picked, axis=-1) <= 0).all())


def test_gates_sum_to_the_scaling_factor(bank):
    _, bias, scores = bank
    _, gates = FR.reference_select(scores, bias, TOP_K, RSF, True)
    tot = np.array(gates).sum(axis=-1)
    assert np.abs(tot - RSF).max() < 1e-5


# --------------------------------------------------------------------------- #
# 3. what the CPU eager path does differently -- documented, not hidden
# --------------------------------------------------------------------------- #

def test_cpu_eager_selects_the_same_expert_SET_but_not_the_same_order(bank):
    """``mx.argpartition`` on CPU is ``std::nth_element`` (cpu/sort.cpp:297): the
    top-k SET is the same as Metal's sort, the ORDER is not.  This is a
    pre-existing CPU/Metal fork property, not something the fusion introduces --
    it is the reason the identity rail is written against Metal semantics."""
    logits, bias, _ = bank
    i_e, _ = group_expert_select(logits, bias, TOP_K, 1, 1, RSF, True)
    i_ms, _ = FR.metal_semantics_select(mx.sigmoid(logits), bias, TOP_K, RSF, True)
    mx.eval(i_e, i_ms)
    a, b = np.array(i_e), np.array(i_ms)
    assert (np.sort(a, axis=-1) == np.sort(b, axis=-1)).all()
    assert not (a == b).all()


def test_cpu_sum_order_differs_from_metal_row_reduce_small(bank):
    """The other half of the same story: MLX's CPU ``Sum`` over an 8-float row is
    not the ascending sequential accumulation ``row_reduce_small`` performs on
    Metal (reduce_row.h:166-181).  Bounded here so a future MLX change that makes
    it matter is visible."""
    _, bias, scores = bank
    _, g_seq = FR.metal_semantics_select(scores, bias, TOP_K, RSF, True,
                                         sum_order="sequential")
    _, g_mx = FR.metal_semantics_select(scores, bias, TOP_K, RSF, True,
                                        sum_order="mx")
    mx.eval(g_seq, g_mx)
    d = np.abs(np.array(g_seq) - np.array(g_mx))
    assert d.max() < 1e-6
    assert float(np.abs(np.array(g_seq)).max()) > 0.1


# --------------------------------------------------------------------------- #
# 4. the gate wiring
# --------------------------------------------------------------------------- #

def _gate_config(**over):
    cfg = dict(
        model_type="glm5_next_text",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=E,
        num_experts_per_tok=TOP_K,
        routed_scaling_factor=RSF,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=16,
        qk_nope_head_dim=16,
        first_k_dense_replace=0,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        index_topk=2048,
        index_head_dim=16,
        index_n_heads=2,
        layer_types=["linear_attention"],
        mlp_layer_types=["sparse"],
        linear_attn_config={
            "num_heads": 4,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
    )
    cfg.update(over)
    return TextConfig.from_dict(cfg)


def test_gate_reads_the_flag_at_construction(monkeypatch):
    cfg = _gate_config()
    monkeypatch.delenv(FR.ROUTER_FUSED_ENV, raising=False)
    assert glm5.Glm5NextMoEGate(cfg).use_fused_router is False
    monkeypatch.setenv(FR.ROUTER_FUSED_ENV, "1")
    assert glm5.Glm5NextMoEGate(cfg).use_fused_router is True


def test_gate_is_byte_identical_on_cpu_with_the_flag_on(monkeypatch):
    """Flag on, CPU: the support gate refuses, the eager path runs, bytes match."""
    cfg = _gate_config()
    mx.random.seed(7)
    gate = glm5.Glm5NextMoEGate(cfg)
    gate.weight = mx.random.normal((E, cfg.hidden_size)).astype(mx.float32)
    gate.e_score_correction_bias = mx.random.normal((E,)) * 0.1
    x = mx.random.normal((1, 1, cfg.hidden_size)).astype(mx.bfloat16)
    mx.eval(gate.weight, gate.e_score_correction_bias, x)

    gate.use_fused_router = False
    i0, g0 = gate(x)
    gate.use_fused_router = True
    i1, g1 = gate(x)
    mx.eval(i0, g0, i1, g1)
    assert bool(mx.array_equal(i0, i1))
    assert bool(mx.array_equal(g0, g1))


def test_gate_calls_the_kernel_when_the_support_gate_passes(monkeypatch):
    """The flag must actually select the fused callable, not just read true."""
    cfg = _gate_config()
    gate = glm5.Glm5NextMoEGate(cfg)
    gate.weight = mx.random.normal((E, cfg.hidden_size)).astype(mx.float32)
    gate.e_score_correction_bias = mx.zeros((E,))
    x = mx.random.normal((1, 1, cfg.hidden_size)).astype(mx.bfloat16)
    mx.eval(gate.weight, gate.e_score_correction_bias, x)

    seen = {}

    def fake_supported(*a, **k):
        return True

    def fake_fused(logits, bias, top_k, rsf, norm):
        seen["rows"] = logits.size // logits.shape[-1]
        seen["top_k"] = top_k
        seen["rsf"] = rsf
        return FR.reference_select(mx.sigmoid(logits), bias, top_k, rsf, norm)

    monkeypatch.setattr(glm5, "fused_router_supported", fake_supported)
    monkeypatch.setattr(glm5, "fused_group_expert_select", fake_fused)
    gate.use_fused_router = True
    inds, gates = gate(x)
    mx.eval(inds, gates)
    assert seen == {"rows": 1, "top_k": TOP_K, "rsf": RSF}
    assert inds.shape == (1, 1, TOP_K)


# --------------------------------------------------------------------------- #
# 5. whole decode step, B=1, depth 512
# --------------------------------------------------------------------------- #

def _model_config():
    layers = ["linear_attention"] * 5 + ["full_attention"]
    return TextConfig.from_dict(
        dict(
            model_type="glm5_next_text",
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=len(layers),
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=64,
            num_experts_per_tok=6,
            routed_scaling_factor=RSF,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            kv_lora_rank=32,
            q_lora_rank=32,
            qk_rope_head_dim=0,
            v_head_dim=16,
            qk_nope_head_dim=16,
            first_k_dense_replace=1,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            index_topk=2048,
            index_head_dim=16,
            index_n_heads=2,
            layer_types=layers,
            mlp_layer_types=["dense"] + ["sparse"] * (len(layers) - 1),
            linear_attn_config={
                "num_heads": 4,
                "head_dim": 8,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
        )
    )


def _select_via(select_fn):
    def __call__(self, x):
        w = self.weight
        if w.dtype != mx.float32:
            w = w.astype(mx.float32)
        logits = x.astype(mx.float32) @ w.T
        return select_fn(
            mx.sigmoid(logits),
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )

    return __call__


def _decode_logits(model, prompt, select_fn, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(glm5.Glm5NextMoEGate, "__call__", _select_via(select_fn))
        cache = model.make_cache()
        out = model(prompt, cache=cache)
        mx.eval(out.logits)
        nxt = mx.argmax(out.logits[:, -1, :], axis=-1)[:, None]
        step = model(nxt, cache=cache)
        mx.eval(step.logits)
        return step.logits


def test_whole_decode_step_fused_algorithm_equals_metal_eager(monkeypatch):
    """B=1, prompt depth 512, 5 sparse MoE layers: the kernel's algorithm and the
    Metal eager router give bit-identical next-token logits."""
    cfg = _model_config()
    mx.random.seed(11)
    model = glm5.LanguageModel(cfg)
    model.eval()

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(tree.dtype)

    model.update(rand(model.parameters()))
    # compile_ffn traces the FFN block; ``reference_select`` materialises through
    # numpy, which is illegal inside an mx.compile trace.  The eager arm is what
    # this rail compares, and compile_ffn changes no arithmetic.
    for layer in model.model.layers:
        layer.compile_ffn = False
    mx.eval(model.parameters())

    prompt = mx.random.randint(0, cfg.vocab_size, (1, 512))
    mx.eval(prompt)

    a = _decode_logits(model, prompt, FR.reference_select, monkeypatch)
    b = _decode_logits(model, prompt, FR.metal_semantics_select, monkeypatch)
    assert bool(mx.array_equal(a, b))


# --------------------------------------------------------------------------- #
# 6. dispatch count -- built on the GPU stream, never evaluated
# --------------------------------------------------------------------------- #

def _graph_labels(*outs):
    import collections
    import io
    import re

    buf = io.StringIO()
    mx.export_to_dot(buf, *outs)
    pat = re.compile(r'(\d+) \[label ="([^"]+)"')
    return collections.Counter(lab for _, lab in pat.findall(buf.getvalue()))


# Same list as bench/hwdossier/v3_dispatch_census.py:132-139.
_VIEW_LABELS = frozenset(
    {
        "Reshape", "Split", "Slice", "Broadcast", "BroadcastAxes", "ExpandDims",
        "Squeeze", "Transpose", "Flatten", "Unflatten", "AsStrided",
        "Contiguous", "Copy", "StopGradient", "Depends", "View",
    }
)

_ROUTER_EAGER_COMPUTE = {
    "Matmul", "Sigmoid", "CompiledBroadcastAddNegative", "ArgPartition",
    "GatherAxis", "Sum", "CompiledBroadcastDivideBroadcastMultiply",
}


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal device to build for")
def test_fused_router_is_two_compute_nodes_on_the_gpu_stream():
    """The census's headline claim, measured rather than substituted.

    Graphs are BUILT on the gpu stream and never evaluated -- ``mx.array`` is
    lazy, so no command buffer is encoded and no GPU time is spent.  ``0 GPU
    minutes`` in the same sense as ``v3_dispatch_census.py``.
    """
    with mx.stream(mx.gpu):
        x = mx.zeros((1, 1, 256), mx.bfloat16)
        w = mx.zeros((E, 256), mx.float32)
        b = mx.zeros((E,), mx.float32)

        eager = group_expert_select(x.astype(mx.float32) @ w.T, b, TOP_K, 1, 1, RSF, True)
        fused = FR.fused_group_expert_select(
            x.astype(mx.float32) @ w.T, b, TOP_K, RSF, True
        )

        le = _graph_labels(*eager)
        lf = _graph_labels(*fused)

    # the router's own primitives, leaves (Full/AsType) excluded
    e_compute = {k for k in le if k not in _VIEW_LABELS} & _ROUTER_EAGER_COMPUTE
    assert e_compute == _ROUTER_EAGER_COMPUTE          # all 7, exactly as the P0 census
    assert sum(le[k] for k in _ROUTER_EAGER_COMPUTE) == 7

    assert lf["CustomKernel"] == 1
    assert lf["Matmul"] == 1
    assert {k for k in _ROUTER_EAGER_COMPUTE if k != "Matmul"} & set(lf) == set()
    # everything the fusion adds is a metadata view on a contiguous array
    assert all(k in _VIEW_LABELS or k in ("Matmul", "CustomKernel", "Full", "AsType")
               for k in lf)


# --------------------------------------------------------------------------- #
# 7. the kernel source itself -- compiled, never run
# --------------------------------------------------------------------------- #

# The signature ``write_signature`` generates for this kernel
# (mlx/backend/common/metal_kernel.cpp:52-175): three template params, two
# ``device`` float inputs (both >= max_constant_array_size = 8 elements), the
# 0-dim ``rsf`` scalar as a ``constant`` reference, two device outputs, and the
# two attributes the body names.
_SIGNATURE = """#include <metal_stdlib>
using namespace metal;

template <int E, int K, bool NORM>
[[kernel]] void glm5_router_topk(
  const device float* logits [[buffer(0)]],
  const device float* bias [[buffer(1)]],
  const constant float& rsf [[buffer(2)]],
  device uint32_t* inds [[buffer(3)]],
  device float* gates [[buffer(4)]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
"""

_INSTANTIATIONS = """
template [[host_name("a")]] [[kernel]]
decltype(glm5_router_topk<288, 8, true>) glm5_router_topk<288, 8, true>;
template [[host_name("b")]] [[kernel]]
decltype(glm5_router_topk<288, 1, false>) glm5_router_topk<288, 1, false>;
template [[host_name("c")]] [[kernel]]
decltype(glm5_router_topk<2048, 16, true>) glm5_router_topk<2048, 16, true>;
"""


def _have_metal_frontend():
    import shutil
    import subprocess

    if shutil.which("xcrun") is None:
        return False
    try:
        return subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True, timeout=60,
        ).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _have_metal_frontend(), reason="no Metal front end")
def test_kernel_source_compiles(tmp_path):
    """Front-end compile only: ``metal -c`` produces AIR, nothing is dispatched
    and no GPU is touched.  This is the only automated check the desk can make
    on kernel text it cannot execute."""
    import subprocess

    src = tmp_path / "k.metal"
    src.write_text(_SIGNATURE + FR._SOURCE + "\n}\n" + _INSTANTIATIONS)
    out = tmp_path / "k.air"
    r = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-Wall", "-Wextra",
         "-c", str(src), "-o", str(out)],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr
    assert r.stderr.strip() == ""
    assert out.exists() and out.stat().st_size > 0


def test_threadgroup_memory_fits_at_the_declared_expert_ceiling():
    """2 float arrays + 1 uchar array of E, plus the 32-lane scratch: must stay
    under Apple silicon's 32 KiB threadgroup allocation."""
    e = FR.MAX_FUSED_EXPERTS
    tg = 4 * e + 4 * e + e + 4 * FR.LANES + 4 * FR.LANES + 4 * 32
    assert tg < 32 * 1024
