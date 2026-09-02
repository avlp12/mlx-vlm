"""The verify block's dense projections, padded past MLX's qmv/qmm crossover.

The lever, in one sentence: MLX v0.32.x picks the quantized-matmul kernel on the
FLATTENED row count ``M``, and below ``get_qmv_batch_limit(K, N)`` it picks one
that re-reads the whole weight matrix once per row -- so a DFlash2 verify block
of 8 or 9 rows pays 2.2x to 2.9x the M=1 weight traffic for nine tokens' work.
Padding the activation to the limit with zero rows and slicing the result back
buys the weight-reusing ``qmm_splitk`` for three rows of dummy arithmetic
(sweep11 P-L1: whole dense chain 31.665 ms at M=9 -> 24.209 ms at M=12).

What this file pins:

1.  ``qmv_batch_limit`` reproduces MLX v0.32.1's own table
    (``mlx/backend/metal/quantized.cpp:85-122``) at the twelve shapes this
    checkpoint actually has.  If MLX retunes that constant, the pad target has to
    move with it, and this is the test that says so.
2.  The pad fires ONLY inside a declared verify window.  S=1 decode and prefill
    keep their exact bytes -- a ragged prefill chunk can present the same M as a
    verify block, so gating on M alone would silently break the receipted
    bit-identity of the chunked speculative prefill.
3.  The eligibility rule: the per-bit-width floor, the "does the pad actually
    cross THIS shape's boundary" test, and the K in {64,128} exclusion
    (``quantized.cpp:1754`` routes those to ``qmv_quad``, whose grid is already
    keyed on M, so the pad would only buy more work).
4.  Pad-and-slice is algebraically exact: at the real shapes, on CPU, where MLX
    has no M-keyed kernel switch, the padded result is BITWISE the unpadded one.
    That is the point of running this gate on CPU -- it separates "the slicing
    algebra is right" from "the GPU picked a different reduction order", and any
    delta seen on Metal is therefore attributable to the routing alone.
5.  Law 23: the resolved knobs reach the log at model construction, not only the
    test that sets them.
"""

import logging
import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.glm5_next import verify_pad as vp
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel

# (K, N, shipped bits, expected get_qmv_batch_limit) for every dense projection
# of GLM-5.3-Flash-vlm-q4-quasar, read off the build's safetensors headers in
# sweep11 P-L1 arm E.
REAL_SHAPES = [
    ("kda_in_proj_fused", 4096, 24896, 8, 12),
    ("kda_o_proj", 8192, 4096, 8, 12),
    ("mla_o_proj", 16384, 4096, 6, 12),
    ("mla_q_b_proj", 1536, 16384, 6, 12),
    ("mla_q_a_proj", 4096, 1536, 6, 18),
    ("mla_indexer_wq_b", 1536, 4096, 8, 18),
    ("mla_kv_a_proj", 4096, 512, 8, 18),
    ("shared_gate_up", 4096, 2048, 6, 18),
    ("shared_down", 2048, 4096, 6, 18),
    ("dense_gate_up", 4096, 12288, 6, 12),
    ("dense_down", 12288, 4096, 6, 12),
    ("lm_head", 4096, 154880, 6, 12),
]


@pytest.fixture(autouse=True)
def _clean_config(monkeypatch):
    """Every test resolves the knobs itself; none inherits another's."""
    for name in (vp.PAD_M_ENV, vp.PAD_MIN_ENV, vp.PAD_MIN_Q8_ENV):
        monkeypatch.delenv(name, raising=False)
    vp.reset_for_tests()
    yield
    vp.reset_for_tests()


def _arm(monkeypatch, pad_m=None, pad_min=None, pad_min_q8=None):
    for name, value in (
        (vp.PAD_M_ENV, pad_m),
        (vp.PAD_MIN_ENV, pad_min),
        (vp.PAD_MIN_Q8_ENV, pad_min_q8),
    ):
        if value is not None:
            monkeypatch.setenv(name, str(value))
    vp.reset_for_tests()


def _is_g15d():
    return vp._arch_table() == (15, "d")


needs_g15d = pytest.mark.skipif(
    not _is_g15d(),
    reason="the shape->limit table is architecture specific; this box is not applegpu_g15*d",
)


# ------------------------------------------------------------------ 1. the table


@needs_g15d
@pytest.mark.parametrize("tag,K,N,bits,limit", REAL_SHAPES)
def test_qmv_batch_limit_matches_mlx_source(tag, K, N, bits, limit):
    """32 / 18 / 12 by shape on arch_size 'd' -- quantized.cpp:104-115."""
    assert vp.qmv_batch_limit(K, N) == limit, tag


@needs_g15d
def test_limit_boundaries_are_inclusive_the_way_mlx_writes_them():
    # `D <= 2048 && O <= 2048` -> 32, `D <= 4096 && O <= 4096` -> 18, else 12.
    assert vp.qmv_batch_limit(2048, 2048) == 32
    assert vp.qmv_batch_limit(2049, 2048) == 18
    assert vp.qmv_batch_limit(4096, 4096) == 18
    assert vp.qmv_batch_limit(4097, 4096) == 12
    assert vp.qmv_batch_limit(4096, 4097) == 12


# ------------------------------------------------------- 2. the eligibility rule


@needs_g15d
def test_pad_is_inert_outside_a_verify_window(monkeypatch):
    _arm(monkeypatch, pad_m=12)
    assert not vp.window_is_open()
    assert vp._plan(9, 4096, 24896, 8) == 0
    assert vp.counters() == {"padded": 0, "declined": 0}


@needs_g15d
def test_floor_is_per_bit_width(monkeypatch):
    _arm(monkeypatch, pad_m=12)
    with vp.verify_window(True):
        # 8-bit shapes cross earlier than MLX's constant: measured 0.4155 ms at
        # M=8 via qmv against 0.3172 via qmm on kda_in_proj_fused.
        assert vp._plan(8, 4096, 24896, 8) == 12
        assert vp._plan(7, 4096, 24896, 8) == 0
        # 4-bit and 6-bit shapes do not: the same M=8 cell is 0.2697 -> 0.3003.
        assert vp._plan(8, 4096, 24896, 4) == 0
        assert vp._plan(9, 4096, 24896, 4) == 12
        # at or above the limit MLX already takes the qmm route
        assert vp._plan(12, 4096, 24896, 8) == 0
        assert vp._plan(64, 4096, 24896, 8) == 0


@needs_g15d
def test_shapes_whose_limit_the_pad_would_not_reach_are_declined(monkeypatch):
    _arm(monkeypatch, pad_m=12)
    with vp.verify_window(True):
        # limit 18: padding to 12 would still be a qmv dispatch, only a wider one.
        assert vp._plan(9, 4096, 2048, 6) == 0
        assert vp._plan(9, 2048, 4096, 6) == 0
    _arm(monkeypatch, pad_m=18)
    with vp.verify_window(True):
        assert vp._plan(9, 4096, 2048, 6) == 18
        # ... and a 12-limit shape then pads to 18, not to 12
        assert vp._plan(9, 4096, 24896, 8) == 18


@needs_g15d
def test_qmv_quad_shapes_are_excluded(monkeypatch):
    _arm(monkeypatch, pad_m=12)
    with vp.verify_window(True):
        # f_b_proj / g_b_proj are [qkv_dim, head_dim=128]; quantized.cpp:1754
        # sends K in {64,128} to qmv_quad, whose grid is already (M, ., B).
        assert vp._plan(9, 128, 8192, 8) == 0
        assert vp._plan(9, 64, 8192, 8) == 0


def test_pad_m_zero_disables_everything(monkeypatch):
    _arm(monkeypatch, pad_m=0)
    assert not vp.config().enabled
    with vp.verify_window(True):
        assert not vp.window_is_open()
        assert vp._plan(9, 4096, 24896, 8) == 0


def test_a_junk_knob_falls_back_to_the_default_instead_of_raising(monkeypatch):
    monkeypatch.setenv(vp.PAD_M_ENV, "twelve")
    vp.reset_for_tests()
    assert vp.config().pad_m == vp.DEFAULT_PAD_M


def test_the_shipped_defaults_are_the_measured_ones(monkeypatch):
    _arm(monkeypatch)
    cfg = vp.config()
    assert (cfg.pad_m, cfg.pad_min, cfg.pad_min_q8) == (12, 9, 8)


# ----------------------------------------------- 3. pad-and-slice is exact math


@needs_g15d
@pytest.mark.parametrize("M", [8, 9, 11])
def test_pad_and_slice_is_bitwise_exact_at_real_shapes(monkeypatch, M):
    """On CPU there is no M-keyed kernel switch, so this isolates the algebra.

    Any delta observed on Metal is therefore the routing, not the slicing.
    """
    _arm(monkeypatch, pad_m=12)
    K, N, bits = 1536, 16384, 8  # mla_q_b_proj's shape, at a size CPU can chew
    mx.random.seed(11)
    x = mx.random.normal((1, M, K)).astype(mx.bfloat16)
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    qw, scales, biases = mx.quantize(w, 64, bits)
    ref = mx.quantized_matmul(
        x, qw, scales, biases, transpose=True, group_size=64, bits=bits
    )
    with vp.verify_window(True):
        got = vp.quantized_matmul(
            x, qw, scales, biases, transpose=True, group_size=64, bits=bits
        )
    assert vp.counters()["padded"] == 1
    assert got.shape == ref.shape == (1, M, N)
    assert mx.array_equal(got, ref).item()


@needs_g15d
def test_project_preserves_shape_and_declines_unquantized(monkeypatch):
    _arm(monkeypatch, pad_m=12)
    layer = nn.Linear(1536, 16384, bias=False)
    mx.eval(layer.parameters())
    x = mx.random.normal((1, 9, 1536)).astype(mx.bfloat16)
    with vp.verify_window(True):
        dense = vp.project(layer, x)
    assert vp.counters()["padded"] == 0
    assert mx.array_equal(dense, layer(x)).item()

    # nn.quantize swaps CHILD modules, so a bare leaf has to be wrapped to be
    # quantized at all -- quantizing it in place silently leaves an nn.Linear,
    # and the assertion below is what catches that.
    holder = nn.Sequential(layer)
    nn.quantize(holder, group_size=64, bits=8)
    mx.eval(holder.parameters())
    qlayer = holder.layers[0]
    assert hasattr(qlayer, "scales")
    ref = qlayer(x)
    with vp.verify_window(True):
        got = vp.project(qlayer, x)
    assert vp.counters()["padded"] == 1
    assert got.shape == ref.shape
    assert mx.array_equal(got, ref).item()


# --------------------------------------------------------- 4. the model, end to end


def _tiny_config():
    """Small everywhere EXCEPT where the limit table needs a dimension over 4096.

    ``hidden_size`` is 192 rather than the usual 128 on purpose: K == 128 is the
    ``qmv_quad`` short-circuit, so a 128-wide toy would decline every projection
    and the test would pass while proving nothing.
    """
    return TextConfig(
        model_type="glm5_next_text",
        vocab_size=4224,
        hidden_size=192,
        intermediate_size=4224,
        moe_intermediate_size=4224,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=0,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_head_dim=64,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        index_topk=6,
        index_head_dim=64,
        index_n_heads=2,
        index_kpool=3,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 1024,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        eos_token_id=1,
    )


@pytest.fixture(scope="module")
def tiny_lm():
    mx.random.seed(0)
    lm = LanguageModel(_tiny_config())
    nn.quantize(lm, group_size=64, bits=8)
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def _tokens(n, seed):
    return (mx.arange(n, dtype=mx.int32) * (seed + 7) % 4000)[None, :] + 1


def _primed_cache(lm, ctx):
    cache = lm.make_cache()
    mx.eval(lm(ctx, cache=cache, capture_gdn_states=False).logits)
    return cache


@needs_g15d
def test_verify_forward_pads_and_agrees_within_the_ulp_class(monkeypatch, tiny_lm):
    """The acceptance gate: argmax equality over 64+ positions, delta sub-ulp.

    Bit-identity is NOT claimed and must not be asserted -- ``qmm_splitk`` and
    ``qmv_wide`` sum in different partitions.  On CPU the delta happens to be
    exactly zero (one kernel serves every M), which is why the ulp bound below is
    written as a bound and not as an equality.
    """
    S, blocks = 9, 8
    ctx0 = _tokens(40, 1)
    stream = _tokens(blocks * S, 3)
    worst, equal, padded = 0.0, 0, 0
    for i in range(blocks):
        ctx = mx.concatenate([ctx0, stream[:, : i * S]], axis=1)
        blk = stream[:, i * S : (i + 1) * S]

        _arm(monkeypatch, pad_m=0)
        off = tiny_lm(
            blk,
            cache=_primed_cache(tiny_lm, ctx),
            capture_layer_ids=[0, 1],
            speculative_verify=True,
        ).logits
        mx.eval(off)

        _arm(monkeypatch, pad_m=12)
        on = tiny_lm(
            blk,
            cache=_primed_cache(tiny_lm, ctx),
            capture_layer_ids=[0, 1],
            speculative_verify=True,
        ).logits
        mx.eval(on)
        padded += vp.counters()["padded"]

        f32 = lambda a: a.astype(mx.float32)
        scale = mx.max(mx.abs(f32(off))).item()
        delta = mx.max(mx.abs(f32(on) - f32(off))).item()
        worst = max(worst, delta / scale)
        equal += int(mx.sum(mx.argmax(on, -1) == mx.argmax(off, -1)).item())

    assert padded == blocks * 8, "the lever must actually fire on this geometry"
    assert equal == blocks * S >= 64
    # bfloat16 has 8 mantissa bits: one ulp is 2**-8 relative.  Two of them.
    assert worst <= 2 * 2**-8, worst


@needs_g15d
def test_s1_decode_is_untouched(monkeypatch, tiny_lm):
    ctx0 = _tokens(40, 1)
    tok = _tokens(1, 3)
    _arm(monkeypatch, pad_m=0)
    off = tiny_lm(
        tok,
        cache=_primed_cache(tiny_lm, ctx0),
        capture_layer_ids=[0, 1],
        speculative_verify=True,
    ).logits
    mx.eval(off)
    _arm(monkeypatch, pad_m=12)
    on = tiny_lm(
        tok,
        cache=_primed_cache(tiny_lm, ctx0),
        capture_layer_ids=[0, 1],
        speculative_verify=True,
    ).logits
    mx.eval(on)
    assert vp.counters()["padded"] == 0
    assert mx.array_equal(on, off).item()


@needs_g15d
def test_prefill_at_a_verify_width_is_untouched(monkeypatch, tiny_lm):
    """A ragged prefill chunk can land on M = 9..11 too, and must not move.

    R30 K6 receipts the chunked speculative prefill as bit-identical to greedy's;
    padding it would spend that receipt for a saving on a path nobody times.
    """
    chunk = mx.concatenate([_tokens(40, 1), _tokens(9, 3)], axis=1)
    _arm(monkeypatch, pad_m=0)
    off = tiny_lm(chunk, cache=tiny_lm.make_cache(), capture_gdn_states=False).logits
    mx.eval(off)
    _arm(monkeypatch, pad_m=12)
    on = tiny_lm(chunk, cache=tiny_lm.make_cache(), capture_gdn_states=False).logits
    mx.eval(on)
    assert vp.counters()["padded"] == 0
    assert mx.array_equal(on, off).item()


# ----------------------------------------------------------------- 5. law 23


def test_the_resolved_knobs_reach_the_log_at_model_construction(monkeypatch, caplog):
    _arm(monkeypatch, pad_m=12)
    with caplog.at_level(logging.INFO, logger=vp.__name__):
        LanguageModel(_tiny_config())
    lines = [r.getMessage() for r in caplog.records]
    assert any(vp.PAD_M_ENV in line and "=12" in line for line in lines), lines


def test_the_verify_width_is_logged_on_first_sight(monkeypatch, caplog, tiny_lm):
    _arm(monkeypatch, pad_m=12)
    ctx0 = _tokens(40, 1)
    cache = _primed_cache(tiny_lm, ctx0)
    with caplog.at_level(logging.INFO, logger=vp.__name__):
        mx.eval(
            tiny_lm(
                _tokens(9, 3),
                cache=cache,
                capture_layer_ids=[0, 1],
                speculative_verify=True,
            ).logits
        )
    lines = [r.getMessage() for r in caplog.records]
    assert any("M=9" in line for line in lines), lines
    assert vp.observed_widths() == {9: 1}
