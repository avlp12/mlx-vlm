"""L13: mx.compile fusion toggles for the hyper-connection (HC) norm/mix/inject
path and the KDA eager-glue path.

``MLX_VLM_GLM5_HC_COMPILE=1`` swaps ``HyperConnection.__call__``'s pre-norm +
mixing-matmul preamble (and, on the non-GPU-kernel / training branch, the
whole sinkhorn + collapse block) for module-level `mx.compile`d equivalents.
``MLX_VLM_GLM5_KDA_GLUE_COMPILE=1`` compiles the pure-array glue in
``Glm5NextLinearAttention.__call__`` around ``gated_delta_update`` for the
eager S>1 (prefill) path only.

Both default OFF. All of this runs on CPU (``mx.default_device() != mx.gpu``),
which is what forces ``HyperConnection`` onto its ``_hc_ops`` branch (the
custom Metal ``_hc_kernel`` path, and the fused-KDA decode kernel, are both
GPU-only and unverified here -- see the report for what that leaves
unverified).
"""

import mlx.core as mx
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.deepseek_v4 import hyper_connection as hc
from mlx_vlm.models.glm5_next.config import TextConfig

MAX_ABS_DIFF_TOL = 1e-3


def _small_config(**overrides):
    cfg = dict(
        model_type="glm5_next_text",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=16,
        qk_nope_head_dim=16,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        index_topk=2048,
        index_head_dim=16,
        index_n_heads=2,
        layer_types=["linear_attention"],
        mlp_layer_types=["dense"],
        linear_attn_config={
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
    )
    cfg.update(overrides)
    return TextConfig.from_dict(cfg)


def _randomize(module, seed=0):
    mx.random.seed(seed)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(tree.dtype)

    module.update(rand(module.parameters()))
    return module


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    d = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
    mx.eval(d)
    return float(d)


@pytest.fixture(autouse=True)
def _reset_toggles():
    saved_hc = hc._HC_COMPILE_ENV
    saved_kda = glm5._KDA_GLUE_COMPILE_ENV
    yield
    hc._HC_COMPILE_ENV = saved_hc
    glm5._KDA_GLUE_COMPILE_ENV = saved_kda


# ---------------------------------------------------------------------------
# HyperConnection / hc_expand
# ---------------------------------------------------------------------------


def test_hc_compile_default_off(monkeypatch):
    monkeypatch.delenv("MLX_VLM_GLM5_HC_COMPILE", raising=False)
    hc._HC_COMPILE_ENV = None
    assert hc._hc_compile_enabled() is False


def test_hc_compile_env_enables(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GLM5_HC_COMPILE", "1")
    hc._HC_COMPILE_ENV = None
    assert hc._hc_compile_enabled() is True


def test_hc_compile_dispatches_to_compiled_functions(monkeypatch):
    """The flag must actually select the compiled callables, not just read true."""
    config = _small_config()
    conn = _randomize(hc.HyperConnection(config))
    mx.random.seed(1)
    x = (mx.random.normal((1, 3, config.hc_mult, config.hidden_size)) * 0.3).astype(
        mx.bfloat16
    )
    mx.eval(x)

    calls = {"pre": 0, "full": 0}
    orig_pre = hc._hc_preamble_compiled
    orig_full = hc._hc_full_ops_compiled

    def spy_pre(*a, **k):
        calls["pre"] += 1
        return orig_pre(*a, **k)

    def spy_full(*a, **k):
        calls["full"] += 1
        return orig_full(*a, **k)

    monkeypatch.setattr(hc, "_hc_preamble_compiled", spy_pre)
    monkeypatch.setattr(hc, "_hc_full_ops_compiled", spy_full)

    hc._HC_COMPILE_ENV = False
    conn(x)
    assert calls == {"pre": 0, "full": 0}

    hc._HC_COMPILE_ENV = True
    conn(x)
    assert calls == {"pre": 1, "full": 1}


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("S", [1, 5, 64])
def test_hc_compile_matches_eager(B, S):
    config = _small_config()
    conn = _randomize(hc.HyperConnection(config))
    mx.random.seed(2)
    x = (
        mx.random.normal((B, S, config.hc_mult, config.hidden_size)) * 0.3
    ).astype(mx.bfloat16)
    mx.eval(x)

    hc._HC_COMPILE_ENV = False
    xc_ref, post_ref, comb_ref = conn(x)
    mx.eval(xc_ref, post_ref, comb_ref)

    hc._HC_COMPILE_ENV = True
    xc_got, post_got, comb_got = conn(x)
    mx.eval(xc_got, post_got, comb_got)

    assert xc_ref.shape == xc_got.shape
    assert post_ref.shape == post_got.shape
    assert comb_ref.shape == comb_got.shape
    assert xc_ref.dtype == xc_got.dtype

    diff = _max_abs_diff(xc_ref, xc_got)
    assert diff < MAX_ABS_DIFF_TOL, f"HC collapse max abs diff {diff} (B={B}, S={S})"
    print(f"[hc_compile] B={B} S={S} max_abs_diff={diff:.3e}")


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("S", [1, 5, 64])
def test_hc_compile_matches_eager_with_folded_norm(B, S):
    """norm_w path (fold_hc_norm=True callers) must agree too."""
    config = _small_config()
    conn = _randomize(hc.HyperConnection(config))
    norm_w = (mx.random.normal((config.hidden_size,)) * 0.1 + 1.0).astype(
        mx.bfloat16
    )
    mx.eval(norm_w)
    mx.random.seed(3)
    x = (
        mx.random.normal((B, S, config.hc_mult, config.hidden_size)) * 0.3
    ).astype(mx.bfloat16)
    mx.eval(x)

    hc._HC_COMPILE_ENV = False
    xc_ref, _, _ = conn(x, norm_w=norm_w, norm_eps=config.rms_norm_eps)
    mx.eval(xc_ref)

    hc._HC_COMPILE_ENV = True
    xc_got, _, _ = conn(x, norm_w=norm_w, norm_eps=config.rms_norm_eps)
    mx.eval(xc_got)

    assert xc_ref.shape == xc_got.shape
    diff = _max_abs_diff(xc_ref, xc_got)
    assert diff < MAX_ABS_DIFF_TOL, f"folded-norm max abs diff {diff} (B={B}, S={S})"


# ---------------------------------------------------------------------------
# KDA glue
# ---------------------------------------------------------------------------


def _kda_layer(config, seed=0):
    return _randomize(glm5.Glm5NextLinearAttention(config), seed=seed)


def test_kda_glue_compile_default_on(monkeypatch):
    # Default ON since 2026-09-05 (operator-approved micro bundle); "0" restores.
    monkeypatch.delenv("MLX_VLM_GLM5_KDA_GLUE_COMPILE", raising=False)
    glm5._KDA_GLUE_COMPILE_ENV = None
    assert glm5._kda_glue_compile_enabled() is True
    monkeypatch.setenv("MLX_VLM_GLM5_KDA_GLUE_COMPILE", "0")
    glm5._KDA_GLUE_COMPILE_ENV = None
    assert glm5._kda_glue_compile_enabled() is False
    glm5._KDA_GLUE_COMPILE_ENV = None


def test_kda_glue_compile_env_enables(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GLM5_KDA_GLUE_COMPILE", "1")
    glm5._KDA_GLUE_COMPILE_ENV = None
    assert glm5._kda_glue_compile_enabled() is True


def test_kda_glue_compile_builds_cache_only_for_S_gt_1_and_flag_set():
    config = _small_config()
    layer = _kda_layer(config)
    mx.random.seed(4)
    x = (mx.random.normal((1, 4, config.hidden_size)) * 0.3).astype(mx.bfloat16)
    mx.eval(x)

    glm5._KDA_GLUE_COMPILE_ENV = False
    layer(x)
    assert layer._kda_glue_pre_c is None
    assert layer._kda_glue_post_c is None

    glm5._KDA_GLUE_COMPILE_ENV = True
    layer(x)
    assert layer._kda_glue_pre_c is not None
    assert layer._kda_glue_post_c is not None


def test_kda_glue_compile_flag_ignored_at_decode_S1():
    """The S>1 guard must hold even when the fused-decode kernel is unreachable
    (as it always is here, on CPU) -- the toggle must not build/use the
    compiled glue at S=1 regardless of why the eager path was taken."""
    config = _small_config()
    layer = _kda_layer(config)
    mx.random.seed(5)
    x = (mx.random.normal((1, 1, config.hidden_size)) * 0.3).astype(mx.bfloat16)
    mx.eval(x)

    glm5._KDA_GLUE_COMPILE_ENV = True
    layer(x)
    assert layer._kda_glue_pre_c is None, "S=1 must not build the compiled glue"
    assert layer._kda_glue_post_c is None, "S=1 must not build the compiled glue"


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("S", [2, 5, 64])
def test_kda_glue_compile_matches_eager(B, S):
    config = _small_config()
    layer_ref = _kda_layer(config, seed=10)
    layer_got = _kda_layer(config, seed=10)  # identical weights (same seed)

    mx.random.seed(6)
    x = (mx.random.normal((B, S, config.hidden_size)) * 0.3).astype(mx.bfloat16)
    mx.eval(x)

    glm5._KDA_GLUE_COMPILE_ENV = False
    out_ref = layer_ref(x)
    mx.eval(out_ref)

    glm5._KDA_GLUE_COMPILE_ENV = True
    out_got = layer_got(x)
    mx.eval(out_got)

    assert out_ref.shape == out_got.shape
    assert out_ref.dtype == out_got.dtype
    diff = _max_abs_diff(out_ref, out_got)
    assert diff < MAX_ABS_DIFF_TOL, f"KDA glue max abs diff {diff} (B={B}, S={S})"
    print(f"[kda_glue_compile] B={B} S={S} max_abs_diff={diff:.3e}")
