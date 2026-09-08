"""K6 -- ``MLX_VLM_GLM5_KDA_GLUE_COMPILE_MIN_ROWS``: a row floor on the KDA glue compile.

L13 measured +1 % prefill with ``MLX_VLM_GLM5_KDA_GLUE_COMPILE`` on; I1310 measured
DFlash2 acceptance 4.82 -> 3.27 per round with the same flag and sent it back OFF.
Those are different shapes: the win is at S = 8,192 (a prefill chunk), the loss at
S = 8 (a speculative verify block).  The floor separates them -- compile only at
S >= N -- and N = 512 is below every prefill chunk and far above every verify block.

Two things are pinned here: the PREDICATE (unset == today, exactly), and what the
compiled glue does to the numbers on CPU, which is reported rather than assumed --
the campaign gate for this arm is the spec rail, per I1310's own rule.
"""
import os

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.glm5_next.config import TextConfig

KEY = "MLX_VLM_GLM5_KDA_GLUE_COMPILE_MIN_ROWS"
BOOL_KEY = "MLX_VLM_GLM5_KDA_GLUE_COMPILE"

_CFG = dict(
    model_type="glm5_next_text",
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=256,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=8,
    routed_scaling_factor=2.5,
    kv_lora_rank=64,
    q_lora_rank=128,
    qk_rope_head_dim=0,
    v_head_dim=32,
    qk_nope_head_dim=32,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-05,
    index_topk=64,
    index_head_dim=64,
    index_n_heads=4,
    layer_types=["linear_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={
        "num_heads": 2,
        "gate_lower_bound": -5.0,
        "head_dim": 64,
        "short_conv_kernel_size": 4,
    },
)
H, D, K = 2, 64, 4


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in (KEY, BOOL_KEY)}
    prev_latch = glm5._KDA_GLUE_COMPILE_ENV
    for k in (KEY, BOOL_KEY):
        os.environ.pop(k, None)
    glm5._KDA_GLUE_COMPILE_ENV = None
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    glm5._KDA_GLUE_COMPILE_ENV = prev_latch


def _layer(seed=0):
    mx.random.seed(seed)
    config = TextConfig.from_dict(dict(_CFG))
    layer = glm5.Glm5NextLinearAttention(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    layer.update(rand(layer.parameters()))
    layer.conv1d.weight = (mx.random.normal(layer.conv1d.weight.shape) * 0.5).astype(mx.bfloat16)
    layer.forget_gate.A_log = (mx.random.normal((H,)) * 0.5).astype(mx.float32)
    layer.forget_gate.dt_bias = (mx.random.normal((H * D,)) * 0.5).astype(mx.float32)
    layer.o_norm.weight = (mx.ones((D,)) + 0.02 * mx.random.normal((D,))).astype(mx.bfloat16)
    mx.eval(layer.parameters())
    return layer, config


def _cache(batch=1, seed=1):
    mx.random.seed(seed)
    cache = ArraysCache(size=2)
    cache[0] = (mx.random.normal((batch, K - 1, 3 * H * D)) * 0.3).astype(mx.bfloat16)
    cache[1] = (mx.random.normal((batch, H, D, D)) * 0.05).astype(mx.float32)
    mx.eval(cache[0], cache[1])
    return cache


def _forward(layer, config, S, *, env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    glm5._KDA_GLUE_COMPILE_ENV = None
    layer._kda_glue_pre_c = None
    layer._kda_glue_post_c = None
    mx.random.seed(7)
    x = (mx.random.normal((1, S, config.hidden_size)) * 0.5).astype(mx.bfloat16)
    cache = _cache()
    y = layer(x, mask=None, cache=cache)
    mx.eval(y, cache[0], cache[1])
    return y, cache


# ------------------------------------------------------------------ predicate
class TestPredicate:
    def test_unset_is_todays_predicate(self):
        # floor unset: the boolean alone decides, and it defaults OFF.
        assert glm5._kda_glue_compile_min_rows() == 0
        for rows in (1, 2, 8, 512, 8192):
            assert glm5._kda_glue_compile_for(rows) is False
        os.environ[BOOL_KEY] = "1"
        glm5._KDA_GLUE_COMPILE_ENV = None
        assert glm5._kda_glue_compile_for(1) is False   # S > 1 guard survives
        for rows in (2, 8, 512, 8192):
            assert glm5._kda_glue_compile_for(rows) is True

    def test_floor_excludes_decode_and_the_verify_block(self):
        os.environ[KEY] = "512"
        for rows in (1, 2, 4, 8, 16, 32, 64, 511):
            assert glm5._kda_glue_compile_for(rows) is False, rows
        for rows in (512, 1024, 8192):
            assert glm5._kda_glue_compile_for(rows) is True, rows

    def test_floor_alone_arms_the_lever(self):
        """One assignment is a complete arm: the boolean need not also be set."""
        os.environ[KEY] = "512"
        assert BOOL_KEY not in os.environ
        assert glm5._kda_glue_compile_for(8192) is True

    @pytest.mark.parametrize("raw,want", [
        (None, 0), ("", 0), ("0", 0), ("-4", 0), ("garbage", 0),
        ("512", 512), (" 1024 ", 1024), ("2", 2),
    ])
    def test_floor_parsing(self, raw, want):
        if raw is None:
            os.environ.pop(KEY, None)
        else:
            os.environ[KEY] = raw
        assert glm5._kda_glue_compile_min_rows() == want

    def test_floor_is_read_per_call(self):
        os.environ[KEY] = "512"
        assert glm5._kda_glue_compile_for(512) is True
        os.environ[KEY] = "1024"
        assert glm5._kda_glue_compile_for(512) is False


# ------------------------------------------------------------------- forwards
class TestForward:
    @pytest.mark.parametrize("S", [8, 64])
    def test_below_the_floor_is_byte_for_byte_the_default(self, S):
        layer, config = _layer()
        ref, cref = _forward(layer, config, S, env={KEY: None, BOOL_KEY: None})
        arm, carm = _forward(layer, config, S, env={KEY: "512", BOOL_KEY: None})
        assert mx.array_equal(arm, ref).item()
        assert mx.array_equal(carm[0], cref[0]).item()
        assert mx.array_equal(carm[1], cref[1]).item()

    @pytest.mark.parametrize("S", [512, 700])
    def test_at_or_above_the_floor_the_glue_is_compiled(self, S):
        """Non-vacuity: the compiled callables really get built at these widths."""
        layer, config = _layer()
        _forward(layer, config, S, env={KEY: "512", BOOL_KEY: None})
        assert layer._kda_glue_pre_c is not None
        assert layer._kda_glue_post_c is not None
        _forward(layer, config, S, env={KEY: None, BOOL_KEY: None})
        assert layer._kda_glue_pre_c is None
        assert layer._kda_glue_post_c is None

    @pytest.mark.parametrize("S", [512, 700])
    def test_compiled_vs_eager_numerics(self, S):
        """Reported, not asserted-to-zero: mx.compile may fuse the elementwise chains.

        The assertion is a loose sanity bound; the exact deviation is printed so the
        Stage-A doc can state it.  If this ever becomes bit-identical the assertion
        below still holds, and the doc line is what changes.
        """
        layer, config = _layer()
        ref, cref = _forward(layer, config, S, env={KEY: None, BOOL_KEY: None})
        arm, carm = _forward(layer, config, S, env={KEY: "512", BOOL_KEY: None})
        d = mx.abs(arm.astype(mx.float32) - ref.astype(mx.float32))
        scale = mx.max(mx.abs(ref.astype(mx.float32))).item() or 1.0
        max_abs = mx.max(d).item()
        print(f"K6 S={S}: identical={bool(mx.array_equal(arm, ref).item())} "
              f"max_abs={max_abs:.3e} max_rel={max_abs / scale:.3e} "
              f"state_identical={bool(mx.array_equal(carm[1], cref[1]).item())}")
        assert max_abs / scale < 1e-2
