"""K5 -- ``MLX_VLM_GLM5_MOE_COMBINE_MATMUL``: the routed-expert combine as one GEMM.

``(y * scores[..., None]).sum(-2)`` multiplies a bf16 [B, S, top_k, hidden] tensor by
FP32 router scores, so MLX promotes and materialises an fp32 tensor twice the size of
y (3.0 GB per sparse layer at chunk 8,192; 1.0 % of the step per the V5 SW
decomposition).  ``scores[..., None, :] @ y`` is the same contraction as one batched
matvec: y is read once, the accumulator lives in the GEMM, and no fp32 copy exists.

This is NOT bit-identical -- the scores are rounded to y's dtype and the reduction
order is the kernel's -- so the tests assert (a) the default path is byte-for-byte the
old expression, (b) the arm agrees with an fp64 reference at least as well as a bf16
result can, and (c) they report the actual deviation for the Stage-A doc / KL gate.
"""
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.deepseek_v32.language import moe_combine

KEY = "MLX_VLM_GLM5_MOE_COMBINE_MATMUL"


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.get(KEY)
    os.environ.pop(KEY, None)
    yield
    if saved is None:
        os.environ.pop(KEY, None)
    else:
        os.environ[KEY] = saved


def _inputs(B=1, S=64, k=8, H=256, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    y = (mx.random.normal((B, S, k, H)) * 0.5).astype(dtype)
    # Router scores as group_expert_select emits them: fp32, non-negative, normalised
    # over the kept k and then scaled by routed_scaling_factor.
    raw = mx.sigmoid(mx.random.normal((B, S, k)).astype(mx.float32))
    scores = (raw / raw.sum(axis=-1, keepdims=True)) * 2.5
    mx.eval(y, scores)
    return y, scores


def _shipped(y, scores):
    return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)


class TestDefaultUnchanged:
    @pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
    @pytest.mark.parametrize("shape", [(1, 64, 8, 256), (2, 33, 6, 128), (1, 1, 8, 256)])
    def test_flag_unset_is_the_shipped_expression(self, dtype, shape):
        y, scores = _inputs(*shape, dtype=dtype)
        got = moe_combine(y, scores)
        want = _shipped(y, scores)
        mx.eval(got, want)
        assert got.dtype == want.dtype == dtype
        assert mx.array_equal(got, want).item()

    def test_explicit_off_is_the_shipped_expression(self):
        os.environ[KEY] = "0"
        y, scores = _inputs()
        assert mx.array_equal(moe_combine(y, scores), _shipped(y, scores)).item()


class TestArm:
    @pytest.mark.parametrize("shape", [(1, 64, 8, 256), (2, 33, 6, 128), (1, 1, 8, 256)])
    def test_shape_and_dtype_are_preserved(self, shape):
        os.environ[KEY] = "1"
        y, scores = _inputs(*shape)
        got = moe_combine(y, scores)
        want = _shipped(y, scores)
        mx.eval(got, want)
        assert got.shape == want.shape
        assert got.dtype == want.dtype

    def test_float32_y_keeps_the_scores_exact(self):
        """With y already fp32 the cast is the identity; only the order can differ."""
        os.environ[KEY] = "1"
        y, scores = _inputs(dtype=mx.float32)
        arm = moe_combine(y, scores)
        ref = _shipped(y, scores)
        mx.eval(arm, ref)
        d = float(mx.max(mx.abs(arm - ref)).item())
        scale = float(mx.max(mx.abs(ref)).item())
        print(f"K5 fp32 y: identical={bool(mx.array_equal(arm, ref).item())} "
              f"max_abs={d:.3e} max_rel={d / scale:.3e}")
        assert d / scale < 1e-5

    def test_bfloat16_deviation_is_reported_and_bounded(self):
        """The real regime: bf16 y, fp32 scores.  Both paths are compared against an
        fp64 reference computed in numpy, because "differs from the shipped path" is
        only interesting if the arm is not the WORSE of the two."""
        os.environ[KEY] = "1"
        y, scores = _inputs()
        arm = moe_combine(y, scores)
        os.environ[KEY] = "0"
        ref = moe_combine(y, scores)
        mx.eval(arm, ref)

        yn = np.array(y.astype(mx.float32), copy=False).astype(np.float64)
        sn = np.array(scores, copy=False).astype(np.float64)
        exact = (yn * sn[..., None]).sum(axis=-2)
        a = np.array(arm.astype(mx.float32), copy=False).astype(np.float64)
        r = np.array(ref.astype(mx.float32), copy=False).astype(np.float64)
        scale = float(np.abs(exact).max())
        d_arm_ref = float(np.abs(a - r).max())
        print(
            f"K5 bf16 y: identical={bool(mx.array_equal(arm, ref).item())} "
            f"arm-vs-shipped max_abs={d_arm_ref:.3e} max_rel={d_arm_ref / scale:.3e} | "
            f"vs exact: arm {float(np.abs(a - exact).max()) / scale:.3e}, "
            f"shipped {float(np.abs(r - exact).max()) / scale:.3e}"
        )
        # bf16 has ~2**-9 relative resolution and the output itself is bf16, so any
        # honest implementation lands within a few bf16 ulps of the other.
        assert d_arm_ref / scale < 2 ** -6

    def test_never_materialises_an_fp32_product(self, monkeypatch):
        """Non-vacuity: the arm must not take the multiply-then-sum route at all."""
        os.environ[KEY] = "1"
        y, scores = _inputs()

        def _boom(*a, **k):  # pragma: no cover - the point is that it is not called
            raise AssertionError("the arm reduced with mx.sum instead of a matmul")

        monkeypatch.setattr(mx.array, "sum", _boom)
        mx.eval(moe_combine(y, scores))


class TestReachesTheModel:
    def test_glm5_moe_layer_uses_the_gate(self):
        from mlx_vlm.models.glm5_next.config import TextConfig
        from mlx_vlm.models.glm5_next.language import Glm5NextMoE

        cfg = TextConfig.from_dict(dict(
            model_type="glm5_next_text", vocab_size=256, hidden_size=64,
            intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1,
            n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=32,
            q_lora_rank=32, qk_rope_head_dim=0, v_head_dim=16, qk_nope_head_dim=16,
            num_experts_per_tok=2, first_k_dense_replace=0,
            max_position_embeddings=4096, rms_norm_eps=1e-05, index_topk=16,
            index_head_dim=16, index_n_heads=2, layer_types=["full_attention"],
            mlp_layer_types=["sparse"],
            linear_attn_config={"num_heads": 2, "gate_lower_bound": -5.0,
                                "head_dim": 32, "short_conv_kernel_size": 4},
        ))
        mx.random.seed(0)
        moe = Glm5NextMoE(cfg)
        mx.eval(moe.parameters())
        x = (mx.random.normal((1, 32, cfg.hidden_size)) * 0.3).astype(mx.bfloat16)
        os.environ.pop(KEY, None)
        ref = moe(x)
        os.environ[KEY] = "1"
        arm = moe(x)
        mx.eval(ref, arm)
        d = float(mx.max(mx.abs(arm.astype(mx.float32) - ref.astype(mx.float32))).item())
        scale = float(mx.max(mx.abs(ref.astype(mx.float32))).item())
        print(f"K5 Glm5NextMoE: identical={bool(mx.array_equal(arm, ref).item())} "
              f"max_abs={d:.3e} max_rel={d / scale:.3e}")
        assert ref.shape == arm.shape
        assert d / scale < 2 ** -6
