"""Equivalence tests for the MQA fold and the fused MLA metal kernel.

Both are OFF by default; these tests flip the env flags on a REAL-weight module (a module
constructed with its default random init, then given non-trivial weights) and require the
three dispatch paths to agree to within a bf16 ulp.  An identity check on a freshly built
zero-weight module is vacuous, so every case here perturbs the weights first.
"""

import importlib
import os
import unittest

import mlx.core as mx
import mlx.nn as nn


def _reload(**env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    import mlx_vlm.models.glm5_next.language as L

    importlib.reload(L)
    return L


def _ulp_bf16(x):
    return float(mx.abs(x).max()) * 2**-8


class TestFusedMLA(unittest.TestCase):
    def setUp(self):
        self.saved = {
            k: os.environ.get(k)
            for k in ("MLX_VLM_GLM5_MQA_FOLD", "MLX_VLM_GLM5_FUSED_MLA")
        }

    def tearDown(self):
        _reload(**self.saved)

    # ---------------------------------------------------------------- kernel level
    def test_kernel_matches_composite(self):
        from mlx_vlm.models.glm5_next.fused_mla_attn import mla_flash_attention

        mx.random.seed(0)
        for G, R, N, D, kind in [
            (2, 32, 128, 512, None),
            (3, 64, 2051, 512, "keyvalid"),
            (1, 128, 512, 512, "causal"),
            (1, 64, 1000, 512, "sparse"),
            (2, 96, 300, 512, None),
            (1, 32, 64, 256, None),
        ]:
            q = mx.random.normal((G, R, D)).astype(mx.bfloat16)
            kv = mx.random.normal((G, N, D)).astype(mx.bfloat16)
            if kind == "causal":
                qi, ki = mx.arange(R)[:, None], mx.arange(N)[None, :]
                m3 = mx.broadcast_to(((qi + (N - R)) >= ki)[None], (1, R, N))
            elif kind == "keyvalid":
                m3 = mx.random.uniform(shape=(G, 1, N)) > 0.1
            elif kind == "sparse":
                m3 = mx.random.uniform(shape=(1, R, N)) > 0.8
                m3 = m3 | (mx.arange(N)[None, None, :] >= N - 4)
            else:
                m3 = None
            scale = 256**-0.5
            got = mla_flash_attention(q, kv, scale, m3)
            m4 = None if m3 is None else m3[:, None]
            ref = mx.fast.scaled_dot_product_attention(
                q[:, None], kv[:, None], kv[:, None], scale=scale, mask=m4
            )[:, 0]
            err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
            self.assertLess(
                err, 4 * _ulp_bf16(ref.astype(mx.float32)),
                f"{G=} {R=} {N=} {D=} {kind=} err={err}",
            )

    def test_kernel_rejects_bad_shapes(self):
        from mlx_vlm.models.glm5_next.fused_mla_attn import mla_flash_attention

        q = mx.zeros((2, 32, 512), mx.bfloat16)
        with self.assertRaises(ValueError):  # head dim mismatch
            mla_flash_attention(q, mx.zeros((2, 64, 256), mx.bfloat16), 0.1)
        with self.assertRaises(ValueError):  # kv groups do not divide q groups
            mla_flash_attention(q, mx.zeros((3, 64, 512), mx.bfloat16), 0.1)
        with self.assertRaises(ValueError):  # float32 not supported
            mla_flash_attention(q.astype(mx.float32), mx.zeros((2, 64, 512)), 0.1)

    # ---------------------------------------------------------------- module level
    def _module(self, L):
        from mlx.utils import tree_map

        from mlx_vlm.models.glm5_next.config import TextConfig

        cfg = TextConfig(
            model_type="glm5_next", vocab_size=128, hidden_size=256,
            intermediate_size=256, moe_intermediate_size=256, num_hidden_layers=1,
            num_attention_heads=8, num_key_value_heads=8, n_shared_experts=1,
            n_routed_experts=4, routed_scaling_factor=1.0, kv_lora_rank=512,
            q_lora_rank=64, qk_rope_head_dim=0, v_head_dim=256, qk_nope_head_dim=256,
            num_experts_per_tok=2, first_k_dense_replace=0, max_position_embeddings=4096,
            rms_norm_eps=1e-6, index_topk=64, index_head_dim=32, index_n_heads=4,
            layer_types=["deepseek_sparse_attention"], mlp_layer_types=["dense"],
            linear_attn_config={}, index_kpool=4,
        )
        m = L.Glm5NextSparseAttention(cfg)

        # REAL weights AND the serving dtype.  Both matter: nn.Linear inits in float32, and
        # _mqa_sdpa's fused branch declines anything that is not bfloat16 -- so a module left
        # at the default dtype makes every one of these assertions vacuously true.  The first
        # version of this test did exactly that and passed while executing none of the code.
        def prep(t):
            if not isinstance(t, mx.array):
                return t
            return (t + 0.1 * mx.random.normal(t.shape)).astype(mx.bfloat16)

        m.update(tree_map(prep, m.parameters()))
        m.eval()
        self.assertFalse(m.training, "production path must not be in training mode")
        return m

    def _run(self, env, x):
        """Returns (output, n_fused_kernel_calls, [q shapes seen by MLX's own sdpa])."""
        L = _reload(**env)
        import mlx_vlm.models.glm5_next.fused_mla_attn as F

        mx.random.seed(3)
        m = self._module(L)

        n = [0]
        orig_k = F.mla_flash_attention

        def wrap_k(*a, **kw):
            n[0] += 1
            return orig_k(*a, **kw)

        F.mla_flash_attention = wrap_k
        shapes = []
        orig_s = L.scaled_dot_product_attention

        def wrap_s(q, k, v, cache=None, scale=None, mask=None, **kw):
            shapes.append(tuple(q.shape))
            return orig_s(q, k, v, cache=cache, scale=scale, mask=mask, **kw)

        L.scaled_dot_product_attention = wrap_s
        try:
            out = m(x, mask=None, cache=None)
            mx.eval(out)
        finally:
            F.mla_flash_attention = orig_k
            L.scaled_dot_product_attention = orig_s
        return out, n[0], shapes

    def test_module_paths_agree_and_are_actually_taken(self):
        mx.random.seed(3)
        x = mx.random.normal((1, 96, 256)).astype(mx.bfloat16)
        mx.eval(x)

        ref, n_ref, sh_ref = self._run(
            {"MLX_VLM_GLM5_MQA_FOLD": None, "MLX_VLM_GLM5_FUSED_MLA": None}, x
        )
        self.assertEqual(n_ref, 0)
        self.assertEqual(sh_ref, [(1, 8, 96, 512)], "baseline should take MLX's own dense call")
        u = _ulp_bf16(ref.astype(mx.float32))

        # dense prefill (L=96 > 1): only the fused kernel can take it; the fold is L==1 only.
        got, n_k, sh = self._run(
            {"MLX_VLM_GLM5_MQA_FOLD": None, "MLX_VLM_GLM5_FUSED_MLA": "1"}, x
        )
        self.assertEqual(n_k, 1, "fused kernel was NOT entered on the dense prefill path")
        self.assertEqual(sh, [], "MLX's own sdpa should not be reached when the kernel takes it")
        err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
        self.assertLess(err, 8 * u, f"fused dense err={err} ulp={u}")

    def test_decode_fold_is_taken(self):
        mx.random.seed(3)
        x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
        mx.eval(x)
        ref, _, sh_ref = self._run(
            {"MLX_VLM_GLM5_MQA_FOLD": None, "MLX_VLM_GLM5_FUSED_MLA": None}, x
        )
        self.assertEqual(sh_ref, [(1, 8, 1, 512)])
        u = _ulp_bf16(ref.astype(mx.float32))

        got, n_k, sh = self._run(
            {"MLX_VLM_GLM5_MQA_FOLD": "1", "MLX_VLM_GLM5_FUSED_MLA": None}, x
        )
        self.assertEqual(n_k, 0)
        self.assertEqual(
            sh, [(1, 1, 8, 512)], "fold should hand MLX a [B,1,H,D] query, not [B,H,1,D]"
        )
        err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
        self.assertLess(err, 8 * u, f"fold err={err} ulp={u}")

        got, n_k, sh = self._run(
            {"MLX_VLM_GLM5_MQA_FOLD": None, "MLX_VLM_GLM5_FUSED_MLA": "1"}, x
        )
        self.assertEqual(n_k, 1, "fused kernel was NOT entered on the L==1 path")
        err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
        self.assertLess(err, 8 * u, f"fused decode err={err} ulp={u}")


if __name__ == "__main__":
    unittest.main()
