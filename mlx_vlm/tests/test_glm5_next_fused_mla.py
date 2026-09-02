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
            for k in (
                "MLX_VLM_GLM5_MQA_FOLD",
                "MLX_VLM_GLM5_FUSED_MLA",
                "MLX_VLM_GLM5_FUSED_MLA_MIN_TG",
            )
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
            {"MLX_VLM_GLM5_MQA_FOLD": None, "MLX_VLM_GLM5_FUSED_MLA": "1",
             "MLX_VLM_GLM5_FUSED_MLA_MIN_TG": "1"}, x
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
            {"MLX_VLM_GLM5_MQA_FOLD": None, "MLX_VLM_GLM5_FUSED_MLA": "1",
             "MLX_VLM_GLM5_FUSED_MLA_MIN_TG": "1"}, x
        )
        self.assertEqual(n_k, 1, "fused kernel was NOT entered on the L==1 path")
        err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
        self.assertLess(err, 8 * u, f"fused decode err={err} ulp={u}")


    def test_small_launch_declines_the_kernel(self):
        """At the default threadgroup floor a tiny launch must fall through, not run slow.

        The test module is 8 heads / 1 batch, so an L==1 step asks for a single threadgroup.
        The kernel is tiled with no split-K and loses badly there (1.42 ms vs 0.31 ms measured
        at the real 64-head shape), so the guard must decline and hand the call back to MLX.
        """
        mx.random.seed(3)
        x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
        mx.eval(x)
        _, n_k, sh = self._run(
            {"MLX_VLM_GLM5_MQA_FOLD": None, "MLX_VLM_GLM5_FUSED_MLA": "1",
             "MLX_VLM_GLM5_FUSED_MLA_MIN_TG": None}, x
        )
        self.assertEqual(n_k, 0, "the launch-size guard did not decline a 1-threadgroup call")
        self.assertEqual(sh, [(1, 8, 1, 512)])

    # ------------------------------------------------------- the fold under a REAL mask
    # Adversarial review (Codex, relayed 2026-09-02) found that nothing exercised the fold with a
    # mask: the module tests run mask=None and the kernel tests take the Gk==G path. These call
    # _mqa_sdpa directly at the exact shapes the model produces, with the exact mask ranks.
    def _fold_vs_composite(self, L_mod, q, kv, mask, tol_ulps=4):
        """Assert fold == composite AND that the fold was actually entered.

        The shape assertion is the whole point: without it a silent decline makes the tolerance
        check vacuously true, which is exactly how the first version of the module test in this
        file passed while executing none of the new code.
        """
        from mlx_vlm.models.base import scaled_dot_product_attention as sdpa

        scale = 256 ** -0.5
        B, H = q.shape[0], q.shape[1]
        ref = sdpa(q, kv, kv, cache=None, scale=scale, mask=mask)
        mx.eval(ref)

        seen = []
        orig = L_mod.scaled_dot_product_attention

        def wrap(a, b, c, cache=None, scale=None, mask=None, **kw):
            seen.append(tuple(a.shape))
            return orig(a, b, c, cache=cache, scale=scale, mask=mask, **kw)

        L_mod.scaled_dot_product_attention = wrap
        L_mod._MQA_FOLD_ENV = True
        L_mod._FUSED_MLA_ENV = False
        try:
            got = L_mod._mqa_sdpa(q, kv, scale, mask)
            mx.eval(got)
        finally:
            L_mod.scaled_dot_product_attention = orig

        self.assertEqual(seen[:1], [(B, 1, H, q.shape[3])],
                         f"fold was NOT entered: inner query shape {seen[:1]}")
        self.assertEqual(got.shape, ref.shape)
        err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
        u = _ulp_bf16(ref.astype(mx.float32))
        self.assertLess(err, tol_ulps * u, f"fold err={err} ulp={u}")
        self.assertGreater(err, 0.0,
                           "bit-identical to the composite means the fold silently declined")
        return err, u

    def test_fold_declines_a_head_dependent_mask(self):
        """A [B, H, 1, N] mask cannot ride the fold unchanged, so it must fall through."""
        L = _reload(MLX_VLM_GLM5_MQA_FOLD="1", MLX_VLM_GLM5_FUSED_MLA=None)
        from mlx_vlm.models.base import scaled_dot_product_attention as sdpa

        mx.random.seed(11)
        q = mx.random.normal((2, 64, 1, 512)).astype(mx.bfloat16)
        kv = mx.random.normal((2, 1, 2048, 512)).astype(mx.bfloat16)
        m = mx.random.uniform(shape=(2, 64, 1, 2048)) > 0.3
        mx.eval(q, kv, m)
        L._MQA_FOLD_ENV = True
        L._FUSED_MLA_ENV = False
        got = L._mqa_sdpa(q, kv, 256 ** -0.5, m)
        ref = sdpa(q, kv, kv, cache=None, scale=256 ** -0.5, mask=m)
        mx.eval(got, ref)
        self.assertEqual(float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max()), 0.0,
                         "a head-dependent mask must be handed back to MLX bit-identically")

    def test_fold_with_bool_mask(self):
        L = _reload(MLX_VLM_GLM5_MQA_FOLD="1", MLX_VLM_GLM5_FUSED_MLA=None)
        mx.random.seed(7)
        q = mx.random.normal((2, 64, 1, 512)).astype(mx.bfloat16)
        kv = mx.random.normal((2, 1, 2048, 512)).astype(mx.bfloat16)
        m = mx.random.uniform(shape=(2, 1, 1, 2048)) > 0.3
        mx.eval(q, kv, m)
        self._fold_vs_composite(L, q, kv, m)

    def test_fold_with_additive_float_mask(self):
        L = _reload(MLX_VLM_GLM5_MQA_FOLD="1", MLX_VLM_GLM5_FUSED_MLA=None)
        mx.random.seed(8)
        q = mx.random.normal((2, 64, 1, 512)).astype(mx.bfloat16)
        kv = mx.random.normal((2, 1, 2048, 512)).astype(mx.bfloat16)
        # additive mask: 0 where visible, a large negative where not -- the shape
        # create_attention_mask produces for a float mask
        blocked = mx.random.uniform(shape=(2, 1, 1, 2048)) > 0.7
        m = mx.where(blocked, mx.array(-1e4, mx.bfloat16), mx.array(0.0, mx.bfloat16))
        mx.eval(q, kv, m)
        self._fold_vs_composite(L, q, kv, m)

    def test_fold_on_the_gathered_path_shape(self):
        """The per-chunk shape _gathered_attention actually builds: B*lc rows, one KV head."""
        L = _reload(MLX_VLM_GLM5_MQA_FOLD="1", MLX_VLM_GLM5_FUSED_MLA=None)
        mx.random.seed(9)
        q = mx.random.normal((256, 64, 1, 512)).astype(mx.bfloat16)
        kv = mx.random.normal((256, 1, 64, 512)).astype(mx.bfloat16)
        valid = mx.random.uniform(shape=(256, 1, 1, 64)) > 0.2
        valid = valid | (mx.arange(64)[None, None, None, :] == 0)   # never a fully-masked row
        mx.eval(q, kv, valid)
        self._fold_vs_composite(L, q, kv, valid)

    def test_fold_declines_the_causal_string_sentinel(self):
        """MLX accepts mask="causal"; the fold branch indexes mask.ndim, which a str lacks.

        create_attention_mask returns the sentinel whenever return_array is false. The glm5_next
        call sites pass return_array=True, so this is a contract fix rather than a live bug -- but
        it must hand back to MLX instead of raising AttributeError.
        """
        L = _reload(MLX_VLM_GLM5_MQA_FOLD="1", MLX_VLM_GLM5_FUSED_MLA="1",
                    MLX_VLM_GLM5_FUSED_MLA_MIN_TG="1")
        mx.random.seed(10)
        q = mx.random.normal((1, 64, 16, 512)).astype(mx.bfloat16)
        kv = mx.random.normal((1, 1, 64, 512)).astype(mx.bfloat16)
        mx.eval(q, kv)
        out = L._mqa_sdpa(q, kv, 256 ** -0.5, "causal")     # must not raise
        mx.eval(out)
        ref = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=256 ** -0.5, mask="causal")
        mx.eval(ref)
        self.assertEqual(
            float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()), 0.0,
            "the sentinel must be handed back to MLX unchanged, not rewritten")


if __name__ == "__main__":
    unittest.main()
