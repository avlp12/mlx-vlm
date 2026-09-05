"""MoE dense-prefill toggle (MLX_VLM_GLM5_MOE_DENSE_PREFILL) for QuantizedSwitchLinear.

The lever: at long-prefill chunk sizes GLM-5.3-Flash's 288-expert/top-8 router sends many
rows to each touched expert, and the hypothesis is that a one-shot bf16 dequant of just the
SELECTED experts + dense `gather_mm` beats the quantized `gather_qmm` kernel there, at the
cost of a transient dequantized buffer. These tests are CPU-only correctness/gating checks;
no GPU timing is asserted here (see the report for the timing harness command).
"""

import importlib
import os
import unittest

import mlx.core as mx
import numpy as np


def _reload(**env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    import mlx_vlm.models.switch_layers as S

    importlib.reload(S)
    return S


class TestGlm5MoEDensePrefill(unittest.TestCase):
    E = 32       # experts
    TOPK = 8
    K = 128      # input_dims (hidden)
    N = 64       # output_dims (moe_intermediate, kept small for CPU speed)

    ENV_KEYS = (
        "MLX_VLM_GLM5_MOE_DENSE_PREFILL",
        "MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS",
        "MLX_VLM_GLM5_MOE_DENSE_MAX_BYTES",
    )

    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in self.ENV_KEYS}

    def tearDown(self):
        _reload(**self.saved)

    def _make_ql(self, S, seed=0):
        rng = np.random.default_rng(seed)
        w_full = mx.array(rng.normal(size=(self.E, self.N, self.K)).astype(np.float32)).astype(
            mx.bfloat16
        )
        ql = S.QuantizedSwitchLinear(self.K, self.N, self.E, bias=True, group_size=64, bits=4)
        w, scales, *biases = mx.quantize(w_full, group_size=64, bits=4, mode="affine")
        ql.weight = w
        ql.scales = scales
        ql.biases = biases[0] if biases else None
        ql.bias = mx.array(rng.normal(size=(self.E, self.N)).astype(np.float32)).astype(
            mx.bfloat16
        )
        mx.eval(ql.parameters())
        return ql

    def _indices(self, rows, seed):
        rng = np.random.default_rng(seed)
        idx_np = rng.integers(0, self.E, size=rows).astype(np.uint32)
        return mx.array(idx_np.reshape(rows // self.TOPK, self.TOPK))

    def _reference(self, S, ql, x, indices):
        """The gather_qmm path, called directly (bypasses the dense-toggle branch)."""
        y = mx.gather_qmm(
            x,
            ql["weight"],
            ql["scales"],
            ql.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=ql.group_size,
            bits=ql.bits,
            mode=ql.mode,
            sorted_indices=False,
        )
        y = y + mx.expand_dims(ql["bias"][indices], -2)
        return y

    # ------------------------------------------------------------------ flags
    def test_flag_defaults_off(self):
        S = _reload(MLX_VLM_GLM5_MOE_DENSE_PREFILL=None)
        self.assertFalse(S._moe_dense_prefill_enabled())

    def test_flag_parsing(self):
        for v, want in (("1", True), ("true", True), ("on", True), ("yes", True),
                        ("0", False), ("false", False), ("garbage", False), (None, False)):
            S = _reload(MLX_VLM_GLM5_MOE_DENSE_PREFILL=v)
            self.assertEqual(S._moe_dense_prefill_enabled(), want, f"{v!r}")

    def test_min_rows_default_and_parsing(self):
        S = _reload(MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS=None)
        self.assertEqual(S._moe_dense_min_rows(), 32)
        S = _reload(MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS="64")
        self.assertEqual(S._moe_dense_min_rows(), 64)
        S = _reload(MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS="garbage")
        self.assertEqual(S._moe_dense_min_rows(), 32)

    def test_max_bytes_default_and_parsing(self):
        S = _reload(MLX_VLM_GLM5_MOE_DENSE_MAX_BYTES=None)
        self.assertEqual(S._moe_dense_max_bytes(), 16 * (1024**3))
        S = _reload(MLX_VLM_GLM5_MOE_DENSE_MAX_BYTES=str(1024))
        self.assertEqual(S._moe_dense_max_bytes(), 1024)

    # ------------------------------------------------------------------ correctness
    def test_dense_matches_quantized_within_tolerance(self):
        """Above MIN_ROWS, the dense path (remap + bulk dequant + gather_mm) should
        closely match gather_qmm's own 4-bit accumulation. They are NOT expected to be
        bit-identical: gather_qmm dequantizes and accumulates per-tile in its own
        kernel, while the dense path does one bulk mx.dequantize then a separate
        gather_mm -- same math, different rounding path. Report max abs diff; assert it
        stays small relative to activation scale (not a chunky mismatch, e.g. a wrong
        expert remap or transpose).

        CPU-BACKEND CAVEAT: called with dtype=mx.float32, not the production bf16 --
        mlx 0.32.1's CPU GatherMM only implements float32 (RuntimeError on bf16). This
        verifies the remap/dequant/matmul LOGIC on CPU; production bf16 numerics and
        timing on Metal are unverified by this test (see the report)."""
        S = _reload(
            MLX_VLM_GLM5_MOE_DENSE_PREFILL="1",
            MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS="4",
            MLX_VLM_GLM5_MOE_DENSE_MAX_BYTES=str(16 * (1024**3)),
        )
        ql = self._make_ql(S, seed=1)
        rows = self.E * 8  # rows_per_expert = 8 >= MIN_ROWS=4
        indices = self._indices(rows, seed=2)
        x = mx.random.normal((rows // self.TOPK, 1, 1, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)

        dense = S._dense_selected_expert_gather_mm(
            x, ql, indices, False, S._moe_dense_max_bytes(), dtype=mx.float32
        )
        ref = self._reference(S, ql, x, indices)
        mx.eval(dense, ref)

        diff = mx.abs(dense.astype(mx.float32) - ref.astype(mx.float32))
        max_abs_diff = float(diff.max())
        scale = float(mx.abs(ref.astype(mx.float32)).max())
        print(f"[dense-vs-qmm] max_abs_diff={max_abs_diff:.6f} ref_max_abs={scale:.6f}")
        # gather_qmm itself dequantizes 4-bit weights and accumulates in fp32 internally,
        # so both arms trace back to the same 4-bit values; divergence should be at the
        # float rounding level, not systematic. Generous bound to catch real bugs (wrong
        # expert remap, wrong transpose, wrong bias) rather than claim tight parity.
        self.assertLess(max_abs_diff, 0.05 * max(scale, 1.0))
        self.assertEqual(dense.shape, ref.shape)

    def test_dense_actually_fires_above_threshold(self):
        """Non-vacuity: at rows_per_expert >= MIN_ROWS the dense path must actually run,
        not silently fall back (which would make the tolerance test above vacuous)."""
        S = _reload(
            MLX_VLM_GLM5_MOE_DENSE_PREFILL="1",
            MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS="4",
        )
        ql = self._make_ql(S, seed=1)
        rows = self.E * 8
        indices = self._indices(rows, seed=2)
        x = mx.random.normal((rows // self.TOPK, 1, 1, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)

        dense_direct = S._dense_selected_expert_gather_mm(
            x, ql, indices, False, S._moe_dense_max_bytes(), dtype=mx.float32
        )
        self.assertIsNotNone(dense_direct, "dense path declined above MIN_ROWS threshold")

    # ------------------------------------------------------------------ fallbacks
    def test_min_rows_threshold_falls_back_to_qmm_bit_exact(self):
        """Below MIN_ROWS, QuantizedSwitchLinear.__call__ must take the gather_qmm
        branch untouched -- bit-exact against the reference, not just close."""
        S = _reload(
            MLX_VLM_GLM5_MOE_DENSE_PREFILL="1",
            MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS="64",  # rows_per_expert will be well below this
        )
        ql = self._make_ql(S, seed=1)
        rows = self.E * 2  # rows_per_expert = 2 < 64
        indices = self._indices(rows, seed=3)
        x = mx.random.normal((rows // self.TOPK, 1, 1, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)

        out = ql(x, indices, sorted_indices=False)
        ref = self._reference(S, ql, x, indices)
        mx.eval(out, ref)
        self.assertTrue(bool(mx.all(out == ref).item()), "fell through to dense below MIN_ROWS")

    def test_max_bytes_cap_falls_back_to_qmm_bit_exact(self):
        """A byte cap too small for even one expert must decline the dense path and take
        gather_qmm, bit-exact."""
        S = _reload(
            MLX_VLM_GLM5_MOE_DENSE_PREFILL="1",
            MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS="4",
            MLX_VLM_GLM5_MOE_DENSE_MAX_BYTES="1",  # smaller than one expert's dequant buffer
        )
        ql = self._make_ql(S, seed=1)
        rows = self.E * 8
        indices = self._indices(rows, seed=2)
        x = mx.random.normal((rows // self.TOPK, 1, 1, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)

        out = ql(x, indices, sorted_indices=False)
        ref = self._reference(S, ql, x, indices)
        mx.eval(out, ref)
        self.assertTrue(bool(mx.all(out == ref).item()), "byte cap did not gate the dense path")

    def test_toggle_default_off_is_bit_exact_to_qmm(self):
        """With the toggle at its default (unset), behavior must be identical to before
        this lever existed -- bit-exact, regardless of row count."""
        S = _reload(**{k: None for k in self.ENV_KEYS})
        self.assertFalse(S._moe_dense_prefill_enabled())
        ql = self._make_ql(S, seed=1)
        rows = self.E * 16  # deliberately well above any plausible MIN_ROWS
        indices = self._indices(rows, seed=4)
        x = mx.random.normal((rows // self.TOPK, 1, 1, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)

        out = ql(x, indices, sorted_indices=False)
        ref = self._reference(S, ql, x, indices)
        mx.eval(out, ref)
        self.assertTrue(bool(mx.all(out == ref).item()))

    # ------------------------------------------------------------------ decode / verify untouched
    def test_decode_s1_untouched_even_with_toggle_on(self):
        """S=1 decode: indices.size == top_k, rows_per_expert = top_k/E, far below any
        sane MIN_ROWS. Must be bit-exact to gather_qmm even with the toggle ON."""
        S = _reload(MLX_VLM_GLM5_MOE_DENSE_PREFILL="1", MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS="32")
        ql = self._make_ql(S, seed=1)
        indices = self._indices(self.TOPK, seed=5)  # 1 "token" worth of top-k routing
        x = mx.random.normal((1, 1, 1, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)

        out = ql(x, indices, sorted_indices=False)
        ref = self._reference(S, ql, x, indices)
        mx.eval(out, ref)
        self.assertTrue(bool(mx.all(out == ref).item()))

    def test_speculative_verify_s8_untouched_even_with_toggle_on(self):
        """S<=8 verify block: indices.size <= 8*top_k, still far below MIN_ROWS*E for a
        288-expert-scale router. Must be bit-exact to gather_qmm even with the toggle ON."""
        S = _reload(MLX_VLM_GLM5_MOE_DENSE_PREFILL="1", MLX_VLM_GLM5_MOE_DENSE_MIN_ROWS="32")
        ql = self._make_ql(S, seed=1)
        rows = 8 * self.TOPK  # S=8 verify block
        indices = self._indices(rows, seed=6)
        x = mx.random.normal((rows // self.TOPK, 1, 1, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)

        out = ql(x, indices, sorted_indices=False)
        ref = self._reference(S, ql, x, indices)
        mx.eval(out, ref)
        self.assertTrue(bool(mx.all(out == ref).item()))


if __name__ == "__main__":
    unittest.main()
