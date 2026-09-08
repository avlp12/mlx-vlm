"""Segment-aligned MoE padding (MLX_VLM_MOE_SEGMENT_ALIGN).

The lever: `affine_gather_qmm_rhs` tiles rows at BM=16 and runs a full K-loop per distinct
expert inside a tile, so a tile straddling an expert boundary pays the K-loop twice. Padding each
expert segment to a multiple of 16 removes the straddle.

These tests assert the two things that make it safe to ship:
  1. the REAL rows of the padded result are bit-identical to the unpadded result, and
  2. the path that actually ships is the data-dependent pad, NOT the static worst-case bound
     (which measured 1.015x at T=2048 against 1.130x -- it throws the win away).
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


class TestMoESegmentAlign(unittest.TestCase):
    E = 32          # small but > 1, so B/E >= 4 is reachable at modest row counts
    TOPK = 4
    K = 128
    N = 64

    def setUp(self):
        self.saved = {"MLX_VLM_MOE_SEGMENT_ALIGN": os.environ.get("MLX_VLM_MOE_SEGMENT_ALIGN")}

    def tearDown(self):
        _reload(**self.saved)

    # ---------------------------------------------------------------- flag
    def test_flag_defaults_off(self):
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN=None)
        self.assertEqual(S._moe_segment_align(), 0)

    def test_flag_parsing(self):
        for v, want in (("16", 16), ("1", 16), ("true", 16), ("on", 16),
                        ("0", 0), ("garbage", 0), ("32", 32)):
            S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN=v)
            self.assertEqual(S._moe_segment_align(), want, f"{v!r}")

    # ---------------------------------------------------------------- order construction
    def test_order_pad_is_segment_aligned_and_repeats_the_last_row(self):
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN="16")
        rng = np.random.default_rng(0)
        idx = np.sort(rng.integers(0, self.E, size=1024)).astype(np.uint32)
        order_pad, real_pos = S._segment_align_order(mx.array(idx), self.E, 16)
        mx.eval(order_pad, real_pos)
        op = np.array(order_pad)
        counts = np.bincount(idx, minlength=self.E)
        padded = ((counts + 15) // 16) * 16

        # every segment in the padded layout is a whole number of 16-row tiles
        self.assertEqual(len(op), int(padded.sum()))
        self.assertEqual(len(op) % 16, 0)
        # the expert of every padded row is right, and each segment starts on a tile boundary
        experts_of_padded = idx[op]
        pos = 0
        for e in range(self.E):
            if padded[e] == 0:
                continue
            self.assertEqual(pos % 16, 0, f"segment for expert {e} is not tile-aligned")
            seg = experts_of_padded[pos:pos + padded[e]]
            self.assertTrue((seg == e).all(), f"expert {e} segment is contaminated")
            pos += padded[e]
        # real_pos selects exactly the real rows, in their original sorted order
        self.assertEqual(len(real_pos), len(idx))
        self.assertTrue((idx[op[np.array(real_pos)]] == idx).all())

    # ---------------------------------------------------------------- bit-exactness
    def _bitexact_at(self, R):
        rng = np.random.default_rng(R)
        idx_np = rng.integers(0, self.E, size=R).astype(np.uint32)
        indices = mx.array(idx_np.reshape(R // self.TOPK, self.TOPK))
        x = mx.random.normal((R // self.TOPK, self.K)).astype(mx.bfloat16)
        wq = mx.random.normal((self.E, self.N, self.K)).astype(mx.bfloat16)
        w, scales, biases = mx.quantize(wq, group_size=64, bits=4)
        mx.eval(x, indices, w, scales, biases)

        rows = {}

        def run(S, tag):
            xx = mx.expand_dims(x, (-2, -3))
            xs, ids, inv = S._gather_sort(xx, indices, num_experts=self.E)
            rows[tag] = xs.shape[0]
            o = mx.gather_qmm(xs, w, scales=scales, biases=biases, rhs_indices=ids,
                              transpose=True, group_size=64, bits=4, sorted_indices=True)
            return S._scatter_unsort(o, inv, indices.shape).squeeze(-2)

        base = run(_reload(MLX_VLM_MOE_SEGMENT_ALIGN="0"), "off")
        pad = run(_reload(MLX_VLM_MOE_SEGMENT_ALIGN="16"), "on")
        mx.eval(base, pad)
        # NON-VACUITY: if padding silently declined, both arms are the same computation and the
        # bit-exactness assertion below is trivially true. Assert the row count actually grew,
        # and grew to exactly sum_e ceil(n_e/16)*16.
        counts = np.bincount(idx_np, minlength=self.E)
        self.assertEqual(rows["off"], R)
        self.assertEqual(rows["on"], int((((counts + 15) // 16) * 16).sum()))
        self.assertGreater(rows["on"], rows["off"],
                           f"padding did not fire at R={R}; the bit-exact check would be vacuous")
        self.assertEqual(base.shape, pad.shape)
        return base, pad

    def test_bit_exact_three_sizes(self):
        for R in (2048, 8192, 16384):        # all satisfy B/E >= 4 at E=32
            base, pad = self._bitexact_at(R)
            self.assertTrue(
                bool(mx.all(base == pad).item()),
                f"padded result is not bit-identical to the unpadded one at R={R}; "
                f"max|d|={float(mx.abs(base.astype(mx.float32) - pad.astype(mx.float32)).max())}",
            )

    # ---------------------------------------------------------------- the shipped path
    def test_ships_the_data_dependent_pad_not_the_static_bound(self):
        """R_pad must be sum_e ceil(n_e/16)*16, NOT the static worst case R + 15E.

        The static bound needs no host sync, which is tempting, but it measured 1.015x at
        T=2048 against 1.130x for the data-dependent pad -- it pads so hard it throws the win
        away. This test fails if somebody swaps it in to remove the sync.
        """
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN="16")
        rng = np.random.default_rng(3)
        R = 8192
        idx = np.sort(rng.integers(0, self.E, size=R)).astype(np.uint32)
        order_pad, _ = S._segment_align_order(mx.array(idx), self.E, 16)
        mx.eval(order_pad)
        counts = np.bincount(idx, minlength=self.E)
        expected = int((((counts + 15) // 16) * 16).sum())
        static_bound = R + 15 * self.E
        self.assertEqual(len(order_pad), expected)
        self.assertLess(len(order_pad), static_bound,
                        "R_pad reached the static worst-case bound -- the win is gone")

    def test_declines_when_the_fast_kernel_would_not_be_reached(self):
        """Below B/E >= 4 mlx does not take affine_gather_qmm_rhs, so padding adds rows for
        nothing. _gather_sort must fall through to the unpadded layout there."""
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN="16")
        R = 4 * self.E - 4                       # just under the threshold
        rng = np.random.default_rng(5)
        indices = mx.array(rng.integers(0, self.E, size=R)
                           .astype(np.uint32).reshape(R // self.TOPK, self.TOPK))
        x = mx.random.normal((R // self.TOPK, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)
        xs, ids, inv = S._gather_sort(mx.expand_dims(x, (-2, -3)), indices,
                                      num_experts=self.E)
        mx.eval(xs, ids, inv)
        self.assertEqual(xs.shape[0], R, "padding was applied below the B/E >= 4 threshold")

    def test_num_experts_omitted_means_no_padding(self):
        """Callers that do not pass num_experts keep MLX's original behaviour exactly."""
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN="16")
        rng = np.random.default_rng(6)
        R = 4096
        indices = mx.array(rng.integers(0, self.E, size=R)
                           .astype(np.uint32).reshape(R // self.TOPK, self.TOPK))
        x = mx.random.normal((R // self.TOPK, self.K)).astype(mx.bfloat16)
        mx.eval(x, indices)
        xs, _, _ = S._gather_sort(mx.expand_dims(x, (-2, -3)), indices)
        mx.eval(xs)
        self.assertEqual(xs.shape[0], R)


    def test_bit_exact_with_a_batch_axis(self):
        """indices arrive as [B, T, topk] under batched prefill, not [T, topk].

        _gather_sort flattens before sorting so the padding is shape-agnostic, but "should be"
        is not a test: batched serving is where this flag is most likely to be turned on and
        least likely to be noticed if it is subtly wrong.
        """
        for B in (2, 8):
            T, R = 64, None
            rng = np.random.default_rng(100 + B)
            idx_np = rng.integers(0, self.E, size=(B, T, self.TOPK)).astype(np.uint32)
            indices = mx.array(idx_np)
            R = int(indices.size)
            x = mx.random.normal((B, T, self.K)).astype(mx.bfloat16)
            wq = mx.random.normal((self.E, self.N, self.K)).astype(mx.bfloat16)
            w, scales, biases = mx.quantize(wq, group_size=64, bits=4)
            mx.eval(x, indices, w, scales, biases)

            rows = {}

            def run(S, tag):
                xx = mx.expand_dims(x, (-2, -3))
                xs, ids, inv = S._gather_sort(xx, indices, num_experts=self.E)
                rows[tag] = xs.shape[0]
                o = mx.gather_qmm(xs, w, scales=scales, biases=biases, rhs_indices=ids,
                                  transpose=True, group_size=64, bits=4, sorted_indices=True)
                return S._scatter_unsort(o, inv, indices.shape).squeeze(-2)

            base = run(_reload(MLX_VLM_MOE_SEGMENT_ALIGN="0"), "off")
            pad = run(_reload(MLX_VLM_MOE_SEGMENT_ALIGN="16"), "on")
            mx.eval(base, pad)
            counts = np.bincount(idx_np.reshape(-1), minlength=self.E)
            self.assertEqual(rows["off"], R)
            self.assertEqual(rows["on"], int((((counts + 15) // 16) * 16).sum()))
            self.assertGreater(rows["on"], rows["off"], f"padding did not fire at B={B}")
            self.assertEqual(base.shape, pad.shape)
            self.assertEqual(base.shape[:2], (B, T), "batch/sequence axes were not restored")
            self.assertTrue(bool(mx.all(base == pad).item()),
                            f"not bit-identical at B={B}")


class TestMoESegmentAlign32(unittest.TestCase):
    """K2 (V5, 2026-09-08): align 32 is the value that matches the shipped tile.

    ``GatherQMM::eval_gpu`` uses bm = 32 (not 16) whenever M/E >= 32 and group_size >= 64
    (mlx quantized.cpp:1697-1701), which is every prefill chunk of this model (M/E = 227.6).
    Padding to 16 therefore leaves ~half the straddles AND breaks ``align_M`` (67,568 % 32
    == 16); padding to 32 removes every straddle and restores ``align_M``.  These tests pin
    the two properties that make the arm shippable: the row layout is what the tile wants,
    and the real rows are still bit-identical to the unpadded ones.
    """

    E = 32
    TOPK = 4
    K = 128
    N = 64

    def setUp(self):
        self.saved = {"MLX_VLM_MOE_SEGMENT_ALIGN": os.environ.get("MLX_VLM_MOE_SEGMENT_ALIGN")}

    def tearDown(self):
        _reload(**self.saved)

    def test_32_is_accepted_by_the_existing_knob(self):
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN="32")
        self.assertEqual(S._moe_segment_align(), 32)

    def test_every_segment_is_a_whole_number_of_32_row_tiles(self):
        """The straddle count is zero and R_pad % 32 == 0, i.e. align_M is restored."""
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN="32")
        rng = np.random.default_rng(11)
        idx = np.sort(rng.integers(0, self.E, size=8192)).astype(np.uint32)
        order_pad, real_pos = S._segment_align_order(mx.array(idx), self.E, 32)
        mx.eval(order_pad, real_pos)
        op = np.array(order_pad)
        counts = np.bincount(idx, minlength=self.E)
        padded = ((counts + 31) // 32) * 32
        self.assertEqual(len(op), int(padded.sum()))
        self.assertEqual(len(op) % 32, 0, "R_pad is not a multiple of bm=32: align_M stays false")
        experts_of_padded = idx[op]
        pos = 0
        for e in range(self.E):
            if padded[e] == 0:
                continue
            self.assertEqual(pos % 32, 0, f"segment for expert {e} is not bm=32 aligned")
            seg = experts_of_padded[pos:pos + padded[e]]
            self.assertTrue((seg == e).all(), f"expert {e} segment is contaminated")
            pos += padded[e]

    def test_only_align_32_guarantees_the_shipped_tile_is_aligned(self):
        """align_M under bm=32: always true at align 32, a coin flip at align 16.

        R_pad(32) is a sum of multiples of 32, so ``align_M`` holds by construction.
        R_pad(16) is a multiple of 32 only when an EVEN number of experts happen to have
        an odd count of 16-row tiles -- so on some draws align 16 accidentally aligns and
        on others it does not.  Both halves are asserted: 32 never fails, and 16 fails on
        at least one ordinary draw.
        """
        rng = np.random.default_rng(11)
        misaligned_16 = 0
        for _ in range(32):
            idx = rng.integers(0, self.E, size=8192)
            counts = np.bincount(idx, minlength=self.E)
            r16 = int((((counts + 15) // 16) * 16).sum())
            r32 = int((((counts + 31) // 32) * 32).sum())
            self.assertEqual(r32 % 32, 0)
            self.assertGreaterEqual(r32, r16)   # 32 pads at least as hard: the memory cost
            misaligned_16 += int(r16 % 32 != 0)
        self.assertGreater(misaligned_16, 0,
                           "align 16 never broke align_M in 32 draws -- re-check the model")

    def _run_arm(self, align, x, indices, w, scales, biases):
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN=align)
        xx = mx.expand_dims(x, (-2, -3))
        xs, ids, inv = S._gather_sort(xx, indices, num_experts=self.E)
        o = mx.gather_qmm(xs, w, scales=scales, biases=biases, rhs_indices=ids,
                          transpose=True, group_size=64, bits=4, sorted_indices=True)
        return S._scatter_unsort(o, inv, indices.shape).squeeze(-2), xs.shape[0]

    def test_bit_exact_against_no_padding_and_against_align_16(self):
        for R in (2048, 8192, 16384):
            rng = np.random.default_rng(R + 1)
            idx_np = rng.integers(0, self.E, size=R).astype(np.uint32)
            indices = mx.array(idx_np.reshape(R // self.TOPK, self.TOPK))
            x = mx.random.normal((R // self.TOPK, self.K)).astype(mx.bfloat16)
            wq = mx.random.normal((self.E, self.N, self.K)).astype(mx.bfloat16)
            w, scales, biases = mx.quantize(wq, group_size=64, bits=4)
            mx.eval(x, indices, w, scales, biases)

            off, rows_off = self._run_arm("0", x, indices, w, scales, biases)
            a16, rows_16 = self._run_arm("16", x, indices, w, scales, biases)
            a32, rows_32 = self._run_arm("32", x, indices, w, scales, biases)
            mx.eval(off, a16, a32)

            counts = np.bincount(idx_np, minlength=self.E)
            self.assertEqual(rows_off, R)
            self.assertEqual(rows_32, int((((counts + 31) // 32) * 32).sum()))
            self.assertGreater(rows_32, rows_16,
                               f"align 32 did not pad more than 16 at R={R}: check is vacuous")
            self.assertTrue(bool(mx.all(off == a32).item()),
                            f"align 32 is not bit-identical to unpadded at R={R}; "
                            f"max|d|={float(mx.abs(off.astype(mx.float32) - a32.astype(mx.float32)).max())}")
            self.assertTrue(bool(mx.all(a16 == a32).item()),
                            f"align 32 differs from align 16 at R={R}")

    def test_bit_exact_with_a_batch_axis(self):
        for B in (2, 8):
            T = 64
            rng = np.random.default_rng(200 + B)
            idx_np = rng.integers(0, self.E, size=(B, T, self.TOPK)).astype(np.uint32)
            indices = mx.array(idx_np)
            x = mx.random.normal((B, T, self.K)).astype(mx.bfloat16)
            wq = mx.random.normal((self.E, self.N, self.K)).astype(mx.bfloat16)
            w, scales, biases = mx.quantize(wq, group_size=64, bits=4)
            mx.eval(x, indices, w, scales, biases)
            off, rows_off = self._run_arm("0", x, indices, w, scales, biases)
            a32, rows_32 = self._run_arm("32", x, indices, w, scales, biases)
            mx.eval(off, a32)
            self.assertEqual(rows_off, int(indices.size))
            self.assertGreater(rows_32, rows_off, f"padding did not fire at B={B}")
            self.assertEqual(off.shape[:2], (B, T))
            self.assertTrue(bool(mx.all(off == a32).item()), f"not bit-identical at B={B}")


if __name__ == "__main__":
    unittest.main()
