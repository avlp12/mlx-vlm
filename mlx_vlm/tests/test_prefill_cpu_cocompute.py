"""CPU co-compute for the prefill MoE expert GEMMs (MLX_VLM_GLM5_PREFILL_CPU_FRACTION).

The lever: prefill is GPU-COMPUTE-bound (~58 % of the measured 23.5 TFLOP/s 4-bit
qmm peak) while 32 AMX cores sit idle at a measured 4.6-5.3 TFLOP/s on the same
bf16 GEMM shapes, and the routed experts are 38.0 % of prefill wall (L7-c).  This
path hands a leading slice of each chunk's expert-sorted rows to a CPU stream.

These tests assert the four things that make it safe to ship behind a default-off
knob, all on the CPU (no model, no server, no GPU):

  1. the knob is OFF by default and parses defensively (a typo can never raise
     inside a request, and can never silently enable the path);
  2. at fraction 0 the emitted result is BYTE-identical to the arm without this
     lever, and the plan builder -- with its one host sync -- is never even called;
  3. the split is taken at a real expert boundary on the ROW cumsum, so a skewed
     router still yields the requested share of rows, and decode-sized calls are
     excluded by the row floor;
  4. with the knob on, the two-arm result matches the single-arm result within the
     tolerance the mechanism allows (the CPU arm rounds the weights to bf16 before
     the GEMM, so this is NOT bit-identity and the campaign KL gate,
     docs/PERF_BASELINE_B0_2026-09-05.md rule 2, is the real gate), and the CPU arm
     really is issued on its own stream.

Run: MLX_DEFAULT_DEVICE=cpu pytest mlx_vlm/tests/test_prefill_cpu_cocompute.py
"""

import importlib
import os
import unittest

import mlx.core as mx
import numpy as np

import mlx_vlm.models.switch_layers as S

ENV_KEYS = (
    "MLX_VLM_GLM5_PREFILL_CPU_FRACTION",
    "MLX_VLM_GLM5_PREFILL_CPU_MIN_ROWS",
    "MLX_VLM_GLM5_PREFILL_CPU_DTYPE",
    "MLX_VLM_MOE_SEGMENT_ALIGN",
)


def _reload(**env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    global S
    S = importlib.reload(S)
    return S


class _EnvCase(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in ENV_KEYS}

    def tearDown(self):
        _reload(**self.saved)


# --------------------------------------------------------------------------- knob
class TestKnob(_EnvCase):
    def test_defaults_off(self):
        s = _reload(**{k: None for k in ENV_KEYS})
        self.assertEqual(s._prefill_cpu_fraction(), 0.0)
        self.assertEqual(s._prefill_cpu_min_rows(), s.DEFAULT_PREFILL_CPU_MIN_ROWS)
        self.assertIsNone(s._prefill_cpu_dtype())

    def test_fraction_parsing_and_clamp(self):
        for raw, want in (
            ("0", 0.0), ("0.0", 0.0), ("0.2", 0.2), ("0.5", 0.5),
            ("0.9", 0.5),          # clamped to MAX_PREFILL_CPU_FRACTION
            ("1", 0.5),
            ("-0.3", 0.0),         # negative reads as off, never as a reversed split
            ("nan", 0.0),          # NaN reads as off, not as a truthy float
            ("", 0.0), ("garbage", 0.0), ("0.2x", 0.0),
        ):
            s = _reload(MLX_VLM_GLM5_PREFILL_CPU_FRACTION=raw)
            self.assertAlmostEqual(s._prefill_cpu_fraction(), want, msg=f"{raw!r}")

    def test_min_rows_parsing(self):
        for raw, want in (("1024", 1024), ("1", 1), ("0", 1), ("-5", 1),
                          ("junk", S.DEFAULT_PREFILL_CPU_MIN_ROWS)):
            s = _reload(MLX_VLM_GLM5_PREFILL_CPU_MIN_ROWS=raw)
            self.assertEqual(s._prefill_cpu_min_rows(), want, f"{raw!r}")

    def test_dtype_parsing(self):
        for raw, want in (("bf16", mx.bfloat16), ("bfloat16", mx.bfloat16),
                          ("f32", mx.float32), ("float32", mx.float32),
                          ("fp16", mx.float16),
                          ("auto", None), ("", None), ("nonsense", None)):
            s = _reload(MLX_VLM_GLM5_PREFILL_CPU_DTYPE=raw)
            self.assertEqual(s._prefill_cpu_dtype(), want, f"{raw!r}")

    def test_stream_is_a_distinct_cpu_stream(self):
        s = _reload(**{k: None for k in ENV_KEYS})
        st = s._prefill_cpu_stream()
        self.assertEqual(st.device, mx.cpu)
        self.assertIs(st, s._prefill_cpu_stream())  # cached, not re-created per call
        self.assertNotEqual(st, mx.default_stream(mx.default_device()))


# --------------------------------------------------------------------------- plan
class TestPlan(_EnvCase):
    E = 32

    def _idx(self, n, seed=0, skew=0.0):
        """Sorted expert ids; ``skew`` is the Zipf exponent (0 = uniform).

        0.5 is the draw the MoE segment-align lever was validated against, and is
        milder than the real GLM-5.3 router looks at chunk 8192.
        """
        rng = np.random.default_rng(seed)
        if skew:
            p = 1.0 / (1.0 + np.arange(self.E)) ** skew
            raw = rng.choice(self.E, size=n, p=p / p.sum())
        else:
            raw = rng.integers(0, self.E, size=n)
        return mx.array(np.sort(raw).astype(np.uint32))

    def test_none_when_off(self):
        s = _reload(**{k: None for k in ENV_KEYS})
        self.assertIsNone(s._cpu_cocompute_plan(self._idx(8192), self.E))

    def test_none_below_row_floor(self):
        s = _reload(**{k: None for k in ENV_KEYS})
        # 8 rows is a B=1 decode step; 65,536 is an 8192-token prefill chunk.
        self.assertIsNone(
            s._cpu_cocompute_plan(self._idx(64), self.E, fraction=0.2, min_rows=4096)
        )
        self.assertIsNotNone(
            s._cpu_cocompute_plan(self._idx(8192), self.E, fraction=0.2, min_rows=4096)
        )

    def test_cut_is_an_expert_boundary_and_under_target(self):
        s = _reload(**{k: None for k in ENV_KEYS})
        for skew in (0.0, 0.5):
            for frac in (0.2, 0.3, 0.5):
                idx = self._idx(8192, seed=1, skew=skew)
                plan = s._cpu_cocompute_plan(idx, self.E, fraction=frac, min_rows=1)
                self.assertIsNotNone(plan, f"skew={skew} frac={frac}")
                counts = np.bincount(np.array(idx), minlength=self.E)
                # rows is exactly the cumulative count of the leading experts
                self.assertEqual(plan["rows"], int(counts[: plan["experts"]].sum()))
                self.assertEqual(int(plan["counts"].sum()), plan["rows"])
                self.assertEqual(len(plan["counts"]), plan["experts"])
                # never over-commits the CPU
                self.assertLessEqual(plan["rows"], int(frac * 8192))
                # and the cut really is at a boundary: adding one more expert
                # would have exceeded the target
                nxt = plan["rows"] + int(counts[plan["experts"]])
                self.assertGreater(nxt, int(frac * 8192))

    def test_row_share_tracks_fraction_under_skew(self):
        """The cut is on rows, not on expert count -- so a skewed router still
        gives ~the requested share of ROWS (the load balance that matters)."""
        s = _reload(**{k: None for k in ENV_KEYS})
        idx = self._idx(65536, seed=7, skew=0.5)
        plan = s._cpu_cocompute_plan(idx, self.E, fraction=0.2, min_rows=1)
        share = plan["rows"] / 65536
        self.assertLessEqual(share, 0.2)
        self.assertGreater(share, 0.10, "cut collapsed far below the requested share")

    def test_declines_rather_than_overcommitting_under_extreme_skew(self):
        """If the FIRST expert alone already exceeds the target there is no
        boundary under it, and the plan declines (single-arm path) rather than
        handing the CPU more than it was asked for -- overshoot is the one failure
        mode that can only lose, because the CPU is ~4x slower per FLOP."""
        s = _reload(**{k: None for k in ENV_KEYS})
        idx = self._idx(8192, seed=1, skew=1.5)  # expert 0 holds >10 % of rows
        self.assertIsNone(s._cpu_cocompute_plan(idx, self.E, fraction=0.1, min_rows=1))
        self.assertIsNotNone(s._cpu_cocompute_plan(idx, self.E, fraction=0.5, min_rows=1))

    def test_none_when_the_cut_would_take_every_expert(self):
        # The knob caps at 0.5, but the helper takes an explicit fraction, so drive
        # the degenerate case directly: a target at or above the total row count
        # would leave the accelerator arm empty.
        s = _reload(**{k: None for k in ENV_KEYS})
        idx = self._idx(8192)
        self.assertIsNone(s._cpu_cocompute_plan(idx, self.E, fraction=1.5, min_rows=1))
        self.assertIsNone(s._cpu_cocompute_plan(idx, self.E, fraction=1.0, min_rows=1))


# --------------------------------------------------------------------------- segment mm
class TestSegmentMM(_EnvCase):
    def test_matches_a_row_by_row_reference(self):
        s = _reload(**{k: None for k in ENV_KEYS})
        rng = np.random.default_rng(3)
        E, K, N = 6, 64, 32
        counts = np.array([3, 0, 5, 1, 0, 4])
        R = int(counts.sum())
        x = mx.array(rng.standard_normal((R, K)).astype(np.float32))
        w = mx.array(rng.standard_normal((E, N, K)).astype(np.float32))
        b = mx.array(rng.standard_normal((E, N)).astype(np.float32))

        got = s._cpu_segment_mm(x, w, counts, b)
        expert_of_row = np.repeat(np.arange(E), counts)
        ref = np.stack(
            [
                np.asarray(x[r]) @ np.asarray(w[expert_of_row[r]]).T
                + np.asarray(b[expert_of_row[r]])
                for r in range(R)
            ]
        )
        self.assertEqual(got.shape, (R, N))
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-4, atol=1e-4)

    def test_skips_empty_experts_and_returns_none_when_all_empty(self):
        s = _reload(**{k: None for k in ENV_KEYS})
        x = mx.zeros((0, 8))
        w = mx.zeros((3, 4, 8))
        self.assertIsNone(s._cpu_segment_mm(x, w, np.array([0, 0, 0])))


# --------------------------------------------------------------------------- end to end
def _build_switch_glu(s, *, quantized, E=32, K=128, N=64, seed=0):
    mx.random.seed(seed)
    glu = s.SwitchGLU(K, N, E, bias=False)
    if quantized:
        glu.gate_proj = glu.gate_proj.to_quantized(group_size=64, bits=4)
        glu.up_proj = glu.up_proj.to_quantized(group_size=64, bits=4)
        glu.down_proj = glu.down_proj.to_quantized(group_size=64, bits=4)
    glu.eval()
    return glu


def _inputs(E, K, tokens=256, top_k=4, seed=1, dtype=mx.float32):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, tokens, K)).astype(np.float32)).astype(dtype)
    inds = mx.array(
        np.stack([rng.choice(E, size=top_k, replace=False) for _ in range(tokens)])
        .astype(np.uint32)[None]
    )
    return x, inds


class TestEndToEnd(_EnvCase):
    E, K, N = 32, 128, 64

    def test_default_off_is_byte_identical(self):
        """OFF must not merely agree -- it must emit the same bytes, and must not
        even build the plan (whose host sync is the thing OFF is paying for)."""
        s = _reload(**{k: None for k in ENV_KEYS})
        glu = _build_switch_glu(s, quantized=True, E=self.E, K=self.K, N=self.N)
        x, inds = _inputs(self.E, self.K)

        ref = np.asarray(glu(x, inds).astype(mx.float32))

        called = []
        real_plan = s._cpu_cocompute_plan
        s._cpu_cocompute_plan = lambda *a, **k: called.append(1) or real_plan(*a, **k)
        try:
            got = np.asarray(glu(x, inds).astype(mx.float32))
        finally:
            s._cpu_cocompute_plan = real_plan

        self.assertEqual(called, [], "plan builder ran with the knob off")
        self.assertEqual(got.tobytes(), ref.tobytes(), "off path is not byte-identical")

    def test_on_matches_off_within_tolerance_quantized(self):
        off = _reload(MLX_VLM_GLM5_PREFILL_CPU_FRACTION=None,
                      MLX_VLM_GLM5_PREFILL_CPU_MIN_ROWS="16",
                      MLX_VLM_GLM5_PREFILL_CPU_DTYPE=None,
                      MLX_VLM_MOE_SEGMENT_ALIGN=None)
        glu = _build_switch_glu(off, quantized=True, E=self.E, K=self.K, N=self.N)
        x, inds = _inputs(self.E, self.K, dtype=mx.bfloat16)
        ref = np.asarray(glu(x, inds).astype(mx.float32))

        for frac in ("0.1", "0.2", "0.3"):
            on = _reload(MLX_VLM_GLM5_PREFILL_CPU_FRACTION=frac,
                         MLX_VLM_GLM5_PREFILL_CPU_MIN_ROWS="16")
            glu_on = _build_switch_glu(on, quantized=True, E=self.E, K=self.K, N=self.N)
            got = np.asarray(glu_on(x, inds).astype(mx.float32))
            self.assertEqual(got.shape, ref.shape)
            # bf16 dequant-then-GEMM against in-kernel dequant: agreement to a few
            # parts per thousand of the signal scale, NOT bit-identity.
            scale = float(np.abs(ref).mean())
            self.assertLess(
                float(np.abs(got - ref).max()), 0.05 * max(scale, 1e-3) * 20,
                f"fraction {frac} diverged beyond the bf16 mechanism tolerance",
            )
            np.testing.assert_allclose(
                got.mean(), ref.mean(), atol=0.02 * max(scale, 1e-3)
            )

    def test_on_matches_off_dense_weights_tightly(self):
        """With DENSE weights the two arms differ only in accumulation order, so
        the agreement is tight -- isolating the quantisation term above."""
        off = _reload(**{k: None for k in ENV_KEYS})
        glu = _build_switch_glu(off, quantized=False, E=self.E, K=self.K, N=self.N)
        x, inds = _inputs(self.E, self.K, dtype=mx.float32)
        ref = np.asarray(glu(x, inds))

        on = _reload(MLX_VLM_GLM5_PREFILL_CPU_FRACTION="0.3",
                     MLX_VLM_GLM5_PREFILL_CPU_MIN_ROWS="16",
                     MLX_VLM_GLM5_PREFILL_CPU_DTYPE="f32")
        glu_on = _build_switch_glu(on, quantized=False, E=self.E, K=self.K, N=self.N)
        got = np.asarray(glu_on(x, inds))
        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)

    def test_row_floor_keeps_decode_on_the_single_arm_path(self):
        s = _reload(MLX_VLM_GLM5_PREFILL_CPU_FRACTION="0.3")  # default floor 4096
        glu = _build_switch_glu(s, quantized=True, E=self.E, K=self.K, N=self.N)
        x, inds = _inputs(self.E, self.K, tokens=8, top_k=4)  # 32 rows: decode-sized
        called = []
        real = s.SwitchGLU._cocompute
        s.SwitchGLU._cocompute = lambda *a, **k: called.append(1) or real(*a, **k)
        try:
            glu(x, inds)
        finally:
            s.SwitchGLU._cocompute = real
        self.assertEqual(called, [], "decode-sized call entered the co-compute arm")

    def test_in_process_flip_and_row_accounting(self):
        """The A/B has to be paired arms inside ONE model load, so the fraction must
        be settable without a module reload -- and the harness's busy estimate is
        only as good as these counters, so the counts are asserted exactly."""
        s = _reload(MLX_VLM_GLM5_PREFILL_CPU_FRACTION=None,
                    MLX_VLM_GLM5_PREFILL_CPU_MIN_ROWS="16")
        glu = _build_switch_glu(s, quantized=True, E=self.E, K=self.K, N=self.N)
        x, inds = _inputs(self.E, self.K)          # 256 tokens x top-4 = 1024 rows

        self.assertEqual(s.set_prefill_cpu_fraction(0.3), 0.3)
        self.assertEqual(s.set_prefill_cpu_fraction(0.9), 0.5)   # same clamp as the env
        self.assertEqual(s.set_prefill_cpu_fraction("junk"), 0.0)

        s.set_prefill_cpu_fraction(0.3)
        s.reset_cocompute_stats()
        mx.eval(glu(x, inds))
        on = s.cocompute_stats()
        self.assertEqual(on["calls"], 1)
        self.assertEqual(on["total_rows"], 1024)
        self.assertGreater(on["cpu_rows"], 0)
        self.assertLessEqual(on["cpu_rows"], int(0.3 * 1024))
        self.assertAlmostEqual(on["cpu_row_fraction"], on["cpu_rows"] / 1024)
        # weight params dequantised = experts x (up + gate + down)
        self.assertEqual(on["cpu_weight_params"],
                         on["cpu_experts"] * 3 * self.K * self.N)

        s.set_prefill_cpu_fraction(0.0)
        s.reset_cocompute_stats()
        mx.eval(glu(x, inds))
        off = s.cocompute_stats()
        self.assertEqual(off["calls"], 0)
        self.assertEqual(off["cpu_rows"], 0)
        self.assertEqual(off["cpu_weight_params"], 0)

        # None restores the environment value (here: unset -> off)
        self.assertEqual(s.set_prefill_cpu_fraction(None), 0.0)

    def test_concurrency_plumbing_is_exercised(self):
        """The CPU arm must be built inside the dedicated CPU stream and issued
        with async_eval before the accelerator tail is built -- that ordering IS
        the overlap.  Assert both, and the segment count."""
        s = _reload(MLX_VLM_GLM5_PREFILL_CPU_FRACTION="0.3",
                    MLX_VLM_GLM5_PREFILL_CPU_MIN_ROWS="16")
        glu = _build_switch_glu(s, quantized=True, E=self.E, K=self.K, N=self.N)
        x, inds = _inputs(self.E, self.K)

        seen_streams = []
        real_mm = s._cpu_segment_mm
        s._cpu_segment_mm = lambda *a, **k: (
            seen_streams.append(mx.default_stream(mx.default_device())) or real_mm(*a, **k)
        )
        async_calls = []
        real_async = s.mx.async_eval
        s.mx.async_eval = lambda *a: async_calls.append(len(a)) or real_async(*a)
        try:
            out = glu(x, inds)
            mx.eval(out)
        finally:
            s._cpu_segment_mm = real_mm
            s.mx.async_eval = real_async

        # three projections -> three segment-mm calls, all on the CPU stream
        self.assertEqual(len(seen_streams), 3)
        for st in seen_streams:
            self.assertEqual(st.device, mx.cpu)
        # weights+activation kicked first (4 arrays), then the CPU result (1)
        self.assertEqual(async_calls, [4, 1])
        self.assertEqual(out.shape, (1, 256, 4, self.K))


if __name__ == "__main__":
    unittest.main()
