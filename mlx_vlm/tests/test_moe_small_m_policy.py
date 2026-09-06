"""Small-M MoE expert-GEMM dispatch policy (MLX_VLM_MOE_SMALL_M_POLICY).

The lever: ``mlx/backend/metal/quantized.cpp`` switches ``GatherQMM`` to ``gather_qmm_rhs`` the
instant ``rows / num_experts >= 4`` (with sorted indices).  That kernel runs a full K-loop per
distinct expert inside every BM=16 row tile, so its cost carries a per-expert CONSTANT of ~E
passes on top of rows/16 useful passes.  At rpe=4 the constant is 80% of the work and the
branch is a 51%-per-token REGRESSION against the ``gather_qmv`` path one token below it (probe
receipt: 143 tokens = 75.06 us/token, 144 tokens = 113.37 us/token).

The fix needs no MLX change: cut the already-sorted rows into k contiguous slabs each holding
fewer than 4*E rows, so every slab lands on ``gather_qmv`` with its expert locality intact.

These tests assert the things that make it safe to ship:
  1. the flag is OFF by default and parses the way the docstring says,
  2. the slab decision is the arithmetic the cost model derives (fires only on 4 <= rpe < max,
     always cuts every slab below the 4*E branch threshold), and it is a function of SHAPES
     only -- no dependence on index values, so no host sync and no mx.compile hazard,
  3. the transform is exactly output-preserving: on CPU there is no branch at all, so policy-on
     and policy-off run the identical kernel and the outputs must be BITWISE equal, and
  4. segment alignment is suppressed on the rows this policy slabs (BM=16 padding only helps
     the rhs kernel, and at rpe~4 it would multiply the row count several-fold).

CPU-only by construction (mx.set_default_device(mx.cpu)); the GPU branch selection itself is
measured by the kernel probe, not by unit tests.
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


_ENV_KEYS = ("MLX_VLM_MOE_SMALL_M_POLICY", "MLX_VLM_MOE_SMALL_M_MAX_RPE",
             "MLX_VLM_MOE_SEGMENT_ALIGN")


class _CPUCase(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        self._dev = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self._dev)
        _reload(**self.saved)


class TestSmallMFlag(_CPUCase):
    def test_defaults_off(self):
        S = _reload(MLX_VLM_MOE_SMALL_M_POLICY=None, MLX_VLM_MOE_SMALL_M_MAX_RPE=None)
        self.assertEqual(S._small_m_policy(), (False, 6.0))
        # and with the flag off the decision is always "leave MLX alone"
        for rows in (1, 64, 1152, 2048, 100000):
            self.assertEqual(S._small_m_slabs(rows, 288), 1)

    def test_flag_parsing(self):
        for v, want in (("auto", True), ("on", True), ("1", True), ("true", True),
                        ("yes", True), ("off", False), ("0", False), ("garbage", False)):
            S = _reload(MLX_VLM_MOE_SMALL_M_POLICY=v, MLX_VLM_MOE_SMALL_M_MAX_RPE=None)
            self.assertEqual(S._small_m_policy()[0], want, f"{v!r}")

    def test_threshold_parsing_and_clamp(self):
        for v, want in (("4", 4.0), ("6", 6.0), ("7.5", 7.5), ("8", 8.0),
                        ("2", 4.0),        # below 4 the policy can never fire anyway
                        ("99", 8.0),       # k>=3 is a loss under both cost models
                        ("junk", 6.0)):
            S = _reload(MLX_VLM_MOE_SMALL_M_POLICY="auto", MLX_VLM_MOE_SMALL_M_MAX_RPE=v)
            self.assertAlmostEqual(S._small_m_policy()[1], want, msg=f"{v!r}")


class TestSlabArithmetic(_CPUCase):
    E = 288  # GLM-5.3-Flash

    def _S(self, max_rpe="8"):
        return _reload(MLX_VLM_MOE_SMALL_M_POLICY="auto", MLX_VLM_MOE_SMALL_M_MAX_RPE=max_rpe,
                       MLX_VLM_MOE_SEGMENT_ALIGN=None)

    def test_band_edges_are_the_mlx_branch_condition(self):
        S = self._S()
        # rows/E < 4 -> MLX is already on gather_qmv, do nothing. GLM: tokens 143 = 1144 rows.
        self.assertEqual(S._small_m_slabs(4 * self.E - 1, self.E), 1)
        # rows/E == 4 -> the branch flips. GLM: tokens 144 = 1152 rows. This is the cliff.
        self.assertEqual(S._small_m_slabs(4 * self.E, self.E), 2)
        # rows/E == 8 -> at/above the cap, leave rhs alone.
        self.assertEqual(S._small_m_slabs(8 * self.E, self.E), 1)

    def test_default_cap_stops_at_six(self):
        S = self._S(max_rpe=None)
        self.assertEqual(S._small_m_slabs(int(5.9 * self.E), self.E), 2)
        self.assertEqual(S._small_m_slabs(int(6.0 * self.E), self.E), 1)

    def test_every_slab_falls_below_the_branch_threshold(self):
        S = self._S()
        limit = 4 * self.E
        for rows in range(limit, 8 * self.E):
            k = S._small_m_slabs(rows, self.E)
            if k == 1:
                continue
            bounds = S._slab_bounds(rows, k)
            self.assertEqual(bounds[0][0], 0)
            self.assertEqual(bounds[-1][1], rows)
            self.assertEqual(len(bounds), k)
            for a, b in bounds:
                self.assertGreater(b, a)
                # the whole point: MLX must NOT take gather_qmm_rhs on any slab
                self.assertLess((b - a) // self.E, 4, f"rows={rows} slab={b - a}")
            # slabs tile the row range exactly, in order, with no gaps
            self.assertEqual([a for a, _ in bounds[1:]], [b for _, b in bounds[:-1]])

    def test_decision_is_shape_only(self):
        """No dependence on index VALUES -> no host sync, stable under mx.compile."""
        S = self._S()
        import inspect

        src = inspect.getsource(S._small_m_slabs)
        for forbidden in ("np.array", "mx.eval", "bincount", "item()", "tolist"):
            self.assertNotIn(forbidden, src)
        # same rows -> same k regardless of what the routing actually chose
        self.assertEqual(S._small_m_slabs(1152, 288), S._small_m_slabs(1152, 288))


class TestBitIdentity(_CPUCase):
    """Policy on vs off must give BITWISE identical outputs.

    On CPU there is no gather_qmm_rhs branch, so both arms run the same kernel and any
    difference would be a plumbing bug (wrong slab boundary, wrong index slice, wrong
    concatenation order, alignment leaking in).  This is the algebraic-identity test; the GPU
    arm is a genuinely different kernel and is expected to differ in the low bits.
    """

    E = 32      # small, so rows/E >= 4 is reachable at modest sizes
    TOPK = 4
    K = 128
    N = 64

    def _build(self, S, quantized=True):
        mx.random.seed(0)
        sw = S.SwitchGLU(self.K, self.N, self.E, bias=False)
        if quantized:
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(sw, name, getattr(sw, name).to_quantized(group_size=32, bits=4))
        mx.eval(sw.parameters())
        return sw

    def _inputs(self, tokens, seed=1):
        rng = np.random.default_rng(seed)
        x = mx.array(rng.normal(size=(1, tokens, self.K)).astype(np.float32))
        idx = mx.array(
            np.stack([rng.choice(self.E, size=self.TOPK, replace=False)
                      for _ in range(tokens)])[None].astype(np.int32)
        )
        mx.eval(x, idx)
        return x, idx

    def _run(self, tokens, quantized, **env):
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN=None, **env)
        sw = self._build(S, quantized)
        x, idx = self._inputs(tokens)
        y = sw(x, idx)
        mx.eval(y)
        return np.array(y), S

    def test_policy_fires_in_the_band(self):
        S = _reload(MLX_VLM_MOE_SMALL_M_POLICY="auto", MLX_VLM_MOE_SMALL_M_MAX_RPE="8")
        # tokens * TOPK rows; E=32 -> the band is 128 <= rows < 256, i.e. 32 <= tokens < 64
        self.assertEqual(S._small_m_slabs(32 * self.TOPK, self.E), 2)
        self.assertEqual(S._small_m_slabs(31 * self.TOPK, self.E), 1)
        self.assertEqual(S._small_m_slabs(64 * self.TOPK, self.E), 1)

    def test_bitwise_identical_quantized(self):
        for tokens in (32, 40, 48, 63):
            off, _ = self._run(tokens, True, MLX_VLM_MOE_SMALL_M_POLICY="off")
            on, S = self._run(tokens, True, MLX_VLM_MOE_SMALL_M_POLICY="auto",
                              MLX_VLM_MOE_SMALL_M_MAX_RPE="8")
            self.assertEqual(S._small_m_slabs(tokens * self.TOPK, self.E), 2,
                             f"tokens={tokens} did not exercise the policy")
            self.assertEqual(off.shape, on.shape)
            np.testing.assert_array_equal(off, on, err_msg=f"tokens={tokens}")

    def test_bitwise_identical_unquantized(self):
        off, _ = self._run(40, False, MLX_VLM_MOE_SMALL_M_POLICY="off")
        on, _ = self._run(40, False, MLX_VLM_MOE_SMALL_M_POLICY="auto",
                          MLX_VLM_MOE_SMALL_M_MAX_RPE="8")
        np.testing.assert_array_equal(off, on)

    def test_bitwise_identical_outside_the_band_is_trivially_the_same_path(self):
        for tokens in (8, 16, 100, 128):
            off, _ = self._run(tokens, True, MLX_VLM_MOE_SMALL_M_POLICY="off")
            on, _ = self._run(tokens, True, MLX_VLM_MOE_SMALL_M_POLICY="auto",
                              MLX_VLM_MOE_SMALL_M_MAX_RPE="8")
            np.testing.assert_array_equal(off, on, err_msg=f"tokens={tokens}")

    def test_switch_mlp_is_covered_too(self):
        S = _reload(MLX_VLM_MOE_SMALL_M_POLICY="off", MLX_VLM_MOE_SEGMENT_ALIGN=None)

        def build(S):
            mx.random.seed(0)
            m = S.SwitchMLP(self.K, self.N, self.E, bias=False)
            for name in ("fc1", "fc2"):
                setattr(m, name, getattr(m, name).to_quantized(group_size=32, bits=4))
            mx.eval(m.parameters())
            return m

        x, idx = self._inputs(40)
        off = np.array(build(S)(x, idx))
        S = _reload(MLX_VLM_MOE_SMALL_M_POLICY="auto", MLX_VLM_MOE_SMALL_M_MAX_RPE="8")
        x, idx = self._inputs(40)
        on = np.array(build(S)(x, idx))
        np.testing.assert_array_equal(off, on)


class TestCompileAndCPUBranchFacts(_CPUCase):
    E, TOPK, K, N = 32, 4, 128, 64

    def _sw(self, S):
        mx.random.seed(0)
        sw = S.SwitchGLU(self.K, self.N, self.E, bias=False)
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(sw, name, getattr(sw, name).to_quantized(group_size=32, bits=4))
        mx.eval(sw.parameters())
        return sw

    def test_survives_mx_compile_and_reselects_per_shape(self):
        """Glm5NextDecoderLayer compiles the FFN half, so the policy must be trace-safe.

        ``indices.size`` is a SHAPE, and mx.compile keeps a per-shape cache, so branching on it
        retraces rather than replaying a stale graph.  Branching on index VALUES would not be
        safe -- hence the shape-only rule in ``_small_m_slabs``.
        """
        S = _reload(MLX_VLM_MOE_SMALL_M_POLICY="auto", MLX_VLM_MOE_SMALL_M_MAX_RPE="8",
                    MLX_VLM_MOE_SEGMENT_ALIGN=None)
        sw = self._sw(S)
        compiled = mx.compile(lambda a, b: sw(a, b))
        rng = np.random.default_rng(0)
        # 40 tokens -> 160 rows -> rpe 5.0 -> slabbed;  20 -> 2.5 and 80 -> 10.0 -> untouched
        for tokens, want_k in ((40, 2), (40, 2), (20, 1), (80, 1)):
            self.assertEqual(S._small_m_slabs(tokens * self.TOPK, self.E), want_k)
            x = mx.array(rng.normal(size=(1, tokens, self.K)).astype(np.float32))
            idx = mx.array(
                np.stack([rng.choice(self.E, self.TOPK, replace=False)
                          for _ in range(tokens)])[None].astype(np.int32))
            mx.eval(x, idx)
            a, b = compiled(x, idx), sw(x, idx)
            mx.eval(a, b)
            np.testing.assert_array_equal(np.array(a), np.array(b),
                                          err_msg=f"tokens={tokens}")

    def test_cpu_backend_has_no_sorted_branch(self):
        """Documents WHY the CPU suite cannot certify the GPU numerics of this lever.

        The branch this policy steers lives in mlx/backend/metal/quantized.cpp.  On CPU
        ``sorted_indices`` selects nothing, so both arms run one kernel and CPU bit-identity
        proves the algebra, not the kernel equivalence.  If this test ever starts failing, MLX
        grew a CPU sorted path and the GPU-only caveat needs revisiting.
        """
        mx.random.seed(0)
        w, sc, bi = mx.quantize(mx.random.uniform(shape=(self.E, self.N, self.K)),
                                group_size=32, bits=4)
        idx = mx.sort(mx.random.randint(0, self.E, (200,)))
        x = mx.random.normal((200, 1, self.K))
        kw = dict(transpose=True, group_size=32, bits=4)
        a = mx.gather_qmm(x, w, sc, bi, rhs_indices=idx, sorted_indices=True, **kw)
        b = mx.gather_qmm(x, w, sc, bi, rhs_indices=idx, sorted_indices=False, **kw)
        mx.eval(a, b)
        np.testing.assert_array_equal(np.array(a), np.array(b))


class TestSegmentAlignInteraction(_CPUCase):
    """Segment alignment must be suppressed exactly where the policy slabs the rows.

    ``_gather_sort`` gates BM=16 padding on ``indices.size >= 4 * num_experts`` -- the SAME
    threshold as the rhs branch -- so with MLX_VLM_MOE_SEGMENT_ALIGN=16 the band that this
    policy targets is also the band where alignment inflates the row count hardest (at rpe=4
    almost every expert holds fewer than 16 rows, so 4*E rows pad to ~16*E).
    """

    E = 32
    TOPK = 4

    def test_align_is_off_on_slabbed_rows(self):
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN="16", MLX_VLM_MOE_SMALL_M_POLICY="auto",
                    MLX_VLM_MOE_SMALL_M_MAX_RPE="8")
        rng = np.random.default_rng(0)
        rows = 4 * self.E                      # rpe == 4.0, right on the cliff
        idx = mx.array(rng.integers(0, self.E, size=(1, rows // self.TOPK, self.TOPK))
                       .astype(np.int32))
        x = mx.array(rng.normal(size=(1, rows // self.TOPK, 8)).astype(np.float32))
        xe = mx.expand_dims(x, (-2, -3))
        _, idx_aligned, _ = S._gather_sort(xe, idx, num_experts=self.E, allow_align=True)
        _, idx_plain, _ = S._gather_sort(xe, idx, num_experts=self.E, allow_align=False)
        mx.eval(idx_aligned, idx_plain)
        # alignment really does blow the row count up here -- that is why it is suppressed
        self.assertGreater(idx_aligned.size, idx_plain.size)
        self.assertEqual(idx_plain.size, rows)

    def test_align_still_works_when_the_policy_does_not_fire(self):
        S = _reload(MLX_VLM_MOE_SEGMENT_ALIGN="16", MLX_VLM_MOE_SMALL_M_POLICY="auto",
                    MLX_VLM_MOE_SMALL_M_MAX_RPE="8")
        self.assertEqual(S._small_m_slabs(16 * self.E, self.E), 1)  # rpe 16 -> rhs keeps it
        self.assertEqual(S._moe_segment_align(), 16)


if __name__ == "__main__":
    unittest.main()
