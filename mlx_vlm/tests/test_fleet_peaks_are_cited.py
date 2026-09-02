"""Every gate peak must still equal the receipt it claims to come from.

SHARD_GB["single"] = 183.0 was one operating point (the batch curve at B=8.1,
512-token prompts, decode) quoted as a constant. Lane 5 then measured 190.52 GiB
at B=1 simply by prefilling 16k -- a gate sized below what the workload reaches,
which is the I891 failure and the shape of the 2026-09-01 freeze.

The defence is not a better constant, it is that the table cannot drift from its
sources: these tests re-open each receipt and compare.
"""

import json
import os
import unittest

from mlx_vlm.tp import fleet


def _dig(doc, locator):
    for k in locator:
        doc = doc[k]
    return doc


class TestPeaksAreDerivedFromReceipts(unittest.TestCase):
    def test_every_entry_matches_its_receipt(self):
        root = fleet._LOG_ROOT
        checked = 0
        for key, gib, rel, loc, src in fleet.SINGLE_BOX_PEAKS:
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                self.skipTest(f"receipt not on this box: {rel}")
            with open(path) as fh:
                doc = json.load(fh)
            got = float(_dig(doc, loc))
            self.assertAlmostEqual(
                got, gib, places=2,
                msg=f"{key} from {src} claims {gib} but {rel}::{loc} says {got}")
            checked += 1
        self.assertGreater(checked, 0, "the table must not be empty")

    def test_units_are_gib_not_gb(self):
        """The audit that had to happen before any refit.

        phys_footprint_gb -- which produced lane 5's peaks -- divides by
        _GB = 1024**3, so its output is GiB and compares directly against
        SHARD_GB. Reading those peaks as decimal GB would have made 190.52 into
        177.4 GiB and reversed the finding.
        """
        self.assertEqual(fleet._GB, 1024.0 ** 3)


class TestSizingRule(unittest.TestCase):
    def test_exact_configuration_is_sized_from_its_own_receipt(self):
        gib, why = fleet.single_box_required_gib(
            batch=1, prompt_tokens=16384, chunk=512)
        self.assertAlmostEqual(gib, 190.52, places=2)
        self.assertIn("SWEEP6_L2_e2e_E1.json", why)

    def test_a_dominating_point_may_size_a_smaller_request(self):
        gib, _ = fleet.single_box_required_gib(batch=1, prompt_tokens=8192, chunk=256)
        self.assertAlmostEqual(gib, 190.52, places=2)

    def test_harness_disagreement_resolves_to_the_HIGHER_measurement(self):
        """The bug real data exposed in the first draft of this rule.

        kda_bench and X3_T1 both measured B=1 @ 512 and disagree by 11.85 GiB
        (173.90 vs 162.05). Returning the smallest dominating measurement reads
        as "tight without being under" and in fact adopts whichever instrument
        read low. Stage 1 takes the max per configuration; only then does
        stage 2 take the min across configurations.
        """
        gib, why = fleet.single_box_required_gib(batch=1, prompt_tokens=512)
        self.assertGreater(gib, 162.05,
                           msg="must not adopt the optimistic harness (X3 read 162.05)")

    def test_a_direct_measurement_is_never_undercut_by_another_harness(self):
        """The second bug real data exposed.

        kda_bench measured B=8 @ 512 at 183.20. X3_T1 measured B=16 @ 512 at
        177.39 -- a LARGER configuration reading LOWER, because the two
        harnesses sit ~11 GiB apart. Dominating across mixed sources returned
        177.39 for a B=8 request: 5.81 GiB below a direct measurement of that
        exact cell. Per-source domination, then max, fixes it.
        """
        gib, why = fleet.single_box_required_gib(batch=8, prompt_tokens=512)
        self.assertGreaterEqual(gib, 183.2)
        self.assertIn("kda_bench", why)

    def test_the_rule_can_be_loose_and_says_so(self):
        """Honest limitation, pinned so it is not mistaken for tightness.

        Taking the max across per-source bounds means a source that only
        measured much larger configurations still contributes its (valid but
        loose) bound. B=1 @ 512 is sized by lane 5's B=1 @ 16k peak because that
        genuinely dominates it. Safe, not tight; it tightens as the table fills.
        """
        gib, why = fleet.single_box_required_gib(batch=1, prompt_tokens=512)
        self.assertAlmostEqual(gib, 190.52, places=2)
        self.assertIn("SWEEP6_L2_e2e_E1.json", why)

    def test_b16_at_8k_is_now_sized_from_lane3(self):
        gib, why = fleet.single_box_required_gib(batch=16, prompt_tokens=8192)
        self.assertAlmostEqual(gib, 258.22, places=2)
        self.assertAlmostEqual(fleet.gate_requirement(gib)["required_gib"], 318.22, 2)
        self.assertIn("X3_T1.json", why)
        self.assertIn("cells.9.peak_gib", why)

    def test_131k_still_raises_because_nothing_measured_it(self):
        with self.assertRaises(fleet.UnmeasuredConfiguration):
            fleet.single_box_required_gib(batch=16, prompt_tokens=131072)

    def test_every_entry_names_its_source_commit_or_run(self):
        for key, gib, rel, loc, src in fleet.SINGLE_BOX_PEAKS:
            self.assertTrue(src and "@" in src,
                            f"{key} must name the run/commit it came from")

    def test_segment_align_is_not_satisfied_by_an_OFF_measurement(self):
        gib, _ = fleet.single_box_required_gib(
            batch=1, prompt_tokens=16384, chunk=512, segment_align=True)
        self.assertAlmostEqual(gib, 196.10, places=2,
                               msg="an ON request must be sized by an ON receipt")

    def test_uncovered_configuration_refuses_rather_than_extrapolating(self):
        with self.assertRaises(fleet.UnmeasuredConfiguration):
            fleet.single_box_required_gib(batch=32, prompt_tokens=8192)
        with self.assertRaises(fleet.UnmeasuredConfiguration):
            fleet.single_box_required_gib(batch=1, prompt_tokens=131072, chunk=512)
        with self.assertRaises(fleet.UnmeasuredConfiguration):
            fleet.single_box_required_gib(batch=1, prompt_tokens=512, speculative=True)

    def test_the_old_constant_would_have_under_requested(self):
        """The finding, asserted so it cannot quietly regress."""
        measured, _ = fleet.single_box_required_gib(
            batch=1, prompt_tokens=16384, chunk=512)
        self.assertGreater(measured, fleet.SHARD_GB["single"],
                           "lane 5's B=1 16k prefill exceeds the old constant")

    def test_the_batch_fit_is_not_used_for_sizing(self):
        """173.0 + 1.23*B predicts 174.2 at B=1; the measured 16k peak is 190.52.

        A line fitted over batch cannot be trusted over prompt length, and this
        16.3 GiB gap is why the fit is documentation and not the gate.
        """
        fit_at_b1 = fleet.single_box_gb(1)
        measured, _ = fleet.single_box_required_gib(
            batch=1, prompt_tokens=16384, chunk=512)
        self.assertGreater(measured - fit_at_b1, 10.0)

    def test_the_fit_would_have_badly_undersized_B16_at_8k(self):
        """173.0 + 1.23*16 = 192.7 GiB; lane 3 measured 258.22."""
        measured, _ = fleet.single_box_required_gib(batch=16, prompt_tokens=8192)
        self.assertGreater(measured - fleet.single_box_gb(16), 60.0)


if __name__ == "__main__":
    unittest.main()


class TestGateRequirementUnits(unittest.TestCase):
    """The GiB/GB conflation, pinned so it cannot recur silently.

    Seen twice on 2026-09-02: lane 5's prereg wrote the fit in GiB and its
    prediction in GB in one sentence; lane 3's gate dry-run converted a GiB peak
    to GB and then added the 60 GiB margin as 60 GB, shaving 4.4 GB off a margin
    that still read "60" on the page.
    """

    def test_both_units_are_reported(self):
        r = fleet.gate_requirement(328.0)
        self.assertEqual(r["required_gib"], 388.0)
        self.assertAlmostEqual(r["required_gb"], 416.61, places=1)

    def test_the_margin_is_gib_and_worth_more_in_gb(self):
        """60 GiB is 64.4 GB; adding 60 GB instead loses 4.4 GB of margin."""
        r = fleet.gate_requirement(328.0)
        naive_gb = 328.0 * (1024 ** 3) / 1e9 + 60.0
        self.assertAlmostEqual(r["required_gb"] - naive_gb, 4.4, places=1)

    def test_lane3_dry_run_numbers_are_reproduced(self):
        """219.1 GiB -> 235.3 GB is the conversion; 295.3 is that plus 60 GB."""
        r = fleet.gate_requirement(219.1)
        self.assertAlmostEqual(219.1 * (1024 ** 3) / 1e9, 235.3, places=1)
        self.assertEqual(r["required_gib"], 279.1)
        self.assertAlmostEqual(r["required_gb"], 299.7, places=1)

    def test_it_says_which_field_the_gate_wants(self):
        self.assertEqual(fleet.gate_requirement(100.0)["pass_to_require_headroom"],
                         "required_gib")
