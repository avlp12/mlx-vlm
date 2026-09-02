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
        for key, (gib, rel, loc) in fleet.SINGLE_BOX_PEAKS.items():
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                self.skipTest(f"receipt not on this box: {rel}")
            with open(path) as fh:
                doc = json.load(fh)
            got = float(_dig(doc, loc))
            self.assertAlmostEqual(
                got, gib, places=2,
                msg=f"{key} claims {gib} but {rel}::{loc} says {got}")
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

    def test_segment_align_is_not_satisfied_by_an_OFF_measurement(self):
        gib, _ = fleet.single_box_required_gib(
            batch=1, prompt_tokens=16384, chunk=512, segment_align=True)
        self.assertAlmostEqual(gib, 196.10, places=2,
                               msg="an ON request must be sized by an ON receipt")

    def test_uncovered_configuration_refuses_rather_than_extrapolating(self):
        with self.assertRaises(fleet.UnmeasuredConfiguration):
            fleet.single_box_required_gib(batch=16, prompt_tokens=16384, chunk=2048)
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


if __name__ == "__main__":
    unittest.main()
