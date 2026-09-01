"""The gathered-attention query chunk is derived from the cache depth.

MLX indexes a gather with 32-bit arithmetic while the operand has fewer than
2**31 elements and takes a slower path at or above it.  In _gathered_attention
the operand is the broadcast ``(B, chunk, Kv, dim)``, so the boundary sits on
``chunk * Kv * dim`` and moves with the cache.  A constant chunk is therefore on
the wrong side of it at some depth no matter which constant is chosen.

Measured on the 320B tree (one load, gate 24576, real source+prose, two
interleaved cycles reproducing to 0.3%; receipt logs/sweep3/R2b_arms_r5.json):
the gather region cost 84.1 s at chunk 512, 82.0 s at 256, 70.8 s at 128 and
59.3 s at 64 -- and the 128 arm's per-chunk cost stepped 6564 -> 8894 ms at
exactly the chunk where 128 * 32768 * 512 == 2**31, and nowhere else.

These tests pin the arithmetic rather than the timing.  They exist because
mlx#4437 concerns this same boundary as a CORRECTNESS issue: bit-exactness
across it was measured (logs/sweep3/R5_identity_cliff_c1.json, max |logit diff|
0.0) and these cases should fail loudly if the reasoning behind that ever moves.
"""
import unittest

from mlx_vlm.models.glm5_next import language as L


class TestGatherQChunk(unittest.TestCase):
    LIMIT = 2**31
    DIM = 512  # kv_lora_rank on GLM-5.3-Flash

    def test_stays_under_the_32_bit_boundary(self):
        # The whole point: chunk * Kv * dim must never reach 2**31 while the
        # floor is not binding.
        for kv in (2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072):
            chunk = L._gather_q_chunk_for(kv, self.DIM)
            if chunk > L._GATHER_Q_CHUNK_MIN:
                self.assertLess(
                    chunk * kv * self.DIM, self.LIMIT,
                    f"chunk {chunk} at Kv {kv} reaches the 32-bit boundary",
                )

    def test_never_exceeds_the_env_knob_and_never_undercuts_the_floor(self):
        for kv in (1, 1024, 32768, 1 << 20):
            chunk = L._gather_q_chunk_for(kv, self.DIM)
            self.assertLessEqual(chunk, L._GATHER_Q_CHUNK)
            self.assertGreaterEqual(chunk, L._GATHER_Q_CHUNK_MIN)

    def test_is_a_power_of_two(self):
        # so the chunk keeps dividing the prefill block evenly and the gathered
        # shapes stay regular
        for kv in (2048, 12345, 32768, 40960, 131072):
            chunk = L._gather_q_chunk_for(kv, self.DIM)
            self.assertEqual(chunk & (chunk - 1), 0, f"{chunk} is not a power of two")

    def test_reproduces_the_measured_fast_values(self):
        # Regression pin against the arms that were actually run.  128 measured
        # fast at Kv 24576-30720 and slow at 32768; 64 measured fast at 32768
        # and above, out to the 40960 the run reached.
        for kv in (24576, 26624, 28672, 30720):
            self.assertEqual(L._gather_q_chunk_for(kv, self.DIM), 128)
        for kv in (32768, 40960):
            self.assertEqual(L._gather_q_chunk_for(kv, self.DIM), 64)
        # And it keeps stepping down: at Kv 65536 a chunk of 64 would land
        # exactly ON the boundary (64 * 65536 * 512 == 2**31), which the 128 arm
        # showed is already the slow side, so the derived value is 32.  This
        # expectation was wrong when first written and the test caught it.
        self.assertEqual(L._gather_q_chunk_for(65536, self.DIM), 32)
        self.assertEqual(L._gather_q_chunk_for(131072, self.DIM), 16)

    def test_short_context_is_unchanged(self):
        # Below Kv 8192 the derived bound is wider than the shipped 512, so the
        # serving configuration keeps exactly the chunk it had.
        old = L._GATHER_Q_CHUNK
        try:
            L._GATHER_Q_CHUNK = 512
            for kv in (2048, 4096):
                self.assertEqual(L._gather_q_chunk_for(kv, self.DIM), 512)
        finally:
            L._GATHER_Q_CHUNK = old

    def test_a_verify_block_is_still_a_single_chunk(self):
        # A speculative verify block is L <= 8.  The floor guarantees the chunk
        # is at least 16 at any depth, so range(0, L, chunk) stays one iteration
        # and the speculative path is untouched by this change.
        for kv in (2048, 32768, 131072, 1 << 21):
            self.assertGreaterEqual(L._gather_q_chunk_for(kv, self.DIM), 8)

    def test_degenerate_inputs_fall_back_to_the_knob(self):
        self.assertEqual(L._gather_q_chunk_for(0, self.DIM), L._GATHER_Q_CHUNK)
        self.assertEqual(L._gather_q_chunk_for(self.DIM, 0), L._GATHER_Q_CHUNK)


if __name__ == "__main__":
    unittest.main()
