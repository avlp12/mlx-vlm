"""Resident session store: the test contract, written before the code.

Coordinator ruling I969(2).  Written before the code; discharged by
``test_session_store.py`` on 2026-09-02.  Writing them first is
deliberate -- three of these (T4, T6, T9) describe ways the feature can be
"working" and silently wrong, and they are much harder to think of once there is
an implementation to look at.

WHAT IS BEING BUILT
-------------------
The shipped vault (context_vault.py) checkpoints at PREFILL boundaries: rungs
land on multiples of prefill_step_size so restore-plus-tail is bit-identical to a
straight-through cold prefill (align_boundaries, :181-199).  That serves "same
document, new suffix" and gives the measured 8-12x TTFT.

The session store adds a second, SEPARATELY LABELLED tier: a checkpoint taken
when a response finishes, keyed by prompt+response, so the next turn re-prefills
only the new user message.  Its guarantee is deliberately different and weaker in
one direction and stronger in another:

    "bit-identical to CONTINUING THE SAME SESSION"

It is NOT bit-identical to a cold prefill of the concatenated transcript, and it
cannot be: a decoded suffix is produced one token at a time, which is a different
chunk decomposition than a 2048-token prefill chunk, before the store is even
involved.  Claiming otherwise would widen the existing cold-prefill guarantee to
cover something it has never covered.  T5/T6 pin both halves of that sentence.

SIZING THIS EXISTS TO CASH IN (post-dedup, see test_vault_latent_dedup.py)
    8k 0.267 GiB | 32k 0.654 GiB | 131k 2.203 GiB
    -> 873 / 356 / 105 resident sessions in a 250 GB budget
    131k revisit TTFT: 520 s cold (131072 tok / 252 tok/s, PA733) -> ~0.04 s
"""

import unittest

# Discharged 2026-09-02 by test_session_store.py, which implements T1-T13 against
# the shipped machinery.  This file is kept as the rationale of record -- the
# WHY that the implementation file points back at -- and each spec now names the
# test that discharges it, so ``pytest -rs`` reads as a contract-to-code map.
DESIGN = "contract discharged -> see mlx_vlm/tests/test_session_store.py"


@unittest.skip(DESIGN)
class TestSessionCapture(unittest.TestCase):
    def test_T1_capture_fires_when_a_response_completes(self):
        """A finished generation inserts a rung keyed by prompt+response tokens.

        Guards the actual bug this feature fixes: today the deepest rung is at
        the last prefill boundary BELOW the prompt, so a returning session
        re-prefills its own last turn plus everything generated since.
        """

    def test_T2_capture_is_skipped_when_the_stream_was_aborted(self):
        """A cancelled or errored generation must not store a rung.

        A truncated response is a prefix no future turn will ever send, so the
        rung is pure eviction pressure -- and if the abort happened mid-token the
        cache may not correspond to any complete token sequence at all.
        """

    def test_T3_ownership_is_transferred_not_copied(self):
        """End-of-turn capture must not deep-copy 2-4 GiB it is about to free.

        capture_fragments (:123-138) snapshots because the live cache keeps
        advancing.  At end of turn it does not: the request is done and the cache
        is about to be dropped, so the store can take the buffers.  Assert the
        stored arrays are the same objects the request held, and that the request
        path no longer references them.
        """


@unittest.skip(DESIGN)
class TestIdentityTiers(unittest.TestCase):
    def test_T4_session_rungs_are_labelled_and_never_serve_a_prefill_query(self):
        """The two tiers must not be interchangeable in the trie.

        THE FAILURE THIS PREVENTS: a session rung answering a cold-prefill lookup
        would hand back a state that is correct for the session but not
        bit-identical to a cold prefill, silently breaking the guarantee the
        shipped vault advertises.  Tier must be part of the lookup key, not a
        field a caller is trusted to check.
        """

    def test_T5_restore_then_continue_is_bit_identical_to_never_stopping(self):
        """The session tier's actual guarantee, stated positively.

        Drive N tokens, capture, restore into a fresh cache, decode M more; then
        drive N+M straight through. The two must be bit-identical (compared on
        bit patterns, per test_vault_latent_dedup.assert_same_bits).
        """

    def test_T6_session_restore_is_NOT_claimed_equal_to_a_cold_prefill(self):
        """The guarantee's boundary, asserted rather than left to the docstring.

        Assert the labelling and the refusal, NOT that the tensors differ -- on a
        small synthetic cache they may coincide, and a test that passes only
        because of a coincidence is the empty-test failure class (I943 sect.7,
        I950 sect.1).
        """

    def test_T7_identity_change_invalidates_session_rungs_too(self):
        """vault_identity (:689-720) already covers weights/code/topology/toggles.

        Pin that session rungs go through the same origin check in restore_into
        (:523-541) and are refused, not restored, when the origin differs.
        """


@unittest.skip(DESIGN)
class TestEviction(unittest.TestCase):
    def test_T8_deep_rungs_of_a_session_are_evicted_before_its_shallow_ones(self):
        """Degrade gracefully from instant to 8-12x, never straight to cold.

        Dropping a session's deepest rung costs the tail; dropping its shallow
        rungs costs everything.  Today _evict_until (:450-465) picks the globally
        oldest rung and knows nothing about sessions.
        """

    def test_T9_evicting_a_session_cannot_orphan_a_live_request(self):
        """A rung restored into an in-flight request must survive eviction.

        Eviction drops the store's reference; if a live cache aliases those
        buffers (T3's ownership transfer makes that possible in the other
        direction too) the request must keep working. This is the interaction
        between the two features and is the one I would expect to be got wrong.
        """

    def test_T10_ttl_expiry_frees_bytes_and_is_observable(self):
        """Per-session TTL, with resident_bytes and stats reflecting the drop."""

    def test_T11_budget_is_never_exceeded_across_mixed_tiers(self):
        """The shipped invariant (test_context_vault.test_lru_eviction_respects_
        budget) must hold with prefill rungs and session rungs in one store."""


@unittest.skip(DESIGN)
class TestServerWiring(unittest.TestCase):
    def test_T12_session_store_stays_off_under_TP(self):
        """generation.py:1286-1290 disables the vault under TP because the server
        request path carries no rungs to rank 1 -- NOT because of a jaccl fault.

        Single-box only until the control side-channel lands.  Pin that the
        session tier inherits the same refusal rather than adding its own.
        """

    def test_T13_a_store_fault_never_fails_a_request(self):
        """_build_vault (:1268-1301) swallows vault faults on load.  The
        end-of-turn capture is on the RESPONSE path, where an exception would
        surface to the client; assert a raising capture is logged and dropped."""


if __name__ == "__main__":
    unittest.main()
