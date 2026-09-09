# TP batch row protocol design gate — 2026-09-09

## Verdict

Implementation is deliberately deferred. A safe EXTEND is not a local cache
conversion: current rank 1 deletes the incumbent cache when a donor prefill
opens its epoch. Adding an EXTEND verb now would merge the donor into missing or
fresh incumbent state and can silently change logits.

The previous lifecycle audit conclusion is confirmed: the frozen ladder remains
blocked for partial completion, greedy live merge, and DFlash late admission.

## Required minimal parts

1. **Multi-epoch ownership.** Rank 0 maps each live cache object identity to a
   stable epoch. Rank 1 retains every announced live epoch instead of replacing
   `self.caches`. A return to an incumbent selects its existing epoch; it must
   never emit MAKE_CACHE for populated state. Explicit release bounds memory.
2. **FILTER verb.** Carry destination epoch plus the exact ordered keep indices.
   DFlash's round loop and `GenerationBatch.filter()` announce before touching
   any rank-0 leaf. Rank 1 recursively calls `filter()` on every corresponding
   leaf, then acknowledges before the next FORWARD. Duplicate/reordered indices
   are preserved exactly.
3. **EXTEND verb.** Carry destination and source epochs. Both caches must already
   exist from their own mirrored prefills. Rank 1 applies the same `_extend_cache`
   operation, deletes the consumed source epoch, and acknowledges before rank 0
   mutates. Empty/unannounced/populated-but-unknown sources refuse before change.
4. **Protocol/lifecycle.** Bump the protocol; mixed revisions fail the frozen
   handshake. EXIT and error cleanup release all epochs. Vault restore and
   speculative rollback remain scoped to their named epoch.

## Required integration evidence

CPU tests must drive actual callers through: heterogeneous partial finish,
FILTER, next decode; donor prefill under a second epoch, EXTEND, next decode;
greedy live merge; all rows finish then a fresh next group. Each compares cache
leaf types, ordered padding, offsets, masks, and tiny deterministic outputs on
both ranks. It must also prove release of consumed/completed epochs and preserve
multimodal, arbitrary embedding, nonempty unknown-cache, and mixed-revision
refusals.

`test_tp_multi_epoch_design_blocker.py` records the immediate blocker: the
second MAKE_CACHE currently discards the first epoch. This is expected evidence
of missing capability, not a readiness test.

No model, Metal, deployment, network, or hardware work was performed.
