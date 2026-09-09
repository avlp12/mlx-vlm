# TP batch lifecycle readiness audit — 2026-09-09

## Verdict

**BLOCKED for the frozen full N1/4/8/16 ladder.** Commit `469b1c97` is safe for
a fixed batch whose rows remain together until completion, and a later group can
open a fresh epoch. It does not support the ladder's live row lifecycle.

## Source contract

- The DFlash arm explicitly enables `MLX_VLM_SPEC_EXTEND_ACTIVE=1` with a 40 ms
  quiet window and 400 ms admission window. `SpeculativeGenerationBatch` queues
  late rows, then `_admission_poll()` calls `_extend_cache()` at a round boundary.
  The following mirrored forward sees a changed padding vector and refuses.
- DFlash's batched round loop filters target caches immediately when only some
  rows finish. The following round's mirrored forward likewise refuses.
- The greedy server combines a prepared batch with the live generation batch
  through `BatchGenerator._extend_generation_batch()` and
  `GenerationBatch.extend()`. The new TP hook refuses that merge before mutation.
- If every initially collected row finishes together, no later forward touches
  the filtered cache; the next independent group creates a distinct empty cache
  and advances the TP epoch normally. Distinct prompts and independent EOS make
  this timing an observation, not a safe workload invariant.

The CPU fixture `test_tp_batch_lifecycle_readiness_audit.py` pins all four
outcomes. Existing real-loop tests `test_dflash_perrow_rollback.py` and
`test_spec_extend_active.py` independently establish that partial-finish filter
and admission merge are live DFlash paths.

## Minimum follow-up scope

The full ladder needs a protocol verb carrying the exact survivor/reorder index
vector and an append/admission description. Rank 1 must apply each operation to
every cache leaf and keep speculative row maps synchronized before the next
forward. Greedy and DFlash callers must announce before local mutation, with a
protocol bump and mixed-revision refusal. Until that is implemented and tested
through real fake-loop partial completion plus late admission, only fixed-row
batch serving is desk-ready; the full ladder is not.

No full TP logits/token identity, model, Metal, network, or hardware execution
was performed.
