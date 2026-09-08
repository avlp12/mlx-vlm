# TP heterogeneous batch cache contract audit

Verdict: **BLOCKED for another full-model heterogeneous TP launch.** The two prior fixes remove the embedding refusal and vector-offset exception, but they expose an older rank-cache asymmetry that can silently change output.

Rank 0 constructs `BatchKVCache(left_padding)` in `generate/ar.py:_make_cache` and passes that cache into the model. Its attention mask is per row, and after a `[2,0]` left pad its offsets become `[1,3]` after the three-column prefill. Rank 1 handles `OP_MAKE_CACHE` with `self.lm.make_cache()`, a regular scalar cache, then forwards the same batched IDs without a mask or left-padding metadata. Its offset is `3`; its prefill mask is a shared causal matrix and its first decode mask is `None`. These are not equivalent to rank 0's `(B,1,L,K)` masks.

The CPU fixture `test_tp_heterogeneous_contract_audit.py` reproduces both prefill and decode differences without a model or Metal. This is a correctness risk rather than a cleanup risk: collective shapes may still match while the two shards compute with different attention masks, allowing plausible but invalid summed logits.

A safe implementation requires both ranks to create the same batch-aware cache with the same per-row left-padding vector. That requires carrying authenticated padding metadata in the control protocol, incrementing `PROTO_VERSION`, constructing corresponding batch cache types on rank 1, and testing cache/mask/output identity through prefill and at least one decode step. Full TP output identity remains unverified and hardware execution is not ready.
