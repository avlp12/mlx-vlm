# GLM-5-Next (`glm5_next`)

MLX support for the GLM-5-Next architecture, as shipped in **GLM-5.3-Flash**.

## Architecture

- **Hybrid decoder** — 34 Kimi-Delta linear-attention (KDA) layers interleaved with 11 DeepSeek sparse-attention (DSA) layers.
- **MLA with NoPE** and a **lightning indexer** (top-`index_topk` key selection over pooled keys).
- **288-expert MoE** (top-8) with a shared expert; **mHC hyper-connections**.
- A **multi-token-prediction (nextn) head** at the final layer, used for self-speculative decoding.

## Usage

```python
from mlx_vlm import generate, load

model, processor = load("zai-org/GLM-5.3-Flash")
print(generate(model, processor, "Explain multi-head latent attention.", max_tokens=256))
```

## Decode optimizations

All are on the compute path (no weight changes) and lossless:

| optimization | effect |
| --- | --- |
| KDA input-projection fusion | the six shared-input KDA projections become one (quantized) matmul via a lossless output-axis weight concat |
| Lightning-indexer chunked prefill | bounds prefill peak memory to `O(chunk · P)` (avoids a one-shot `O(S · P)` blow-up at long context) |
| Lightning-indexer incremental decode | per-step pool cost `O(T)` → `O(index_kpool)` (reuses stable complete pools) |
| Short-context dense-MLA bypass | when the cache fits within `index_topk` the indexer would select every token, so it is skipped and DSA falls through to dense MLA |
| Last-token `lm_head` | skip the vocab-wide projection on discarded prefill positions |
| FFN-block compile | `mx.compile` the stateless FFN half, scoped to single-stream decode |

### Fused KDA decode step (opt-in: `MLX_VLM_GLM5_FUSED_KDA=1`)

After the input projections, a KDA layer's decode step is a long tail of tiny
kernels -- causal conv1d window update, silu, two L2 norms, the safe forget gate,
`sigmoid` beta, the gated delta-rule state update, gated RMSNorm -- roughly 30 GPU
dispatches per layer, 34 layers per token. The arithmetic is trivial; the launch
count is the cost.

`fused_kda.py` folds that whole chain into **one** `mx.fast.metal_kernel` launch
per layer. One threadgroup owns one head, so both cross-`head_dim` reductions (the
L2 norms over the key axis, the RMSNorm over the value axis) stay in threadgroup
memory, while the `[head_dim, head_dim]` recurrent state streams through registers
exactly once. State accumulation stays fp32, as in the eager kernel.

The kernel is a rounding-faithful transcription, so its output *and* the fp32
recurrent state it carries are **bit-identical** to the eager path -- verified over
32 consecutive decode steps in
[`test_glm5_next_fused_kda.py`](../../tests/test_glm5_next_fused_kda.py).
Getting that exact means matching where MLX rounds: `mx.conv1d` writes bf16,
`nn.silu` rounds twice, `gated_delta_kernel` writes `y` in the input dtype, and
`mx.sigmoid` evaluates in bf16 (not fp32) for a bf16 input. It also means matching
*which* `exp` MLX used -- its `Sigmoid` is written with `metal::exp`, which is
precise inside MLX's prebuilt library but the fast approximation inside anything
JIT'd (`mx.compile`, custom kernels) -- and disabling fma contraction in the norm
reductions, since `(x*x).sum(-1)` rounds the square before the add.

Decode-only and conservative: it engages only for `B=1, S=1` with no SSM mask and
no speculative capture, and falls back to the eager path otherwise (prefill, the
`S>1` verify block, batched/left-padded decode, a drafter's capture forward, or a
checkpoint whose `A_log`/`dt_bias` were not kept in fp32). Default is off.

## Self-speculative decoding (MTP)

GLM-5.3-Flash ships one trained nextn (MTP) layer inside the target checkpoint.
`mlx_vlm.convert` drops it during a normal conversion, so it is extracted into a
standalone drafter and used for self-speculative decoding. Each round the drafter
proposes one token from the target's hidden state and the target verifies the
`[bonus, draft]` block in a single forward.

- **Short-block verify gather.** The DSA verify path gathers each query's
  top-`index_topk` selected latents rather than masking over all keys, so
  verifying a short speculative block stays `O(index_topk)` instead of
  `O(context)`.
- **KDA block verify + rollback.** The linear-attention layers verify a block on
  the shared fused gated-delta kernel and roll the per-step recurrent state back
  to the accepted prefix on a partial-accept.
- **Never-lose adaptive pause.** With a single nextn head the block is 2 tokens,
  so a round only helps when the draft is accepted often enough to clear the
  verify overhead -- and that break-even shifts with context length and batch
  size. The orchestrator calibrates the drafting-vs-plain cost once, then runs a
  plain decode step whenever recent acceptance can't clear it (re-probing to
  resume). The target verifies every token, so the gate only trades drafting
  throughput and never drops below the baseline by more than the plain-step
  overhead.

See [`mlx_vlm/speculative/drafters/glm5_next_mtp/README.md`](../../speculative/drafters/glm5_next_mtp/README.md)
for the split tool and usage.

## Speculative decoding (DFlash2)

GLM-5.3-Flash can also be driven by a **DFlash2** drafter -- a separate,
multi-token block drafter (block size 8) trained against the target and published
as `incoai/GLM-5.3-Flash-DFlash2`. Unlike the single-token nextn head it proposes
a whole block per round, so acceptance clears the verify overhead by a wide
margin.

Each round the drafter reads the target's hidden state at the checkpoint's
`target_layer_ids` and proposes a block; the target verifies the block in one
forward and, on a partial accept, rolls the KDA recurrent state and the
MLA/indexer caches back to the accepted prefix (the same rollback MTP uses).
Output is bit-identical to plain decode.

Target-side support is small: the model captures the per-layer hidden at
`target_layer_ids` and returns it (with the KDA states) from the verify forward;
the drafter, its round loop, and the converter are shared across DFlash targets.
The drafter runs in bf16 as published -- it does not need quantizing to pair with
a quantized target.

## Continuous batching

Runs the batched `BatchGenerator` path unmodified. The lightning indexer's incremental
pool and the DSA decode mask are batch-aware, so grow/shrink of the batch
(`filter`/`extend`) stays correct.
