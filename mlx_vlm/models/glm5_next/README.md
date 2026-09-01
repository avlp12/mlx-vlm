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

Decode-only and conservative: it engages only for `B=1, S=1` with no SSM mask, and
falls back to the eager path otherwise (prefill, the `S>1` verify block,
batched/left-padded decode, or a checkpoint whose `A_log`/`dt_bias` were not kept
in fp32). Default is off.

#### Folding the small projections in (`MLX_VLM_GLM5_FUSED_KDA_QPROJ=1`)

A second opt-in also pulls `f_b_proj` and `g_b_proj` -- two `Linear(head_dim,
num_heads*head_dim)` affine-quantized GEMVs whose only consumer is this kernel --
inside the launch, leaving the chain as literally one dispatch per layer.

The in-kernel GEMV is written as a transcription of MLX's affine `qmv_quad`
(`load_vector` + `qdot` + `quad_sum`, `bits == 8`): one quad per output row, quad
lane `l` owning `x[VPT*l : VPT*l+VPT]` with `VPT = head_dim/4`. Same partition and
same accumulation order, so it stays bit-identical. Writing it as an ordinary
per-element `x * (scale*q + bias)` dot instead disagreed with `mx.quantized_matmul`
on ~0.01% of elements -- small, but enough to flip greedy tokens on 2 of 5 synthetic
seeds, which is why the exact form is the one that ships. MLX only dispatches
`qmv_quad` for `head_dim` in {64, 128} at 8 bits, so the fold declines outside that
and the layer keeps the two GEMVs outside the kernel. It also stays off on the
speculative capture path, which hands `f_b_proj`'s output back in `gdn_sink`.

Speculative decoding keeps the fusion on its single-token steps: a capture variant
of the kernel also emits the `gdn_sink` tensors (post-conv `q`/`k`/`v` straight out
of threadgroup memory, plus the pre-conv window `concatenate([conv_state, mixed])`)
that `rollback_speculative_cache` replays on a partial accept. Those are bit-identical
to the eager sink too, as is the rollback state they reconstruct. The `S>1` verify
block itself stays eager.

## Prefill optimizations

### Segment-aligned routed-expert GEMM (opt-in: `MLX_VLM_GLM5_MOE_GEMM=1`)

`SwitchGLU` sorts the routed rows by expert and calls
`mx.gather_qmm(x[N, 1, K], w[E, O, K], sorted_indices=True)`. On a non-`nax` GPU
(M3/M2 Ultra) that takes `GatherQMM::eval_gpu`'s `M == 1` branch, which dispatches
`gather_qmm_rhs` with `bm=16, bn=32, bk=32, wm=1, wn=2`. That kernel walks the
*distinct experts inside each 16-row block* and runs a **full** `16 x 32 x K`
block-gemm per distinct expert, storing only that expert's row slice. Every expert
boundary that lands strictly inside a 16-row block therefore costs one extra full
block-gemm.

GLM-5.3-Flash prefill hits this hard: 288 routed experts, top-8, prefill chunk 2048
gives 16384 sorted rows in 1024 blocks with 287 interior boundaries, i.e. ~1311
block-gemms for 1024 blocks of useful work.

Measured on M3 Ultra / mlx 0.32.0 at the real shapes
(`x[16384, 4096] @ w[288, 2048, 4096]`, 4-bit g64 affine):

| variant                                                | ms    | TFLOPS |
| ------------------------------------------------------ | ----- | ------ |
| sorted `gather_qmm`, random top-8 route (counts 36..78) | 16.65 | 16.5   |
| same, every count forced to a multiple of 16            | 13.15 | 20.9   |
| dense `quantized_matmul`, same M/N/K (`bm=32`)          | 12.18 | 22.6   |
| bf16 `matmul`, same M/N/K                               | 11.92 | 23.1   |

So the ~69%-of-dense gap reported in ml-explore/mlx#4246 is not the `bm=16` tile
(that reaches 93% of dense on its own) -- it is the redundant boundary block-gemms.

`MLX_VLM_GLM5_MOE_GEMM=1` swaps `SwitchGLU` for `Glm5NextTiledSwitchGLU`, which pads
every expert's run of sorted rows out to a multiple of `R` (`MLX_VLM_GLM5_MOE_GEMM_ROWS`,
default 16) with zero rows, so no block ever spans two experts. Wasted rows drop from
~16 per boundary to `(R - c_e mod R) mod R`: 1.265x -> 1.132x. The padding is folded
into the sort gather `SwitchGLU` already performs and the unpad into the unsort gather,
so no extra bulk traffic is added.

Whole-layer A/B at the production shape (`x[1, chunk, 4096]`, top-8, E=288, 4-bit g64,
M3 Ultra, ms per MoE layer; chunk 2048 is three independent runs, 58.23 / 58.14 / 58.05
stock):

| chunk | rows/expert | stock  | `R=16`          | `R=32`          |
| ----- | ----------- | ------ | --------------- | --------------- |
| 512   | 14.2        | 23.95  | 17.92 (1.337x)  | 24.64 (0.972x)  |
| 1024  | 28.4        | 36.37  | 29.80 (1.220x)  | 30.24 (1.203x)  |
| 1536  | 42.7        | 46.87  | 40.78 (1.150x)  | 46.61 (1.006x)  |
| 2048  | 56.9        | 58.28  | 52.67 (1.107x)  | 52.50 (1.110x)  |
| 3072  | 85.3        | 80.67  | 75.13 (1.074x)  | 74.96 (1.076x)  |
| 4096  | 113.8       | 102.90 | 97.89 (1.051x)  | 96.97 (1.061x)  |
| 8192  | 227.6       | 192.10 | 188.41 (1.020x) | 180.92 (1.062x) |

Those track `1 + 16*(E-1)/N` (stock) vs `1 + E*R/2/N` (padded) to within ~2%: the win is
biggest where the route is thinnest. At the default chunk 2048 the routed-expert path is
~53.7% of a 4551 ms chunk (450 tok/s), so `R=16` is ~+5% end-to-end prefill.

`R=16` keeps `M == 1` and therefore the `gather_qmm_rhs` (`bm=16`) kernel; `R >= 32`
hands mlx `x[T, R, K]`, which falls through to the general `gather_qmm` dispatch
(`bm=32 bn=32 wm=2 wn=2`, the dense `qmm_t` tile) -- a 1.064x better tile but coarser
padding. `MLX_VLM_GLM5_MOE_GEMM_ROWS` defaults to `auto`, which compares the two padded
row totals (both already available from the per-expert counts, so no extra sync) and
takes `R=32` only when its padding costs less than the 1.064x it buys back. On the seven
measured points that rule picks the winner or a tie every time: `R=16` up to ~85
rows/expert, `R=32` from ~114 up.

**Numerics.** Only the row layout changes -- the same `BlockMMA` runs over the same
`BK=32` K steps into the same fp32 accumulator -- so the output is **bit-identical** to
the stock path. `mlx_vlm/tests/test_glm5_next_moe_gemm.py` pins bit-identity across 5
seeds at `R=auto/16/32`, on a hot/cold route that leaves experts empty, pins a 64-token
greedy walk through 8 stacked MoE blocks over 5 seeds, and pins bit-identical logits
from a real 4-layer `LanguageModel` (3 sparse MoE layers, 4-bit g64) at 256/1024/2048
token prefill.

**Scope.** Sorted/prefill only. The path declines routes with fewer than `R * E * 3/4`
routed rows (3456 rows = 432 tokens at top-8; override with
`MLX_VLM_GLM5_MOE_GEMM_MIN`). Below that the route is thin enough that a whole `R`-row
tile per *active* expert costs more than the boundary passes it removes, and below
`4 * E` rows mlx does not take the `gather_qmm_rhs` branch for the stock path at all
(`B / E >= 4` in `GatherQMM::eval_gpu`). Measured: 0.696x at 3.6 rows/expert, 0.679x at
7.1, 1.286x at 10.7, 1.337x at 14.2 -- the crossover is between 7.1 and 10.7 and the gate
sits at 12 with margin. Decode and the speculative verify block are therefore untouched,
and so is the MTP drafter's own decode step.

**Live verdict (2026-08-31).** The synthetic +5% does **not** survive end to end, and
the reason is measured, not guessed. Four-arm in-situ ablation through the same
`generate()` the A/B harness uses (GLM-5.3-Flash-vlm-q4-quasar, M3 Ultra, 4 reps,
spread <0.4%; receipts `~/glm53flash/prep/serving/logs/moegemm_ablate2.json`):

| arm                                   | PAD prompt @2048 | real code @2048 |
| ------------------------------------- | ---------------- | --------------- |
| A stock                               | 477.29 tok/s     | 471.56 tok/s    |
| B stock + only the host syncs the path forces | 471.56 (-1.20%) | 467.78 (-0.80%) |
| C toggle ON                           | 473.18 (**-0.86%**) | 481.64 (**+2.14%**) |
| D routed experts -> zeros             | 866.12           | 886.81          |
| => live routed-expert share of prefill | **44.9%**       | **46.8%**       |
| layout benefit alone (C/B)            | +0.34%           | +2.96%          |

Three corrections fall out:

1. **The routed-expert share is 44.9-46.8%, not the 53.6%** estimated by dividing a
   synthetic layer by an assumed 4551 ms chunk. Every projected gain scales down ~0.85x.
2. **The plan forces host syncs costing a flat -0.8 to -1.2%**, which is unconditional:
   `T` (the padded tile count) is data dependent, so its `.item()` cannot be hoisted out
   of the layer loop, and `MLX_VLM_GLM5_MOE_GEMM_ROWS=auto` spends three syncs per layer
   rather than one. This tax is structural, not a tuning bug.
3. **The benefit is routing dependent**, and the A/B harness's own prompt is the
   worst case for it. Live per-layer expert histograms
   (`~/glm53flash/prep/serving/logs/moegemm_routing_hist.json`) over the harness's
   repetitive `PAD` template vs a real source file vs real prose:

   | prompt | active experts / 288 | stock waste | pad16 waste | predicted gain |
   | ------ | -------------------- | ----------- | ----------- | -------------- |
   | PAD (harness A/B), chunk 0 | 252.7 | 1.230 | 1.136 | 1.083 |
   | PAD, chunk 2 of an 8k prompt | 157.9 | 1.143 | 1.098 | 1.041 |
   | real code | 284.2 | 1.259 | 1.134 | **1.111** |
   | real prose | 283.9 | 1.258 | 1.136 | **1.108** |

   The repeated pad sentence routes to ~55-160 fewer experts per chunk, which shrinks
   the boundary term the padding removes; real text does not, and matches the synthetic
   model exactly.

So: keep the toggle **OFF by default**; it is a ~+2% prefill win on real text and a
~-1% loss on highly repetitive text, and the -1% floor is unavoidable. Do not judge it
on the `PAD` benchmark again.

**The waste model, pinned against mlx core.** A bm=16/32/64 sweep of
`affine_gather_qmm_rhs` (experiment build at `~/src/mlx-core-pr`, runtime
`MLX_GATHER_QMM_BM`; receipt `bench/stage3_core_bm.json`) discriminates two candidate
models over 18 points:

| model | form | mean abs error |
| ----- | ---- | -------------- |
| boundary-pass (this file) | `sum_e (blocks expert e touches) * bm / N` | **3.3%** |
| row-padding | `1 + bm / (2 * rows_per_expert)` | 16.9% |

The row-padding form mispredicts every bm-aligned case (it claims waste where the
measurement shows none) and underestimates the stock bm=16 waste at 56.9 rows/expert as
1.14x when it is **1.263x**. The kernel does not pad; it *re-runs a full block-gemm per
distinct expert in the block*, which is why the correct term is ~2x larger.

**No core-side prize on M3.** The same sweep, useful TFLOPS at the real gate/up shape:

| rows/expert | bm=16 | bm=32 | bm=64 |
| ----------- | ----- | ----- | ----- |
| 32          | 21.15 | 22.31 | 12.10 |
| 56.9 (chunk 2048) | **17.00** | 14.81 | 11.52 |
| 64          | 21.64 | 22.89 | 24.15 |
| 128         | 22.12 | 23.28 | 24.60 |
| realistic chunk-2048 route | **17.12** | 14.82 | 11.54 |

Bigger tiles win only when the counts are tile-aligned; at the MoE-typical 56.9
rows/expert they lose (0.87x at bm=32, 0.68x at bm=64) because the boundary re-runs
scale with `bm`. Mirroring the nax heuristic (`bm = (M/E<64) ? 32 : 64`,
`quantized.cpp:1497`) to the non-nax path would be a regression on M3. `bm=16` is
already the right choice; ml-explore/mlx#4246 has no fix to wait for.

**Prefill chunk size is not a lever either.** `--prefill-step-size` sweep on the
serving stack, 2 reps in both orders (receipts
`~/glm53flash/prep/serving/logs/chunksweep_step*_rep*.json`):

| prompt | step 2048 | step 4096 | step 8192 |
| ------ | --------- | --------- | --------- |
| 8195 tok  | 463.69 tok/s / 191.9 GB | 462.46 (-0.26%) / 198.2 GB | 438.34 (**-5.47%**) / 210.6 GB |
| 32781 tok | 345.87 tok/s / 200.2 GB | 345.07 (-0.23%) / 211.1 GB | 329.69 (**-4.68%**) / 229.7 GB |

The 1.21x drop in MoE us/token at chunk 8192 is real but is more than paid for by the
rest of the stack, and peak memory grows 19-30 GB. **Keep `prefill_step_size=2048.**

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

## MoE prefill GEMM (`MLX_VLM_GLM5_MOE_GEMM`) — ON by default since 2026-09-01

Segment-aligned routed-expert GEMM for prefill. Set the variable to `0` to
restore the stock path.

Measured twice on **real text** (source code + prose), paired and interleaved,
chunk 2048: round 1 **+2.14%**; round 2 **+2.77% / +3.92% / +3.14%** (3/3 pairs,
median +2.77%). Round 1 saw the same real-text win and still recommended OFF on
the strength of a −0.86% result from a repeated PAD sentence — that weighting was
wrong, and the campaign law "never judge this toggle on PAD templates" exists
precisely because degenerate routing is unrepresentative.

**Two caveats, first-class:**

* **The sign flips on degenerate prompts.** A repeated PAD sentence activates
  158–253 of 288 experts per chunk and regresses ~0.86%; real text activates
  281–285 and gains 2–4%. Genuinely repetitive traffic will be slightly slower.
* **It is structurally noisier.** Run-to-run spread is 2.71–4.14% on against
  0.41–1.64% off, because `T` (the padded tile count) is data dependent and its
  `.item()` cannot be hoisted out of the layer loop — three host syncs per layer
  instead of one. A property of the design, not measurement error.

Receipts: `bench/moegemm_realtext_{off,on}_r{0,1,2}.json`,
`bench/moegemm_ablate2.json`, `bench/moegemm_routing_hist.json`.
Harness: `bench/moegemm_realtext_ab.py`.
