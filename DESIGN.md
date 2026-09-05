# Fused KDA prefill kernel — design, cost model, risks, measurement plan

Branch `claude/kda-prefill-fused-0905` off `glm5-serve-unified` @ faa876f2.
Lever: I1292 — prefill KDA recurrence 9 % + glue 6 % of prefill time.

---

## 0. Correction to the brief, which changes the shape of the whole lever

The brief describes the prefill path as running "the eager **chunked** gated-delta
recurrence" and asks the fused version to keep "the same math order" as the
reference chunked algorithm — "chunk size, intra-chunk matmuls, inter-chunk
state propagation".

**There is no chunked (matmul) formulation anywhere in this tree.**
`mlx_vlm/models/gated_delta.py` contains exactly two implementations:

* `gated_delta_ops` (gated_delta.py:221) — a Python `for t in range(T)` loop of
  single-step ops, used only when Metal is unavailable or `head_dim % 32 != 0`;
* `gated_delta_kernel` (gated_delta.py:178) — a **single Metal dispatch carrying
  its own sequential `for t` scan** (gated_delta.py:69-101), one thread per
  (key lane, value row), state in registers.

`gated_delta_update` (gated_delta.py:269) picks the kernel for any `S`, so
**prefill's recurrence is already one fused dispatch per layer per chunk.**
"Chunked at 8192" refers to the *prompt* being fed in 8192-token blocks, not to
an intra-chunk matmul. So:

* there is no intra-chunk matmul order to preserve, and no numeric deviation to
  gate — the fused kernel can and does keep the recurrence **bit-identical**;
* the 3.1 s / 6.3 s "kda_recurrence" line in L7-c is **not** unfused overhead.
  It is the cost of one already-fused sequential scan. Fusing cannot delete it;
  it can only be made cheaper by changing its *geometry* or by removing traffic
  around it;
* the honest prize is the **glue** (2.0 s / 4.3 s) plus whatever the geometry
  change moves on the recurrence — which may be negative. See §4.

This is the single most important thing for the lead to carry into the
measurement: **this kernel is not the verify-block kernel with a bigger S, and
the verify-block receipt does not transfer.** At S ≤ 8 the glue is launch-bound
(≈33 dependent launches × 14.72 µs). At S = 8192 those same 33 launches cost
0.5 ms against a measured ≈59 ms of glue per layer — i.e. **99 % of the prefill
glue is not dispatch overhead.** It is memory traffic and elementwise work on
`[1, S, H·D]` tensors. Different physics, same-looking kernel.

---

## 1. What the eager prefill path does, per KDA layer, per chunk

`Glm5NextLinearAttention.__call__`, language.py:1500-1560 (post-patch numbering):

```
mixed  = concat(q_o, k_o, v_o)                         [B,S,3HD]
conv_input = concat(conv_state, mixed)                 [B,S+K-1,3HD]
cache[0]   = conv_input[:, -(K-1):]
_kda_glue_pre  (mx.compile'd, language.py:983):
    conv1d -> silu -> split -> 3x reshape
    a = f_b_proj(fa_o)
    q = (l2norm(q.fp32) * D**-0.5).bf16 ;  k = l2norm(k.fp32).bf16
gated_delta_update(...)  ->  gated_delta_kernel   [ONE dispatch, seq. scan]
_kda_glue_post (mx.compile'd, language.py:1005):
    gate = g_b_proj(ga_o)
    o_norm(out, gate)   # Glm5NextRMSNormGated, language.py:861: ~12 dispatches
    o_proj
```

At S=8192, H=64, D=128 (GLM-5.3-Flash `linear_attn_config`), H·D = 8192, so
every `[1, S, H·D]` bf16 tensor is **134 MB**. The chain above materialises
roughly a dozen of them (mixed, conv_input, conv_out, silu's two, q, k, v, a,
the l2norm intermediates, out, the norm's four fp32 intermediates, gate).
`mx.compile` fuses the elementwise runs within each glue half, but not across
the conv, the reductions, or the recurrence.

## 2. Dispatch geometry of the fused kernel

`mlx_vlm/models/glm5_next/fused_kda_prefill.py`.

Two arms behind one flag, because they trade the two costs against each other
and which wins is not derivable:

| arm | `NV` | launches/layer | threadgroups (B=1) | resident threads | q/k/a reads |
|---|---|---|---|---|---|
| `NV=1`  (max fusion)      | 1 | 1 | H = 64        | H·32·TY = 65 536  | 1× |
| `NV=D/TY` (max parallel)  | 4 | 2 | H·NV = 256    | 262 144           | 4× |
| eager baseline            | — | ~34 | 2048 (recurrence) + full-tensor grids | 262 144 | 1× |

* threadgroup `(32, TY, 1)`, grid `(32, TY, B·H·NV)`, `TY=32` → 1024 threads/TG.
* `z` is laid out `(b,h)`-major, `nv`-minor so the `NV` threadgroups that re-read
  the same q/k/a rows are adjacent and share cache lines.
* threadgroup `nv` owns value rows `[nv·D/NV, (nv+1)·D/NV)`; lane `l` owns key
  elements `[NDK·l, NDK·l+NDK)`, `NDK = D/32 = 4`.
* registers: `st[NDV][NDK]`, `NDV = (D/NV)/TY`. **NV=4 → `st[1][4]` = 4 floats**
  (vs 16 in the verify-block kernel at the same D). Register pressure is *lower*
  than the kernel already shipping.
* `S` is a **runtime scalar**, never templated — prefill tails are ragged and a
  templated S would cold-compile a pipeline per width.
* threadgroup memory: `4·D` floats + `(K-1)·3·D` T = 2048 + 2304 B ≈ 4.3 KB.

**Why the value-axis split exists.** `fused_kda_verify_block` runs *one*
threadgroup per (row, head). At B=1 that is 64 threadgroups on an 80-core GPU,
and 65 536 resident threads against `gated_delta_kernel`'s 262 144 — a 4×
cut in the resident thread count of the very scan it is replacing. At S ≤ 8 that
is irrelevant (the kernel is launch-bound). At S = 8192 it is the whole
question. `NV = D/TY` restores `gated_delta_kernel`'s exact partition — one
value row and 32 key elements per thread — so the recurrence runs at the same
occupancy it runs at today, with the glue folded in.

**Cost of the split.** The pre-recurrence glue reduces over the *key* axis,
which every threadgroup needs whole, so conv/silu/L2/gate are recomputed `NV`
times and q/k/a are re-read `NV` times. The gated RMSNorm reduces over the
*value* axis, which is now split, so it moves to a second launch
(`fused_kda_prefill_norm`, one simdgroup per `(b,s,h)` row, fully parallel)
reading a `[B,S,H,D]` bf16 intermediate — **which `gated_delta_kernel` already
writes today** (`y[dv] = static_cast<InT>(out)`, gated_delta.py:89), so the
split costs no traffic the eager path was not already paying, and still replaces
the norm's ~12 dispatches with 1.

## 3. Memory traffic, per KDA layer, one 8192-token chunk (H=64, D=128, bf16)

Unit = one `[1, 8192, 8192]` bf16 tensor = 134 MB.

| | eager | NV=1 | NV=4 |
|---|---|---|---|
| reads of mq/mk/mv | 3 (as `mixed`, +1 for the concat) | 3 | 12 |
| `a`, `gate` | 2 | 2 | 4 + 1 |
| conv_input / conv_out / silu | ≈4 R + 4 W | 0 | 0 |
| q, k, v, l2norm intermediates | ≈6 R + 6 W | 0 | 0 |
| recurrence out `y` | 1 W + 1 R | 0 | 1 W + 1 R |
| RMSNorm fp32 intermediates | ≈6 R + 6 W (fp32 ⇒ ×2) | 0 | 0 |
| final `y` to o_proj | 1 W | 1 W | 1 W |
| state `[H,D,D]` fp32 (4 MB) | 1 R + 1 W | 1 R + 1 W | 1 R + 1 W |
| **total ≈** | **≈ 45 units ≈ 6.0 GB** | **≈ 6 units ≈ 0.8 GB** | **≈ 20 units ≈ 2.7 GB** |

At ~700 GB/s that is ≈8.6 ms (eager) → ≈1.1 ms (NV=1) / ≈3.9 ms (NV=4) per
layer, i.e. 293 / 39 / 133 ms per chunk over 34 layers.

**Reconcile with L7-c before trusting this.** L7-c measured glue at 2.0 s per
8 k prefill = **59 ms per layer**, ~7× the bandwidth model above. So the eager
glue is *not* purely bandwidth-bound either; the surplus is most plausibly
(a) `mx.conv1d` on 24 576 depthwise channels, which MLX lowers poorly, and
(b) the hand-rolled `Glm5NextRMSNormGated` running fp32 intermediates. Both are
absorbed entirely by the fused kernel, which is *upside* relative to the table —
but it also means the table is a lower bound on the prize and **cannot be used
as the prediction.** Predict from the table; expect between it and 2.0 s.

## 4. Expected effect, stated as a falsifiable prediction

At 8 k (35 s total, recurrence 3.1 s, glue 2.0 s):

* **glue → ≈0.1-0.4 s** in both arms (it is fully absorbed; what is left is the
  extra read amplification at NV=4 and the norm launch).
* **recurrence**: NV=4 keeps the eager partition, so predict **flat ±15 %**.
  NV=1 cuts resident threads 4× while giving each thread 4× the ILP (the `j`
  loop's four value rows are independent chains), so predict **flat to 2× worse**
  — this is exactly the uncertainty the two arms exist to resolve.
* net at 8 k: NV=4 predicted **−1.6 to −1.9 s of 35 s = −4.6 to −5.4 %**;
  NV=1 predicted anywhere from −5.4 % to **+4 % (a regression)**.
* **Falsifier**: if the NV=4 arm does not beat the off arm by ≥3 % at 8 k with
  the two-load CI excluding 0, the lever is dead as built, and the residual
  hypothesis is that the glue's 59 ms/layer is dominated by something the kernel
  did not absorb (measure `mx.conv1d` alone before iterating).

## 5. Risks

1. **Occupancy at B=1 (highest).** 256 threadgroups (NV=4) on 80 cores is fine;
   64 (NV=1) is not. Mitigated by shipping both arms; not eliminated.
2. **The kernel has never been compiled.** No GPU was available for this work
   (both boxes in a measurement). Metal compile errors, `simd_sum` divergence
   and the `maxTotalThreadsPerThreadgroup` limit are all unverified. The probe
   (`fused_kda_prefill_probe`) catches the last one by halving TY, and every
   `except RuntimeError` path falls back to eager, so a build failure degrades
   rather than crashes — but **first GPU run may well be a compile fix.**
3. **`S` loop length.** 8192 iterations of a loop with 5 threadgroup barriers is
   40 960 barriers per dispatch. If barrier cost dominates, NV=1's advantage
   (fewer threadgroups → cheaper barriers is *false*; barrier cost scales with
   threads per TG, which is 1024 in both) does not materialise. Untested.
4. **fp32 state size.** `[H,D,D]` fp32 = 4 MB per layer, loaded once per chunk in
   both arms — unchanged from today, not a new risk.
5. **Register pressure.** Lower than the shipping verify-block kernel at NV=4
   (`st[1][4]` vs `st[4][4]`); the NV=1 arm is identical to it. Low risk.
6. **Wrong-flag blast radius.** Default OFF, B≤1, no speculative sink, S≥2, and
   three independent predicates (`_fused_kda_eligible` S=1,
   `_fused_kda_block_eligible` 2..16, `_fused_kda_prefill_eligible` ≥2 checked
   last) — the decode and verify paths cannot be reached by this change.
7. **Numeric deviation: none claimed and none taken.** Every arithmetic line is
   copied from `fused_kda._BLOCK_SOURCE`, which is pinned bit-identical by
   `test_glm5_next_fused_kda_block.py`; the split is partition-preserving. The
   parity tests assert `array_equal`, not a tolerance. If they need a tolerance,
   something is wrong, not merely imprecise.

## 6. Measurement plan

**Gate 0 — parity (must pass before any timing).**

```
MLX_DEFAULT_DEVICE=gpu python -m pytest \
  mlx_vlm/tests/test_glm5_next_fused_kda_prefill.py -q
```
34 tests; 20 already pass on CPU, 14 need the GPU. Bit-exactness at
S ∈ {2,5,64,300,2048} × NV ∈ {default,1}, cross-chunk state carry, masked
tokens, and end-to-end first-token logits.

**Gate 1 — L7-b harness, chunk 8192, arms off/on, 8 k and 32 k, 2 loads.**

```
MLX_VLM_GLM5_PREFILL_CHUNK=8192 \
MLX_VLM_GLM5_FUSED_KDA_PREFILL=0   # arm A (control)
MLX_VLM_GLM5_FUSED_KDA_PREFILL=1   # arm B (NV = D/TY, default)
MLX_VLM_GLM5_FUSED_KDA_PREFILL=1 MLX_VLM_GLM5_FUSED_KDA_PREFILL_NV=1   # arm C
```
Interleave the arms **inside one load** (the flag is read once into
`glm5._FUSED_KDA_PREFILL_ENV`, so an in-process A/B needs the same
module-global flip the block kernel's kill-switch test uses, not a
process-per-arm); two loads only to bound cross-load variance. Report
prefill wall time and the L7-c per-phase split (kda_recurrence / glue / proj)
so the two components are attributed separately — a win that lands entirely on
"glue" and leaves "kda_recurrence" flat is the predicted result and confirms
the geometry reasoning; a win that moves "kda_recurrence" means the L7-c
attribution itself was wrong.

**Gate 2 — L9-c-style KL gate.** Bit-exactness makes KL ≡ 0 by construction, so
this is a *check on the claim*, not on the model: any non-zero KL means the
kernel is not doing what the parity tests say and the arm must be reverted, not
tuned.

## 7. Files

* `mlx_vlm/models/glm5_next/fused_kda_prefill.py` — new, 2 Metal kernels
  (scan × {fused-norm, split-norm}, plus the standalone gated-RMSNorm), the
  geometry chooser, the threadgroup probe, and the two entry points.
* `mlx_vlm/models/glm5_next/language.py` — 5 env knobs + `_fused_kda_prefill_*`
  predicate/step, hooked after the verify-block predicate.
* `mlx_vlm/tests/test_glm5_next_fused_kda_prefill.py` — new, 34 tests.
