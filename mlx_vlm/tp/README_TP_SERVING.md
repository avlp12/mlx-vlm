# TP=2 serving mode

`MLX_VLM_GLM5_TP_HOSTS=10.0.0.1,10.0.0.2` turns it on. Absent, nothing changes.

    MLX_VLM_GLM5_TP_HOSTS          two or more IPs on the fast link; presence = on
    MLX_VLM_GLM5_TP_WORKER_PYTHON  interpreter for rank 1
    MLX_VLM_GLM5_TP_WORKER_SRC     source tree for rank 1
    MLX_VLM_GLM5_TP_WORKER_MODEL   peer's checkpoint path (defaults to rank 0's)
    MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD  control-plane payload width (8192)
    MLX_VLM_GLM5_TP_STEP_TIMEOUT_S  watchdog bound on one announced step (300)
    MLX_VLM_TP_HEAVY_RSS_GB        sequencing-guard threshold (20)

Rank 1 is started for you over ssh. It runs `python -m mlx_vlm.tp.worker`, which
imports nothing from `mlx_vlm.server` — a rank-1 box needs no HTTP stack.

## The one invariant

**The control plane is a data collective.** Every verb rides the same `all_sum`
the sharded layers already need: rank 0 fills a fixed-width int32 vector, rank 1
contributes zeros, the sum hands both the same message. There is no side
channel, so the control stream cannot desynchronise from the data stream — it
*is* one of them. A feature that cannot be built without bending this does not
get built.

One consequence shapes every verb: the vector is one-directional, because rank 1
contributes zeros. When rank 0 needs an *answer* — "do you hold this vault
rung?" — the answer is a second collective with the roles swapped, not a socket.
`OP_VAULT_RESTORE` is the only verb that needs one.

    op   verb            carries                       who acts
    0    EXIT            —                             rank 1 stops
    1    MAKE_CACHE      epoch                         rank 1 builds an EMPTY cache
    2    FORWARD         epoch, (b,s), ids, capture    both ranks run the forward
    3    ROLLBACK        epoch, accepted[], block      each rank rolls back its own half
    4    VAULT_STORE     epoch, name, prefix_len       each rank checkpoints its own half
    5    VAULT_RESTORE   epoch, name, prefix_len       each rank restores its own half, + ack

`PROTO_VERSION`, the header width and the payload width are agreed in preflight
over a vector whose width is frozen forever. A revision skew between the boxes
would otherwise mismatch the very first control collective — which on jaccl is a
hang, not an error.

## What the mirror owes, and what it refuses

Rank 1's cache must always be reconstructible from what rank 0 announced. Three
things can break that, and each has a verb or a refusal rather than a silent
divergence:

* a forward rank 1 cannot reproduce from token ids (**multimodal
  `inputs_embeds`**) → refused. Checked per call, never cached: whether a prefill
  is multimodal is a property of the request, not of the model.
* a **mutation of the cache outside a forward** (speculative rollback, vault
  restore) → announced.
* a cache that arrives **already populated but never announced** → refused.
  `OP_MAKE_CACHE` says "build an EMPTY cache", so rank 1 would start from
  nothing while rank 0 starts from history. Known producers: continuous batching
  admitting a second request mid-generation (`generate/ar.py::_extend_cache`
  merges the batch caches in place and returns a new list), and APC warm caches.
* **KV-cache quantization** (`KV_BITS`) → refused at startup. `generate/ar.py`
  calls `maybe_quantize_kv_cache` on the prompt cache *between* forwards and
  rank 1 is never told.

## Speculative decoding in TP mode (rank-0-only drafter)

**Premise, verified in code.** In the megatron sharding the residual stream is
replicated on both ranks after every reduce, so rank-0-local hidden capture needs
no new collectives:

* `tp/glm5_next.py::shard_layer` wraps `layer.self_attn` and `layer.mlp` in
  `AllReduce`, so both of a decoder layer's contributions to the residual are
  all-summed before they are added (`models/glm5_next/language.py`, lines 1409–1412
  and 1425–1428).
* Everything else the residual passes through — `attn_hc`/`ffn_hc`, both
  RMSNorms, `embed_tokens`, `lm_head` — is deliberately **not** sharded
  (`tp/glm5_next.py` module docstring, and `shard_layer` touches nothing else).
* The DFlash capture site is `language.py:1492`, `hidden_sink.append(h.mean(axis=2))`,
  executed **after** `h = layer(...)` returns — i.e. after both reduces. So the
  hidden rank 0 captures is the same tensor rank 1 holds, and reading it costs
  nothing on the wire.

Confirmed empirically by the stage-3/4 driver: rank0 and rank1 emitted
byte-identical tokens over 257 tokens.

**What is not replicated** is the KDA recurrent state: it is head-split, so each
rank holds only its own heads' history. That is why rollback is a *verb* rather
than a rank-0-local operation — each rank replays its own heads and trims its own
copy of the (replicated) DSA latent, and only `(accepted, block_size)` crosses.

**What rank 1 is told about capture** is a flag, not the layer id list. The ids
belong to the drafter, which is rank-0-only. The flag makes rank 1 pass
`capture_layer_ids=[]`, which allocates `gdn_sink` without capturing hidden
states. That is numerically inert — the sinks are appended to and never read back
into the residual, and `_fused_kda_eligible` explicitly `del`s `gdn_sink` — but
`gdn_sink is not None` is exactly what makes a KDA layer stash the block inputs
its own rollback needs, so it must be set on **both** ranks or neither.

## Vault in TP mode (per-rank, shard-local)

Each rank saves and restores *its own* half of the state; the only thing that
crosses is the 128-bit name of the boundary. Under TP each rank already holds
exactly the half it computed, so shipping state would move bytes the receiver
cannot use and the sender already has.

**Identity separation.** `context_vault.vault_identity` folds in a topology
descriptor (`tp/vault.py::topology_descriptor`: size, rank, layer split,
reduce count, model path), so a single-box rung, rank 0's rung and rank 1's rung
cannot even *name* the same boundary. Changing the topology drops the store,
because every rung already in it was named under the old one.

**Cross-topology restore is refused, not merely missed.** A checkpoint can reach
`restore_into` without ever having been in that vault — from the peer tier, or
from the other rank — and the shapes match either way, so the check is on
provenance: `VaultCheckpoint.origin` is stamped at insert and compared at
restore. A mismatch logs both identities and falls back to a cold prefill.

**The ack exists because the two stores evict independently.** "Rank 0 holds the
rung" does not imply "rank 1 holds it". Rank 1 restores first and answers over a
roles-swapped `all_sum`; on a miss rank 0 discards its own restore and both ranks
prefill cold. Serving a warm rank 0 against a cold rank 1 would sum halves of
different states into a fluent wrong answer.

## Teardown, and the sequencing guard

On 2026-08-31 an HTTP end-to-end server hung in shutdown with its model still
resident, the next load started anyway, and the box froze hard enough to need a
power cycle. Both halves of that are now closed:

* **Shutdown actually unloads.** The FastAPI lifespan `finally` calls
  `unload_model_sync()`; the unload path announces `OP_EXIT` to the peer, exits
  the `wired_limit` context (which is now *owned* by the mirror — held as a bare
  local it was collected the moment the loader returned, quietly restoring the
  old limit, so the process that believed it had wired the model had not), drops
  the model reference, and logs a teardown report carrying `active_memory_gb`
  and `rss_gb`. The `atexit` hook holds a **weakref**: a bound method there pins
  the mirror, and through it every weight tensor, for the life of the
  interpreter — defeating every unload the server performs.
* **The next load refuses to start until the box is quiet.**
  `mlx_vlm.tp.fleet` is the shared preflight, the same rule the EAGLE-3 bulk
  wrapper already used (`refuse while any python RSS > 20 GB`), made reusable,
  extended to the peer box, and made to return a receipt so "the guard ran and
  found the fleet quiet" is distinguishable from "nobody called the guard". It
  refuses rather than assumes when a box cannot be inspected.

Every heavy-run driver should call it:

    python -m mlx_vlm.tp.fleet --hosts 10.0.0.1,10.0.0.2 --label mylabel   # rc 75 if busy
    python -m mlx_vlm.tp.fleet --wait --hosts ... --label ...              # queue instead

`maybe_load_tp` calls `require_quiet_fleet` itself before launching the worker.

## Fault behaviour

A dead peer leaves the fast fence's GPU kernel spinning on a shared counter, and
nothing on the host can preempt that (see `tp/transport.py::Deadman`). The
serving loop therefore carries a watchdog: one long-lived thread and a tuple
store per step (a per-forward `threading.Timer` would be a measurable fraction of
a 6 ms B=8 step), which after `MLX_VLM_GLM5_TP_STEP_TIMEOUT_S` logs the in-flight
verb and exits 75 (`EX_TEMPFAIL`) so a supervisor can restart the server rather
than leave a wedged 94 GiB process. **Run the server under a supervisor** — the
recovery is a restart, by construction.

Transport, worker launch, sharded load or fleet-preflight failure logs and
returns None, and the server comes up single-box. Refusing to serve because a
second box is unreachable would be the worse failure.

## Measured (3 pairs, alternated order, real-text 512-token prompt)

| | TP=2 | single | ratio |
|---|---|---|---|
| B=1 decode | 38.0 tok/s | 29.9 | **+27.4%** |
| B=8 decode | 161.1 tok/s | 109.0 | **+47.7%** |
| B=1 prefill | 633 tok/s | 378 | **+67.4%** |
| peak memory | 94.5 GiB/box | 183.2 GiB | **48% less per box** |

Spread within each arm is ≤0.5%.

## What TP mode still does not do

* **Reproduce single-box tokens.** `all_sum` adds two partial sums where one
  device summed all 4096; at the measured one-ULP top-2 gap the argmax flips.
  Cross-lane token identity is a TP-off property. The TP-mode invariant is
  rank0 == rank1.
* **Concurrent generations with rolling admission.** Mode-level TP assumes one
  live cache. A second request joining a running batch merges the caches outside
  any forward, which the mirror now *refuses* (it used to desync silently: on
  2026-08-31 one concurrent request returned and the other timed out at 90 s).
  Serve a fixed batch, or `MLX_VLM_MAX_NUM_SEQS=1`. Making this work means
  announcing batch composition — a real design, and a much larger protocol.
* **Multimodal prefill.**

## Long-context checklist

* `MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD` caps `b * s` per announced forward,
  not `s`. Default 8192; `DEFAULT_PREFILL_STEP_SIZE` is 2048, so B=1..4 chunked
  prefill fits and **B=8 × 2048 = 16384 does not**. Raise it on **both** ranks
  (it is agreed in preflight, so a one-sided change is refused rather than
  hung).
* `tp/glm5_next.py:103` sets `attn._fused_ready = False` after sharding, so the
  fused-KDA kernel re-selects and recompiles at half `H`. Keep it on — it was on
  for every measured run — but expect a one-off recompile at the first new shape.
