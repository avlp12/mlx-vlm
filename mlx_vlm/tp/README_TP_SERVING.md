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

**What a restore does not carry.** `KVCacheCloneAdapter.clone` copies exactly
`keys`, `values` and `offset`. The DSA layer's cache is
`CacheList(KVCache, KVCache)` whose second entry is the indexer cache, carrying
`_pool`, `_fpool` and `_no_pad` — derived state the model sets during a forward.
None of it survives. Tested (`tests/test_vault_indexer_state.py`): the pool is
rebuilt from the restored keys and **a restored cache selects the same KV blocks
as a live one**, so this costs time on the first post-restore forward, not
agreement. `_no_pad` is read with `getattr(..., False)`, so a restored cache
takes the slow visibility path until it is recomputed.

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
  `mlx_vlm.tp.fleet` is the shared preflight: refuse (or queue) while any
  process on either box holds more than 20 GB, returning a receipt so "the
  guard ran and found the fleet quiet" is distinguishable from "nobody called
  the guard", and refusing rather than assuming when a box cannot be inspected.

  **It does not use RSS, and the reason matters.** The EAGLE-3 bulk wrapper's
  guard is `ps -A -o rss,command | awk '$1>20000000 && /[Pp]ython/`, and that
  cannot fire: measured, 8 GiB of live `mx.array` moves a process's RSS by
  0.01 GB, and a loaded 85.5 GiB shard reports about 3 GB. MLX allocates Metal
  buffers and macOS does not count them in `resident_size`. The guard that was
  supposed to prevent the freeze was structurally incapable of it. What does
  track the allocation — `footprint(1)`, `vmmap --summary`, `top`'s MEM column
  and `proc_pid_rusage`'s `ri_phys_footprint` — all agreed to within 0.5% on
  the same 8 GiB. This module reads `top` for the fleet scan and
  `proc_pid_rusage` for "how big am I", and it was checked live against a
  deliberate 32 GiB holder (detected at 32.0 GB; RSS read 0.03 GB).

Every heavy-run driver should call it:

    python -m mlx_vlm.tp.fleet --hosts 10.0.0.1,10.0.0.2 --label mylabel   # rc 75 if busy
    python -m mlx_vlm.tp.fleet --wait --hosts ... --label ...              # queue instead

`maybe_load_tp` calls `require_quiet_fleet` itself before launching the worker.

## Operating cost: wired memory, and why abrupt kills used to age the box

A worker that is killed rather than asked to exit **leaks its entire shard as
wired memory that survives the process**. Measured 2026-09-01 on the peer:
wired 206.3 GB / free 112.0 GB before a bare SIGTERM of a worker holding an
85.5 GiB shard, wired 305.6 GB / free 9.5 GB after. The box could not take
another load, and only a reboot reclaimed it.

The cause is ordinary: Python's default SIGTERM disposition terminates without
unwinding, so `wired_limit.__exit__` never runs and the shard is never dropped.
The clean path — rank 0 announces `OP_EXIT`, the loop returns, the `finally`
runs — releases fully (the same box sat at 7.1 GB wired after a clean cycle).
The worker now installs SIGTERM/SIGINT/SIGHUP handlers that raise `SystemExit`
so the signal path *is* the clean path; verified live, a bare SIGTERM of a real
materialized shard returned 86.1 GB.

This is the whole explanation for the campaign's memory history: the freeze, the
95% swap incident, and 449 GB of wired held by no visible process were all
workers that got a signal instead of a verb. **Prefer `OP_EXIT` / `/unload`
anyway** — the handler cannot run while the process is blocked inside a
collective, so a genuinely wedged rank still costs a reboot.

## Running heavy sweeps safely

Three freezes in this campaign came from the same shape: a box approved for a
load it could not actually finish. The rules that came out of them:

* **Gate on headroom for the load, not a floor.** `require_headroom(load_gb,
  margin_gb)` asks for `free >= load + margin`. A fixed floor says nothing about
  what is left *after* an 86 GiB shard lands — a box at 118 GB free passes a
  100 GB floor and then runs at 32 GB. `SHARD_GB` carries the usual sizes.
* **Watch the trend, not just the level.** `DebtWatch` records wired at sweep
  start and stops the sweep once it grows by more than a shard. Every watchdog
  abort leaks one, so an N-arm sweep degrades monotonically and each individual
  gate call still says yes.
* **Short step timeouts for long context** — 120 s, not 900. A wedged rank holds
  its shard and spins the GPU fence for the whole window; at 32k a prefill chunk
  is ~18 s, so 120 s is generous and caps the damage at two minutes.
* **Measure a curve, not a monolith.** `prep/tp2/lc_curve.py` drives the chunk
  loop directly, times every chunk, and stops on a wall-clock budget. The old
  monolithic 32k/65k prefill produced one number per hour and could not be
  abandoned safely.
* **No concurrent GPU work on a box running a long-context arm** — not even a
  test suite. The 2026-09-01 freeze had a wedged rank spinning the fence next to
  another lane's kernels on the same device.

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
* **Vault restore, live.** The protocol works end to end — rank 1 stores under
  the announced name, restores it, and re-prefills only the tail (observed:
  `VAULT_STORE name=874732f45a45`, then `VAULT_RESTORE`, then a forward of
  `s=1678` for a 5774-token prompt restored at 4096). But the first forward
  *after* a restore stalled indefinitely on 2026-09-01 and the cause is not
  known. Ruled out: dropped indexer pool state (a restored cache selects the
  same blocks), and unmirrored forward kwargs (the whitelist would name them).
  Do not enable `MLX_VLM_GLM5_VAULT` under TP until this is understood.

## The gather gate is per-lane: TP defaults to 65536

Measured 2026-09-01 as a per-chunk prefill cost curve (`prep/tp2/lc_curve.py`),
2048-token chunks:

| lane | dense cost | gather plateau | crossover |
|---|---|---|---|
| single-box | 3924 + **252.8**·chunk ms | 8670 ms | **~38k** |
| TP=2 | 2306 + **122.9**·chunk ms | 6394 ms | **~68k** |

TP splits the attention heads, so the dense `O(S*T)` term halves (252.8 → 122.9,
a 2.06× ratio) while the gather path's fixed cost falls only 1.36× — it is
dominated by indexer and gather work that parallelises less. The crossover
therefore moves right. Leaving the single-box default of 32768 in place under TP
costs **13.9%** of prefill time to 65k (153.8 s vs a projected 132.4 s
all-dense).

`shard_model` raises the gate to 65536 for the TP lane; an explicit
`MLX_VLM_GLM5_GATHER_MIN_CONTEXT` always wins. It is applied at sharding rather
than passed through the environment, so **both ranks get it without a
passthrough** — passthroughs are how rank 1 once received `NAME=` and died at
`int('')`.

**Where the curves stop, and what is extrapolation.** They were measured to 65k.
Beyond that: extrapolating dense to 131k (chunk 64) gives
`2306 + 122.9*64 = 10.2 s/chunk` against a flat **6.4 s** gather plateau, so
gather is well ahead past the ~68k crossover and 65536 engages it about where it
starts paying. The 131k end of that is arithmetic, not measurement.

The knee was verified to move with the gate — 32768 → knee at kv_len 32768,
16384 → knee at kv_len 16384 — which is what identifies dense-attention depth
(rather than, say, indexer pooling) as the thing that grows.

## Long-context checklist

* `MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD` caps `b * s` per announced forward,
  not `s`. Default 8192; `DEFAULT_PREFILL_STEP_SIZE` is 8192 since 2026-09-05
  (it was 2048 when this section was written), so only **B=1** chunked prefill
  fits at the default -- B=2 × 8192 = 16384 does not, and neither did the old
  B=8 × 2048. Either lower `PREFILL_STEP_SIZE` for batched TP prefill or raise
  the cap; if raising it, do so on **both** ranks
  (it is agreed in preflight, so a one-sided change is refused rather than
  hung).
* `tp/glm5_next.py:103` sets `attn._fused_ready = False` after sharding, so the
  fused-KDA kernel re-selects and recompiles at half `H`. Keep it on — it was on
  for every measured run — but expect a one-off recompile at the first new shape.
