# The TP mirror against mlx-lm's distributed inference

A read-only study, 2026-09-01. No code changed; every verdict below is either
"keep ours and here is why" or "adopt, and here is the size of it".

**What I read.** mlx-lm at `44b42cc` (version 0.32.0), the whole distributed
path: `mlx_lm/utils.py::sharded_load`, `mlx_lm/models/deepseek_v32.py`
(`shard`, `pipeline`, the MoE call), `mlx_lm/chat.py`, `mlx_lm/cli_ui.py`,
`mlx_lm/server.py`, plus `mlx/nn/layers/distributed.py` and
`mlx/_distributed_utils/launch.py` from mlx 0.32.2, which are the primitives
underneath it. Our side is `mlx_vlm/server/tp_mode.py`, `mlx_vlm/tp/*` and the
sharded call sites in `mlx_vlm/models/glm5_next/language.py`.

**The closest model on their side is not our model.** `mlx_lm/models/glm_moe_dsa.py`
is 54 lines of config that subclasses `deepseek_v32.Model` unchanged — no
`index_kpool`, no linear-attention layers, no hyper-connections. So it is a
GLM-4-family MoE with DSA, not GLM-5.3-Flash's 34 KDA + 11 DSA stack. Where I
compare model code below, it is against `deepseek_v32.py`, and the differences
that follow from the architecture are labelled as such rather than counted as
design choices.

## 1. The one thing we independently agree on, and it is the load-bearing one

Our design note says the control plane rides the data collective so that there
is no side channel to desynchronise (`mlx_vlm/server/tp_mode.py:17-19`,
`mlx_vlm/tp/worker.py:10-13`). mlx-lm's **server** does the same thing:

```python
# mlx_lm/server.py:463-481
def _share_object(self, obj):
    ...
    if self._rank == 0:
        data = mx.array(pickle.dumps(obj))
        mx.eval(mx.distributed.all_sum(data.size))
        mx.eval(mx.distributed.all_sum(data))
```

Rank 0 pickles the request and ships it over `all_sum`; every other rank
reconstructs it (`server.py:475-481`), and `_share_request` (`server.py:483-494`)
is what every rank calls instead of reading its own queue (`server.py:452-461`).
That is the same principle we wrote down, reached independently. It is worth
saying plainly because it is the part of our design that was most obviously
"ours" and it turns out to be convergent.

**But their `chat` CLI does the opposite**, and I would have got this wrong if
I had only read the server:

```python
# mlx_lm/cli_ui.py:278-281
def prompt(self) -> str:
    if self._rank == 0:
        return corridor_input(self._console)
    return input("")
```

Rank ≥1 reads its **own stdin**. What makes that work is the launcher, which
fans the user's keystrokes out to every rank over the SSH pipes:

```python
# mlx/_distributed_utils/launch.py:251-256
rlist, _, _ = select([sys.stdin.fileno()], [], [], 0.1)
for fd in rlist:
    stdin_buffer = os.read(fd, 8192)
    for q in stdin_queues:
        q.put(stdin_buffer)
```

So `chat` is SPMD with an out-of-band control channel, kept in agreement by
(a) identical program, (b) identical stdin bytes from the launcher, (c) one
seed set on every rank (`mlx_lm/chat.py:108`), and (d) logits that are equal
across ranks after the collective. Four assumptions, any one of which can fail
quietly. The server does not rely on any of them.

## 2. Sharding strategy

|  | mlx-lm | ours |
|---|---|---|
| attention | `q_b_proj` all-to-sharded, `o_proj` sharded-to-all, `num_heads //= N`, `embed_q`/`unembed_out` sliced by head range (`deepseek_v32.py:588-608`) | `shard_kda` / `shard_dsa` per layer type, `num_heads //= size` plus a depthwise-conv split the KDA layers need (`mlx_vlm/tp/glm5_next.py:66-140`) |
| dense MLP | gate/up all-to-sharded, down sharded-to-all (`deepseek_v32.py:611-620`) | same shape (`mlx_vlm/tp/glm5_next.py:142-147`) |
| MoE | `shard_inplace` on shared + switch experts, and the module owns its own `all_sum` (`deepseek_v32.py:623-640`, `deepseek_v32.py:366-379`) | `shard_moe` + an `AllReduce` wrapper around the whole mlp (`mlx_vlm/tp/glm5_next.py:149-183`) |
| DSA indexer | **not sharded at all.** `shard()` never touches `Indexer`, so every rank computes it with every head | **sharded**, with a reduce installed on the module (`mlx_vlm/tp/glm5_next.py:120-140`) and applied before the top-k (`mlx_vlm/models/glm5_next/language.py:1205-1206`) |
| pipeline parallelism | supported, layer-range split with `send`/`recv_like` and a closing `all_gather` (`deepseek_v32.py:429-478`) | not implemented |
| coverage | 20 of 127 model files define `shard`, 5 define `pipeline` | one model |

Two of those deserve verdicts rather than a row in a table.

**The indexer.** Theirs replicates it; ours splits its heads and pays an
all-reduce per DSA layer. Ours is not a free win — it creates a correctness
hazard their design cannot have, because each rank contracts only its own half
of the head axis and *ranking a partial sum selects different KV blocks on each
rank*. That is why the reduce has to happen before the top-k, and why the fast
decode path issues a reduce it does not otherwise need, purely so the collective
count cannot depend on which path a rank took (`mlx_vlm/models/glm5_next/language.py:943-957`).
Keeping it is a measured call, not a preference: the gather-path plateau under
TP=2 is 6394 ms/chunk against 8670 single-box (`~/glm53flash/logs/tp2/lc_curve_*_g32768.json`,
1.36×), and the indexer is the part of that path our own note calls the one that
"parallelises less" (`mlx_vlm/tp/glm5_next.py:196-199`). **Verdict: KEEP.**
Worth recording that the simpler design exists and is one `shard()` line away
if the hazard ever costs more than 1.36× is worth.

**Pipeline parallelism.** They have it and we do not, and it is not a small
feature — it is what makes a model bigger than one box's memory runnable at all,
and their loader only downloads the shard a rank needs (`mlx_lm/utils.py:609-624`).
**Verdict: PARK, with the reason written down.** At 2×512 GB our model fits in
one box; pipelining would buy capacity we are not short of, at the cost of
serialising the layer stack across the link. It becomes interesting at 4 boxes
or at a model past ~400 GB, and `deepseek_v32.py:429-478` is the reference to
start from.

## 3. Collective count

Structurally they issue about 2 per layer: one in the attention's
sharded-to-all `o_proj` (`mlx/nn/layers/distributed.py:333` and `:585` for the
quantized form) and one in the MoE (`deepseek_v32.py:377`). Nothing for the
indexer, and `sum_gradients` (`deepseek_v32.py:368`) is inference-inert.

We issue 2 per layer for the same reasons, **plus** one per DSA layer for the
indexer, plus one per forward for the control message. Our instrumented count
is 135 per prefill chunk against 92 on the first
(`~/glm53flash/logs/tp2/lc_curve_tp_g32768.json`, `collectives` field), and the
indexer reduce was previously costed at +11 per decode step, i.e. 5% at B=1 and
0.6% at B=8. So our per-forward collective bill is roughly 1.2× theirs by
construction, bought for the 1.36× above and for a control plane that cannot
desynchronise.

One structural difference in the *shape* of the control traffic: their
`_share_object` costs **two** collectives per message (size, then payload) and
can carry any picklable object; ours costs **one**, because the message is a
fixed-width int32 vector (`mlx_vlm/tp/worker.py:194-228`) with strings packed
16 bits per word (`mlx_vlm/tp/worker.py:174-190`). Ours is cheaper and less
general; theirs is more general and allows rank ≥1 to re-run the whole request
path rather than only the forward.

## 4. Cache handling

This is where the designs diverge most, and it follows from the mirror being
asymmetric.

Because mlx-lm is SPMD, cache identity is not a problem they have: every rank
builds its own `prompt_cache` from the same request and the same code, so the
caches correspond by construction. `LRUPromptCache` lives on every rank alike.

Our rank 1 runs no request path, so cache identity has to be carried explicitly.
That produced machinery they have no analogue for:

* `OP_MAKE_CACHE` / `OP_ROLLBACK` / `OP_VAULT_STORE` / `OP_VAULT_RESTORE`
  (`mlx_vlm/tp/worker.py:43-46`) — a mutation of the cache outside a forward is
  announced so rank 1 performs the same mutation on its own half
  (`mlx_vlm/server/tp_mode.py:28-31`);
* `_cache_is_empty` (`mlx_vlm/server/tp_mode.py:102-110`) — rank 0 may only tell
  rank 1 to make an empty cache if rank 0's own cache is empty too, or the two
  halves start from different histories and their partial sums are two halves of
  different computations;
* a by-name allowlist of forward kwargs, everything else refused
  (`mlx_vlm/server/tp_mode.py:219-222`), because two different forwards issue
  different numbers of collectives.

None of that is overhead we could delete by copying them; it is the price of the
asymmetry, and the asymmetry is what lets the HTTP server, batching, samplers
and stop criteria stay single-rank. **Verdict: KEEP**, and note that if we ever
wanted their generality, `_share_object` is the pattern — ship the request, not
the forward.

## 5. Launch and lifecycle

| | mlx-lm | ours |
|---|---|---|
| process launch | `mlx.launch`, SSH per host, stdin/stdout multiplexed by the launcher (`mlx/_distributed_utils/launch.py:54-63, 226-256`) | our own worker spawn with an env allowlist (`mlx_vlm/tp/worker.py`) |
| group init | `mx.distributed.init()` inside the program | `init_tp` with an explicit preflight |
| readiness | one `all_sum(1.0)` on the **CPU stream** after load, commented "Synchronize processes to avoid timeout" (`mlx_lm/utils.py:645-646`) | a preflight `all_sum` with a value check *and* a `Deadman` timeout, plus a protocol-version handshake (`mlx_vlm/tp/worker.py:96-155`) |
| shape mismatch | `ValueError` at shard time if a dimension is not divisible by the group size (`mlx/nn/layers/distributed.py:223, 307, 403, 538`) | same via the underlying primitives |
| stall handling | none I found | a watchdog that aborts rank 0 rather than leaving rank 1 holding a shard (`mlx_vlm/server/tp_mode.py:136-192`) |
| load-time preflight | none | fleet headroom / swap / GPU-liveness gate (`mlx_vlm/tp/fleet.py`) |

Two things here are worth taking.

**ADOPT (small): the CPU-stream barrier after load.** `mlx_lm/utils.py:646` runs
its post-load rendezvous on `stream=mx.cpu`. Our preflight `all_sum`
(`mlx_vlm/tp/worker.py:115-116`) does not pin a stream. Metal collectives are
CPU-stream-only anyway, so this is very likely a no-op for us — but it is one
word, it is what the reference does, and "very likely" is the wrong confidence
for a rendezvous that hangs rather than errors when it is wrong. Cost: one
keyword. **Not applied here** — this file changes no code.

**ADOPT (real): the adaptive sync budget.** `TimeBudget`
(`mlx_lm/server.py:234-268`) solves a problem we have not solved: how often the
ranks need to agree on wall-clock progress. It runs a fixed number of iterations,
then every `sync_frequency` loops does one `all_sum` of elapsed time and rescales
the iteration count toward a 0.5 s budget. It is 35 lines and it converts "how
often should I check" from a constant into a measurement. We currently have a
watchdog (abort on stall) but nothing that *tunes* rank-agreement frequency;
this is the piece that would let a TP admission controller make time-based
decisions without adding a collective per step. **Verdict: ADOPT as a design
input for the TP admission work**, not as a copy — our loop shape differs.

## 6. What we have that they do not, kept deliberately

* **"Do not compile a collective."** `mlx_vlm/tp/glm5_next.py:161-170` disables
  the FFN `mx.compile` under TP after a live failure on 2026-09-01: the first
  generation completed and the second stalled a few decode steps in, with rank 1
  waiting for a collective rank 0 never issued. mlx-lm has no equivalent guard
  because it does not compile a block containing a distributed op. If they ever
  do, this is the receipt.
* **The gather gate as a per-lane constant, not a per-model one.**
  `mlx_vlm/tp/glm5_next.py:186-208` records the crossover arithmetic and sets
  65536 for the TP lane against 32768 single-box. mlx-lm has no such knob because
  their DSA path has no equivalent gate.
* **A context vault with a boundary hash** that includes topology, so a cache
  checkpointed under one rank layout cannot be restored under another
  (`mlx_vlm/tp/vault.py`, `mlx_vlm/tp/mirror_vault.py`). mlx-lm has
  `save_prompt_cache` (`mlx_lm/cache_prompt.py:146`) with no distributed
  awareness at all.
* **The fleet gate.** Nothing on their side stops two heavy loads landing on one
  box; we froze a box learning that (`mlx_vlm/tp/fleet.py:1-30`).

## 7. One free cross-check, since I was in both files today

Their indexer bypasses on the **inclusive** predicate:

```python
# mlx_lm/models/deepseek_v32.py:104
if k.shape[2] <= self.index_topk:
    return None
```

which is the same predicate and the same side of it as ours
(`mlx_vlm/models/glm5_next/language.py:1072`). The reference implementation
agrees with the boundary the parity cells in
`mlx_vlm/tests/test_glm5_next_idx_fast.py` now pin. They select over positions
with `argpartition` where we select over pools with `argsort`; that is the
`index_kpool` architecture difference, not a disagreement.

## Verdict summary

| item | verdict | basis |
|---|---|---|
| control on the data collective | **converged** — they do it too, in the server | `mlx_lm/server.py:463-481` |
| SPMD chat via launcher stdin fan-out | **decline** | four silent-failure assumptions vs zero side channels |
| indexer replicated instead of sharded | **keep ours** | 1.36× measured on the gather plateau |
| pipeline parallelism | **park, reason recorded** | we are not memory-short at 2×512 GB |
| variable-width pickled control messages | **park** | ours is 1 collective vs their 2; adopt only if we need generality |
| CPU-stream post-load barrier | **adopt** (one keyword, unapplied here) | `mlx_lm/utils.py:646` |
| `TimeBudget` adaptive sync | **adopt as design input** for TP admission | `mlx_lm/server.py:234-268` |
| explicit cache-identity verbs | **keep ours** | forced by the mirror asymmetry, no analogue needed on their side |
| fleet gate / watchdog / no-compile-a-collective | **keep ours** | each bought with an outage |
