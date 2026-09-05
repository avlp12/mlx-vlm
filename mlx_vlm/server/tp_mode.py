"""TP=2 serving mode: rank 0 is the API server, rank 1 mirrors its forwards.

Enabled by presence of ``MLX_VLM_GLM5_TP_HOSTS`` (family style, e.g.
``10.0.0.1,10.0.0.2``).  Absent or empty means the server behaves exactly as it
does today -- no import of the tp package, no transport, no behavioural change.

WHY A MIRROR RATHER THAN A DISTRIBUTED SERVER.  The whole stack above the model
(HTTP, batching, samplers, caches, stop criteria) is intricate and single-rank
by construction.  Re-entering all of it on rank 1 would mean keeping two copies
of that state in agreement.  Instead rank 1 runs *only* the model forward, and
rank 0 tells it what to run over the one collective the sharded forward already
needs.  Every all_sum inside a sharded layer is then naturally matched, because
both ranks are executing the same forward on the same inputs -- which is exactly
the property the stage-3/4 driver validated (rank0 tokens were byte-identical to
rank1's over 257 tokens).

CONTROL PLANE.  A fixed-width int32 vector carried by all_sum: rank 0 fills it,
rank 1 contributes zeros, the sum hands both the same message.  No side channel,
no second transport to keep alive, and it cannot desynchronise from the data
collectives because it *is* one.

WHAT THE MIRROR OWES.  Rank 1's cache must always be reconstructible from what
rank 0 announced.  Three things can break that, and each has a verb or a
refusal here rather than a silent divergence:

* a forward rank 1 cannot reproduce from token ids (multimodal ``inputs_embeds``)
  -> refused;
* a *mutation* of the cache outside a forward (speculative rollback, vault
  restore) -> announced, so rank 1 performs the same mutation on its own half;
* a cache that appears from nowhere already populated (a merged continuous
  batch, an APC warm cache) -> refused, because ``OP_MAKE_CACHE`` only tells
  rank 1 to make an *empty* one.  See ``_require_reconstructible``.

REPRODUCIBILITY.  TP mode does not reproduce single-box tokens: all_sum adds two
partial sums where one device summed all 4096, and at a one-ULP top-2 gap the
argmax flips (measured: 16.875 vs 16.75, exactly one bf16 ULP at that
magnitude).  Cross-lane token identity is a TP-off property.  The TP-mode
invariant is rank0 == rank1, asserted by the identity test.
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import os
import subprocess
import threading
import time
import weakref
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

from ..tp.worker import (  # noqa: F401  re-exported for the rank-0 side
    ENV_HOSTS, ENV_MAX_TOK, ENV_RANK, ENV_WORKER_PY, ENV_WORKER_SRC,
    ENV_WORKER_MODEL, FLAG_CAPTURE, FLAG_HAS_NAME, HEADER, NAME_WORDS,
    PROTO_VERSION,
    OP_EXIT, OP_FORWARD, OP_MAKE_CACHE, OP_ROLLBACK, OP_VAULT_RESTORE,
    OP_VAULT_STORE, Ctrl, TPDesync, TPUnavailable,
    _ack_recv, _ctrl_recv, _ctrl_send, _max_tok, decode, encode,
    name_to_words, preflight, tp_enabled, tp_hosts, tp_rank, words_to_name,
    worker_loop,
)

# How long a single announced step may take before we conclude the peer is
# gone.  Generous: a 65k prefill chunk on a half shard is seconds, not
# milliseconds, and a false abort is worse than a slow one.
ENV_STEP_TIMEOUT = "MLX_VLM_GLM5_TP_STEP_TIMEOUT_S"


def _step_timeout() -> float:
    try:
        return max(10.0, float(os.environ.get(ENV_STEP_TIMEOUT, "300")))
    except ValueError:
        return 300.0


# --------------------------------------------------------------- cache shape
def _trace_collectives() -> bool:
    return os.environ.get("MLX_VLM_GLM5_TP_TRACE", "") not in ("", "0", "false")


def _reset_forward_counter() -> None:
    try:
        from ..tp.transport import reset_forward_counter

        reset_forward_counter()
    except Exception:
        pass


def _collectives() -> int:
    """all_sums constructed so far, or -1 when the transport is not up."""
    try:
        from ..tp.transport import collective_count

        return collective_count()
    except Exception:
        return -1


def _cache_is_empty(cache) -> bool:
    """Is this prompt cache freshly made -- i.e. exactly what rank 1 would build?

    ``OP_MAKE_CACHE`` says "make an empty cache".  It is only a faithful
    instruction if rank 0's cache is empty too.  Anything else (a continuous
    batch that just merged a second row via ``_extend_cache``, an APC warm
    cache, a vault rung restored without announcing it) means rank 1 would start
    from nothing while rank 0 starts from history -- and their partial sums
    would be halves of different computations.
    """
    for c in cache or ():
        if c is None:
            continue
        sub = getattr(c, "caches", None)
        if sub is not None:                       # CacheList
            if not _cache_is_empty(sub):
                return False
            continue
        off = getattr(c, "offset", None)
        if off is not None:
            if int(off) != 0:
                return False
            continue
        entries = getattr(c, "cache", None)       # ArraysCache (KDA)
        if entries is not None:
            if any(e is not None for e in entries):
                return False
            continue
        # Unknown cache type: treat as non-empty.  Guessing "empty" here would
        # convert a new cache kind into a silent desync.
        return False
    return True


class _Watchdog:
    """Bound the wall time of one announced step, with O(1) cost per step.

    A dead peer leaves the fast fence's GPU kernel spinning on a shared counter;
    nothing on the host can preempt that (see ``tp.transport.Deadman``), so the
    only available recovery is to exit and let a supervisor restart us.  What
    this adds over arming a timer per forward is that it costs a tuple store on
    the hot path instead of a thread launch: at B=8 the server takes ~20 steps a
    second, and a per-step ``threading.Timer`` is a measurable fraction of one.
    """

    def __init__(self, timeout_s: float, poll_s: float = 1.0, on_timeout=None):
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._inflight: Optional[tuple] = None
        self._stop = threading.Event()
        self._on_timeout = on_timeout or self._abort
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="tp-watchdog", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def arm(self, label: str):
        self._inflight = (label, time.monotonic())

    def disarm(self):
        self._inflight = None

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            inflight = self._inflight
            if inflight is None:
                continue
            label, started = inflight
            waited = time.monotonic() - started
            if waited > self.timeout_s and self._inflight is inflight:
                self._on_timeout(label, waited)

    @staticmethod
    def _abort(label: str, waited: float):
        msg = (f"[tp watchdog] rank 0: '{label}' has been in flight for "
               f"{waited:.0f}s (> {_step_timeout():.0f}s). The peer is stalled "
               f"or dead and the GPU-side fence is spinning, which nothing on "
               f"this host can preempt. Exiting 75 so a supervisor can restart "
               f"the server instead of leaving a wedged 94 GiB process.")
        print(msg, flush=True)
        logger.error(msg)
        # Reap the peer FIRST.  os._exit skips atexit and every finally, so
        # nothing else will: observed live on 2026-09-01, an aborted rank 0
        # left rank 1 holding 98 GB, and the next run's fleet preflight
        # (correctly) refused to start at all.  "Supervisor restarts the
        # server" is only a recovery if the box it restarts onto is empty.
        _reap_peer_workers(tp_hosts())
        # And drop OUR OWN wired budget, for the same reason: measured
        # 2026-09-01, a watchdog abort left this box 88 GB deeper in wired
        # memory that no process owned and only a reboot reclaimed.  Best
        # effort by construction -- the main thread is wedged inside a
        # collective, so ``wired_limit.__exit__`` cannot run (it synchronises
        # the stream) and all this thread can do is lower the budget and hope
        # the allocator unwires.  Logged either way so the next abort tells us
        # whether it worked.
        try:
            import mlx.core as mx

            from ..tp.fleet import phys_footprint_gb

            before = phys_footprint_gb()
            mx.set_wired_limit(0)
            logger.error("[tp watchdog] wired budget dropped; footprint "
                         "%.1f -> %.1f GiB", before, phys_footprint_gb())
        except Exception:
            logger.warning("[tp watchdog] could not drop the wired budget",
                           exc_info=True)
        os._exit(75)  # EX_TEMPFAIL


# Keyword arguments rank 0 may consume alone, with the reason each one is safe.
# Everything else is refused BY NAME, because the mirror hands rank 1 nothing but
# token ids: an argument that changes the computation and is not on this list
# makes the two ranks run different forwards, and two different forwards issue
# different numbers of collectives.  That does not surface as a wrong answer --
# it surfaces as the *next* collective pairing a send with the wrong recv, i.e.
# a jaccl error or a hang, a long way from the cause.
_RANK0_ONLY_KWARGS = {
    # Verified per call to equal embed_tokens(inputs); rank 1 rebuilds it.
    "inputs_embeds": "verified equal to embed_tokens(inputs)",
    # Mirrored as FLAG_CAPTURE; the ids belong to the rank-0-only drafter.
    "capture_layer_ids": "mirrored as a flag",
    # glm5_next's LanguageModel.__call__ pops and ignores it.
    "speculative_verify": "popped and ignored by the model",
    # Accepted by LanguageModel.__call__ and never forwarded to the stack
    # (models/glm5_next/language.py builds self.model(...) without it).
    "mask": "accepted and ignored by glm5_next",
    # Read by the prefill driver, never by the model.
    "n_to_process": "not read by the model",
    # Slice the lm_head output only; lm_head is replicated, no collective.
    "num_logits_to_keep": "slices replicated lm_head output",
    "logits_to_keep": "slices replicated lm_head output",
    # Append to a sink; never read back into the residual.
    "return_hidden": "appends to a sink, numerically inert",
    "return_shared_kv": "appends to a sink, numerically inert",
    # Skips the replicated lm_head; no collective either way.
    "skip_logits": "skips the replicated lm_head",
}


class MirroredLanguageModel:
    """Rank-0 wrapper: announce each forward, then run it locally.

    Also the single place where cache-mutating verbs are intercepted, because
    the mirror is only sound if *every* change to rank 0's cache has a matching
    announcement.  Attribute access falls through to the wrapped model, so a
    mutator that is added upstream and not intercepted here would silently
    become a desync -- which is why ``rollback_speculative_cache`` is spelled
    out rather than inherited.
    """

    def __init__(self, lm, *, wire=None, shard_report=None, watchdog=None):
        self._lm = lm
        self._epoch = 0
        self._last_cache_id = None
        # A strong reference to the cache we last announced.  id() alone is not
        # a safe identity key: once a cache is freed its address can be reused
        # by the next one, and an accidental id() match would skip MAKE_CACHE
        # and leave rank 1 decoding into the previous conversation -- a silent
        # desync rather than an error.  Holding the reference makes the address
        # unrecyclable for exactly as long as the comparison depends on it.
        # Cost is bounded: one extra cache stays alive between generations, and
        # during steady decode it is the same object we are already using.
        self._last_cache_obj = None
        self._lock = threading.RLock()
        self._closed = False
        self.shard_report = shard_report
        # The entered ``wired_limit`` context.  Held here, not dropped as a
        # local: a bare local is collected the moment the loader returns, and
        # the generator's ``finally`` then quietly restores the old limit --
        # so the process that thinks it wired the model has not.  Owning it
        # makes both the raise and the release explicit and shutdown-ordered.
        self._wire = wire
        self._watchdog = watchdog
        # Bound method in atexit would pin ``self`` -- and through it the whole
        # 94 GiB shard -- for the life of the interpreter, defeating every
        # unload the server performs.  A weakref hook releases the moment the
        # mirror is dropped, and unregisters cleanly on an ordinary shutdown.
        ref = weakref.ref(self)

        def _atexit_shutdown():
            m = ref()
            if m is not None:
                m.shutdown()

        self._atexit_hook = _atexit_shutdown
        atexit.register(_atexit_shutdown)

    def __getattr__(self, name):
        return getattr(self._lm, name)

    def supports_per_row_speculative_rollback(self, caches) -> bool:
        """No: rank 1 cannot represent a per-row length, so nobody may use one.

        Per-row rollback needs a batched (left-paddable) KV cache on BOTH sides.
        The wire is not the obstacle -- ``OP_ROLLBACK`` already carries the whole
        per-row list (``_accepted_list`` -> ``ids``, decoded back into a list by
        ``tp/worker.py``), so a ragged vector would cross intact.  Rank 1's
        CACHES are: the worker builds ``self.lm.make_cache()`` and nothing else
        (``tp/worker.py`` cache_for / new_cache / _vault_restore), i.e. the
        scalar-offset ``KVCache`` with one offset for the batch.  Handed a ragged
        list it would raise inside ``rollback_speculative_cache`` -- mid-round,
        on the peer, after rank 0 had already rolled back and moved on.

        So the mirror declines for the pair, and the batched loop clamps under
        TP exactly as it did before.  Lifting this is a rank-1 change (build the
        batch caches there), not a protocol one; the two ranks must answer this
        question identically, and the only way to be sure of that today is for
        rank 0 to answer for both.
        """
        del caches
        return False

    # ------------------------------------------------------------ discipline
    def _embeds_are_just_the_ids(self, inputs, embeds) -> bool:
        """Is ``inputs_embeds`` exactly what rank 1 gets by embedding the ids?

        generate_step passes inputs_embeds on every prefill (generate/ar.py, the
        chunk loop and the final step), so refusing it outright would refuse
        every request.  For text-only glm5_next it is literally
        ``embed_tokens(inputs)`` (models/glm5_next/language.py: ``h =
        self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds``),
        so rank 1 reproduces the identical hidden by embedding the broadcast ids
        itself.  For a multimodal prefill it is NOT -- image embeddings are
        spliced in -- and rank 1 could not reconstruct it from ids at all.

        Checked per call, never cached.  Whether a prefill is multimodal is a
        property of the *request*, not of the model: on a VLM checkpoint the
        first request can be text-only and the second can carry an image, and a
        cached "yes" would wave that second one through -- precisely the silent
        desync this class exists to prevent.  The cost is one embedding gather
        per prefill chunk against forty-five MoE layers of work.
        """
        import mlx.core as mx

        try:
            inner = getattr(self._lm, "model", None)
            emb = getattr(inner, "embed_tokens", None)
            if emb is None:
                return False
            ref = emb(inputs)
            return bool(ref.shape == embeds.shape and mx.all(ref == embeds).item())
        except Exception:
            logger.warning("tp: inputs_embeds check failed", exc_info=True)
            return False

    def _require_reconstructible(self, cache) -> None:
        if _cache_is_empty(cache):
            return
        raise TPDesync(
            "TP mode was handed a cache that is already populated but was "
            "never announced. OP_MAKE_CACHE tells rank 1 to build an EMPTY "
            "cache, so rank 1 would start from nothing while rank 0 starts "
            "from history. Known producers: continuous batching admitting a "
            "second request mid-generation (generate/ar.py _extend_cache "
            "merges the batch caches in place and hands back a new list), and "
            "APC warm caches. Serve those single-box, or set "
            "MLX_VLM_MAX_NUM_SEQS=1 to keep the batch composition fixed.")

    # ------------------------------------------------------------ the forward
    def __call__(self, inputs=None, cache=None, **kw):
        if self._closed:
            raise TPUnavailable("TP mirror already released its peer")
        embeds = kw.get("inputs_embeds")
        if inputs is None:
            raise TPUnavailable("TP mode needs token ids to mirror a forward")
        if embeds is not None and not self._embeds_are_just_the_ids(inputs, embeds):
            raise TPUnavailable(
                "TP mode cannot mirror inputs_embeds that are not embed_tokens("
                "inputs) -- multimodal prefill is unsupported in TP mode")
        # A capturing forward is a speculative verify.  Rank 1 is told to
        # capture too (flag, not the id list: the drafter lives on rank 0) so
        # that its KDA layers stash the block inputs its own rollback needs.
        capture = kw.get("capture_layer_ids") is not None
        unknown = [k for k, v in kw.items()
                   if v is not None and k not in _RANK0_ONLY_KWARGS]
        if unknown:
            raise TPDesync(
                f"TP mode was handed forward kwargs it does not mirror: "
                f"{sorted(unknown)}. Rank 1 receives token ids and nothing "
                f"else, so an argument that changes the computation makes the "
                f"two ranks run different forwards -- which shows up as a "
                f"mispaired collective (a jaccl error or a hang) rather than a "
                f"wrong answer. Add it to _RANK0_ONLY_KWARGS with the reason it "
                f"is inert, give it a control verb, or keep this path "
                f"single-box.")
        with self._lock:
            self._ensure_epoch(cache)
            shape = getattr(inputs, "shape", ("?", "?"))
            self._announce(OP_FORWARD, inputs,
                           flags=FLAG_CAPTURE if capture else 0,
                           label=f"announce forward b={shape[0]} s={shape[1]}")
            # The watchdog has to cover the FORWARD ITSELF, not just the
            # announcement.  The announcement is one collective; the forward is
            # 101, and a count mismatch stalls inside them, not before them.
            # Observed 2026-09-01: the first forward after a vault restore hung
            # for 19 minutes with the watchdog disarmed, because it had already
            # been disarmed when the control send returned.
            with self._guard(f"forward b={shape[0]} s={shape[1]}"):
                n0 = _collectives()
                _reset_forward_counter()
                out = self._lm(inputs, cache=cache, **kw)
                _force_same_graph(out)
                if _trace_collectives():
                    logger.info("rank0 forward b=%s s=%s collectives=%d",
                                shape[0], shape[1], _collectives() - n0)
            return out

    @contextlib.contextmanager
    def _guard(self, label: str):
        if self._watchdog is None:
            yield
            return
        self._watchdog.arm(label)
        try:
            yield
        finally:
            self._watchdog.disarm()

    def _ensure_epoch(self, cache) -> None:
        """Announce a fresh cache if this is one rank 1 has not been told about."""
        cid = id(cache)
        if cid == self._last_cache_id and cache is self._last_cache_obj:
            return
        self._require_reconstructible(cache)
        self._epoch += 1
        self._last_cache_id = cid
        self._last_cache_obj = cache
        self._announce(OP_MAKE_CACHE, None, label="make_cache")

    def _release_peer(self, why: str) -> None:
        """Best-effort EXIT so rank 1 is never left blocked in the control wait.

        Rank 1 spends its life inside a blocking all_sum waiting for the next
        verb.  If rank 0 stops issuing verbs -- an early return, a raise, a path
        that forgets to announce -- rank 1 waits forever.  That is the whole TP
        hang family: the ranks never diverge, one simply stops driving, and the
        collective does exactly what a collective is defined to do.

        Reproduced directly: with rank 0 alive-but-idle after N-1 collectives,
        rank 1 blocked on its Nth and had to be killed.  So every failure path
        that could stop the driver emits EXIT on the way out.
        """
        if self._closed:
            return
        self._closed = True
        try:
            _ctrl_send(OP_EXIT, self._epoch, None)
            logger.warning("tp: released peer with EXIT after %s", why)
        except Exception:
            logger.warning("tp: could not release peer after %s; reaping instead",
                           why, exc_info=True)
            try:
                _reap_peer_workers(tp_hosts())
            except Exception:
                logger.warning("tp: peer reap also failed", exc_info=True)

    def _announce(self, op, ids, *, flags=0, arg0=0, name="", label="") -> None:
        if self._watchdog is not None:
            self._watchdog.arm(label or f"op{op}")
        try:
            _ctrl_send(op, self._epoch, ids, flags=flags, arg0=arg0, name=name)
        except BaseException as e:
            # The verb did not land, so rank 1 is either still waiting for this
            # one or will wait for the next that never comes. Release it before
            # propagating -- an orphaned peer holds its whole shard.
            self._release_peer(f"{type(e).__name__} while announcing op{op}")
            raise
        finally:
            if self._watchdog is not None:
                self._watchdog.disarm()

    # ------------------------------------------------------- speculative verbs
    def rollback_speculative_cache(self, caches, gdn_states, accepted,
                                   block_size: int) -> int:
        """Announce the rejection, then roll this rank's own half back.

        The rolled-back state is shard-local on both sides -- the KDA recurrence
        is head-split and each rank replays only its own heads, the DSA latent
        is replicated and each rank trims its own copy -- so nothing but the two
        integers crosses.  ``accepted`` may be an int, a list, or an mx.array
        (batched rounds); it is normalised to a list because that is what the
        target's own implementation reduces it to.
        """
        acc = _accepted_list(accepted)
        with self._lock:
            if caches is not self._last_cache_obj:
                raise TPDesync(
                    "TP rollback on a cache that is not the announced one; "
                    "rank 1 would roll back a different conversation.")
            self._announce(OP_ROLLBACK, acc, arg0=int(block_size),
                           label=f"announce rollback a={acc} bs={block_size}")
            with self._guard(f"rollback a={acc} bs={block_size}"):
                return self._lm.rollback_speculative_cache(
                    caches, gdn_states, accepted, block_size)

    # ------------------------------------------------------------ vault verbs
    def tp_mirror_vault(self, vault):
        """Wrap rank 0's token-shaped vault so its rungs are announced."""
        from ..tp.mirror_vault import MirroredVault

        return MirroredVault(vault, self)

    def announce_vault_store(self, name: str, prefix_len: int) -> None:
        with self._lock:
            n0 = _collectives()
            self._announce(OP_VAULT_STORE, None, arg0=int(prefix_len), name=name,
                           label=f"vault_store {name[:8]}@{prefix_len}")
            if _trace_collectives():
                logger.info("rank0 vault_store %s@%s collectives=%d (1 = the "
                            "announce itself; anything more is not local)",
                            name[:12], prefix_len, _collectives() - n0)

    def announce_vault_restore(self, cache, name: str, prefix_len: int) -> bool:
        """Tell rank 1 to rebuild its half, and believe its answer.

        Returns False when rank 1 does not hold the rung.  The two vaults evict
        independently, so "rank 0 has it" does not imply "rank 1 has it"; the
        ack is the only way to know, and serving a warm rank 0 against a cold
        rank 1 would sum halves of different states into fluent nonsense.
        """
        with self._lock:
            self._epoch += 1
            self._last_cache_id = id(cache)
            self._last_cache_obj = cache
            self._announce(OP_VAULT_RESTORE, None, arg0=int(prefix_len),
                           name=name, label=f"vault_restore {name[:8]}")
            if self._watchdog is not None:
                self._watchdog.arm("vault_restore ack")
            try:
                ok = bool(_ack_recv())
            finally:
                if self._watchdog is not None:
                    self._watchdog.disarm()
            if not ok:
                # Rank 1 missed. Forget the epoch so the next forward announces
                # a fresh MAKE_CACHE and both ranks prefill cold together.
                self._last_cache_id = None
                self._last_cache_obj = None
                logger.info("tp: peer vault miss for %s; cold prefill", name[:12])
            return ok

    # --------------------------------------------------------------- teardown
    def shutdown(self) -> bool:
        """Stop rank 1, release the wired limit, and drop the shard.

        Ordered, because the order is the point.  Announce EXIT first so the
        peer stops waiting in a collective; then leave ``wired_limit``, which
        synchronises the stream and puts the wired budget back; then drop the
        reference to the model so the caller's ``gc.collect()`` /
        ``mx.clear_cache()`` can actually return the memory.  A shutdown that
        skips the last step is what left 183 GiB resident while the next load
        started, and froze the box.

        Returns whether EXIT was actually sent *this call*.  Idempotent, but
        NOT a no-op on repeat: a mirror already marked ``_closed`` (typically
        by ``_release_peer`` after a failed announce) must still run its local
        teardown here -- otherwise the wire context and ``self._lm`` are never
        dropped, and the caller's ``gc.collect()`` frees nothing.  The bool
        return lets the caller log "peer told to exit" only when that is
        actually what happened, instead of unconditionally.
        """
        with self._lock:
            already_closed = self._closed
            self._closed = True
            exit_sent = False
            if not already_closed:
                try:
                    _ctrl_send(OP_EXIT, self._epoch, None)
                    exit_sent = True
                except Exception:  # teardown must never mask the real error
                    logger.warning("tp: EXIT broadcast failed", exc_info=True)
            if self._watchdog is not None:
                self._watchdog.stop()
                self._watchdog = None
            if self._wire is not None:
                try:
                    self._wire.__exit__(None, None, None)
                except Exception:
                    logger.warning("tp: releasing wired limit failed", exc_info=True)
                self._wire = None
            self._last_cache_obj = None
            self._last_cache_id = None
            self._lm = None
            if self._atexit_hook is not None:
                try:
                    atexit.unregister(self._atexit_hook)
                except Exception:
                    pass
                self._atexit_hook = None
            return exit_sent


def _force_same_graph(out) -> None:
    """Evaluate the forward here, because laziness is rank-local and
    collectives are not.

    MLX only executes the ops an evaluated output depends on.  During a chunked
    prefill the caller *discards* the model's return value and evaluates only
    the caches (generate/ar.py: ``model.language_model(...)`` with no
    assignment, then ``mx.eval([c.state for c in prompt_cache])``).  The last
    decoder layer's MLP output feeds nothing else, so its ``all_sum`` has no
    evaluated consumer and rank 0 simply never runs it -- while rank 1, which
    evaluates its logits, runs all 101.

    One collective out of phase does not produce a wrong answer.  It produces a
    *later* recv paired with a send of the wrong size: observed live as
    ``[jaccl] Recv failed with error code -12`` raised from inside the DSA
    indexer, several layers away from the reduce that went missing.  A short
    prompt hides it completely, which is why every TP validation up to this
    point passed: they were all one chunk long, and a one-chunk prefill ends in
    a sampled token, so the logits were evaluated after all.

    Forcing evaluation here makes rank 0's executed graph equal to rank 1's by
    construction, for every caller, rather than for the callers we happened to
    test.
    """
    import mlx.core as mx

    logits = getattr(out, "logits", None)
    if logits is not None:
        mx.eval(logits)
    elif isinstance(out, mx.array):
        mx.eval(out)


def _accepted_list(accepted) -> List[int]:
    if isinstance(accepted, int):
        return [int(accepted)]
    if hasattr(accepted, "reshape") and hasattr(accepted, "tolist"):
        return [int(v) for v in accepted.reshape(-1).tolist()]
    return [int(v) for v in accepted]


def launch_worker(model_path: str, hosts: List[str]) -> subprocess.Popen:
    """Start rank 1 over ssh, the way the pipeline tail is started."""
    py = os.environ.get(ENV_WORKER_PY, "/Users/m3ms/venv_mlx321/bin/python")
    src = os.environ.get(ENV_WORKER_SRC, "/Users/m3ms/src/mlx-vlm-tp2serve")
    host = hosts[1]
    remote_model = os.environ.get(ENV_WORKER_MODEL) or model_path
    # Forward only the variables that are actually SET.  Emitting
    # ``NAME=`` for an unset one puts an empty string in rank 1's environment,
    # and a consumer that parses rather than tests -- glm5_next reads
    # ``int(os.environ.get("MLX_VLM_GLM5_GATHER_MIN_CONTEXT", "32768"))`` --
    # gets int('') and dies at import.  Rank 0 then waits on a peer that never
    # joined until the watchdog fires.  Observed 2026-09-01: every TP run that
    # did not happen to set the gate was broken this way.
    passthrough = [
        "MLX_VLM_GLM5_TP_TRACE", "MLX_VLM_GLM5_TP_TRACE_DEEP",
        "MLX_VLM_GLM5_IDX_FAST", "MLX_VLM_GLM5_SYNC_TRACE",
        "MLX_VLM_GLM5_GATHER_MIN_CONTEXT", "MLX_VLM_GLM5_VAULT",
    ]
    extra = " ".join(f"{k}={os.environ[k]}" for k in passthrough
                     if os.environ.get(k, "") != "")
    budget = os.environ.get("MLX_VLM_GLM5_TP_PEER_VAULT_BUDGET_GB", "")
    if budget:
        extra += f" MLX_VLM_GLM5_VAULT_BUDGET_GB={budget}"
    inner = (
        f"cd {src} && MLX_VLM_GLM5_FUSED_KDA=1 PYTHONPATH={src} "
        f"{ENV_HOSTS}='{','.join(hosts)}' {ENV_RANK}=1 "
        f"MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD={_max_tok()} "
        + (extra + " " if extra else "") +
        f"nohup {py} -u -m mlx_vlm.tp.worker --model {remote_model} "
        f">> ~/tp_worker.log 2>&1 &"
    )
    cmd = ["ssh", "-o", "BatchMode=yes", f"m3ms@{host}", inner]
    logger.info("tp: launching rank1 on %s", host)
    return subprocess.Popen(cmd)


def _refuse_unmirrorable_env() -> None:
    """Startup guards for settings whose effects happen outside a forward."""
    if os.environ.get("KV_BITS"):
        raise TPUnavailable(
            "KV cache quantization is not mirrored: generate_step calls "
            "maybe_quantize_kv_cache on the prompt cache between forwards "
            "(generate/ar.py), and rank 1 is never told. Unset KV_BITS to "
            "serve TP, or serve single-box with KV quantization.")


def maybe_load_tp(model_path: str):
    """Return (model, processor, config) in TP mode, or None to serve single-box.

    Any failure -- transport, worker launch, sharded load -- logs and returns
    None.  Refusing to start the server because a second box is unreachable
    would be a worse failure than serving at single-box speed.
    """
    if not tp_enabled():
        return None
    hosts = tp_hosts()
    worker = None
    try:
        import mlx.core as mx

        from ..context_vault import set_tp_topology
        from ..generate import wired_limit
        from ..tp.fleet import require_quiet_fleet
        from ..tp.load import load_sharded, materialize
        from ..tp.transport import tp_rank as _r, tp_size
        from ..tp.vault import topology_descriptor

        _refuse_unmirrorable_env()
        # Two 94 GiB shards fit; a 94 GiB shard beside a leftover 183 GiB
        # single-box resident does not, and the box freezes rather than swaps.
        logger.info("tp: fleet preflight %s",
                    require_quiet_fleet(hosts, label="tp serving load"))
        _require_live_gpus(hosts)

        worker = launch_worker(model_path, hosts)
        info = preflight(hosts, 0)
        logger.info("tp: group up %s", info)
        model, report = load_sharded(model_path, _r(), tp_size())
        peak = materialize(model)
        logger.info("tp: sharded %s peak %.1f GiB", report, peak)
        # Every vault identity from here on carries which half of which model
        # this process holds, so a TP rung and a single-box rung -- and rank 0's
        # and rank 1's -- can never name the same boundary.
        set_tp_topology(topology_descriptor(report, model_path))
        wire = wired_limit(model, [mx.default_stream(mx.default_device())])
        wire.__enter__()
        watchdog = _Watchdog(_step_timeout()).start()
        inner = model.language_model if hasattr(model, "language_model") else model
        mirrored = MirroredLanguageModel(
            inner, wire=wire, shard_report=report, watchdog=watchdog)
        if hasattr(model, "language_model"):
            model.language_model = mirrored
        else:
            model = mirrored
        processor = _load_processor_like_utils_load(model_path, model)
        return model, processor, model.config if hasattr(model, "config") else None
    except Exception as e:
        logger.error("tp: unavailable (%s); serving single-box", e, exc_info=True)
        _reap_worker(worker, hosts)
        return None


def _require_live_gpus(hosts) -> None:
    """Refuse if either box's Metal device has stopped executing work.

    Memory is not the only way a box goes unusable.  Measured 2026-09-01:
    gesicht reached a state where a 4x4 ``mx.eval`` never returned, with 302 GB
    free and nothing resident.  Every memory check passed, and a load into that
    box would have hung for the full step timeout and then aborted -- costing a
    shard's worth of leaked memory on the way out.  Seconds to check, minutes
    to discover otherwise.
    """
    from ..tp.fleet import gpu_responsive

    if os.environ.get("MLX_VLM_GLM5_TP_SKIP_GPU_CHECK", "") not in ("", "0"):
        return
    if not gpu_responsive():
        raise TPUnavailable(
            "this box's Metal device is not executing work (a 4x4 eval did "
            "not return). Nothing will run here until it is rebooted; serving "
            "single-box would hang the same way.")
    if len(hosts or []) >= 2:
        py = os.environ.get(ENV_WORKER_PY, "/Users/m3ms/venv_mlx321/bin/python")
        if not gpu_responsive(f"m3ms@{hosts[1]}", python=py):
            raise TPUnavailable(
                f"the peer {hosts[1]}'s Metal device is not executing work; "
                f"rank 1 would load its shard and then hang. Reboot it, or "
                f"serve single-box.")


def _reap_worker(worker, hosts) -> None:
    """Make sure a failed bring-up does not leave a shard on the peer.

    ``launch_worker`` starts rank 1 with ``nohup ... &``, so the local ssh
    client exits immediately and terminating it terminates nothing: the remote
    worker is already detached, and by the time an error surfaces here it may be
    most of the way through materialising 85 GiB.  It would then sit blocked in
    a collective forever, holding that memory, and the next load's fleet
    preflight would (correctly) refuse to start.

    SIGTERM only, never SIGKILL: the worker's own ``finally`` drops the shard
    and releases the wired limit, and we want it to run.
    """
    if worker is not None:
        try:
            worker.terminate()
        except Exception:
            pass
    _reap_peer_workers(hosts)


def _reap_peer_workers(hosts) -> None:
    """SIGTERM any rank-1 worker on the peer. Never SIGKILL: the worker's own
    ``finally`` drops the shard and releases the wired limit, and we want it
    to run."""
    if len(hosts or []) < 2:
        return
    try:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             f"m3ms@{hosts[1]}", 'pkill -TERM -f "mlx_vlm.tp.worker"'],
            capture_output=True, timeout=30)
        logger.info("tp: sent SIGTERM to any rank-1 worker on %s", hosts[1])
    except Exception:
        logger.warning(
            "tp: could not reap the rank-1 worker on %s -- check it by hand "
            "before the next load (python -m mlx_vlm.tp.fleet)", hosts[1],
            exc_info=True)


def _load_processor_like_utils_load(model_path: str, model):
    """Build the processor the way ``utils.load()`` does, or the best available.

    Returning a bare ``TokenizerWrapper`` is not equivalent: the server calls
    the processor, and a wrapper is not callable.  So the full ``AutoProcessor``
    is the target.

    It is not always reachable.  ``Glm5NextProcessor`` pulls in a *video*
    sub-processor that transformers gates behind torch + torchvision, and a
    text-serving venv reasonably has neither -- in which case
    ``AutoProcessor.from_pretrained`` raises ImportError for a component this
    model path never touches.  Falling back to the tokenizer keeps text serving
    working (which is what the single-box path did before the upgrade) and says
    exactly what was lost, rather than refusing TP over an optional backend.
    """
    from ..utils import get_model_path, load_image_processor, load_processor, \
        load_tokenizer

    mp = get_model_path(model_path)
    eos = getattr(model, "config", None)
    eos = getattr(eos, "eos_token_id", None)
    try:
        processor = load_processor(mp, True, eos_token_ids=eos)
    except (ImportError, OSError, ValueError) as e:
        logger.warning(
            "tp: full processor unavailable (%s); falling back to the "
            "tokenizer. Text serving is unaffected; image/video inputs are "
            "not supported in this environment (and are refused by the mirror "
            "in TP mode anyway).", str(e).strip().splitlines()[0][:160])
        return load_tokenizer(mp)
    image_processor = load_image_processor(mp)
    if image_processor is not None:
        processor.image_processor = image_processor
    return processor


def shutdown_tp(model) -> bool:
    """Shut the mirror down if ``model`` has one. Safe on any model.

    Called from the server's unload path, so that dropping a model group also
    stops the peer instead of leaving it blocked in a collective with a shard
    resident.

    Returns whether EXIT was actually sent to the peer this call -- NOT
    merely whether ``model`` was TP-mirrored.  ``shutdown()`` always runs its
    local teardown (idempotently), but a mirror already closed (e.g. by
    ``_release_peer`` after a failed announce) sends no further EXIT, and the
    caller's "peer told to exit" log line should not claim otherwise.
    """
    lm = getattr(model, "language_model", model)
    if not isinstance(lm, MirroredLanguageModel):
        return False
    return lm.shutdown()
