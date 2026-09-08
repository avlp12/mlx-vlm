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
    OP_VAULT_STORE, Ctrl, TPDesync, TPPeerGone, TPUnavailable,
    _ack_recv, _ctrl_recv, _ctrl_send, _max_tok, clear_peer_gone, decode,
    encode, mark_peer_gone, name_to_words, peer_gone, preflight, tp_enabled,
    tp_hosts, tp_rank, words_to_name, worker_loop,
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

    WHY IT DID NOT FIRE ON 2026-09-08.  The timeout arm only looks at
    ``_inflight``, and ``_inflight`` is only set by ``_announce``/``_guard`` --
    i.e. by forwards that go THROUGH the mirror.  The MTP verify did not: it
    reached past the wrapper (``speculative/mtp.py`` -> ``lm.speculative_verify_
    hidden`` -> the raw model) and spun in an unarmed, unannounced collective, so
    the loop below saw ``None`` on every one of its 2400 polls and had nothing to
    time.  The thread was running and healthy the whole time; there was simply
    nothing armed.

    Two changes follow from that.  The forward path is fixed so nothing bypasses
    the mirror (that is the real repair), and the loop below also polls a PEER
    verdict, which does not depend on anything being armed: a peer that
    announced EXITING is a fact about the pair, not about a step.  When nothing
    is in flight the verdict is only RECORDED -- the next control verb then
    raises ``TPPeerGone`` and the server unwinds normally -- because a peer that
    exits while the server is idle (an orderly shutdown) must not abort us.
    """

    def __init__(self, timeout_s: float, poll_s: float = 1.0, on_timeout=None,
                 peer_probe=None, on_peer_gone=None):
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._inflight: Optional[tuple] = None
        self._stop = threading.Event()
        self._on_timeout = on_timeout or self._abort
        # Injectable so the fire path is testable without a group.  Default is
        # the module gate, which reads the heartbeat's already-received state
        # and does no I/O.
        self._peer_probe = peer_probe or peer_gone
        self._on_peer_gone = on_peer_gone or mark_peer_gone
        self.peer_gone_reason: Optional[str] = None
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
            self.poll_once()

    def poll_once(self) -> None:
        """One watchdog tick.  Split out so a test can drive it directly."""
        inflight = self._inflight
        gone = None
        try:
            gone = self._peer_probe()
        except Exception:  # pragma: no cover - a probe must never kill the loop
            logger.debug("[tp watchdog] peer probe failed", exc_info=True)
        if gone is not None and self.peer_gone_reason is None:
            self.peer_gone_reason = gone
            self._on_peer_gone(gone)
        if inflight is None:
            return
        label, started = inflight
        waited = time.monotonic() - started
        if gone is not None and self._inflight is inflight:
            # A step IS in flight and the peer is gone: the thread running it is
            # already inside the collective and cannot be unwound from here.
            # Exit, the same as a timeout -- but now, instead of after 300 s.
            self._on_timeout(f"{label} (peer gone: {gone})", waited)
            return
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
    # Prefill-leg memory knob (speculative.utils.prefill_capture_kwargs): asks
    # the model NOT to retain the sequence-shaped KDA rollback stash it would
    # otherwise build alongside a per-layer capture.  It changes what is kept,
    # not what is computed or reduced -- rank 1 simply keeps (or drops) its own
    # stash; the collective sequence is identical.  Verified 2026-09-05 when the
    # L3 tp_spec arm was refused on exactly this key.
    "capture_gdn_states": "prefill stash retention only; no collective",
    # Slice the lm_head output only; lm_head is replicated, no collective.
    "num_logits_to_keep": "slices replicated lm_head output",
    "logits_to_keep": "slices replicated lm_head output",
    # Append to a sink; never read back into the residual.
    "return_hidden": "appends to a sink, numerically inert",
    "return_shared_kv": "appends to a sink, numerically inert",
    # Skips the replicated lm_head; no collective either way.
    "skip_logits": "skips the replicated lm_head",
}


# Attributes that run a sharded forward and are NOT implemented on the mirror.
# Handing one of these out is how a rank-0-only forward gets issued: the caller
# reaches past the wrapper, 101 all_sums happen with nothing announced, and the
# ranks are out of phase from then on.  Refused by name in ``__getattr__``.
#
# ``speculative_verify_hidden`` / ``speculative_verify_logits`` are deliberately
# ABSENT: they are implemented on the mirror (they announce, then delegate), so
# ``__getattr__`` is never reached for them.  Anything added upstream that runs a
# forward belongs here until it is given the same treatment.
# It is EMPTY today, and that is the correct state: every hook that exists in
# this tree is either mirrored or inert, and ``_refuse_unmirrored_speculative_
# hooks`` refuses anything else AT LOAD.  This exists as the second rail, for a
# hook attached to an instance after load (where the load-time scan of the class
# cannot see it) -- add the name here and it becomes a refusal instead of a
# forty-minute hang.
_UNMIRRORABLE_FORWARD_HOOKS = frozenset()

# Attributes that are safe to hand out because they issue NO collective.  Kept
# as a list so the load-time check below can say which hooks it accepted and
# why, instead of silently allowing everything it does not recognise.
_INERT_SPECULATIVE_HOOKS = {
    # final norm + the replicated lm_head (models/glm5_next/language.py:3604);
    # lm_head is replicated in the shard plan, so this reduces nothing.
    "speculative_logits_from_hidden": "replicated norm + lm_head, no collective",
    "speculative_argmax_from_hidden": "argmax over the replicated head",
    # identity for glm5_next; a pure elementwise reshape elsewhere.
    "speculative_draft_hidden": "reshapes the drafter's hidden, no collective",
}

# The hooks the mirror announces.  Named so the load-time refusal can tell a
# hook it mirrors from one it has never heard of.
_MIRRORED_SPECULATIVE_HOOKS = ("speculative_verify_hidden",
                               "speculative_verify_logits")


class _ReadOnlyStack:
    """``mirror.model``: read the inner stack, do not run it.

    Reads are what the drafter and the MTP helpers need (embed_tokens, layers,
    norm, layer_type).  A CALL is a full unannounced forward, which is the shape
    of the 2026-09-08 hang, so it raises with the reason instead.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    @property
    def __class__(self):
        """Answer isinstance() as the wrapped stack.

        A proxy that fails ``isinstance(x, nn.Module)`` does not raise -- it
        makes some caller take its OTHER branch, quietly, and a quiet branch
        change on the serving path is exactly the class of bug this whole file
        exists to prevent.  Special methods (including ``__call__`` below) are
        still looked up on the real type, so the refusal is unaffected.
        """
        return type(object.__getattribute__(self, "_inner"))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __call__(self, *args, **kwargs):
        raise TPDesync(
            "TP mode refuses a direct call to language_model.model(...): it runs "
            "the sharded stack on rank 0 with no OP_FORWARD announced, so rank 1 "
            "never runs the matching reduces and the two ranks are out of phase "
            "from the next collective on (2026-09-08: rank 1 refused with "
            "TPDesync and rank 0 spun at 200% CPU for 40 minutes). Call the "
            "language model itself -- the mirror announces that path -- or add a "
            "verb for this one.")


def _force_same_graph_hidden(out) -> None:
    """The ``_force_same_graph`` of the verify path.

    Same argument, different output shape: the verify hooks return a tuple whose
    first element is the pre-final-norm hidden, and with ``skip_logits=True``
    there are no logits to evaluate at all.  Evaluating the hidden forces every
    reduce in the stack (it is the last decoder layer's output), which is what
    makes rank 0's executed graph equal to rank 1's -- rank 1 evaluates its
    logits unconditionally (tp/worker.py, OP_FORWARD).
    """
    import mlx.core as mx

    first = out[0] if isinstance(out, tuple) and out else out
    if isinstance(first, mx.array):
        mx.eval(first)


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
        self._model_proxy = None
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
        """Fall through to the wrapped model -- except for the forwards.

        This method is where the 2026-09-08 MTP hang came from.  Attribute
        fall-through is what makes the wrapper transparent, and it is also what
        makes it *leaky*: ``speculative/mtp.py`` asks the language model for
        ``speculative_verify_hidden`` (mtp.py:92) and gets a bound method of the
        RAW model, which then runs the whole sharded stack -- 101 all_sums --
        with no OP_FORWARD announced.  Rank 1, sitting in its control wait,
        paired its next control reduce with one of those, read a control vector
        that was never one, and refused with ``TPDesync`` naming the last shape
        it had agreed on (batch 1, seqlen 15: the 15-token prompt of that run).
        DFlash2 survives the same wiring only because it verifies through
        ``lm(...)`` (speculative/dflash.py:1206) and therefore through
        ``__call__`` below.

        So the hooks that run a forward are implemented on this class (they
        announce, then delegate), and any *other* attribute that is known to run
        one is refused BY NAME here rather than handed out.  The rule matches the
        one ``_RANK0_ONLY_KWARGS`` already applies to keyword arguments: an
        unmirrored forward is not a wrong answer, it is a hang a long way from
        its cause, so it has to be impossible to reach by accident.
        """
        if name in _UNMIRRORABLE_FORWARD_HOOKS:
            raise TPDesync(
                f"TP mode will not hand out {name!r}: it runs a sharded forward "
                f"on rank 0 alone, and rank 1 is only told about forwards that go "
                f"through the mirror. Implement it on MirroredLanguageModel (see "
                f"speculative_verify_hidden) so it announces OP_FORWARD first, or "
                f"serve this drafter single-box.")
        return getattr(self._lm, name)

    @property
    def model(self):
        """The inner stack, readable but not callable.

        ``speculative/mtp.py`` falls back to ``lm.model(...)`` when a model has
        no verify hook (mtp.py:119,129), and ``mtp.py:62`` reads
        ``lm.model.layers`` on every round.  The reads are harmless -- layers,
        embed_tokens, norm -- and the drafter's ``reset()`` needs them
        (speculative/drafters/glm5_next_mtp: ``target_model.language_model.model
        .embed_tokens``).  The CALL is the unannounced forward.  So the proxy
        passes reads through and refuses ``__call__`` by name.
        """
        if self._lm is None:
            raise TPUnavailable("TP mirror has been shut down")
        inner = self._lm.model
        proxy = self._model_proxy
        if proxy is None or proxy._inner is not inner:
            proxy = self._model_proxy = _ReadOnlyStack(inner)
        return proxy

    # ------------------------------------------------- speculative verify
    def speculative_verify_hidden(self, inputs, cache):
        """Announce the verify block, then run it.  MTP's hot path.

        Rank 1 answers OP_FORWARD by running ``self.lm(ids, cache, capture_
        layer_ids=[])`` (tp/worker.py handle/OP_FORWARD).  Rank 0 runs
        ``Glm5NextExactSpeculativeVerifier`` -- ``language_model.model(inputs,
        cache=cache, gdn_sink=[], hidden_sink=[])`` plus, on the logits variant,
        the replicated lm_head.  Those two run the SAME 101 reduces in the same
        order: the sinks are appended to and never read back into the residual
        (the argument ``_RANK0_ONLY_KWARGS`` already makes for
        ``capture_layer_ids`` and ``return_hidden``), and ``skip_logits`` only
        skips a replicated head.  FLAG_CAPTURE is always set, because a verify
        can be rejected and OP_ROLLBACK on rank 1 refuses without a captured
        round.
        """
        return self._mirrored_verify(
            "speculative_verify_hidden", inputs, cache,
            lambda: self._lm.speculative_verify_hidden(inputs, cache))

    def speculative_verify_logits(self, inputs, cache, sampler):
        return self._mirrored_verify(
            "speculative_verify_logits", inputs, cache,
            lambda: self._lm.speculative_verify_logits(inputs, cache, sampler))

    def _mirrored_verify(self, what: str, inputs, cache, run):
        if self._closed:
            raise TPUnavailable("TP mirror already released its peer")
        if inputs is None:
            raise TPUnavailable("TP mode needs token ids to mirror a forward")
        with self._lock:
            self._ensure_epoch(cache)
            shape = getattr(inputs, "shape", ("?", "?"))
            self._announce(OP_FORWARD, inputs, flags=FLAG_CAPTURE,
                           label=f"announce {what} b={shape[0]} s={shape[1]}")
            with self._guard(f"{what} b={shape[0]} s={shape[1]}"):
                n0 = _collectives()
                _reset_forward_counter()
                out = run()
                _force_same_graph_hidden(out)
                if _trace_collectives():
                    logger.info("rank0 %s b=%s s=%s collectives=%d",
                                what, shape[0], shape[1], _collectives() - n0)
            return out

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
            # Announce EXITING on the side-channel too, and take our beacon
            # down.  The datagram cannot block, and it is what stops rank 1
            # waiting out its own dead_s bound after an orderly shutdown.
            try:
                from ..tp import heartbeat as _hb

                _hb.shutdown_beacon(announce_exit=True)
            except Exception:
                logger.debug("tp: heartbeat shutdown failed", exc_info=True)
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
        # RANK 0's HALF OF THE SIDE-CHANNEL.  Until now ``init_beacon`` was
        # called in exactly one place, ``tp/worker.py`` (rank 1), so the
        # heartbeat was one-directional: rank 1 could tell that rank 0 had
        # stopped driving, and rank 0 could tell nothing at all.  That is why the
        # 2026-09-08 hang was unrecoverable -- rank 1 announced EXITING as it
        # released its shard (worker_loop's finally) and the announcement had no
        # listener.  Starting one here costs two daemon threads and 44 bytes at
        # 4 Hz, and it is what makes ``peer_gone()`` (and therefore TPPeerGone)
        # able to answer.  Best effort: a side-channel that cannot start must
        # never stop the serve, exactly as it does not on rank 1.
        clear_peer_gone()
        _start_rank0_beacon(hosts)
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
        _refuse_unmirrored_speculative_hooks(inner)
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


def _start_rank0_beacon(hosts) -> bool:
    """Rank 0's half of the side-channel.  Best effort; returns whether it is up.

    Never raises: a heartbeat that cannot start must not stop a serve, which is
    the same rule ``tp/worker.py`` applies on rank 1.  What is lost when it does
    not start is only detection speed -- the step timeout still bounds an armed
    step -- so the failure is logged and serving continues.
    """
    # One switch, rank 0 only.  ``MLX_VLM_TP_HB`` disables the beacon on both
    # ranks and is not forwarded to the worker by ``launch_worker``, so it cannot
    # answer "is rank 0's beacon implicated?" on its own.  This can: set
    # MLX_VLM_TP_HB_RANK0=0 and everything else about the arm is unchanged,
    # including rank 1's beacon, which has been running since long before this.
    if os.environ.get("MLX_VLM_TP_HB_RANK0", "1") in ("0", "false", "no", "off"):
        logger.info("tp: rank-0 heartbeat beacon disabled by "
                    "MLX_VLM_TP_HB_RANK0; a peer that exits will only be "
                    "noticed by the step timeout")
        return False
    try:
        from ..tp import heartbeat as _hb

        b = _hb.init_beacon(0, len(hosts or []) or 2)
        if b is None:
            return False
        b.note(_hb.STATE_IDLE)
        return True
    except Exception:
        logger.warning("tp: rank-0 heartbeat beacon could not start; a peer that "
                       "exits will only be noticed by the step timeout",
                       exc_info=True)
        return False


def _refuse_unmirrored_speculative_hooks(inner) -> None:
    """Refuse AT LOAD any speculative hook the mirror does not know about.

    The MTP hang had no error to read because the bypass was silent: a hook the
    mirror had never heard of was handed out by ``__getattr__`` and ran a
    sharded forward on rank 0 alone.  The hooks that exist today are each either
    mirrored or provably collective-free, and both lists are spelled out above.
    A hook that is on neither list is, by construction, one nobody has checked --
    so it is a named refusal here (seconds, before a four-minute load) rather
    than a hang forty minutes in.
    """
    unknown = sorted(
        n for n in dir(type(inner))
        if n.startswith("speculative_")
        and n not in _INERT_SPECULATIVE_HOOKS
        and n not in _MIRRORED_SPECULATIVE_HOOKS
        and callable(getattr(inner, n, None))
    )
    if unknown:
        raise TPUnavailable(
            f"{type(inner).__name__} exposes speculative hooks TP mode does not "
            f"know how to mirror: {unknown}. A hook that runs the sharded stack "
            f"outside MirroredLanguageModel issues collectives rank 1 is never "
            f"told about, which is a hang rather than a wrong answer "
            f"(2026-09-08, MTP under TP=2). Either implement it on the mirror "
            f"(see speculative_verify_hidden), or add it to "
            f"_INERT_SPECULATIVE_HOOKS with the reason it issues no collective, "
            f"or serve this model single-box.")


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
