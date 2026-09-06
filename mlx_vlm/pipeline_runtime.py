"""Head-side runtime for two-box layer-pipelined prefill.

This is the half that lives inside a normal ``generate()`` call. The other half
is ``python -m mlx_vlm.pipeline_prefill --role tail`` running on the peer box.

Enable by setting ``MLX_VLM_PIPELINE_HOSTS`` (the worker's control endpoint);
everything else has a default::

    MLX_VLM_PIPELINE_HOSTS=10.0.0.2:39210          # required, enables the feature
    MLX_VLM_PIPELINE_RING=10.0.0.1:39400,10.0.0.2:39401   # ring backend (default transport)
    MLX_VLM_PIPELINE_SPLIT=23                      # int, or "auto" for the micro-sweep
    MLX_VLM_PIPELINE_MIN_TOKENS=16384              # below this, stay single-box
    MLX_VLM_PIPELINE_CALIB=~/.cache/mlx_vlm/pipeline_splits.json
    MLX_VLM_PIPELINE_MODEL_SHA256=<verified manifest SHA256>  # required
    MLX_VLM_PIPELINE_SOURCE_REVISION=<source commit SHA1>     # required
    MLX_VLM_PIPELINE_IO_TIMEOUT=120               # socket/queue timeout seconds

The model/source identities are caller attestations: the launcher must verify
the actual local files on each host before supplying these pins. Schema 1 only
supports cold, unpadded B=1 text with bf16 activations and ordinary KDA/DSA caches.
Socket I/O and queue waits are bounded; an in-flight Metal kernel or ring call
cannot be preempted by this Python protocol. Use socket transport for supervised
cooperative runs. A failed request's partially filled head cache is unusable.

Prefill only. When prefill ends, stage B ships its caches back and decode
continues on this box with all layers resident -- single-stream decode gains
nothing from layer pipelining because both stage latencies serialize inside
every token step.

Receipts (twin M3 Ultra 512GB, tbnet, GLM-5.3-Flash q4, split 23, chunk 2048,
repro_mlxvlm_dflash2.py --skip-spec, 32768-token prompt)::

    metric              single box     two box      ratio
    prefill tok/s          348.3        555.6       1.60x
    TTFT                   94.13 s      59.00 s     -37%
    decode tok/s            24.89        25.19      1.01x  (unaffected)
    peak memory           200.2 GB     197.9 GB

Identity: on the same-tree loopback arrangement the pipelined path reproduces
the single-box text_preview byte-for-byte with generation_tps 26.716 vs 26.722
-- the cache handoff restores stage B's state exactly, so decode is untouched.

The 1.60x is lower than the 1.85x the internal stage accounting suggests
because splitting the stack forces the boundary tensor to be materialized every
chunk instead of letting MLX keep the whole 45-layer graph in flight; measured
at ~7% of aggregate stage time on real text (nil on random tokens). 59.00 s is
~99% of this split's structural floor (max stage 55.2 s + fill 3.2 s).

Transport: ring and the raw Python socket are indistinguishable under real
prefill load -- 31.9 vs 30.6/33.4 ms per 64 MiB boundary, tail_wait 2.81 vs
2.67/2.93 s, 582.0 vs 582.1/579.0 tok/s. The 28-41 ms in-flight cost is
memory-bandwidth contention with the prefill itself, not a transport property
(both fall to ~12 ms when the box is only running matmuls). Ring's one clear
win is deserialization: 0.01 ms vs 5.3 ms per chunk, since it hands back an
mx.array instead of a 64 MiB host copy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx

from .pipeline_prefill import (
    EOF_IDX,
    HDR,
    MAGIC,
    PrefillEnvelope,
    _abort_socket,
    _check_peer_identity,
    _hex,
    _queue_put,
    _recv_json,
    _send_json,
    handoff_recv,
    expected_state_meta,
    install_state,
    ring_group,
    ring_send,
    setup_ring_env,
    to_wire,
    token_bytes,
)

_DISABLED = object()
# Kept as a module attribute because tests patch it, but it no longer LATCHES a
# decision: see ``maybe_open_pipeline``.
_CTX = None


def pipeline_language_model(model):
    """The half of ``model`` that owns the pipeline hooks, or ``None``.

    ``generate_step`` is handed the VLM wrapper, so every hook here used to be
    spelled ``model.language_model.<hook>``.  The SERVER does not: it builds its
    ``BatchGenerator`` on ``model.language_model`` directly
    (``server/generation.py:2454``), so the object that reaches
    ``PromptProcessingBatch`` -- and therefore the batch-path pipeline call site --
    IS the language model.  Both spellings have to resolve to the same object or
    the served path would bypass itself forever with ``no_pipeline_hook``.
    """
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "pipeline_prefill_head"):
        return lm
    if hasattr(model, "pipeline_prefill_head"):
        return model
    return None


# ------------------------------------------------------------------ settings

# ``docs/POLICY_long_prompt_pp_routing_2026-09-05.md`` rule 1: a first prefill
# of >= 16k tokens routes to the two-box path.  The pairs behind that rule are
# 32,779 tok (339.1 -> 743.1 input tok/s, 2.19x) and 131,072 tok (~295 -> 756.9,
# 2.57x); 16k-32k has no PP measurement yet (the doc queues it as L5-a), so 16k
# is the policy FLOOR and not a measured crossover.  The old 4096 default sat
# below every number in that table and would have routed prompts the policy
# says nothing about.
DEFAULT_MIN_TOKENS = 16384


class PipelineSettings:
    def __init__(
        self,
        peer,
        ring,
        split,
        min_tokens,
        calib_path,
        transport,
        model_sha256=None,
        source_revision=None,
        io_timeout=120.0,
    ):
        self.peer = peer
        self.ring = ring
        self.split = split
        self.min_tokens = min_tokens
        self.calib_path = calib_path
        self.transport = transport
        self.model_sha256 = model_sha256
        self.source_revision = source_revision
        self.io_timeout = float(io_timeout)
        if not math.isfinite(self.io_timeout) or self.io_timeout <= 0:
            raise ValueError("pipeline io_timeout must be positive and finite")

    @classmethod
    def from_env(cls):
        hosts = os.environ.get("MLX_VLM_PIPELINE_HOSTS", "").strip()
        if not hosts:
            return None
        host, _, port = hosts.partition(":")
        ring = os.environ.get("MLX_VLM_PIPELINE_RING", "").strip() or None
        split = os.environ.get("MLX_VLM_PIPELINE_SPLIT", "auto").strip()
        calib = os.environ.get(
            "MLX_VLM_PIPELINE_CALIB",
            str(Path.home() / ".cache/mlx_vlm/pipeline_splits.json"),
        )
        return cls(
            peer=(host, int(port or 39210)),
            ring=ring,
            split=split,
            min_tokens=int(
                os.environ.get("MLX_VLM_PIPELINE_MIN_TOKENS", str(DEFAULT_MIN_TOKENS))
            ),
            calib_path=calib,
            transport="ring" if ring else "socket",
            model_sha256=os.environ.get("MLX_VLM_PIPELINE_MODEL_SHA256"),
            source_revision=os.environ.get("MLX_VLM_PIPELINE_SOURCE_REVISION"),
            io_timeout=float(os.environ.get("MLX_VLM_PIPELINE_IO_TIMEOUT", "120")),
        )


# --------------------------------------------------------- split calibration


def _ctx_bucket(tokens: int) -> int:
    """Round to a power-of-two bucket: the split optimum moves with context
    (DSA cost grows with the cache, KDA is flat), so a cached split is only
    valid near the context it was measured at."""
    b = 4096
    while b < tokens:
        b *= 2
    return b


def _prime_dsa_caches(model, caches, prefix_tokens: int):
    """Fill the DSA latent/indexer caches with `prefix_tokens` of random state.

    Only DSA needs priming: its per-chunk cost depends on how deep the cache is
    (mask width, and whether the 32768 gather gate has tripped), while KDA's
    recurrent state is a fixed-size tensor whose cost is context-independent, so
    an empty KDA cache is already cost-faithful.
    """
    if prefix_tokens <= 0:
        return
    lm = pipeline_language_model(model)
    cfg = lm.args
    layers = lm.model.layers
    idx_dim = 2 * cfg.index_head_dim + 1
    for i, layer in enumerate(layers):
        if layer is None or layer.is_linear:
            continue
        c = caches[i]
        lat = mx.random.normal((1, 1, prefix_tokens, cfg.kv_lora_rank)).astype(
            mx.bfloat16
        )
        c[0].state = (lat, lat)
        packed = mx.random.normal((1, 1, prefix_tokens, idx_dim)).astype(mx.bfloat16)
        c[1].state = (packed, mx.zeros((1, 1, prefix_tokens, 0), dtype=mx.bfloat16))


def calibrate_split(
    model,
    *,
    ctx: int,
    chunk: int = 2048,
    candidates: Optional[List[int]] = None,
    reps: int = 2,
    peer_speed: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Pick the layer split by timing cumulative prefixes on this box.

    One forward pass per rep with an ``mx.eval`` barrier at each candidate
    boundary gives every candidate's prefix cost at once, so the whole sweep is
    a couple of chunk-forwards rather than one per candidate. Both halves are
    timed here (this box holds all the layers), which also removes box-to-box
    speed differences from the comparison -- ``peer_speed`` scales the tail if
    the boxes are not twins.
    """
    lm = pipeline_language_model(model).model
    n = len(lm.layers)
    if candidates is None:
        mid = n // 2
        candidates = [k for k in range(mid - 3, mid + 4) if 1 <= k < n]
    bounds = sorted(set(candidates))

    per_rep = []
    for _ in range(reps):
        caches = model.make_cache()
        _prime_dsa_caches(model, caches, max(0, ctx - chunk))
        ids = mx.random.randint(1000, 150000, (1, chunk)).astype(mx.int32)
        mx.eval(ids, [c.state for c in caches])
        cum, lo, h, t0 = [], 0, None, time.perf_counter()
        for k in bounds + [n]:
            h = lm.pipeline_forward(h, caches, lo, k, inputs=ids)
            mx.eval(h)
            cum.append(time.perf_counter() - t0)
            lo = k
        per_rep.append(cum)
        del caches
        mx.clear_cache()

    # median across reps, per boundary
    merged = [
        sorted(c[j] for c in per_rep)[len(per_rep) // 2] for j in range(len(bounds) + 1)
    ]
    total = merged[-1]
    scored = []
    for j, k in enumerate(bounds):
        head, tail = merged[j], (total - merged[j]) * peer_speed
        scored.append(
            {"split": k, "head_s": head, "tail_s": tail, "max_s": max(head, tail)}
        )
    best = min(scored, key=lambda d: d["max_s"])
    out = {
        "split": best["split"],
        "ctx": ctx,
        "chunk": chunk,
        "total_s": total,
        "candidates": scored,
        "reps": reps,
    }
    if verbose:
        print("[pipeline] calibration " + json.dumps(out), flush=True)
    return out


def resolve_split(model, settings: PipelineSettings, tokens: int, verbose=False) -> int:
    if settings.split != "auto":
        return int(settings.split)
    n = pipeline_language_model(model).pipeline_num_layers
    bucket = _ctx_bucket(tokens)
    key = f"{getattr(model, 'model_path', '?')}|{n}|{bucket}"
    path = Path(os.path.expanduser(settings.calib_path))
    store = {}
    if path.exists():
        try:
            store = json.loads(path.read_text())
        except (ValueError, OSError):
            store = {}
    if key in store:
        return int(store[key]["split"])
    res = calibrate_split(model, ctx=bucket, verbose=verbose)
    store[key] = res
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, indent=1))
    except OSError:
        pass
    return int(res["split"])


# --------------------------------------------------------------- head client


# The keys a ping ack may carry.  The check stays as strict as the equality it
# replaces -- an unknown key is still a protocol error -- but ``degraded`` and
# ``rail`` are now part of the contract (pipeline_prefill.tail_session_factory).
_PING_ACK_KEYS = frozenset({"cmd", "ok", "degraded", "rail"})


class PipelineHead:
    def __init__(self, settings: PipelineSettings, split: int, n_layers: int):
        self.settings = settings
        self.split = split
        self.n_layers = n_layers
        self.transport = settings.transport
        self.sock = None
        self.stats = {"chunks": [], "wire_send_s": 0.0, "head_gpu_s": 0.0}
        self._q = None
        self._th = None
        self._err = []
        self._active = False
        self._stop = threading.Event()
        self.envelope = None
        self._model = None
        # The tail's own verdict on its rail, as of the last ``ping`` that
        # asked for it.  ``False``/``None`` on a tail that predates A7 and on a
        # connection that has not been pinged, which is the same thing an
        # operator would assume: no evidence of degradation is not evidence of
        # degradation, and the request goes ahead.
        self.peer_degraded = False
        self.peer_rail = None

    # -- connection ---------------------------------------------------------
    def connect(self, timeout=60.0):
        identity = dict(
            schema=1,
            model_sha256=_hex(self.settings.model_sha256, 64, "model_sha256"),
            source_revision=_hex(self.settings.source_revision, 40, "source_revision"),
            split=self.split,
            n_layers=self.n_layers,
        )
        if self.transport == "ring":
            setup_ring_env(self.settings.ring, 0)
        deadline = time.monotonic() + timeout
        while True:
            try:
                s = socket.create_connection(
                    self.settings.peer,
                    timeout=min(
                        self.settings.io_timeout, max(0.01, deadline - time.monotonic())
                    ),
                )
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        s.settimeout(self.settings.io_timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        try:
            _send_json(s, {"cmd": "hello", "transport": self.transport, **identity})
            ack = _recv_json(s)
            if not ack.get("ok"):
                raise ValueError("pipeline hello refused")
            _check_peer_identity(
                ack,
                self.settings.model_sha256,
                self.settings.source_revision,
                self.split,
                self.n_layers,
            )
            if self.transport == "ring":
                ring_group()
        except BaseException:
            self.abort()
            raise
        return self

    def ping(self):
        """Is this connection still a connection, and is its rail worth using?

        Cheap enough to run before every reuse, which is the only way a pooled
        socket's death can be discovered while falling back is still free.

        ``rail: true`` asks the A7 tail to ride its own rail verdict on the
        ack.  It is a REQUEST rather than a standing arrangement because the
        tail answers a bare ping with exactly ``{"cmd": "ping", "ok": True}``
        and the head used to compare the reply for equality: an unconditional
        extra key would have broken every head not updated in the same breath.
        Asking makes the compatibility run the other way too -- a tail that
        predates A7 ignores the flag and answers the two-key form, which is
        read here as "no rail evidence" and NOT as a fault, because a tail that
        cannot report a degraded rail is not thereby a degraded tail.

        Returns ``True`` for "the connection is alive"; the verdict is left on
        ``peer_degraded``/``peer_rail`` rather than returned, so a caller that
        only wanted liveness (and every existing one did) is unchanged.
        """
        if self.sock is None:
            raise OSError("pipeline connection is closed")
        if self._active:
            raise RuntimeError("pipeline ping requires an idle peer")
        _send_json(self.sock, {"cmd": "ping", "rail": True})
        ack = _recv_json(self.sock)
        if (
            not isinstance(ack, dict)
            or ack.get("cmd") != "ping"
            or ack.get("ok") is not True
            or set(ack) - _PING_ACK_KEYS
        ):
            raise ValueError("pipeline ping not acknowledged")
        self.peer_degraded = bool(ack.get("degraded"))
        self.peer_rail = ack.get("rail")
        return True

    def close(self):
        if self.sock is not None:
            try:
                if not self._active:
                    _send_json(self.sock, {"cmd": "bye"})
                    ack = _recv_json(self.sock)
                    if ack != {"cmd": "bye", "ok": True}:
                        raise ValueError("pipeline bye not acknowledged")
            finally:
                self.abort()

    def abort(self):
        """Cancel I/O; never retry using this request's partially filled cache."""
        self._stop.set()
        if self.sock is not None:
            _abort_socket(self.sock)
        if self._th is not None and self._th.is_alive():
            self._th.join(timeout=self.settings.io_timeout)
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        self._active = False
        self._q = None

    # -- one prefill --------------------------------------------------------
    def begin(self, tokens: int, chunk: int, *, input_ids):
        if self._active or self.sock is None:
            raise RuntimeError("pipeline begin requires an idle connected peer")
        if input_ids.shape != (1, tokens):
            raise ValueError("pipeline prompt/token count mismatch")
        self.envelope = PrefillEnvelope.create(
            model_sha256=self.settings.model_sha256,
            source_revision=self.settings.source_revision,
            split=self.split,
            n_layers=self.n_layers,
            input_ids=input_ids[:, :-1],
            chunk=chunk,
        )
        self.stats = {
            "chunks": [],
            "wire_send_s": 0.0,
            "head_gpu_s": 0.0,
            "envelope": self.envelope.to_dict(),
            "pipeline_used": True,
        }
        self._token_hash = hashlib.sha256()
        self._model = None
        self._stop.clear()
        self._active = True
        self._started = time.perf_counter()
        _send_json(
            self.sock,
            {
                "cmd": "run",
                "tokens": tokens,
                "chunk": chunk,
                "split": self.split,
                "envelope": self.envelope.to_dict(),
                "handoff": True,
                "transport": self.transport,
            },
        )
        ack = _recv_json(self.sock)
        if not ack.get("ok") or ack.get("request_id") != self.envelope.request_id:
            self.abort()
            raise ValueError("pipeline run not acknowledged")
        self._q = queue.Queue(maxsize=2)
        self._err = []
        self._th = threading.Thread(target=self._sender, daemon=True)
        self._th.start()

    def _sender(self):
        try:
            stream = mx.new_stream(mx.cpu) if self.transport == "ring" else None
            while True:
                if self._stop.is_set():
                    return
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    self.sock.sendall(HDR.pack(MAGIC, EOF_IDX, 0, 0, 0, 0, 0))
                    return
                idx, h, nb = item
                B, S, HC, D = h.shape
                t0 = time.perf_counter()
                self.sock.sendall(
                    HDR.pack(MAGIC, idx, B, S, HC, D, 0 if nb is None else nb.nbytes)
                )
                if self.transport == "ring":
                    ring_send(h, 1, stream)
                else:
                    self.sock.sendall(memoryview(nb).cast("B"))
                self.stats["wire_send_s"] += time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            self._err.append(repr(e))
            if self.sock is not None:
                _abort_socket(self.sock)

    def local_caches(self, cache):
        return [c for c in cache[: self.split] if c is not None]

    def prefill_chunk(self, model, input_ids, inputs_embeds, cache):
        if not self._active:
            raise RuntimeError("pipeline request is not active")
        idx = len(self.stats["chunks"])
        if idx >= len(self.envelope.chunks) or input_ids.shape != (
            1,
            self.envelope.chunks[idx],
        ):
            raise ValueError("pipeline input chunk schedule mismatch")
        if self._model is not None and self._model is not model:
            raise ValueError("pipeline model changed during request")
        self._model = model
        self._token_hash.update(token_bytes(input_ids))
        t0 = time.perf_counter()
        h = pipeline_language_model(model).pipeline_prefill_head(
            inputs=input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            split=self.split,
        )
        mx.eval(h)
        self.stats["head_gpu_s"] += time.perf_counter() - t0
        if self._err:
            raise RuntimeError(f"pipeline peer failed: {self._err[0]}")
        if h.ndim != 4 or h.shape[:2] != input_ids.shape or h.dtype != mx.bfloat16:
            raise ValueError(
                "pipeline boundary must be B=1 bf16 with matching chunk length"
            )
        self.stats["chunks"].append(h.shape[1])
        _queue_put(
            self._q,
            (idx, h, None if self.transport == "ring" else to_wire(h)),
            self._err,
            self.settings.io_timeout,
        )

    def finalize(self, cache):
        """End the stream, pull stage B's caches back, install them for decode."""
        if (
            not self._active
            or tuple(self.stats["chunks"]) != self.envelope.chunks
            or self._token_hash.hexdigest() != self.envelope.token_sha256
        ):
            raise ValueError("pipeline completed token/depth/chunk mismatch")
        _queue_put(self._q, None, self._err, self.settings.io_timeout)
        self._th.join(timeout=self.settings.io_timeout)
        if self._th.is_alive():
            self.abort()
            raise TimeoutError("pipeline sender did not retire")
        if self._err:
            raise RuntimeError(f"pipeline peer failed: {self._err[0]}")
        done = _recv_json(self.sock)
        if done.get("cmd") != "done":
            raise ValueError("pipeline did not complete")
        PrefillEnvelope.from_dict(done.get("envelope")).require_match(self.envelope)
        t0 = time.perf_counter()
        schema = expected_state_meta(self._model, self.envelope)
        ho = handoff_recv(
            self.sock, rebuild=True, expected=self.envelope, expected_meta=schema
        )
        states = ho.pop("states")
        report = _recv_json(self.sock)
        PrefillEnvelope.from_dict(report.get("envelope")).require_match(self.envelope)
        t_eval = time.perf_counter()
        mx.eval(list(states.values()))
        ho["handoff_eval_s"] = time.perf_counter() - t_eval
        t_install = time.perf_counter()
        install_state(
            cache,
            states,
            fresh_caches=self._model.make_cache(),
            expected=self.envelope,
            expected_meta=schema,
        )
        ho["handoff_install_s"] = time.perf_counter() - t_install
        ho["handoff_total_s"] = time.perf_counter() - t0
        self.stats["handoff"] = ho
        self.stats["tail"] = report
        self.stats["prefill_wall_s"] = time.perf_counter() - self._started
        self._active = False
        return self.stats


# ------------------------------------------------------------------ receipts


class PipelineMetrics:
    """Counters a running server can be asked for.

    A pipeline that silently bypassed itself for a month looks exactly like a
    pipeline that was never enabled, so the bypass reasons are counted by name
    rather than logged: ``pp_bypass_reason`` is the histogram that tells an
    operator WHICH gate refused, not merely that something did.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.used = 0
            self.failed = 0
            self.reconnects = 0
            self.breaker_trips = 0
            self.handoff_bytes = 0
            self.wire_s = 0.0
            self.bypass = {}
            self.breaker_state = "closed"

    def note_bypass(self, reason: Optional[str]):
        if not reason:
            return
        with self._lock:
            self.bypass[reason] = self.bypass.get(reason, 0) + 1

    def note_reconnect(self):
        with self._lock:
            self.reconnects += 1

    def note_trip(self):
        with self._lock:
            self.breaker_trips += 1

    def note_failure(self):
        with self._lock:
            self.failed += 1

    def note_success(self, stats: Optional[dict] = None):
        with self._lock:
            self.used += 1
            if not stats:
                return
            self.wire_s += float(stats.get("wire_send_s") or 0.0)
            handoff = stats.get("handoff") or {}
            self.handoff_bytes += int(handoff.get("handoff_bytes") or 0)
            self.wire_s += float(handoff.get("handoff_wire_recv_s") or 0.0)

    def set_breaker_state(self, state: str):
        with self._lock:
            self.breaker_state = state

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pp_used": self.used,
                "pp_failed": self.failed,
                "pp_reconnects": self.reconnects,
                "pp_bypass_reason": dict(self.bypass),
                "pp_handoff_bytes": self.handoff_bytes,
                "pp_wire_s": round(self.wire_s, 6),
                "pp_breaker_state": self.breaker_state,
                "pp_breaker_trips": self.breaker_trips,
            }


METRICS = PipelineMetrics()


def pipeline_metrics_snapshot() -> dict:
    """Read-only receipts for the server runtime snapshot."""
    snap = METRICS.snapshot()
    snap["pp_pool_idle"] = POOL.idle_count()
    snap["pp_enabled"] = bool(os.environ.get("MLX_VLM_PIPELINE_HOSTS", "").strip())
    return snap


# ------------------------------------------------------------ circuit breaker


class CircuitBreaker:
    """N consecutive failures disable the peer for T seconds.

    Without it, a tail that died at 03:00 is re-dialled once per request until
    someone notices: every request pays the full connect timeout before falling
    back to the single box, which is strictly worse than never having had a
    peer.  Open trips straight to bypass; after the cooldown ONE request is let
    through (half-open) and its outcome decides.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 60.0, clock=time.monotonic):
        self.threshold = max(1, int(threshold))
        self.cooldown = float(cooldown)
        self.clock = clock
        self._lock = threading.Lock()
        self.failures = 0
        self.opened_at = None
        self._trial = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self._trial or self.clock() - self.opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        with self._lock:
            state = self._state_locked()
            if state == "open":
                return False
            if state == "half_open":
                # exactly one probe in flight
                if self._trial:
                    return False
                self._trial = True
            return True

    def record_success(self):
        with self._lock:
            self.failures = 0
            self.opened_at = None
            self._trial = False
            state = "closed"
        METRICS.set_breaker_state(state)

    def record_failure(self):
        tripped = False
        with self._lock:
            self._trial = False
            self.failures += 1
            if self.failures >= self.threshold:
                tripped = self.opened_at is None
                self.opened_at = self.clock()
            state = self._state_locked()
        if tripped:
            METRICS.note_trip()
        METRICS.set_breaker_state(state)

    def reset(self):
        with self._lock:
            self.failures = 0
            self.opened_at = None
            self._trial = False
        METRICS.set_breaker_state("closed")


# --------------------------------------------------------------- connection pool


def _pool_key(settings: PipelineSettings, split: int, n_layers: int):
    return (
        settings.peer,
        settings.transport,
        int(split),
        int(n_layers),
        settings.model_sha256,
        settings.source_revision,
    )


class PipelinePool:
    """One live connection per tail, reused across requests.

    ``maybe_open_pipeline`` used to open a socket per request and ``close()``
    sent ``bye``, which is why the cooperation driver had to monkey-patch
    ``close`` to keep the peer alive.  The connection is now a resource with a
    lifetime longer than the request: ``close()`` on a leased head returns it
    here, and only ``shutdown()`` says ``bye``.
    """

    def __init__(self, breaker: Optional[CircuitBreaker] = None):
        self._lock = threading.Lock()
        self._idle = {}
        self.breaker = breaker or CircuitBreaker(
            threshold=int(os.environ.get("MLX_VLM_PIPELINE_BREAKER_FAILS", "3")),
            cooldown=float(os.environ.get("MLX_VLM_PIPELINE_BREAKER_COOLDOWN", "60")),
        )

    def idle_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._idle.values())

    def acquire(self, settings, split, n_layers, *, verbose=False):
        """A connected, verified-idle head, or None (caller stays single-box)."""
        key = _pool_key(settings, split, n_layers)
        with self._lock:
            pooled = self._idle.get(key) or []
            head = pooled.pop() if pooled else None
            self._idle[key] = pooled
        if head is not None:
            # A pooled socket can have died since the last request; find out
            # here, where falling back is still free, not inside ``begin``.
            try:
                head.ping()
            except (OSError, ValueError, TimeoutError) as exc:
                if verbose:
                    print(f"[pipeline] pooled peer stale: {exc!r}", flush=True)
                _discard(head)
                METRICS.note_reconnect()
            else:
                if not getattr(head, "peer_degraded", False):
                    return head
                # The tail is ALIVE and answering; what it is telling us is that
                # its own rail p95 is past the bound.  Committing a >= 16k
                # prefill to it would buy a slow request in exchange for a fast
                # fallback, so the connection goes back in the pool (nothing is
                # wrong with the socket, and discarding it would make the
                # recovery cost a reconnect) and the request stays single-box.
                #
                # It is fed to the breaker rather than merely counted because
                # the breaker is the thing that already knows how to stop
                # ASKING: without it, every request pays a ping round trip and a
                # bypass for a rail whose verdict has hysteresis on it anyway.
                with self._lock:
                    self._idle.setdefault(key, []).append(head)
                self.breaker.record_failure()
                METRICS.note_bypass("peer_degraded")
                if verbose:
                    print(
                        f"[pipeline] bypass=peer_degraded rail={head.peer_rail!r}",
                        flush=True,
                    )
                return None
        if not self.breaker.allow():
            METRICS.note_bypass("breaker_open")
            if verbose:
                print("[pipeline] bypass=breaker_open", flush=True)
            return None
        try:
            head = PipelineHead(settings, split, n_layers).connect()
        except (OSError, ValueError, TimeoutError) as exc:
            # The request falls back to single-box prefill; a peer that is down
            # must never be able to fail a request that this box can serve.
            self.breaker.record_failure()
            METRICS.note_bypass("peer_unreachable")
            if verbose:
                print(f"[pipeline] bypass=peer_unreachable ({exc!r})", flush=True)
            return None
        self.breaker.record_success()
        return head

    def release(self, head, key, ok: bool):
        """Give a connection back.  A request that did not finish leaves the
        peer's state undefined, so its connection is discarded, never reused."""
        if head is None:
            return
        if not ok or head.sock is None:
            _discard(head)
            self.breaker.record_failure()
            METRICS.note_failure()
            return
        with self._lock:
            self._idle.setdefault(key, []).append(head)

    def shutdown(self):
        """Say ``bye`` to every pooled peer.  The only place that does."""
        with self._lock:
            heads = [h for hs in self._idle.values() for h in hs]
            self._idle = {}
        for head in heads:
            try:
                head.close()
            except Exception:  # noqa: BLE001
                _discard(head)


def _discard(head):
    try:
        head.abort()
    except Exception:  # noqa: BLE001
        pass


POOL = PipelinePool()

# One pipelined prefill at a time.  The tail is ONE stage-B stack: a second
# concurrent request cannot be served by it, and queueing behind the first would
# turn a fallback (cheap, single-box, correct) into a wait of up to a whole
# 131k prefill.  Non-blocking by construction, therefore.
_PP_INFLIGHT = threading.Lock()


def acquire_pipeline_slot() -> bool:
    """Take the single PP prefill slot, or count ``pp_busy`` and refuse."""
    if _PP_INFLIGHT.acquire(blocking=False):
        return True
    note_pipeline_bypass("pp_busy")
    return False


def release_pipeline_slot() -> None:
    """Give the slot back.  Idempotent: it runs in a ``finally``."""
    try:
        _PP_INFLIGHT.release()
    except RuntimeError:
        pass


class PooledPipelineHead:
    """A per-request lease over a pooled connection.

    Presents exactly the surface ``generate_step`` already uses -- ``begin``,
    ``prefill_chunk``, ``local_caches``, ``finalize``, ``close`` -- so the call
    site does not learn that the socket outlives it.  ``close`` is the request
    boundary and NEVER raises: it runs in the caller's ``finally``, where an
    exception would replace the real failure with a bookkeeping one.
    """

    def __init__(self, pool: PipelinePool, head: PipelineHead, key):
        self._pool = pool
        self._head = head
        self._key = key
        self._ok = False
        self.stats = head.stats

    # -- delegation ---------------------------------------------------------
    @property
    def split(self):
        return self._head.split

    def begin(self, tokens, chunk, *, input_ids):
        self._ok = False
        return self._head.begin(tokens, chunk, input_ids=input_ids)

    def local_caches(self, cache):
        return self._head.local_caches(cache)

    def prefill_chunk(self, model, input_ids, inputs_embeds, cache):
        return self._head.prefill_chunk(model, input_ids, inputs_embeds, cache)

    def finalize(self, cache):
        stats = self._head.finalize(cache)
        self.stats = stats
        self._ok = True
        METRICS.note_success(stats)
        return stats

    # -- lifetime -----------------------------------------------------------
    def close(self):
        try:
            self._pool.release(self._head, self._key, self._ok)
        except Exception as exc:  # noqa: BLE001
            _discard(self._head)
            print(f"[pipeline] release failed: {exc!r}", flush=True)
        finally:
            self._head = None

    def shutdown(self):
        """Explicit end of life for the whole pool: sends ``bye``."""
        if self._head is not None:
            self._pool.release(self._head, self._key, self._ok)
            self._head = None
        self._pool.shutdown()


# ------------------------------------------------------------------ factory


def pipeline_bypass_reason(*args, **kwargs):
    """The handoff only represents cold, unpadded, unquantized text B=1 state.

    Counted here rather than at the call site: this function is the single
    gate, so the histogram cannot drift away from the decision it describes.
    """
    reason = _pipeline_bypass_reason(*args, **kwargs)
    METRICS.note_bypass(reason)
    return reason


def note_pipeline_bypass(reason: Optional[str]) -> Optional[str]:
    """Count a refusal that is a RESOURCE fact rather than a request fact.

    ``breaker_open``, ``peer_unreachable`` and ``disabled`` are already counted
    where they are decided rather than inside ``pipeline_bypass_reason`` -- they
    are not properties of the request.  ``pp_busy`` is the same kind of fact and
    is counted the same way, so the histogram stays the single place an operator
    reads to learn WHICH gate refused.
    """
    METRICS.note_bypass(reason)
    return reason


def _pipeline_bypass_reason(
    *,
    ladder,
    capture,
    warm,
    pixel_values,
    mask,
    cache,
    input_ids,
    kv_quantized,
    right_pad=False,
):
    if ladder:
        return "apc_checkpoint_ladder"
    if capture:
        return "speculative_hidden_capture"
    if warm:
        return "warm_prefix"
    # New in the batch call site (A4) and additive: ``generate_step`` never sees
    # a right-padded batch, so it leaves this at its default and its reason
    # strings are unchanged.  ``prompt_step``'s batch can be right-padded (mixed
    # warm/cold prefill), and a right pad is invisible to every check below --
    # the mask is None, the cache carries the padding as metadata the handoff
    # schema has no field for, and B is still 1.
    if right_pad:
        return "right_pad_batch"
    if pixel_values is not None:
        return "multimodal_input"
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        return "batch_not_one"
    if kv_quantized:
        return "quantized_kv"
    if mask is not None and not bool(mx.all(mask).item()):
        return "padded_or_custom_mask"
    from .models.cache import ArraysCache, CacheList, KVCache

    def supported(c):
        return (
            type(c) in (ArraysCache, KVCache)
            or type(c) is CacheList
            and all(supported(child) for child in c.caches)
        )

    if not all(supported(c) for c in cache):
        return "unsupported_cache_type"

    def padded(c):
        if type(c) is CacheList:
            return any(padded(child) for child in c.caches)
        return any(
            getattr(c, attr, None) is not None for attr in ("left_padding", "lengths")
        )

    if any(padded(c) for c in cache):
        return "cache_padding_metadata"

    # state dereferences fresh KV keys; empty() checks only the first slot of
    # composite caches. Inspect every typed component and its offset instead.
    def populated(c):
        if type(c) is CacheList:
            return any(populated(child) for child in c.caches)
        if type(c) is ArraysCache:
            return any(value is not None for value in c.cache)
        return c.keys is not None or c.values is not None or c.offset != 0

    if any(populated(c) for c in cache):
        return "populated_prompt_cache"
    return None


def maybe_open_pipeline(model, total_tokens: int, verbose: bool = False):
    """Return a leased, connected pipeline head, or None to stay single-box.

    Every ``None`` here is a named bypass in ``pp_bypass_reason``.  Nothing in
    this function may raise on a peer problem: a tail that is down, slow or
    tripped must cost the request a fallback, not a failure.
    """
    # NO STICKY DISABLE.  This used to latch ``_CTX = _DISABLED`` on the first
    # refusal and return None forever after, so a process that reached here once
    # before ``MLX_VLM_PIPELINE_HOSTS`` was set -- a test module, a server whose
    # peer is configured after the model loads, an operator turning the feature
    # on -- could never enable the pipeline again.  Worse, the later requests
    # were not even COUNTED: the latch returned before ``note_bypass``, so the
    # histogram an operator reads to find out which gate refused went silent.
    # The whole check is a few environment reads against a >= 16k-token prefill.
    settings = PipelineSettings.from_env()
    if settings is None:
        METRICS.note_bypass("disabled")
        return None
    lm = pipeline_language_model(model)
    if lm is None:
        if verbose:
            print(
                "[pipeline] model has no pipeline hook; staying single-box", flush=True
            )
        METRICS.note_bypass("no_pipeline_hook")
        return None
    if total_tokens < settings.min_tokens:
        if verbose:
            print("[pipeline] bypass=below_min_tokens", flush=True)
        METRICS.note_bypass("below_min_tokens")
        return None
    try:
        _hex(settings.model_sha256, 64, "model_sha256")
        _hex(settings.source_revision, 40, "source_revision")
    except (ValueError, TypeError) as exc:
        # An unset or malformed pin is a MISCONFIGURATION, and this function's
        # contract is that a peer problem costs a fallback and never a failure.
        # It used to propagate, which turned a missing launcher variable into a
        # 500 on a request this box can serve by itself.
        METRICS.note_bypass("identity_unpinned")
        if verbose:
            print(f"[pipeline] bypass=identity_unpinned ({exc!r})", flush=True)
        return None
    try:
        split = resolve_split(model, settings, total_tokens, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        METRICS.note_bypass("split_unresolved")
        if verbose:
            print(f"[pipeline] bypass=split_unresolved ({exc!r})", flush=True)
        return None
    head = POOL.acquire(settings, split, lm.pipeline_num_layers, verbose=verbose)
    if head is None:
        return None
    if verbose:
        print(
            f"[pipeline] split={split} transport={settings.transport} "
            f"peer={settings.peer[0]}:{settings.peer[1]}",
            flush=True,
        )
    return PooledPipelineHead(
        POOL, head, _pool_key(settings, split, lm.pipeline_num_layers)
    )


def shutdown_pipeline_pool():
    """Say ``bye`` to every pooled tail.  Server shutdown calls this."""
    POOL.shutdown()
