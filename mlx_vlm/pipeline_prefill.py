"""Two-box layer-pipelined prefill prototype for glm5_next.

Splits the 45-layer glm5_next decoder stack across two Macs joined by a fast
link (Thunderbolt IP on this fleet) and runs standard chunked prefill so that
chunk ``i+1`` is on stage A while chunk ``i`` is on stage B.

Why prefill only: single-stream decode gains nothing from layer pipelining
(the two stage latencies serialize inside every token step). Prefill is a
stream of independent chunks, so the two stages can be kept busy at once.

What crosses the wire: glm5_next carries mHC (hyper-connection) streams
*with* the hidden state, so the inter-layer activation is
``(B, S, hc_mult, hidden_size)`` -- not ``(B, S, hidden_size)``. At
hc_mult=4/hidden=4096/bf16 that is ``S * 32 KiB`` per boundary
(64 MiB for a 2048-token chunk), 4x what a plain-residual model would send.

Caches stay local: KDA recurrent state and DSA latent/indexer KV are
per-layer, so a layer split needs no cache exchange during prefill. Only the
boundary activation moves.

Roles::

    # stage B (tail) first -- it listens
    python -m mlx_vlm.pipeline_prefill --role tail  --model PATH --split 23 --port 39200
    # stage A (head)
    python -m mlx_vlm.pipeline_prefill --role head  --model PATH --split 23 \
        --peer 10.0.0.2 --port 39200 --tokens 8192 --chunk 2048
    # single-process reference using the identical stage/chunk code
    python -m mlx_vlm.pipeline_prefill --role single --model PATH --tokens 8192 --chunk 2048

Measured on the twin M3 Ultra 512GB fleet (gesicht 10.0.0.1 = stage A layers
0:23, epsilon 10.0.0.2 = stage B layers 23:45), GLM-5.3-Flash q4, chunk 2048,
Thunderbolt IP link.  Prefill tok/s, warm, median of replicates::

    ctx     1 box (best of both)   2 box pipelined   speedup   ideal 2N/(N+1)
    8192          419.0                 613.5         1.46x        1.60x  (N=4)
    32768         316.0                 579.6         1.83x        1.88x  (N=16)
    131072        230.6                 427.6         1.85x        1.97x  (N=64)

Against the *sum of its own two stage times* the schedule runs at 95-100% of
the ideal two-stage pipeline: 1.55x / 1.85x / 1.89x at 8k / 32k / 131k.  The
only structural loss is the fill/drain bubble, which is 1/(N+1) of the
schedule -- so short prompts gain least.

Wire: 64.0 MiB per 2048-token boundary (32 KiB/token), one direction, 4.0 GiB
total for a 131k prefill.  10.9 ms idle over Thunderbolt IP with the MLX ring
backend, 12.0 ms with a raw Python socket, 28-41 ms in-flight here because the
sender/receiver threads contend with the compute thread for the GIL.  Even the
degraded number is 0.7-1.3% of a stage's per-chunk compute.

Split 23 balances the two halves within 5% (head 5 DSA + 3 dense MLP + 15 KDA,
tail 6 DSA + 16 KDA).  The optimum drifts with context because DSA cost grows
with context while KDA is flat and the halves hold 5 vs 6 DSA layers.

--handoff: shipping stage B's caches back for single-box decode costs 186 MiB /
59 ms at 8k, 547 MiB / 192 ms at 32k, 1988 MiB / 429 ms at 131k -- 0.14% of a
131k prefill.  DSA cache is 2562 B/token/layer, KDA state is a flat
4.14 MiB/layer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx
import numpy as np

from . import pipeline_admission as admission
from .models.base import create_attention_mask, create_ssm_mask
from .utils import load_model

MAGIC = b"GP51"
HDR = struct.Struct("!4sIIIIIQ")  # magic, chunk_idx, B, S, HC, D, nbytes
EOF_IDX = 0xFFFFFFFF
MAX_JSON_BYTES = 4 * 1024 * 1024
ADMISSION_REFUSED_EXIT = 3  # a refused start is not a crash and not a success
# How long a request ALREADY IN FLIGHT may keep running after a stop has been
# asked for.  30 s covers the remaining chunks of a 131k pipelined prefill at
# the measured rates; past it the connection is aborted and the head prefills
# single-box (``pp_failed``), which is the documented failure model.
DEFAULT_DRAIN_TIMEOUT_S = 30.0


def _hex(value, length, name):
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"invalid pipeline {name}")
    return value


def token_bytes(input_ids):
    """Canonical cold B=1 token prefix; never hashes embeddings or padding."""
    ids = np.asarray(input_ids)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype.kind not in "iu":
        raise ValueError("pipeline requires B=1 integer token IDs")
    if np.any(ids < 0) or np.any(ids > 2147483647):
        raise ValueError("pipeline token ID out of range")
    return ids.astype("<i4", copy=False).tobytes()


@dataclass(frozen=True)
class PrefillEnvelope:
    """Request identity. Model/source pins are supplied by a verified manifest.

    The tail sees activations, so the token digest is a head-side attestation,
    not an independent tokenization check on the tail.
    """

    schema: int
    request_id: str
    model_sha256: str
    source_revision: str
    split: int
    n_layers: int
    batch: int
    depth: int
    token_sha256: str
    chunks: tuple

    def __post_init__(self):
        _hex(self.request_id, 32, "request_id")
        _hex(self.model_sha256, 64, "model_sha256")
        _hex(self.source_revision, 40, "source_revision")
        _hex(self.token_sha256, 64, "token_sha256")
        for name in ("schema", "split", "n_layers", "batch", "depth"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"invalid pipeline {name}")
        if (
            self.schema != 1
            or self.batch != 1
            or not 0 < self.split < self.n_layers <= 1024
        ):
            raise ValueError("unsupported pipeline schema/batch/layers")
        if not 0 < self.depth <= 2**24 or not 0 < len(self.chunks) <= 65536:
            raise ValueError("invalid pipeline depth/chunks")
        if (
            any(type(n) is not int or n <= 0 for n in self.chunks)
            or sum(self.chunks) != self.depth
        ):
            raise ValueError("pipeline depth/chunks mismatch")

    @classmethod
    def create(
        cls, *, model_sha256, source_revision, split, n_layers, input_ids, chunk
    ):
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("invalid pipeline chunk size")
        raw = token_bytes(input_ids)
        depth = len(raw) // 4
        chunks = tuple(min(chunk, depth - p) for p in range(0, depth, chunk))
        return cls(
            1,
            uuid.uuid4().hex,
            model_sha256,
            source_revision,
            split,
            n_layers,
            1,
            depth,
            hashlib.sha256(raw).hexdigest(),
            chunks,
        )

    def to_dict(self):
        out = asdict(self)
        out["chunks"] = list(self.chunks)
        return out

    @classmethod
    def from_dict(cls, obj):
        if not isinstance(obj, dict) or set(obj) != set(cls.__dataclass_fields__):
            raise ValueError("missing or unknown pipeline envelope fields")
        if not isinstance(obj["chunks"], (tuple, list)):
            raise ValueError("invalid pipeline chunks")
        return cls(**{**obj, "chunks": tuple(obj["chunks"])})

    def require_match(self, expected):
        if self != expected:
            raise ValueError("pipeline envelope mismatch")


def _check_peer_identity(message, model_sha256, source_revision, split, n_layers):
    wanted = dict(
        schema=1,
        model_sha256=_hex(model_sha256, 64, "model_sha256"),
        source_revision=_hex(source_revision, 40, "source_revision"),
        split=split,
        n_layers=n_layers,
    )
    if any(message.get(k) != v for k, v in wanted.items()):
        raise ValueError("pipeline peer identity mismatch")
    return wanted


def _abort_socket(sock):
    """Wake socket workers; model owners unwind normally at a chunk boundary."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _queue_put(q, item, errors, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if errors:
            raise RuntimeError(f"pipeline worker failed: {errors[0]}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("pipeline queue progress timeout")
        try:
            q.put(item, timeout=min(0.1, remaining))
            return
        except queue.Full:
            pass


def _check_stop_file(path):
    if path and Path(path).exists():
        raise InterruptedError("pipeline cooperative STOP requested")


class ShutdownGate:
    """Why the tail is stopping, and how long the request in flight still has.

    A2b.  The tail had two unrelated ways of being told to stop and they never
    met: the cooperative STOP *file*, which every socket wait polls, and the
    signal *flag*, which only the accept loop between connections ever read.  A
    pooled head keeps ONE connection open across requests
    (``pipeline_runtime.PipelinePool``), so ``session`` -- not ``accept`` -- is
    where a busy tail lives, and the flag was never looked at there.  That is
    how a SIGTERMed tail went on serving for eight minutes with
    ``shutdown_reason: "signal_15"`` already on its own health line (B3
    fallback drills, 2026-09-07 15:25-16:02: D2/D3/D4/D5 all void, and the
    flock it still held refused every replacement tail).

    Both sources live here now, behind one object that is passed where the stop
    file used to go -- so every place that already polled the file observes the
    signal too -- and the answer depends on what the tail is doing:

    ===================  ==================================================
    idle (no request)    stop -> raise at the next poll (<= 0.25 s)
    inside a request     stop -> keep going, until ``drain_timeout``
    inside a request     drain expired, or a second signal -> raise now
    ===================  ==================================================

    Plain attribute stores only, never a lock: :meth:`request` runs inside a
    signal handler, and taking a lock the interrupted thread may already hold
    is a deadlock.  Single stores are atomic under the GIL, which is all the
    handler needs.
    """

    def __init__(
        self,
        stop_file=None,
        drain_timeout: float = DEFAULT_DRAIN_TIMEOUT_S,
        clock=time.monotonic,
    ):
        self.stop_file = stop_file or None
        self.drain_timeout = max(0.0, float(drain_timeout))
        self.clock = clock
        self.reason = None
        self.stopping = False
        self.hard = False
        self.deadline = None
        self.in_request = False
        self.requests = 0

    # -- control (signal-handler safe) --------------------------------------
    def request(self, reason: str = "shutdown") -> str:
        """First ask starts the drain; a second one is the hard stop.

        Returns ``"soft"`` or ``"hard"``.  Idempotent in the sense that
        matters: the reason and the deadline are only ever set once by the
        first ask, so a supervisor that signals twice cannot extend the drain.
        """
        self.requests += 1
        if not self.stopping:
            self.reason = reason
            self.deadline = self.clock() + self.drain_timeout
            self.stopping = True
            return "soft"
        self.hard = True
        self.deadline = self.clock()
        return "hard"

    def enter_request(self):
        """A request owns the connection: the drain window applies from here."""
        self.in_request = True

    def leave_request(self):
        self.in_request = False

    # -- verdicts ------------------------------------------------------------
    @property
    def drained(self) -> bool:
        if not self.stopping:
            return False
        if self.hard or self.deadline is None:
            return True
        return self.clock() >= self.deadline

    def remaining(self):
        """Seconds left of the drain window, or None when not stopping."""
        if not self.stopping or self.deadline is None:
            return None
        return max(0.0, self.deadline - self.clock())

    def check(self):
        """Raise if this wait must not continue.  Called from every I/O poll."""
        _check_stop_file(self.stop_file)
        if not self.stopping:
            return
        if not self.in_request:
            raise InterruptedError(
                f"pipeline shutdown ({self.reason}): retiring an idle connection"
            )
        if self.hard:
            raise InterruptedError(
                f"pipeline shutdown ({self.reason}): second signal, aborting the "
                "request in flight"
            )
        if self.drained:
            raise InterruptedError(
                f"pipeline shutdown ({self.reason}): drain timeout "
                f"{self.drain_timeout:g}s expired, aborting the request in flight"
            )

    def snapshot(self) -> dict:
        return {
            "stopping": self.stopping,
            "hard_stop": self.hard,
            "drain_timeout_s": round(self.drain_timeout, 3),
            "drain_remaining_s": (
                None if self.remaining() is None else round(self.remaining(), 3)
            ),
        }


def _check_stop(stop):
    """``stop`` is a :class:`ShutdownGate` (the service) or a stop-file path
    (the bench tail, and every caller that predates A2b)."""
    if isinstance(stop, ShutdownGate):
        stop.check()
    else:
        _check_stop_file(stop)


class StopAwareSocket:
    """Tail-owned stop checks during idle and transfer, without signal I/O.

    The second argument is a :class:`ShutdownGate` for the service and a bare
    stop-file path for the bench roles; both are polled at most 0.25 s apart,
    inside every recv/send wait, so a stop is observed while the socket is idle
    and not only between connections.
    """

    def __init__(self, sock, stop, timeout):
        self.sock, self.stop, self.timeout = sock, stop, timeout
        sock.settimeout(min(0.25, timeout))

    @property
    def stop_file(self):
        """Back-compat alias: this used to be the only kind of stop there was."""
        return self.stop

    def gettimeout(self):
        return self.timeout

    def recv_into(self, view, n):
        deadline = time.monotonic() + self.timeout
        while True:
            _check_stop(self.stop)
            if time.monotonic() >= deadline:
                raise TimeoutError("pipeline socket receive timeout")
            try:
                return self.sock.recv_into(view, n)
            except socket.timeout:
                pass

    def sendall(self, data):
        view = memoryview(data).cast("B")
        sent = 0
        deadline = time.monotonic() + self.timeout
        while sent < len(view):
            _check_stop(self.stop)
            if time.monotonic() >= deadline:
                raise TimeoutError("pipeline socket send timeout")
            try:
                n = self.sock.send(view[sent : sent + 2**20])
            except socket.timeout:
                continue
            if not n:
                raise ConnectionError("pipeline peer closed during send")
            sent += n

    def shutdown(self, how):
        return self.sock.shutdown(how)

    def close(self):
        return self.sock.close()


# ---------------------------------------------------------------- model side


def load_stage(model_path: str, lo: int, hi: int, prune: bool = True):
    """Load glm5_next lazily and keep only layers [lo, hi).

    ``lazy=True`` skips ``mx.eval(model.parameters())``, so the discarded
    half's weights are never materialized -- dropping the module references
    before the first forward keeps a stage's resident set to its own half.
    """
    t0 = time.perf_counter()
    model = load_model(Path(model_path), lazy=True)
    caches = model.make_cache()
    lm = model.language_model.model
    n_layers = len(lm.layers)
    local = list(range(lo, hi))
    for i in range(n_layers):
        if i < lo or i >= hi:
            caches[i] = None
    if prune:
        for i in range(n_layers):
            if i < lo or i >= hi:
                lm.layers[i] = None
        # the vision tower is unused for a text prefill benchmark
        model.vision_model = None
        if lo > 0:
            lm.embed_tokens = None
        if hi < n_layers:
            model.language_model.lm_head = None
        gc.collect()
        mx.clear_cache()
    load_s = time.perf_counter() - t0
    return model, caches, local, n_layers, load_s


class Stage:
    """One contiguous slice of the decoder stack plus its local caches."""

    def __init__(self, model, caches, local: List[int], n_layers: int):
        self.model = model
        self.lm = model.language_model.model
        self.caches = caches
        self.local = local
        self.n_layers = n_layers
        self.is_head = local[0] == 0
        self.is_tail = local[-1] == n_layers - 1
        self.hc_mult = self.lm.hc_mult
        # Masks depend only on (N, cache offset), and offsets are identical for
        # every layer of a kind, so any local layer of that kind can supply it.
        self.ssm_local = next((i for i in local if self.lm.layers[i].is_linear), None)
        self.fa_local = next(
            (i for i in local if not self.lm.layers[i].is_linear), None
        )

    def __call__(
        self,
        h: mx.array,
        inputs: Optional[mx.array] = None,
        hidden_sink: Optional[list] = None,
        capture_layer_ids: Optional[list] = None,
    ) -> mx.array:
        """Run this stage over one chunk (delegates to Glm5NextModel).

        The capture arguments are passed on ONLY when a sink is open, so a
        request with no hidden-reading drafter -- and every model whose
        ``pipeline_forward`` predates the capture -- sees exactly the call it
        saw before.
        """
        kwargs = {}
        if hidden_sink is not None:
            kwargs = dict(hidden_sink=hidden_sink, capture_layer_ids=capture_layer_ids)
        return self.lm.pipeline_forward(
            h, self.caches, self.local[0], self.local[-1] + 1, inputs=inputs, **kwargs
        )

    def finish(self, h: mx.array) -> mx.array:
        """Tail only: collapse the hc streams, final norm, LM head on the last position."""
        return self.model.language_model._logits(self.lm.pipeline_finish(h)[:, -1:, :])

    def eval_state(self):
        st = []
        for c in self.caches:
            if c is not None:
                st.append(c.state)
        mx.eval(st)


def boundary_bytes(S: int, hc: int, D: int, itemsize: int = 2) -> int:
    return S * hc * D * itemsize


# ------------------------------------------------ speculative hidden capture
#
# A hidden-reading drafter (DFlash2, MTP) is primed on the target's own
# activations over the prompt, so a prefill that ran half its layers on another
# box has to bring that half back or the request cannot use the drafter at all.
# Until A6 it did not: ``speculative_hidden_capture`` refused every such request,
# which in the DEFAULT served config (DFlash2) is every request.
#
# WHAT IS SENT, and why it is not the whole capture.  The drafter reads only its
# trailing ``keep`` rows (DFlash2: ``sliding_window - 1`` = 2047; MTP:
# ``mtp_prime_window()`` = 2048), so only those rows have to cross -- once, at
# the end of the request.  Per captured layer that is ``keep * hidden * 2`` B:
# 16.8 MB at 2047x4096 bf16, so 50.3 MB for DFlash2's three tail-side layers and
# 16.8 MB for MTP's one.  The alternative -- shipping each chunk's capture with
# the chunk -- is 3.22 GB on a 131k prompt, 1.55x the KV handoff itself, for rows
# that are thrown away as soon as a later chunk covers the window.
#
# WHICH BOX CAPTURES WHAT.  DFlash2's ``target_layer_ids`` are [5, 14, 24, 33,
# 42] and the shipped split is 23, so the set STRADDLES the boundary: 5 and 14
# are on the head, 24/33/42 on the tail.  Both halves therefore capture, each
# over its own layers, and the head merges the two ordered lists.  MTP's capture
# is the pre-final-norm hidden after the LAST layer, which only the tail has.


class CaptureUnsupported(ValueError):
    """The peer cannot serve the capture this request asked for."""


@dataclass(frozen=True)
class CaptureSpec:
    """What a request wants captured, in the form both boxes agree on.

    ``kind`` is ``"layers"`` (a per-layer capture: dflash/eagle3
    ``capture_layer_ids``) or ``"hidden"`` (MTP's ``return_hidden``, the
    pre-final-norm mHC-collapsed hidden after the last layer).  ``keep`` is the
    drafter's own window; the head derives it from
    ``PrefillHiddenAccumulator.keep`` so the window that comes back is the one
    the accumulator would have trimmed to, and never a different one.
    """

    kind: str
    layers: tuple
    keep: int

    MAX_KEEP = 1 << 20

    @classmethod
    def parse(cls, obj, *, n_layers: int) -> "CaptureSpec":
        if not isinstance(obj, dict) or set(obj) - {"schema", "kind", "layers", "keep"}:
            raise CaptureUnsupported("invalid capture request")
        if obj.get("schema") != 1:
            raise CaptureUnsupported("unsupported capture schema")
        kind = obj.get("kind")
        if kind not in ("layers", "hidden"):
            raise CaptureUnsupported(f"unsupported capture kind {kind!r}")
        keep = obj.get("keep")
        if type(keep) is not int or not 0 < keep <= cls.MAX_KEEP:
            raise CaptureUnsupported("invalid capture window")
        layers = obj.get("layers") or []
        if not isinstance(layers, list) or any(type(i) is not int for i in layers):
            raise CaptureUnsupported("invalid capture layer ids")
        if any(not 0 <= i < n_layers for i in layers):
            raise CaptureUnsupported("capture layer id outside the stack")
        if kind == "layers" and not layers:
            raise CaptureUnsupported("a per-layer capture names no layers")
        if kind == "hidden" and layers:
            raise CaptureUnsupported("a whole-hidden capture names layers")
        if list(layers) != sorted(set(layers)):
            raise CaptureUnsupported("capture layer ids are not ascending/unique")
        return cls(kind, tuple(layers), int(keep))

    def to_dict(self) -> dict:
        return {
            "schema": 1,
            "kind": self.kind,
            "layers": list(self.layers),
            "keep": self.keep,
        }

    def head_layers(self, split: int) -> list:
        """Captured layers stage A owns.  Empty for ``hidden``."""
        return [i for i in self.layers if i < split]

    def tail_layers(self, split: int) -> list:
        return [i for i in self.layers if i >= split]

    def tail_tensors(self, split: int) -> int:
        return 1 if self.kind == "hidden" else len(self.tail_layers(split))


class TrailingHiddenWindow:
    """The last ``keep`` captured rows of a chunked prefill, and nothing else.

    Deliberately the same arithmetic as
    ``speculative.utils.PrefillHiddenAccumulator``: hold the chunk pieces, drop
    a whole leading piece as soon as what follows it already covers the window,
    and trim once at the end -- never per chunk, because a chunk boundary is not
    the prompt end.  It is a SECOND implementation only because it has to run in
    the tail process, which loads no drafter and must not import the speculative
    package to serve a prefill; ``test_pipeline_hidden_capture`` pins the two
    against each other on random data so they cannot drift.

    Bounded memory is the point: ``keep`` rows per captured layer, so a 131k
    prompt costs the same as a 16k one (16.8 MB per layer at 2047x4096 bf16).
    """

    def __init__(self, keep: int):
        self.keep = int(keep) if keep and int(keep) > 0 else None
        self._layers = None
        self._widths = []
        self.total_rows = 0
        self.dropped_rows = 0

    def append(self, captured) -> None:
        if not captured:
            return
        if self._layers is None:
            self._layers = [[] for _ in captured]
        if len(captured) != len(self._layers):
            raise ValueError("pipeline capture width changed mid-prompt")
        width = int(captured[0].shape[1])
        for slot, h in zip(self._layers, captured):
            if int(h.shape[1]) != width:
                raise ValueError("pipeline capture layers disagree on length")
            slot.append(h)
        self._widths.append(width)
        self.total_rows += width
        resident = self.total_rows - self.dropped_rows
        while len(self._widths) > 1 and (
            self.keep is not None and resident - self._widths[0] >= self.keep
        ):
            head = self._widths.pop(0)
            for slot in self._layers:
                slot.pop(0)
            self.dropped_rows += head
            resident -= head

    def pending(self) -> list:
        """The most recent chunk's pieces, for ``mx.eval``."""
        if self._layers is None:
            return []
        return [slot[-1] for slot in self._layers if slot]

    def window(self) -> list:
        """Per-layer copies of the trailing ``min(keep, total_rows)`` rows.

        Copied (``mx.contiguous``) and evaluated for the reason the accumulator
        gives: an mx slice is a view that pins its parent buffer, and an
        unevaluated one pins every intermediate behind it -- either would keep
        the whole prefill alive behind a 16.8 MB window.
        """
        if self._layers is None:
            return []
        out = []
        for slot in self._layers:
            h = slot[0] if len(slot) == 1 else mx.concatenate(slot, axis=1)
            if self.keep is not None and self.keep < int(h.shape[1]):
                h = h[:, -self.keep :]
            out.append(mx.contiguous(h))
        mx.eval(out)
        return out


def capture_meta(spec: CaptureSpec, ids: list, window: list, rows: int) -> dict:
    """Describe a window so the receiver can size the payload before reading it."""
    if not window:
        return {"schema": 1, "kind": spec.kind, "layers": [], "rows": int(rows),
                "width": 0, "dim": 0, "dtype": "bfloat16"}
    dtype = str(window[0].dtype).rsplit(".", 1)[-1]
    if dtype not in _ITEMSIZES:
        raise ValueError("unsupported capture dtype")
    if any(h.ndim != 3 or h.shape[0] != 1 for h in window):
        raise ValueError("a capture window is [1, rows, dim]")
    if any(str(h.dtype).rsplit(".", 1)[-1] != dtype for h in window):
        raise ValueError("capture window layers disagree on dtype")
    if any(int(h.shape[1]) != int(window[0].shape[1]) for h in window) or any(
        int(h.shape[2]) != int(window[0].shape[2]) for h in window
    ):
        raise ValueError("capture window layers disagree on shape")
    return {
        "schema": 1,
        "kind": spec.kind,
        "layers": [int(i) for i in ids],
        "rows": int(rows),
        "width": int(window[0].shape[1]),
        "dim": int(window[0].shape[2]),
        "dtype": dtype,
    }


def capture_send(sock, window: list) -> int:
    """Raw payloads, in the order ``meta["layers"]`` names them."""
    total = 0
    for h in window:
        nb = np.array(h.view(mx.uint8), copy=False)
        sock.sendall(memoryview(nb).cast("B"))
        total += int(h.nbytes)
    return total


def capture_recv(sock, meta, *, spec: CaptureSpec, expect_ids, expect_rows, expect_dim):
    """Read the peer's window, or refuse it.  Never trusts the peer's arithmetic.

    Every number the sender supplies is compared against one the receiver
    derived independently (which layers it asked the peer for, how many rows the
    schedule it dictated covers, how wide its own capture is).  A window that
    does not match is not a slower prefill, it is a drafter primed on the wrong
    rows -- so it fails the request into the single-box fallback instead.
    """
    if not isinstance(meta, dict) or set(meta) != {
        "schema", "kind", "layers", "rows", "width", "dim", "dtype",
    }:
        raise ValueError("invalid pipeline capture descriptor")
    if meta["schema"] != 1 or meta["kind"] != spec.kind:
        raise ValueError("pipeline capture schema/kind mismatch")
    ids, rows, width, dim = meta["layers"], meta["rows"], meta["width"], meta["dim"]
    if not isinstance(ids, list) or [int(i) for i in ids] != list(expect_ids):
        raise ValueError("pipeline capture layer ids mismatch")
    if type(rows) is not int or rows != int(expect_rows):
        raise ValueError("pipeline capture depth mismatch")
    # How many tensors the peer owes: one per layer it was asked for, or exactly
    # one for the whole-hidden (MTP) capture, which lives after the last layer
    # and therefore always on the tail.
    n = 1 if spec.kind == "hidden" else len(expect_ids)
    if type(width) is not int or type(dim) is not int:
        raise ValueError("invalid pipeline capture shape")
    if n == 0:
        # The split left this peer none of the captured layers.  It still has to
        # say so, and say it in the empty form, so a peer that simply forgot the
        # window cannot pass for one that had nothing to send.
        if (width, dim, meta["dtype"]) != (0, 0, "bfloat16"):
            raise ValueError("pipeline capture claims rows for no layers")
        return [], {"capture_bytes": 0, "capture_recv_s": 0.0,
                    "capture_rows": 0, "capture_tensors": 0}
    if width != min(spec.keep, int(expect_rows)):
        raise ValueError("pipeline capture window width mismatch")
    if expect_dim is not None and dim != int(expect_dim):
        raise ValueError("pipeline capture feature width mismatch")
    if not 0 < dim <= 2 ** 20 or meta["dtype"] not in _ITEMSIZES:
        raise ValueError("invalid pipeline capture shape/dtype")
    dt = _DTYPES[meta["dtype"]]
    nbytes = width * dim * _ITEMSIZES[meta["dtype"]]
    if nbytes * n > 8 * 2 ** 30:
        raise ValueError("pipeline capture exceeds byte limit")
    out = []
    t0 = time.perf_counter()
    for _ in range(n):
        buf = bytearray(nbytes)
        _recv_exact(sock, memoryview(buf), nbytes)
        flat = mx.array(np.frombuffer(buf, dtype=np.uint8))
        out.append(flat.view(dt).reshape((1, width, dim)))
    return out, {
        "capture_bytes": nbytes * n,
        "capture_recv_s": time.perf_counter() - t0,
        "capture_rows": width,
        "capture_tensors": n,
    }


# ------------------------------------------------------------- wire helpers


def _recv_exact(sock, view, n):
    got = 0
    timeout = sock.gettimeout()
    deadline = None if timeout is None else time.monotonic() + timeout
    while got < n:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("pipeline payload progress timeout")
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("peer closed mid-payload")
        got += k
    return got


def _send_json(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(b)) + b)


def _recv_json(sock):
    raw = bytearray(4)
    _recv_exact(sock, memoryview(raw), 4)
    (n,) = struct.unpack("!I", bytes(raw))
    if not 0 < n <= MAX_JSON_BYTES:
        raise ValueError("pipeline JSON frame exceeds limit")
    buf = bytearray(n)
    _recv_exact(sock, memoryview(buf), n)
    return json.loads(bytes(buf))


def to_wire(h: mx.array):
    """bf16 -> uint16 view -> numpy (zero-copy on unified memory)."""
    u = h.view(mx.uint16)
    return np.array(u, copy=False)


def from_wire(buf: bytearray, shape) -> mx.array:
    a = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
    return mx.array(a).view(mx.bfloat16)


# ----------------------------------------------------------- ring transport

_RING_GROUP = None


def ring_group():
    """Lazily join the MLX ring backend.

    Requires ``MLX_HOSTFILE`` (a file holding e.g. ``[["10.0.0.1:39400"],
    ["10.0.0.2:39401"]]``) and ``MLX_RANK``; ``--ring-hosts`` writes that file
    for you. Deliberately *not* jaccl: that backend wants RDMA device
    enumeration and a Thunderbolt Bridge, which is out of bounds on this fleet.
    The ring backend is plain TCP over whatever IPs you hand it, so it rides
    the existing tbnet with no interface changes at all.
    """
    global _RING_GROUP
    if _RING_GROUP is None:
        _RING_GROUP = mx.distributed.init(backend="ring")
    return _RING_GROUP


def setup_ring_env(ring_hosts: Optional[str], rank: int):
    if not ring_hosts:
        return
    hosts = [h.strip() for h in ring_hosts.split(",") if h.strip()]
    path = Path(os.environ.get("TMPDIR", "/tmp")) / "mlx_pipeline_ring_hosts.json"
    path.write_text(json.dumps([[h] for h in hosts]))
    os.environ["MLX_HOSTFILE"] = str(path)
    os.environ["MLX_RANK"] = str(rank)


def ring_send(h: mx.array, dst: int, stream):
    # Send/Recv have no GPU implementation -- they must run on a CPU stream, and
    # MLX streams are thread-local, so the stream is created inside the worker
    # thread that uses it. Measured: 12.7 ms for the 64 MiB boundary tensor with
    # the GPU fully loaded, and no measurable slowdown of the compute stream.
    mx.eval(mx.distributed.send(h, dst, group=ring_group(), stream=stream))


def ring_recv(shape, src: int, stream) -> mx.array:
    tmpl = mx.zeros(shape, dtype=mx.bfloat16)
    h = mx.distributed.recv_like(tmpl, src, group=ring_group(), stream=stream)
    mx.eval(h)
    return h


# --------------------------------------------------- stage-B cache handoff


def describe_state(obj):
    """Structure descriptor for a cache state tree (arrays -> shape/dtype)."""
    if obj is None:
        return {"k": "none"}
    if isinstance(obj, mx.array):
        return {
            "k": "arr",
            "dtype": str(obj.dtype).rsplit(".", 1)[-1],
            "shape": list(obj.shape),
            "nbytes": int(obj.nbytes),
        }
    if isinstance(obj, (list, tuple)):
        return {"k": "seq", "items": [describe_state(o) for o in obj]}
    raise TypeError(f"unhandled cache state node: {type(obj)}")


def collect_arrays(obj, out):
    """Arrays in the same order describe_state walks them."""
    if isinstance(obj, mx.array):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            collect_arrays(o, out)
    return out


def rebuild_state(desc, it):
    k = desc["k"]
    if k == "none":
        return None
    if k == "arr":
        return next(it)
    return [rebuild_state(d, it) for d in desc["items"]]


def collect_state(caches):
    """(layer index, descriptor, arrays) for every populated local cache."""
    entries = []
    for i, c in enumerate(caches):
        if c is None:
            continue
        st = c.state
        entries.append((i, describe_state(st), collect_arrays(st, [])))
    return entries


def handoff_send(sock, caches, *, envelope):
    """Stretch goal: ship stage B's KDA/DSA caches back so decode can run on box A.

    Always on the control socket: measured 4.6 GB/s at 131k, and it is a
    one-shot cost (0.18% of a 131k prefill), so it is not worth the extra
    ring-template plumbing.
    """
    t_pack = time.perf_counter()
    entries = collect_state(caches)
    mx.eval([a for _, _, arrs in entries for a in arrs])
    meta = [{"layer": i, "desc": d} for i, d, _ in entries]
    _validate_meta(meta, envelope)
    pack_s = time.perf_counter() - t_pack
    _send_json(sock, {"cmd": "handoff", "envelope": envelope.to_dict(), "meta": meta})
    total = 0
    t0 = time.perf_counter()
    for _, _, arrs in entries:
        for a in arrs:
            if a.nbytes == 0:
                continue
            nb = np.array(a.view(mx.uint8), copy=False)
            sock.sendall(memoryview(nb).cast("B"))
            total += a.nbytes
    dt = time.perf_counter() - t0
    return {
        "handoff_pack_s": pack_s,
        "handoff_send_s": dt,
        "handoff_bytes": total,
        "handoff_tensors": sum(len(a) for _, _, a in entries),
    }


_DTYPES = {
    n: getattr(mx, n)
    for n in ("bfloat16", "float16", "float32", "uint8", "uint16", "int32", "int64")
}
_ITEMSIZES = dict(bfloat16=2, float16=2, float32=4, uint8=1, uint16=2, int32=4, int64=8)


def _validate_desc(desc, level=0):
    if not isinstance(desc, dict) or level > 8:
        raise ValueError("invalid pipeline state descriptor")
    kind = desc.get("k")
    if kind == "none" and set(desc) == {"k"}:
        return 0
    if kind == "seq" and set(desc) == {"k", "items"}:
        items = desc["items"]
        if not isinstance(items, list) or len(items) > 64:
            raise ValueError("invalid pipeline state sequence")
        return sum(_validate_desc(d, level + 1) for d in items)
    if kind != "arr" or set(desc) != {"k", "dtype", "shape", "nbytes"}:
        raise ValueError("invalid pipeline array descriptor")
    shape, dtype, nbytes = desc["shape"], desc["dtype"], desc["nbytes"]
    if dtype not in _ITEMSIZES or not isinstance(shape, list) or len(shape) > 8:
        raise ValueError("invalid pipeline dtype/shape")
    if any(type(d) is not int or d < 0 or d > 2**24 for d in shape):
        raise ValueError("invalid pipeline shape")
    if (
        type(nbytes) is not int
        or nbytes != math.prod(shape) * _ITEMSIZES[dtype]
        or nbytes > 8 * 2**30
    ):
        raise ValueError("pipeline shape/nbytes mismatch")
    return nbytes


def _validate_meta(meta, envelope):
    if not isinstance(meta, list) or len(meta) != envelope.n_layers - envelope.split:
        raise ValueError("pipeline handoff layer count mismatch")
    layers = []
    total = 0
    for ent in meta:
        if (
            not isinstance(ent, dict)
            or set(ent) != {"layer", "desc"}
            or type(ent["layer"]) is not int
        ):
            raise ValueError("invalid pipeline layer descriptor")
        layers.append(ent["layer"])
        total += _validate_desc(ent["desc"])
    if sorted(layers) != list(range(envelope.split, envelope.n_layers)):
        raise ValueError("pipeline handoff layer indices mismatch")
    if total > 64 * 2**30:
        raise ValueError("pipeline handoff exceeds byte limit")


def expected_state_meta(model, envelope):
    """Exact GLM KDA/DSA shape and dtype schema, independent of peer metadata.

    Schema 1 is bf16 text prefill with fp32 recurrent accumulation. Quantized
    caches and other state formats require a different, explicitly supported
    schema rather than coercion at the receiver.
    """

    def arr(shape, dtype="bfloat16"):
        return dict(
            k="arr",
            shape=list(shape),
            dtype=dtype,
            nbytes=math.prod(shape) * _ITEMSIZES[dtype],
        )

    def seq(*items):
        return dict(k="seq", items=list(items))

    result = []
    # The head may be the VLM wrapper (``generate_step``) or the language model
    # itself (the server hands ``BatchGenerator`` ``model.language_model``), and
    # the tail always passes its own loaded wrapper; resolve rather than assume.
    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "pipeline_prefill_head"):
        lm = model
    layers = lm.model.layers
    if len(layers) != envelope.n_layers:
        raise ValueError("pipeline model layer count mismatch")
    for i in range(envelope.split, envelope.n_layers):
        layer = layers[i]
        if layer is None:
            # Prototype heads prune tail modules before loading weights, but
            # retain the validated architecture config needed for the schema.
            cfg = lm.args
            linear = cfg.layer_types[i] == "linear_attention"
            if linear:
                h, d, k = (
                    cfg.linear_num_heads,
                    cfg.linear_head_dim,
                    cfg.linear_conv_kernel_dim,
                )
            else:
                rank, index_dim = cfg.kv_lora_rank, cfg.index_head_dim
        else:
            a = layer.self_attn
            linear = layer.is_linear
            if linear:
                h, d, k = a.num_heads, a.head_dim, a.conv_kernel_size
            else:
                rank, index_dim = a.kv_lora_rank, a.indexer.head_dim
        if linear:
            desc = seq(arr((1, k - 1, 3 * h * d)), arr((1, h, d, d), "float32"))
        else:
            latent = arr((1, 1, envelope.depth, rank))
            packed = arr((1, 1, envelope.depth, 2 * index_dim + 1))
            desc = seq(
                seq(latent, latent),
                seq(packed, arr((1, 1, envelope.depth, 0), "float32")),
            )
        result.append(dict(layer=i, desc=desc))
    return result


def _require_state_meta(meta, expected_meta):
    if sorted(meta, key=lambda e: e["layer"]) != sorted(
        expected_meta, key=lambda e: e["layer"]
    ):
        raise ValueError("pipeline cache schema dtype/shape mismatch")


def _walk_arrays(desc, fn, out):
    if desc["k"] == "arr":
        out.append(fn(desc))
    elif desc["k"] == "seq":
        for d in desc["items"]:
            _walk_arrays(d, fn, out)
    return out


def handoff_recv(sock, rebuild: bool = False, *, expected, expected_meta):
    """Receive stage B's caches. ``rebuild`` materializes them for decode."""
    msg = _recv_json(sock)
    if msg.get("cmd") != "handoff":
        raise ValueError("expected pipeline handoff")
    PrefillEnvelope.from_dict(msg.get("envelope")).require_match(expected)
    _validate_meta(msg.get("meta"), expected)
    _require_state_meta(msg["meta"], expected_meta)
    total = 0
    wire_s = rebuild_s = 0.0
    states = {}
    t0 = time.perf_counter()
    for ent in msg["meta"]:
        descs = _walk_arrays(ent["desc"], lambda d: d, [])
        arrays = []
        for d in descs:
            n = d["nbytes"]
            dt = _DTYPES[d["dtype"]]
            if n == 0:
                arrays.append(
                    mx.zeros(tuple(d["shape"]), dtype=dt) if rebuild else None
                )
                continue
            buf = bytearray(n)
            t_wire = time.perf_counter()
            _recv_exact(sock, memoryview(buf), n)
            wire_s += time.perf_counter() - t_wire
            total += n
            if rebuild:
                t_rebuild = time.perf_counter()
                flat = mx.array(np.frombuffer(buf, dtype=np.uint8))
                arrays.append(flat.view(dt).reshape(tuple(d["shape"])))
                rebuild_s += time.perf_counter() - t_rebuild
            else:
                arrays.append(None)
        if rebuild:
            states[ent["layer"]] = rebuild_state(ent["desc"], iter(arrays))
    dt = time.perf_counter() - t0
    return {
        "handoff_wire_recv_s": wire_s,
        "handoff_rebuild_s": rebuild_s,
        "handoff_recv_s": dt,
        "handoff_bytes": total,
        "handoff_tensors": sum(
            len(_walk_arrays(e["desc"], lambda d: d, [])) for e in msg["meta"]
        ),
        "handoff_MB_per_s": (total / 2**20) / dt if dt else None,
        "states": states if rebuild else None,
    }


def install_state(caches, states, *, fresh_caches, expected, expected_meta):
    """Validate into fresh caches, then atomically replace only tail entries.

    Never reuse indexer auxiliary pools, padding metadata, or offsets from a
    prior request. Unsupported cache types fail closed; this is cold B=1 only.
    """
    from .models.cache import ArraysCache, CacheList, KVCache

    layers = list(range(expected.split, expected.n_layers))
    if (
        len(caches) != expected.n_layers
        or len(fresh_caches) != expected.n_layers
        or sorted(states) != layers
    ):
        raise ValueError("pipeline install layer mismatch")
    _require_state_meta(
        [dict(layer=i, desc=describe_state(st)) for i, st in states.items()],
        expected_meta,
    )

    def cache_nodes(c):
        return [c] + (
            [n for child in c.caches for n in cache_nodes(child)]
            if type(c) is CacheList
            else []
        )

    old_ids = {id(n) for c in caches if c is not None for n in cache_nodes(c)}
    if any(id(n) in old_ids for i in layers for n in cache_nodes(fresh_caches[i])):
        raise ValueError("pipeline install requires fresh unaliased caches")

    def prepare(cache, state):
        if not cache.empty() or any(
            getattr(cache, attr, None) is not None
            for attr in ("left_padding", "lengths")
        ):
            raise ValueError("pipeline install requires empty unpadded caches")
        if type(cache) is CacheList:
            if not isinstance(state, (list, tuple)) or len(state) != len(cache.caches):
                raise ValueError("pipeline cache structure mismatch")
            for child, value in zip(cache.caches, state):
                prepare(child, value)
        elif type(cache) is KVCache:
            if (
                not isinstance(state, (list, tuple))
                or len(state) != 2
                or any(
                    not isinstance(a, mx.array)
                    or a.ndim != 4
                    or a.shape[0] != 1
                    or a.shape[-2] != expected.depth
                    for a in state
                )
            ):
                raise ValueError("pipeline KV state depth/shape mismatch")
            cache.state = state
        elif type(cache) is ArraysCache:
            if (
                not isinstance(state, (list, tuple))
                or len(state) != len(cache.cache)
                or any(
                    not isinstance(a, mx.array) or a.ndim < 2 or a.shape[0] != 1
                    for a in state
                )
            ):
                raise ValueError("pipeline recurrent state shape mismatch")
            cache.state = state
        else:
            raise ValueError("unsupported pipeline cache type")
        for attr in ("_pool", "_fpool", "_no_pad"):
            if hasattr(cache, attr):
                delattr(cache, attr)

    for i in layers:
        if fresh_caches[i] is caches[i]:
            raise ValueError("pipeline install requires fresh caches")
        prepare(fresh_caches[i], states[i])
    # Evaluate before committing replacement so deferred rebuild errors cannot
    # leave a partially installed destination.
    mx.eval([fresh_caches[i].state for i in layers])
    for i in layers:
        caches[i] = fresh_caches[i]


# ------------------------------------------------------------------- roles


def make_prompt(tokens: int, seed: int, vocab: int = 150000) -> mx.array:
    rng = np.random.default_rng(seed)
    ids = rng.integers(low=1000, high=vocab, size=(1, tokens), dtype=np.int64)
    return mx.array(ids.astype(np.int32))


def run_head(args):
    _hex(args.model_sha256, 64, "model_sha256")
    _hex(args.source_revision, 40, "source_revision")
    lo, hi = 0, args.split
    model, caches, local, n_layers, load_s = load_stage(args.model, lo, hi, args.prune)
    stage = Stage(model, caches, local, n_layers)
    print(f"[head] layers {lo}:{hi} of {n_layers} loaded in {load_s:.1f}s", flush=True)

    sock = socket.socket()
    sock.settimeout(args.io_timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    deadline = time.time() + args.connect_timeout
    while True:
        try:
            sock.connect((args.peer, args.port))
            break
        except OSError as e:
            if time.time() > deadline:
                raise
            time.sleep(2.0)
    print(f"[head] connected to {args.peer}:{args.port}", flush=True)
    _send_json(
        sock,
        {
            "cmd": "hello",
            "transport": args.transport,
            "schema": 1,
            "split": args.split,
            "n_layers": n_layers,
            "model_sha256": args.model_sha256,
            "source_revision": args.source_revision,
        },
    )
    hi = _recv_json(sock)
    _check_peer_identity(
        hi, args.model_sha256, args.source_revision, args.split, n_layers
    )
    if args.transport == "ring":
        g = ring_group()
        print(f"[head] ring rank {g.rank()}/{g.size()}", flush=True)

    results = []
    for tokens in args.tokens:
        res = _head_one(args, stage, sock, tokens)
        results.append(res)
        print(json.dumps(res), flush=True)
        # fresh caches for the next length
        _reset_caches(stage, args.model)
    _send_json(sock, {"cmd": "bye"})
    if _recv_json(sock) != {"cmd": "bye", "ok": True}:
        raise ValueError("pipeline bye not acknowledged")
    sock.close()
    out = {"role": "head", "split": args.split, "load_s": load_s, "runs": results}
    _dump(args, out)


def _reset_caches(stage: Stage, model_path: str = ""):
    # model.make_cache() walks model.layers, which now has None holes after
    # pruning, so rebuild the local entries by hand.
    from .models.cache import ArraysCache, CacheList, KVCache

    new = [None] * stage.n_layers
    for i in stage.local:
        layer = stage.lm.layers[i]
        new[i] = (
            ArraysCache(size=2) if layer.is_linear else CacheList(KVCache(), KVCache())
        )
    stage.caches = new
    gc.collect()
    mx.clear_cache()


def _head_one(args, stage: Stage, sock, tokens: int):
    chunk = args.chunk
    prompt = make_prompt(tokens, args.seed)
    envelope = PrefillEnvelope.create(
        model_sha256=args.model_sha256,
        source_revision=args.source_revision,
        split=args.split,
        n_layers=stage.n_layers,
        input_ids=prompt,
        chunk=chunk,
    )
    n_chunks = (tokens + chunk - 1) // chunk
    _send_json(
        sock,
        {
            "cmd": "run",
            "envelope": envelope.to_dict(),
            "tokens": tokens,
            "chunk": chunk,
            "split": args.split,
            "n_chunks": n_chunks,
            "handoff": bool(args.handoff),
            "transport": args.transport,
        },
    )
    ack = _recv_json(sock)
    if not ack.get("ok") or ack.get("request_id") != envelope.request_id:
        raise ValueError("pipeline run not acknowledged")

    sendq: "queue.Queue" = queue.Queue(maxsize=args.depth)
    send_times = []
    err = []

    def sender():
        try:
            # MLX streams are thread-local: the comm stream must be made here.
            stream = mx.new_stream(mx.cpu) if args.transport == "ring" else None
            while True:
                item = sendq.get(timeout=args.io_timeout)
                if item is None:
                    sock.sendall(HDR.pack(MAGIC, EOF_IDX, 0, 0, 0, 0, 0))
                    return
                idx, keep, nb = item
                B, S, HC, D = keep.shape
                t0 = time.perf_counter()
                # The shape header always rides the control socket (16 bytes,
                # sub-ms) so chunk shapes need not be predicted by the peer.
                nbytes = 0 if nb is None else nb.nbytes
                sock.sendall(HDR.pack(MAGIC, idx, B, S, HC, D, nbytes))
                if args.transport == "ring":
                    ring_send(keep, 1, stream)
                else:
                    sock.sendall(memoryview(nb).cast("B"))
                send_times.append(time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            err.append(repr(e))

    th = threading.Thread(target=sender, daemon=True)
    th.start()

    per_chunk = []
    t_start = time.perf_counter()
    pos = 0
    for idx in range(n_chunks):
        n = min(chunk, tokens - pos)
        ids = prompt[:, pos : pos + n]
        t0 = time.perf_counter()
        h = stage(None, inputs=ids)
        mx.eval(h)
        stage.eval_state()
        t_gpu = time.perf_counter() - t0
        nb = None if args.transport == "ring" else to_wire(h)
        t1 = time.perf_counter()
        _queue_put(sendq, (idx, h, nb), err, args.io_timeout)
        t_block = time.perf_counter() - t1
        per_chunk.append({"idx": idx, "n": n, "gpu_s": t_gpu, "block_s": t_block})
        pos += n
        mx.clear_cache()
    t_head_done = time.perf_counter() - t_start
    _queue_put(sendq, None, err, args.io_timeout)
    th.join(timeout=args.io_timeout)
    if th.is_alive():
        _abort_socket(sock)
        raise TimeoutError("pipeline sender did not retire")
    if err:
        raise RuntimeError(err[0])

    # tail signals "last chunk retired" before any optional handoff so the
    # pipeline wall clock is not polluted by the stretch-goal transfer
    done = _recv_json(sock)
    PrefillEnvelope.from_dict(done.get("envelope")).require_match(envelope)
    t_total = time.perf_counter() - t_start
    handoff = (
        handoff_recv(
            sock,
            expected=envelope,
            expected_meta=expected_state_meta(stage.model, envelope),
        )
        if args.handoff
        else None
    )
    if handoff is not None:
        handoff.pop("states", None)
    tail = _recv_json(sock)

    hc = stage.hc_mult
    D = stage.lm.layers[stage.local[0]].input_layernorm.weight.shape[0]
    return {
        "tokens": tokens,
        "chunk": chunk,
        "n_chunks": n_chunks,
        "split": args.split,
        "head_gpu_s": sum(c["gpu_s"] for c in per_chunk),
        "head_block_s": sum(c["block_s"] for c in per_chunk),
        "head_done_s": t_head_done,
        "total_s": t_total,
        "tok_per_s": tokens / t_total,
        "transport": args.transport,
        "wire_bytes_per_chunk": boundary_bytes(chunk, hc, D),
        "wire_send_s": sum(send_times),
        "wire_send_each": send_times,
        "head_chunks": per_chunk,
        "handoff": handoff,
        "tail": tail,
    }


class TailDaemon:
    """A resident accept loop for the tail stage.

    The bench tail served exactly one connection: ``listen(1)``, one
    ``accept()``, ``break``, and a ``--connect-timeout`` deadline that killed
    the process if a head was late.  A production tail holds tens of GB of
    pruned weights, so tearing it down between requests is the expensive part
    of the whole rig.  This class keeps the stage resident and re-arms
    ``accept`` after every connection, whatever ended it -- ``bye``, EOF, or a
    peer that died mid-chunk.  A failed connection is a connection-scoped
    event: it is logged, counted, and the next head is served.

    Shutdown is explicit and unloads BEFORE the process exits: a bare SIGTERM
    to a process holding MLX buffers leaks wired memory that only a reboot
    reclaims, so the signal only sets the flag and the loop's own ``finally``
    releases the stage.  A2b: the flag lives in a :class:`ShutdownGate` that
    the SESSION and every socket wait share, because the accept loop is
    exactly where a tail with a pooled head never is -- see the gate's
    docstring for the eight minutes of post-SIGTERM serving that proved it.
    """

    POLL_S = 0.25

    def __init__(
        self,
        srv,
        session,
        *,
        connect_timeout: float = 0.0,
        idle_timeout: float = 0.0,
        once: bool = False,
        unload=None,
        log=None,
        clock=time.monotonic,
        admission_report=None,
        sampler=None,
        gate=None,
        drain_timeout: float = DEFAULT_DRAIN_TIMEOUT_S,
    ):
        self.srv = srv
        self.session = session
        self.connect_timeout = float(connect_timeout or 0.0)
        self.idle_timeout = float(idle_timeout or 0.0)
        self.once = bool(once)
        self.unload = unload
        self.log = log or (lambda line: print(line, flush=True))
        self.clock = clock
        # A7: what the start-time gates decided, and the live rail verdict.
        # Both ride the health line so an operator (and, via ``ping``, the head)
        # never has to guess why a tail is up or whether it is worth using.
        self.admission_report = admission_report
        self.sampler = sampler
        # A2b: one stop verdict for the whole tail.  The session and the
        # per-request sockets hold the SAME gate, so a signal is observed
        # wherever the daemon happens to be, not only between connections.
        self.gate = (
            gate
            if gate is not None
            else ShutdownGate(drain_timeout=drain_timeout, clock=clock)
        )
        self.state = "starting"
        self.peer = None
        self.shutdown_reason = None
        self.started = clock()
        self.last_active = self.started
        self.counters = {
            "connections": 0,
            "requests": 0,
            "connection_errors": 0,
            "last_error": None,
        }

    # -- control ------------------------------------------------------------
    def request_shutdown(self, reason: str = "shutdown") -> str:
        """Ask the daemon to stop.  Safe from a signal handler (flags only).

        Returns ``"soft"`` (drain the request in flight) or ``"hard"`` (a
        second ask: abort it now).  Never does I/O -- a print here can deadlock
        on the stdout lock the interrupted thread already holds.
        """
        if self.shutdown_reason is None:
            self.shutdown_reason = reason
        return self.gate.request(reason)

    @property
    def stopping(self) -> bool:
        return self.gate.stopping

    def status(self) -> dict:
        """The health line.  Schema (every key always present):

        ``role`` str, ``state`` one of starting/listening/serving/stopping/
        stopped, ``peer`` str|null, ``uptime_s`` float, ``idle_s`` float,
        ``shutdown_reason`` str|null, ``connections`` int, ``requests`` int,
        ``connection_errors`` int, ``last_error`` str|null, ``degraded`` bool,
        ``rail`` object|null (:meth:`RailSampler.snapshot`), ``admission``
        object|null (:meth:`AdmissionReport.to_dict`), and A2b's four stop
        fields: ``stopping`` bool, ``hard_stop`` bool, ``drain_timeout_s``
        float, ``drain_remaining_s`` float|null.

        INVARIANT (A2b, and the shape the B3 drills caught): a line with a
        ``shutdown_reason`` is never in state ``listening``.  A tail that has
        been told to stop is either draining a request (``serving``), on its
        way out (``stopping``), or gone (``stopped``) -- never back at accept.
        """
        return {
            "role": "tail",
            "state": self.state,
            "peer": self.peer,
            "uptime_s": round(self.clock() - self.started, 3),
            "idle_s": round(self.clock() - self.last_active, 3),
            "shutdown_reason": self.shutdown_reason,
            **self.gate.snapshot(),
            "degraded": self.degraded,
            "rail": self.sampler.snapshot() if self.sampler is not None else None,
            "admission": (
                self.admission_report.to_dict()
                if self.admission_report is not None
                else None
            ),
            **self.counters,
        }

    @property
    def degraded(self) -> bool:
        return bool(self.sampler is not None and self.sampler.degraded)

    def _emit_status(self):
        self.log("[tail-health] " + json.dumps(self.status()))

    # -- loop ---------------------------------------------------------------
    def serve_forever(self):
        self.srv.settimeout(self.POLL_S)
        self.state = "listening"
        self._emit_status()
        try:
            while not self.gate.stopping:
                conn = self._accept()
                if conn is None:
                    self._check_deadlines()
                    continue
                sock, addr = conn
                self.counters["connections"] += 1
                self.peer = str(addr)
                self.state = "serving"
                self.last_active = self.clock()
                self._emit_status()
                try:
                    served = self.session(sock, addr)
                    self.counters["requests"] += int(served or 0)
                except BaseException as exc:  # noqa: BLE001
                    # One head's bad day is not the daemon's: count it, name
                    # it, re-arm.  KeyboardInterrupt still stops the service.
                    if isinstance(exc, InterruptedError) and self.gate.stopping:
                        # A2b: our own shutdown ending an IDLE connection is
                        # not the peer failing.  It is still recorded -- an
                        # operator reading the health line wants to know how
                        # the connection ended -- but it does not inflate the
                        # error counter a rail verdict is read from.  A request
                        # actually cut by the drain deadline does not come
                        # through here: it surfaces as the RuntimeError the
                        # receiver thread's error is re-raised as, and counts.
                        self.counters["last_error"] = repr(exc)
                        self.log(f"[tail] connection retired by shutdown: {exc}")
                    else:
                        self.counters["connection_errors"] += 1
                        self.counters["last_error"] = repr(exc)
                        self.log(f"[tail] connection error: {exc!r}")
                    if isinstance(exc, KeyboardInterrupt):
                        self.request_shutdown("interrupt")
                finally:
                    _close_quietly(sock)
                    self.peer = None
                    self.last_active = self.clock()
                    # Never "listening" while a stop is pending: that line is
                    # the one an operator read as "SIGTERM did nothing".
                    self.state = "stopping" if self.gate.stopping else "listening"
                    self._emit_status()
                if self.once:
                    self.request_shutdown("once")
        finally:
            self.state = "stopping"
            if self.shutdown_reason is None:
                self.shutdown_reason = "loop_exit"
            if self.gate.hard:
                self.log(
                    "[tail] hard stop: the request in flight was aborted "
                    "(second signal)"
                )
            self._emit_status()
            try:
                self.srv.close()
            except OSError:
                pass
            if self.unload is not None:
                self.unload()
            self.state = "stopped"
            self._emit_status()
        return self.status()

    def _accept(self):
        try:
            sock, addr = self.srv.accept()
        except (socket.timeout, TimeoutError):
            return None
        except OSError as exc:
            if self.gate.stopping:
                return None
            raise exc
        return sock, addr

    def _check_deadlines(self):
        now = self.clock()
        if (
            self.connect_timeout > 0
            and self.counters["connections"] == 0
            and now - self.started >= self.connect_timeout
        ):
            # Only the FIRST connection can time out, and only when the
            # operator asked for a deadline; a resident tail that has served a
            # head never dies of a quiet hour.
            raise TimeoutError("pipeline tail accept timeout")
        if self.idle_timeout > 0 and now - self.last_active >= self.idle_timeout:
            self.request_shutdown("idle_timeout")


def _close_quietly(sock):
    if sock is None:
        return
    try:
        _abort_socket(sock)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def install_tail_signal_handlers(daemon, signums=(signal.SIGTERM, signal.SIGINT)):
    """SIGTERM asks; the loop unloads.  Never let the runtime be killed while
    it owns MLX buffers -- a bare SIGTERM there leaks wired memory that only a
    reboot reclaims.  Returns the previous handlers so a caller can restore.

    A2b, what the signal now MEANS in each state (the CLI help says the same):

    * idle, whether between connections or holding a pooled one -- the daemon
      exits within ~1 s, having unloaded the stage;
    * inside a request -- the request keeps running and is answered if it
      finishes within ``--drain-timeout`` (default 30 s), then the connection
      retires and the daemon exits; past the drain the connection is aborted
      and the head falls back to single-box (``pp_failed``);
    * a SECOND signal -- abort the request in flight immediately, then unload.
      Still not a kill: the unload always runs, because the wired memory of a
      SIGKILLed MLX process is only reclaimed by a reboot.
    """
    previous = {}
    for num in signums:
        def _handler(signo, frame, _daemon=daemon):
            # Flags only.  Logging here would take the stdout lock that the
            # interrupted thread may already hold; the loop logs instead.
            _daemon.request_shutdown(f"signal_{signo}")

        try:
            previous[num] = signal.signal(num, _handler)
        except ValueError:
            # not the main thread: the caller is embedding us, and owns signals
            pass
    return previous


class HealthServer:
    """The health socket and its thread, so a caller can actually join it.

    A2b: the loop used to stop accepting the instant ``daemon.stopping``
    flipped, which made "SIGTERM was received" and "the tail died" look the
    same to a probe -- and left the thread's lifetime unobservable.  It now
    answers until the daemon has reached ``stopped``, so the last thing a probe
    can read is the shutdown itself, and the owner closes and joins it.
    """

    def __init__(self, sock):
        self.socket = sock
        self.thread = None
        self.closed = False

    def fileno(self):
        return self.socket.fileno()

    def getsockname(self):
        return self.socket.getsockname()

    def close(self):
        """Idempotent; the loop wakes on the closed descriptor or its 0.25 s
        poll, whichever is first."""
        self.closed = True
        try:
            self.socket.close()
        except OSError:
            pass

    def join(self, timeout=None) -> bool:
        if self.thread is None:
            return True
        self.thread.join(timeout)
        return not self.thread.is_alive()


def serve_health(daemon, port: int, bind: str = "127.0.0.1"):
    """One line of JSON per connection, then close.  A health check must not
    be able to wedge the service, so it never reads from the client."""
    if not port:
        return None
    hs = socket.socket()
    hs.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    hs.bind((bind, int(port)))
    hs.listen(8)
    hs.settimeout(0.25)
    handle = HealthServer(hs)

    def loop():
        while not handle.closed and daemon.state != "stopped":
            try:
                c, _ = hs.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                break
            try:
                c.sendall((json.dumps(daemon.status()) + "\n").encode())
            except OSError:
                pass
            finally:
                _close_quietly(c)
        handle.close()

    th = threading.Thread(target=loop, daemon=True, name="tail-health")
    handle.thread = th
    th.start()
    return handle


def tail_session_factory(args, stage, n_layers, load_s, stop, sampler=None):
    """Build the per-connection handler: hello, then run/bye until the peer
    leaves.  Returns the number of requests the connection served.

    ``stop`` is A2b's :class:`ShutdownGate` -- or, for callers that predate it,
    the bare stop-file path.  The connection, not the request, is what a stop
    retires: with a pooled head this loop is where the daemon spends its life
    (``pipeline_runtime.PipelinePool`` keeps one socket across requests), so it
    asks the gate before every command it waits for, and again as soon as the
    request in flight has been answered.
    """
    gate = stop if isinstance(stop, ShutdownGate) else None

    def _retiring(g):
        """A stop that arrived while this connection was IDLE: nothing is in
        flight, so there is nothing to drain and nothing to abort."""
        return g is not None and g.stopping and not g.in_request

    def session(raw_sock, addr):
        served = 0
        raw_sock.settimeout(args.io_timeout)
        raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock = StopAwareSocket(raw_sock, stop, args.io_timeout)
        print(f"[tail] peer {addr}", flush=True)
        hello = _recv_json(sock)
        if hello.get("cmd") != "hello" or hello.get("transport") not in (
            "socket",
            "ring",
        ):
            raise ValueError("invalid pipeline hello")
        identity = _check_peer_identity(
            hello, args.model_sha256, args.source_revision, args.split, n_layers
        )
        args.transport = hello["transport"]
        _send_json(sock, {"ok": True, "load_s": load_s, **identity})
        if args.transport == "ring":
            ring_group()
        try:
            while True:
                # Idle between requests is a state a stop must be able to end:
                # a pooled head can leave this connection open for hours, and
                # before A2b that was the state a signal was never observed in.
                # The gate answers here and inside the recv poll below (<=
                # 0.25 s); a stop-file path keeps its old meaning exactly.
                if _retiring(gate):
                    print(
                        "[tail] shutdown: retiring the idle connection",
                        flush=True,
                    )
                    return served
                _check_stop(stop)
                try:
                    req = _recv_json(sock)
                except InterruptedError:
                    # The stop landed while we were blocked on the head.  That
                    # is a clean end of a connection, not a failed one: the
                    # requests this connection did serve are still counted and
                    # the daemon logs a retirement rather than an error.
                    if not _retiring(gate):
                        raise
                    print(
                        "[tail] shutdown: retiring the idle connection",
                        flush=True,
                    )
                    return served
                if req.get("cmd") == "bye":
                    _send_json(sock, {"cmd": "bye", "ok": True})
                    return served
                if req.get("cmd") == "ping":
                    # Liveness for a pooled head: the connection outlives the
                    # request now, so the head must be able to ask whether it
                    # still has a peer before it commits a prefill to it.
                    #
                    # A7: a head that ASKS gets the tail's own rail verdict
                    # back with the ack.  ``degraded`` true means the tail is
                    # alive and still serving, but its recent p95 wire time is
                    # past the bound -- the head should feed that to its
                    # CircuitBreaker (record_failure) and prefill single-box
                    # rather than commit a request to a rail it already knows is
                    # slow.  ``rail`` is the evidence behind the flag; see
                    # pipeline_admission.RailSampler.snapshot for the fields.
                    #
                    # Opt-in, not unconditional: the shipped head compares the
                    # ack for EQUALITY with {"cmd": "ping", "ok": True}
                    # (pipeline_runtime.PipelineHead.ping), so an extra key on
                    # every reply would break every head that has not been
                    # updated yet.  The head side of this is a one-line change:
                    # send {"cmd": "ping", "rail": True} and read ``degraded``.
                    ack = {"cmd": "ping", "ok": True}
                    if req.get("rail"):
                        ack["degraded"] = bool(
                            sampler is not None and sampler.degraded
                        )
                        ack["rail"] = sampler.snapshot() if sampler else None
                    _send_json(sock, ack)
                    continue
                if req.get("cmd") != "run" or req.get("transport") != args.transport:
                    raise ValueError("invalid pipeline run")
                envelope = PrefillEnvelope.from_dict(req.get("envelope"))
                _check_peer_identity(
                    envelope.to_dict(),
                    args.model_sha256,
                    args.source_revision,
                    args.split,
                    n_layers,
                )
                # A6.  The capture is negotiated BEFORE the ack, where a refusal
                # is free: the head has not sent a chunk yet, so it can prefill
                # the whole prompt on one box for the price of this round trip.
                # A refusal that arrived at ``done`` instead would have cost the
                # request its entire pipelined prefill.  Named, and not a raised
                # connection error, so the head can put the reason in its
                # bypass histogram and this connection survives to serve the
                # next request.
                capture = None
                if req.get("capture") is not None:
                    try:
                        capture = CaptureSpec.parse(
                            req["capture"], n_layers=n_layers
                        )
                    except CaptureUnsupported as exc:
                        _send_json(
                            sock,
                            {
                                "ok": False,
                                "error": "capture_unsupported",
                                "detail": str(exc),
                                "request_id": envelope.request_id,
                            },
                        )
                        continue
                _check_stop(stop)
                # Reset BEFORE acknowledging ownership of every request,
                # including the first/no-prune request. No stale head layers
                # enter handoff.
                _reset_caches(stage, args.model)
                if gate is not None:
                    # From the ack to the reply this connection owns a request:
                    # a stop DRAINS it (up to --drain-timeout) instead of
                    # cutting it, and a shutdown that lands while the reply is
                    # being written cannot throw the finished work away.
                    gate.enter_request()
                try:
                    _send_json(
                        sock, {"ok": True, "request_id": envelope.request_id}
                    )
                    rep = _tail_one(args, stage, sock, req, capture)
                    served += 1
                    if sampler is not None:
                        # Sample AFTER the work and BEFORE the reply, so the
                        # next ping already reflects the request that just ran.
                        sampler.observe(
                            rep.get("wire_recv_s"), rep.get("tail_total_s")
                        )
                    _send_json(sock, rep)
                finally:
                    if gate is not None:
                        gate.leave_request()
                print(json.dumps(rep), flush=True)
                if gate is not None and gate.stopping:
                    # Graceful: the request that was in flight when the signal
                    # arrived is finished and answered.  The connection retires
                    # here -- the head's next ping fails, so it reconnects or
                    # bypasses with peer_unreachable -- and the accept loop
                    # sees the flag on its very next turn.
                    print(
                        "[tail] drained the request in flight; retiring the "
                        "connection and shutting down",
                        flush=True,
                    )
                    return served
        finally:
            # Whatever ended this connection, the next head gets an empty
            # stage: a half-populated cache must never be reachable by a
            # request that did not fill it.
            _reset_caches(stage, args.model)

    return session


def run_tail(args, on_ready=None):
    # Validate pins before loading any weights. Identity is supplied by the
    # caller's verified model manifest; no path-name equivalence is inferred.
    _hex(args.model_sha256, 64, "model_sha256")
    _hex(args.source_revision, 40, "source_revision")
    stop_file = getattr(args, "stop_file", None)
    _check_stop(stop_file)
    # A7 service-start admission. Every gate here answers a question that stops
    # being answerable the moment weights are resident, so all three run first
    # and a refusal costs nothing but a process start: one tail per box (flock),
    # one heavy model per box (process scan), and the registered wired budget.
    # Off by default for programmatic callers; ``main()`` turns it on for the
    # service, and MLX_VLM_PIPELINE_ADMISSION=0 turns it back off.
    report = None
    if _admission_enabled(args):
        report = admission.admit(args)
        print("[tail-admission] " + json.dumps(report.to_dict()), flush=True)
    try:
        return _run_tail_admitted(args, stop_file, report, on_ready)
    except BaseException:
        if report is not None:
            report.release()
        raise


def _admission_enabled(args) -> bool:
    value = getattr(args, "admission", None)
    if value is None:
        value = os.environ.get("MLX_VLM_PIPELINE_ADMISSION", "0")
    return str(value).lower() not in ("0", "false", "no", "none", "")


def _run_tail_admitted(args, stop_file, report, on_ready):
    # A2b.  One gate for the process: the signal handlers, the accept loop, the
    # session and every socket poll read the same flags, and the drain window
    # starts the moment the first signal lands.
    drain = getattr(args, "drain_timeout", None)
    gate = ShutdownGate(
        stop_file, DEFAULT_DRAIN_TIMEOUT_S if drain is None else drain
    )
    lo, hi = args.split, args.layers
    model, caches, local, n_layers, load_s = load_stage(args.model, lo, hi, args.prune)
    stage = Stage(model, caches, local, n_layers)
    print(f"[tail] layers {lo}:{hi} of {n_layers} loaded in {load_s:.1f}s", flush=True)

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(getattr(args, "backlog", 8))
    print(f"[tail] listening on {args.bind}:{args.port}", flush=True)

    holder = {"stage": stage, "model": model, "caches": caches}

    def unload():
        # No signals: the owner releases its local references and returns.
        # The session closure holds the Stage, and the Stage holds the weights,
        # so clearing this frame's names is not enough -- the Stage's own
        # references have to go or the pruned layers stay wired for as long as
        # the interpreter lives.
        st = holder.pop("stage", None)
        if st is not None:
            st.caches = []
            st.model = None
            st.lm = None
        holder.pop("model", None)
        holder.pop("caches", None)
        gc.collect()
        mx.clear_cache()

    def unload_and_unlock():
        # The lock is released only AFTER the weights are gone: a successor
        # that grabbed the lock the instant we set the flag would load its own
        # shard against ours and blow the box's wired budget.  The release is a
        # ``finally`` because a lock this process still holds after it has
        # stopped serving locks the box out of its own restart -- which is what
        # the B3 drills hit from the other side (a live-but-deaf tail refusing
        # every replacement).
        try:
            unload()
        finally:
            if report is not None:
                report.release()

    sampler = admission.rail_sampler_from_args(args)
    daemon = TailDaemon(
        srv,
        tail_session_factory(args, stage, n_layers, load_s, gate, sampler),
        connect_timeout=getattr(args, "connect_timeout", 0.0) or 0.0,
        idle_timeout=getattr(args, "idle_timeout", 0.0) or 0.0,
        once=bool(getattr(args, "once", False)),
        unload=unload_and_unlock,
        admission_report=report,
        sampler=sampler,
        gate=gate,
    )
    install_tail_signal_handlers(daemon)
    health = serve_health(daemon, getattr(args, "health_port", 0) or 0, bind=args.bind)
    if on_ready is not None:
        # An embedding caller (or a test) needs a handle on the running
        # service to shut it down; the daemon is built here, so it is handed
        # over here.
        on_ready(daemon)
    stage = None
    del caches, model
    try:
        return daemon.serve_forever()
    finally:
        # The health thread deliberately outlives the loop (it is what answers
        # the "stopping"/"stopped" line), but it must not outlive the process's
        # exit path: close its listener and join it here.
        if health is not None:
            health.close()
            if not health.join(2.0):
                print("[tail] health thread did not retire", flush=True)


def _tail_one(args, stage: Stage, sock, req, capture: Optional[CaptureSpec] = None):
    envelope = PrefillEnvelope.from_dict(req.get("envelope"))
    # A6.  This half's share of a hidden-reading drafter's capture: the layers
    # of ``capture.layers`` that live on THIS side of the split, or -- for MTP's
    # whole-hidden capture -- the pre-final-norm hidden after the last layer,
    # which is always here.  ``capture_ids`` is ascending, because the merge on
    # the head is a concatenation of the two halves' ordered lists and that is
    # only layer-id order if each half is ordered.
    capture_ids = []
    capture_window = None
    if capture is not None:
        local = set(stage.local)
        capture_ids = [i for i in capture.layers if i in local]
        if capture.kind == "hidden" or capture_ids:
            capture_window = TrailingHiddenWindow(capture.keep)
    recvq: "queue.Queue" = queue.Queue(maxsize=args.depth)
    recv_times = []
    err = []
    stop = threading.Event()
    # A2b: the same stop the socket polls, so a chunk boundary is a stop point
    # too -- ``sock`` carries the gate for the service, and the bench roles
    # keep their stop-file path.
    request_stop = getattr(sock, "stop", None)
    if request_stop is None:
        request_stop = getattr(args, "stop_file", None)
    timeout = args.io_timeout
    expected_hc = stage.hc_mult
    expected_d = stage.lm.layers[stage.local[0]].input_layernorm.weight.shape[0]

    def receiver():
        try:
            stream = mx.new_stream(mx.cpu) if args.transport == "ring" else None
            hdrbuf = bytearray(HDR.size)
            next_idx = 0
            while True:
                _recv_exact(sock, memoryview(hdrbuf), HDR.size)
                magic, idx, B, S, HC, D, nbytes = HDR.unpack(bytes(hdrbuf))
                if magic != MAGIC:
                    raise ValueError("pipeline boundary magic mismatch")
                if idx == EOF_IDX:
                    if next_idx != len(envelope.chunks) or any((B, S, HC, D, nbytes)):
                        raise ValueError("pipeline premature/invalid EOF")
                    _queue_put(recvq, None, err, timeout)
                    return
                if (
                    next_idx >= len(envelope.chunks)
                    or idx != next_idx
                    or B != 1
                    or S != envelope.chunks[idx]
                    or HC != expected_hc
                    or D != expected_d
                    or nbytes != (0 if args.transport == "ring" else B * S * HC * D * 2)
                ):
                    raise ValueError("pipeline boundary shape/chunk/bytes mismatch")
                next_idx += 1
                # header already arrived -> this times the payload transfer only
                t0 = time.perf_counter()
                if args.transport == "ring":
                    payload = ring_recv((B, S, HC, D), 0, stream)
                else:
                    buf = bytearray(nbytes)
                    _recv_exact(sock, memoryview(buf), nbytes)
                    payload = buf
                recv_times.append(time.perf_counter() - t0)
                if stop.is_set():
                    return
                _queue_put(recvq, (idx, payload, (B, S, HC, D)), err, timeout)
        except Exception as e:  # noqa: BLE001
            err.append(repr(e))
            _abort_socket(sock)
            try:
                recvq.put_nowait(None)
            except queue.Full:
                pass

    th = threading.Thread(target=receiver, daemon=True)
    th.start()

    per_chunk = []
    t_start = time.perf_counter()
    last_logits = None
    try:
        while True:
            _check_stop(request_stop)
            if err:
                raise RuntimeError(err[0])
            t0 = time.perf_counter()
            item = recvq.get(timeout=timeout)
            t_wait = time.perf_counter() - t0
            if item is None:
                break
            idx, buf, shape = item
            t1 = time.perf_counter()
            h = buf if isinstance(buf, mx.array) else from_wire(buf, shape)
            mx.eval(h)
            t_deser = time.perf_counter() - t1
            t2 = time.perf_counter()
            sink = [] if capture_window is not None else None
            out = stage(h, hidden_sink=sink, capture_layer_ids=capture_ids)
            # The capture is evaluated WITH the chunk, never later: an
            # unevaluated capture is a graph node that pins every intermediate
            # behind it, so a window of lazy pieces would hold the whole
            # prefill's activations instead of ``keep`` rows per layer.
            mx.eval(out if sink is None else [out, *sink])
            if capture_window is not None:
                capture_window.append(sink)
            stage.eval_state()
            t_gpu = time.perf_counter() - t2
            last_logits = out
            per_chunk.append(
                {
                    "idx": idx,
                    "n": shape[1],
                    "wait_s": t_wait,
                    "deser_s": t_deser,
                    "gpu_s": t_gpu,
                }
            )
            mx.clear_cache()
    except BaseException:
        stop.set()
        err.append("tail consumer aborted")
        _abort_socket(sock)
        raise
    finally:
        th.join(timeout=timeout)
    t_total = time.perf_counter() - t_start
    if th.is_alive():
        raise TimeoutError("pipeline receiver did not retire")
    if err:
        raise RuntimeError(err[0])

    tok = None
    if last_logits is not None:
        lg = stage.finish(last_logits)
        mx.eval(lg)
        tok = int(mx.argmax(lg[0, -1]).item())
    done = {"cmd": "done", "envelope": envelope.to_dict()}
    window = capture_window.window() if capture_window is not None else []
    cap = None
    if capture is not None:
        # ONE end-of-request frame, and it rides on ``done`` because there is
        # nothing else going back before it: between the run ack and here the
        # tail sends nothing at all, and the head does not read until finalize.
        # A per-chunk reply would be 3.22 GB on a 131k prompt for rows the very
        # next chunk makes unreachable.
        rows = sum(envelope.chunks)
        done["capture"] = capture_meta(capture, capture_ids, window, rows)
    _send_json(sock, done)
    cap_bytes = capture_send(sock, window) if window else 0
    window = None
    if capture is not None:
        cap = {"capture_bytes": cap_bytes, "capture_tensors": len(capture_ids)}
    ho = (
        handoff_send(sock, stage.caches, envelope=envelope)
        if req.get("handoff")
        else None
    )
    return {
        "envelope": envelope.to_dict(),
        "handoff": ho,
        "capture": cap,
        "tail_gpu_s": sum(c["gpu_s"] for c in per_chunk),
        "tail_wait_s": sum(c["wait_s"] for c in per_chunk),
        "tail_deser_s": sum(c["deser_s"] for c in per_chunk),
        "tail_total_s": t_total,
        "wire_recv_s": sum(recv_times),
        "wire_recv_each": recv_times,
        "argmax_token": tok,
        "tail_chunks": per_chunk,
    }


def run_single(args):
    """Same stage/chunk code, one process, layers [lo, hi) -- the honest baseline."""
    lo = args.lo
    hi = args.hi if args.hi is not None else args.layers
    model, caches, local, n_layers, load_s = load_stage(args.model, lo, hi, args.prune)
    stage = Stage(model, caches, local, n_layers)
    print(
        f"[single] layers {lo}:{hi} of {n_layers} loaded in {load_s:.1f}s", flush=True
    )

    results = []
    for tokens in args.tokens:
        prompt = make_prompt(tokens, args.seed)
        chunk = args.chunk
        n_chunks = (tokens + chunk - 1) // chunk
        per_chunk = []
        pos = 0
        h = None
        t_start = time.perf_counter()
        for idx in range(n_chunks):
            n = min(chunk, tokens - pos)
            t0 = time.perf_counter()
            if stage.is_head:
                h = stage(None, inputs=prompt[:, pos : pos + n])
            else:
                # feed a synthetic boundary tensor so a middle/tail slice can be
                # timed standalone (shape-accurate, values irrelevant to cost)
                D = stage.lm.layers[local[0]].input_layernorm.weight.shape[0]
                fake = mx.random.normal((1, n, stage.hc_mult, D)).astype(mx.bfloat16)
                h = stage(fake)
            mx.eval(h)
            stage.eval_state()
            per_chunk.append({"idx": idx, "n": n, "gpu_s": time.perf_counter() - t0})
            pos += n
            mx.clear_cache()
        tok = None
        if stage.is_tail:
            lg = stage.finish(h)
            mx.eval(lg)
            tok = int(mx.argmax(lg[0, -1]).item())
        t_total = time.perf_counter() - t_start
        res = {
            "tokens": tokens,
            "chunk": chunk,
            "lo": lo,
            "hi": hi,
            "total_s": t_total,
            "tok_per_s": tokens / t_total,
            "gpu_s": sum(c["gpu_s"] for c in per_chunk),
            "argmax_token": tok,
            "chunks": per_chunk,
        }
        results.append(res)
        print(json.dumps(res), flush=True)
        _reset_caches(stage, args.model)
    _dump(
        args, {"role": "single", "lo": lo, "hi": hi, "load_s": load_s, "runs": results}
    )


def _dump(args, obj):
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(obj, indent=2))
        print(f"wrote {args.out}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["head", "tail", "single"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--split", type=int, default=23, help="first layer of the tail stage"
    )
    p.add_argument("--layers", type=int, default=45)
    p.add_argument("--lo", type=int, default=0, help="single-role: first layer")
    p.add_argument(
        "--hi", type=int, default=None, help="single-role: end layer (exclusive)"
    )
    p.add_argument("--tokens", type=int, nargs="+", default=[8192])
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--peer", default="10.0.0.2")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=39200)
    p.add_argument("--depth", type=int, default=2, help="in-flight chunk queue depth")
    p.add_argument(
        "--transport",
        choices=["socket", "ring"],
        default="ring",
        help="boundary-tensor transport; 'socket' is the fallback",
    )
    p.add_argument(
        "--ring-hosts",
        default=None,
        help="comma separated ip:port per rank, e.g. 10.0.0.1:39400,10.0.0.2:39401",
    )
    p.add_argument(
        "--connect-timeout",
        type=float,
        default=0.0,
        help="deadline for the FIRST head only; 0 = wait forever (resident tail)",
    )
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        help="tail-role: shut down cleanly after this many idle seconds; 0 = never",
    )
    p.add_argument(
        "--drain-timeout",
        type=float,
        default=DEFAULT_DRAIN_TIMEOUT_S,
        help="tail-role: on SIGTERM/SIGINT, how long a request ALREADY IN "
        "FLIGHT may keep running before its connection is aborted (the head "
        "then prefills single-box and counts pp_failed). An idle tail exits "
        "within ~1s either way; a SECOND signal aborts the request "
        "immediately. Both paths unload the stage before exiting -- the "
        "process is never left to be killed while it owns MLX buffers "
        f"[default {DEFAULT_DRAIN_TIMEOUT_S:g}s, 0 = abort at once]",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="tail-role: serve a single connection then exit (the bench behaviour)",
    )
    p.add_argument(
        "--health-port",
        type=int,
        default=0,
        help="tail-role: one-line JSON status socket; 0 = stdout only",
    )
    # -- A7: service-start admission ------------------------------------
    p.add_argument(
        "--admission",
        dest="admission",
        action="store_true",
        default=os.environ.get("MLX_VLM_PIPELINE_ADMISSION", "1") not in ("0", ""),
        help="tail-role: run the start-time gates (flock, heavy-process "
        "preflight, wired budget) before loading weights [default on]",
    )
    p.add_argument(
        "--no-admission", dest="admission", action="store_false",
        help="skip the start-time gates entirely",
    )
    p.add_argument(
        "--lock-file",
        default=None,
        help="one-tail-per-box flock path "
        f"[default {admission.DEFAULT_LOCK_PATH}, env MLX_VLM_PIPELINE_LOCK]",
    )
    p.add_argument(
        "--no-lock", action="store_true",
        help="do not take the per-box tail lock (a deliberate second service)",
    )
    p.add_argument(
        "--allow-shared-box", action="store_true",
        help="admit even when another heavy MLX model process is resident",
    )
    p.add_argument(
        "--wired-cap-bytes",
        type=int,
        default=None,
        help="wired budget this box may reach after loading "
        f"[default {admission.REGISTERED_WIRED_POLICY['requested_limit_bytes']} "
        "(450 GiB, the registered run05 policy), env "
        "MLX_VLM_PIPELINE_WIRED_CAP_BYTES]",
    )
    p.add_argument(
        "--shard-bytes",
        type=int,
        default=None,
        help="override the measured stage footprint (env "
        "MLX_VLM_PIPELINE_SHARD_BYTES); default is read from the safetensors "
        "headers, no tensor data",
    )
    p.add_argument(
        "--allow-wired-overcommit", action="store_true",
        help="record the wired refusal but start anyway",
    )
    p.add_argument(
        "--rail-window", type=int, default=None,
        help=f"requests in the rail p95 window [default {admission.DEFAULT_RAIL_WINDOW}]",
    )
    p.add_argument(
        "--rail-p95-s", type=float, default=None,
        help="p95 wire seconds above which the tail reports degraded "
        f"[default {admission.DEFAULT_RAIL_P95_BOUND_S}]",
    )
    p.add_argument("--io-timeout", type=float, default=120.0)
    p.add_argument(
        "--model-sha256", default=os.environ.get("MLX_VLM_PIPELINE_MODEL_SHA256")
    )
    p.add_argument(
        "--source-revision", default=os.environ.get("MLX_VLM_PIPELINE_SOURCE_REVISION")
    )
    p.add_argument(
        "--stop-file",
        default=os.environ.get("MLX_VLM_PIPELINE_STOP_FILE"),
        help="tail-local cooperative cancellation file; checked at I/O/chunk boundaries",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--no-prune", dest="prune", action="store_false")
    p.add_argument(
        "--handoff",
        action="store_true",
        help="after prefill, ship stage B's caches back to stage A and time it",
    )
    args = p.parse_args(argv)
    if not math.isfinite(args.io_timeout) or args.io_timeout <= 0 or args.depth < 1:
        p.error("io-timeout and queue depth must be positive")
    if args.connect_timeout < 0 or args.idle_timeout < 0:
        p.error("connect-timeout and idle-timeout must be >= 0 (0 disables)")
    if not math.isfinite(args.drain_timeout) or args.drain_timeout < 0:
        p.error("drain-timeout must be a finite number of seconds >= 0")

    mx.random.seed(args.seed)
    if args.role != "single" and args.transport == "ring":
        setup_ring_env(args.ring_hosts, 0 if args.role == "head" else 1)
    if args.role == "head":
        run_head(args)
    elif args.role == "tail":
        try:
            run_tail(args)
        except admission.AdmissionRefused as exc:
            # Exit non-zero with the sentence, not a traceback: the caller of a
            # refused start is a launchd job or an operator, and neither is
            # served by a stack.
            print(f"[tail-admission] REFUSED {exc}", file=sys.stderr, flush=True)
            print(
                "[tail-admission] "
                + json.dumps({"admitted": False, "gate": exc.gate,
                              "detail": exc.detail}, default=str),
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(ADMISSION_REFUSED_EXIT)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
