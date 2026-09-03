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

from .models.base import create_attention_mask, create_ssm_mask
from .utils import load_model

MAGIC = b"GP51"
HDR = struct.Struct("!4sIIIIIQ")  # magic, chunk_idx, B, S, HC, D, nbytes
EOF_IDX = 0xFFFFFFFF
MAX_JSON_BYTES = 4 * 1024 * 1024


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


def _check_stop(path):
    if path and Path(path).exists():
        raise InterruptedError("pipeline cooperative STOP requested")


class StopAwareSocket:
    """Tail-owned stop-file checks during idle and transfer, without signals."""

    def __init__(self, sock, stop_file, timeout):
        self.sock, self.stop_file, self.timeout = sock, stop_file, timeout
        sock.settimeout(min(0.25, timeout))

    def gettimeout(self):
        return self.timeout

    def recv_into(self, view, n):
        deadline = time.monotonic() + self.timeout
        while True:
            _check_stop(self.stop_file)
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
            _check_stop(self.stop_file)
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

    def __call__(self, h: mx.array, inputs: Optional[mx.array] = None) -> mx.array:
        """Run this stage over one chunk (delegates to Glm5NextModel)."""
        return self.lm.pipeline_forward(
            h, self.caches, self.local[0], self.local[-1] + 1, inputs=inputs
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
    layers = model.language_model.model.layers
    if len(layers) != envelope.n_layers:
        raise ValueError("pipeline model layer count mismatch")
    for i in range(envelope.split, envelope.n_layers):
        layer = layers[i]
        if layer is None:
            # Prototype heads prune tail modules before loading weights, but
            # retain the validated architecture config needed for the schema.
            cfg = model.language_model.args
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


def run_tail(args):
    # Validate pins before loading any weights. Identity is supplied by the
    # caller's verified model manifest; no path-name equivalence is inferred.
    _hex(args.model_sha256, 64, "model_sha256")
    _hex(args.source_revision, 40, "source_revision")
    stop_file = getattr(args, "stop_file", None)
    _check_stop(stop_file)
    lo, hi = args.split, args.layers
    model, caches, local, n_layers, load_s = load_stage(args.model, lo, hi, args.prune)
    stage = Stage(model, caches, local, n_layers)
    print(f"[tail] layers {lo}:{hi} of {n_layers} loaded in {load_s:.1f}s", flush=True)

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"[tail] listening on {args.bind}:{args.port}", flush=True)
    srv.settimeout(min(0.25, args.connect_timeout))
    sock = None
    try:
        deadline = time.monotonic() + args.connect_timeout
        while True:
            _check_stop(stop_file)
            if time.monotonic() >= deadline:
                raise TimeoutError("pipeline tail accept timeout")
            try:
                sock, addr = srv.accept()
                break
            except socket.timeout:
                pass
        sock.settimeout(args.io_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock = StopAwareSocket(sock, stop_file, args.io_timeout)
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
        while True:
            req = _recv_json(sock)
            if req.get("cmd") == "bye":
                _send_json(sock, {"cmd": "bye", "ok": True})
                break
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
            _check_stop(stop_file)
            # Reset BEFORE acknowledging ownership of every request, including
            # the first/no-prune request. No stale head layers enter handoff.
            _reset_caches(stage, args.model)
            _send_json(sock, {"ok": True, "request_id": envelope.request_id})
            rep = _tail_one(args, stage, sock, req)
            _send_json(sock, rep)
            print(json.dumps(rep), flush=True)
    finally:
        if sock is not None:
            _abort_socket(sock)
            sock.close()
        srv.close()
        # No signals: the owner releases its local references and returns.
        stage.caches = []
        del stage, caches, model
        gc.collect()
        mx.clear_cache()


def _tail_one(args, stage: Stage, sock, req):
    envelope = PrefillEnvelope.from_dict(req.get("envelope"))
    recvq: "queue.Queue" = queue.Queue(maxsize=args.depth)
    recv_times = []
    err = []
    stop = threading.Event()
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
            _check_stop(getattr(args, "stop_file", None))
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
            out = stage(h)
            mx.eval(out)
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
    _send_json(sock, {"cmd": "done", "envelope": envelope.to_dict()})
    ho = (
        handoff_send(sock, stage.caches, envelope=envelope)
        if req.get("handoff")
        else None
    )
    return {
        "envelope": envelope.to_dict(),
        "handoff": ho,
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
    p.add_argument("--connect-timeout", type=float, default=1800.0)
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

    mx.random.seed(args.seed)
    if args.role != "single" and args.transport == "ring":
        setup_ring_env(args.ring_hosts, 0 if args.role == "head" else 1)
    if args.role == "head":
        run_head(args)
    elif args.role == "tail":
        run_tail(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
