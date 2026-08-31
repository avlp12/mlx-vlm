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

Transport, 32k, interleaved jaccl/ring/jaccl/ring in one quiet window (head
stage 50.51-50.65 s across all four, so conditions are matched)::

    transport   tok/s   wire_send/chunk   tail_wait   internal speedup
    jaccl       578.6        9.4 ms         2.63 s        1.844x
    ring        580.5       28.4 ms         2.83 s        1.843x

RDMA cuts the in-flight boundary transfer 3.0x and moves end-to-end throughput
by nothing, because the pipeline already hides the wire behind compute. That is
the expected result and the reason to adopt jaccl here is headroom, not speed:
the wire stops being the thing that would break first if the split moved or the
chunk shrank. Idle the three transports are much closer (jaccl 7.19, ring 10.9,
socket 12.0 ms) -- the gap only opens under prefill's memory-bandwidth pressure,
which the copy-based transports pay and DMA does not.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import ctypes
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx
import numpy as np

from .models.base import create_attention_mask, create_ssm_mask
from .utils import load_model

MAGIC = b"GP51"
HDR = struct.Struct("!4sIIIIIQ")  # magic, chunk_idx, B, S, HC, D, nbytes
EOF_IDX = 0xFFFFFFFF


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
    if prune:
        for i in range(n_layers):
            if i < lo or i >= hi:
                lm.layers[i] = None
                caches[i] = None
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
        self.fa_local = next((i for i in local if not self.lm.layers[i].is_linear), None)

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
    while got < n:
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


# ---------------------------------------------------------- fast-link transport

_LINK_GROUP = None
_LIBRDMA = "/usr/lib/librdma.dylib"


class _Gid(ctypes.Union):
    _fields_ = [("raw", ctypes.c_ubyte * 16)]


def rdma_ipv4_gids() -> dict:
    """{rdma device name: IPv4 address of its IPv4-mapped GID}.

    macOS exposes one RDMA device per Thunderbolt port (``rdma_en2..en7``) and
    each carries an IPv4-mapped GID derived from whatever IP is configured on
    the matching ``enN``.  These exist independently of the "Thunderbolt Bridge"
    network service -- verified with that service disabled on both boxes -- so
    jaccl rides the tbnet addresses that are already up, with no interface,
    bridge or service change.  jaccl refuses a port without such a GID
    ("[jaccl] No IPv4-mapped GID for this device").
    """
    out = {}
    try:
        lib = ctypes.CDLL(_LIBRDMA)
    except OSError:
        return out
    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ibv_get_device_name.restype = ctypes.c_char_p
    lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    lib.ibv_open_device.restype = ctypes.c_void_p
    lib.ibv_open_device.argtypes = [ctypes.c_void_p]
    lib.ibv_query_gid.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.c_int,
        ctypes.POINTER(_Gid),
    ]
    n = ctypes.c_int(0)
    lst = lib.ibv_get_device_list(ctypes.byref(n))
    if not lst:
        return out
    for i in range(n.value):
        name = lib.ibv_get_device_name(lst[i]).decode()
        ctx = lib.ibv_open_device(lst[i])
        if not ctx:
            continue
        for idx in range(8):
            g = _Gid()
            if lib.ibv_query_gid(ctx, 1, idx, ctypes.byref(g)) != 0:
                continue
            raw = bytes(g.raw)
            if raw[:10] == b"\x00" * 10 and raw[10:12] == b"\xff\xff":
                out[name] = socket.inet_ntoa(raw[12:16])
                break
    return out


def _peer_ips(spec: str):
    """Accept both "10.0.0.1,10.0.0.2" and the legacy "ip:port,ip:port"."""
    ips, ports = [], []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.partition(":")
        ips.append(host)
        ports.append(int(port) if port else None)
    return ips, ports


def setup_link_env(transport: str, peers: str, rank: int, jaccl_port: int = 39500,
                   rdma_dev: Optional[str] = None):
    """Fill in the env protocol mlx.launch would set, minus the launcher.

    The launcher insists on one python path for every host, which does not hold
    here (different users, different venvs), so each box is started on its own
    and only the env is shared.
    """
    if transport == "socket" or not peers:
        return transport
    ips, ports = _peer_ips(peers)
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    os.environ["MLX_RANK"] = str(rank)
    if transport == "jaccl":
        dev = rdma_dev or next(
            (d for d, ip in rdma_ipv4_gids().items() if ip == ips[rank]), None
        )
        if dev is None:
            print(
                f"[link] no RDMA device carries {ips[rank]} "
                f"({rdma_ipv4_gids()}); falling back to ring",
                flush=True,
            )
            transport = "ring"
        else:
            n = len(ips)
            # rdma[i][j] = device rank i uses to reach rank j; None on the
            # diagonal (mlx asserts "RDMA device of self should be null")
            table = [[None if i == j else dev for j in range(n)] for i in range(n)]
            path = tmp / "mlx_vlm_pipeline_ibv.json"
            path.write_text(json.dumps(table))
            os.environ["MLX_IBV_DEVICES"] = str(path)
            os.environ["MLX_JACCL_COORDINATOR"] = f"{ips[0]}:{jaccl_port}"
    if transport == "ring":
        pp = [p if p is not None else 39400 + i for i, p in enumerate(ports)]
        path = tmp / "mlx_vlm_pipeline_ring.json"
        path.write_text(json.dumps([[f"{h}:{p}"] for h, p in zip(ips, pp)]))
        os.environ["MLX_HOSTFILE"] = str(path)
    return transport


def link_group(backend: str = "ring"):
    """Lazily join the fast link. Deliberately not via mlx.launch (see above)."""
    global _LINK_GROUP
    if _LINK_GROUP is None:
        _LINK_GROUP = mx.distributed.init(backend=backend)
    return _LINK_GROUP


def link_send(h: mx.array, dst: int, stream):
    # Send/Recv have no GPU implementation -- they must run on a CPU stream, and
    # MLX streams are thread-local, so the stream is created inside the worker
    # thread that uses it.  The pipeline pays exactly one CPU<->GPU crossing per
    # chunk (~0.2 ms against a 2.4-4.6 s chunk), which is why this structure is
    # fine here and fatal for tensor parallelism.
    mx.eval(mx.distributed.send(h, dst, group=_LINK_GROUP, stream=stream))


def link_recv(shape, src: int, stream) -> mx.array:
    tmpl = mx.zeros(shape, dtype=mx.bfloat16)
    h = mx.distributed.recv_like(tmpl, src, group=_LINK_GROUP, stream=stream)
    mx.eval(h)
    return h


# back-compat aliases
setup_ring_env = setup_link_env
ring_group = link_group
ring_send = link_send
ring_recv = link_recv


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


def handoff_send(sock, caches):
    """Stretch goal: ship stage B's KDA/DSA caches back so decode can run on box A.

    Always on the control socket: measured 4.6 GB/s at 131k, and it is a
    one-shot cost (0.18% of a 131k prefill), so it is not worth the extra
    ring-template plumbing.
    """
    entries = collect_state(caches)
    mx.eval([a for _, _, arrs in entries for a in arrs])
    meta = [{"layer": i, "desc": d} for i, d, _ in entries]
    _send_json(sock, {"cmd": "handoff", "meta": meta})
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
        "handoff_send_s": dt,
        "handoff_bytes": total,
        "handoff_tensors": sum(len(a) for _, _, a in entries),
    }


_DTYPES = {
    n: getattr(mx, n)
    for n in ("bfloat16", "float16", "float32", "uint8", "uint16", "int32", "int64")
}


def _walk_arrays(desc, fn, out):
    if desc["k"] == "arr":
        out.append(fn(desc))
    elif desc["k"] == "seq":
        for d in desc["items"]:
            _walk_arrays(d, fn, out)
    return out


def handoff_recv(sock, rebuild: bool = False):
    """Receive stage B's caches. ``rebuild`` materializes them for decode."""
    msg = _recv_json(sock)
    assert msg.get("cmd") == "handoff", msg
    total = 0
    states = {}
    t0 = time.perf_counter()
    for ent in msg["meta"]:
        descs = _walk_arrays(ent["desc"], lambda d: d, [])
        arrays = []
        for d in descs:
            n = d["nbytes"]
            dt = _DTYPES[d["dtype"]]
            if n == 0:
                arrays.append(mx.zeros(tuple(d["shape"]), dtype=dt))
                continue
            buf = bytearray(n)
            _recv_exact(sock, memoryview(buf), n)
            total += n
            if rebuild:
                flat = mx.array(np.frombuffer(buf, dtype=np.uint8))
                arrays.append(flat.view(dt).reshape(tuple(d["shape"])))
            else:
                arrays.append(None)
        if rebuild:
            states[ent["layer"]] = rebuild_state(ent["desc"], iter(arrays))
    dt = time.perf_counter() - t0
    return {
        "handoff_recv_s": dt,
        "handoff_bytes": total,
        "handoff_tensors": sum(
            len(_walk_arrays(e["desc"], lambda d: d, [])) for e in msg["meta"]
        ),
        "handoff_MB_per_s": (total / 2**20) / dt if dt else None,
        "states": states if rebuild else None,
    }


def install_state(caches, states):
    """Install received stage-B state into the head's own cache list."""
    for i, st in states.items():
        caches[i].state = st


# ------------------------------------------------------------------- roles


def make_prompt(tokens: int, seed: int, vocab: int = 150000) -> mx.array:
    rng = np.random.default_rng(seed)
    ids = rng.integers(low=1000, high=vocab, size=(1, tokens), dtype=np.int64)
    return mx.array(ids.astype(np.int32))


def run_head(args):
    lo, hi = 0, args.split
    model, caches, local, n_layers, load_s = load_stage(args.model, lo, hi, args.prune)
    stage = Stage(model, caches, local, n_layers)
    print(f"[head] layers {lo}:{hi} of {n_layers} loaded in {load_s:.1f}s", flush=True)

    sock = socket.socket()
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
    _send_json(sock, {"cmd": "hello", "transport": args.transport, "split": args.split})
    hi = _recv_json(sock)
    assert hi.get("ok"), hi
    if args.transport != "socket":
        g = link_group(args.transport)
        print(f"[head] {args.transport} rank {g.rank()}/{g.size()}", flush=True)

    results = []
    for tokens in args.tokens:
        res = _head_one(args, stage, sock, tokens)
        results.append(res)
        print(json.dumps(res), flush=True)
        # fresh caches for the next length
        _reset_caches(stage, args.model)
    _send_json(sock, {"cmd": "bye"})
    out = {"role": "head", "split": args.split, "load_s": load_s, "runs": results}
    _dump(args, out)


def _reset_caches(stage: Stage, model_path: str = ""):
    # model.make_cache() walks model.layers, which now has None holes after
    # pruning, so rebuild the local entries by hand.
    from .models.cache import ArraysCache, CacheList, KVCache

    new = [None] * stage.n_layers
    for i in stage.local:
        layer = stage.lm.layers[i]
        new[i] = ArraysCache(size=2) if layer.is_linear else CacheList(KVCache(), KVCache())
    stage.caches = new
    gc.collect()
    mx.clear_cache()


def _head_one(args, stage: Stage, sock, tokens: int):
    chunk = args.chunk
    prompt = make_prompt(tokens, args.seed)
    n_chunks = (tokens + chunk - 1) // chunk
    _send_json(
        sock,
        {
            "cmd": "run",
            "tokens": tokens,
            "chunk": chunk,
            "split": args.split,
            "n_chunks": n_chunks,
            "handoff": bool(args.handoff),
            "transport": args.transport,
        },
    )
    ack = _recv_json(sock)
    assert ack.get("ok"), ack

    sendq: "queue.Queue" = queue.Queue(maxsize=args.depth)
    send_times = []
    err = []

    def sender():
        try:
            # MLX streams are thread-local: the comm stream must be made here.
            stream = mx.new_stream(mx.cpu) if args.transport != "socket" else None
            while True:
                item = sendq.get()
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
                if args.transport != "socket":
                    link_send(keep, 1, stream)
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
        nb = None if args.transport != "socket" else to_wire(h)
        t1 = time.perf_counter()
        sendq.put((idx, h, nb))  # blocks when the tail is behind -> backpressure
        t_block = time.perf_counter() - t1
        per_chunk.append({"idx": idx, "n": n, "gpu_s": t_gpu, "block_s": t_block})
        pos += n
        mx.clear_cache()
    t_head_done = time.perf_counter() - t_start
    sendq.put(None)
    th.join()
    if err:
        raise RuntimeError(err[0])

    # tail signals "last chunk retired" before any optional handoff so the
    # pipeline wall clock is not polluted by the stretch-goal transfer
    done = _recv_json(sock)
    assert done.get("cmd") == "done", done
    t_total = time.perf_counter() - t_start
    handoff = handoff_recv(sock) if args.handoff else None
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
    lo, hi = args.split, args.layers
    model, caches, local, n_layers, load_s = load_stage(args.model, lo, hi, args.prune)
    stage = Stage(model, caches, local, n_layers)
    print(f"[tail] layers {lo}:{hi} of {n_layers} loaded in {load_s:.1f}s", flush=True)

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"[tail] listening on {args.bind}:{args.port}", flush=True)
    sock, addr = srv.accept()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[tail] peer {addr}", flush=True)
    hello = _recv_json(sock)
    assert hello.get("cmd") == "hello", hello
    assert hello["split"] == args.split, (hello, args.split)
    args.transport = hello["transport"]
    _send_json(sock, {"ok": True, "load_s": load_s})
    if args.transport != "socket":
        g = link_group(args.transport)
        print(f"[tail] {args.transport} rank {g.rank()}/{g.size()}", flush=True)

    while True:
        req = _recv_json(sock)
        if req.get("cmd") == "bye":
            break
        assert req["split"] == args.split, (req, args.split)
        _send_json(sock, {"ok": True, "load_s": load_s})
        rep = _tail_one(args, stage, sock, req)
        _send_json(sock, rep)
        print(json.dumps(rep), flush=True)
        _reset_caches(stage, args.model)
    sock.close()


def _tail_one(args, stage: Stage, sock, req):
    recvq: "queue.Queue" = queue.Queue(maxsize=args.depth)
    recv_times = []
    err = []

    def receiver():
        try:
            stream = mx.new_stream(mx.cpu) if args.transport != "socket" else None
            hdrbuf = bytearray(HDR.size)
            while True:
                _recv_exact(sock, memoryview(hdrbuf), HDR.size)
                magic, idx, B, S, HC, D, nbytes = HDR.unpack(bytes(hdrbuf))
                assert magic == MAGIC, magic
                if idx == EOF_IDX:
                    recvq.put(None)
                    return
                # header already arrived -> this times the payload transfer only
                t0 = time.perf_counter()
                if args.transport != "socket":
                    payload = link_recv((B, S, HC, D), 0, stream)
                else:
                    buf = bytearray(nbytes)
                    _recv_exact(sock, memoryview(buf), nbytes)
                    payload = buf
                recv_times.append(time.perf_counter() - t0)
                recvq.put((idx, payload, (B, S, HC, D)))
        except Exception as e:  # noqa: BLE001
            err.append(repr(e))
            recvq.put(None)

    th = threading.Thread(target=receiver, daemon=True)
    th.start()

    per_chunk = []
    t_start = time.perf_counter()
    last_logits = None
    while True:
        t0 = time.perf_counter()
        item = recvq.get()
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
            {"idx": idx, "wait_s": t_wait, "deser_s": t_deser, "gpu_s": t_gpu}
        )
        mx.clear_cache()
    t_total = time.perf_counter() - t_start
    th.join(timeout=5)
    if err:
        raise RuntimeError(err[0])

    tok = None
    if last_logits is not None:
        lg = stage.finish(last_logits)
        mx.eval(lg)
        tok = int(mx.argmax(lg[0, -1]).item())
    _send_json(sock, {"cmd": "done"})
    ho = handoff_send(sock, stage.caches) if req.get("handoff") else None
    return {
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
    print(f"[single] layers {lo}:{hi} of {n_layers} loaded in {load_s:.1f}s", flush=True)

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
    _dump(args, {"role": "single", "lo": lo, "hi": hi, "load_s": load_s, "runs": results})


def _dump(args, obj):
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(obj, indent=2))
        print(f"wrote {args.out}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["head", "tail", "single"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", type=int, default=23, help="first layer of the tail stage")
    p.add_argument("--layers", type=int, default=45)
    p.add_argument("--lo", type=int, default=0, help="single-role: first layer")
    p.add_argument("--hi", type=int, default=None, help="single-role: end layer (exclusive)")
    p.add_argument("--tokens", type=int, nargs="+", default=[8192])
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--peer", default="10.0.0.2")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=39200)
    p.add_argument("--depth", type=int, default=2, help="in-flight chunk queue depth")
    p.add_argument(
        "--transport",
        choices=["socket", "ring", "jaccl"],
        default="ring",
        help="boundary-tensor transport; jaccl is Thunderbolt RDMA, "
        "ring is TCP over the same IPs, socket is the plain-python fallback",
    )
    p.add_argument(
        "--peers",
        default=None,
        help="comma separated ip per rank (rank 0 first), e.g. 10.0.0.1,10.0.0.2",
    )
    p.add_argument(
        "--ring-hosts",
        default=None,
        help="legacy alias for --peers; ip:port form is accepted for ring",
    )
    p.add_argument("--jaccl-port", type=int, default=39500)
    p.add_argument(
        "--rdma-dev",
        default=None,
        help="override the autodetected rdma_enN (matched by IPv4-mapped GID)",
    )
    p.add_argument("--connect-timeout", type=float, default=1800.0)
    p.add_argument("--out", default=None)
    p.add_argument("--no-prune", dest="prune", action="store_false")
    p.add_argument(
        "--handoff",
        action="store_true",
        help="after prefill, ship stage B's caches back to stage A and time it",
    )
    args = p.parse_args(argv)

    mx.random.seed(args.seed)
    if args.role != "single" and args.transport != "socket":
        args.transport = setup_link_env(
            args.transport,
            args.peers or args.ring_hosts,
            0 if args.role == "head" else 1,
            jaccl_port=args.jaccl_port,
            rdma_dev=args.rdma_dev,
        )
    if args.role == "head":
        run_head(args)
    elif args.role == "tail":
        run_tail(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
