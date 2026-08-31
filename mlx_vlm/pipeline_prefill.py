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
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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
        """Run this stage over one chunk.

        head: ``inputs`` is int token ids -> embed + hc-broadcast.
        tail/middle: ``h`` is the already-expanded (B, S, hc, D) boundary tensor.
        """
        if self.is_head:
            h = self.lm.embed_tokens(inputs)

        fa_cache = self.caches[self.fa_local] if self.fa_local is not None else None
        fa_mask = (
            create_attention_mask(h, fa_cache[0] if fa_cache else None, return_array=True)
            if self.fa_local is not None
            else None
        )
        ssm_mask = (
            create_ssm_mask(h, self.caches[self.ssm_local])
            if self.ssm_local is not None
            else None
        )

        if self.is_head:
            h = mx.broadcast_to(
                h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
            )
            h = mx.contiguous(h)

        for i in self.local:
            layer = self.lm.layers[i]
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=self.caches[i])
        return h

    def finish(self, h: mx.array) -> mx.array:
        """Tail only: pool the hc streams, final norm, LM head on the last position."""
        h = h.mean(axis=2)
        h = self.lm.norm(h)
        return self.model.language_model._logits(h[:, -1:, :])

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
        },
    )
    ack = _recv_json(sock)
    assert ack.get("ok"), ack

    sendq: "queue.Queue" = queue.Queue(maxsize=args.depth)
    send_times = []
    err = []

    def sender():
        try:
            while True:
                item = sendq.get()
                if item is None:
                    sock.sendall(HDR.pack(MAGIC, EOF_IDX, 0, 0, 0, 0, 0))
                    return
                idx, keep, nb = item
                B, S, HC, D = keep.shape
                t0 = time.perf_counter()
                sock.sendall(HDR.pack(MAGIC, idx, B, S, HC, D, nb.nbytes))
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
        nb = to_wire(h)
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

    tail = _recv_json(sock)
    t_total = time.perf_counter() - t_start

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
        "wire_bytes_per_chunk": boundary_bytes(chunk, hc, D),
        "wire_send_s": sum(send_times),
        "wire_send_each": send_times,
        "head_chunks": per_chunk,
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
            hdrbuf = bytearray(HDR.size)
            while True:
                t0 = time.perf_counter()
                _recv_exact(sock, memoryview(hdrbuf), HDR.size)
                magic, idx, B, S, HC, D, nbytes = HDR.unpack(bytes(hdrbuf))
                assert magic == MAGIC, magic
                if idx == EOF_IDX:
                    recvq.put(None)
                    return
                buf = bytearray(nbytes)
                _recv_exact(sock, memoryview(buf), nbytes)
                recv_times.append(time.perf_counter() - t0)
                recvq.put((idx, buf, (B, S, HC, D)))
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
        h = from_wire(buf, shape)
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
    return {
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
    p.add_argument("--connect-timeout", type=float, default=1800.0)
    p.add_argument("--out", default=None)
    p.add_argument("--no-prune", dest="prune", action="store_false")
    args = p.parse_args(argv)

    mx.random.seed(args.seed)
    if args.role == "head":
        run_head(args)
    elif args.role == "tail":
        run_tail(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
