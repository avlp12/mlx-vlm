"""Head-side runtime for two-box layer-pipelined prefill.

This is the half that lives inside a normal ``generate()`` call. The other half
is ``python -m mlx_vlm.pipeline_prefill --role tail`` running on the peer box.

Enable by setting ``MLX_VLM_PIPELINE_HOSTS`` (the worker's control endpoint);
everything else has a default::

    MLX_VLM_PIPELINE_HOSTS=10.0.0.2:39210          # required, enables the feature
    MLX_VLM_PIPELINE_RING=10.0.0.1:39400,10.0.0.2:39401   # ring backend (default transport)
    MLX_VLM_PIPELINE_SPLIT=23                      # int, or "auto" for the micro-sweep
    MLX_VLM_PIPELINE_MIN_TOKENS=4096               # below this, stay single-box
    MLX_VLM_PIPELINE_CALIB=~/.cache/mlx_vlm/pipeline_splits.json

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

import json
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
    _recv_json,
    _send_json,
    handoff_recv,
    install_state,
    ring_group,
    ring_send,
    setup_ring_env,
    to_wire,
)

_DISABLED = object()
_CTX = None


# ------------------------------------------------------------------ settings


class PipelineSettings:
    def __init__(self, peer, ring, split, min_tokens, calib_path, transport):
        self.peer = peer
        self.ring = ring
        self.split = split
        self.min_tokens = min_tokens
        self.calib_path = calib_path
        self.transport = transport

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
            min_tokens=int(os.environ.get("MLX_VLM_PIPELINE_MIN_TOKENS", "4096")),
            calib_path=calib,
            transport="ring" if ring else "socket",
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
    cfg = model.language_model.args
    layers = model.language_model.model.layers
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
    lm = model.language_model.model
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
    merged = [sorted(c[j] for c in per_rep)[len(per_rep) // 2] for j in range(len(bounds) + 1)]
    total = merged[-1]
    scored = []
    for j, k in enumerate(bounds):
        head, tail = merged[j], (total - merged[j]) * peer_speed
        scored.append({"split": k, "head_s": head, "tail_s": tail, "max_s": max(head, tail)})
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
    n = model.language_model.pipeline_num_layers
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

    # -- connection ---------------------------------------------------------
    def connect(self, timeout=60.0):
        if self.transport == "ring":
            setup_ring_env(self.settings.ring, 0)
        s = socket.socket()
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        deadline = time.time() + timeout
        while True:
            try:
                s.connect(self.settings.peer)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(1.0)
        self.sock = s
        _send_json(s, {"cmd": "hello", "transport": self.transport, "split": self.split})
        ack = _recv_json(s)
        assert ack.get("ok"), ack
        if self.transport == "ring":
            ring_group()
        return self

    def close(self):
        if self.sock is not None:
            try:
                _send_json(self.sock, {"cmd": "bye"})
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    # -- one prefill --------------------------------------------------------
    def begin(self, tokens: int, chunk: int):
        _send_json(
            self.sock,
            {
                "cmd": "run",
                "tokens": tokens,
                "chunk": chunk,
                "split": self.split,
                "n_chunks": -1,          # ar.py drives the schedule; EOF ends it
                "handoff": True,
                "transport": self.transport,
            },
        )
        ack = _recv_json(self.sock)
        assert ack.get("ok"), ack
        self._q = queue.Queue(maxsize=2)
        self._err = []
        self._th = threading.Thread(target=self._sender, daemon=True)
        self._th.start()

    def _sender(self):
        try:
            stream = mx.new_stream(mx.cpu) if self.transport == "ring" else None
            while True:
                item = self._q.get()
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

    def local_caches(self, cache):
        return [c for c in cache[: self.split] if c is not None]

    def prefill_chunk(self, model, input_ids, inputs_embeds, cache):
        t0 = time.perf_counter()
        h = model.language_model.pipeline_prefill_head(
            inputs=input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            split=self.split,
        )
        mx.eval(h)
        self.stats["head_gpu_s"] += time.perf_counter() - t0
        if self._err:
            raise RuntimeError(f"pipeline peer failed: {self._err[0]}")
        idx = len(self.stats["chunks"])
        self.stats["chunks"].append(h.shape[1])
        self._q.put((idx, h, None if self.transport == "ring" else to_wire(h)))

    def finalize(self, cache):
        """End the stream, pull stage B's caches back, install them for decode."""
        self._q.put(None)
        self._th.join()
        if self._err:
            raise RuntimeError(f"pipeline peer failed: {self._err[0]}")
        done = _recv_json(self.sock)
        assert done.get("cmd") == "done", done
        t0 = time.perf_counter()
        ho = handoff_recv(self.sock, rebuild=True)
        install_state(cache, ho.pop("states"))
        mx.eval([c.state for c in cache])
        ho["handoff_install_s"] = time.perf_counter() - t0
        report = _recv_json(self.sock)
        self.stats["handoff"] = ho
        self.stats["tail"] = report
        return self.stats


# ------------------------------------------------------------------ factory


def maybe_open_pipeline(model, total_tokens: int, verbose: bool = False):
    """Return a connected PipelineHead, or None to stay single-box."""
    global _CTX
    if _CTX is _DISABLED:
        return None
    settings = PipelineSettings.from_env()
    if settings is None:
        _CTX = _DISABLED
        return None
    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "pipeline_prefill_head"):
        if verbose:
            print("[pipeline] model has no pipeline hook; staying single-box", flush=True)
        _CTX = _DISABLED
        return None
    if total_tokens < settings.min_tokens:
        return None
    split = resolve_split(model, settings, total_tokens, verbose=verbose)
    head = PipelineHead(settings, split, lm.pipeline_num_layers).connect()
    if verbose:
        print(
            f"[pipeline] split={split} transport={settings.transport} "
            f"peer={settings.peer[0]}:{settings.peer[1]}",
            flush=True,
        )
    return head
