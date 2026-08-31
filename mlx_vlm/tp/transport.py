"""Tensor-parallel transport: an mx.distributed group over Thunderbolt RDMA.

macOS exposes one RDMA device per Thunderbolt port (``rdma_en2 .. rdma_en7``)
whose GID table carries an IPv4-mapped GID derived from whatever IP is already
configured on the matching ``enN`` interface.  Those devices exist
*independently of the "Thunderbolt Bridge" network service* -- verified with
that service disabled on both boxes -- so a jaccl group needs no interface,
bridge or service changes at all: point it at the tbnet IPs that are already up.

Env protocol (this is what ``mlx.launch --backend jaccl`` would set, minus the
launcher, so each box can be started with its own venv and paths)::

    MLX_IBV_DEVICES=<file containing [[null,"rdma_en5"],["rdma_en5",null]]>
    MLX_JACCL_COORDINATOR=<rank-0 ip>:<port>
    MLX_RANK=<0|1>

``init_tp`` fills all three in, autodetecting the local device by matching the
IPv4-mapped GID against the local address, and falls back to the ring backend
(plain TCP over the same IPs) if RDMA is unavailable.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

import logging

logger = logging.getLogger(__name__)

_GROUP = None
_BACKEND = None

_LIBRDMA = "/usr/lib/librdma.dylib"


class _Gid(ctypes.Union):
    _fields_ = [("raw", ctypes.c_ubyte * 16)]


def rdma_devices() -> List[str]:
    """Names of the RDMA devices this box exposes (empty if no provider)."""
    try:
        lib = ctypes.CDLL(_LIBRDMA)
    except OSError:
        return []
    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ibv_get_device_name.restype = ctypes.c_char_p
    lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    n = ctypes.c_int(0)
    lst = lib.ibv_get_device_list(ctypes.byref(n))
    if not lst:
        return []
    return [lib.ibv_get_device_name(lst[i]).decode() for i in range(n.value)]


def device_ipv4_gids() -> dict:
    """{device name: IPv4 address of its IPv4-mapped GID}.

    jaccl refuses a port with no IPv4-mapped GID ("[jaccl] No IPv4-mapped GID
    for this device"), and that GID is derived from the IP on the matching
    ``enN``, so this map is exactly "which RDMA device carries which of my IPs".
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


def detect_device(local_ip: str) -> Optional[str]:
    for dev, ip in device_ipv4_gids().items():
        if ip == local_ip:
            return dev
    return None


def _write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return str(path)


def init_tp(
    hosts: Optional[List[str]] = None,
    rank: Optional[int] = None,
    *,
    backend: str = "jaccl",
    device: Optional[str] = None,
    coordinator_port: int = 39500,
    ring_ports: Optional[List[int]] = None,
    strict: bool = False,
):
    """Join (or return) the tensor-parallel group.

    ``hosts`` are the per-rank IPs on the fast link, e.g.
    ``["10.0.0.1", "10.0.0.2"]``.  ``backend="jaccl"`` uses Thunderbolt RDMA and
    silently degrades to ``"ring"`` (TCP over the same IPs) when RDMA is not
    usable, unless ``strict``.
    """
    global _GROUP, _BACKEND
    if _GROUP is not None:
        return _GROUP

    # MLX's Metal fence has two modes. The default one signals an MTLSharedEvent
    # at command-buffer granularity; the "fast" one spins a tiny GPU kernel on a
    # shared-memory counter, so a CPU-stream op inside a GPU chain no longer
    # costs a command-buffer round trip. Distributed collectives are CPU-stream
    # ops (AllReduce/Send/Recv have no GPU implementation on Metal), so every
    # reduce in a sharded forward pays that crossing: measured 161 us with the
    # default fence and 4.9 us with the fast one, which is the difference
    # between 45 ms and 3 ms of comm per 101-reduce decode step.
    #
    # mlx reads the flag once, lazily, in a function-local static
    # (mlx/utils.h: metal_fast_synch), on the first fence construction -- which
    # happens well after import and after this call -- so setting it here works.
    # Requires Metal 3.2+ / macOS 15+; mlx falls back on its own if unsupported.
    if os.environ.get("MLX_VLM_TP_FAST_SYNCH", "1") == "1":
        if "MLX_METAL_FAST_SYNCH" not in os.environ:
            os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    hosts = hosts or os.environ.get("MLX_VLM_TP_HOSTS", "").split(",")
    hosts = [h.strip() for h in hosts if h.strip()]
    if rank is None:
        rank = int(os.environ.get("MLX_RANK", "0"))
    if not hosts:
        raise ValueError("init_tp needs hosts (or MLX_VLM_TP_HOSTS)")

    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    if backend == "jaccl":
        dev = device or detect_device(hosts[rank])
        if dev is None:
            if strict:
                raise RuntimeError(
                    f"no RDMA device carries {hosts[rank]}; "
                    f"available: {device_ipv4_gids()}"
                )
            backend = "ring"
        else:
            # rdma[i][j] = the device rank i uses to reach rank j; None on the
            # diagonal (mlx asserts "RDMA device of self should be null").
            n = len(hosts)
            table = [[None if i == j else dev for j in range(n)] for i in range(n)]
            os.environ["MLX_IBV_DEVICES"] = _write_json(
                tmp / "mlx_vlm_tp_ibv.json", table
            )
            os.environ["MLX_JACCL_COORDINATOR"] = f"{hosts[0]}:{coordinator_port}"
            os.environ["MLX_RANK"] = str(rank)

    if backend == "ring":
        ports = ring_ports or [39400 + i for i in range(len(hosts))]
        os.environ["MLX_HOSTFILE"] = _write_json(
            tmp / "mlx_vlm_tp_ring.json", [[f"{h}:{p}"] for h, p in zip(hosts, ports)]
        )
        os.environ["MLX_RANK"] = str(rank)

    _GROUP = mx.distributed.init(backend=backend)
    _BACKEND = backend
    return _GROUP


def group():
    return _GROUP


def backend() -> Optional[str]:
    return _BACKEND


def tp_size() -> int:
    return _GROUP.size() if _GROUP is not None else 1


def tp_rank() -> int:
    return _GROUP.rank() if _GROUP is not None else 0


# How many all_sums this process has *constructed*.  Not the same as how many
# it executed -- MLX is lazy and prunes nodes nothing evaluates, which is a
# distinction this campaign has already been bitten by once.  Counting
# construction is still the cheap first question when two ranks stop pairing:
# if the counts differ, the ranks are running different forwards; if they
# agree, the divergence is in execution and needs a different probe.
_ALL_SUM_CALLS = 0
# Index within the current forward, so a stalled pair can be compared position
# by position rather than in aggregate: "rank 0 reached #57 and rank 1 reached
# #58" names the reduce, and the shape names which kind it was.
_FWD_IDX = 0
_DEEP_ENV = None


def _deep_trace() -> bool:
    global _DEEP_ENV
    if _DEEP_ENV is None:
        _DEEP_ENV = os.environ.get("MLX_VLM_GLM5_TP_TRACE_DEEP", "") not in (
            "", "0", "false", "False")
    return _DEEP_ENV


def collective_count() -> int:
    return _ALL_SUM_CALLS


def reset_forward_counter() -> None:
    global _FWD_IDX
    _FWD_IDX = 0


def all_sum(x: mx.array) -> mx.array:
    """The one collective the sharded forward needs."""
    global _ALL_SUM_CALLS, _FWD_IDX
    if _GROUP is None or _GROUP.size() == 1:
        return x
    _ALL_SUM_CALLS += 1
    _FWD_IDX += 1
    if _deep_trace():
        logger.info("all_sum #%d shape=%s", _FWD_IDX, tuple(x.shape))
    return mx.distributed.all_sum(x, group=_GROUP)


# ---------------------------------------------------------------- deadman


class Deadman:
    """Bound the wall time of a region that blocks inside a collective.

    A stalled peer leaves the fast fence's one-thread GPU kernel spinning on a
    shared counter.  Measured tolerance on M3 Ultra is at least 40 s with
    correct results and no Metal watchdog kill (the watchdog kill seen earlier
    in this campaign was the *default* fence, a different mechanism), but a peer
    that dies outright spins forever.

    This cannot preempt the GPU spin -- nothing on the host can -- so it does
    the next best thing: it turns an indefinite hang into a fast, diagnosed
    abort, which also frees the surviving peer instead of deadlocking the pair.
    A genuinely bounded wait would need a max-iteration counter in mlx's
    fence_wait kernel; that is an upstream change, not something a user of the
    library can do.
    """

    def __init__(self, seconds: float = 120.0, label: str = "", on_timeout="abort"):
        self.seconds = seconds
        self.label = label
        self.on_timeout = on_timeout
        self._timer = None
        self._armed = False

    def _fire(self):
        if not self._armed:
            return
        rank = tp_rank() if _GROUP is not None else "?"
        msg = (
            f"[tp deadman] rank {rank}: '{self.label}' exceeded {self.seconds}s. "
            f"A peer is stalled or dead and the GPU-side fence is spinning; "
            f"aborting so the surviving peer fails fast instead of deadlocking."
        )
        print(msg, flush=True)
        traceback.print_stack()
        if self.on_timeout == "abort":
            os._exit(75)  # EX_TEMPFAIL

    def __enter__(self):
        self._armed = True
        self._timer = threading.Timer(self.seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, *exc):
        self._armed = False
        if self._timer is not None:
            self._timer.cancel()
        return False


def guarded_eval(arrays, seconds: float = 120.0, label: str = "eval"):
    """mx.eval under a deadman."""
    with Deadman(seconds, label):
        mx.eval(arrays)
