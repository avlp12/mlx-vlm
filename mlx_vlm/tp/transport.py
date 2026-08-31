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
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

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


def all_sum(x: mx.array) -> mx.array:
    """The one collective the sharded forward needs."""
    if _GROUP is None or _GROUP.size() == 1:
        return x
    return mx.distributed.all_sum(x, group=_GROUP)
