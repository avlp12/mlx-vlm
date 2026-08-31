"""Rank-1 side of TP=2 serving: env, control codec, preflight, worker loop.

Deliberately free of any dependency on ``mlx_vlm.server``.  The worker holds a
shard and executes forwards; it has no business importing an HTTP framework, and
on a box that only ever runs rank 1 those packages need not be installed at all.
``mlx_vlm.server.tp_mode`` re-exports from here for the rank-0 side.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

ENV_HOSTS = "MLX_VLM_GLM5_TP_HOSTS"
ENV_RANK = "MLX_VLM_GLM5_TP_RANK"
ENV_WORKER_PY = "MLX_VLM_GLM5_TP_WORKER_PYTHON"
ENV_WORKER_SRC = "MLX_VLM_GLM5_TP_WORKER_SRC"
ENV_MAX_TOK = "MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD"
# The peer may hold the same checkpoint under a different path
# (different home directory); default to rank 0's path.
ENV_WORKER_MODEL = "MLX_VLM_GLM5_TP_WORKER_MODEL"

OP_EXIT, OP_MAKE_CACHE, OP_FORWARD = 0, 1, 2
HEADER = 4  # [op, cache_epoch, batch, seqlen]


def tp_hosts() -> List[str]:
    raw = os.environ.get(ENV_HOSTS, "")
    return [h.strip() for h in raw.split(",") if h.strip()]


def tp_enabled() -> bool:
    return len(tp_hosts()) > 1


def tp_rank() -> int:
    return int(os.environ.get(ENV_RANK, "0"))


def _max_tok() -> int:
    try:
        return max(64, int(os.environ.get(ENV_MAX_TOK, "8192")))
    except ValueError:
        return 8192


class TPUnavailable(RuntimeError):
    """Raised when TP cannot be brought up; the caller serves single-box."""


def preflight(hosts: List[str], rank: int, timeout_s: float = 60.0) -> dict:
    """Seconds-long transport check before anything expensive.

    A four-minute sharded load per box is a bad place to discover the group will
    not form, so the same check the stage-3/4 campaign used runs at startup.
    """
    import mlx.core as mx

    from ..tp.transport import Deadman, all_sum, backend, init_tp, tp_size

    init_tp(hosts=hosts, rank=rank, backend="jaccl")
    x = mx.full((1, 1, 64), float(rank + 1))
    with Deadman(timeout_s, "tp preflight all_sum"):
        y = all_sum(x)
        mx.eval(y)
    got = float(y[0, 0, 0])
    want = float(sum(range(1, tp_size() + 1)))
    if abs(got - want) > 1e-6:
        raise TPUnavailable(f"preflight all_sum {got} != {want}")
    return {"size": tp_size(), "backend": backend(), "all_sum": got,
            "fast_synch": os.environ.get("MLX_METAL_FAST_SYNCH")}


# ------------------------------------------------------------------ control
def encode(op: int, epoch: int, shape=None, flat=None, n: Optional[int] = None):
    """Control message as a flat int32 vector. Separated from transport so the
    codec is testable without a group."""
    n = _max_tok() if n is None else n
    buf = [0] * (HEADER + n)
    b = s = 0
    if flat is not None:
        b, s = int(shape[0]), int(shape[1])
        if len(flat) > n:
            raise TPUnavailable(
                f"forward of {len(flat)} tokens exceeds {ENV_MAX_TOK}={n}")
        buf[HEADER:HEADER + len(flat)] = [int(v) for v in flat]
    buf[0], buf[1], buf[2], buf[3] = int(op), int(epoch), b, s
    return buf


def decode(row):
    """Inverse of encode: (op, epoch, batch, seqlen, flat_ids)."""
    op, epoch, b, s = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
    ids = [int(v) for v in row[HEADER:HEADER + b * s]] if (b and s) else None
    return op, epoch, b, s, ids


def _ctrl_send(op: int, epoch: int, ids) -> None:
    """Rank 0: publish a control message through the data collective."""
    import mlx.core as mx

    from ..tp.transport import all_sum

    flat = shape = None
    if ids is not None:
        flat = ids.reshape(-1).tolist()
        shape = (ids.shape[0], ids.shape[1])
    msg = mx.array([encode(op, epoch, shape, flat)], dtype=mx.int32)
    out = all_sum(msg)
    mx.eval(out)


def _ctrl_recv():
    """Rank 1: contribute zeros and read the sum."""
    import mlx.core as mx

    from ..tp.transport import all_sum

    n = _max_tok()
    out = all_sum(mx.zeros((1, HEADER + n), dtype=mx.int32))
    mx.eval(out)
    row = out[0].tolist()
    op, epoch, b, s, flat = decode(row)
    ids = mx.array(flat, dtype=mx.int32).reshape(b, s) if flat else None
    return op, epoch, b, s, ids


def worker_loop(model_path: str, hosts: List[str], rank: int) -> None:
    """Rank 1: hold a shard and execute whatever rank 0 announces."""
    import mlx.core as mx

    from ..generate import wired_limit
    from ..tp.load import load_sharded, materialize
    from ..tp.transport import tp_rank as _r, tp_size

    info = preflight(hosts, rank)
    logger.info("tp worker: joined %s", info)
    model, report = load_sharded(model_path, _r(), tp_size())
    peak = materialize(model)
    logger.info("tp worker: sharded %s peak %.1f GiB", report, peak)
    lm = model.language_model if hasattr(model, "language_model") else model
    wire = wired_limit(model, [mx.default_stream(mx.default_device())])
    wire.__enter__()
    caches, epoch = {}, -1
    try:
        while True:
            op, ep, b, s, ids = _ctrl_recv()
            if op == OP_EXIT:
                logger.info("tp worker: EXIT")
                return
            if op == OP_MAKE_CACHE:
                caches = {ep: lm.make_cache()}   # one live cache in mode-level TP
                epoch = ep
                continue
            if op == OP_FORWARD:
                c = caches.get(ep)
                if c is None:
                    c = caches[ep] = lm.make_cache()
                out = lm(ids, cache=c)
                mx.eval(out.logits)
    finally:
        try:
            wire.__exit__(None, None, None)
        except Exception:
            pass




def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    hosts = tp_hosts()
    if len(hosts) < 2:
        raise SystemExit(f"{ENV_HOSTS} must list >= 2 hosts")
    worker_loop(a.model, hosts, tp_rank())


if __name__ == "__main__":
    main()
