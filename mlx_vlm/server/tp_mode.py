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

REPRODUCIBILITY.  TP mode does not reproduce single-box tokens: all_sum adds two
partial sums where one device summed all 4096, and at a one-ULP top-2 gap the
argmax flips (measured: 16.875 vs 16.75, exactly one bf16 ULP at that
magnitude).  Cross-lane token identity is a TP-off property.  The TP-mode
invariant is rank0 == rank1, asserted by the identity test.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

ENV_HOSTS = "MLX_VLM_GLM5_TP_HOSTS"
ENV_RANK = "MLX_VLM_GLM5_TP_RANK"
ENV_WORKER_PY = "MLX_VLM_GLM5_TP_WORKER_PYTHON"
ENV_WORKER_SRC = "MLX_VLM_GLM5_TP_WORKER_SRC"
ENV_MAX_TOK = "MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD"

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


class MirroredLanguageModel:
    """Rank-0 wrapper: announce each forward, then run it locally.

    Only the plain ``inputs`` path is mirrored.  ``inputs_embeds`` and
    ``capture_layer_ids`` are refused rather than silently diverging the ranks,
    which is why speculative decoding is disabled in TP mode for now.
    """

    def __init__(self, lm):
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
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._lm, name)

    def __call__(self, inputs=None, cache=None, **kw):
        if inputs is None or kw.get("inputs_embeds") is not None:
            raise TPUnavailable(
                "TP mode mirrors token-id forwards only; inputs_embeds is not "
                "supported (vision / spec paths must stay off)")
        if kw.get("capture_layer_ids") is not None:
            raise TPUnavailable(
                "TP mode does not mirror hidden capture; speculative decoding "
                "is disabled in TP mode (TODO: mirror the capture + rollback)")
        with self._lock:
            cid = id(cache)
            if cid != self._last_cache_id or cache is not self._last_cache_obj:
                self._epoch += 1
                self._last_cache_id = cid
                self._last_cache_obj = cache
                _ctrl_send(OP_MAKE_CACHE, self._epoch, None)
            _ctrl_send(OP_FORWARD, self._epoch, inputs)
            return self._lm(inputs, cache=cache, **kw)

    def shutdown(self) -> None:
        with self._lock:
            try:
                _ctrl_send(OP_EXIT, self._epoch, None)
            except Exception:  # teardown must never mask the real error
                logger.warning("tp: EXIT broadcast failed", exc_info=True)


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


def launch_worker(model_path: str, hosts: List[str]) -> subprocess.Popen:
    """Start rank 1 over ssh, the way the pipeline tail is started."""
    py = os.environ.get(ENV_WORKER_PY, "/Users/m3ms/venv_mlx321/bin/python")
    src = os.environ.get(ENV_WORKER_SRC, "/Users/m3ms/src/mlx-vlm-tp2serve")
    host = hosts[1]
    inner = (
        f"cd {src} && MLX_VLM_GLM5_FUSED_KDA=1 PYTHONPATH={src} "
        f"{ENV_HOSTS}='{','.join(hosts)}' {ENV_RANK}=1 "
        f"MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD={_max_tok()} "
        f"nohup {py} -m mlx_vlm.server.tp_worker --model {model_path} "
        f"> ~/tp_worker.log 2>&1 &"
    )
    cmd = ["ssh", "-o", "BatchMode=yes", f"m3ms@{host}", inner]
    logger.info("tp: launching rank1 on %s", host)
    return subprocess.Popen(cmd)


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

        from ..generate import wired_limit
        from ..tp.load import load_sharded, materialize
        from ..tp.transport import tp_rank as _r, tp_size
        from ..utils import get_model_path, load_tokenizer

        worker = launch_worker(model_path, hosts)
        info = preflight(hosts, 0)
        logger.info("tp: group up %s", info)
        model, report = load_sharded(model_path, _r(), tp_size())
        peak = materialize(model)
        logger.info("tp: sharded %s peak %.1f GiB", report, peak)
        wire = wired_limit(model, [mx.default_stream(mx.default_device())])
        wire.__enter__()
        inner = model.language_model if hasattr(model, "language_model") else model
        mirrored = MirroredLanguageModel(inner)
        if hasattr(model, "language_model"):
            model.language_model = mirrored
        else:
            model = mirrored
        processor = load_tokenizer(get_model_path(model_path))
        return model, processor, model.config if hasattr(model, "config") else None
    except Exception as e:
        logger.error("tp: unavailable (%s); serving single-box", e, exc_info=True)
        if worker is not None:
            try:
                worker.terminate()
            except Exception:
                pass
        return None
