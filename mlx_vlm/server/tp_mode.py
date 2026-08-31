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

from ..tp.worker import (  # noqa: F401  re-exported for the rank-0 side
    ENV_HOSTS, ENV_MAX_TOK, ENV_RANK, ENV_WORKER_PY, ENV_WORKER_SRC,
    HEADER, OP_EXIT, OP_FORWARD, OP_MAKE_CACHE, TPUnavailable,
    ENV_WORKER_MODEL,
    _ctrl_recv, _ctrl_send, _max_tok, decode, encode, preflight,
    tp_enabled, tp_hosts, tp_rank, worker_loop,
)

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


def launch_worker(model_path: str, hosts: List[str]) -> subprocess.Popen:
    """Start rank 1 over ssh, the way the pipeline tail is started."""
    py = os.environ.get(ENV_WORKER_PY, "/Users/m3ms/venv_mlx321/bin/python")
    src = os.environ.get(ENV_WORKER_SRC, "/Users/m3ms/src/mlx-vlm-tp2serve")
    host = hosts[1]
    remote_model = os.environ.get(ENV_WORKER_MODEL) or model_path
    inner = (
        f"cd {src} && MLX_VLM_GLM5_FUSED_KDA=1 PYTHONPATH={src} "
        f"{ENV_HOSTS}='{','.join(hosts)}' {ENV_RANK}=1 "
        f"MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD={_max_tok()} "
        f"nohup {py} -m mlx_vlm.tp.worker --model {remote_model} "
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
