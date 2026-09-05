"""Server-side defaults for MLX's Metal command-buffer limits.

Operator-approved serving default (2026-09-05, GLM-5.3-Flash campaign):
``MLX_MAX_MB_PER_BUFFER=1024`` and ``MLX_MAX_OPS_PER_BUFFER=50000`` unless the
environment already sets them.  Measured on the p512/gen64 rail, B=1, same
server, ABBA-paired, 3 lifecycles per setting: plain greedy 28.55 -> 31.42 tok/s
(+10.0 %), DFlash2 speculative 64.03 -> 65.09 (+1.7 %), prefill -2..3 %, peak
memory +6 GB, greedy output byte-identical.  Receipts:
bench/hwdossier/receipts/sweep11/L2_BUFFER_SWEEP_20260905 (private campaign
repo).  Caveat recorded there and in models/glm5_next/language.py:91: 2048 MB
buffers cost 7.5 % at B=16 in an earlier batched measurement; 1024 at B>1 is
not yet measured, so batched deployments should re-check or pin the env.

This module must stay free of ``mlx`` imports: it runs BEFORE the package (and
therefore Metal) is imported, from ``mlx_vlm/server/__main__.py``.  MLX reads
both variables lazily on first Metal use, so setting them here is early enough
for ``python -m mlx_vlm.server``; an embedding process that already touched
Metal keeps whatever it had.
"""

import os

DEFAULT_MAX_MB_PER_BUFFER = "1024"
DEFAULT_MAX_OPS_PER_BUFFER = "50000"

_KEYS = (
    ("MLX_MAX_MB_PER_BUFFER", DEFAULT_MAX_MB_PER_BUFFER),
    ("MLX_MAX_OPS_PER_BUFFER", DEFAULT_MAX_OPS_PER_BUFFER),
)


def apply_default_metal_buffer_env(environ=None):
    """Set the two limits if absent. Returns ``{key: (value, defaulted)}``."""
    env = os.environ if environ is None else environ
    out = {}
    for key, default in _KEYS:
        present = env.get(key)
        if present is None or present == "":
            env[key] = default
            out[key] = (default, True)
        else:
            out[key] = (present, False)
    return out


def describe_metal_buffer_env(environ=None) -> str:
    env = os.environ if environ is None else environ
    return ", ".join(f"{k}={env.get(k, '<unset>')}" for k, _ in _KEYS)
