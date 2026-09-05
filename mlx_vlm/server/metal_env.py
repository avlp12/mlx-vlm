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

This module must stay free of ``mlx`` imports.  NOTE (measured 2026-09-05):
``python -m mlx_vlm.server`` imports the package -- and initialises Metal --
before ``server/__main__.py`` runs, so setting the variables there is too late
(a served greedy rail showed no gain).  ``__main__`` therefore re-execs the
interpreter once with the variables set when they are absent
(``needs_reexec``); an embedding process that already touched Metal keeps
whatever it had, and a process that pins either variable is never re-exec'd.

NOTE (2026-09-06): the console script ``mlx_vlm.server`` (``mlx_vlm.server:main``
-> ``app.main`` -> ``cli.main``), ``python -m mlx_vlm.server.app``, and
``from mlx_vlm.server import main`` all bypass ``server/__main__.py`` entirely,
so none of them ever saw the re-exec.  ``maybe_reexec_with_metal_env`` below is
the same logic factored out so every entry point can call it; a re-exec
restarts the interpreter, so it is safe to call even after this process has
already imported ``mlx`` -- the *fresh* process sees the environment before it
imports anything.
"""

import os
import sys

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


REEXEC_MARK = "MLX_VLM_SERVER_METAL_ENV_APPLIED"


def needs_reexec(environ=None) -> bool:
    """True when neither limit is set and this process has not re-exec'd yet."""
    env = os.environ if environ is None else environ
    if env.get(REEXEC_MARK):
        return False
    return all(not env.get(k) for k, _ in _KEYS)


def unset_keys(environ=None):
    """Return the subset of the two limit names currently unset in ``environ``."""
    env = os.environ if environ is None else environ
    return [k for k, _ in _KEYS if not env.get(k)]


def metal_env_warning(environ=None):
    """A warning string when the default was not applied, else ``None``.

    Meant for entry points that run after the point where the re-exec should
    already have happened (e.g. ``cli.main``): if either variable is still
    unset there, this process was not started (or re-exec'd) through a path
    that applies the default.
    """
    missing = unset_keys(environ)
    if not missing:
        return None
    return (
        "Metal command-buffer default not applied for %s (unset). The default "
        "(MLX_MAX_MB_PER_BUFFER=%s, MLX_MAX_OPS_PER_BUFFER=%s) is only applied "
        "automatically by the `python -m mlx_vlm.server` entry point (or an "
        "equivalent re-exec); set the variable(s) explicitly for other entry "
        "points if you want the default."
        % (", ".join(missing), DEFAULT_MAX_MB_PER_BUFFER, DEFAULT_MAX_OPS_PER_BUFFER)
    )


def _reexec_argv(module, argv=None):
    """Build the argv to re-exec as ``python -m <module> <original args>``.

    ``argv[0]`` is discarded on purpose: for the console-script entry point it
    is the path to the generated wrapper script (e.g. ``.../bin/mlx_vlm.server``),
    not something meaningful to pass to ``-m``. Every entry form re-execs the
    same way, as ``[sys.executable, "-m", module, *argv[1:]]``.
    """
    src = sys.argv if argv is None else argv
    return [sys.executable, "-m", module, *src[1:]]


def maybe_reexec_with_metal_env(module="mlx_vlm.server", environ=None, argv=None, execv=None):
    """Re-exec the interpreter (as ``python -m <module>``) with the Metal
    buffer default applied, if it is not already set and this process has not
    re-exec'd yet.

    Call this at the very top of any entry point, before any Metal-touching
    import or work. Returns ``False`` when no re-exec was necessary, so the
    caller can continue normally. On an actual re-exec, ``os.execv`` replaces
    the process image and this function does not return -- unless a fake
    ``execv`` is injected (for tests), in which case it returns ``True`` after
    calling it.
    """
    env = os.environ if environ is None else environ
    if not needs_reexec(env):
        return False
    apply_default_metal_buffer_env(env)
    env[REEXEC_MARK] = "1"
    exec_fn = os.execv if execv is None else execv
    exec_fn(sys.executable, _reexec_argv(module, argv))
    return True
