# Metal command-buffer limits must be in the environment before Metal is
# initialised, and `python -m mlx_vlm.server` has ALREADY imported the package
# (and touched Metal) by the time this module runs.  So when the limits are
# absent we set them and re-exec the interpreter once; the re-exec'd process
# sees them from the start.  A process that pins either variable is left alone.
# Numbers and the operator approval: server/metal_env.py.
import os
import sys

from .metal_env import apply_default_metal_buffer_env, needs_reexec

if needs_reexec(os.environ):
    apply_default_metal_buffer_env(os.environ)
    os.environ["MLX_VLM_SERVER_METAL_ENV_APPLIED"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "mlx_vlm.server", *sys.argv[1:]])

from . import main  # noqa: E402

if __name__ == "__main__":
    main()
