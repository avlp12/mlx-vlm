# Metal command-buffer limits must be in the environment before Metal is
# initialised, and `python -m mlx_vlm.server` has ALREADY imported the package
# (and touched Metal) by the time this module runs.  So when the limits are
# absent we set them and re-exec the interpreter once; the re-exec'd process
# sees them from the start.  A process that pins either variable is left alone.
# Numbers, the operator approval, and the shared helper: server/metal_env.py.
from .metal_env import maybe_reexec_with_metal_env

maybe_reexec_with_metal_env("mlx_vlm.server")

from . import main  # noqa: E402

if __name__ == "__main__":
    main()
