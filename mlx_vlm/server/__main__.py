# Metal command-buffer limits must be in the environment before the package
# (and with it mlx) is imported -- see server/metal_env.py for the numbers and
# the operator approval they come from.
from .metal_env import apply_default_metal_buffer_env

apply_default_metal_buffer_env()

from . import main  # noqa: E402

if __name__ == "__main__":
    main()
