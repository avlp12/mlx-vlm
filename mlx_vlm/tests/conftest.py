import os

# float32 matmul runs at TF32 precision on hardware with matrix units, which is
# looser than the float32 references these tests compare against. Set rather than
# setdefault, so an inherited MLX_ENABLE_TF32=1 doesn't leak into the test run.
os.environ["MLX_ENABLE_TF32"] = "0"


def pytest_sessionfinish(session, exitstatus):
    """Release thread-local MLX resources before Python finalization."""
    del session, exitstatus

    import mlx.core as mx

    clear_streams = getattr(mx, "clear_streams", None)
    if clear_streams is not None:
        clear_streams()


# --- device pin (2026-09-03) -------------------------------------------------
# MLX does not read MLX_DEFAULT_DEVICE itself (verified on 0.32.1.dev20260902:
# with it exported, mx.default_device() still answers Device(gpu, 0)).  Desk-agent
# test runs that only exported it took the GPU and tripped the fleet preflight on
# a box in a measurement window (2026-09-02 23:39).  Honour it here so
# ``MLX_DEFAULT_DEVICE=cpu pytest ...`` really is CPU-only, and print the effective
# device in the report header so a wrong device is visible in every run.
import os as _os

import mlx.core as _mx


def pytest_configure(config):
    want = _os.environ.get("MLX_DEFAULT_DEVICE", "").strip().lower()
    if want == "cpu":
        _mx.set_default_device(_mx.cpu)
    elif want == "gpu":
        _mx.set_default_device(_mx.gpu)


def pytest_report_header(config):
    return f"mlx default device: {_mx.default_device()}"
