from mlx_vlm.server import metal_env


def test_defaults_applied_when_absent():
    env = {}
    out = metal_env.apply_default_metal_buffer_env(env)
    assert env == {"MLX_MAX_MB_PER_BUFFER": "1024", "MLX_MAX_OPS_PER_BUFFER": "50000"}
    assert out["MLX_MAX_MB_PER_BUFFER"] == ("1024", True)
    assert out["MLX_MAX_OPS_PER_BUFFER"] == ("50000", True)


def test_existing_values_are_respected():
    env = {"MLX_MAX_MB_PER_BUFFER": "2048", "MLX_MAX_OPS_PER_BUFFER": "100000"}
    out = metal_env.apply_default_metal_buffer_env(env)
    assert env["MLX_MAX_MB_PER_BUFFER"] == "2048"
    assert out["MLX_MAX_MB_PER_BUFFER"] == ("2048", False)
    assert out["MLX_MAX_OPS_PER_BUFFER"] == ("100000", False)


def test_empty_string_counts_as_unset():
    env = {"MLX_MAX_MB_PER_BUFFER": ""}
    metal_env.apply_default_metal_buffer_env(env)
    assert env["MLX_MAX_MB_PER_BUFFER"] == "1024"


def test_describe_lists_both_keys():
    env = {"MLX_MAX_MB_PER_BUFFER": "1024"}
    text = metal_env.describe_metal_buffer_env(env)
    assert "MLX_MAX_MB_PER_BUFFER=1024" in text
    assert "MLX_MAX_OPS_PER_BUFFER=<unset>" in text


def test_module_does_not_import_mlx():
    import sys
    import importlib
    importlib.reload(metal_env)
    assert "mlx.core" not in getattr(metal_env, "__dict__", {})
    src = open(metal_env.__file__).read()
    assert "import mlx" not in src
    del sys


def test_needs_reexec_only_when_both_absent_and_not_marked():
    assert metal_env.needs_reexec({}) is True
    assert metal_env.needs_reexec({"MLX_MAX_MB_PER_BUFFER": "2048"}) is False
    assert metal_env.needs_reexec({"MLX_MAX_OPS_PER_BUFFER": "100"}) is False
    assert metal_env.needs_reexec({metal_env.REEXEC_MARK: "1"}) is False
