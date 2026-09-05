import sys

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


def test_unset_keys():
    assert set(metal_env.unset_keys({})) == {
        "MLX_MAX_MB_PER_BUFFER",
        "MLX_MAX_OPS_PER_BUFFER",
    }
    assert metal_env.unset_keys({"MLX_MAX_MB_PER_BUFFER": "2048"}) == [
        "MLX_MAX_OPS_PER_BUFFER"
    ]
    assert metal_env.unset_keys(
        {"MLX_MAX_MB_PER_BUFFER": "2048", "MLX_MAX_OPS_PER_BUFFER": "100"}
    ) == []


def test_metal_env_warning_none_when_both_set():
    env = {"MLX_MAX_MB_PER_BUFFER": "1024", "MLX_MAX_OPS_PER_BUFFER": "50000"}
    assert metal_env.metal_env_warning(env) is None


def test_metal_env_warning_names_supported_entry_point():
    text = metal_env.metal_env_warning({})
    assert text is not None
    assert "MLX_MAX_MB_PER_BUFFER" in text
    assert "MLX_MAX_OPS_PER_BUFFER" in text
    assert "python -m mlx_vlm.server" in text


def test_reexec_argv_discards_argv0_and_uses_module():
    argv = ["/some/venv/bin/mlx_vlm.server", "--port", "9000"]
    built = metal_env._reexec_argv("mlx_vlm.server", argv)
    assert built == [sys.executable, "-m", "mlx_vlm.server", "--port", "9000"]


def test_reexec_argv_defaults_to_sys_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ignored-argv0", "--help"])
    built = metal_env._reexec_argv("mlx_vlm.server")
    assert built == [sys.executable, "-m", "mlx_vlm.server", "--help"]


def test_maybe_reexec_skips_when_already_set():
    env = {"MLX_MAX_MB_PER_BUFFER": "2048", "MLX_MAX_OPS_PER_BUFFER": "100"}
    calls = []
    result = metal_env.maybe_reexec_with_metal_env(
        "mlx_vlm.server", environ=env, execv=lambda *a: calls.append(a)
    )
    assert result is False
    assert calls == []


def test_maybe_reexec_skips_when_marker_present():
    env = {metal_env.REEXEC_MARK: "1"}
    calls = []
    result = metal_env.maybe_reexec_with_metal_env(
        "mlx_vlm.server", environ=env, execv=lambda *a: calls.append(a)
    )
    assert result is False
    assert calls == []


def test_maybe_reexec_applies_default_sets_marker_and_execs():
    env = {}
    calls = []
    argv = ["/some/venv/bin/mlx_vlm.server", "--help"]
    result = metal_env.maybe_reexec_with_metal_env(
        "mlx_vlm.server",
        environ=env,
        argv=argv,
        execv=lambda *a: calls.append(a),
    )
    assert result is True
    assert env["MLX_MAX_MB_PER_BUFFER"] == "1024"
    assert env["MLX_MAX_OPS_PER_BUFFER"] == "50000"
    assert env[metal_env.REEXEC_MARK] == "1"
    assert calls == [
        (sys.executable, [sys.executable, "-m", "mlx_vlm.server", "--help"])
    ]
