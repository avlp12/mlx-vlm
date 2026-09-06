"""Service-start admission for the resident tail (L38 A7).

The campaign proved a box was safe to load with a supervisor, an idle window
and a flock.  A served tail has none of those, so three gates run before the
weights: one tail per box, one heavy model per box, and the registered wired
budget.  These tests pin all three -- and pin that a refusal is cheap, i.e. that
nothing was loaded -- plus the rail sampler that keeps reporting after start-up
and the shape of the health line an operator and the head both read.
"""

import json
import os
import socket
from types import SimpleNamespace

import pytest

from mlx_vlm import pipeline_admission as adm
from mlx_vlm import pipeline_prefill as pp

GIB = 1024 ** 3


def _args(tmp_path, **over):
    base = dict(
        model=str(tmp_path / "model"),
        split=23,
        layers=45,
        prune=True,
        port=39200,
        lock_file=str(tmp_path / "pp_tail.lock"),
        no_lock=False,
        allow_shared_box=False,
        wired_cap_bytes=None,
        shard_bytes=None,
        allow_wired_overcommit=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _vm_stat(wired_pages, page=16384):
    return (
        f"Mach Virtual Memory Statistics: (page size of {page} bytes)\n"
        "Pages free:                            1000.\n"
        f"Pages wired down:                {wired_pages}.\n"
    )


def _ps(*lines):
    return lambda: "".join(lines)


def _quiet_ps():
    return _ps(f"{os.getpid()} 1 python -m mlx_vlm.pipeline_prefill --role tail\n")


# ------------------------------------------------------------------- flock


def test_a_second_tail_is_refused_and_the_first_keeps_the_box(tmp_path):
    args = _args(tmp_path)
    first = adm.admit(
        args, ps_reader=_quiet_ps(), vm_stat_reader=lambda: _vm_stat(100)
    )
    try:
        assert first.admitted is True
        assert first.gates["flock"]["ok"] is True
        with pytest.raises(adm.AdmissionRefused) as caught:
            adm.admit(
                _args(tmp_path),
                ps_reader=_quiet_ps(),
                vm_stat_reader=lambda: _vm_stat(100),
            )
    finally:
        first.release()
    exc = caught.value
    assert exc.gate == "flock"
    assert "already owns this box" in exc.message
    # the refusal names the holder and the lock, and says nothing was loaded
    assert exc.detail["holder"]["pid"] == os.getpid()
    assert exc.detail["lock_path"] == str(tmp_path / "pp_tail.lock")
    assert "no weights were loaded" in exc.message
    assert exc.detail["report"]["admitted"] is False
    assert exc.detail["report"]["refused_gate"] == "flock"
    # released: the box is free again
    adm.admit(
        _args(tmp_path), ps_reader=_quiet_ps(), vm_stat_reader=lambda: _vm_stat(100)
    ).release()


def test_the_lock_is_advisory_so_a_killed_tail_does_not_lock_the_box_out(tmp_path):
    path = tmp_path / "pp_tail.lock"
    lock = adm.BoxLock(path).acquire()
    lock.release()
    # a fresh process-level acquire succeeds; the file survives, the lock does not
    assert path.exists()
    second = adm.BoxLock(path).acquire()
    try:
        assert json.loads(path.read_text())["pid"] == os.getpid()
    finally:
        second.release()


def test_env_lock_path_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_VLM_PIPELINE_LOCK", str(tmp_path / "other.lock"))
    assert adm.default_lock_path() == str(tmp_path / "other.lock")
    monkeypatch.delenv("MLX_VLM_PIPELINE_LOCK")
    assert adm.default_lock_path() == adm.DEFAULT_LOCK_PATH


# ------------------------------------------------------------ wired policy


def test_the_registered_policy_is_the_one_the_campaign_registered():
    # 483183820800 == 450 GiB exactly; drift here means the served tail is
    # budgeting against a number nobody registered.
    assert adm.REGISTERED_WIRED_POLICY["requested_limit_bytes"] == 450 * GIB
    assert adm.REGISTERED_WIRED_POLICY["scope"] == "run05_only"
    assert adm.REGISTERED_WIRED_POLICY["require_at_most_device_recommendation"]


def test_wired_refusal_when_current_plus_shard_exceeds_the_cap():
    d = adm.wired_decision(
        shard_bytes=200 * GIB, current_wired=300 * GIB, cap_bytes=450 * GIB
    )
    assert d["refused"] is True and d["ok"] is False
    assert d["projected_bytes"] == 500 * GIB
    assert d["headroom_bytes"] == -50 * GIB
    assert "exceeds the registered cap 450.0 GiB" in d["reason"]


def test_wired_admission_with_headroom_and_the_exact_boundary():
    d = adm.wired_decision(
        shard_bytes=150 * GIB, current_wired=10 * GIB, cap_bytes=450 * GIB
    )
    assert d["ok"] is True and d["refused"] is False
    assert d["headroom_bytes"] == 290 * GIB
    edge = adm.wired_decision(
        shard_bytes=440 * GIB, current_wired=10 * GIB, cap_bytes=450 * GIB
    )
    assert edge["ok"] is True and edge["headroom_bytes"] == 0


def test_device_recommendation_clamps_the_registered_cap():
    d = adm.wired_decision(
        shard_bytes=200 * GIB,
        current_wired=10 * GIB,
        device_info=lambda: {"max_recommended_working_set_size": 180 * GIB},
    )
    assert d["cap_bytes"] == 180 * GIB
    assert d["cap_source"].endswith("+device_recommendation")
    assert d["refused"] is True


def test_an_unknown_shard_footprint_is_reported_not_silently_passed():
    d = adm.wired_decision(shard_bytes=None, current_wired=10 * GIB)
    assert d["shard_known"] is False
    assert d["ok"] is True and d["refused"] is False
    assert "unknown" in d["reason"]


def test_overcommit_flag_admits_but_still_records_the_breach():
    d = adm.wired_decision(
        shard_bytes=500 * GIB, current_wired=10 * GIB, allow_overcommit=True
    )
    assert d["ok"] is True and d["refused"] is False
    assert d["overcommit_allowed"] is True and "exceeds" in d["reason"]


def test_wired_bytes_reads_vm_stat_the_way_the_campaign_does():
    assert adm.wired_bytes(lambda: _vm_stat(3, page=16384)) == 3 * 16384
    with pytest.raises(adm.AdmissionRefused):
        adm.wired_bytes(lambda: "nothing parseable here")


def _write_safetensors(path, tensors):
    """A real safetensors header with no payload -- the reader only maps the
    header, so the file body may be absent."""
    import struct

    header = {}
    off = 0
    for name, nbytes in tensors:
        header[name] = {"dtype": "BF16", "shape": [nbytes // 2],
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_shard_footprint_counts_only_the_layers_this_stage_keeps(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    _write_safetensors(
        root / "model-00001.safetensors",
        [
            ("language_model.model.layers.0.mlp.w", 1000),
            ("language_model.model.layers.23.mlp.w", 2000),
            ("language_model.model.layers.44.mlp.w", 4000),
            ("language_model.model.embed_tokens.weight", 8000),
            ("language_model.lm_head.weight", 16000),
            ("vision_model.blocks.0.w", 32000),
            ("language_model.model.norm.weight", 64),
        ],
    )
    tail, detail = adm.shard_footprint_bytes(root, 23, 45, n_layers=45, prune=True)
    # layers 23 and 44, plus lm_head (a tail stage keeps it) and the final norm;
    # layer 0, embed_tokens (lo>0) and the vision tower are all dropped
    assert tail == 2000 + 4000 + 16000 + 64
    assert detail["source"] == "safetensors_header" and detail["shard_files"] == 1
    head, _ = adm.shard_footprint_bytes(root, 0, 23, n_layers=45, prune=True)
    # layer 0 plus embed_tokens plus norm; lm_head belongs to the tail
    assert head == 1000 + 8000 + 64
    whole, _ = adm.shard_footprint_bytes(root, 0, 45, n_layers=45, prune=False)
    assert whole == 1000 + 2000 + 4000 + 8000 + 16000 + 32000 + 64


def test_an_absent_model_directory_is_unknown_not_zero(tmp_path):
    value, detail = adm.shard_footprint_bytes(tmp_path / "nope", 23, 45)
    assert value is None and detail["source"] == "unknown"


# --------------------------------------------------------------- preflight


HEAVY = "/usr/bin/python3.9 -m mlx_vlm.server --model /x --port 8085\n"


def test_preflight_refuses_when_a_heavy_model_is_already_resident(tmp_path):
    ps = _ps(f"{os.getpid()} 1 python -m mlx_vlm.pipeline_prefill --role tail\n",
             f"4242 1 {HEAVY}")
    with pytest.raises(adm.AdmissionRefused) as caught:
        adm.admit(_args(tmp_path), ps_reader=ps,
                  vm_stat_reader=lambda: _vm_stat(100))
    exc = caught.value
    assert exc.gate == "preflight"
    assert "mlx_vlm.server(pid 4242)" in exc.message
    assert "one heavy model load per box" in exc.message
    assert "no weights were loaded" in exc.message
    # the lock the flock gate took is handed back on refusal
    assert adm.BoxLock(tmp_path / "pp_tail.lock").acquire().release() is None


def test_allow_shared_box_admits_the_same_situation(tmp_path):
    ps = _ps(f"{os.getpid()} 1 self\n", f"4242 1 {HEAVY}")
    report = adm.admit(_args(tmp_path, allow_shared_box=True), ps_reader=ps,
                       vm_stat_reader=lambda: _vm_stat(100))
    try:
        assert report.admitted is True
        assert report.gates["preflight"]["ok"] is True
        assert report.gates["preflight"]["found"][0]["pid"] == 4242
    finally:
        report.release()


def test_the_gate_never_refuses_because_of_its_own_launcher():
    """pp_cooperation_child runs the tail IN ITS OWN PROCESS and under a
    pp_cooperation_supervisor parent; both match the pattern list."""
    me = os.getpid()
    ps = _ps(
        f"{me} 777 python bench/ops/pp_cooperation_child.py --box epsilon\n",
        "777 1 python bench/ops/pp_cooperation_supervisor.py --box epsilon\n",
        "888 1 python -m gdn_e2e_arms\n",
    )
    found = adm.heavy_processes(ps_reader=ps, own_pid=me)
    assert [p["pid"] for p in found] == [888]


def test_pattern_list_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("MLX_VLM_PIPELINE_PREFLIGHT_PATTERNS", "weird_thing")
    ps = _ps(f"{os.getpid()} 1 self\n", f"4242 1 {HEAVY}",
             "99 1 /usr/bin/python3 run weird_thing now\n")
    found = adm.heavy_processes(ps_reader=ps)
    assert [p["pid"] for p in found] == [99]


def test_the_documented_fleet_patterns_are_all_present():
    for name in ("mlx_vlm.server", "gdn_e2e_arms", "l7b_prefill", "pp_cooperation"):
        assert name in adm.HEAVY_MODEL_PATTERNS
    assert "rsync" in adm.BOX_BUSY_PATTERNS
    assert "rsync" not in adm.HEAVY_MODEL_PATTERNS


# ------------------------------------------------------------ rail sampler


def test_the_sampler_stays_healthy_until_the_window_has_evidence():
    s = adm.RailSampler(window=8, p95_bound_s=1.0, min_samples=4)
    for _ in range(3):
        assert s.observe(99.0) is False  # slow, but not yet enough samples
    assert s.snapshot()["degraded"] is False
    assert s.observe(99.0) is True  # the 4th sample crosses min_samples
    assert s.snapshot()["p95_wire_s"] == 99.0


def test_degraded_latches_on_p95_and_clears_only_below_the_recovery_band():
    s = adm.RailSampler(window=10, p95_bound_s=1.0, min_samples=4,
                        recover_factor=0.8)
    for _ in range(10):
        s.observe(0.2)
    assert s.degraded is False
    for _ in range(3):
        s.observe(5.0)
    assert s.degraded is True, "p95 over a 10-wide window crosses on 1 slow in 20"
    # sitting just under the bound must NOT clear it -- that is the hysteresis
    for _ in range(10):
        s.observe(0.9)
    assert s.degraded is True
    for _ in range(10):
        s.observe(0.5)
    assert s.degraded is False
    snap = s.snapshot()
    assert snap["transitions"] == 2 and snap["observed"] == 33
    assert snap["window"] == 10 and snap["p95_bound_s"] == 1.0


def test_the_window_slides_so_an_old_stall_stops_counting():
    s = adm.RailSampler(window=4, p95_bound_s=1.0, min_samples=1)
    s.observe(10.0)
    assert s.degraded is True
    for _ in range(4):
        s.observe(0.1)
    assert s.degraded is False and s.snapshot()["samples"] == 4


def test_non_finite_and_missing_samples_are_ignored():
    s = adm.RailSampler(window=4, p95_bound_s=1.0, min_samples=1)
    assert s.observe(None) is False
    assert s.observe(float("nan")) is False
    assert s.snapshot()["samples"] == 0


def test_prefill_seconds_ride_along_for_the_operator():
    s = adm.RailSampler(window=4, p95_bound_s=99.0, min_samples=1)
    s.observe(0.4, 12.0)
    snap = s.snapshot()
    assert snap["last_wire_s"] == 0.4 and snap["last_prefill_s"] == 12.0
    assert snap["p95_prefill_s"] == 12.0


# --------------------------------------------- the daemon's health and ping


def test_health_line_schema(tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    report = adm.admit(_args(tmp_path), ps_reader=_quiet_ps(),
                       vm_stat_reader=lambda: _vm_stat(100))
    try:
        sampler = adm.RailSampler(window=4, p95_bound_s=1.0, min_samples=1)
        daemon = pp.TailDaemon(srv, lambda s, a: 0, admission_report=report,
                               sampler=sampler)
        line = json.loads(json.dumps(daemon.status()))  # must be JSON-clean
        assert set(line) == {
            "role", "state", "peer", "uptime_s", "idle_s", "shutdown_reason",
            "degraded", "rail", "admission", "connections", "requests",
            "connection_errors", "last_error",
        }
        assert line["role"] == "tail" and line["degraded"] is False
        assert line["rail"]["p95_bound_s"] == 1.0
        a = line["admission"]
        assert a["admitted"] is True and a["refused_gate"] is None
        assert set(a["gates"]) == {"flock", "preflight", "wired"}
        assert a["gates"]["wired"]["cap_bytes"] == 450 * GIB
        assert a["gates"]["wired"]["policy_scope"] == "run05_only"
        assert a["lock_path"] == str(tmp_path / "pp_tail.lock")
        sampler.observe(50.0)
        assert daemon.status()["degraded"] is True
        assert daemon.status()["rail"]["degraded"] is True
    finally:
        report.release()
        srv.close()


def test_status_is_unchanged_shape_when_no_admission_ran():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        line = pp.TailDaemon(srv, lambda s, a: 0).status()
        assert line["admission"] is None and line["rail"] is None
        assert line["degraded"] is False
    finally:
        srv.close()


class _FakeSock:
    """Speaks the tail's length-prefixed JSON to a session, in-process."""

    def __init__(self, messages):
        self._inbox = list(messages)
        self.sent = []

    def settimeout(self, *_):
        pass

    def setsockopt(self, *_):
        pass


def test_ping_reply_carries_the_degraded_flag_and_its_evidence(monkeypatch):
    sampler = adm.RailSampler(window=4, p95_bound_s=1.0, min_samples=1)
    args = SimpleNamespace(
        model="unused", model_sha256="a" * 64, source_revision="b" * 40,
        split=1, io_timeout=5.0, transport="socket", stop_file=None, depth=2,
    )
    hello = {"cmd": "hello", "transport": "socket", "schema": 1,
             "model_sha256": "a" * 64, "source_revision": "b" * 40,
             "split": 1, "n_layers": 3}
    inbox = [hello, {"cmd": "ping"}, {"cmd": "ping", "rail": True},
             {"cmd": "bye"}]
    sent = []
    monkeypatch.setattr(pp, "_recv_json", lambda sock: inbox.pop(0))
    monkeypatch.setattr(pp, "_send_json", lambda sock, obj: sent.append(obj))
    monkeypatch.setattr(pp, "_reset_caches", lambda *a: None)
    monkeypatch.setattr(pp, "StopAwareSocket", lambda raw, stop, to: raw)

    session = pp.tail_session_factory(args, object(), 3, 0.0, None, sampler)
    served = session(_FakeSock([]), ("127.0.0.1", 1))
    assert served == 0
    pings = [m for m in sent if m.get("cmd") == "ping"]
    assert len(pings) == 2
    # a bare ping keeps the shipped two-key contract byte for byte, so a head
    # that has not been updated (PipelineHead.ping compares for EQUALITY) is
    # unaffected by this change
    assert pings[0] == {"cmd": "ping", "ok": True}
    # a head that asks gets the flag and its evidence
    assert pings[1]["ok"] is True
    assert pings[1]["degraded"] is False
    assert pings[1]["rail"]["samples"] == 0


def test_a_slow_request_flips_the_flag_a_later_ping_reports(monkeypatch):
    sampler = adm.RailSampler(window=4, p95_bound_s=1.0, min_samples=1)
    args = SimpleNamespace(
        model="unused", model_sha256="a" * 64, source_revision="b" * 40,
        split=1, io_timeout=5.0, transport="socket", stop_file=None, depth=2,
    )
    env = {"schema": 1, "request_id": "d" * 32, "model_sha256": "a" * 64,
           "source_revision": "b" * 40, "split": 1, "n_layers": 3, "batch": 1,
           "depth": 4, "token_sha256": "c" * 64, "chunks": [2, 2]}
    inbox = [
        {"cmd": "hello", "transport": "socket", **env},
        {"cmd": "run", "transport": "socket", "envelope": env},
        {"cmd": "ping", "rail": True},
        {"cmd": "bye"},
    ]
    sent = []
    monkeypatch.setattr(pp, "_recv_json", lambda sock: inbox.pop(0))
    monkeypatch.setattr(pp, "_send_json", lambda sock, obj: sent.append(obj))
    monkeypatch.setattr(pp, "_reset_caches", lambda *a: None)
    monkeypatch.setattr(pp, "StopAwareSocket", lambda raw, stop, to: raw)
    monkeypatch.setattr(
        pp, "_tail_one",
        lambda a, stage, sock, req: {"wire_recv_s": 40.0, "tail_total_s": 90.0},
    )
    session = pp.tail_session_factory(args, object(), 3, 0.0, None, sampler)
    assert session(_FakeSock([]), ("127.0.0.1", 1)) == 1
    ping = [m for m in sent if m.get("cmd") == "ping"][0]
    assert ping["degraded"] is True
    assert ping["rail"]["p95_wire_s"] == 40.0
    assert ping["rail"]["last_prefill_s"] == 90.0


# ------------------------------------------------------ refusal loads nothing


def test_a_refused_start_never_reaches_load_stage(tmp_path, monkeypatch):
    """The whole point of a start-time gate: the refusal is cheaper than the
    load it prevents."""
    loaded = []
    monkeypatch.setattr(pp, "load_stage", lambda *a: loaded.append(a))
    holder = adm.BoxLock(tmp_path / "pp_tail.lock").acquire()
    args = _args(tmp_path, admission=True, stop_file=None, connect_timeout=0.0,
                 idle_timeout=0.0, once=False, health_port=0, bind="127.0.0.1",
                 model_sha256="a" * 64, source_revision="b" * 40,
                 transport="socket", io_timeout=5.0, depth=2)
    try:
        with pytest.raises(adm.AdmissionRefused) as caught:
            pp.run_tail(args)
    finally:
        holder.release()
    assert caught.value.gate == "flock"
    assert loaded == []


def test_admission_is_off_for_programmatic_callers_and_on_for_the_cli(monkeypatch):
    monkeypatch.delenv("MLX_VLM_PIPELINE_ADMISSION", raising=False)
    assert pp._admission_enabled(SimpleNamespace()) is False
    assert pp._admission_enabled(SimpleNamespace(admission=True)) is True
    assert pp._admission_enabled(SimpleNamespace(admission=False)) is False
    monkeypatch.setenv("MLX_VLM_PIPELINE_ADMISSION", "1")
    assert pp._admission_enabled(SimpleNamespace()) is True


def test_cli_exposes_the_admission_flags_and_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(pp, "run_tail", lambda args: seen.update(vars(args)))
    monkeypatch.delenv("MLX_VLM_PIPELINE_ADMISSION", raising=False)
    pp.main(["--role", "tail", "--model", "/x", "--transport", "socket"])
    # the SERVICE entry point admits by default; a programmatic caller does not
    assert seen["admission"] is True
    assert seen["lock_file"] is None and seen["no_lock"] is False
    assert seen["allow_shared_box"] is False
    assert seen["wired_cap_bytes"] is None and seen["shard_bytes"] is None
    assert seen["allow_wired_overcommit"] is False
    assert seen["rail_window"] is None and seen["rail_p95_s"] is None
    seen.clear()
    pp.main([
        "--role", "tail", "--model", "/x", "--transport", "socket",
        "--no-admission", "--allow-shared-box", "--allow-wired-overcommit",
        "--lock-file", "/tmp/x.lock", "--no-lock",
        "--wired-cap-bytes", "123", "--shard-bytes", "456",
        "--rail-window", "7", "--rail-p95-s", "1.5",
    ])
    assert seen["admission"] is False and seen["no_lock"] is True
    assert seen["wired_cap_bytes"] == 123 and seen["shard_bytes"] == 456
    assert seen["rail_window"] == 7 and seen["rail_p95_s"] == 1.5


def test_env_switches_the_cli_default_off(monkeypatch):
    seen = {}
    monkeypatch.setattr(pp, "run_tail", lambda args: seen.update(vars(args)))
    monkeypatch.setenv("MLX_VLM_PIPELINE_ADMISSION", "0")
    pp.main(["--role", "tail", "--model", "/x", "--transport", "socket"])
    assert seen["admission"] is False


def test_the_cli_refusal_exits_non_zero_with_a_sentence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pp, "run_tail",
        lambda args: (_ for _ in ()).throw(
            adm.AdmissionRefused("flock", "another pipeline tail already owns this box")
        ),
    )
    with pytest.raises(SystemExit) as caught:
        pp.main(["--role", "tail", "--model", "/x", "--transport", "socket"])
    assert caught.value.code == pp.ADMISSION_REFUSED_EXIT != 0


def test_a_queue_script_that_greps_for_the_patterns_is_not_a_heavy_model():
    """Observed live on gesicht: the fleet's queue scripts wait with
    ``zsh -c ... grep -qE "mlx_vlm.server|gdn_e2e_arms|..."``, so their OWN
    command line contains every pattern in the list.  Those scripts filter
    ``grep -v "zsh -c source"`` and ``grep -v grep`` for exactly this reason."""
    ps = _ps(
        f"{os.getpid()} 1 self\n",
        "98024 1 /bin/zsh -c source /Users/x/.claude/shell-snapshots/s.sh && "
        "while ps -axo command= | grep -qE \"mlx_vlm.server|l7b_prefill\"; "
        "do sleep 30; done\n",
        "1234 1 grep -E mlx_vlm.server\n",
        "95718 1 /opt/homebrew/.../Python bench/hwdossier/l7b_prefill_chunk_sweep.py "
        "--target-tokens 4096\n",
        "77 1 /usr/bin/rsync -a /a /b\n",
    )
    assert [p["pid"] for p in adm.heavy_processes(ps_reader=ps)] == [95718]
    # rsync is a box-busy pattern, and it is matched by its own argv0
    busy = adm.heavy_processes(patterns=adm.BOX_BUSY_PATTERNS, ps_reader=ps)
    assert sorted(p["pid"] for p in busy) == [77, 95718]
