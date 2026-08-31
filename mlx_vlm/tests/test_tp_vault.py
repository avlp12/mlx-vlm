"""Vault-in-TP: per-rank shard-local stores that cannot contaminate each other.

The design is one sentence long -- each rank saves and restores its own half,
and only the *name* of the boundary crosses the wire -- so the tests are about
the two ways that can go wrong:

* two stores that should be distinct agreeing on a rung name (a single-box rung
  restored into a shard, or rank 1's half restored into rank 0), and
* the two ranks disagreeing about whether a rung exists, which would put one
  rank on a warm cache and the other on a cold one.

Both are silent failures: the shapes match either way, so the restore "works"
and the answer is merely wrong.
"""

import pytest
from types import SimpleNamespace

from mlx_vlm import context_vault as V
from mlx_vlm.context_vault_wire import boundary_hash
from mlx_vlm.tp import vault as TV
from mlx_vlm.tp.mirror_vault import MirroredVault


@pytest.fixture(autouse=True)
def _clean_topology():
    V.set_tp_topology("tp1")
    V.reset_vault()
    TV.reset_shard_vault()
    yield
    V.set_tp_topology("tp1")
    V.reset_vault()
    TV.reset_shard_vault()


# ------------------------------------------------------------ hash separation
_REPORT_R0 = {"rank": 0, "size": 2, "kda_layers": 34, "dsa_layers": 11,
              "moe_layers": 45, "dense_layers": 0, "all_reduces_per_step": 101}
_REPORT_R1 = dict(_REPORT_R0, rank=1)
_REPORT_TP4 = dict(_REPORT_R0, size=4)


def test_topology_descriptor_separates_rank_size_and_split():
    d0 = TV.topology_descriptor(_REPORT_R0, "/m")
    assert len({d0,
                TV.topology_descriptor(_REPORT_R1, "/m"),
                TV.topology_descriptor(_REPORT_TP4, "/m"),
                TV.topology_descriptor(None, "/m")}) == 4


def test_single_box_and_tp_identities_are_different():
    """The load-bearing separation: a rung computed by a whole model and a rung
    computed by half of one are not interchangeable, and must not be nameable
    the same way."""
    single = V.vault_identity("/models/quasar")
    V.set_tp_topology(TV.topology_descriptor(_REPORT_R0, "/models/quasar"))
    tp0 = V.vault_identity("/models/quasar")
    V.set_tp_topology(TV.topology_descriptor(_REPORT_R1, "/models/quasar"))
    tp1 = V.vault_identity("/models/quasar")
    assert len({single, tp0, tp1}) == 3


def test_boundary_names_cannot_collide_across_topologies():
    """Same model, same tokens, same depth -- three different rung names."""
    toks = list(range(64))
    names = []
    for report in (None, _REPORT_R0, _REPORT_R1, _REPORT_TP4):
        V.set_tp_topology(TV.topology_descriptor(report, "/m"))
        names.append(boundary_hash(toks, 32, V.vault_identity("/m")))
    assert len(set(names)) == 4


def test_changing_topology_drops_the_store():
    """Every rung already stored was named under the old topology, so keeping
    them would be paying memory for entries nobody can ever look up again."""
    V.set_tp_topology(TV.topology_descriptor(_REPORT_R0, "/m"))
    v = V.get_vault("id-a")
    v.insert([1, 2, 3, 4], 2, [])
    V.set_tp_topology(TV.topology_descriptor(_REPORT_R1, "/m"))
    assert V.get_vault("id-a").rungs == 0


# ------------------------------------------------------- foreign-origin refusal
def test_restore_refuses_a_checkpoint_from_another_vault():
    """A checkpoint can reach ``restore_into`` without ever having been in this
    vault -- from the peer tier, or from the other rank.  Shapes match, so the
    check has to be on provenance."""
    a, b = V.ContextVault("identity-A"), V.ContextVault("identity-B")
    a.insert([1, 2, 3, 4], 2, [])
    cp = a.lookup([1, 2, 3, 4])
    assert cp is not None and cp.origin == "identity-A"
    assert b.restore_into([], cp) is False
    assert b.stats.rejected_foreign == 1


def test_restore_accepts_its_own_checkpoint():
    a = V.ContextVault("identity-A")
    a.insert([1, 2, 3, 4], 2, [])
    cp = a.lookup([1, 2, 3, 4])
    assert a.restore_into([], cp) is True


def test_a_pre_origin_checkpoint_is_still_accepted():
    """Empty origin means "stored before this field existed"; refusing those
    would turn a code upgrade into a silent cache-wipe."""
    a = V.ContextVault("identity-A")
    a.insert([1, 2, 3, 4], 2, [])
    cp = a.lookup([1, 2, 3, 4])
    cp.origin = ""
    assert a.restore_into([], cp) is True


# ------------------------------------------------------------ the shard vault
class _FakeCacheEntry:
    def __init__(self):
        self.offset = 0


def _stub_fragments(monkeypatch, nbytes=1024):
    monkeypatch.setattr(V, "capture_fragments", lambda caches, n: ["frag", n])
    monkeypatch.setattr(V, "eval_fragments", lambda frags: None)
    monkeypatch.setattr(V, "fragments_nbytes", lambda frags: nbytes)
    restored = []
    monkeypatch.setattr(V, "restore_fragments",
                        lambda caches, frags: restored.append((caches, frags)) or True)
    return restored


def test_shard_vault_round_trips_by_name(monkeypatch):
    restored = _stub_fragments(monkeypatch)
    sv = TV.ShardVault("topo-r1")
    assert sv.store("aa" * 16, 2048, ["c"]) is True
    assert sv.has("aa" * 16, 2048) is True
    assert sv.restore("aa" * 16, 2048, ["fresh"]) is True
    assert restored == [(["fresh"], ["frag", 2048])]


def test_shard_vault_misses_on_a_depth_mismatch(monkeypatch):
    _stub_fragments(monkeypatch)
    sv = TV.ShardVault("topo-r1")
    sv.store("aa" * 16, 2048, ["c"])
    assert sv.restore("aa" * 16, 4096, ["fresh"]) is False
    assert sv.misses == 1


def test_shard_vault_evicts_lru_within_budget(monkeypatch):
    _stub_fragments(monkeypatch, nbytes=100)
    sv = TV.ShardVault("topo-r1", budget_bytes=250)
    for i in range(3):
        sv.store(f"{i:032x}", 1024, ["c"])
    assert sv.rungs == 2 and sv.evictions == 1
    assert sv.has(f"{0:032x}", 1024) is False    # the oldest went first


def test_shard_vault_is_rebuilt_when_the_topology_changes():
    a = TV.shard_vault("topo-r0")
    assert TV.shard_vault("topo-r0") is a
    assert TV.shard_vault("topo-r1") is not a


# ------------------------------------------------------- the mirrored vault
class _Mirror:
    def __init__(self, peer_has=True):
        self.stores, self.restores = [], []
        self.peer_has = peer_has

    def announce_vault_store(self, name, prefix_len):
        self.stores.append((name, prefix_len))

    def announce_vault_restore(self, cache, name, prefix_len):
        self.restores.append((name, prefix_len))
        return self.peer_has


def _mirrored(peer_has=True):
    base = V.ContextVault("identity-tp0")
    return MirroredVault(base, _Mirror(peer_has)), base


def test_mirrored_insert_announces_before_storing_locally():
    """Announced while both caches are still exactly at ``prefix_len``: rank 1
    is blocked in the control collective immediately after the same prefill
    chunk, so the message finds it in the state the rung describes."""
    mv, base = _mirrored()
    toks = list(range(16))
    assert mv.insert(toks, 8, []) is True
    assert mv._mirror.stores == [
        (boundary_hash(toks, 8, base.identity), 8)]
    assert base.rungs == 1


def test_mirrored_restore_requires_a_matching_lookup():
    """A rung named after somebody else's prompt would restore the wrong state
    on rank 1 while rank 0 restored the right one."""
    mv, base = _mirrored()
    base.insert(list(range(16)), 8, [])
    cp = base.lookup(list(range(16)))
    assert mv.restore_into([], cp) is False      # no lookup through the mirror
    assert mv._mirror.restores == []


def test_mirrored_restore_hits_both_ranks():
    mv, base = _mirrored(peer_has=True)
    toks = list(range(16))
    mv.insert(toks, 8, [])
    cp = mv.lookup(toks)
    assert cp is not None
    assert mv.restore_into([], cp) is True
    assert mv._mirror.restores == [(boundary_hash(toks, 8, base.identity), 8)]


def test_mirrored_restore_falls_back_cold_when_the_peer_missed():
    """The gate: rank 0 holding a rung does not mean rank 1 holds it.  The two
    stores evict independently, and half a warm cache is worse than none."""
    mv, base = _mirrored(peer_has=False)
    toks = list(range(16))
    mv.insert(toks, 8, [])
    cp = mv.lookup(toks)
    assert mv.restore_into([], cp) is False
    assert mv.peer_misses == 1
    assert mv.stats_dict()["peer_misses"] == 1


def test_mirrored_vault_delegates_everything_else():
    mv, base = _mirrored()
    assert mv.identity == base.identity
    assert mv.rungs == 0


# ------------------------------------------------------------- dispatch hook
def test_dispatch_hook_is_a_noop_without_a_mirror():
    """TP off must cost exactly one getattr that returns None."""
    plain = SimpleNamespace(language_model=SimpleNamespace())
    assert getattr(plain.language_model, "tp_mirror_vault", None) is None


def test_mirror_exposes_the_wrap_hook(monkeypatch):
    from mlx_vlm.server import tp_mode as T

    monkeypatch.setattr(T, "_ctrl_send", lambda *a, **k: None)
    m = T.MirroredLanguageModel(SimpleNamespace(config="c"))
    wrapped = m.tp_mirror_vault(V.ContextVault("id"))
    assert isinstance(wrapped, MirroredVault)
