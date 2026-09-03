"""Harvest provenance on APC exact entries and vault rungs (finding L1b-1).

The measurement these tests exist for is
``~/glm53flash/logs/sweep11/L1b1_apc_poison_RESULT.json`` (2026-09-03, gesicht,
GLM-5.3-Flash-vlm-q4-quasar, one process, four arms with cache resets between
them):

    a 3,091-token APC exact CHECKPOINT entry harvested inside a B=2 prefill made
    every later SOLO serve of that prompt return sha 122e772a; the same entry
    harvested at B=1 returned 5d7c209c.  4/4 serves each way, reversed by
    clearing the cache and re-harvesting alone.  The carrier is BATCH WIDTH at
    the checkpoint column, not right padding: an equal-suffix zero-padding B=2
    batch (``right_pad_per_row=[0, 0]``), whose own rows decode 32/32 correctly,
    poisons the entry to the same sha.

Nothing here changes what is harvested.  These tests pin that every exact entry
and every vault rung now RECORDS where it came from, that the two policies keyed
on that fact behave in both directions, and -- the load-bearing one -- that with
every knob at its default the serve path is exactly what it was.

CPU only (``MLX_DEFAULT_DEVICE=cpu``; ``conftest`` pins it and prints it in the
report header).  No model load: the harvest tests drive real prefill chunks
through the same 4-layer Qwen3.5 miniature the vault wiring tests use.
"""

from __future__ import annotations

import json
import logging
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from mlx_vlm import harvest_provenance as HP
from mlx_vlm import vault_disk as VD
from mlx_vlm.apc import APCManager, apc_lookup_plan
from mlx_vlm.context_vault import (
    ContextVault,
    VaultTier,
    capture_fragments,
)
from mlx_vlm.generate.ar import BatchGenerator, PromptProcessingBatch, PromptProgress
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _tiny_lm():
    """4-layer Qwen3.5: mixed ArraysCache + KVCache, so ``apc_mode`` is ``exact``.

    The same miniature ``test_vault_server_wiring`` drives real chunks through.
    ``get_rope_index`` is left in place because these tests run the model's own
    ``__call__``.
    """
    from mlx_vlm.models import qwen3_5

    text_config = qwen3_5.TextConfig(
        model_type="qwen3_5",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        head_dim=8,
        full_attention_interval=4,
    )
    config = qwen3_5.ModelConfig(
        text_config=text_config,
        vision_config=qwen3_5.VisionConfig(
            model_type="qwen3_5", depth=1, hidden_size=16,
            intermediate_size=32, out_hidden_size=16, num_heads=2,
        ),
        model_type="qwen3_5",
    )
    return qwen3_5.LanguageModel(text_config, config)


def _meta(ids, checkpoint_len, prefix_len=0):
    return {
        "full_input_ids": list(ids),
        "prefix_len": int(prefix_len),
        "extra_hash": 0,
        "apc_blocks": [],
        "checkpoint_len": int(checkpoint_len),
        "vault_rungs": [],
    }


def _prefill(model, rows, manager, *, checkpoint_len, prefill_step_size=32,
             right_pad=False):
    """Run a real prefill of ``len(rows)`` rows and return the batch.

    ``rows`` are the per-row PREFILL inputs (cold rows, so each is its whole
    prompt).  ``right_pad=True`` builds the mixed-warm/cold geometry with an
    explicit ``right_pad_per_row``; the default is the EQUAL-SUFFIX, zero-padding
    shape -- arm D of the measurement, the one that proves width and not padding
    is the carrier.
    """
    n = max(len(r) for r in rows)
    embeds = mx.zeros((len(rows), n, model.args.hidden_size), dtype=mx.float32)
    meta = [_meta(r, checkpoint_len) for r in rows]
    kw = {}
    if right_pad:
        kw["right_pad_per_row"] = [n - len(r) for r in rows]
        kw["suffix_lens"] = [len(r) for r in rows]
    b = PromptProcessingBatch(
        model=model,
        uids=list(range(len(rows))),
        input_ids=[list(r) for r in rows],
        max_tokens=[4] * len(rows),
        inputs_embeds=embeds,
        prompt_kwargs={},
        prefill_step_size=prefill_step_size,
        apc_meta=meta,
        apc_manager=manager,
        apc_mode="exact",
        **kw,
    )
    while b.needs_processing():
        if b.prompt_step() == 0:
            break
    return b


def _entries(manager):
    return list(manager._exact_cache.values())


# --------------------------------------------------------------------------
# 1. the fields are recorded, and they say the truth about the batch
# --------------------------------------------------------------------------
class TestProvenanceIsRecorded:
    def test_a_b1_checkpoint_records_width_1_and_no_pads(self):
        model = _tiny_lm()
        m = APCManager(num_blocks=8, block_size=16)
        ids = list(range(1, 97))
        _prefill(model, [ids], m, checkpoint_len=64)

        ents = _entries(m)
        assert len(ents) == 1, "the mid-prefill checkpoint fires exactly once"
        p = ents[0].provenance
        assert HP.is_complete(p), p
        assert p["harvest_batch_width"] == 1
        assert p["harvest_right_pad"] == 0
        assert p["harvest_left_pad"] == 0
        assert p["harvest_git_head"] == HP.git_head()
        assert p["harvest_at"] > 0
        assert len(ents[0].token_ids) == 64

    def test_an_equal_suffix_b2_checkpoint_records_width_2_and_no_pads(self):
        """Arm D: the batch that is UNPADDED and still a batch.

        ``harvest_right_pad == 0`` here is the whole point.  A policy that keyed
        on padding would call this harvest clean; the measurement says it is the
        one that poisons.
        """
        model = _tiny_lm()
        m = APCManager(num_blocks=8, block_size=16)
        a = list(range(1, 97))
        b = list(range(200, 296))
        assert len(a) == len(b)
        _prefill(model, [a, b], m, checkpoint_len=64)

        ents = _entries(m)
        assert len(ents) == 2
        for e in ents:
            p = e.provenance
            assert HP.is_complete(p), p
            assert p["harvest_batch_width"] == 2
            assert p["harvest_right_pad"] == 0

    def test_a_right_padded_b2_checkpoint_records_the_rows_own_pad(self):
        model = _tiny_lm()
        m = APCManager(num_blocks=8, block_size=16)
        short = list(range(1, 81))
        long = list(range(200, 296))
        _prefill(model, [short, long], m, checkpoint_len=64, right_pad=True)

        ents = _entries(m)
        assert len(ents) == 2, "both rows reach column 64 before their suffix ends"
        by_head = {e.token_ids[0]: e.provenance for e in ents}
        for prov in by_head.values():
            assert prov["harvest_batch_width"] == 2
        # the short row is 16 columns short of the batch width and says so; the
        # long row sets the width and carries no pad.
        assert by_head[1]["harvest_right_pad"] == 16
        assert by_head[200]["harvest_right_pad"] == 0

    def test_the_end_of_prefill_harvest_also_records_provenance(self):
        """The second store site: ``generate()``'s post-prefill harvest.

        Driven directly rather than through ``generate()`` so the test does not
        depend on a sampler; the store it exercises is the same one.
        """
        model = _tiny_lm()
        m = APCManager(num_blocks=8, block_size=16)
        ids = list(range(1, 97))
        b = _prefill(model, [ids], m, checkpoint_len=0)
        assert _entries(m) == [], "no checkpoint was configured"

        cache = b._apc_prompt_cache_for_store(0)
        assert cache is not None
        m.store_exact_cache(
            ids, cache, harvest_provenance=b._harvest_provenance(0)
        )
        p = _entries(m)[0].provenance
        assert p["harvest_batch_width"] == 1

    def test_an_entry_stored_with_no_provenance_reads_back_as_unknown(self):
        m = APCManager(num_blocks=8, block_size=16)
        ids = list(range(64))
        c = ArraysCache(size=1)
        c[0] = mx.zeros((1, 2, len(ids), 8))
        assert m.store_exact_cache(ids, [c])
        e = _entries(m)[0]
        assert e.provenance is None
        assert HP.batch_width_of(e.provenance) is None
        assert HP.width_key(e.provenance) == "unknown"


# --------------------------------------------------------------------------
# 2. metrics
# --------------------------------------------------------------------------
class TestMetrics:
    def _store(self, m, ids, prov):
        c = ArraysCache(size=1)
        c[0] = mx.zeros((1, 2, max(1, len(ids)), 8))
        return m.store_exact_cache(ids, [c], harvest_provenance=prov)

    def test_the_apc_block_counts_entries_by_harvest_width(self, monkeypatch):
        monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "8")
        m = APCManager(num_blocks=8, block_size=16)
        self._store(m, list(range(0, 64)), HP.make(1))
        self._store(m, list(range(100, 164)), HP.make(2))
        self._store(m, list(range(200, 264)), HP.make(2))
        self._store(m, list(range(300, 364)), None)

        snap = m.stats_snapshot()
        assert snap["exact_entries"] == 4
        assert snap["exact_entries_by_harvest_width"] == {
            "1": 1, "2": 2, "unknown": 1
        }

    def test_the_snapshot_reports_both_policy_settings(self, monkeypatch):
        m = APCManager(num_blocks=8, block_size=16)
        snap = m.stats_snapshot()
        assert snap["serve_b1_from_b1_only"] is False
        assert snap["persist_max_harvest_width"] == 1

        monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "1")
        monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "4")
        snap = m.stats_snapshot()
        assert snap["serve_b1_from_b1_only"] is True
        assert snap["persist_max_harvest_width"] == 4

    def test_the_histogram_is_json_serialisable(self, monkeypatch):
        monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "4")
        m = APCManager(num_blocks=8, block_size=16)
        self._store(m, list(range(64)), HP.make(3))
        # /v1/metrics goes out as JSON; an int key would silently become a
        # string somewhere less visible than here.
        blob = json.dumps(m.stats_snapshot())
        assert '"exact_entries_by_harvest_width": {"3": 1}' in blob


# --------------------------------------------------------------------------
# 3. the server's own log line
# --------------------------------------------------------------------------
class TestLogLine:
    def _log(self, caplog, **kw):
        from mlx_vlm.server.generation import ResponseGenerator

        caplog.set_level(logging.INFO, logger="mlx_vlm.server")
        ResponseGenerator._log_prefill_completed(
            7, {"request_id": "abc"}, PromptProgress(uid=7, **kw)
        )
        return caplog.text

    def test_a_warm_serve_reports_the_width_it_was_harvested_at(self, caplog):
        text = self._log(
            caplog, prompt_tokens=3107, prompt_tps=100.0, prompt_time=1.0,
            cached_tokens=3091, cached_from_width=2,
        )
        assert "cached_tokens=3091 cached_from_width=2" in text

    def test_a_width_1_serve_says_1_not_a_dash(self, caplog):
        text = self._log(
            caplog, prompt_tokens=3107, prompt_tps=100.0, prompt_time=1.0,
            cached_tokens=3091, cached_from_width=1,
        )
        assert "cached_tokens=3091 cached_from_width=1" in text

    def test_a_cold_serve_reports_a_dash(self, caplog):
        text = self._log(
            caplog, prompt_tokens=3107, prompt_tps=100.0, prompt_time=1.0,
            cached_tokens=0, cached_from_width=None,
        )
        assert "cached_tokens=0 cached_from_width=-" in text

    def test_the_width_reaches_the_log_line_from_a_real_lookup(self, monkeypatch):
        """End to end within the desk's reach: store at width 2, look up, and
        confirm the number the log line would print is 2 rather than absent."""
        monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "4")
        m = APCManager(num_blocks=8, block_size=16)
        ids = list(range(128))
        c = ArraysCache(size=1)
        c[0] = mx.zeros((1, 2, 64, 8))
        m.store_exact_cache(ids[:64], [c], harvest_provenance=HP.make(2))

        plan = apc_lookup_plan(
            m, ids, extra_hash=0, apc_mode="exact", safe_lookup_min=0,
            suffix_is_text_only=lambda pl: True,
            prefix_has_media=lambda pl: False,
        )
        assert plan is not None and plan["prefix_len"] == 64
        assert HP.batch_width_of(plan["harvest_provenance"]) == 2


# --------------------------------------------------------------------------
# 4. the determinism knob
# --------------------------------------------------------------------------
class TestServeB1FromB1Only:
    def _mgr(self, monkeypatch):
        monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "8")
        return APCManager(num_blocks=8, block_size=16)

    def _store(self, m, ids, width):
        c = ArraysCache(size=1)
        c[0] = mx.zeros((1, 2, max(1, len(ids)), 8))
        assert m.store_exact_cache(
            ids, [c],
            harvest_provenance=None if width is None else HP.make(width),
        )

    def _plan(self, m, ids, width):
        return apc_lookup_plan(
            m, ids, extra_hash=0, apc_mode="exact", safe_lookup_min=0,
            suffix_is_text_only=lambda pl: True,
            prefix_has_media=lambda pl: False,
            serve_batch_width=width,
        )

    def test_off_by_default_a_b1_request_takes_the_deepest_entry(
        self, monkeypatch
    ):
        m = self._mgr(monkeypatch)
        ids = list(range(256))
        self._store(m, ids[:64], 1)
        self._store(m, ids[:128], 2)
        plan = self._plan(m, ids, 1)
        assert plan["prefix_len"] == 128, "default is throughput: deepest wins"
        assert HP.batch_width_of(plan["harvest_provenance"]) == 2

    def test_on_a_b1_request_falls_through_to_the_deepest_width_1_entry(
        self, monkeypatch
    ):
        m = self._mgr(monkeypatch)
        ids = list(range(256))
        self._store(m, ids[:32], 1)
        self._store(m, ids[:64], 1)
        self._store(m, ids[:128], 2)
        monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "1")
        plan = self._plan(m, ids, 1)
        assert plan is not None
        assert plan["prefix_len"] == 64, "longest strict prefix among width-1"
        assert HP.batch_width_of(plan["harvest_provenance"]) == 1

    def test_on_with_no_width_1_entry_the_request_goes_cold(self, monkeypatch):
        m = self._mgr(monkeypatch)
        ids = list(range(256))
        self._store(m, ids[:128], 2)
        monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "1")
        assert self._plan(m, ids, 1) is None
        # and it says so
        assert m.stats.rejects_by_reason.get("serve_b1_from_b1_only", 0) >= 1

    def test_on_an_unknown_provenance_entry_is_refused_not_assumed(
        self, monkeypatch
    ):
        m = self._mgr(monkeypatch)
        ids = list(range(256))
        self._store(m, ids[:128], None)
        monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "1")
        assert self._plan(m, ids, 1) is None

    def test_on_a_wider_admission_is_unaffected(self, monkeypatch):
        """The knob is about SOLO admissions.  A B=2 admission keeps the fast
        path -- it is already a batched prefill, so refusing a batch-harvested
        entry there would cost throughput for no determinism."""
        m = self._mgr(monkeypatch)
        ids = list(range(256))
        self._store(m, ids[:128], 2)
        monkeypatch.setenv("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", "1")
        plan = self._plan(m, ids, 2)
        assert plan is not None and plan["prefix_len"] == 128

    def _bare_generator(self):
        """A ``BatchGenerator`` built with ``__new__``.

        Its ``__init__`` reads ``mx.device_info()["max_recommended_working_set_size"]``
        (``generate/common.py:198``), a key this venv's mlx does not have -- the
        known CPU-lane artefact that already fails 13 cases in
        ``test_generate.py``.  Only the two attributes the admission path reads
        are set, which is also the sharper test: it cannot pass by accident
        through some other field.
        """
        g = BatchGenerator.__new__(BatchGenerator)
        g.apc_manager = APCManager(num_blocks=8, block_size=16)
        g.vault = None
        g.apc_mode = "exact"
        g._wire_stack = None  # __del__ -> close() reads it
        return g

    def test_the_admission_looks_a_row_up_at_the_window_width(self):
        seen = []
        g = self._bare_generator()
        g._apc_pick_for = lambda s, serve_batch_width=1: seen.append(
            serve_batch_width
        )
        seqs = [
            (0, list(range(64)), 4, {}, None, None),
            (1, list(range(100, 164)), 4, {}, None, None),
        ]
        assert g._build_mixed_prompt_batch(seqs) is None  # no warm row
        assert seen == [2, 2]

        seen.clear()
        assert g._build_mixed_prompt_batch(seqs[:1]) is None
        assert seen == [1]

    def test_the_width_reaches_the_lookup(self, monkeypatch):
        seen = []

        def fake_plan(manager, ids_list, **kw):
            seen.append(kw["serve_batch_width"])
            return None

        monkeypatch.setattr("mlx_vlm.apc.apc_lookup_plan", fake_plan)
        g = self._bare_generator()
        g._apc_extra_hash = lambda kw: 0
        g._apc_safe_prefix_lookup_min = lambda ids: 0
        g._apc_media_token_ids = lambda: []
        seq = (0, list(range(64)), 4, {}, None, None)
        assert g._apc_pick_for(seq, serve_batch_width=3) is None
        assert seen == [3]


# --------------------------------------------------------------------------
# 5. the disk vault
# --------------------------------------------------------------------------
def _glm5_shaped_cache():
    caches = [ArraysCache(size=2) for _ in range(2)]
    caches += [CacheList(KVCache(), KVCache())]
    return caches


def _drive(caches, n_tokens, seed=0):
    for i, c in enumerate(caches):
        if isinstance(c, ArraysCache):
            key = mx.random.key(seed * 131 + i)
            c[0] = mx.random.normal((1, 2, 4, 4), key=key, dtype=mx.float32)
            c[1] = mx.random.normal((1, 2, 4, 4), key=key, dtype=mx.float32)
        else:
            for j, sub in enumerate(c.caches):
                key = mx.random.key(seed * 977 + i * 13 + j)
                k = mx.random.normal((1, 2, n_tokens, 4), key=key)
                v = mx.random.normal((1, 2, n_tokens, 4), key=key)
                sub.update_and_fetch(k, v)
    mx.eval([c[0] for c in caches if isinstance(c, ArraysCache)])
    return caches


def _frags(depth, seed=0):
    return capture_fragments(_drive(_glm5_shaped_cache(), depth, seed), depth)


class _DiskVaultCase:
    @pytest.fixture(autouse=True)
    def _fresh_counters(self):
        # ``DiskPrefixVault`` defaults to the MODULE-GLOBAL ``STATS`` (by
        # design: "the directory is unset" is a state with no vault object to
        # hang a counter on).  Reset around each case so a refusal count is this
        # case's and not the file's running total.
        VD.reset_disk_stats()
        yield
        VD.reset_disk_stats()

    def _vault(self, tmp, cap=1 << 40):
        v = ContextVault(identity="ident-x", budget_bytes=cap)
        dv = VD.DiskPrefixVault(tmp, "ident-x", model_identity="m", git_head="g")
        v.disk = dv
        return v, dv


class TestDiskVaultPersistGate(_DiskVaultCase):
    def test_a_width_2_rung_is_not_persisted_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            v, dv = self._vault(tmp)
            toks = list(range(64))
            v.insert(toks, 32, _frags(32, 1), harvest_provenance=HP.make(2))
            assert dv.save_async(toks, v.lookup(toks)) is False
            dv.flush()
            assert dv.entry_files() == []
            assert dv.stats.disk_refusals.get("harvest_width_not_durable", 0) == 1

    def test_a_width_1_rung_is_persisted_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            v, dv = self._vault(tmp)
            toks = list(range(64))
            v.insert(toks, 32, _frags(32, 2), harvest_provenance=HP.make(1))
            assert dv.save_async(toks, v.lookup(toks)) is True
            dv.flush()
            assert len(dv.entry_files()) == 1

    def test_an_unknown_provenance_rung_is_not_persisted_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            v, dv = self._vault(tmp)
            toks = list(range(64))
            v.insert(toks, 32, _frags(32, 3))
            assert dv.save_async(toks, v.lookup(toks)) is False
            dv.flush()
            assert dv.entry_files() == []

    def test_the_knob_lets_a_wider_harvest_through(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "2")
        with tempfile.TemporaryDirectory() as tmp:
            v, dv = self._vault(tmp)
            toks = list(range(64))
            v.insert(toks, 32, _frags(32, 4), harvest_provenance=HP.make(2))
            assert dv.save_async(toks, v.lookup(toks)) is True
            dv.flush()
            assert len(dv.entry_files()) == 1
            hdr = VD.read_header(dv.entry_files()[0])
            assert hdr["harvest_batch_width"] == 2
            assert hdr["harvest_provenance_complete"] is True

    def test_zero_disables_the_gate_including_for_unknown(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "0")
        with tempfile.TemporaryDirectory() as tmp:
            v, dv = self._vault(tmp)
            toks = list(range(64))
            v.insert(toks, 32, _frags(32, 5))
            assert dv.save_async(toks, v.lookup(toks)) is True
            dv.flush()
            assert len(dv.entry_files()) == 1

    def test_eviction_of_a_width_2_rung_names_its_refusal(self):
        """The eviction hook is the DEFAULT save path.  The rung is dropped, the
        refusal is counted, and nothing raises -- losing it costs one cold
        prefill on a later process; keeping it costs a warm start that is
        bit-different from the cold run and survives every restart."""
        from mlx_vlm.context_vault import fragments_nbytes

        with tempfile.TemporaryDirectory() as tmp:
            frags = _frags(32, 6)
            v, dv = self._vault(tmp, cap=fragments_nbytes(frags) + 8)
            v.insert(list(range(64)), 32, frags, harvest_provenance=HP.make(2))
            v.insert(list(range(100, 164)), 32, _frags(32, 7),
                     harvest_provenance=HP.make(2))
            dv.flush()
            assert v.stats.evictions >= 1
            assert dv.entry_files() == []
            assert dv.stats.disk_refusals.get("harvest_width_not_durable", 0) >= 1


class TestDiskVaultRestoreRefusal(_DiskVaultCase):
    def _write_then_strip(self, tmp, strip: bool):
        v, dv = self._vault(tmp)
        toks = list(range(64))
        v.insert(toks, 32, _frags(32, 8), harvest_provenance=HP.make(1))
        assert dv.save_async(toks, v.lookup(toks))
        dv.flush()
        path = dv.entry_files()[0]
        if strip:
            _rewrite_header(path, drop=HP.FIELDS)
        return dv, toks, path

    def test_a_header_without_provenance_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            dv, toks, path = self._write_then_strip(tmp, strip=True)
            hdr = VD.read_header(path)
            assert "harvest_batch_width" not in hdr
            key = hdr["key"]
            assert dv.load_entry(key, expect_prompt_sha=hdr["prompt_sha256"],
                                 tier="prefill") is None
            assert dv.stats.disk_refusals["harvest_provenance_missing"] == 1

    def test_the_same_blob_with_provenance_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            dv, toks, path = self._write_then_strip(tmp, strip=False)
            hdr = VD.read_header(path)
            got = dv.load_entry(hdr["key"], expect_prompt_sha=hdr["prompt_sha256"],
                                tier="prefill")
            assert got is not None
            assert dv.stats.disk_refusals.get("harvest_provenance_missing", 0) == 0

    def test_the_knob_at_zero_also_relaxes_the_restore_refusal(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            dv, toks, path = self._write_then_strip(tmp, strip=True)
            monkeypatch.setenv("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", "0")
            hdr = VD.read_header(path)
            assert dv.load_entry(hdr["key"], expect_prompt_sha=hdr["prompt_sha256"],
                                 tier="prefill") is not None

    def test_a_restored_rung_keeps_its_provenance_so_it_can_be_resaved(self):
        """Without this, a restore-then-evict cycle launders a width-1 entry
        into an unknown-provenance one and the gate refuses to re-persist the
        very blob it just read."""
        with tempfile.TemporaryDirectory() as tmp:
            v, dv = self._vault(tmp)
            toks = list(range(64))
            v.insert(toks, 32, _frags(32, 9), harvest_provenance=HP.make(1))
            assert dv.save_async(toks, v.lookup(toks))
            dv.flush()

            fresh = ContextVault(identity="ident-x", budget_bytes=1 << 40)
            cp = dv.restore_into_vault(fresh, toks, VaultTier.PREFILL)
            assert cp is not None
            assert HP.batch_width_of(cp.harvest_provenance) == 1


# --------------------------------------------------------------------------
# 6. with everything at its default, nothing moved
# --------------------------------------------------------------------------
class TestDefaultsAreUnchanged:
    def test_a_manifest_with_no_provenance_is_byte_identical(self):
        from mlx_vlm.context_vault_wire import pack_fragments

        frags = _frags(32, 10)
        m_old, p_old = pack_fragments(frags)
        m_new, p_new = pack_fragments(frags, None)
        assert m_old == m_new
        assert "harvest_provenance" not in m_old
        assert mx.array_equal(p_old, p_new)

    def test_a_manifest_with_provenance_carries_it_and_nothing_else_moves(self):
        from mlx_vlm.context_vault_wire import pack_fragments

        frags = _frags(32, 11)
        plain, payload_a = pack_fragments(frags)
        withp, payload_b = pack_fragments(frags, HP.make(2))
        assert withp["harvest_provenance"]["harvest_batch_width"] == 2
        assert withp["offsets"] == plain["offsets"]
        assert withp["tree"] == plain["tree"]
        assert withp["total_bytes"] == plain["total_bytes"]
        assert mx.array_equal(payload_a, payload_b)

    def test_the_serve_path_is_unchanged_with_the_knob_off(self, monkeypatch):
        """The regression this whole change must not cause: with defaults, the
        entry a request gets and the cache it gets back are what they were."""
        monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "8")
        ids = list(range(256))
        results = []
        for prov in (None, HP.make(1), HP.make(2), HP.make(8)):
            m = APCManager(num_blocks=8, block_size=16)
            c = ArraysCache(size=1)
            c[0] = mx.arange(2 * 128 * 8).reshape(1, 2, 128, 8).astype(mx.float32)
            m.store_exact_cache(ids[:128], [c], harvest_provenance=prov)
            warm, plen = m.lookup_exact_cache(ids, extra_hash=0)
            assert warm is not None
            results.append((plen, mx.sum(warm[0][0]).item()))
        assert len(set(results)) == 1, (
            "provenance changed which entry served or what it restored to: "
            f"{results}"
        )

    def test_an_old_style_store_and_lookup_still_works(self):
        m = APCManager(num_blocks=8, block_size=16)
        ids = list(range(128))
        c = ArraysCache(size=1)
        c[0] = mx.zeros((1, 2, 64, 8))
        assert m.store_exact_cache(ids[:64], [c], extra_hash=3)
        warm, plen = m.lookup_exact_cache(ids, extra_hash=3)
        assert plen == 64 and warm is not None

    def test_prompt_progress_defaults_to_no_width(self):
        p = PromptProgress(uid=1, prompt_tokens=10)
        assert p.cached_from_width is None

    def test_a_cold_row_reports_no_width_even_with_entries_around(self):
        model = _tiny_lm()
        m = APCManager(num_blocks=8, block_size=16)
        ids = list(range(1, 97))
        b = _prefill(model, [ids], m, checkpoint_len=64)
        b.record_prompt_time(0.1)
        progress = b.prompt_progress()
        assert len(progress) == 1
        assert progress[0].cached_tokens == 0
        assert progress[0].cached_from_width is None


# --------------------------------------------------------------------------
# 7. L1b1-b: the latent inconsistency, pinned rather than fixed
# --------------------------------------------------------------------------
class TestTheNonRightPadCheckpointBranchStaysUnreachable:
    """``_checkpoint_column_for_len``'s non-right-pad branch disagrees with
    ``_row_real_tokens_processed`` by ``prefix_len`` (open item L1b1-b).

    The column branch returns ``left_pad + target_len``; stopping there makes
    ``_row_real_tokens_processed`` return ``prefix_len + min(suffix_len,
    target_len)``, which equals ``checkpoint_len == target_len`` only when
    ``prefix_len == 0``.  So on a WARM row that branch could place a column the
    store would then never accept, and the checkpoint would silently never fire.

    It is unreachable today, and this test pins WHY rather than fixing the
    arithmetic: every warm row is built by ``_build_mixed_prompt_batch``, which
    always passes ``right_pad_per_row`` as a LIST (``[0]`` at B=1), so the
    right-pad branch governs every warm row that exists.  If that ever stops
    being true this test fails and the arithmetic gets looked at with a
    reproduction in hand -- which is the condition the ruling set for touching
    it.  NOT measured on a box; read from source and pinned here.
    """

    def _batch(self, right_pad, prefix_len):
        b = PromptProcessingBatch.__new__(PromptProcessingBatch)
        b._right_pad_per_row = right_pad
        b._left_padding_per_row = [4]
        b._suffix_lens = [100]
        b._processed_prompt_columns = 0
        b._apc_mode = "exact"
        b._apc_meta = [{"prefix_len": prefix_len, "checkpoint_len": 64}]
        return b

    def test_the_two_agree_on_a_cold_row(self):
        b = self._batch(None, prefix_len=0)
        col = b._checkpoint_column_for_len(0, b._apc_meta[0], 64)
        assert col == 4 + 64
        b._processed_prompt_columns = col
        assert b._row_real_tokens_processed(0) == 64

    def test_the_two_disagree_by_prefix_len_on_a_warm_row(self):
        b = self._batch(None, prefix_len=30)
        col = b._checkpoint_column_for_len(0, b._apc_meta[0], 64)
        assert col == 4 + 64
        b._processed_prompt_columns = col
        # 30 + 64 = 94, not 64: the store's ``!= checkpoint_len`` guard would
        # skip, and the checkpoint would never fire on this branch.
        assert b._row_real_tokens_processed(0) == 94

    def test_the_right_pad_branch_that_actually_governs_agrees(self):
        b = self._batch([0], prefix_len=30)
        col = b._checkpoint_column_for_len(0, b._apc_meta[0], 64)
        assert col == 64 - 30 == 34
        b._processed_prompt_columns = col
        assert b._row_real_tokens_processed(0) == 64

    def test_a_warm_row_is_always_built_with_a_right_pad_list(self):
        """The reachability guard.  ``_build_mixed_prompt_batch`` computes
        ``right_pad_per_row = [max_suffix - s for s in suffix_lens]`` and passes
        it unconditionally, so it is ``[0]`` at B=1 rather than ``None``."""
        import inspect

        from mlx_vlm.generate.ar import BatchGenerator

        src = inspect.getsource(BatchGenerator._build_mixed_prompt_batch)
        assert "right_pad_per_row = [max_suffix_len - s for s in suffix_lens]" in src
        assert "right_pad_per_row=right_pad_per_row" in src
        assert "right_pad_per_row=None" not in src


# --------------------------------------------------------------------------
# helpers that rewrite a blob header in place (tests only)
# --------------------------------------------------------------------------
def _rewrite_header(path: Path, *, drop=()) -> None:
    """Drop fields from a written entry's JSON header, keeping the layout.

    The header area is padded to ``tokens_offset``, so a SHORTER header fits
    without moving a byte of the tokens or the payload.  That is what makes this
    a faithful stand-in for a blob written by a build that predates the field.
    """
    raw = bytearray(Path(path).read_bytes())
    (hlen,) = struct.unpack("<Q", bytes(raw[8:16]))
    hdr = json.loads(bytes(raw[16 : 16 + hlen]).decode("utf-8"))
    for f in drop:
        hdr.pop(f, None)
    hdr.pop("harvest_provenance_complete", None)
    blob = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    assert len(blob) <= hlen, "a shorter header must fit the reserved area"
    blob = blob + b" " * (hlen - len(blob))  # JSON ignores trailing space
    raw[16 : 16 + hlen] = blob
    Path(path).write_bytes(bytes(raw))
