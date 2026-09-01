"""The Warm Context Vault on the SERVER's generation path.

Until now the vault existed only on ``generate/dispatch.py::stream_generate``.
The server's continuous-batching path builds its own ``BatchGenerator`` and
never saw it, so a server that had the toggle on still cold-prefilled every
request.  These tests pin the two halves of the wiring and, just as important,
pin that with the vault OFF nothing about the old path moved.

What is deliberately NOT covered here, because it is not wired:
  * the speculative loop (``SpeculativeGenerationBatch``) -- a separate class
    with its own prompt processing;
  * TP=2 -- rung stores and restores have to be announced to rank 1 over the
    control collective, and the server request path does not do that, so
    ``_build_vault`` refuses under TP.
"""

from __future__ import annotations

from unittest.mock import Mock

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import BatchGenerator, PromptProcessingBatch


# --------------------------------------------------------------------------
# A vault stand-in that records what it is asked to do.  Real ContextVault
# behaviour is covered by test_context_vault.py; what these tests are about is
# whether the server path CALLS it, at the right lengths, with the right rows.
# --------------------------------------------------------------------------
class _Hit:
    def __init__(self, prefix_len):
        self.prefix_len = prefix_len


class _SpyVault:
    def __init__(self, hit_at=None, restore_ok=True):
        self.inserted = []          # (tuple(full_ids), prefix_len, frags is None)
        self.lookups = []
        self._hit_at = hit_at
        self._restore_ok = restore_ok

    def lookup(self, tokens):
        self.lookups.append(list(tokens))
        return _Hit(self._hit_at) if self._hit_at else None

    def restore_into(self, caches, checkpoint):
        return self._restore_ok

    def insert(self, tokens, prefix_len, fragments):
        self.inserted.append((tuple(tokens), int(prefix_len), fragments is None))
        return True


def _tiny_lm(nope: bool = True):
    """A 4-layer Qwen3.5 LM: mixed ArraysCache + KVCache, so apc_mode is 'exact'.

    ``nope=True`` drops ``get_rope_index`` from the instance, which is how a
    NoPE model such as GLM-5-Next presents: nothing has to be primed before the
    prompt is trimmed to the uncached suffix.  Qwen3.5 really does carry mRoPE
    metadata, so it is the right stand-in for both sides of that guard -- pass
    ``nope=False`` to get the model that must be refused.
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
    model = qwen3_5.LanguageModel(text_config, config)
    if nope:
        model.get_rope_index = None
    return model


def _proc():
    p = Mock()
    p.tokenizer = Mock()
    p.tokenizer.stopping_criteria = Mock()
    p.tokenizer.stopping_criteria.add_eos_token_ids = Mock()
    return p


def _gen(model, vault=None, **kw):
    return BatchGenerator(model, _proc(), vault=vault, **kw)


# ------------------------------------------------------------------ rungs
def test_rung_ladder_is_the_dispatch_ladder_shifted_past_the_warm_prefix():
    model = _tiny_lm()
    from mlx_vlm.context_vault import boundary_ladder

    g = _gen(model, vault=_SpyVault(), prefill_step_size=64)
    n = 32768
    expected = boundary_ladder(n, step=64)
    assert len(expected) >= 2, expected
    assert g._vault_rungs_for(list(range(n)), 0) == expected
    # a warm start drops the rungs it already contains
    cut = expected[len(expected) // 2]
    assert g._vault_rungs_for(list(range(n)), cut) == [b for b in expected if b > cut]


def test_no_vault_means_no_rungs():
    g = _gen(_tiny_lm(), vault=None, prefill_step_size=64)
    assert g._vault_rungs_for(list(range(32768)), 0) == []


# ------------------------------------------------------------------ lookup
def test_vault_supplies_a_warm_start_when_apc_is_off():
    model = _tiny_lm()
    v = _SpyVault(hit_at=64)
    g = _gen(model, vault=v, prefill_step_size=32)
    ids = list(range(1, 257))
    pick = g._apc_pick_for((0, ids, 16, {}, None, None))
    assert pick is not None, "a vault hit must produce a plan with APC off"
    assert pick["prefix_len"] == 64
    assert pick["source"] == "vault"
    assert pick["matched_blocks"] == []
    assert pick["warm_cache"] is not None
    assert v.lookups == [ids]


def test_a_vault_miss_leaves_the_request_cold():
    g = _gen(_tiny_lm(), vault=_SpyVault(hit_at=None), prefill_step_size=32)
    assert g._apc_pick_for((0, list(range(1, 257)), 16, {}, None, None)) is None


def test_a_failed_restore_leaves_the_request_cold():
    g = _gen(_tiny_lm(), vault=_SpyVault(hit_at=64, restore_ok=False),
             prefill_step_size=32)
    assert g._apc_pick_for((0, list(range(1, 257)), 16, {}, None, None)) is None


def test_a_rung_that_covers_the_whole_prompt_is_refused():
    """Restoring the entire prompt would leave zero tokens to prefill."""
    ids = list(range(1, 129))
    g = _gen(_tiny_lm(), vault=_SpyVault(hit_at=len(ids)), prefill_step_size=32)
    assert g._apc_pick_for((0, ids, 16, {}, None, None)) is None


def test_the_vault_only_deepens_an_apc_hit_never_shallows_it():
    """A shallower rung than APC already found must lose."""
    model = _tiny_lm()
    g = _gen(model, vault=_SpyVault(hit_at=32), prefill_step_size=32)
    ids = list(range(1, 257))
    deeper = {"matched_blocks": [], "warm_cache": None, "prefix_len": 128,
              "extra_hash": 0, "full_input_ids": ids}
    assert g._vault_pick_for(ids, {}, deeper) is deeper


def test_a_model_needing_rope_priming_is_refused():
    """dispatch.py primes _rope_deltas from the full prompt before trimming;
    this path has no such hook, so a model that needs it must not be trimmed.

    Qwen3.5 is not a hypothetical here -- it carries mRoPE metadata, and the
    unmodified model is what this refuses.
    """
    model = _tiny_lm(nope=False)
    assert callable(getattr(model, "get_rope_index", None))
    g = _gen(model, vault=_SpyVault(hit_at=64), prefill_step_size=32)
    assert g._apc_pick_for((0, list(range(1, 257)), 16, {}, None, None)) is None


# ------------------------------------------------------- checkpoint columns
def _batch(model, ids, vault, prefill_step_size=32, rungs=None):
    """A one-row PromptProcessingBatch that will actually run forwards.

    These tests drive real prefill chunks, so the model here is the unmodified
    Qwen3.5 (``nope=False``): the lookup tests above stub ``get_rope_index`` off
    to stand in for a NoPE model, and that stub is not something the model's own
    ``__call__`` survives.
    """
    n = len(ids)
    embeds = mx.zeros((1, n, model.args.hidden_size), dtype=mx.float32)
    meta = [{
        "full_input_ids": list(ids),
        "prefix_len": 0,
        "extra_hash": 0,
        "apc_blocks": [],
        "checkpoint_len": 0,
        "vault_rungs": list(rungs if rungs is not None else []),
    }]
    return PromptProcessingBatch(
        model=model, uids=[0], input_ids=[list(ids)], max_tokens=[4],
        inputs_embeds=embeds, prompt_kwargs={},
        prefill_step_size=prefill_step_size,
        apc_meta=meta, apc_manager=None, apc_mode="exact", vault=vault,
    )


def test_chunking_lands_exactly_on_each_rung():
    model = _tiny_lm(nope=False)
    ids = list(range(1, 129))
    v = _SpyVault()
    b = _batch(model, ids, v, prefill_step_size=64, rungs=[32, 96])
    assert b._next_apc_checkpoint_column() == 32
    steps = []
    while b.needs_processing():
        got = b.prompt_step()
        if got == 0:
            break
        steps.append(got)
    # A short first chunk to land exactly on rung 32, then a full 64 that lands
    # exactly on rung 96.  ``prompt_step`` stops once the remainder fits in one
    # step and no rung is pending -- the tail belongs to ``generate()``.
    assert steps == [32, 64]
    assert [pl for _, pl, _ in v.inserted] == [32, 96]
    assert all(t == tuple(ids) for t, _, _ in v.inserted)
    assert all(frags_none is False for _, _, frags_none in v.inserted), (
        "capture_fragments returned None -- a rung was stored empty"
    )


def test_a_rung_is_stored_once():
    model = _tiny_lm(nope=False)
    ids = list(range(1, 129))
    v = _SpyVault()
    b = _batch(model, ids, v, prefill_step_size=32, rungs=[32])
    while b.needs_processing():
        if b.prompt_step() == 0:
            break
    assert [pl for _, pl, _ in v.inserted] == [32]


# ------------------------------------------------------- the OFF guarantee
def test_with_the_vault_off_the_checkpoint_column_is_unchanged():
    """The whole point of gating: a vault-less batch must compute exactly the
    column it computed before, including when rung metadata is somehow present.
    """
    model = _tiny_lm(nope=False)
    ids = list(range(1, 129))
    off = _batch(model, ids, None, prefill_step_size=64, rungs=[32, 96])
    assert off._next_apc_checkpoint_column() is None
    steps = []
    while off.needs_processing():
        got = off.prompt_step()
        if got == 0:
            break
        steps.append(got)
    # Plain prefill_step_size chunking, undisturbed by the rung metadata that is
    # sitting right there in the meta -- one 64 and then the remainder is small
    # enough that prompt_step is done.  The vault-on run above splits the same
    # prompt into [32, 64] instead, which is exactly the difference being gated.
    assert steps == [64]


def test_with_the_vault_off_nothing_is_stored():
    model = _tiny_lm(nope=False)
    v = _SpyVault()
    b = _batch(model, list(range(1, 129)), None, prefill_step_size=32, rungs=[32])
    while b.needs_processing():
        if b.prompt_step() == 0:
            break
    assert v.inserted == []


# ------------------------------------------------------------ server wiring
def test_server_refuses_the_vault_under_tp(monkeypatch):
    import mlx_vlm.server.generation as sg
    import mlx_vlm.context_vault as cv

    monkeypatch.setattr(cv, "vault_enabled", lambda: True)
    monkeypatch.setattr("mlx_vlm.server.tp_mode.tp_enabled", lambda: True)
    assert sg.ResponseGenerator._build_vault(Mock()) is None


def test_server_builds_no_vault_when_the_toggle_is_off(monkeypatch):
    import mlx_vlm.server.generation as sg
    import mlx_vlm.context_vault as cv

    monkeypatch.setattr(cv, "vault_enabled", lambda: False)
    assert sg.ResponseGenerator._build_vault(Mock()) is None
