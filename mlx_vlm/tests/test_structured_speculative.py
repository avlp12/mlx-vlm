"""Structured output x DFlash speculative decoding (D1-D8).

What is being pinned down
-------------------------
R1, the bug that motivated the whole change: ``generate_step`` applied the
request's logits processors to the FIRST bonus token and then handed the round
loop nothing, and ``PromptProcessingBatch`` did not pass ``logits_processors``
to ``SpeculativeGenerationBatch`` at all.  A structured request served with a
draft model therefore emitted one constrained token followed by an unconstrained
stream -- silently.  A grammar processor is now threaded through the round loop
and the shapes that cannot be served refuse by name.

The rail is ON by default as of the LU-panel promotion (text sha 4/4 identical
to the AR+mask reference, 2.2x per token, natural panel unchanged).
``MLX_VLM_SPEC_STRUCTURED=0`` is the only escape hatch, and it is exact: the
gate returns before it looks at the processor list, no ledger is built, nothing
new can raise, and the server's original refusal message comes back verbatim.
``test_toggle_off_*`` are the guards on that, and the strongest of them compare
emitted token streams, not exception types.

A request with no grammar processor is not a structured request: penalty-only
processors pass through the gate untouched in either configuration, so the
promotion does not turn ``repetition_penalty`` + a draft model into a failure.

Everything below runs on stub targets and a ``StubLedger`` -- a scripted
legal-set schedule with the same interface as the llguidance-backed ledger -- so
the exactness properties are checked without a model or llguidance.  The
llguidance-backed identities at the bottom skip when the library or the
tokenizer is not available.

The reference every emitted-stream test compares against is the MASKED
autoregressive chain: one token at a time, grammar mask applied to the target's
own logits, no drafter anywhere.  That is the answer the server owes the client;
speculation must not move it.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate.ar import _PositionedTargetSampler, SpeculativeGenerationBatch
from mlx_vlm.server import generation as server_generation
from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.speculative import structured_ledger as SL
from mlx_vlm.speculative import utils as spec_utils
from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig
from mlx_vlm.speculative.structured_ledger import (
    SPEC_STRUCTURED_ENV,
    StructuredLedgerError,
    StructuredSpeculationRefused,
    StubLedger,
    apply_block_mask,
    spec_structured_enabled,
    unpack_bitmask,
)

SEED = 20260906
VOCAB = 24
HIDDEN = 4
THINK_END = 19


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(SPEC_STRUCTURED_ENV, raising=False)
    SL._reset_env_warnings()
    yield
    SL._reset_env_warnings()


# --------------------------------------------------------------------------
# stub target / drafters
# --------------------------------------------------------------------------
class _Markov1Target:
    """Logits depend only on the immediately previous token.

    For a Markov-1 target the masked chain ``x_i = argmax(mask_i + f(x_{i-1}))``
    is fully determined by the seed, so ANY drafter -- or none -- must reproduce
    it.  Same trick ``test_sampling_coupling`` uses, with the mask added.
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.table = mx.array((rng.normal(size=(VOCAB, VOCAB)) * 2.0).astype(np.float32))
        self.forwards = 0

    def logits_for(self, token_ids):
        return self.table[token_ids]

    def __call__(self, ids, cache=None, **kw):
        self.forwards += 1
        length = int(ids.shape[1])
        return SimpleNamespace(
            logits=self.logits_for(ids),
            hidden_states=[mx.zeros((1, length, HIDDEN))],
            gdn_states=["gdn"],
        )

    def rollback_speculative_cache(self, *args, **kwargs):
        return 0


class _StubDrafter:
    """DFlash2's contract: one proposal per block position, from the anchor."""

    def __init__(self, target, kind="oracle", block_size=8, seed=0):
        self.target = target
        self.kind = kind
        self.config = SimpleNamespace(
            target_layer_ids=[0],
            block_size=block_size,
            runtime_block_size=block_size,
            vocab_size=VOCAB,
        )
        self.accept_lens = []
        self.draft_lens = []
        self.dflash_deferred_walk = True
        self.draft_block_calls = 0
        self.masks_seen = []
        rng = np.random.default_rng(1000 + seed)
        self.noise = mx.array((rng.normal(size=(VOCAB, VOCAB)) * 2.0).astype(np.float32))

    def reset(self, model):
        return ["draft-cache"]

    def _scores(self, previous):
        table = self.target.table if self.kind == "oracle" else self.noise
        return table[mx.array([previous], dtype=mx.int32)]

    def draft_block(self, last_bonus, hidden, cache, bs, sampler, token_dtype, **kw):
        self.draft_block_calls += 1
        self.masks_seen.append(kw.get("structured_position0_mask"))
        previous = (
            int(last_bonus)
            if isinstance(last_bonus, int)
            else int(last_bonus.reshape(-1)[0])
        )
        propose = getattr(sampler, "sample_proposal", None)
        tokens = []
        for _ in range(bs - 1):
            scores = self._scores(previous)
            drawn = propose(scores) if callable(propose) else sampler(scores)
            previous = int(mx.array(drawn).reshape(-1)[0])
            tokens.append(previous)
        return mx.array([tokens], dtype=token_dtype)


class _ScriptedDrafter(_StubDrafter):
    """Proposes a fixed row every round -- used to plant an illegal draft."""

    def __init__(self, target, row, block_size=8):
        super().__init__(target, kind="noise", block_size=block_size)
        self.row = list(row)

    def draft_block(self, last_bonus, hidden, cache, bs, sampler, token_dtype, **kw):
        self.draft_block_calls += 1
        self.masks_seen.append(kw.get("structured_position0_mask"))
        return mx.array([self.row[: bs - 1]], dtype=token_dtype)


def _greedy_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _legal_first_bonus(ledger, default=3):
    """The prefill masked the first bonus, so it is grammar-legal by
    construction; the stub runs have to honour that."""
    if not ledger.active:
        return default
    legal = ledger.legal_at(())
    return default if legal is None else legal[0]


def _run_rounds(
    drafter,
    ledger,
    *,
    sampler=_greedy_sampler,
    greedy=True,
    max_tokens=40,
    block_size=8,
    first_bonus=None,
):
    if first_bonus is None:
        first_bonus = _legal_first_bonus(ledger)
    model = SimpleNamespace(language_model=drafter.target)
    rounds = dflash_utils._dflash_rounds(
        model,
        drafter,
        [SimpleNamespace(offset=0)],
        mx.zeros((1, 1, HIDDEN)),
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_block_size=block_size,
        use_model_initial_block_size=False,
        greedy_sampling=greedy,
        structured_ledger=ledger,
    )
    tokens = [first_bonus]
    try:
        for tok, _ in rounds:
            tokens.append(int(tok))
    finally:
        rounds.close()
    return tokens


# --------------------------------------------------------------------------
# masked autoregressive references (no drafter at all)
# --------------------------------------------------------------------------
def _masked_greedy_reference(target, ledger, count, first_bonus=None):
    if first_bonus is None:
        first_bonus = _legal_first_bonus(ledger)
    tokens = [first_bonus]
    ledger.commit([first_bonus])
    while len(tokens) < count:
        logits = target.logits_for(mx.array([tokens[-1]], dtype=mx.int32))
        masked = apply_block_mask(logits, ledger.next_token_mask())
        token = int(mx.argmax(masked, axis=-1).reshape(-1)[0])
        tokens.append(token)
        ledger.commit([token])
    return tokens


def _masked_sampled_reference(target, ledger, sampler, count, first_bonus=None):
    if first_bonus is None:
        first_bonus = _legal_first_bonus(ledger)
    tokens = [first_bonus]
    ledger.commit([first_bonus])
    position = 1
    while len(tokens) < count:
        logits = target.logits_for(mx.array([tokens[-1]], dtype=mx.int32))
        masked = apply_block_mask(logits, ledger.next_token_mask())
        logprobs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
        drawn = sampler.sample_target(logprobs, row_ids=[0], positions=[position])
        token = int(mx.array(drawn).reshape(-1)[0])
        tokens.append(token)
        ledger.commit([token])
        position += 1
    return tokens


# --------------------------------------------------------------------------
# legal-set schedules
# --------------------------------------------------------------------------
def _mixed_legal(prefix):
    """Bimodal, like a real grammar: some positions forced, some open.

    Every third position is a single forced token (the fast-forward case); the
    rest allow a four-token set.  Keyed off the prefix LENGTH so the schedule is
    a pure function of how far the grammar has come, which is what the ledger
    tracks.
    """
    step = len(prefix)
    if step % 3 == 2:
        return [(step * 7 + 5) % VOCAB]
    return sorted({(step * 7 + 5 * i) % VOCAB for i in range(4)})


def _open_legal(prefix):
    step = len(prefix)
    return sorted({(step * 11 + 3 * i) % VOCAB for i in range(5)})


def _all_allow(prefix):
    return None


# --------------------------------------------------------------------------
# D1 -- the toggle
# --------------------------------------------------------------------------
def test_the_rail_is_on_by_default_and_empty_counts_as_unset():
    """Promoted on the LU panel: unset means ON.  Only an explicit falsy value
    turns it off, and empty/whitespace is still "unset", not "off"."""
    assert spec_structured_enabled({}) is True
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: ""}) is True
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: "   "}) is True
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: "1"}) is True
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: "TRUE"}) is True
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: " on "}) is True
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: "0"}) is False
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: "off"}) is False
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: " FALSE "}) is False
    assert spec_structured_enabled({SPEC_STRUCTURED_ENV: "no"}) is False


def test_an_invalid_toggle_warns_once_and_falls_back_to_the_default(caplog):
    with caplog.at_level("WARNING", logger="mlx_vlm.speculative.structured_ledger"):
        for _ in range(5):
            assert spec_structured_enabled({SPEC_STRUCTURED_ENV: "maybe"}) is True
    warnings = [r for r in caplog.records if "maybe" in r.getMessage()]
    assert len(warnings) == 1
    # the warning has to name the value it actually fell back to
    assert "'1'" in warnings[0].getMessage()


# --------------------------------------------------------------------------
# D1/D3/D7 + R1 -- the refusals
# --------------------------------------------------------------------------
class _FakeGrammarProcessor:
    """Duck-typed ``LLGuidanceLogitsProcessor``: grammar + llg_tokenizer."""

    grammar = "root ::= 'x'"
    llg_tokenizer = SimpleNamespace(vocab_size=VOCAB)

    def __call__(self, input_ids, logits):  # pragma: no cover - never called
        return logits


def _refusal(**kwargs):
    defaults = dict(
        batch_size=1,
        draft_kind="dflash",
        call_site="test",
    )
    defaults.update(kwargs)
    processors = defaults.pop("processors", [_FakeGrammarProcessor()])
    return SL.resolve_structured_processor(processors, **defaults)


def test_no_processors_means_nothing_to_thread_and_no_refusal():
    assert _refusal(processors=[]) is None
    assert _refusal(processors=None) is None
    assert _refusal(processors=[None]) is None


def test_the_toggle_off_gate_never_raises_whatever_it_is_handed(monkeypatch):
    """D1's hard rule: ``MLX_VLM_SPEC_STRUCTURED=0`` == the code that shipped.

    Every shape that refuses with the rail on -- a grammar processor, a penalty,
    both, B > 1, MTP, eagle3 -- must pass straight through here and return
    ``None``, so the round loop takes exactly the branch it took at base
    71732451.  (R1's silent partial constraint is still there in that
    configuration; the variable is the only thing that changes behaviour.)
    """
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "0")

    def penalty(input_ids, logits):  # pragma: no cover - never called
        return logits

    assert _refusal() is None
    assert _refusal(processors=[penalty]) is None
    assert _refusal(processors=[penalty, _FakeGrammarProcessor()]) is None
    two_grammars = [_FakeGrammarProcessor(), _FakeGrammarProcessor()]
    assert _refusal(processors=two_grammars) is None
    assert _refusal(batch_size=8) is None
    assert _refusal(draft_kind="mtp") is None
    assert _refusal(draft_kind="eagle3") is None
    assert _refusal(draft_kind=None) is None


def test_the_toggle_off_gate_does_not_even_look_at_the_processors(monkeypatch):
    """Not just "does not raise" -- does not touch them.

    A processor whose attribute access explodes must survive the gate, which is
    the mechanical statement of "no new code runs under
    ``MLX_VLM_SPEC_STRUCTURED=0``".
    """
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "0")

    class _Landmine:
        def __getattr__(self, name):  # pragma: no cover - the point is it never runs
            raise AssertionError(f"the toggle-off gate inspected .{name}")

    assert _refusal(processors=[_Landmine()]) is None


def test_a_penalty_only_request_is_not_this_features_business(monkeypatch):
    """Load-bearing now that the rail is ON by default.

    A request with no grammar processor is not a structured request, so the gate
    hands it back untouched and it keeps whatever the speculative path did with
    it before.  Refusing here would turn every ``repetition_penalty`` request
    served with a draft model into a hard failure, which is not what the panel
    promoted -- only ``grammar + penalty`` refuses.
    """
    monkeypatch.delenv(SPEC_STRUCTURED_ENV, raising=False)
    assert spec_structured_enabled() is True

    def penalty(input_ids, logits):  # pragma: no cover - never called
        return logits

    assert _refusal(processors=[penalty]) is None
    assert _refusal(processors=[penalty, penalty]) is None
    # and it stays out of the way for every drafter kind and batch size
    assert _refusal(processors=[penalty], draft_kind="mtp") is None
    assert _refusal(processors=[penalty], batch_size=8) is None


def test_b_greater_than_one_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "1")
    with pytest.raises(StructuredSpeculationRefused) as excinfo:
        _refusal(batch_size=4)
    message = str(excinfo.value)
    assert "batch_size=4" in message
    assert SPEC_STRUCTURED_ENV in message


def test_mtp_with_structured_is_refused(monkeypatch):
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "1")
    with pytest.raises(StructuredSpeculationRefused) as excinfo:
        _refusal(draft_kind="mtp")
    assert "MTP" in str(excinfo.value)


def test_eagle3_with_structured_is_refused(monkeypatch):
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "1")
    with pytest.raises(StructuredSpeculationRefused) as excinfo:
        _refusal(draft_kind="eagle3")
    assert "dflash" in str(excinfo.value)


def test_a_penalty_processor_alongside_the_grammar_is_refused(monkeypatch):
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "1")

    def penalty(input_ids, logits):  # pragma: no cover - never called
        return logits

    with pytest.raises(StructuredSpeculationRefused) as excinfo:
        _refusal(processors=[penalty, _FakeGrammarProcessor()])
    assert "silently dropped" in str(excinfo.value)


def test_the_grammar_processor_is_returned_with_the_toggle_on(monkeypatch):
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "1")
    processor = _FakeGrammarProcessor()
    assert _refusal(processors=[processor]) is processor
    # the batch path's list-of-per-row-lists shape resolves the same way
    assert _refusal(processors=[[processor]]) is processor


def _run_entry_point(
    drafter, logits_processors, *, max_tokens=24, block_size=8, bonus=3
):
    """Drive the real ``generate_step`` entry point, not ``_dflash_rounds``.

    This is the path that used to drop the list on the floor, so it is the one
    the toggle-off equivalence has to be measured on.
    """
    rounds = spec_utils.run_speculative_rounds(
        SimpleNamespace(language_model=drafter.target),
        drafter,
        [SimpleNamespace(offset=0)],
        mx.zeros((1, 1), dtype=mx.int32),
        mx.array([bonus], dtype=mx.int32),
        mx.zeros((1, VOCAB)),
        SimpleNamespace(hidden_states=[mx.zeros((1, 1, HIDDEN))]),
        draft_kind="dflash",
        max_tokens=max_tokens,
        sampler=_greedy_sampler,
        draft_block_size=block_size,
        sampler_is_greedy=True,
        logits_processors=logits_processors,
    )
    tokens = []
    try:
        for tok, _ in rounds:
            tokens.append(int(tok))
    finally:
        rounds.close()
    return tokens


def test_toggle_off_a_penalty_processor_emits_exactly_the_base_stream(monkeypatch):
    """THE regression guard the whole gate exists for.

    With ``MLX_VLM_SPEC_STRUCTURED=0``, ``generate_step``'s speculative branch
    handed a penalty processor -- or a grammar one -- emits the same tokens as
    the same run with no processor at all, which is what base 71732451 does,
    because it dropped the list here.  No refusal, no ledger, no behaviour
    difference.  This is the escape hatch the promotion has to keep intact.
    """
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "0")

    def penalty(input_ids, logits):  # pragma: no cover - never called
        return logits

    base = _run_entry_point(
        _StubDrafter(_Markov1Target(seed=21), "noise", seed=21), None
    )
    with_penalty = _run_entry_point(
        _StubDrafter(_Markov1Target(seed=21), "noise", seed=21), [penalty]
    )
    with_grammar = _run_entry_point(
        _StubDrafter(_Markov1Target(seed=21), "noise", seed=21),
        [_FakeGrammarProcessor()],
    )
    assert base
    assert with_penalty == base
    assert with_grammar == base


def test_toggle_off_the_batch_path_emits_exactly_the_base_stream(monkeypatch):
    """Same equivalence on ``run_speculative_server_rounds`` (the batch side)."""
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "0")

    def penalty(input_ids, logits):  # pragma: no cover - never called
        return logits

    def _run(processors):
        drafter = _StubDrafter(_Markov1Target(seed=22), "noise", seed=22)
        rounds = spec_utils.run_speculative_server_rounds(
            SimpleNamespace(language_model=drafter.target),
            drafter,
            [SimpleNamespace(offset=0)],
            mx.zeros((1, 1, HIDDEN)),
            draft_kind="dflash",
            first_bonus=mx.array([3], dtype=mx.int32),
            max_tokens=24,
            sampler=_greedy_sampler,
            draft_block_size=8,
            greedy_sampling=True,
            logits_processors=processors,
        )
        tokens = []
        try:
            for toks, _ in rounds:
                tokens.extend(int(t) for t in toks)
        finally:
            rounds.close()
        return tokens

    base = _run(None)
    assert base
    assert _run([[penalty]]) == base
    assert _run([[_FakeGrammarProcessor()]]) == base


def _speculative_batch(processors, **overrides):
    kwargs = dict(
        model=SimpleNamespace(),
        draft_model=SimpleNamespace(),
        draft_kind="dflash",
        uids=[0],
        first_tokens=mx.array([3], dtype=mx.int32),
        prompt_cache=[],
        sampler=_greedy_sampler,
        stop_criteria=lambda t: False,
        max_tokens=[8],
        hidden=mx.zeros((1, 1, HIDDEN)),
        shared_kv_states=None,
        prompt_tokens=mx.zeros((1, 1), dtype=mx.int32),
        logits_processors=processors,
    )
    kwargs.update(overrides)
    return SpeculativeGenerationBatch(**kwargs)


def test_toggle_off_the_batch_constructor_accepts_anything(monkeypatch):
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "0")

    def penalty(input_ids, logits):  # pragma: no cover - never called
        return logits

    assert _speculative_batch([[_FakeGrammarProcessor()]]) is not None
    assert _speculative_batch([[penalty]]) is not None
    assert _speculative_batch([[_FakeGrammarProcessor()]], draft_kind="mtp") is not None


def test_by_default_the_batch_constructor_refuses_an_unsupported_drafter():
    """The mirror image with the rail on (no env set at all)."""
    with pytest.raises(StructuredSpeculationRefused):
        _speculative_batch([[_FakeGrammarProcessor()]], draft_kind="mtp")


def test_by_default_a_penalty_only_batch_is_still_admitted():
    def penalty(input_ids, logits):  # pragma: no cover - never called
        return logits

    assert _speculative_batch([[penalty]]) is not None


def test_the_single_stream_entry_point_refuses_an_unsupported_shape(monkeypatch):
    """``run_speculative_rounds`` -- the ``generate_step`` side, toggle ON."""
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "1")
    rounds = spec_utils.run_speculative_rounds(
        SimpleNamespace(),
        SimpleNamespace(),
        [],
        mx.zeros((1, 1), dtype=mx.int32),
        mx.array([3], dtype=mx.int32),
        mx.zeros((1, VOCAB)),
        None,
        draft_kind="mtp",
        max_tokens=8,
        sampler=_greedy_sampler,
        logits_processors=[_FakeGrammarProcessor()],
    )
    with pytest.raises(StructuredSpeculationRefused):
        next(rounds)


def test_the_server_entry_point_refuses_an_unsupported_shape(monkeypatch):
    """``run_speculative_server_rounds`` -- the batch side, toggle ON."""
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "1")
    rounds = spec_utils.run_speculative_server_rounds(
        SimpleNamespace(),
        SimpleNamespace(),
        [],
        mx.zeros((1, 1, HIDDEN)),
        draft_kind="dflash",
        first_bonus=mx.array([3, 4], dtype=mx.int32),
        max_tokens=8,
        sampler=_greedy_sampler,
        logits_processors=[[_FakeGrammarProcessor()], [_FakeGrammarProcessor()]],
    )
    with pytest.raises(StructuredSpeculationRefused):
        next(rounds)


def test_speculative_generation_batch_refuses_at_construction(monkeypatch):
    """Toggle ON, B > 1: fail where the batch was admitted, not on next()."""
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "1")
    with pytest.raises(StructuredSpeculationRefused):
        _speculative_batch(
            [[_FakeGrammarProcessor()], [_FakeGrammarProcessor()]],
            uids=[0, 1],
            first_tokens=mx.array([3, 4], dtype=mx.int32),
            max_tokens=[8, 8],
        )


def _server_stub(draft_model, draft_kind="dflash"):
    return SimpleNamespace(
        wait_until_ready=lambda: None,
        draft_model=draft_model,
        draft_kind=draft_kind,
    )


def _server_generate(stub, args):
    return server_generation.ResponseGenerator.generate(stub, "hello", args=args)


def _grammar_args():
    return server_generation.GenerationArguments(
        max_tokens=8, logits_processors=[_FakeGrammarProcessor()]
    )


def test_the_server_refusal_message_is_byte_identical_with_the_toggle_off(monkeypatch):
    """``MLX_VLM_SPEC_STRUCTURED=0`` restores the pre-promotion server exactly,
    message included -- clients that key on the string keep working."""
    monkeypatch.setenv(SPEC_STRUCTURED_ENV, "0")
    with pytest.raises(ValueError) as excinfo:
        _server_generate(_server_stub(SimpleNamespace()), _grammar_args())
    assert str(excinfo.value) == (
        "Structured response_format is not supported with speculative decoding."
    )


def test_the_server_serves_structured_speculative_requests_by_default(monkeypatch):
    """The promotion: a dflash server takes a structured request with no env set.

    It still fails further in (the stub is not a real server), but it must be
    past the structured gate -- neither the old refusal nor a new one.
    """
    monkeypatch.delenv(SPEC_STRUCTURED_ENV, raising=False)
    with pytest.raises(Exception) as excinfo:
        _server_generate(_server_stub(SimpleNamespace()), _grammar_args())
    message = str(excinfo.value)
    assert not isinstance(excinfo.value, StructuredSpeculationRefused)
    assert "not supported with speculative decoding" not in message


@pytest.mark.parametrize("draft_kind", ["mtp", "eagle3", "lookup"])
def test_the_server_refuses_an_unsupported_drafter_kind_by_name(
    monkeypatch, draft_kind
):
    monkeypatch.delenv(SPEC_STRUCTURED_ENV, raising=False)
    with pytest.raises(StructuredSpeculationRefused) as excinfo:
        _server_generate(_server_stub(SimpleNamespace(), draft_kind), _grammar_args())
    message = str(excinfo.value)
    if draft_kind == "mtp":
        assert "MTP" in message
    else:
        assert repr(draft_kind) in message


def test_the_server_refuses_the_legacy_dflash_arm_by_name(monkeypatch):
    """v1 was measured on the continuous-batching loop only; the
    MLX_VLM_DFLASH_CONTINUOUS_BATCHING=0 control arm is not on that panel."""
    monkeypatch.delenv(SPEC_STRUCTURED_ENV, raising=False)
    monkeypatch.setenv("MLX_VLM_DFLASH_CONTINUOUS_BATCHING", "0")
    assert server_generation._uses_continuous_batching_loop("dflash") is False
    with pytest.raises(StructuredSpeculationRefused) as excinfo:
        _server_generate(_server_stub(SimpleNamespace()), _grammar_args())
    assert "MLX_VLM_DFLASH_CONTINUOUS_BATCHING" in str(excinfo.value)


def test_a_server_without_a_draft_model_is_untouched_by_any_of_this(monkeypatch):
    monkeypatch.delenv(SPEC_STRUCTURED_ENV, raising=False)
    with pytest.raises(Exception) as excinfo:
        _server_generate(_server_stub(None), _grammar_args())
    assert not isinstance(excinfo.value, StructuredSpeculationRefused)


# --------------------------------------------------------------------------
# D4 -- the ledger itself
# --------------------------------------------------------------------------
def test_block_rows_align_with_the_verify_columns():
    """Row i is the legal set after committed + drafts[:i] -- the alignment of
    ``verify_input = [b] + drafts`` and ``verify_out.logits[:, i]``."""
    ledger = StubLedger(VOCAB, _open_legal)
    ledger.commit([3])
    drafts = [_open_legal((3,))[0], _open_legal((3, _open_legal((3,))[0]))[0]]
    mask = ledger.masks_for_block(drafts)
    allowed = unpack_bitmask(mask, VOCAB).tolist()
    assert [i for i, v in enumerate(allowed[0]) if v] == _open_legal((3,))
    assert [i for i, v in enumerate(allowed[1]) if v] == _open_legal((3, drafts[0]))
    assert [i for i, v in enumerate(allowed[2]) if v] == _open_legal(
        (3, drafts[0], drafts[1])
    )


def test_rows_after_an_illegal_draft_are_vacuous_not_wrong():
    """llguidance fills all-ones after the draft path leaves the grammar; the
    stub mirrors it, so acceptance stops at the illegal position rather than at
    a bogus empty set one row later."""
    ledger = StubLedger(VOCAB, _open_legal)
    ledger.commit([3])
    illegal = next(t for t in range(VOCAB) if t not in _open_legal((3,)))
    allowed = unpack_bitmask(ledger.masks_for_block([illegal, 0]), VOCAB).tolist()
    assert [i for i, v in enumerate(allowed[0]) if v] == _open_legal((3,))
    assert all(allowed[1])
    assert all(allowed[2])


def test_an_empty_legal_set_raises_instead_of_decoding_garbage():
    """R2: an all -inf row is not an error to argmax (returns 0) or to
    categorical (returns garbage), so it has to be caught here."""
    ledger = StubLedger(VOCAB, lambda prefix: [] if len(prefix) == 2 else [5])
    ledger.commit([5])
    ledger.commit([5])
    with pytest.raises(StructuredLedgerError, match="empty legal set"):
        ledger.masks_for_block([])


def test_a_fresh_mask_array_is_returned_every_round():
    """R3: MLX is lazy, so a recycled shared buffer would mask a round with the
    NEXT round's grammar state."""
    ledger = StubLedger(VOCAB, _open_legal)
    ledger.commit([3])
    first = ledger.masks_for_block([])
    first_values = first.tolist()
    ledger.commit([_open_legal((3,))[0]])
    second = ledger.masks_for_block([])
    assert first is not second
    assert first.tolist() == first_values  # unchanged by the second round


def test_forced_tokens_walk_the_singleton_run():
    ledger = StubLedger(VOCAB, lambda prefix: [(len(prefix) + 1) % VOCAB])
    assert ledger.forced_tokens(4) == [1, 2, 3, 4]
    assert ledger.forced_tokens(2) == [1, 2]
    assert ledger.forced_tokens(0) == []


def test_forced_tokens_stop_at_the_first_open_position():
    ledger = StubLedger(
        VOCAB, lambda prefix: [7] if len(prefix) < 2 else [1, 2, 3]
    )
    assert ledger.forced_tokens(8) == [7, 7]


# --------------------------------------------------------------------------
# R9 -- the thinking off-by-one
# --------------------------------------------------------------------------
def test_the_grammar_starts_on_the_token_after_the_thinking_end_token():
    """``ThinkingAwareLogitsProcessor`` activates ON the end token and
    constrains the NEXT one, and never feeds the end token to the matcher."""
    ledger = StubLedger(VOCAB, _open_legal, thinking_end_token_id=THINK_END)

    ledger.commit([7])
    assert ledger.active is False
    assert all(unpack_bitmask(ledger.next_token_mask(), VOCAB).tolist()[0])

    ledger.commit([THINK_END])
    assert ledger.active is True
    assert ledger.consumed == 0  # the end token itself is NOT consumed
    allowed = unpack_bitmask(ledger.next_token_mask(), VOCAB).tolist()[0]
    assert [i for i, v in enumerate(allowed) if v] == _open_legal(())


def test_activation_inside_a_drafted_block_lands_on_the_right_row():
    """The predecessor of row i is drafts[i-1]; the grammar starts at the row
    whose predecessor IS the end token -- not one row earlier or later."""
    ledger = StubLedger(VOCAB, _open_legal, thinking_end_token_id=THINK_END)
    ledger.commit([7])
    first_legal = _open_legal(())[0]
    allowed = unpack_bitmask(
        ledger.masks_for_block([THINK_END, first_legal]), VOCAB
    ).tolist()
    assert all(allowed[0])  # predecessor 7: still inside thinking
    assert [i for i, v in enumerate(allowed[1]) if v] == _open_legal(())
    assert [i for i, v in enumerate(allowed[2]) if v] == _open_legal((first_legal,))


def test_the_thinking_gate_survives_a_block_that_never_activates():
    ledger = StubLedger(VOCAB, _open_legal, thinking_end_token_id=THINK_END)
    ledger.commit([7])
    allowed = unpack_bitmask(ledger.masks_for_block([8, 9]), VOCAB).tolist()
    assert all(all(row) for row in allowed)
    assert ledger.active is False


def test_a_thinking_stream_matches_the_masked_reference_through_activation():
    target = _Markov1Target(seed=5)
    drafter = _StubDrafter(target, kind="noise", block_size=8, seed=5)
    ledger = StubLedger(VOCAB, _open_legal, thinking_end_token_id=THINK_END)
    emitted = _run_rounds(drafter, ledger, max_tokens=32, first_bonus=THINK_END)
    reference = _masked_greedy_reference(
        _Markov1Target(seed=5),
        StubLedger(VOCAB, _open_legal, thinking_end_token_id=THINK_END),
        len(emitted),
        first_bonus=THINK_END,
    )
    assert emitted == reference


# --------------------------------------------------------------------------
# D5/D8 -- the emitted stream is the masked reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["oracle", "noise"])
@pytest.mark.parametrize("deferred", [True, False])
def test_greedy_emitted_stream_equals_the_masked_autoregressive_reference(
    monkeypatch, kind, deferred
):
    monkeypatch.setenv("MLX_VLM_DFLASH_DEFERRED", "1" if deferred else "0")
    target = _Markov1Target(seed=1)
    drafter = _StubDrafter(target, kind=kind, block_size=8, seed=1)
    emitted = _run_rounds(drafter, StubLedger(VOCAB, _mixed_legal), max_tokens=48)
    reference = _masked_greedy_reference(
        _Markov1Target(seed=1), StubLedger(VOCAB, _mixed_legal), len(emitted)
    )
    assert emitted == reference


def test_greedy_masked_stream_differs_from_the_unmasked_one():
    """Guard: if the mask never reached the sampler these tests would pass
    vacuously."""
    bonus = _mixed_legal(())[0]
    masked = _masked_greedy_reference(
        _Markov1Target(seed=1), StubLedger(VOCAB, _mixed_legal), 32, first_bonus=bonus
    )
    unmasked = _masked_greedy_reference(
        _Markov1Target(seed=1), StubLedger(VOCAB, _all_allow), 32, first_bonus=bonus
    )
    assert masked != unmasked


@pytest.mark.parametrize("kind", ["oracle", "noise"])
def test_sampled_emitted_stream_equals_the_masked_reference(kind):
    """R8: the coupling holds under the mask -- same masked logits, same token
    order, same per-position keys on both sides."""
    target = _Markov1Target(seed=2)
    drafter = _StubDrafter(target, kind=kind, block_size=8, seed=2)
    sampler = _PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=SEED, coupled=True
    )
    emitted = _run_rounds(
        drafter,
        StubLedger(VOCAB, _mixed_legal),
        sampler=sampler,
        greedy=False,
        max_tokens=40,
    )
    reference = _masked_sampled_reference(
        _Markov1Target(seed=2),
        StubLedger(VOCAB, _mixed_legal),
        _PositionedTargetSampler(
            temperature=1.0, top_p=0.95, seed=SEED, coupled=True
        ),
        len(emitted),
    )
    assert emitted == reference


def test_the_masked_sampled_stream_does_not_depend_on_the_drafter():
    """The coupling invariant, restated under masks: swapping the drafter must
    not move a single emitted token."""
    sampler_a = _PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=SEED, coupled=True
    )
    sampler_b = _PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=SEED, coupled=True
    )
    oracle = _run_rounds(
        _StubDrafter(_Markov1Target(seed=3), "oracle", seed=3),
        StubLedger(VOCAB, _mixed_legal),
        sampler=sampler_a,
        greedy=False,
        max_tokens=40,
    )
    noise = _run_rounds(
        _StubDrafter(_Markov1Target(seed=3), "noise", seed=3),
        StubLedger(VOCAB, _mixed_legal),
        sampler=sampler_b,
        greedy=False,
        max_tokens=40,
    )
    assert oracle == noise


def test_every_emitted_token_is_grammar_legal():
    target = _Markov1Target(seed=4)
    drafter = _StubDrafter(target, kind="noise", block_size=8, seed=4)
    emitted = _run_rounds(drafter, StubLedger(VOCAB, _mixed_legal), max_tokens=40)
    audit = StubLedger(VOCAB, _mixed_legal)
    audit.commit([emitted[0]])
    for token in emitted[1:]:
        legal = audit.legal_at(audit.history)
        assert legal is None or token in legal
        audit.commit([token])


def test_acceptance_stops_at_the_first_illegal_draft():
    """The target mask at the illegal position excludes the drafted token, so
    the walk cannot run past it -- which is why drafter-side masking is an
    acceptance optimisation and not a correctness requirement."""
    ledger = StubLedger(VOCAB, _open_legal)
    bonus = _legal_first_bonus(ledger)
    reference = _masked_greedy_reference(
        _Markov1Target(seed=6), StubLedger(VOCAB, _open_legal), 6, first_bonus=bonus
    )

    # draft the target's own choice at position 0 (so it is accepted) and a
    # grammar-illegal token at position 1 (so it cannot be)
    good0 = reference[1]
    legal1 = _open_legal((bonus, good0))
    illegal = next(t for t in range(VOCAB) if t not in legal1)
    drafter = _ScriptedDrafter(
        _Markov1Target(seed=6), [good0, illegal] + [0] * 5, block_size=8
    )
    emitted = _run_rounds(drafter, ledger, max_tokens=6, first_bonus=bonus)

    assert emitted == reference
    assert emitted[1] == good0  # position 0 accepted
    assert emitted[2] != illegal  # position 1 could not be
    assert drafter.accept_lens[0] == 1  # acceptance stopped exactly there


# --------------------------------------------------------------------------
# D6 Tier A -- grammar fast-forward
# --------------------------------------------------------------------------
def test_the_fast_forward_run_becomes_the_block_and_skips_the_drafter():
    target = _Markov1Target(seed=7)
    drafter = _StubDrafter(target, kind="noise", block_size=8, seed=7)
    # forced everywhere: the drafter must never be asked for a proposal
    ledger = StubLedger(VOCAB, lambda prefix: [(len(prefix) * 5 + 2) % VOCAB])
    emitted = _run_rounds(drafter, ledger, max_tokens=25)
    assert drafter.draft_block_calls == 0
    reference = _masked_greedy_reference(
        _Markov1Target(seed=7),
        StubLedger(VOCAB, lambda prefix: [(len(prefix) * 5 + 2) % VOCAB]),
        len(emitted),
    )
    assert emitted == reference
    # a fully forced grammar accepts every drafted token: the emitted stream is
    # exactly the forced walk, bonus included
    assert emitted == [(i * 5 + 2) % VOCAB for i in range(len(emitted))]


def test_the_drafter_is_still_used_where_the_grammar_is_open():
    target = _Markov1Target(seed=8)
    drafter = _StubDrafter(target, kind="noise", block_size=8, seed=8)
    _run_rounds(drafter, StubLedger(VOCAB, _open_legal), max_tokens=25)
    assert drafter.draft_block_calls > 0


def test_a_mixed_grammar_alternates_between_the_two_paths():
    target = _Markov1Target(seed=9)
    drafter = _StubDrafter(target, kind="noise", block_size=8, seed=9)
    emitted = _run_rounds(drafter, StubLedger(VOCAB, _mixed_legal), max_tokens=48)
    assert drafter.draft_block_calls > 0
    reference = _masked_greedy_reference(
        _Markov1Target(seed=9), StubLedger(VOCAB, _mixed_legal), len(emitted)
    )
    assert emitted == reference


def test_fast_forward_rounds_do_not_lose_the_drafter_context():
    """A skipped ``draft_block`` also skips the drafter's cross-attention ingest
    of that round's target hidden.  The round loop buffers it and hands the
    whole run to the next real draft call."""
    target = _Markov1Target(seed=10)

    seen = []

    class _RecordingDrafter(_StubDrafter):
        def draft_block(self, last_bonus, hidden, cache, bs, sampler, token_dtype, **kw):
            seen.append(int(hidden.shape[1]))
            return super().draft_block(
                last_bonus, hidden, cache, bs, sampler, token_dtype, **kw
            )

    drafter = _RecordingDrafter(target, kind="noise", block_size=8, seed=10)
    _run_rounds(drafter, StubLedger(VOCAB, _mixed_legal), max_tokens=48)
    assert seen
    # at least one call had to absorb more than one round's worth of context
    assert max(seen) > 1


# --------------------------------------------------------------------------
# D6 Tier B -- the drafter's position-0 mask
# --------------------------------------------------------------------------
def _selector_config():
    return ModelConfig.from_dict(
        {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "is_causal": False,
            "hidden_size": 8,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "vocab_size": VOCAB,
            "max_position_embeddings": 4096,
            "num_target_layers": 1,
            "layer_types": ["full_attention"],
            "sliding_window": None,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
            "dflash_config": {
                "block_size": 8,
                "runtime_block_size": 8,
                "conv_group_size": 4,
                "conv_kernel_size": 2,
                "mask_token_id": VOCAB - 1,
                "selector_rank": 4,
                "selector_top_k": 4,
                "target_layer_ids": [0],
            },
        }
    )


def _selector():
    drafter = DFlash2DraftModel(_selector_config())
    return drafter.candidate_selector


def _mask_for(tokens):
    return mx.array(SL._pack_rows([sorted(tokens)], VOCAB))


def test_the_position0_mask_keeps_the_drafter_proposal_legal():
    mx.random.seed(0)
    selector = _selector()
    hidden = mx.random.normal((1, 3, 8))
    logits = mx.random.normal((1, 3, VOCAB))
    anchor = mx.array([2], dtype=mx.int32)

    unconstrained = selector.select(hidden, logits, anchor, _greedy_sampler)
    first = int(unconstrained[0, 0])
    legal = sorted({t for t in range(VOCAB) if t % 5 == 0 and t != first})
    constrained = selector.select(
        hidden, logits, anchor, _greedy_sampler, structured_position0_mask=_mask_for(legal)
    )
    assert int(constrained[0, 0]) in legal
    assert int(unconstrained[0, 0]) not in legal  # the mask actually moved it


def test_the_position0_mask_falls_back_to_the_legal_argmax():
    """R5: when no top-k candidate is legal the masked row would be all -inf and
    argmax would silently return slot 0.  Collapse onto the mask's own argmax."""
    mx.random.seed(1)
    selector = _selector()
    hidden = mx.random.normal((1, 2, 8))
    logits = mx.random.normal((1, 2, VOCAB))
    anchor = mx.array([2], dtype=mx.int32)

    candidates = mx.argpartition(logits, -selector.top_k, axis=-1)[
        ..., -selector.top_k :
    ]
    top_k_at_0 = set(int(t) for t in candidates[0, 0].tolist())
    outside = sorted(set(range(VOCAB)) - top_k_at_0)
    expected = int(
        mx.argmax(
            mx.where(
                unpack_bitmask(_mask_for(outside), VOCAB),
                logits[:, 0],
                mx.array(-float("inf"), dtype=logits.dtype),
            ),
            axis=-1,
        ).reshape(-1)[0]
    )
    drafted = selector.select(
        hidden, logits, anchor, _greedy_sampler,
        structured_position0_mask=_mask_for(outside),
    )
    assert int(drafted[0, 0]) == expected


def test_the_round_loop_passes_the_position0_mask_only_to_drafters_that_ask():
    target = _Markov1Target(seed=11)
    drafter = _StubDrafter(target, kind="noise", block_size=8, seed=11)
    _run_rounds(drafter, StubLedger(VOCAB, _open_legal), max_tokens=24)
    assert drafter.masks_seen
    assert all(mask is None for mask in drafter.masks_seen)

    drafter = _StubDrafter(target, kind="noise", block_size=8, seed=11)
    drafter.supports_structured_draft_mask = True
    _run_rounds(drafter, StubLedger(VOCAB, _open_legal), max_tokens=24)
    assert drafter.masks_seen
    assert all(mask is not None for mask in drafter.masks_seen)
    assert all(int(mask.shape[0]) == 1 for mask in drafter.masks_seen)


def test_a_masked_drafter_still_emits_the_masked_reference():
    """Tier B may only change WHICH proposal the target agrees with."""
    target = _Markov1Target(seed=12)
    plain = _StubDrafter(target, kind="noise", block_size=8, seed=12)
    masked = _StubDrafter(_Markov1Target(seed=12), kind="noise", block_size=8, seed=12)
    masked.supports_structured_draft_mask = True
    a = _run_rounds(plain, StubLedger(VOCAB, _open_legal), max_tokens=32)
    b = _run_rounds(masked, StubLedger(VOCAB, _open_legal), max_tokens=32)
    assert a == b


# --------------------------------------------------------------------------
# mask application
# --------------------------------------------------------------------------
def test_apply_block_mask_sets_illegal_logits_to_negative_infinity():
    mask = mx.array(SL._pack_rows([[1, 3], None], 8))
    logits = mx.arange(16, dtype=mx.float32).reshape(2, 8)
    masked = apply_block_mask(logits, mask).tolist()
    assert masked[0] == [
        -float("inf"), 1.0, -float("inf"), 3.0,
        -float("inf"), -float("inf"), -float("inf"), -float("inf"),
    ]
    assert masked[1] == list(range(8, 16))


def test_the_portable_path_is_what_a_cpu_only_session_takes(monkeypatch):
    """``mx.fast.metal_kernel`` has no CPU implementation, and this MLX build
    (0.32.1) does not act on ``MLX_DEFAULT_DEVICE`` -- ``mx.default_device()``
    still says gpu -- so the CPU signal is honoured explicitly here.  Every test
    in this file therefore runs the portable path, which is the point."""
    monkeypatch.setenv("MLX_DEFAULT_DEVICE", "cpu")
    assert SL.metal_mask_kernel_enabled() is False
    monkeypatch.setenv("MLX_DEFAULT_DEVICE", " CPU ")
    assert SL.metal_mask_kernel_enabled() is False


def test_the_metal_kernel_is_selected_on_a_gpu_default_device(monkeypatch):
    monkeypatch.delenv("MLX_DEFAULT_DEVICE", raising=False)
    expected = bool(mx.metal.is_available()) and (
        mx.default_device() == mx.Device(mx.DeviceType.gpu, 0)
    )
    assert SL.metal_mask_kernel_enabled() is expected


def test_the_round_loop_never_dispatches_a_metal_kernel_on_the_cpu_rail(monkeypatch):
    """The measurement runs are CPU-only; a Metal dispatch from here would be a
    protocol violation, not just a slow path."""
    monkeypatch.setenv("MLX_DEFAULT_DEVICE", "cpu")

    def _boom(*args, **kwargs):  # pragma: no cover - the point is it never runs
        raise AssertionError("the CPU rail dispatched the Metal mask kernel")

    monkeypatch.setattr("mlx_vlm.structured._apply_llguidance_mask", _boom)
    tokens = _run_rounds(
        _StubDrafter(_Markov1Target(seed=23), "noise", seed=23),
        StubLedger(VOCAB, _mixed_legal),
        max_tokens=32,
    )
    assert tokens


@pytest.mark.skipif(
    (os.environ.get("MLX_DEFAULT_DEVICE") or "").strip().lower() == "cpu"
    or not mx.metal.is_available(),
    reason="CPU-only rail: this is the one test that would dispatch Metal",
)
def test_the_two_mask_paths_agree_elementwise():
    """The GPU kernel and the portable fallback are the same function.

    Deliberately the ONLY test here that touches Metal, and it skips on the
    CPU-only rail -- so this is the identity a GPU session has to confirm before
    the kernel path is measured.
    """
    from mlx_vlm.structured import _apply_llguidance_mask

    rng = np.random.default_rng(0)
    vocab = 200
    mask = mx.array(
        SL._pack_rows(
            [
                sorted(rng.choice(vocab, size=17, replace=False).tolist()),
                None,
                [0],
                [vocab - 1],
            ],
            vocab,
        )
    )
    logits = mx.array(rng.normal(size=(4, vocab)).astype(np.float32))
    assert (
        SL.apply_block_mask_portable(logits, mask).tolist()
        == _apply_llguidance_mask(logits, mask).tolist()
    )


# --------------------------------------------------------------------------
# llguidance-backed identities (skipped when the library is unavailable)
# --------------------------------------------------------------------------
LLG_TOKENIZER_DIR = os.environ.get(
    "MLX_VLM_TEST_LLG_TOKENIZER", "/Users/gesicht/glm53flash/meta_tok"
)

_LLG_CACHE = {}


def _llg_tokenizer():
    pytest.importorskip("llguidance")
    pytest.importorskip("llguidance.hf")
    transformers = pytest.importorskip("transformers")
    if not os.path.isdir(LLG_TOKENIZER_DIR):
        pytest.skip(f"no tokenizer at {LLG_TOKENIZER_DIR}")
    if "tok" not in _LLG_CACHE:
        import llguidance.hf

        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(LLG_TOKENIZER_DIR)
        _LLG_CACHE["tok"] = llguidance.hf.from_tokenizer(hf_tokenizer)
    return _LLG_CACHE["tok"]


SMALL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "kind": {"type": "string", "enum": ["alpha", "beta"]},
    },
    "required": ["name", "count", "kind"],
    "additionalProperties": False,
}


def _llg_matcher(llg_tokenizer):
    import json

    import llguidance as llg

    grammar = llg.JsonCompiler(
        separators=(", ", ": "), whitespace_pattern=""
    ).compile(json.dumps(SMALL_SCHEMA))
    return llg.LLMatcher(llg_tokenizer, grammar)


def test_fill_par_with_draft_tokens_equals_the_sequential_fill():
    """The whole ledger rests on this identity: one parallel draft-path fill ==
    fill / consume / fill / consume on a private copy."""
    llg_tokenizer = _llg_tokenizer()
    import llguidance.numpy as llnp

    matcher = _llg_matcher(llg_tokenizer)
    drafts = matcher.compute_ff_tokens()[:3]
    assert drafts, "the schema must force at least one token to make this test real"

    parallel = llnp.allocate_token_bitmask(len(drafts) + 1, llg_tokenizer.vocab_size)
    llnp.fill_next_token_bitmask_par_with_draft_tokens(
        SL._shared_executor(), [(matcher, 0, list(drafts))], parallel
    )

    sequential = llnp.allocate_token_bitmask(len(drafts) + 1, llg_tokenizer.vocab_size)
    walker = _llg_matcher(llg_tokenizer)
    for row in range(len(drafts) + 1):
        llnp.fill_next_token_bitmask(walker, sequential, row)
        if row < len(drafts):
            walker.consume_token(int(drafts[row]))
            assert not walker.get_error()

    assert np.array_equal(np.asarray(parallel), np.asarray(sequential))


def test_the_draft_path_fill_preserves_the_matcher_state():
    llg_tokenizer = _llg_tokenizer()
    import llguidance.numpy as llnp

    matcher = _llg_matcher(llg_tokenizer)
    before = matcher.compute_ff_tokens()
    buffer = llnp.allocate_token_bitmask(4, llg_tokenizer.vocab_size)
    llnp.fill_next_token_bitmask_par_with_draft_tokens(
        SL._shared_executor(), [(matcher, 0, [11, 12, 13])], buffer
    )
    assert not matcher.get_error()
    assert matcher.compute_ff_tokens() == before


def test_a_json_schema_forces_a_run_of_tokens_with_the_real_tokenizer():
    """D6 Tier A's premise: a real schema hands out multi-token forced runs."""
    llg_tokenizer = _llg_tokenizer()
    ledger = SL.StructuredLedger(
        _llg_matcher(llg_tokenizer), vocab_size=llg_tokenizer.vocab_size
    )
    forced = ledger.forced_tokens(8)
    assert len(forced) >= 2

    # the block mask agrees with the forced run: row 0 allows exactly forced[0]
    mask = ledger.masks_for_block(forced[:3])
    row0 = np.asarray(np.array(mask)[0]).view(np.uint8)
    assert int(np.unpackbits(row0).sum()) == 1
    allowed = unpack_bitmask(mask, llg_tokenizer.vocab_size)
    assert bool(allowed[0, forced[0]])


def test_the_real_ledger_commits_and_keeps_going():
    llg_tokenizer = _llg_tokenizer()
    ledger = SL.StructuredLedger(
        _llg_matcher(llg_tokenizer), vocab_size=llg_tokenizer.vocab_size
    )
    forced = ledger.forced_tokens(4)
    ledger.commit(forced)
    assert ledger.consumed == len(forced)
    assert not ledger.stopped
    mask = ledger.masks_for_block([])
    assert int(np.unpackbits(np.array(mask)[0].view(np.uint8)).sum()) > 0


def test_the_real_ledger_drives_the_round_loop_on_a_stub_target():
    """End to end on the real grammar: every emitted token is one llguidance
    would have accepted, with a target that has never seen JSON."""
    llg_tokenizer = _llg_tokenizer()
    vocab = int(llg_tokenizer.vocab_size)

    class _WideTarget:
        def __init__(self):
            rng = np.random.default_rng(0)
            self.row = mx.array(rng.normal(size=(vocab,)).astype(np.float32))

        def __call__(self, ids, cache=None, **kw):
            length = int(ids.shape[1])
            offsets = mx.arange(length, dtype=mx.float32)[None, :, None]
            return SimpleNamespace(
                logits=self.row[None, None, :] + offsets,
                hidden_states=[mx.zeros((1, length, HIDDEN))],
                gdn_states=["gdn"],
            )

        def rollback_speculative_cache(self, *args, **kwargs):
            return 0

    target = _WideTarget()
    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[0], block_size=8, runtime_block_size=8),
        accept_lens=[],
        draft_lens=[],
        dflash_deferred_walk=True,
        reset=lambda m: ["draft-cache"],
        draft_block=lambda b, h, c, bs, s, dt, **kw: mx.array(
            [[int(b)] * (bs - 1)], dtype=dt
        ),
    )
    ledger = SL.StructuredLedger(
        _llg_matcher(llg_tokenizer), vocab_size=vocab
    )
    bonus = ledger.forced_tokens(1)[0]
    rounds = dflash_utils._dflash_rounds(
        SimpleNamespace(language_model=target),
        drafter,
        [SimpleNamespace(offset=0)],
        mx.zeros((1, 1, HIDDEN)),
        first_bonus=bonus,
        max_tokens=24,
        sampler=_greedy_sampler,
        draft_block_size=8,
        use_model_initial_block_size=False,
        greedy_sampling=True,
        structured_ledger=ledger,
    )
    tokens = [bonus]
    try:
        for tok, _ in rounds:
            tokens.append(int(tok))
    finally:
        rounds.close()

    audit = _llg_matcher(llg_tokenizer)
    for token in tokens:
        assert audit.consume_token(int(token)) is not False
        assert not audit.get_error()
