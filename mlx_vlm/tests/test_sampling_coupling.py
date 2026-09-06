"""Sampled-mode speculative coupling: F1 (argmax draft) and F2 (shared-key Gumbel).

The defect these tests pin down: under sampling the fork drew the DFlash2 draft
proposal and the target sample from INDEPENDENT RNG streams (the proposal keyed
the target's stream XOR ``0x0DFA5202``) and accepted only on exact token
equality.  That is unbiased -- the emitted token is always the target's own draw
-- but the acceptance rate is the *collision probability*

    alpha_indep = sum_t p_t q_t   <=   max_t p_t   <=   exp(-H_2(p))

instead of the maximal coupling ``alpha_max = sum_t min(p_t, q_t)``.  On the
vectors used below that is 0.03 against 0.78: a 26x acceptance loss that no
amount of drafter quality can recover.

``mx.random.categorical`` is Gumbel-max, so one shared key over ONE INDEX SPACE
couples the two draws.  Two things break that coupling and both are tested here:

  * the XOR, which gives the two draws different Gumbel noise; and
  * the shipped top-p sampler, which argsorts the probabilities and samples on
    the SORTED axis, so slot j carries a different token for p than for q.
    ``test_top_p_sort_order_is_a_regression_guard`` is the guard: removing the
    XOR *alone* leaves acceptance at the broken value.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_vlm import sampling_coupling as SC
from mlx_vlm.generate.ar import _PositionedTargetSampler, _position_keys
from mlx_vlm.server import generation as server_generation
from mlx_vlm.speculative import dflash as dflash_utils
from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig

SEED = 20260906
VOCAB = 64


# --------------------------------------------------------------------------
# distributions with a known maximal coupling
# --------------------------------------------------------------------------
def _pq(sum_min_target=0.78):
    """A target p and a proposal q with ``sum_t min(p_t, q_t) ~= 0.78``."""
    rng = np.random.default_rng(SEED)
    p = rng.dirichlet(np.ones(VOCAB) * 0.6)
    r = rng.dirichlet(np.ones(VOCAB) * 0.6)
    q = 0.61 * p + 0.39 * r
    return p, q


def _sum_min(p, q):
    return float(np.minimum(p, q).sum())


def _sum_prod(p, q):
    return float((p * q).sum())


def _logprob_rows(vector, n):
    row = np.log(np.asarray(vector, dtype=np.float64))[None, :].astype(np.float32)
    return mx.repeat(mx.array(row), n, axis=0)


def _match_rate(sampler, p, q, n):
    rows = [0] * n
    positions = list(range(n))
    target = sampler.sample_target(
        _logprob_rows(p, n), row_ids=rows, positions=positions
    )
    proposal = sampler.sample_proposal(
        _logprob_rows(q, n), row_ids=rows, positions=positions
    )
    mx.eval(target, proposal)
    return float(mx.mean((target == proposal).astype(mx.float32)))


class _SortedAxisCoupledSampler(_PositionedTargetSampler):
    """The half-fix: XOR removed, but top-p still samples on the sorted axis.

    This is the arm that must stay BROKEN.  It exists so that anyone who
    "simplifies" ``top_p_logits_token_order`` back into the argsorted form has a
    failing test instead of a silent 20x acceptance regression.
    """

    def _sample_top_p_one_token_order(self, logprobs, key):
        return self._sample_top_p_one(logprobs, key)


# --------------------------------------------------------------------------
# env contract: both toggles inert by default
# --------------------------------------------------------------------------
def test_both_toggles_are_off_by_default(monkeypatch):
    monkeypatch.delenv(SC.COUPLING_ENV, raising=False)
    monkeypatch.delenv(SC.DRAFT_MODE_ENV, raising=False)
    assert SC.sampled_coupling_enabled() is False
    assert SC.dflash_sampled_draft_mode() == SC.DRAFT_MODE_INDEPENDENT


def test_the_draft_mode_default_follows_the_coupling_toggle(monkeypatch):
    monkeypatch.delenv(SC.DRAFT_MODE_ENV, raising=False)
    monkeypatch.setenv(SC.COUPLING_ENV, "1")
    assert SC.dflash_sampled_draft_mode() == SC.DRAFT_MODE_GUMBEL
    monkeypatch.setenv(SC.COUPLING_ENV, "0")
    assert SC.dflash_sampled_draft_mode() == SC.DRAFT_MODE_INDEPENDENT


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("independent", SC.DRAFT_MODE_INDEPENDENT),
        ("argmax", SC.DRAFT_MODE_ARGMAX),
        ("GUMBEL", SC.DRAFT_MODE_GUMBEL),
        ("  argmax  ", SC.DRAFT_MODE_ARGMAX),
    ],
)
def test_the_draft_mode_is_read_from_the_environment(monkeypatch, raw, expected):
    monkeypatch.delenv(SC.COUPLING_ENV, raising=False)
    monkeypatch.setenv(SC.DRAFT_MODE_ENV, raw)
    assert SC.dflash_sampled_draft_mode() == expected


def test_an_invalid_draft_mode_falls_back_and_does_not_raise(monkeypatch):
    monkeypatch.delenv(SC.COUPLING_ENV, raising=False)
    monkeypatch.setenv(SC.DRAFT_MODE_ENV, "maximal")
    assert SC.dflash_sampled_draft_mode() == SC.DRAFT_MODE_INDEPENDENT


@pytest.mark.parametrize("top_p", [1.0, 0.95])
def test_the_default_sampler_still_draws_the_independent_streams(monkeypatch, top_p):
    """With both toggles unset, every draw is byte-identical to the old code."""
    monkeypatch.delenv(SC.COUPLING_ENV, raising=False)
    p, q = _pq()
    sampler = _PositionedTargetSampler(temperature=1.0, top_p=top_p, seed=11)
    assert sampler.coupled is False
    rows, positions = [0] * 64, list(range(64))

    # target: the shipped sorted-axis nucleus draw, keyed by the plain seed
    keys = _position_keys(11, rows, positions)
    if top_p < 1.0:
        expected_target = mx.vmap(sampler._sample_top_p_one, in_axes=(0, 0))(
            _logprob_rows(p, 64), keys
        )
    else:
        expected_target = mx.vmap(sampler._sample_one, in_axes=(0, 0))(
            _logprob_rows(p, 64), keys
        )
    got_target = sampler.sample_target(
        _logprob_rows(p, 64), row_ids=rows, positions=positions
    )
    assert mx.array_equal(got_target, expected_target).item()

    # proposal: token-order categorical keyed by seed XOR 0x0DFA5202
    xor_keys = _position_keys(11 ^ 0x0DFA5202, rows, positions)
    expected_proposal = mx.vmap(sampler._sample_one, in_axes=(0, 0))(
        _logprob_rows(q, 64), xor_keys
    )
    got_proposal = sampler.sample_proposal(
        _logprob_rows(q, 64), row_ids=rows, positions=positions
    )
    assert mx.array_equal(got_proposal, expected_proposal).item()


def test_the_server_sampler_carries_the_same_contract(monkeypatch):
    monkeypatch.delenv(SC.COUPLING_ENV, raising=False)
    off = server_generation._PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=3
    )
    assert off.coupled is False
    monkeypatch.setenv(SC.COUPLING_ENV, "1")
    on = server_generation._PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=3
    )
    assert on.coupled is True
    p, q = _pq()
    assert _match_rate(on, p, q, 512) > 0.6 > _match_rate(off, p, q, 512)


# --------------------------------------------------------------------------
# F2: the coupling itself
# --------------------------------------------------------------------------
def test_shared_key_gumbel_reaches_the_maximal_coupling():
    """4 000 positions, per-position keys, no nucleus truncation."""
    p, q = _pq()
    alpha_max = _sum_min(p, q)
    alpha_collision = _sum_prod(p, q)
    assert 0.75 <= alpha_max <= 0.82, alpha_max

    independent = _PositionedTargetSampler(
        temperature=1.0, top_p=1.0, seed=SEED, coupled=False
    )
    coupled = _PositionedTargetSampler(
        temperature=1.0, top_p=1.0, seed=SEED, coupled=True
    )
    shipped = _match_rate(independent, p, q, 4000)
    fixed = _match_rate(coupled, p, q, 4000)

    # the shipped rail accepts at the collision probability, not the coupling
    assert shipped < 4 * alpha_collision, (shipped, alpha_collision)
    assert fixed >= 0.85 * alpha_max, (fixed, alpha_max)
    assert fixed > 15 * shipped, (fixed, shipped)


def test_top_p_sort_order_is_a_regression_guard():
    """top_p 0.95: dropping the XOR alone is NOT enough.

    Arm 1 (shipped)      independent keys, sorted-axis draw -> broken.
    Arm 2 (half fix)     shared key, sorted-axis draw       -> STILL broken.
    Arm 3 (token order)  shared key, token-id-order draw    -> coupled.

    Arm 2 is the reason this test exists: a shared key over an argsorted axis
    couples nothing, because slot j is a different token for p than for q.
    """
    p, q = _pq()
    alpha_max = _sum_min(p, q)
    n = 4000

    shipped = _match_rate(
        _PositionedTargetSampler(
            temperature=1.0, top_p=0.95, seed=SEED, coupled=False
        ),
        p,
        q,
        n,
    )
    half_fix = _match_rate(
        _SortedAxisCoupledSampler(
            temperature=1.0, top_p=0.95, seed=SEED, coupled=True
        ),
        p,
        q,
        n,
    )
    token_order = _match_rate(
        _PositionedTargetSampler(
            temperature=1.0, top_p=0.95, seed=SEED, coupled=True
        ),
        p,
        q,
        n,
    )

    assert shipped < 0.10, shipped
    assert half_fix < 0.10, (half_fix, "sorted-axis sampling broke the coupling")
    assert token_order >= 0.85 * alpha_max, (token_order, alpha_max)


def test_the_token_order_nucleus_mask_keeps_exactly_the_sorted_mask():
    """Same kept token set, same weights -- only the Gumbel slot changes."""
    p, _ = _pq()
    logprobs = _logprob_rows(p, 1)[0]
    top_p, temperature = 0.95, 1.0

    token_order = SC.top_p_logits_token_order(logprobs, top_p, temperature)

    probs = mx.softmax(logprobs / temperature, axis=-1)
    order = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    sorted_mask = mx.log(
        mx.where(cumulative > 1 - top_p, sorted_probs, mx.zeros_like(sorted_probs))
    )

    # gathering the token-order weights in sorted order must give the sorted ones
    gathered = mx.take_along_axis(token_order, order, axis=-1)
    assert mx.array_equal(gathered, sorted_mask).item()


def _emitted_histograms(n):
    p, _ = _pq()
    rows, positions = [0] * n, list(range(n))
    logprobs = _logprob_rows(p, n)
    shipped = _PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=SEED, coupled=False
    ).sample_target(logprobs, row_ids=rows, positions=positions)
    token_order = _PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=SEED, coupled=True
    ).sample_target(logprobs, row_ids=rows, positions=positions)
    mx.eval(shipped, token_order)
    a = np.bincount(np.array(shipped).reshape(-1), minlength=VOCAB)
    b = np.bincount(np.array(token_order).reshape(-1), minlength=VOCAB)
    return a, b


def _two_sample_chi_square(a, b):
    keep = (a + b) > 0
    a, b = a[keep].astype(float), b[keep].astype(float)
    na, nb, total = a.sum(), b.sum(), a.sum() + b.sum()
    pooled = (a + b) / total
    ea, eb = na * pooled, nb * pooled
    stat = float(((a - ea) ** 2 / ea).sum() + ((b - eb) ** 2 / eb).sum())
    return stat, int(keep.sum() - 1)


def test_the_emitted_marginal_is_unchanged_by_the_coupling():
    """The emitted token stays the TARGET's own draw, same distribution.

    20 000 draws through the shipped sorted-axis sampler against 20 000 through
    the token-order one.  The gate is the two-sample chi-square, not the raw TV
    distance: for a 64-token nucleus the TV between two INDEPENDENT 20 000-draw
    histograms of the SAME law has a noise floor near 0.018
    (2/sqrt(N*pi) * sum_t sqrt(p_t)), so a "TV < 0.01" gate at that sample size
    would fail on identical distributions.  TV is asserted at its noise band and
    driven under 0.011 by a 100 000-draw run below.
    """
    a, b = _emitted_histograms(20000)
    tv = 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())
    stat, dof = _two_sample_chi_square(a, b)
    assert stat < 2.0 * dof, (stat, dof)
    assert tv < 0.03, tv


def test_the_emitted_marginal_tv_shrinks_like_sampling_noise():
    """5x the draws must roughly halve the TV -- noise, not a real shift."""
    a, b = _emitted_histograms(100000)
    tv = 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())
    stat, dof = _two_sample_chi_square(a, b)
    assert tv < 0.011, tv
    assert stat < 2.0 * dof, (stat, dof)


def test_the_coupled_draw_is_still_the_targets_own_draw():
    """Unbiasedness: coupling changes the PROPOSAL's luck, never the emission."""
    p, _ = _pq()
    n = 512
    rows, positions = [0] * n, list(range(n))
    logprobs = _logprob_rows(p, n)
    sampler = _PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=SEED, coupled=True
    )
    first = sampler.sample_target(logprobs, row_ids=rows, positions=positions)
    second = sampler.sample_target(logprobs, row_ids=rows, positions=positions)
    mx.eval(first, second)
    assert mx.array_equal(first, second).item()


def test_scatter_candidates_to_vocab_places_scores_at_token_ids():
    scores = mx.array([[1.0, 2.0, 3.0]])
    candidates = mx.array([[5, 0, 9]], dtype=mx.int32)
    full = SC.scatter_candidates_to_vocab(scores, candidates, 12)
    assert full.shape == (1, 12)
    values = np.array(full)[0]
    assert values[5] == 1.0 and values[0] == 2.0 and values[9] == 3.0
    assert np.isneginf(values[[1, 2, 3, 4, 6, 7, 8, 10, 11]]).all()


# --------------------------------------------------------------------------
# F1: argmax drafting and the fused Viterbi step
# --------------------------------------------------------------------------
HIDDEN = 16
DRAFTER_VOCAB = 32
TARGET_LAYER_IDS = [0, 1]
NUM_TARGET_LAYERS = 4
BLOCK_TOTAL = 8


def _drafter_config(layers=2):
    return ModelConfig.from_dict(
        {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "is_causal": False,
            "hidden_size": HIDDEN,
            "intermediate_size": 32,
            "num_hidden_layers": layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "vocab_size": DRAFTER_VOCAB,
            "max_position_embeddings": 4096,
            "num_target_layers": NUM_TARGET_LAYERS,
            "layer_types": ["full_attention"] * layers,
            "sliding_window": None,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000},
            "dflash_config": {
                "block_size": BLOCK_TOTAL,
                "runtime_block_size": BLOCK_TOTAL,
                "conv_group_size": 4,
                "conv_kernel_size": 2,
                "mask_token_id": DRAFTER_VOCAB - 1,
                "selector_rank": 4,
                "selector_top_k": 4,
                "target_layer_ids": TARGET_LAYER_IDS,
            },
        }
    )


class _EmbedOnlyTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            embed_tokens=nn.Embedding(DRAFTER_VOCAB, HIDDEN),
            layers=[None] * NUM_TARGET_LAYERS,
        )
        self.lm_head = nn.Linear(HIDDEN, DRAFTER_VOCAB, bias=False)
        self.config = SimpleNamespace(
            hidden_size=HIDDEN,
            vocab_size=DRAFTER_VOCAB,
            num_hidden_layers=NUM_TARGET_LAYERS,
        )

    def rollback_speculative_cache(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _selector_inputs(seed=0, batch=1, length=3):
    mx.random.seed(seed)
    drafter = DFlash2DraftModel(_drafter_config())
    drafter.bind(_EmbedOnlyTarget())
    mx.eval(drafter.parameters())
    hidden = mx.random.normal((batch, length, HIDDEN))
    logits = mx.random.normal((batch, length, DRAFTER_VOCAB))
    anchors = mx.array([1] * batch, dtype=mx.int32)
    return drafter.candidate_selector, hidden, logits, anchors


def test_the_argmax_draft_sampler_exposes_no_sample_proposal():
    sampler = dflash_utils._ArgmaxDraftSampler()
    assert getattr(sampler, "sample_proposal", None) is None
    logits = mx.array([[[0.0, 3.0, 1.0]]])
    assert mx.array_equal(sampler(logits), mx.array([[1]])).item()


def test_argmax_mode_restores_the_fused_viterbi_step(monkeypatch):
    """``sample_proposal`` present == fused step skipped (dflash2.py:222-229)."""
    monkeypatch.setenv("MLX_VLM_DFLASH_COMPILE", "1")
    monkeypatch.setattr(
        "mlx_vlm.speculative.drafters.dflash2.dflash2._COMPILE_ENV", None, raising=False
    )
    selector, hidden, logits, anchors = _selector_inputs()

    positioned = dflash_utils._make_draft_sampler(
        _PositionedTargetSampler(temperature=1.0, top_p=0.95, seed=1, coupled=False),
        row_ids=[0],
        positions=[0],
        mode=SC.DRAFT_MODE_INDEPENDENT,
    )
    selector.select(hidden, logits, anchors, positioned)
    assert selector._fused_step is None, "a sampled proposal must stay eager"

    argmax_sampler = dflash_utils._make_draft_sampler(
        object(), row_ids=[0], positions=[0], mode=SC.DRAFT_MODE_ARGMAX
    )
    assert isinstance(argmax_sampler, dflash_utils._ArgmaxDraftSampler)
    drafted = selector.select(hidden, logits, anchors, argmax_sampler)
    assert selector._fused_step is not None, "argmax drafting must fuse again"

    # and it is the greedy walk: identical to passing a bare argmax callable
    selector._fused_step = None
    monkeypatch.setenv("MLX_VLM_DFLASH_COMPILE", "0")
    monkeypatch.setattr(
        "mlx_vlm.speculative.drafters.dflash2.dflash2._COMPILE_ENV", None, raising=False
    )
    eager = selector.select(
        hidden, logits, anchors, lambda x: mx.argmax(x, axis=-1)
    )
    assert mx.array_equal(drafted, eager).item()


def test_gumbel_mode_lifts_the_selector_scores_onto_the_token_axis():
    """Coupling needs one index space: candidate slots are not token ids."""
    selector, hidden, logits, anchors = _selector_inputs()
    seen = []

    class _Recorder:
        coupled_full_vocab = True

        def sample_proposal(self, scored):
            seen.append(scored)
            return mx.argmax(scored, axis=-1)

    drafted = selector.select(hidden, logits, anchors, _Recorder())
    assert len(seen) == int(hidden.shape[1])
    for scored in seen:
        assert scored.shape == (1, DRAFTER_VOCAB)
        finite = np.isfinite(np.array(scored)[0])
        # exactly selector_top_k candidates carry a score, the rest are -inf
        assert int(finite.sum()) == 4, int(finite.sum())
    assert drafted.shape == (1, int(hidden.shape[1]))

    # the same walk without the lift picks the same tokens (argmax is
    # permutation-invariant), which is what makes the lift a pure re-indexing
    class _Plain:
        def sample_proposal(self, scored):
            return mx.argmax(scored, axis=-1)

    assert mx.array_equal(drafted, selector.select(hidden, logits, anchors, _Plain())).item()


def test_the_positioned_draft_sampler_advertises_the_mode():
    target = _PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=1, coupled=True
    )
    gumbel = dflash_utils._make_draft_sampler(
        target, row_ids=[0], positions=[0], mode=SC.DRAFT_MODE_GUMBEL
    )
    assert gumbel.coupled_full_vocab is True
    independent = dflash_utils._make_draft_sampler(
        target, row_ids=[0], positions=[0], mode=SC.DRAFT_MODE_INDEPENDENT
    )
    assert independent.coupled_full_vocab is False


# --------------------------------------------------------------------------
# greedy must be untouched
# --------------------------------------------------------------------------
class _ScriptedPair:
    """A target+drafter stub: the target always agrees with the draft."""

    def __init__(self, hidden=4, vocab=64):
        self.hidden, self.vocab = hidden, vocab
        self.round = 0
        self.last_draft = []
        self.samplers_seen = []

    def make_cache(self):
        return ["target-cache"]

    def _onehot(self, row):
        out = mx.zeros((1, len(row), self.vocab))
        idx = mx.array(row, dtype=mx.int32)
        return out + (mx.arange(self.vocab)[None, None, :] == idx[None, :, None]) * 10.0

    def __call__(self, ids, cache=None, **kw):
        length = ids.shape[1]
        row = list(self.last_draft) + [40 + self.round]
        row += [50] * (length - len(row))
        self.round += 1
        return SimpleNamespace(
            logits=self._onehot(row[:length]),
            hidden_states=[mx.zeros((1, length, self.hidden))],
            gdn_states=["gdn"],
        )

    def rollback_speculative_cache(self, *args, **kwargs):
        return 0

    def draft_block(self, b, hidden, draft_cache, bs, sampler, token_dtype, **kw):
        self.samplers_seen.append(sampler)
        self.last_draft = [10 + i for i in range(bs - 1)]
        return mx.array([self.last_draft], dtype=token_dtype)


def _run_greedy_rounds(max_tokens=12, block_size=4):
    pair = _ScriptedPair()
    model = SimpleNamespace(language_model=pair)
    drafter = SimpleNamespace(
        config=SimpleNamespace(
            target_layer_ids=[0],
            block_size=block_size,
            runtime_block_size=block_size,
        ),
        accept_lens=[],
        draft_lens=[],
        dflash_deferred_walk=True,
        reset=lambda m: ["draft-cache"],
        draft_block=pair.draft_block,
    )
    rounds = dflash_utils._dflash_rounds(
        model,
        drafter,
        [SimpleNamespace(offset=0)],
        mx.zeros((1, 1, 4)),
        first_bonus=7,
        max_tokens=max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        draft_block_size=block_size,
        use_model_initial_block_size=False,
        greedy_sampling=True,
    )
    tokens = []
    try:
        for tok, _ in rounds:
            tokens.append(int(tok))
    finally:
        rounds.close()
    return pair, tokens


def test_greedy_rounds_never_reach_the_sampled_coupling_code(monkeypatch):
    """The greedy branch must not call the F1/F2 resolvers at all."""

    def _boom(*args, **kwargs):  # pragma: no cover - the point is it never runs
        raise AssertionError("greedy decoding touched the sampled-coupling path")

    monkeypatch.setattr(dflash_utils, "_make_draft_sampler", _boom)
    monkeypatch.setattr(dflash_utils, "dflash_sampled_draft_mode", _boom)

    pair, tokens = _run_greedy_rounds()
    assert tokens
    # the drafter saw the caller's own greedy sampler, not a wrapper
    assert pair.samplers_seen
    assert all(callable(s) and not hasattr(s, "sample_proposal") for s in pair.samplers_seen)


def test_greedy_output_is_bit_identical_with_the_toggles_on(monkeypatch):
    monkeypatch.delenv(SC.COUPLING_ENV, raising=False)
    monkeypatch.delenv(SC.DRAFT_MODE_ENV, raising=False)
    _, baseline = _run_greedy_rounds()
    monkeypatch.setenv(SC.COUPLING_ENV, "1")
    monkeypatch.setenv(SC.DRAFT_MODE_ENV, "gumbel")
    _, coupled = _run_greedy_rounds()
    assert baseline == coupled


def _tiny_drafter(seed=0):
    mx.random.seed(seed)
    drafter = DFlash2DraftModel(_drafter_config())
    drafter.bind(_EmbedOnlyTarget())
    mx.eval(drafter.parameters())
    return drafter


@pytest.mark.parametrize(
    "mode", [SC.DRAFT_MODE_INDEPENDENT, SC.DRAFT_MODE_ARGMAX, SC.DRAFT_MODE_GUMBEL]
)
def test_a_full_draft_block_runs_in_every_mode(mode):
    """Shape contract end to end: [B, V] proposals reach the positioned sampler."""
    drafter = _tiny_drafter()
    target = _PositionedTargetSampler(
        temperature=1.0, top_p=0.95, seed=SEED, coupled=mode == SC.DRAFT_MODE_GUMBEL
    )
    draft_sampler = dflash_utils._make_draft_sampler(
        target, row_ids=[0], positions=[5], mode=mode
    )
    tokens = drafter.draft_block(
        3,
        mx.random.normal((1, 1, HIDDEN * len(TARGET_LAYER_IDS))),
        drafter.make_cache(),
        BLOCK_TOTAL,
        draft_sampler,
        mx.int32,
    )
    mx.eval(tokens)
    assert tokens.shape == (1, BLOCK_TOTAL - 1)
    ids = np.array(tokens).reshape(-1)
    assert ((ids >= 0) & (ids < DRAFTER_VOCAB)).all(), ids


def test_the_round_loop_resolver_follows_the_environment(monkeypatch):
    target = _PositionedTargetSampler(temperature=1.0, top_p=0.95, seed=1)
    monkeypatch.delenv(SC.DRAFT_MODE_ENV, raising=False)
    monkeypatch.setenv(SC.COUPLING_ENV, "1")
    built = dflash_utils._make_draft_sampler(target, row_ids=[0], positions=[0])
    assert built.mode == SC.DRAFT_MODE_GUMBEL and built.coupled_full_vocab

    monkeypatch.setenv(SC.DRAFT_MODE_ENV, "argmax")
    assert isinstance(
        dflash_utils._make_draft_sampler(target, row_ids=[0], positions=[0]),
        dflash_utils._ArgmaxDraftSampler,
    )

    monkeypatch.delenv(SC.COUPLING_ENV, raising=False)
    monkeypatch.delenv(SC.DRAFT_MODE_ENV, raising=False)
    built = dflash_utils._make_draft_sampler(target, row_ids=[0], positions=[0])
    assert built.mode == SC.DRAFT_MODE_INDEPENDENT and not built.coupled_full_vocab
