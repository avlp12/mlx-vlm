"""Chunked prefill of a RIGHT-PADDED batch must sample each row's last real token.

``BatchGenerator._build_mixed_prompt_batch`` (``generate/ar.py:2656-2833``) admits a
warm APC-prefix row and a cold row into ONE ``PromptProcessingBatch``, right-padding
each row's suffix to the longest one so the suffix RoPE positions line up
(``:2689-2690`` sets ``right_pad_per_row``, forwarded at ``:2825``).  APC is on by default
(``server/runtime_config.py:196``), so this batch is what a served pair of
concurrent requests takes whenever either of them hits the prefix cache.

A right-padded row's real tokens stop before the padded sequence does, so the row's
last real token is NOT the last column.  ``PromptProcessingBatch.generate`` picked it
at ``seq - 1 - right_pad[i]`` where ``seq`` is the width of the FINAL forward.
Unchunked that is the whole prompt and the index is right.  Once
``prefill_step_size`` chunks the prefill (default 2048,
``generate/common.py:25``), ``seq`` is only the leftover and the index goes
NEGATIVE: the row samples some other position, or a padding one.

Measured on this fixture at 422b69fa, greedy, no drafter: unchunked emits
``[[125, 29, 100, 54], [99, 115, 87, 81]]``.  Chunked, the padded row emits
something else at every chunk size tried (4, 8, 12, 16, 24, 32) -- and NOT THE SAME
something else on a re-run: an out-of-range ``take_along_axis`` reads unspecified
memory, so two consecutive runs of the identical arm gave the padded row
``[64, 48, 112, 8]`` and ``[50, 102, 109, 14]`` at step 8, and ``[40, 73, 76, 43]``
and ``[111, 31, 12, 125]`` at step 12.  The UNPADDED row is ``[99, 115, 87, 81]``
in every arm of every run, and every array of the prompt cache is bit-equal to the
unchunked arm -- so the prefill was right and the SELECTION was wrong.  A served
request would get a nondeterministically wrong completion, which is why no fixed
"wrong answer" is pinned below: the assertion is against the UNCHUNKED arm.

The fix keeps the chunk forward's arguments byte-identical (nothing about the
capture reaches the model, so the cache cannot move) and instead captures, per row,
the ``[1, vocab]`` slice of the chunk that contains that row's last real token.
This module pins:

1.  chunked emitted tokens == unchunked emitted tokens, per row, at six chunk sizes
    and over several decode steps -- the test that fails at 422b69fa;
2.  the prompt cache stays bit-equal chunked vs unchunked (the fix must not have
    bought (1) by changing the prefill);
3.  the same at B = 3 with two different right pads, so the loop is not accidentally
    right only for one boundary per prefill;
4.  the unchunked arm's PROMPT CACHE is byte-identical to the one recorded before
    the fix (see RIGHT PADDING IS DECLINED below for why the decode tokens are no
    longer part of that pin);
5.  a batch that is NOT right-padded takes no capture at all -- the code path an
    ordinary cold batch runs is untouched.

DEVICE (added 2026-09-03, after a GPU-quiet Metal run of the suite).  Claims 1-5
are about WHICH COLUMN is sampled, which is integer indexing and device-free.
The numbers they were originally written against are not: on Metal the KDA
(linear-attention) scan split at a chunk boundary drifts from the single scan,
so the chunked prompt cache is not bit-equal to the unchunked one and the
recorded CPU digests do not reproduce.  Measured here on an M3 Ultra, mlx
0.32.1.dev20260902:

  * chunked vs unchunked prompt cache, B = 2: 3 of 10 arrays move, worst
    9.54e-07 (bit-equal on CPU).  Bounded, not asserted equal, on GPU.
  * the PROMPT SELECTION itself -- the ``[B, vocab]`` the sampler is handed --
    agrees with the unchunked arm to <= 1.43e-06 at every chunk size on BOTH
    devices, with identical argmax.  That is the fix's actual claim and it is
    now asserted directly (``test_the_selected_prompt_logits_...``) instead of
    being inferred from four decode steps.
  * four decode steps on the B = 3 fixture are NOT a device-stable comparison:
    this is a 2-layer randomly-initialised model, and row 0's third sampled
    token sits on a 0.074-logprob margin that a 1e-06 state drift flips.  The
    Metal UNCHUNKED arm itself emits [52, 106, 5, 83] where CPU emits
    [52, 106, 60, 71].  The first two tokens agree everywhere; on GPU only
    those are asserted for B = 3, and the docstring says so rather than the
    test quietly comparing two chaotic trajectories.  B = 2 is stable on both
    devices and keeps all four.
  * the pre-fix digest tables are device-keyed.  The GPU column was measured the
    same way the CPU one was -- the pristine 422b69fa tree, same interpreter,
    same machine, 2026-09-03 -- and the post-fix tree reproduces all 14 chunked
    cache digests and both unchunked digests byte for byte on GPU as well.  The
    "the fix did not move the prefill" guard therefore holds on both devices.

RIGHT PADDING IS DECLINED (added 2026-09-03).  This module measures WHICH COLUMN
a right-padded batch samples, and that is all it ever measured.  The batch itself
is now known to be wrong for a different reason, one that no column arithmetic
can reach: the KDA (linear-attention) layers fold the padding into a recurrent
state and a conv window taken AT the padded column, and a recurrent state cannot
be rolled back the way ``BatchKVCache.finalize()`` rolls a K/V buffer.  A
right-padded row's PROMPT logits are right to 9.5e-07 and its DECODE trajectory
parts company from a singleton run at the first step, by 0.09 to 1.6 on a 6.3
logit scale depending on the pad.  ``BatchGenerator`` no longer builds such a
batch for this model (``generate/ar.py::_apply_right_pad_policy``); the fixtures
here build one directly, which is still the right thing for a module about
column selection, but it means the DECODE TOKENS of a padded row are not a
correct answer and must not be pinned as if they were.

So two pins moved:

  * ``test_the_unchunked_arm_...`` pinned ``_digest(rows, arrays)`` -- the
    emitted tokens AND the cache.  The cache half is unchanged and is now pinned
    on its own (``UNCHUNKED_CACHE_DIGEST``, verified byte-identical to the
    pre-fix build); the token half is replaced by the claim that actually holds,
    namely that the row carrying NO right padding decodes exactly like the same
    row run alone, and that every row's FIRST token does.
  * ``test_right_padding_attached_after_construction_...`` pinned two token ids
    recorded at 422b69fa.  That fixture attaches ``_right_pad_per_row`` AFTER
    construction, so the cache is never told about the padding -- its arithmetic
    is nobody's served arithmetic, and the pin duly went stale at 47b503a3
    ([71, 99] -> [95, 99], a live failure in this file before this commit).  It
    now asserts the arithmetic it was always about: no capture was taken, and the
    sampler is handed exactly ``logits[i, width - 1 - right_pad[i]]`` of the
    final forward.
"""

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate.ar import PromptProcessingBatch, _right_pad_prompts
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel

STEPS = [4, 8, 12, 16, 24, 32]

# ``mx.default_device()`` is pinned by conftest from MLX_DEFAULT_DEVICE.
ON_GPU = mx.default_device() == mx.gpu
DEV = "gpu" if ON_GPU else "cpu"

# Chunked-vs-unchunked prompt-cache drift from splitting the KDA scan.  Zero on
# CPU at B = 2; worst measured on Metal 9.54e-07 (B = 2) and 8.35e-07 (B = 3).
CACHE_DRIFT_TOL = 2e-06
# Selection claim: the [B, vocab] handed to the sampler for the prompt.  Worst
# measured 1.43e-06 on both devices, at every chunk size, both fixtures.
SELECTION_TOL = 1e-05
# Decode steps that are a device-stable comparison on the B = 3 fixture.  See
# the DEVICE section of the module docstring.
B3_STABLE_TOKENS = 2 if ON_GPU else 4


def _tiny_text_config():
    return TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=0,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_head_dim=64,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        index_topk=6,
        index_head_dim=64,
        index_n_heads=2,
        index_kpool=3,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 64,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        eos_token_id=1,
    )


def _lm():
    mx.random.seed(0)
    lm = LanguageModel(_tiny_text_config())
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def _cache_arrays(prompt_cache):
    """Every array a prompt cache holds, flattened, for a bytewise comparison."""
    out = []
    for entry in prompt_cache:
        stack = [entry.state]
        while stack:
            item = stack.pop()
            if isinstance(item, mx.array):
                out.append(item)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
    return out


# ------------------------------------------------------------------- fixtures
#
# Shaped exactly as ``_build_mixed_prompt_batch`` shapes one: per-row whole prompt,
# per-row APC prefix length, the suffix that is actually prefilled, and the right
# padding that squares the suffixes off.  ``warm_cache`` is deliberately not
# supplied -- the warm row's prefix K/V is absent, so its arithmetic is not what a
# served warm row would compute.  Nothing here depends on that: every claim is a
# comparison between two arms of the SAME fixture.

B2_FULL = [list(range(3, 43)), list(range(7, 47))]  # both whole prompts are 40 long
B2_PREFIX = [16, 0]  # row 0 is warm: 16 tokens already in the cache
B2_SUFFIX = [B2_FULL[i][B2_PREFIX[i] :] for i in range(2)]  # 24 and 40

# Three rows, three different prefixes -> two distinct non-zero right pads, so two
# different chunks own a boundary.
B3_FULL = [list(range(3, 43)), list(range(5, 45)), list(range(7, 47))]
B3_PREFIX = [18, 7, 0]
B3_SUFFIX = [B3_FULL[i][B3_PREFIX[i] :] for i in range(3)]  # 22, 33, 40


def _pads(suffixes):
    width = max(len(s) for s in suffixes)
    return [width - len(s) for s in suffixes], width


def _run(step, suffixes, *, emit=4):
    """Prefill ``suffixes`` right-padded, at ``prefill_step_size=step``, then decode.

    Returns ``(tokens_per_row, cache_arrays)`` where ``tokens_per_row[i]`` is the
    list of ``emit`` token ids row ``i`` emitted, greedily.
    """
    pads, width = _pads(suffixes)
    lm = _lm()
    padded = _right_pad_prompts(suffixes, max_length=width)
    batch = PromptProcessingBatch(
        model=lm,
        uids=list(range(len(suffixes))),
        input_ids=suffixes,
        max_tokens=[emit] * len(suffixes),
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=step,
        right_pad_per_row=list(pads),
        suffix_lens=[len(s) for s in suffixes],
    )
    # No drafter, so nothing declines the chunking.
    assert batch.prefill_step_size == step
    chunks = 0
    while batch.needs_processing():
        batch.prompt_step()
        chunks += 1
    gen_batch = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
    )
    arrays = _cache_arrays(gen_batch.prompt_cache)
    mx.eval(arrays)

    rows = [[] for _ in suffixes]
    for _ in range(emit):
        tokens, _, _, _ = gen_batch._step()
        for i, t in enumerate(tokens):
            rows[i].append(int(t))
    return rows, arrays, chunks


def _prompt_selection_logprobs(step, suffixes):
    """The ``[B, vocab]`` the sampler is handed for the PROMPT, plus the columns.

    This is what the fix changes and nothing else: ``generate()`` assembles one
    row of logits per row of the batch, and this returns exactly that array
    together with ``_last_real_column`` so the index arithmetic can be asserted
    on its own.
    """
    pads, width = _pads(suffixes)
    lm = _lm()
    padded = _right_pad_prompts(suffixes, max_length=width)
    batch = PromptProcessingBatch(
        model=lm,
        uids=list(range(len(suffixes))),
        input_ids=suffixes,
        max_tokens=[1] * len(suffixes),
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=step,
        right_pad_per_row=list(pads),
        suffix_lens=[len(s) for s in suffixes],
    )
    columns = None if batch._last_real_column is None else list(batch._last_real_column)
    while batch.needs_processing():
        batch.prompt_step()
    seen = {}

    def sampler(logprobs):
        seen["lp"] = mx.array(logprobs)
        return mx.argmax(logprobs, axis=-1)

    batch.generate(sampler=sampler, stop_criteria=lambda token: False)
    mx.eval(seen["lp"])
    return seen["lp"], columns, width


def _cache_digest(arrays):
    """Digest of the prompt cache alone -- what the fix must leave untouched."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(repr(tuple(a.shape)).encode())
        h.update(memoryview(np.asarray(a.astype(mx.float32))).tobytes())
    return h.hexdigest()


# ``_digest(rows, arrays)`` -- tokens AND cache -- used to live here.  It was the
# only caller of a token pin on a right-padded row's decode, which is a wrong
# answer (module docstring, RIGHT PADDING IS DECLINED), so it went with the pin.


# ------------------------------------------------------- 1. the emitted tokens


@pytest.mark.parametrize("step", STEPS)
def test_chunked_right_padded_prefill_emits_the_unchunked_tokens(step):
    """THE defect.  At 422b69fa the padded row's first token is wrong at every step.

    Both rows, four decode steps each -- not just the first token, because the
    first token is what the wrong logits produce and everything after it is what
    that token produces.
    """
    unchunked, _, n_unchunked = _run(None, B2_SUFFIX)
    chunked, _, n_chunked = _run(step, B2_SUFFIX)

    assert n_unchunked == 0 and n_chunked >= 1, (n_unchunked, n_chunked)
    assert chunked == unchunked, f"step={step}"


def test_the_unpadded_row_was_never_the_broken_one():
    """A negative control: the defect is specific to a row that carries padding."""
    unchunked, _, _ = _run(None, B2_SUFFIX)
    for step in STEPS:
        chunked, _, _ = _run(step, B2_SUFFIX)
        assert chunked[1] == unchunked[1], f"step={step}: the unpadded row moved"


# --------------------------------------------------- 2. the cache did not move


@pytest.mark.parametrize("step", STEPS)
def test_the_chunked_prompt_cache_tracks_the_unchunked_one(step):
    """The prefill was always right; the fix must not have bought (1) by moving it.

    B = 2 only.  At B = 3 the KDA scan lands a chunk boundary inside a row at two of
    the six step sizes and the split scan is not the same arithmetic as one scan
    over the whole prompt -- see ``test_three_rows_...`` below, which measures that
    and pins it against the PRE-FIX build rather than against the unchunked arm.

    On CPU this is BIT equality, and that is not relaxed.  On Metal the same KDA
    split drift shows up at B = 2 as well -- 3 of the 10 cache arrays move, worst
    9.54e-07, on cache array 5 (shape (2, 1, 40, 129)) -- so there the assertion
    is the documented ``CACHE_DRIFT_TOL`` bound.  Neither number is caused by the
    fix: ``test_the_prompt_cache_is_what_the_pre_fix_build_wrote`` pins all 14
    arms byte for byte against the pre-fix build ON THE SAME DEVICE.
    """
    _, unchunked, _ = _run(None, B2_SUFFIX)
    _, chunked, _ = _run(step, B2_SUFFIX)

    assert len(chunked) == len(unchunked) == 10
    for i, (a, b) in enumerate(zip(chunked, unchunked)):
        assert a.shape == b.shape, f"cache array {i}"
        if not ON_GPU:
            assert mx.array_equal(a, b), f"cache array {i}, step={step}"
            continue
        if a.size == 0:
            continue
        drift = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        assert drift < CACHE_DRIFT_TOL, f"cache array {i}, step={step}: {drift}"


# Digests of the CHUNKED prompt cache, recorded on 422b69fa BEFORE the fix
# (pristine ``generate/ar.py`` restored into this worktree, measured, restored).
# The fix captures logits out of a chunk's OUTPUT and passes the model exactly the
# arguments it passed before, so every one of these must survive it byte for byte.
# This is the guard that the selection fix did not become a prefill change.
# The kernels differ per device, so the digests do: the CPU column is the
# original one, the GPU column was measured the same way on 2026-09-03 (M3 Ultra,
# mlx 0.32.1.dev20260902).  That the GPU column has SEVEN distinct values where
# the CPU one has one is itself the KDA-split drift, present before the fix.
PRE_FIX_CACHE = {
    ("cpu", "B2", None): "9c582cbff6ae7517",
    ("cpu", "B2", 4): "9c582cbff6ae7517",
    ("cpu", "B2", 8): "9c582cbff6ae7517",
    ("cpu", "B2", 12): "9c582cbff6ae7517",
    ("cpu", "B2", 16): "9c582cbff6ae7517",
    ("cpu", "B2", 24): "9c582cbff6ae7517",
    ("cpu", "B2", 32): "9c582cbff6ae7517",
    ("cpu", "B3", None): "f7900c633298b31b",
    ("cpu", "B3", 4): "fc43c6b7ac48598c",
    ("cpu", "B3", 8): "f7900c633298b31b",
    ("cpu", "B3", 12): "756c54ee44d28d9f",
    ("cpu", "B3", 16): "f7900c633298b31b",
    ("cpu", "B3", 24): "f7900c633298b31b",
    ("cpu", "B3", 32): "f7900c633298b31b",
    ("gpu", "B2", None): "a2b236c52c7c94be",
    ("gpu", "B2", 4): "a104ac619cdcdd23",
    ("gpu", "B2", 8): "a104ac619cdcdd23",
    ("gpu", "B2", 12): "a104ac619cdcdd23",
    ("gpu", "B2", 16): "a104ac619cdcdd23",
    ("gpu", "B2", 24): "b7b9f2af3c7eb1e3",
    ("gpu", "B2", 32): "cafcaf05f92c1327",
    ("gpu", "B3", None): "6ac67756acdcd285",
    ("gpu", "B3", 4): "39b739dce5013d79",
    ("gpu", "B3", 8): "39b739dce5013d79",
    ("gpu", "B3", 12): "f2642349c7c7a717",
    ("gpu", "B3", 16): "26c3927b2da62ed0",
    ("gpu", "B3", 24): "6ac67756acdcd285",
    ("gpu", "B3", 32): "26c3927b2da62ed0",
}


@pytest.mark.parametrize("step", [None] + STEPS)
@pytest.mark.parametrize("fixture", ["B2", "B3"])
def test_the_prompt_cache_is_what_the_pre_fix_build_wrote(fixture, step, monkeypatch):
    # The digests above were recorded on the UNFUSED KDA path.  Since 2026-09-05
    # the fused kernel is the default (rounding-level differences in the cache
    # bytes, selection/argmax/singleton agreement unaffected -- see the other
    # tests in this file), so pin the path this chunked-select regression test is
    # actually about.
    import mlx_vlm.models.glm5_next.language as _lang

    monkeypatch.setenv("MLX_VLM_GLM5_FUSED_KDA", "0")
    monkeypatch.setattr(_lang, "_FUSED_KDA_ENV", None)
    suffixes = B2_SUFFIX if fixture == "B2" else B3_SUFFIX
    _, arrays, _ = _run(step, suffixes)
    assert _cache_digest(arrays)[:16] == PRE_FIX_CACHE[(DEV, fixture, step)]


# ------------------------------------------------ 2b. the selection, directly


@pytest.mark.parametrize("step", [None] + STEPS)
@pytest.mark.parametrize("fixture", ["B2", "B3"])
def test_the_selected_prompt_logits_are_the_unchunked_ones(fixture, step):
    """The fix's claim, stated as itself rather than as four decode steps.

    ``generate()`` hands the sampler one ``[1, vocab]`` row per batch row, taken
    at that row's last real column.  Chunking must not change which column that
    is, so the array must match the unchunked arm -- to the KDA-split drift, and
    with an identical argmax.  Device-free: the column list is integer
    arithmetic and is asserted exactly on both devices.
    """
    suffixes = B2_SUFFIX if fixture == "B2" else B3_SUFFIX
    pads, width = _pads(suffixes)
    ref, _, _ = _prompt_selection_logprobs(None, suffixes)
    got, columns, _ = _prompt_selection_logprobs(step, suffixes)

    assert columns == [width - 1 - p for p in pads], columns
    assert got.shape == ref.shape
    drift = float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))))
    assert drift < SELECTION_TOL, f"{fixture} step={step}: selection moved by {drift}"
    assert mx.array_equal(mx.argmax(got, axis=-1), mx.argmax(ref, axis=-1)), (
        f"{fixture} step={step}: the sampled token moved"
    )


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("fixture", ["B2", "B3"])
def test_a_chunked_batch_row_agrees_with_a_singleton_chunked_run(fixture, step):
    """Cross-check that owes nothing to the unchunked arm.

    Running row ``i`` ALONE at the same chunk size gives a batch with no right
    padding at all -- ``_last_real_column is None``, the untouched code path --
    so its first emitted token is the row's answer by construction, on any
    device.  Only the first token is compared: a singleton decodes with a
    different batch width from row ``i`` of the group, so the trajectories part
    company after it (measured: they do, from token 2, on both devices).
    """
    suffixes = B2_SUFFIX if fixture == "B2" else B3_SUFFIX
    batched, _, _ = _run(step, suffixes)
    singles = [_run(step, [s])[0][0] for s in suffixes]
    assert [r[0] for r in batched] == [r[0] for r in singles], f"step={step}"


# ------------------------------------------------ 3. three rows, two pad sizes


@pytest.mark.parametrize("step", STEPS)
def test_three_rows_with_two_different_right_pads(step):
    """Two boundaries in two different chunks, plus one row that ends at the width.

    The contract -- the emitted tokens -- holds at every step.  The prompt cache
    does NOT come out bit-equal to the unchunked arm at steps 4 and 12: the KDA
    (linear-attention) state is a running scan, and splitting it at a chunk boundary
    is not the same arithmetic as one scan over the whole prompt.  That drift is
    PRE-EXISTING (``test_the_prompt_cache_is_what_the_pre_fix_build_wrote`` pins
    those two arms to the byte against 422b69fa), it is bounded here at 2e-06, and
    it is the same drift the greedy chunked path already shipped.

    ``B3_STABLE_TOKENS`` decode steps are compared, which is all four on CPU and
    the first two on Metal.  That is not a relaxation of the claim, it is the end
    of what this fixture can measure: on a 2-layer randomly-initialised model
    row 0's third sampled token sits on a 0.074-logprob margin, and the Metal
    unchunked arm itself lands on the other side of it from the CPU unchunked arm
    ([52, 106, 5, 83] vs [52, 106, 60, 71]) -- neither of which involves chunking.
    The selection claim at full strength is
    ``test_the_selected_prompt_logits_are_the_unchunked_ones``, which passes at
    every step on both devices with the selected logits agreeing to 1.43e-06.
    """
    pads, _ = _pads(B3_SUFFIX)
    assert pads == [18, 7, 0], pads

    unchunked, unchunked_cache, _ = _run(None, B3_SUFFIX)
    chunked, chunked_cache, _ = _run(step, B3_SUFFIX)

    n = B3_STABLE_TOKENS
    assert [r[:n] for r in chunked] == [r[:n] for r in unchunked], f"step={step}"
    for i, (a, b) in enumerate(zip(chunked_cache, unchunked_cache)):
        assert a.shape == b.shape, f"cache array {i}"
        if a.size == 0:
            continue
        drift = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        assert drift < 2e-06, f"cache array {i}, step={step}: {drift}"


# ------------------------------------------------- 4. the unchunked arm is pinned
#
# The unchunked path is the one the selection fix must not touch: it takes no
# capture, and ``take`` reduces term for term to the ``seq - 1 - right_pad[i]``
# it was.  What is pinned is the PROMPT CACHE alone.
#
# It used to be ``_digest(rows, arrays)`` -- the emitted tokens and the cache
# together, recorded on 422b69fa.  The token half was a pin on a WRONG answer:
# a right-padded row's decode is corrupted by the KDA state (see RIGHT PADDING IS
# DECLINED in the module docstring), so those ids were never the row's answer and they
# moved the moment either defect was touched.  The cache half is the part that
# carries the claim, and it is unchanged: measured on the pristine 47b503a3 tree
# and on this one, CPU, the two digests below are byte-identical.
#
# Device-keyed for the same reason ``PRE_FIX_CACHE`` is.  The GPU values are the
# ``PRE_FIX_CACHE`` entries at ``step=None`` extended to full length; they are
# marked ``None`` here rather than guessed, and the GPU arm falls back to the
# 16-hex prefix pin that ``PRE_FIX_CACHE`` already carries.
UNCHUNKED_CACHE_DIGEST = {
    ("cpu", "B2"): (
        "9c582cbff6ae7517ad070827c637489444bf9c6c0d0324a4db9dbb470f5bb1a3"
    ),
    ("cpu", "B3"): (
        "f7900c633298b31bc1770168b0a8610df5b86de551a36735862bf6d1cfa2a316"
    ),
    ("gpu", "B2"): None,
    ("gpu", "B3"): None,
}


@pytest.mark.parametrize("fixture", ["B2", "B3"])
def test_the_unchunked_prompt_cache_is_byte_identical_to_the_pre_fix_build(fixture):
    suffixes = B2_SUFFIX if fixture == "B2" else B3_SUFFIX
    _, arrays, chunks = _run(None, suffixes)
    assert chunks == 0
    expected = UNCHUNKED_CACHE_DIGEST[(DEV, fixture)]
    digest = _cache_digest(arrays)
    if expected is None:
        assert digest[:16] == PRE_FIX_CACHE[(DEV, fixture, None)]
    else:
        assert digest == expected


@pytest.mark.parametrize("fixture", ["B2", "B3"])
def test_the_unchunked_arm_decodes_like_the_singletons_where_it_can(fixture):
    """What replaces the token pin: the claims that are actually true.

    * every row's FIRST token is the token that row emits when run alone -- the
      prompt selection is right, which is this module's subject;
    * the row that carries NO right padding matches its singleton for all four
      steps, so the batching itself is sound;
    * a PADDED row's later tokens are not asserted, and must not be: its KDA
      state absorbed the padding.  ``test_glm5_rightpad_decline`` measures that
      residual and is where the claim about it lives.
    """
    suffixes = B2_SUFFIX if fixture == "B2" else B3_SUFFIX
    pads, _ = _pads(suffixes)
    batched, _, _ = _run(None, suffixes)
    singles = [_run(None, [s])[0][0] for s in suffixes]

    assert [r[0] for r in batched] == [r[0] for r in singles]
    for i, pad in enumerate(pads):
        if pad == 0:
            assert batched[i] == singles[i], f"row {i} carries no padding"


# ------------------------------------------- 5. a batch without right padding


def test_a_batch_without_right_padding_captures_nothing():
    """The ordinary cold batch keeps the exact code path it had.

    ``_last_real_column`` is ``None``, so ``_rows_ending_in_chunk`` is empty on
    every chunk and ``prompt_step`` schedules the single argument it always did.
    """
    lm = _lm()
    rows = [list(range(3, 27)), list(range(7, 47))]
    from mlx_vlm.generate.ar import _left_pad_prompts

    padded = _left_pad_prompts(rows)
    batch = PromptProcessingBatch(
        model=lm,
        uids=[0, 1],
        input_ids=rows,
        max_tokens=[2, 2],
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=8,
    )
    assert batch._last_real_column is None
    assert batch._captured_last_logits == []
    while batch.needs_processing():
        assert batch._rows_ending_in_chunk(8) == []
        batch.prompt_step()
    assert batch._captured_last_logits == []


def test_right_padding_attached_after_construction_still_selects_the_last_real_token():
    """``_right_pad_per_row`` poked in after ``__init__`` (tests do this).

    No chunk could have captured anything in that case -- the whole prompt is in
    the final forward -- so ``generate()`` falls back to the exact formula it used
    before this fix rather than dereferencing a column list it never built.

    Asserted as ARITHMETIC, not as a token pin.  This fixture constructs the
    batch WITHOUT ``right_pad_per_row`` and attaches it afterwards, so the cache
    is never told there is padding: no ``prepare()``, no roll in ``finalize()``,
    and the row's own K/V still counts the padded columns.  Nothing about that
    is the served arithmetic, and the two token ids this used to pin (recorded
    at 422b69fa) had already gone stale by 47b503a3 -- [71, 99] -> [95, 99], a
    live failure in this file.  What the test is actually about is the fallback
    index, so that is what it now checks: the sampler is handed exactly
    ``logits[i, width - 1 - right_pad[i]]`` of the final forward, bit for bit.
    """
    lm = _lm()
    suffixes = B2_SUFFIX
    pads, width = _pads(suffixes)
    padded = _right_pad_prompts(suffixes, max_length=width)
    batch = PromptProcessingBatch(
        model=lm,
        uids=[0, 1],
        input_ids=suffixes,
        max_tokens=[1, 1],
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=None,
    )
    # The construction-time bookkeeping never happened: no column list, no
    # capture slots, so ``generate()`` must rebuild the index from the forward.
    assert batch._last_real_column is None
    assert batch._captured_last_logits == []
    batch._right_pad_per_row = list(pads)

    # The model's own output for the final (and only) forward, so the selected
    # row can be compared against the column it is supposed to come from.
    seen = {}
    model = batch.model

    class _Spy:
        def __call__(self, *args, **kwargs):
            output = model(*args, **kwargs)
            seen["logits"] = output.logits if hasattr(output, "logits") else output
            return output

        def __getattr__(self, name):
            return getattr(model, name)

    batch.model = _Spy()
    sampled = {}

    def sampler(logprobs):
        sampled["logprobs"] = mx.array(logprobs)
        return mx.argmax(logprobs, axis=-1)

    batch.generate(sampler=sampler, stop_criteria=lambda token: False)

    full = seen["logits"]
    assert full.shape[1] == width
    assert batch._captured_last_logits == []
    # ``generate`` hands the sampler log-probs, so undo the normalisation the
    # same way it was applied and compare against the raw column.
    expected = mx.stack(
        [full[i, width - 1 - int(pad)] for i, pad in enumerate(pads)], axis=0
    )
    expected = expected - mx.logsumexp(expected, axis=-1, keepdims=True)
    mx.eval(expected, sampled["logprobs"])
    assert mx.array_equal(sampled["logprobs"], expected)
