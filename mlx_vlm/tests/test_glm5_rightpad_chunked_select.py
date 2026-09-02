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
4.  the unchunked arm is byte-identical to the number recorded before the fix;
5.  a batch that is NOT right-padded takes no capture at all -- the code path an
    ordinary cold batch runs is untouched.
"""

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate.ar import PromptProcessingBatch, _right_pad_prompts
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel

STEPS = [4, 8, 12, 16, 24, 32]


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


def _cache_digest(arrays):
    """Digest of the prompt cache alone -- what the fix must leave untouched."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(repr(tuple(a.shape)).encode())
        h.update(memoryview(np.asarray(a.astype(mx.float32))).tobytes())
    return h.hexdigest()


def _digest(rows, arrays):
    h = hashlib.sha256(repr(rows).encode())
    for a in arrays:
        h.update(repr(tuple(a.shape)).encode())
        h.update(repr(a.dtype).encode())
        h.update(memoryview(np.asarray(a.astype(mx.float32))).tobytes())
    return h.hexdigest()


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
def test_the_chunked_prompt_cache_stays_bit_equal_to_the_unchunked_one(step):
    """The prefill was always right; the fix must not have bought (1) by moving it.

    B = 2 only.  At B = 3 the KDA scan lands a chunk boundary inside a row at two of
    the six step sizes and the split scan is not the same arithmetic as one scan
    over the whole prompt -- see ``test_three_rows_...`` below, which measures that
    and pins it against the PRE-FIX build rather than against the unchunked arm.
    """
    _, unchunked, _ = _run(None, B2_SUFFIX)
    _, chunked, _ = _run(step, B2_SUFFIX)

    assert len(chunked) == len(unchunked) == 10
    for i, (a, b) in enumerate(zip(chunked, unchunked)):
        assert a.shape == b.shape, f"cache array {i}"
        assert mx.array_equal(a, b), f"cache array {i}, step={step}"


# Digests of the CHUNKED prompt cache, recorded on 422b69fa BEFORE the fix
# (pristine ``generate/ar.py`` restored into this worktree, measured, restored).
# The fix captures logits out of a chunk's OUTPUT and passes the model exactly the
# arguments it passed before, so every one of these must survive it byte for byte.
# This is the guard that the selection fix did not become a prefill change.
PRE_FIX_CACHE = {
    ("B2", None): "9c582cbff6ae7517",
    ("B2", 4): "9c582cbff6ae7517",
    ("B2", 8): "9c582cbff6ae7517",
    ("B2", 12): "9c582cbff6ae7517",
    ("B2", 16): "9c582cbff6ae7517",
    ("B2", 24): "9c582cbff6ae7517",
    ("B2", 32): "9c582cbff6ae7517",
    ("B3", None): "f7900c633298b31b",
    ("B3", 4): "fc43c6b7ac48598c",
    ("B3", 8): "f7900c633298b31b",
    ("B3", 12): "756c54ee44d28d9f",
    ("B3", 16): "f7900c633298b31b",
    ("B3", 24): "f7900c633298b31b",
    ("B3", 32): "f7900c633298b31b",
}


@pytest.mark.parametrize("step", [None] + STEPS)
@pytest.mark.parametrize("fixture", ["B2", "B3"])
def test_the_prompt_cache_is_what_the_pre_fix_build_wrote(fixture, step):
    suffixes = B2_SUFFIX if fixture == "B2" else B3_SUFFIX
    _, arrays, _ = _run(step, suffixes)
    assert _cache_digest(arrays)[:16] == PRE_FIX_CACHE[(fixture, step)]


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
    """
    pads, _ = _pads(B3_SUFFIX)
    assert pads == [18, 7, 0], pads

    unchunked, unchunked_cache, _ = _run(None, B3_SUFFIX)
    chunked, chunked_cache, _ = _run(step, B3_SUFFIX)

    assert chunked == unchunked, f"step={step}"
    for i, (a, b) in enumerate(zip(chunked_cache, unchunked_cache)):
        assert a.shape == b.shape, f"cache array {i}"
        if a.size == 0:
            continue
        drift = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        assert drift < 2e-06, f"cache array {i}, step={step}: {drift}"


# ------------------------------------------------- 4. the unchunked arm is pinned
#
# Recorded on 422b69fa BEFORE the fix (pristine ``generate/ar.py`` restored into
# this worktree, this module run, then restored).  The unchunked path is the one
# the fix must not touch: it takes no capture, and ``take`` reduces term for term
# to the ``seq - 1 - right_pad[i]`` it was.
UNCHUNKED_B2_DIGEST = "39dc6566d9ae2d0d7a3887154f8443f60a306f6242c4d9bddc147ff212a810fa"
POST_HOC_RIGHT_PAD_TOKENS = [71, 99]
UNCHUNKED_B3_DIGEST = "fc992fe112ba53004b4f1fc53563a6110b3dbc94f07ea10c356680a1b0cc09f5"


def test_the_unchunked_arm_is_byte_identical_to_the_pre_fix_build():
    rows, arrays, chunks = _run(None, B2_SUFFIX)
    assert chunks == 0
    assert _digest(rows, arrays) == UNCHUNKED_B2_DIGEST

    rows3, arrays3, chunks3 = _run(None, B3_SUFFIX)
    assert chunks3 == 0
    assert _digest(rows3, arrays3) == UNCHUNKED_B3_DIGEST


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
    assert batch._last_real_column is None
    batch._right_pad_per_row = list(pads)
    gen_batch = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
    )
    tokens, _, _, _ = gen_batch._step()
    # Recorded on 422b69fa BEFORE the fix.  (Not [125, 99]: constructing without
    # ``right_pad_per_row`` skips the cache ``prepare()`` this batch would
    # normally get, so the arithmetic is not the served one -- the point of the
    # test is that the SELECTION arithmetic is unchanged.)
    assert [int(t) for t in tokens] == POST_HOC_RIGHT_PAD_TOKENS
