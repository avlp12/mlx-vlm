"""The two chunked-prefill fixes meeting on one forward, at B = 2, right-padded.

``glm5-rightpad-chunk-select`` (8a00006c) and ``glm5-dflash-chunked-prefill-batch``
(e8316a17 + 4f3a27bd) both rewrote ``PromptProcessingBatch``'s chunk loop, from
opposite ends, and neither branch ever ran the other's arm:

* **A** samples each row's LAST REAL token.  On a right-padded batch that column can
  land in an earlier chunk than the final forward, so ``prompt_step`` keeps a
  ``[1, vocab]`` slice out of whichever chunk contains it.  A needs no drafter.
* **B** carries the speculative capture on EVERY chunk and stitches the pieces back,
  so a hidden-reading drafter is handed the whole prompt rather than the last
  forward.  B needs no right padding.

The merge has to let both read the SAME chunk output, in order, and release it
before the eval.  This file is the arm where both are live at once: a right-padded
mixed warm/cold B = 2 batch, chunked, with a hidden-reading stub drafter attached.

It is also what the narrowing of B's §9 refusal rests on.  That refusal declined
BOTH the chunking and the trim for a right-padded batch; with A merged, the chunked
selection is exact, and with ``keep=None`` the accumulator's stitch reproduces the
full padded width column for column, so only the TRIM half survives.  Everything
that narrowing claims is asserted below.
"""

import mlx.core as mx

from mlx_vlm.generate.ar import (
    PromptProcessingBatch,
    _left_pad_prompts,
    _right_pad_prompts,
)
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel

CAPTURE_IDS = [0, 1]
HIDDEN = 128
STEP = 8  # four chunks of 8 plus the final forward, over a 40-wide prefill

# The mixed warm/cold shape ``BatchGenerator._build_mixed_prompt_batch`` builds:
# row 0 is warm (16 tokens already in the cache, 24 to prefill), row 1 is cold (40).
# ``right_pad_per_row = [max_suffix - suffix_i]``, so row 0 carries 16 pad columns
# and its last real token sits at absolute column 23 -- inside chunk 3, not the
# final forward.
FULL = [list(range(3, 43)), list(range(7, 47))]
PREFIX = [16, 0]
SUFFIX = [FULL[i][PREFIX[i] :] for i in range(2)]
MAX_SUFFIX = max(len(s) for s in SUFFIX)
RIGHT_PAD = [MAX_SUFFIX - len(s) for s in SUFFIX]
META = [{"full_input_ids": FULL[i], "prefix_len": PREFIX[i]} for i in range(2)]


def _tiny_text_config():
    return TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=HIDDEN,
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


class _Spy:
    """Records every forward's kwargs and every capture the model returns."""

    def __init__(self, lm):
        self._lm = lm
        self.calls = []
        self.captures = []
        self.gdn = []

    def __call__(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        out = self._lm(*args, **kwargs)
        self.captures.append(getattr(out, "hidden_states", None))
        self.gdn.append(getattr(out, "gdn_states", None))
        return out

    def __getattr__(self, name):
        return getattr(self._lm, name)


class _HiddenReadingStubDrafter:
    """The smallest object ``speculative/utils.py`` accepts as a dflash drafter.

    ``prefill_context_keep`` returns a real window, so a batch that does NOT refuse
    the trim would trim -- which is exactly what this fixture must show it does not.
    """

    def __init__(self, keep=15):
        from types import SimpleNamespace

        self.config = SimpleNamespace(target_layer_ids=list(CAPTURE_IDS))
        self._keep = keep

    def prefill_context_keep(self):
        return self._keep

    def adopt_pretruncated_context(self, cache, skip):  # pragma: no cover - unused
        raise AssertionError("the trim is refused on this batch; nothing to adopt")


def _first_tokens(gen_batch):
    tokens = getattr(gen_batch, "first_tokens", None)
    if tokens is None:
        tokens = gen_batch._next_tokens
    mx.eval(tokens)
    return tokens.tolist()


def _right_padded_pair(*, step, drafter):
    lm = _lm()
    spy = _Spy(lm)
    padded = _right_pad_prompts(SUFFIX, max_length=MAX_SUFFIX)
    draft_kwargs = (
        dict(draft_model=drafter, draft_kind="dflash") if drafter is not None else {}
    )
    batch = PromptProcessingBatch(
        model=spy,
        uids=[0, 1],
        input_ids=SUFFIX,
        max_tokens=[4, 4],
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=step,
        right_pad_per_row=list(RIGHT_PAD),
        suffix_lens=[len(s) for s in SUFFIX],
        apc_meta=META,
        **draft_kwargs,
    )
    while batch.needs_processing():
        batch.prompt_step()
    # ``generate()`` releases the per-request capture state with everything else,
    # so what the chunk loop kept has to be read before it runs.
    after_loop = dict(
        last_real_column=batch._last_real_column,
        captured=[
            None if c is None else tuple(c.shape) for c in batch._captured_last_logits
        ],
        final_forward_width=batch._inputs_embeds.shape[1],
    )
    gen_batch = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
    )
    return gen_batch, spy, batch, after_loop


def _singleton(row, *, step):
    """The same row, alone, chunked -- the arm a served comparison would run.

    Alone there is no right padding at all, so this arm never touches A's capture
    path: it is an independent computation of what row ``row`` should emit.
    """
    lm = _lm()
    ids = [SUFFIX[row]]
    batch = PromptProcessingBatch(
        model=lm,
        uids=[0],
        input_ids=ids,
        max_tokens=[4],
        inputs_embeds=lm.model.embed_tokens(_left_pad_prompts(ids)),
        prompt_kwargs={},
        prefill_step_size=step,
        apc_meta=[{"full_input_ids": FULL[row], "prefix_len": PREFIX[row]}],
    )
    while batch.needs_processing():
        batch.prompt_step()
    gen_batch = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
    )
    return _first_tokens(gen_batch)[0]


def test_both_chunk_loop_fixes_survive_on_one_right_padded_capturing_batch():
    """A's selection, B's capture and the gdn refusal, all on the same forwards.

    (i)   every row emits what it emits when run ALONE, chunked -- which is A's
          claim, and the one that fails at either branch's parent (the padded row's
          take index goes negative once the loop has eaten the leading columns);
    (ii)  the drafter is handed the WHOLE prompt: the capture is full width and,
          per row, bit-equal to the driver's own per-chunk captures concatenated on
          the time axis -- which is B's claim -- with the trim refused, so the
          offset is 0 and the whole capture is the drafter's context;
    (iii) no KDA rollback stash is built on any forward.
    """
    gen_batch, spy, batch, after_loop = _right_padded_pair(
        step=STEP, drafter=_HiddenReadingStubDrafter()
    )

    # The prefill really chunked, and row 0's last real token really did land in an
    # earlier chunk than the final forward -- otherwise (i) proves nothing.  Note
    # ``40 - 1 - 16 = 23`` against the whole padded width, while the pre-fix formula
    # would have read ``final_forward_width - 1 - 16``, which is negative here.
    assert batch.prefill_step_size == STEP
    assert len(spy.calls) == 5, [c.get("n_to_process") for c in spy.calls]
    assert after_loop["last_real_column"] == [23, 39]
    assert after_loop["captured"] == [(1, 128), None]  # row 1 ends in the final forward
    assert after_loop["final_forward_width"] - 1 - RIGHT_PAD[0] < 0

    # ---- (i) A's selection: the chunked batch agrees with the singleton run, per row
    batched = _first_tokens(gen_batch)
    alone = [_singleton(row, step=STEP) for row in (0, 1)]
    assert batched == alone, f"batched {batched} != singleton {alone}"

    # ---- (ii) B's capture: whole prompt, both rows, trim refused so offset is 0
    assert gen_batch.hidden.shape == (2, MAX_SUFFIX, len(CAPTURE_IDS) * HIDDEN)
    assert gen_batch.target_hidden_offset == 0
    assert batch._prefill_hidden.keep is None, "the trim half of the refusal"
    # The capture is asked for on EVERY forward, not just the last one.
    for i, call in enumerate(spy.calls):
        assert call["capture_layer_ids"] == list(CAPTURE_IDS), f"forward {i}"
    # And what the drafter gets is exactly the pieces those forwards produced,
    # concatenated on the time axis -- per row, bit-equal, no rows dropped.
    per_layer = [
        mx.concatenate([c[layer] for c in spy.captures], axis=1)
        for layer in range(len(CAPTURE_IDS))
    ]
    stitched = mx.concatenate(per_layer, axis=-1)
    mx.eval(stitched, gen_batch.hidden)
    assert stitched.shape == gen_batch.hidden.shape
    for row in range(2):
        assert mx.array_equal(
            stitched[row : row + 1], gen_batch.hidden[row : row + 1]
        ), f"row {row}"

    # A warm row's PROMPT is prefix + suffix, not the right-padded suffix it prefills.
    assert gen_batch.prompt_tokens.shape == (2, len(FULL[0]))
    for row, whole in enumerate(FULL):
        assert gen_batch.prompt_tokens[row].tolist() == whole, f"row {row}"

    # ---- (iii) no gdn sink, on any forward
    for i, call in enumerate(spy.calls):
        assert call["capture_gdn_states"] is False, f"forward {i}"
    for i, sink in enumerate(spy.gdn):
        assert not sink, f"forward {i} built a gdn stash: {sink!r}"
