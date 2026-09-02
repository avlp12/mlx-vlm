"""The prefill leg must not build the KDA rollback stash.

``Glm5NextLinearAttention`` stashes eleven tensors per layer into ``gdn_sink`` so
that ``rollback_speculative_cache`` can replay a rejected speculative round.  Six
of those are shaped over the sequence, and two are ``mx.split`` VIEWS that pin
their parents -- ``v`` pins the conv output ``[B, S, 3*H*D]`` and ``b_o`` pins the
fused in-projection ``[B, S, 3*H*D + 2*head_dim + H]``.  At a decode step or a
verify block that is nothing.  On a whole-prompt prefill it is the dominant
retained allocation of the request (6.7 MB per prompt token on GLM-5.3-Flash),
and it is dead: every consumer of ``gdn_states`` reads it off a VERIFY forward.

``capture_gdn_states=False`` (default True) tells the model not to open the sink.
These tests pin (a) that the flag does exactly that and changes nothing else,
(b) that the two view-pinned parents really stop being held -- measured in bytes,
not asserted from the shapes -- and (c) that the helper only offers the kwarg to
models that advertise it.

The (b) group is a MEMORY measurement, and on Metal a naive
``get_active_memory`` delta reads the allocator rather than the retention: the
backend is asynchronous and the unevaluated tail of the graph is not resident
yet.  ``_held_once`` below states what it does about that, and with it the
numbers agree between CPU and Metal to within 1 % and are stable across
repetitions on both -- so these tests stay device-agnostic rather than being
demoted to CPU-only.
"""

import mlx.core as mx
import pytest

from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel
from mlx_vlm.speculative.utils import prefill_capture_kwargs

CAPTURE_IDS = [0, 1]


def _tiny_text_config(hidden=128, linear_heads=2, linear_head_dim=64):
    return TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=hidden,
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
            "num_heads": linear_heads,
            "head_dim": linear_head_dim,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        eos_token_id=1,
    )


def _lm(config=None):
    mx.random.seed(0)
    lm = LanguageModel(config or _tiny_text_config())
    lm.eval()
    mx.eval(lm.parameters())
    return lm


def _prompt(S):
    return (mx.arange(S, dtype=mx.int32) % 127)[None, :] + 1


# ---------------------------------------------------------------- (a) the flag


def test_capture_gdn_states_false_suppresses_the_sink_and_nothing_else():
    lm = _lm()
    prompt = _prompt(16)

    on = lm(prompt, cache=lm.make_cache(), capture_layer_ids=CAPTURE_IDS)
    off = lm(
        prompt,
        cache=lm.make_cache(),
        capture_layer_ids=CAPTURE_IDS,
        capture_gdn_states=False,
    )
    mx.eval(on.logits, off.logits, on.hidden_states, off.hidden_states)

    # The sink: opened on the verify default, not opened for prefill.
    assert on.gdn_states, "default must still build the rollback stash"
    assert len(on.gdn_states) == 1  # one linear_attention layer in this config
    assert len(on.gdn_states[0]) == 11  # the eleven-member rollback tuple
    assert not off.gdn_states, f"expected empty/None sink, got {off.gdn_states!r}"

    # Everything a prefill caller actually reads is untouched, bitwise.
    assert len(off.hidden_states) == len(on.hidden_states) == len(CAPTURE_IDS)
    for h in off.hidden_states:
        assert h.shape == (1, prompt.shape[1], lm.args.hidden_size)
    for a, b in zip(on.hidden_states, off.hidden_states):
        assert mx.array_equal(a, b)
    assert mx.array_equal(on.logits, off.logits)


def test_default_is_capture_on_so_verify_paths_are_unchanged():
    lm = _lm()
    block = _prompt(4)
    cache = lm.make_cache()
    lm(_prompt(8), cache=cache)
    out = lm(block, cache=cache, capture_layer_ids=CAPTURE_IDS)
    assert out.gdn_states and len(out.gdn_states) == 1
    out2 = lm(block, cache=lm.make_cache(), capture_layer_ids=CAPTURE_IDS,
              capture_gdn_states=True)
    assert out2.gdn_states


def test_no_capture_list_means_no_sink_either_way():
    lm = _lm()
    plain = lm(_prompt(8), cache=lm.make_cache())
    assert plain.gdn_states is None
    assert plain.hidden_states is None


# --------------------------------------------------------- (b) the retention


def _stash_parent_bytes(lm, S):
    """Bytes of the two ``mx.split`` parents the sink pins, per KDA layer."""
    attn = lm.model.layers[0].self_attn
    itemsize = lm.model.embed_tokens.weight.dtype.size
    conv_out = attn.conv_dim  # [B, S, 3*H*D] -- pinned by the stashed `v`
    in_proj = (  # [B, S, 3*H*D + 2*head_dim + H] -- pinned by the stashed `b_o`
        3 * attn.qkv_dim + attn.head_dim + attn.head_dim + attn.num_heads
    )
    return (conv_out + in_proj) * S * itemsize


def _held_once(lm, prompt, **extra):
    """Live bytes still referenced once the forward's result is in hand.

    Three things make this a measurement rather than a reading of the allocator:

    * ``mx.synchronize()`` before and after.  On Metal the command buffer is
      asynchronous, so without it ``get_active_memory`` samples a moving target
      -- the pre-fix version of this helper returned a NEGATIVE delta on one of
      six repetitions on an M3 Ultra.
    * force eval of exactly the retained object.  ``mx.eval(out.logits)`` alone
      leaves ``gdn_states`` as UNEVALUATED graph nodes, so how many of the split
      parents are resident at the sampling instant depends on backend scheduling.
      Evaluating the sink is what the claim is about: the stash's views pin their
      parents, and materialising them is what the server's downstream rollback
      consumer does anyway.
    * ``mx.clear_cache()`` on both sides, so the allocator's free pool is not
      counted as live on either.

    And the delta is taken across DROPPING the result, not against a baseline
    sampled before the forward: on Metal a stale residue from a previous arm can
    survive one ``clear_cache`` and inflate such a baseline (that is how the
    pre-fix helper produced a negative "held" of -1,582,856 B for the on arm of
    ``test_the_view_pinned_parents_are_no_longer_held``).  Sampling on both sides
    of ``del out, cache`` cancels anything that was already resident, and answers
    the question the tests actually ask: how many bytes does holding this
    forward's result cost.

    With all four, the number is bit-stable across repetitions on CPU and on
    Metal alike (six repetitions each, 2026-09-03).
    """
    mx.synchronize()
    mx.clear_cache()
    cache = lm.make_cache()
    out = lm(prompt, cache=cache, capture_layer_ids=CAPTURE_IDS, **extra)
    mx.eval(out.logits)
    if out.gdn_states:
        mx.eval([t for layer in out.gdn_states for t in layer if isinstance(t, mx.array)])
    mx.synchronize()
    mx.clear_cache()
    with_result = mx.get_active_memory()
    del out, cache
    mx.synchronize()
    mx.clear_cache()
    without_result = mx.get_active_memory()
    return with_result - without_result


def _held_bytes(lm, prompt, reps=3, **extra):
    """The floor of a few repetitions: any residue from a previous arm can only
    make a sample bigger, never smaller."""
    return min(_held_once(lm, prompt, **extra) for _ in range(reps))


def test_the_view_pinned_parents_are_no_longer_held():
    lm = _lm()
    S = 256
    prompt = _prompt(S)
    # Warm up: the fused in-projection concatenates its six weights once and
    # caches them on the module, and that allocation must not land in a measured
    # arm.
    mx.eval(lm(prompt, cache=lm.make_cache(), capture_layer_ids=CAPTURE_IDS).logits)
    mx.clear_cache()

    parents = _stash_parent_bytes(lm, S)
    on = _held_bytes(lm, prompt)
    off = _held_bytes(lm, prompt, capture_gdn_states=False)

    assert off < on, f"suppressing the sink freed nothing (on={on}, off={off})"
    # The load-bearing claim: what stops being held is at least the two parents
    # the stash pinned through views, per token of prompt.
    assert on - off >= parents, (
        f"freed {on - off} B, but the two split parents alone are {parents} B "
        f"(on={on}, off={off}, S={S})"
    )
    # Measured on this config (fp32, H=2, D=64, S=256), 2026-09-03:
    #   CPU  on 3,198,050 B  off 1,870,946 B  freed 1,327,104 B
    #   GPU  on 3,197,042 B  off 1,886,322 B  freed 1,310,720 B
    # i.e. 1,296 (CPU) / 1,280 (GPU) elements per token against 898
    # elements/token for the parents alone.  The six stashed tensors would be
    # 1,666 elements/token; the remaining gap is not identified and is left
    # unclaimed rather than asserted -- the inequality above is what matters.
    assert (on - off) / S >= _stash_parent_bytes(lm, 1)


def test_retention_scales_with_prompt_length_before_and_not_after():
    """The whole point: the stash is O(S) and what replaces it is not."""
    lm = _lm()
    mx.eval(lm(_prompt(64), cache=lm.make_cache(), capture_layer_ids=CAPTURE_IDS).logits)
    mx.clear_cache()

    on_128 = _held_bytes(lm, _prompt(128))
    on_256 = _held_bytes(lm, _prompt(256))
    off_128 = _held_bytes(lm, _prompt(128), capture_gdn_states=False)
    off_256 = _held_bytes(lm, _prompt(256), capture_gdn_states=False)

    grew_on = on_256 - on_128
    grew_off = off_256 - off_128
    # Measured 2026-09-03: CPU grew_on 1,415,971 B / grew_off 776,995 B;
    # GPU grew_on 1,414,451 B / grew_off 775,475 B.  The slope drop is
    # 638,976 B on BOTH devices, to the byte, against 459,776 B for the two
    # split parents alone.
    assert grew_on > 0
    assert grew_off < grew_on, (
        f"per-token growth did not fall: on {grew_on} B over 128 tokens, "
        f"off {grew_off} B"
    )
    # The stash's own per-token cost must be gone from the slope.
    assert grew_on - grew_off >= _stash_parent_bytes(lm, 128)


# ------------------------------------------------------------- (c) the helper


def test_helper_only_offers_the_kwarg_to_models_that_advertise_it():
    class Advertises:
        supports_capture_gdn_states = True

    class Silent:
        pass

    caps = {"capture_layer_ids": [1, 2]}
    assert prefill_capture_kwargs(Advertises(), caps) == {
        "capture_layer_ids": [1, 2],
        "capture_gdn_states": False,
    }
    # A model that forwards **kwargs into its decoder stack would raise on an
    # unknown one, so it is never offered.
    assert prefill_capture_kwargs(Silent(), caps) == caps
    # MTP asks for return_hidden, not a capture list: no sink to suppress.
    mtp = {"return_hidden": True, "return_shared_kv": True}
    assert prefill_capture_kwargs(Advertises(), mtp) == mtp
    assert prefill_capture_kwargs(Advertises(), {}) == {}
    # Never mutates the caller's dict.
    assert caps == {"capture_layer_ids": [1, 2]}


def test_glm5_next_advertises_support():
    assert LanguageModel.supports_capture_gdn_states is True
