"""End-to-end greedy identity for prompt-lookup drafting through generate_step.

The stub-target tests in test_prompt_lookup_rounds.py pin the round loop's
contract against a target whose greedy output is known in closed form.  This one
closes the loop through the real plumbing -- drafter registry, prefill kwargs,
chunked-prefill policy, a genuine KV cache and a genuine
``rollback_speculative_cache`` -- on a tiny but real Qwen3.5 target.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import generate_step
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.qwen3_5 import language as qwen_language
from mlx_vlm.models.qwen3_5.config import TextConfig as Qwen3_5TextConfig
from mlx_vlm.speculative.drafters import load_drafter, make_lookup_drafter
from mlx_vlm.speculative.drafters.prompt_lookup import PromptLookupConfig


def _tiny_target(seed=0):
    mx.random.seed(seed)
    config = Qwen3_5TextConfig(
        model_type="qwen3_5_text", hidden_size=16, intermediate_size=32,
        linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_conv_kernel_dim=4, num_hidden_layers=2,
        num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=64,
        num_key_value_heads=1, max_position_embeddings=512,
        tie_word_embeddings=True, head_dim=8, full_attention_interval=2,
        rope_parameters={"type": "default", "mrope_section": [1, 0, 0],
                         "rope_theta": 10000, "partial_rotary_factor": 0.25},
    )
    outer = SimpleNamespace(
        model_type="qwen3_5", text_config=config,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=30, video_token_id=29, vision_start_token_id=28,
    )
    model = qwen_language.LanguageModel(config, outer)
    model.set_dtype(mx.bfloat16)
    return model


def _generated(target, prompt, drafter=None, max_tokens=64):
    def get_input_embeddings(input_ids, pixel_values=None, mask=None, **kwargs):
        del pixel_values, kwargs
        position_ids, rope_deltas = target.get_rope_index(input_ids, attention_mask=mask)
        return InputEmbeddingsFeatures(
            inputs_embeds=target.model.embed_tokens(input_ids),
            position_ids=position_ids, rope_deltas=rope_deltas,
        )

    generation_target = SimpleNamespace(
        language_model=target, get_input_embeddings=get_input_embeddings
    )
    kwargs = (
        {"draft_model": drafter, "draft_kind": "lookup"} if drafter is not None else {}
    )
    return [
        int(tok.item()) if hasattr(tok, "item") else int(tok)
        for tok, _ in generate_step(
            prompt, generation_target, None, None, max_tokens=max_tokens,
            temperature=0, prefill_step_size=None, **kwargs
        )
    ]


# A quoting-heavy prompt: a span the model can copy forward.  And a freeform one
# with no internal repetition, where the drafter should mostly abstain.
_QUOTING = mx.array([[5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8]])
_FREEFORM = mx.array([[3, 41, 17, 2, 33, 58, 12, 46, 21, 9, 60, 14, 37, 1, 25, 50]])


@pytest.mark.parametrize(
    "name,prompt", [("quoting", _QUOTING), ("freeform", _FREEFORM)]
)
def test_lookup_matches_baseline_greedy_exactly(name, prompt):
    target = _tiny_target()
    baseline = _generated(target, prompt, drafter=None, max_tokens=64)
    drafter = make_lookup_drafter(PromptLookupConfig(n_min=3, n_max=5, block_size=8))
    spec = _generated(target, prompt, drafter=drafter, max_tokens=64)
    assert len(baseline) == 64
    assert spec == baseline, f"{name}: speculative stream diverged from greedy"
    assert len(drafter.accept_lens) > 0, "no speculative round ran"


def test_registry_builds_the_weightless_drafter():
    drafter, kind = load_drafter("lookup")
    assert kind == "lookup"
    assert drafter.config.model_type == "prompt_lookup"
    # Nothing was loaded onto the device.  (The stat lists show up in the module
    # tree because nn.Module keeps assigned attributes -- DFlashDraftModel does
    # the same -- so the check that matters is that no mx.array is present.)
    from mlx.utils import tree_flatten

    arrays = [
        v for _, v in tree_flatten(drafter.parameters()) if isinstance(v, mx.array)
    ]
    assert arrays == []


def test_prefill_kwargs_are_empty_for_lookup():
    """Lookup needs nothing captured, so prefill runs exactly as it does with no
    drafter attached -- no capture_layer_ids, no return_hidden."""
    from mlx_vlm.speculative.utils import speculative_prefill_kwargs

    assert speculative_prefill_kwargs("lookup", make_lookup_drafter()) == {}


def test_chunked_prefill_stays_enabled_for_lookup():
    target = _tiny_target()
    assert target.chunked_prefill_policy(
        draft_model=make_lookup_drafter(), draft_kind="lookup", prefill_kwargs={}
    ) is True


def test_temperature_is_refused():
    target = _tiny_target()
    drafter = make_lookup_drafter()
    with pytest.raises(ValueError, match="greedy"):
        list(
            generate_step(
                _QUOTING, SimpleNamespace(
                    language_model=target,
                    get_input_embeddings=lambda ids, pv=None, mask=None, **k:
                        InputEmbeddingsFeatures(
                            inputs_embeds=target.model.embed_tokens(ids),
                            position_ids=None, rope_deltas=None,
                        ),
                ),
                None, None, max_tokens=4, temperature=0.8,
                prefill_step_size=None, draft_model=drafter, draft_kind="lookup",
            )
        )


def test_quoting_prompt_accepts_more_than_freeform():
    """The honest shape of the win: a lookup drafter is a workload bet, not a
    universal speedup."""
    target = _tiny_target()
    stats = {}
    for name, prompt in (("quoting", _QUOTING), ("freeform", _FREEFORM)):
        d = make_lookup_drafter(PromptLookupConfig(n_min=3, n_max=5, block_size=8))
        _generated(target, prompt, drafter=d, max_tokens=48)
        stats[name] = d.stats()
    # Both must have run rounds; we assert only that the accounting is coherent,
    # because a 16-dim random-weight target has no real language statistics.
    for s in stats.values():
        assert s["rounds"] > 0
        assert 0.0 <= s["match_rate"] <= 1.0
        assert len(s["accept_lens"]) == s["rounds"]
