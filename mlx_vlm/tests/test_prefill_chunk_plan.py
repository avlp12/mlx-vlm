"""L35 -- the two prefill wins, tested without a GPU or a real model.

(a) ``num_logits_to_keep`` on the forward that an UNCHUNKED prefill runs (the
    chunk LOOP never needed it -- it drops ``chunk_out`` before the eval, and MLX
    does not compute an unreferenced graph).
(b) the tail-chunk merge.

BOTH ARE ON BY DEFAULT since the rule-13 rail of 2026-09-07 (ledger I1437), so
"env unset" below is the SHIPPED path and every base-behaviour assertion sets
the env to "0" explicitly.  ``TestBaseParityWhenBothAreOff`` is the contract
that "0" on both really is a6634a75: same chunk plan, same keep-kwargs, same
forward widths.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from mlx_vlm.generate.ar import PromptProcessingBatch
from mlx_vlm.generate.common import (
    next_prefill_chunk,
    plan_prefill_chunks,
    prefill_logits_keep_enabled,
    prefill_logits_keep_kwargs,
    prefill_tail_merge_enabled,
    prefill_tail_min,
    prefill_tail_mode,
    tp_forward_token_room,
)

STEP = 8192


def _ranges(plan):
    """The chunk plan as absolute [start, end) ranges."""
    out = []
    pos = 0
    for n in plan:
        out.append((pos, pos + n))
        pos += n
    return out


class TestChunkPlanDisabled:
    """With the merge off every plan is exactly ``min(step, remaining)``."""

    @pytest.mark.parametrize(
        "remaining", [1, 2, 8191, 8192, 8193, 16384, 33305, 131071]
    )
    def test_matches_the_old_expression(self, remaining):
        expected = []
        left = remaining
        while left > 0:
            n = min(STEP, left)
            expected.append(n)
            left -= n
        assert plan_prefill_chunks(remaining, STEP, enabled=False) == expected

    def test_explicit_zero_is_disabled(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "0")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "1024")
        # 33,305 = 4*8192 + 537: the PFINAL as-fed 32k prompt's tail.
        assert plan_prefill_chunks(33305, STEP) == [STEP, STEP, STEP, STEP, 537]

    @pytest.mark.parametrize("off", ["0", "false", "FALSE", "no", "off", " 0 "])
    def test_the_spellings_that_turn_it_off(self, monkeypatch, off):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", off)
        assert prefill_tail_merge_enabled() is False
        assert next_prefill_chunk(STEP + 537, STEP) == STEP

    def test_env_unset_is_ENABLED_now(self, monkeypatch):
        # The default flip (I1437).  Same prompt as above, merged tail.
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", raising=False)
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "1024")
        assert prefill_tail_merge_enabled() is True
        assert plan_prefill_chunks(33305, STEP) == [STEP, STEP, STEP, 8729]

    def test_empty_string_is_the_default_not_off(self, monkeypatch):
        # A launcher that emits ``NAME=`` for a variable it did not set must not
        # silently disable a default-ON lever (the shape of the TP passthrough
        # bug in server/tp_mode.py::launch_worker).
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "")
        assert prefill_tail_merge_enabled() is True
        assert prefill_logits_keep_enabled() is True


class TestChunkPlanMerged:
    def test_shorter_than_one_step_is_one_chunk(self):
        assert plan_prefill_chunks(5000, STEP, enabled=True) == [5000]

    @pytest.mark.parametrize("k", [1, 2, 4, 16])
    def test_exact_multiple_is_untouched(self, k):
        assert plan_prefill_chunks(k * STEP, STEP, enabled=True) == [STEP] * k

    def test_tail_below_threshold_is_folded_into_the_previous_chunk(self):
        # 537 < 1024 -> the last two cells become one 8,729-wide chunk.
        assert plan_prefill_chunks(33305, STEP, tail_min=1024, enabled=True) == [
            STEP,
            STEP,
            STEP,
            8729,
        ]

    def test_tail_at_or_above_threshold_is_left_alone(self):
        assert plan_prefill_chunks(
            4 * STEP + 1024, STEP, tail_min=1024, enabled=True
        ) == [STEP] * 4 + [1024]
        assert plan_prefill_chunks(
            4 * STEP + 2000, STEP, tail_min=1024, enabled=True
        ) == [STEP] * 4 + [2000]

    def test_only_the_last_cell_moves(self):
        plan = plan_prefill_chunks(10 * STEP + 3, STEP, tail_min=1024, enabled=True)
        assert plan[:-1] == [STEP] * 9
        assert plan[-1] == STEP + 3

    def test_balance_mode_splits_the_last_cell(self):
        plan = plan_prefill_chunks(
            2 * STEP + 3, STEP, tail_min=1024, mode="balance", enabled=True
        )
        assert plan == [STEP, 4098, 4097]
        assert max(plan) <= STEP  # memory-neutral, unlike grow

    def test_a_merged_chunk_never_exceeds_step_plus_threshold(self):
        for tail in range(1, 1024):
            plan = plan_prefill_chunks(
                3 * STEP + tail, STEP, tail_min=1024, enabled=True
            )
            assert max(plan) < STEP + 1024

    @pytest.mark.parametrize("mode", ["grow", "balance"])
    @pytest.mark.parametrize("remaining", [1, 7, STEP - 1, STEP, STEP + 1, 33305])
    def test_ranges_concatenate_to_the_whole_prompt(self, mode, remaining):
        plan = plan_prefill_chunks(
            remaining, STEP, tail_min=1024, mode=mode, enabled=True
        )
        assert all(n > 0 for n in plan)
        assert sum(plan) == remaining
        ranges = _ranges(plan)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == remaining
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))

    def test_tail_min_zero_disables_the_merge(self):
        assert plan_prefill_chunks(STEP + 3, STEP, tail_min=0, enabled=True) == [
            STEP,
            3,
        ]


class TestEnvReading:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", raising=False)
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MODE", raising=False)
        assert prefill_tail_min() == 1024
        assert prefill_tail_mode() == "grow"

    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "not-a-number")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MODE", "sideways")
        assert prefill_tail_min() == 1024
        assert prefill_tail_mode() == "grow"

    def test_merge_defaults_on_keep_defaults_on(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", raising=False)
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", raising=False)
        assert prefill_tail_merge_enabled() is True
        assert prefill_logits_keep_enabled() is True

    def test_env_on(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "1")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "600")
        assert next_prefill_chunk(STEP + 537, STEP) == STEP + 537
        assert next_prefill_chunk(STEP + 700, STEP) == STEP

    def test_env_off(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "0")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "600")
        assert next_prefill_chunk(STEP + 537, STEP) == STEP
        assert next_prefill_chunk(STEP + 700, STEP) == STEP


class TestLogitsKeepGate:
    class _Keeper:
        supports_num_logits_to_keep = True

    class _Plain:
        pass

    def test_on_by_default(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", raising=False)
        assert prefill_logits_keep_kwargs(self._Keeper(), 8192) == {
            "num_logits_to_keep": 1
        }

    @pytest.mark.parametrize("off", ["0", "false", "No", "OFF"])
    def test_explicit_off_restores_the_old_argument_list(self, monkeypatch, off):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", off)
        assert prefill_logits_keep_enabled() is False
        assert prefill_logits_keep_kwargs(self._Keeper(), 8192) == {}

    def test_on_for_a_wide_forward(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "1")
        assert prefill_logits_keep_kwargs(self._Keeper(), 8192) == {
            "num_logits_to_keep": 1
        }

    def test_withheld_at_width_one(self, monkeypatch):
        # A decode step (and a post-chunk prefill) must keep the exact argument
        # list it had, so its kernels are the ones that were measured.
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "1")
        assert prefill_logits_keep_kwargs(self._Keeper(), 1) == {}

    def test_withheld_for_a_model_that_does_not_advertise_it(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "1")
        assert prefill_logits_keep_kwargs(self._Plain(), 8192) == {}


class TestPromptStepUsesThePlan:
    def _batch(self, tokens, step):
        return PromptProcessingBatch(
            model=MagicMock(),
            uids=[1],
            input_ids=[list(range(tokens))],
            max_tokens=[1],
            inputs_embeds=mx.ones((1, tokens, 4)),
            prompt_kwargs={},
            prefill_step_size=step,
            warm_cache=[SimpleNamespace(state=mx.array([1]))],
        )

    def test_disabled_leaves_a_short_tail(self, monkeypatch):
        # The batch loop stops chunking as soon as what is left fits in one step
        # (``needs_processing``), so the 3-token tail is handed to the FINAL
        # forward instead of becoming a third-of-a-chunk of its own.
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "0")
        monkeypatch.setattr(mx, "async_eval", MagicMock())
        monkeypatch.setattr(mx, "clear_cache", MagicMock())
        batch = self._batch(12, 8)
        assert batch.prompt_step() == 8
        assert batch.needs_processing() is False
        assert batch._inputs_embeds.shape[1] == 4

    def test_merge_folds_the_tail(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "1")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "4")
        monkeypatch.setattr(mx, "async_eval", MagicMock())
        monkeypatch.setattr(mx, "clear_cache", MagicMock())
        batch = self._batch(12, 8)
        assert batch.prompt_step() == 11
        assert batch.needs_processing() is False

    def test_merge_folds_the_tail_with_the_env_unset(self, monkeypatch):
        # Same as above with nothing exported: the default is now ON.
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", raising=False)
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "4")
        monkeypatch.setattr(mx, "async_eval", MagicMock())
        monkeypatch.setattr(mx, "clear_cache", MagicMock())
        batch = self._batch(12, 8)
        assert batch.prompt_step() == 11

    def test_a_checkpoint_column_still_wins(self, monkeypatch):
        # The APC/vault rung is clamped AFTER the plan, and the clamp can only
        # shorten -- so a checkpointing request quietly gets the old plan back
        # rather than a rung that was never taken.
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "1")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "4")
        monkeypatch.setattr(mx, "async_eval", MagicMock())
        monkeypatch.setattr(mx, "clear_cache", MagicMock())
        batch = self._batch(12, 8)
        batch._next_apc_checkpoint_column = lambda: 8
        batch._store_apc_exact_checkpoints = MagicMock()
        assert batch.prompt_step() == 8


class TestBatchGenerateKeepKwarg:
    class _Sentinel(Exception):
        pass

    class _Cache:
        state = mx.array([1])

        def prepare(self, *a, **k):
            return None

        def finalize(self, *a, **k):
            return None

    def _batch(self, right_pad=None):
        recorded = {}

        def model(*args, **kwargs):
            recorded.update(kwargs)
            raise TestBatchGenerateKeepKwarg._Sentinel()

        model.supports_num_logits_to_keep = True
        batch = PromptProcessingBatch(
            model=model,
            uids=[1],
            input_ids=[[1, 2, 3, 4]],
            max_tokens=[1],
            inputs_embeds=mx.ones((1, 4, 4)),
            prompt_kwargs={},
            prefill_step_size=None,
            warm_cache=[TestBatchGenerateKeepKwarg._Cache()],
            right_pad_per_row=right_pad,
        )
        return batch, recorded

    def _run(self, batch):
        with pytest.raises(TestBatchGenerateKeepKwarg._Sentinel):
            batch.generate(lambda x: mx.array([0]), MagicMock())

    def test_absent_when_disabled(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "0")
        batch, recorded = self._batch()
        self._run(batch)
        assert "num_logits_to_keep" not in recorded

    def test_present_with_the_env_unset(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", raising=False)
        batch, recorded = self._batch()
        self._run(batch)
        assert recorded["num_logits_to_keep"] == 1

    def test_still_withheld_when_right_padded_by_default(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", raising=False)
        batch, recorded = self._batch(right_pad=[1])
        self._run(batch)
        assert "num_logits_to_keep" not in recorded

    def test_present_on_an_unchunked_prefill(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "1")
        batch, recorded = self._batch()
        self._run(batch)
        assert recorded["num_logits_to_keep"] == 1

    def test_withheld_when_right_padded(self, monkeypatch):
        # The row's last real token is not the last column, so a keep-1 slice
        # would not contain the row it needs.
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "1")
        batch, recorded = self._batch(right_pad=[1])
        self._run(batch)
        assert "num_logits_to_keep" not in recorded


# --------------------------------------------------------------- generate_step
#
# The loop PFINAL measured (bench/hwdossier/receipts/sweep11/
# PFINAL_PREFILL_PATH_20260906): unlike the batch driver it chunks all the way
# down to one token, so a short tail IS a chunk of its own there -- the 537-token
# chunk that cost 3.33 ms/token against 2.38 for the full ones.


def _chunk_widths(prompt_len, step, monkeypatch):
    import sys
    from unittest.mock import patch

    from mlx_vlm.generate import ar as ar_module

    generate_module = sys.modules["mlx_vlm.generate"]

    hidden_width = 4
    widths = []

    model = MagicMock()
    model.no_chunked_prefill = False
    model.chunked_prefill_policy.return_value = True
    model.language_model.supports_logits_to_keep = False
    model.language_model.supports_num_logits_to_keep = True

    def _forward(*args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        n = embeds.shape[1] if embeds is not None else args[0].shape[1]
        widths.append((n, kwargs.get("num_logits_to_keep")))
        keep = kwargs.get("num_logits_to_keep") or 0
        rows = keep if keep else n
        return SimpleNamespace(
            logits=mx.zeros((1, rows, 4)),
            hidden_states=None,
            shared_kv_states={},
            cross_attention_states=None,
            encoder_outputs=None,
        )

    model.language_model.side_effect = _forward

    embedding_output = MagicMock()
    embedding_output.inputs_embeds = mx.zeros((1, prompt_len, hidden_width))
    embedding_output.to_dict.return_value = {}
    model.get_input_embeddings.return_value = embedding_output

    with (
        patch.object(generate_module.cache, "make_prompt_cache", return_value=[]),
        patch.object(generate_module, "make_logits_processors", return_value=[]),
        patch.object(
            generate_module, "make_sampler", return_value=lambda _: mx.array([0])
        ),
        patch.object(ar_module.mx, "clear_cache", MagicMock()),
    ):
        list(
            generate_module.generate_step(
                input_ids=mx.array([list(range(prompt_len))], dtype=mx.int32),
                model=model,
                pixel_values=None,
                mask=None,
                max_tokens=1,
                prefill_step_size=step,
            )
        )
    return widths


class TestGenerateStepLoop:
    def test_disabled_leaves_the_short_tail_as_its_own_chunk(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "0")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "0")
        widths = _chunk_widths(20, 8, monkeypatch)
        # 19 columns chunked (the loop always holds the last token back), then
        # the 1-token _step, then one decode step (max_tokens=1).
        assert [w for w, _ in widths] == [8, 8, 3, 1, 1]
        assert all(keep is None for _, keep in widths)

    def test_merge_folds_the_tail_into_the_previous_chunk(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "1")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "4")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "0")
        widths = _chunk_widths(20, 8, monkeypatch)
        assert [w for w, _ in widths] == [8, 11, 1, 1]
        assert sum(w for w, _ in widths[:-2]) == 19

    def test_the_shipped_default_merges_and_keeps(self, monkeypatch):
        # Nothing exported: (b) folds the 3-token tail, and (a) is inert here
        # because after the loop every forward this path makes is one column.
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", raising=False)
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", raising=False)
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", "4")
        widths = _chunk_widths(20, 8, monkeypatch)
        assert widths == [(8, None), (11, None), (1, None), (1, None)]

    def test_unchunked_prefill_asks_for_one_row_when_enabled(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "1")
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", raising=False)
        # prompt (6) <= step (8): ``should_chunk`` is false, so the WHOLE prompt
        # goes through the single wide forward -- the one that used to project
        # 6 x vocab to read one row.
        widths = _chunk_widths(6, 8, monkeypatch)
        assert widths == [(6, 1), (1, None)]

    def test_unchunked_prefill_asks_for_one_row_by_default(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", raising=False)
        widths = _chunk_widths(6, 8, monkeypatch)
        assert widths == [(6, 1), (1, None)]

    def test_unchunked_prefill_is_untouched_when_disabled(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "0")
        widths = _chunk_widths(6, 8, monkeypatch)
        assert widths == [(6, None), (1, None)]

    def test_decode_steps_never_carry_the_kwarg(self, monkeypatch):
        # Chunked prefill: every forward after the loop is one column wide, so
        # (a) is withheld on all of them whether or not (b) merged the tail.
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "1")
        for merge in ("0", "1"):
            monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", merge)
            widths = _chunk_widths(20, 8, monkeypatch)
            assert [keep for _, keep in widths] == [None] * len(widths)


# ------------------------------------------------ the model side of L35(a)


def _tiny_glm5_next():
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import LanguageModel

    mx.random.seed(0)
    cfg = glm5_next.TextConfig(
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
        num_nextn_predict_layers=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return LanguageModel(cfg)


class TestGlm5NextKeepsLogits:
    PROMPT = mx.array([[3, 5, 7, 9, 11, 13, 15, 17]])

    def test_advertises_the_capability(self):
        assert _tiny_glm5_next().supports_num_logits_to_keep is True

    def test_keep_one_returns_the_last_row(self):
        lm = _tiny_glm5_next()
        full = lm(self.PROMPT, cache=lm.make_cache()).logits
        kept = lm(self.PROMPT, cache=lm.make_cache(), num_logits_to_keep=1).logits
        assert kept.shape == (1, 1, 128)
        # CLOSE, NOT EQUAL.  Slicing the hidden before the projection changes the
        # GEMM's M (8 -> 1), and that moves the last ulp: measured 4.77e-07 here,
        # on a float32 CPU matmul at hidden_size 128.  This is the whole reason
        # L35(a) is opt-in rather than an unconditional "free" win -- on a
        # near-tie it can pick a different token (I1098 kept the same kwarg out
        # of a correctness fix for the same reason).
        assert float(mx.max(mx.abs(full[:, -1:] - kept))) < 1e-5

    def test_the_generic_kwarg_name_is_accepted_too(self):
        lm = _tiny_glm5_next()
        a = lm(self.PROMPT, cache=lm.make_cache(), num_logits_to_keep=1).logits
        b = lm(self.PROMPT, cache=lm.make_cache(), logits_to_keep=1).logits
        assert mx.array_equal(a, b).item()

    def test_no_kwarg_is_the_full_prompt(self):
        lm = _tiny_glm5_next()
        assert lm(self.PROMPT, cache=lm.make_cache()).logits.shape == (1, 8, 128)


class TestChunkPlanIsNotBitExact:
    """The contract L35(b) can actually keep, stated as a test.

    The brief this lever was written from assumed the chunk-carry prefill is
    bit-identical across chunk sizes, so a re-planned tail would be free of
    consequence.  It is not, on this model:

    * L7B (receipts/sweep11/L7B_PREFILL_CHUNK_20260905) recorded "identity
      across chunk sizes: **False**" at both 8k and 32k;
    * L7B3 put a chunked prefill's first divergence from an UNCHUNKED reference
      at token 35 (chunk 2048) / 45 (chunk 8192), mean KL 0.025 nats;
    * I1098 measured 1.37e-06 between a B=3 chunked cache and an unchunked one;
    * and the fixture below reproduces the mechanism in-process.

    (The L23e receipts' "identity across chunk sizes: True" line does not
    contradict this: those arms swept a single chunk size, so the check had
    nothing to compare against.)

    So the assertion is a TOLERANCE, not an equality.  The lever nevertheless
    ships ON (I1437) because the rule-13 rail -- natural gen1024 panel, 4/4
    identical completion text sha, identical speculative acceptance -- is the
    gate that a chaos-limited decomposition can actually be judged by; the
    numbers below are why "bit-identical" is NOT claimed anywhere.
    """

    def _cache_state(self, lm, prompt, plan):
        c = lm.make_cache()
        pos = 0
        for n in plan:
            lm(prompt[:, pos : pos + n], cache=c)
            pos += n
        mx.eval([x.state for x in c])
        flat = []
        for x in c:
            state = x.state
            state = state if isinstance(state, (list, tuple)) else [state]
            for t in state:
                if t is None:
                    continue
                for u in t if isinstance(t, (list, tuple)) else [t]:
                    if u is not None and u.size:
                        flat.append(u)
        return flat

    def test_two_plans_agree_to_a_tolerance_not_to_the_bit(self):
        lm = _tiny_glm5_next()
        n = 24
        prompt = mx.array([[(i * 7 + 3) % 127 for i in range(n)]])
        # [16, 8] is the grid plan; [17, 7] is what a tail merge produces.
        a = self._cache_state(lm, prompt, [16, 8])
        b = self._cache_state(lm, prompt, [17, 7])
        assert len(a) == len(b) and a
        worst = max(
            float(mx.max(mx.abs(u.astype(mx.float32) - v.astype(mx.float32))))
            for u, v in zip(a, b)
        )
        assert worst < 1e-4, worst


# ------------------------------------------------ the "0 0" revert contract


class TestBaseParityWhenBothAreOff:
    """``LOGITS_KEEP=0 TAIL_MERGE=0`` == a6634a75, on every path this file sees.

    This is the revert knob named in the promotion decision (I1437) and in
    generate/common.py's L35 header, so it is asserted rather than asserted-by-
    comment: the chunk plan is ``min(step, remaining)`` again, the keep-kwarg is
    absent again, and generate_step's forward widths and argument lists are the
    ones the old expressions produced.
    """

    @pytest.fixture(autouse=True)
    def _both_off(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_LOGITS_KEEP", "0")
        monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", "0")
        # Left at their defaults on purpose: with the merge off they must not be
        # read at all.
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", raising=False)
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MODE", raising=False)

    @pytest.mark.parametrize(
        "remaining", [1, 2, 537, 8191, 8192, 8193, 8729, 16384, 33305, 131071]
    )
    @pytest.mark.parametrize("step", [8192, 2048])
    def test_the_chunk_plan_is_the_old_expression(self, remaining, step):
        expected = []
        left = remaining
        while left > 0:
            n = min(step, left)
            expected.append(n)
            left -= n
        assert plan_prefill_chunks(remaining, step) == expected

    @pytest.mark.parametrize("width", [1, 2, 512, 8192])
    def test_the_keep_kwarg_is_absent_at_every_width(self, width):
        keeper = SimpleNamespace(supports_num_logits_to_keep=True)
        assert prefill_logits_keep_kwargs(keeper, width) == {}

    def test_generate_step_widths_and_kwargs_are_the_base_ones(self, monkeypatch):
        # chunked: 19 columns at step 8, then the 1-token _step, then decode
        assert _chunk_widths(20, 8, monkeypatch) == [
            (8, None),
            (8, None),
            (3, None),
            (1, None),
            (1, None),
        ]
        # unchunked: the whole prompt through one wide forward, no keep kwarg
        assert _chunk_widths(6, 8, monkeypatch) == [(6, None), (1, None)]


# ------------------------------------------------------- the TP=2 forward cap
#
# tp/worker.py::encode RAISES TPUnavailable when a forward's b*s exceeds
# MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD minus the ECHO_WORDS reserved for the
# shape agreement.  A GROWN chunk is the only width this module can produce that
# is wider than ``step``, so it is the only way (b) could turn a TP deployment
# that worked into a raise.  It falls back instead.


class TestTpForwardCap:
    ROOM = 8192 - 3  # _max_tok() - ECHO_WORDS at the shipped defaults

    def test_room_is_none_off_tp(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_GLM5_TP_HOSTS", raising=False)
        assert tp_forward_token_room() is None

    def test_room_is_none_for_a_single_host(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_GLM5_TP_HOSTS", "box0")
        assert tp_forward_token_room() is None

    def test_room_is_the_cap_minus_the_echo_words(self, monkeypatch):
        from mlx_vlm.tp import worker

        monkeypatch.setenv("MLX_VLM_GLM5_TP_HOSTS", "box0,box1")
        monkeypatch.setenv("MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD", "16384")
        assert tp_forward_token_room() == 16384 - worker.ECHO_WORDS

    def test_the_room_matches_what_encode_actually_accepts(self, monkeypatch):
        """The number this module uses is the number the codec enforces."""
        from mlx_vlm.tp import worker

        monkeypatch.setenv("MLX_VLM_GLM5_TP_HOSTS", "box0,box1")
        monkeypatch.setenv("MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD", "128")
        room = tp_forward_token_room()
        worker.encode(worker.OP_FORWARD, 1, (1, room), list(range(room)))
        with pytest.raises(worker.TPUnavailable):
            worker.encode(worker.OP_FORWARD, 1, (1, room + 1), list(range(room + 1)))

    def test_a_grown_chunk_that_fits_is_left_alone(self):
        # 8189 room, step 4096: a grown 4096+537 chunk fits, so grow wins.
        assert (
            next_prefill_chunk(4096 + 537, 4096, tail_min=1024, enabled=True,
                               tp_room=self.ROOM)
            == 4096 + 537
        )

    def test_a_grown_chunk_over_the_cap_falls_back_to_balance(self):
        # The shipped step is 8192 and the shipped cap leaves room for 8189, so
        # ANY grown chunk overflows in TP=2.  It must re-plan, not raise.
        n = next_prefill_chunk(
            STEP + 537, STEP, tail_min=1024, enabled=True, tp_room=self.ROOM
        )
        assert n == (STEP + 537 + 1) // 2 == 4365
        assert n <= self.ROOM

    def test_the_fallback_is_a_complete_plan_that_sums(self):
        plan = plan_prefill_chunks(
            4 * STEP + 537, STEP, tail_min=1024, enabled=True, tp_room=self.ROOM
        )
        assert sum(plan) == 4 * STEP + 537
        assert plan == [STEP, STEP, STEP, 4365, 4364]
        # every chunk before the last cell is untouched, so the vault ladder's
        # multiples-of-step rungs still exist
        assert plan[:3] == [STEP] * 3

    def test_the_batch_dimension_is_what_the_cap_counts(self):
        # b*s, not s: a 2-row batch halves the room a chunk may occupy.
        assert (
            next_prefill_chunk(2048 + 100, 2048, tail_min=1024, enabled=True,
                               batch=1, tp_room=4096)
            == 2148
        )
        assert (
            next_prefill_chunk(2048 + 100, 2048, tail_min=1024, enabled=True,
                               batch=2, tp_room=4096)
            == 1074
        )

    def test_it_never_returns_wider_than_the_unmerged_plan(self):
        # A cap too small for ``step`` itself is a pre-existing misconfiguration;
        # the merge must not make it worse, so the worst case is the base width.
        for tail in (1, 100, 537, 1023):
            n = next_prefill_chunk(
                STEP + tail, STEP, tail_min=1024, enabled=True, tp_room=16
            )
            assert n <= STEP

    def test_balance_mode_is_capped_too(self):
        n = next_prefill_chunk(
            STEP + 537, STEP, tail_min=1024, mode="balance", enabled=True, tp_room=16
        )
        assert n <= STEP

    def test_the_default_on_plan_never_widens_a_tp_forward(self, monkeypatch):
        """End to end, through the ENV: no plan is wider than ``step`` in TP.

        ``step`` itself, not the merge, is what the cap has to be sized for --
        and at the shipped defaults it is NOT: DEFAULT_PREFILL_STEP_SIZE is 8192
        while the default cap leaves room for 8189, which is why
        tp/README_TP_SERVING.md tells a TP operator to raise
        MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD or lower PREFILL_STEP_SIZE.  That
        is a pre-existing configuration fact.  What this asserts is that the
        default-ON merge does not make it worse: with the env exactly as shipped
        the widest chunk is still ``step``.
        """
        monkeypatch.setenv("MLX_VLM_GLM5_TP_HOSTS", "box0,box1")
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MERGE", raising=False)
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", raising=False)
        monkeypatch.delenv("MLX_VLM_GLM5_PREFILL_TAIL_MODE", raising=False)
        monkeypatch.delenv("MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD", raising=False)
        for tail in range(1, 1024, 37):
            plan = plan_prefill_chunks(2 * STEP + tail, STEP, batch=1)
            assert sum(plan) == 2 * STEP + tail
            assert max(plan) <= STEP, (tail, plan)
        # and with a cap that IS big enough for the step, nothing overflows
        monkeypatch.setenv("MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD", "16384")
        room = tp_forward_token_room()
        for tail in range(1, 1024, 37):
            plan = plan_prefill_chunks(2 * STEP + tail, STEP, batch=1)
            assert max(plan) <= room, (tail, plan)
