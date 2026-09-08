"""M1/M2 -- prefill-only top-k levers, and the definition of "prefill" they share.

M1 ``MLX_VLM_GLM5_PREFILL_MOE_TOPK``  : MoE router top-8 -> 7/6, prefill forwards only.
M2 ``MLX_VLM_GLM5_PREFILL_DSA_TOPK``  : DSA indexer top-2048 -> 1536, prefill only.

Neither model carries a prefill flag, and "S > 1" is not prefill -- the speculative
verify block is S = 8, the shape I1310 showed is quality-sensitive.  Both levers
therefore share ONE row floor (``MLX_VLM_GLM5_PREFILL_TOPK_MIN_ROWS``, default 512),
so a run can never have one of them think it is in prefill and the other not.

What is asserted: the default path is byte-for-byte unchanged at every width; the arm
fires only at or above the floor; the arm really cuts the k (non-vacuity); the kept
scores are renormalised over the kept k exactly as the model does at k = 8; and a
bad value raises instead of silently doing nothing.
"""
import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_vlm.models.deepseek_v32.language as dsv32
import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.glm5_next.config import TextConfig

MOE_KEY = "MLX_VLM_GLM5_PREFILL_MOE_TOPK"
DSA_KEY = "MLX_VLM_GLM5_PREFILL_DSA_TOPK"
FLOOR_KEY = "MLX_VLM_GLM5_PREFILL_TOPK_MIN_ROWS"


@pytest.fixture(autouse=True)
def _clean_env():
    keys = (MOE_KEY, DSA_KEY, FLOOR_KEY)
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ------------------------------------------------------------------- the floor
class TestPrefillFloor:
    def test_default_floor_is_512(self):
        assert dsv32.prefill_row_floor() == 512
        assert dsv32.is_prefill_rows(512) is True
        assert dsv32.is_prefill_rows(511) is False

    def test_decode_and_verify_widths_are_never_prefill(self):
        for rows in (1, 2, 4, 8, 16, 32, 64):
            assert dsv32.is_prefill_rows(rows) is False, rows

    def test_shipped_prefill_widths_are_prefill(self):
        for rows in (1024, 2048, 8192):
            assert dsv32.is_prefill_rows(rows) is True, rows

    @pytest.mark.parametrize("raw,want", [
        (None, 512), ("", 512), ("garbage", 512), ("0", 512), ("-1", 512),
        ("1024", 1024), (" 64 ", 64),
    ])
    def test_floor_parsing(self, raw, want):
        if raw is None:
            os.environ.pop(FLOOR_KEY, None)
        else:
            os.environ[FLOOR_KEY] = raw
        assert dsv32.prefill_row_floor() == want


# ---------------------------------------------------------------------- M1
class TestMoeTopKSelection:
    def test_unset_returns_the_configured_k_at_every_width(self):
        for rows in (1, 8, 512, 8192):
            assert dsv32.prefill_moe_top_k(8, rows) == 8

    def test_arm_applies_only_at_or_above_the_floor(self):
        os.environ[MOE_KEY] = "7"
        for rows in (1, 8, 64, 511):
            assert dsv32.prefill_moe_top_k(8, rows) == 8, rows
        for rows in (512, 8192):
            assert dsv32.prefill_moe_top_k(8, rows) == 7, rows

    def test_six_is_accepted(self):
        os.environ[MOE_KEY] = "6"
        assert dsv32.prefill_moe_top_k(8, 8192) == 6

    @pytest.mark.parametrize("bad", ["0", "9", "-1"])
    def test_out_of_range_raises_rather_than_silently_doing_nothing(self, bad):
        os.environ[MOE_KEY] = bad
        with pytest.raises(ValueError):
            dsv32.prefill_moe_top_k(8, 8192)

    def test_garbage_raises(self):
        os.environ[MOE_KEY] = "seven"
        with pytest.raises(ValueError):
            dsv32.prefill_moe_top_k(8, 8192)

    def test_below_the_floor_a_bad_value_is_never_even_read(self):
        """Decode must not be able to crash on an env meant for prefill."""
        os.environ[MOE_KEY] = "nonsense"
        assert dsv32.prefill_moe_top_k(8, 1) == 8


def _gate_config(top_k=8, n_routed=16, n_group=1):
    return TextConfig.from_dict(dict(
        model_type="glm5_next_text", vocab_size=256, hidden_size=64,
        intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1,
        n_routed_experts=n_routed, routed_scaling_factor=2.5, kv_lora_rank=32,
        q_lora_rank=32, qk_rope_head_dim=0, v_head_dim=16, qk_nope_head_dim=16,
        num_experts_per_tok=top_k, first_k_dense_replace=0, n_group=n_group,
        topk_group=n_group, max_position_embeddings=4096, rms_norm_eps=1e-05,
        index_topk=16, index_head_dim=16, index_n_heads=2,
        layer_types=["full_attention"], mlp_layer_types=["sparse"],
        linear_attn_config={"num_heads": 2, "gate_lower_bound": -5.0,
                            "head_dim": 32, "short_conv_kernel_size": 4},
    ))


class TestMoeGate:
    def _gate(self, seed=0):
        mx.random.seed(seed)
        cfg = _gate_config()
        gate = glm5.Glm5NextMoEGate(cfg)
        gate.weight = mx.random.normal((cfg.n_routed_experts, cfg.hidden_size)) * 0.5
        gate.e_score_correction_bias = mx.random.normal((cfg.n_routed_experts,)) * 0.1
        mx.eval(gate.parameters())
        return gate, cfg

    def test_default_is_bit_identical_at_prefill_width(self):
        gate, cfg = self._gate()
        x = (mx.random.normal((1, 600, cfg.hidden_size)) * 0.3).astype(mx.bfloat16)
        i0, s0 = gate(x)
        os.environ[MOE_KEY] = "7"
        os.environ[FLOOR_KEY] = "100000"      # floor above this width: arm inert
        i1, s1 = gate(x)
        mx.eval(i0, s0, i1, s1)
        assert mx.array_equal(i0, i1).item() and mx.array_equal(s0, s1).item()

    def test_decode_width_is_untouched_while_the_arm_is_set(self):
        gate, cfg = self._gate()
        x = (mx.random.normal((1, 8, cfg.hidden_size)) * 0.3).astype(mx.bfloat16)
        i0, s0 = gate(x)
        os.environ[MOE_KEY] = "6"
        i1, s1 = gate(x)
        mx.eval(i0, s0, i1, s1)
        assert i0.shape[-1] == 8
        assert mx.array_equal(i0, i1).item() and mx.array_equal(s0, s1).item()

    @pytest.mark.parametrize("k", [7, 6])
    def test_prefill_width_selects_k_experts_and_renormalises_over_them(self, k):
        gate, cfg = self._gate()
        x = (mx.random.normal((1, 600, cfg.hidden_size)) * 0.3).astype(mx.bfloat16)
        i8, s8 = gate(x)
        os.environ[MOE_KEY] = str(k)
        ik, sk = gate(x)
        mx.eval(i8, s8, ik, sk)
        assert i8.shape[-1] == 8 and ik.shape[-1] == k
        # norm_topk_prob + routed_scaling_factor: the kept scores sum to the scaling
        # factor whatever k is -- i.e. renormalised over the kept k exactly as at k=8.
        for arr in (s8, sk):
            tot = np.array(arr.sum(axis=-1).astype(mx.float32), copy=False)
            assert np.allclose(tot, cfg.routed_scaling_factor, atol=2e-3), tot[:5]
        # Non-vacuity: the kept set is a SUBSET of the k=8 set (same ranking, fewer
        # kept), so the arm really drops the lowest-scoring experts.
        top8 = set(np.array(i8, copy=False).reshape(-1, 8)[0].tolist())
        topk = set(np.array(ik, copy=False).reshape(-1, k)[0].tolist())
        assert topk < top8

    def test_second_k_compiles_its_own_trace_and_leaves_k8_alone(self):
        """mx.compile hashes the python int into the trace key, so k=7 cannot
        poison the k=8 trace the decode path uses."""
        gate, cfg = self._gate()
        wide = (mx.random.normal((1, 600, cfg.hidden_size)) * 0.3).astype(mx.bfloat16)
        narrow = (mx.random.normal((1, 4, cfg.hidden_size)) * 0.3).astype(mx.bfloat16)
        i_dec_before, s_dec_before = gate(narrow)
        mx.eval(i_dec_before, s_dec_before)
        os.environ[MOE_KEY] = "6"
        i_pre, _ = gate(wide)          # compiles the k=6 trace
        mx.eval(i_pre)
        i_dec_after, s_dec_after = gate(narrow)
        mx.eval(i_dec_after, s_dec_after)
        assert i_pre.shape[-1] == 6
        assert mx.array_equal(i_dec_before, i_dec_after).item()
        assert mx.array_equal(s_dec_before, s_dec_after).item()


# ---------------------------------------------------------------------- M2
def _indexer_config(index_topk=64):
    return TextConfig.from_dict(dict(
        model_type="glm5_next_text", vocab_size=256, hidden_size=64,
        intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1,
        n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=32,
        q_lora_rank=32, qk_rope_head_dim=0, v_head_dim=16, qk_nope_head_dim=16,
        num_experts_per_tok=2, first_k_dense_replace=0,
        max_position_embeddings=4096, rms_norm_eps=1e-05, index_topk=index_topk,
        index_head_dim=16, index_n_heads=2, layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
        linear_attn_config={"num_heads": 2, "gate_lower_bound": -5.0,
                            "head_dim": 32, "short_conv_kernel_size": 4},
    ))


class TestDsaTopKSelection:
    def test_unset_returns_the_configured_topk(self):
        for rows in (1, 8, 512, 8192):
            assert glm5._prefill_dsa_topk(2048, rows) == 2048

    def test_arm_applies_only_at_or_above_the_floor(self):
        os.environ[DSA_KEY] = "1536"
        for rows in (1, 8, 511):
            assert glm5._prefill_dsa_topk(2048, rows) == 2048, rows
        for rows in (512, 8192):
            assert glm5._prefill_dsa_topk(2048, rows) == 1536, rows

    @pytest.mark.parametrize("bad", ["0", "-8", "4096", "sixteen"])
    def test_bad_values_raise(self, bad):
        os.environ[DSA_KEY] = bad
        with pytest.raises(ValueError):
            glm5._prefill_dsa_topk(2048, 8192)

    def test_below_the_floor_a_bad_value_is_never_read(self):
        os.environ[DSA_KEY] = "nonsense"
        assert glm5._prefill_dsa_topk(2048, 1) == 2048

    def test_the_floor_is_shared_with_m1(self):
        os.environ[FLOOR_KEY] = "1024"
        os.environ[DSA_KEY] = "1536"
        os.environ[MOE_KEY] = "7"
        assert glm5._prefill_dsa_topk(2048, 600) == 2048
        assert dsv32.prefill_moe_top_k(8, 600) == 8
        assert glm5._prefill_dsa_topk(2048, 1024) == 1536
        assert dsv32.prefill_moe_top_k(8, 1024) == 7


class TestIndexerForward:
    def _indexer(self, index_topk=64, seed=0):
        mx.random.seed(seed)
        cfg = _indexer_config(index_topk)
        idx = glm5.Glm5NextIndexer(cfg)

        def rand(tree):
            if isinstance(tree, dict):
                return {k: rand(v) for k, v in tree.items()}
            if isinstance(tree, list):
                return [rand(v) for v in tree]
            return mx.random.normal(tree.shape) * 0.1

        idx.update(rand(idx.parameters()))
        mx.eval(idx.parameters())
        return idx, cfg

    def _run(self, idx, cfg, S):
        mx.random.seed(5)
        x = mx.random.normal((1, S, cfg.hidden_size)) * 0.3
        qr = mx.random.normal((1, S, cfg.q_lora_rank)) * 0.3
        out = idx(x, qr, None, cache=None)
        mx.eval(out)
        return out

    def test_default_width_at_a_prefill_length(self):
        idx, cfg = self._indexer()
        out = self._run(idx, cfg, 600)
        # index_topk + (kpool - 1) tail
        assert out.shape[-1] == cfg.index_topk + cfg.index_kpool - 1

    def test_arm_narrows_only_the_prefill_path(self):
        idx, cfg = self._indexer()
        ref = self._run(idx, cfg, 600)
        os.environ[DSA_KEY] = "32"
        arm = self._run(idx, cfg, 600)
        assert ref.shape[-1] == 67 and arm.shape[-1] == 35
        # the kept selections are a subset of the reference's, per query row
        r = set(np.array(ref, copy=False).reshape(-1).tolist()) - {-1}
        a = set(np.array(arm, copy=False).reshape(-1).tolist()) - {-1}
        assert a <= r

    def test_arm_is_inert_below_the_floor(self):
        idx, cfg = self._indexer()
        ref = self._run(idx, cfg, 200)
        os.environ[DSA_KEY] = "32"
        arm = self._run(idx, cfg, 200)
        assert mx.array_equal(ref, arm).item()

    def test_default_is_bit_identical_with_the_floor_raised(self):
        idx, cfg = self._indexer()
        ref = self._run(idx, cfg, 600)
        os.environ[DSA_KEY] = "32"
        os.environ[FLOOR_KEY] = "100000"
        arm = self._run(idx, cfg, 600)
        assert mx.array_equal(ref, arm).item()

    def test_decode_after_a_narrowed_prefill_keeps_the_configured_topk(self):
        """The three decode-fast consumers read self.index_topk unconditionally.

        Run a real prefill into a cache with the arm set, then one decode step: the
        decode step must return the CONFIGURED width, not the narrowed one.
        """
        from mlx_vlm.models.cache import KVCache

        idx, cfg = self._indexer()
        os.environ[DSA_KEY] = "32"
        cache = KVCache()
        mx.random.seed(5)
        x = mx.random.normal((1, 600, cfg.hidden_size)) * 0.3
        qr = mx.random.normal((1, 600, cfg.q_lora_rank)) * 0.3
        pre = idx(x, qr, None, cache=cache)
        mx.eval(pre)
        assert pre.shape[-1] == 35, "prefill did not take the narrowed path"

        x1 = mx.random.normal((1, 1, cfg.hidden_size)) * 0.3
        qr1 = mx.random.normal((1, 1, cfg.q_lora_rank)) * 0.3
        dec = idx(x1, qr1, None, cache=cache)
        mx.eval(dec)
        assert dec.shape[-1] == cfg.index_topk + cfg.index_kpool - 1, (
            f"decode width {dec.shape[-1]} was narrowed by a prefill-only lever"
        )
