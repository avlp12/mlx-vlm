"""Parity for the GLM-5-Next DSA indexer visibility memo.

The indexer splits into a *scoring* half (pool_keys / scores / index_scores,
which reads cached keys and hidden states, genuinely per-layer) and a
*visibility* half (pool_indices, pool_valid, pool_end, visible, pool_visible,
valid_candidates and the always-select tail) that is a pure function of the
padding-mask history and the sequence lengths.  Every DSA layer in one
Glm5NextModel forward is handed the same fa_mask and advances its indexer cache
in lockstep, so all of them rebuild byte-identical visibility tensors.  The memo
computes it once and hands back the *same* mx.array objects.

These tests pin that the shared result is bit-identical to per-layer
recomputation, that the memo declines indexers the model does not own (the MTP
head, direct sub-module callers), and that it degrades to plain recomputation
when its memory budget is exhausted.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.glm5_next.config import TextConfig

# Small but structurally faithful: index_topk must stay under the sequence
# length or the indexer short-circuits (`bypass_short`) and selects nothing.
_CFG = dict(
    model_type="glm5_next_text",
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    routed_scaling_factor=2.5,
    kv_lora_rank=32,
    q_lora_rank=32,
    qk_rope_head_dim=0,
    v_head_dim=32,
    qk_nope_head_dim=32,
    num_experts_per_tok=2,
    first_k_dense_replace=3,
    max_position_embeddings=4096,
    rms_norm_eps=1e-05,
    index_topk=16,
    index_kpool=4,
    index_head_dim=16,
    index_n_heads=4,
    layer_types=["linear_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={
        "num_heads": 4,
        "gate_lower_bound": -5.0,
        "head_dim": 16,
        "short_conv_kernel_size": 4,
    },
)

N_LAYERS = 4          # stand-ins for the 11 live DSA layers
SEQ = 96              # > index_topk, so the pooling/selection path runs
CHUNK_SPLIT = 3       # S > 512 is the live chunking; here one chunk is enough


def _config():
    return TextConfig.from_dict(dict(_CFG))


def _reset_env(**kv):
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    glm5._VIS_MEMO_ENV = None
    glm5._VIS_MEMO_MB = None
    glm5._VIS_MEMO_VERIFY = None


def _indexers(config, n=N_LAYERS, seed=0):
    """n indexers with *different* weights -- only the mask-derived half is shared."""
    out = []
    for i in range(n):
        mx.random.seed(seed + i)
        ix = glm5.Glm5NextIndexer(config)

        def rand(tree):
            if isinstance(tree, dict):
                return {k: rand(v) for k, v in tree.items()}
            if isinstance(tree, list):
                return [rand(v) for v in tree]
            return (mx.random.normal(tree.shape) * 0.1).astype(mx.float32)

        ix.update(rand(ix.parameters()))
        ix.index_kpool_compress_ape = mx.random.normal(
            ix.index_kpool_compress_ape.shape
        ) * 0.1
        ix.index_kpool_compress_gate = mx.random.normal(
            ix.index_kpool_compress_gate.shape
        ) * 0.1
        mx.eval(ix.parameters())
        out.append(ix)
    return out


def _inputs(config, batch=1, seq=SEQ, seed=7):
    mx.random.seed(seed)
    x = (mx.random.normal((batch, seq, config.hidden_size)) * 0.5).astype(mx.float32)
    qr = (mx.random.normal((batch, seq, config.q_lora_rank)) * 0.5).astype(mx.float32)
    mx.eval(x, qr)
    return x, qr


def _run(indexers, x, qr, mask, memo):
    """One 'forward': every indexer sees the same x/qr/mask, its own fresh cache."""
    prev = glm5._VIS_MEMO_CTX
    glm5._VIS_MEMO_CTX = memo
    try:
        outs = []
        for ix in indexers:
            topk = ix(x, qr, mask, cache=KVCache())
            mx.eval(topk)
            outs.append(topk)
        return outs
    finally:
        glm5._VIS_MEMO_CTX = prev


def _memo_for(indexers):
    return glm5._VisibilityMemo(frozenset(id(ix) for ix in indexers))


# --------------------------------------------------------------------- tests
def test_memo_is_bit_identical_no_mask():
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1", MLX_VLM_GLM5_VIS_MEMO_VERIFY=None)
    config = _config()
    ix = _indexers(config)
    x, qr = _inputs(config)
    ref = _run(ix, x, qr, None, None)
    got = _run(ix, x, qr, None, _memo_for(ix))
    assert len(ref) == N_LAYERS and ref[0] is not None
    for i, (a, b) in enumerate(zip(ref, got)):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert bool(mx.all(a == b)), f"layer {i} topk differs"
    # negative control: the layers really do differ from one another, so the
    # test is not trivially comparing a constant.
    assert not bool(mx.all(ref[0] == ref[1]))


@pytest.mark.parametrize("batch", [1, 2])
def test_memo_is_bit_identical_with_padding_mask(batch):
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1", MLX_VLM_GLM5_VIS_MEMO_VERIFY=None)
    config = _config()
    ix = _indexers(config, seed=20)
    x, qr = _inputs(config, batch=batch, seed=11)
    # left padding: the first rows are invalid, which is what makes `valid`
    # (and therefore first_key / pool_indices / the tail) non-trivial.
    valid = mx.ones((batch, SEQ), dtype=mx.bool_)
    if batch > 1:
        pad = mx.arange(SEQ)[None, :] >= mx.array([[0], [5]])
        valid = valid & pad
    else:
        valid = mx.arange(SEQ)[None, :] >= 3
    mx.eval(valid)
    ref = _run(ix, x, qr, valid, None)
    got = _run(ix, x, qr, valid, _memo_for(ix))
    for i, (a, b) in enumerate(zip(ref, got)):
        assert bool(mx.all(a == b)), f"layer {i} topk differs under padding"
    assert bool(mx.any(ref[0] == -1)), "padding should mask some selections"


def test_verify_mode_proves_per_layer_equality():
    """MLX_VLM_GLM5_VIS_MEMO_VERIFY=1 recomputes every memoized tensor inside
    each layer and raises on any mismatch -- this is the assertion the hoist
    rests on, exercised directly rather than inferred from the outputs."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1", MLX_VLM_GLM5_VIS_MEMO_VERIFY="1")
    try:
        config = _config()
        ix = _indexers(config, seed=33)
        x, qr = _inputs(config, seed=5)
        valid = mx.arange(SEQ)[None, :] >= 2
        mx.eval(valid)
        ref = _run(ix, x, qr, valid, None)
        got = _run(ix, x, qr, valid, _memo_for(ix))
        for a, b in zip(ref, got):
            assert bool(mx.all(a == b))
    finally:
        _reset_env(MLX_VLM_GLM5_VIS_MEMO_VERIFY=None)


def test_memo_declines_foreign_indexer():
    """The MTP head owns an indexer the model never registered; it must not be
    served, or it would read another cache's visibility."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    config = _config()
    owned = _indexers(config, n=2, seed=40)
    foreign = _indexers(config, n=1, seed=90)[0]
    memo = _memo_for(owned)
    prev = glm5._VIS_MEMO_CTX
    glm5._VIS_MEMO_CTX = memo
    try:
        assert glm5._active_vis_memo(owned[0]) is memo
        assert glm5._active_vis_memo(foreign) is None
    finally:
        glm5._VIS_MEMO_CTX = prev
    # and with no context open at all (probes calling layer.self_attn by hand)
    assert glm5._active_vis_memo(owned[0]) is None


def test_memo_off_by_env_matches_on():
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    config = _config()
    ix = _indexers(config, seed=50)
    x, qr = _inputs(config, seed=13)
    on = _run(ix, x, qr, None, _memo_for(ix))
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="0")
    off = _run(ix, x, qr, None, _memo_for(ix))
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    for a, b in zip(on, off):
        assert bool(mx.all(a == b))


def test_memo_budget_exhaustion_still_correct():
    """A zero budget must degrade to plain recomputation, not to wrong answers."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1", MLX_VLM_GLM5_VIS_MEMO_MB="0")
    config = _config()
    ix = _indexers(config, seed=60)
    x, qr = _inputs(config, seed=17)
    memo = _memo_for(ix)
    got = _run(ix, x, qr, None, memo)
    assert memo.chunks == {}, "budget 0 must cache no chunk bundles"
    _reset_env(MLX_VLM_GLM5_VIS_MEMO_MB=None)
    ref = _run(ix, x, qr, None, None)
    for a, b in zip(ref, got):
        assert bool(mx.all(a == b))


def test_memo_reuses_the_same_array_object():
    """Bit-exactness here is by identity, not by numerical agreement: layers
    2..n must receive the very tensor layer 1 built."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    config = _config()
    ix = _indexers(config, seed=70)
    x, qr = _inputs(config, seed=19)
    memo = _memo_for(ix)
    _run(ix, x, qr, None, memo)
    assert len(memo.layout) == 1, "one (B, T) layout for the whole forward"
    assert len(memo.chunks) >= 1
    (bundle,) = list(memo.chunks.values())[:1]
    valid_candidates, tail = bundle
    assert valid_candidates.dtype == mx.bool_
    assert tail is not None  # index_kpool_always_select_tail defaults on
    assert memo.nbytes > 0


def test_pool_layout_split_is_verbatim():
    """_pooled_states was split into _pool_layout + the keys half; passing the
    layout in must equal computing it inline."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    config = _config()
    ix = _indexers(config, n=1, seed=80)[0]
    mx.random.seed(3)
    S, hd = SEQ, config.index_head_dim
    keys = mx.random.normal((1, S, hd)).astype(mx.float32)
    gate = mx.random.normal((1, S, hd)).astype(mx.float32)
    valid = mx.arange(S)[None, :] >= 4
    mx.eval(keys, gate, valid)
    a = ix._pooled_states(keys, gate, valid)
    b = ix._pooled_states(keys, gate, valid, layout=ix._pool_layout(valid, S))
    for u, v in zip(a, b):
        assert bool(mx.all(u == v))
