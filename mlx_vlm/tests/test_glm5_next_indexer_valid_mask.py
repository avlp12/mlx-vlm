"""The GLM-5-Next DSA indexer must know which columns are PADDING.

``Glm5NextIndexer.__call__`` packs a per-token validity flag into channel
``2 * head_dim`` of its own KV cache.  That flag is the only thing that tells
the pool layout where a row's real tokens begin (``first_key = argmax(valid)``),
the only thing intersected into ``visible``, and the only thing standing between
a padding column and the top-k.

It used to be taken from ``mask`` *only* when ``mask.shape == (B, S)``.  A DSA
layer is handed ``fa_mask`` -- the 4-D causal+left-pad array that
``BatchKVCache.make_mask`` builds, ``[B, 1, S, T]`` -- so that test was False for
every batched forward and ``valid_cur`` fell back to all-ones.  A LEFT-PADDED
prompt, which is how ``BatchKVCache`` and ``generate/ar.py`` align ragged rows,
was therefore marked *valid*:

* pad columns became pooling candidates and were selected;
* at ``L == 1`` the dense path still masked them out of the attention, but they
  had already consumed top-k slots that real tokens would otherwise have had;
* at ``L > 1`` the GATHERED path (``_gathered_attention``) takes no attention
  mask at all, so a batched prefill or a speculative verify block genuinely
  attended prompt padding.

The oracle here is the one a served batch has to satisfy: a padded row in a
ragged B=2 batch must produce what that row produces when it is run alone,
unpadded.  Nothing about speculation is involved -- ragged prompt lengths are
enough.
"""

import hashlib

import mlx.core as mx
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models import glm5_next
from mlx_vlm.models.glm5_next.language import LanguageModel

# Row 0 is the long row; row 1 is short and therefore LEFT-PADDED by 5.  Five is
# deliberately not a multiple of ``index_kpool`` (3), so a pool layout anchored
# at buffer 0 cannot accidentally agree with one anchored at the row's first
# real token.
ROW0 = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 2]
ROW1 = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
PAD = len(ROW0) - len(ROW1)
# A four-token continuation: the shape of a speculative verify block, and the
# one that takes the gathered path at ANY context length (``L <= 8``).
CONT0 = [2, 2, 2, 2]
CONT1 = [5, 9, 13, 17]

# A batched B=2 reduction and a B=1 reference are not bit-equal in float32; the
# same tolerance, and the same reason, as the ragged/per-row rollback tests.
# Measured worst deviation with the fix in place: 9.5e-07 on a logit scale of
# 1.93.  Without it: 2.13, i.e. six orders of magnitude larger.
_ROW_TOL = 1e-4


def _tiny_glm5_next():
    """1 KDA layer + 1 DSA layer, hidden 128, float32.

    ``index_topk 6`` / ``index_kpool 3`` puts the indexer in its ACTIVE regime
    (T > index_topk) from the first forward and leaves ``select_k = 2`` pools out
    of 5-6 candidates, so the top-k genuinely discriminates rather than selecting
    everything.  float32 because the assertion below is numerical.
    """
    config = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=32,
        hidden_size=128,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=0,
        v_head_dim=16,
        qk_nope_head_dim=16,
        qk_head_dim=16,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        index_topk=6,
        index_head_dim=16,
        index_n_heads=2,
        index_kpool=3,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 16,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        eos_token_id=1,
    )
    mx.random.seed(7)
    model = LanguageModel(config)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def model():
    """One model for the module: it is stateless between tests (each test brings
    its own caches) and building it is the slow part."""
    return _tiny_glm5_next()


@pytest.fixture
def force_gather(monkeypatch):
    """Take the gathered path at prefill.

    ``_GATHER_MIN_CONTEXT`` (6144 by default) is the depth above which a long
    query block gathers instead of masking densely; a verify block of ``L <= 8``
    gathers at any depth.  A 16-token fixture cannot reach 6144, so the knob --
    which is a documented env override, ``MLX_VLM_GLM5_GATHER_MIN_CONTEXT`` --
    stands in for the served long-context prefill.
    """
    monkeypatch.setattr(glm5, "_GATHER_MIN_CONTEXT", 0)


def _batched(model):
    """Prefill a ragged B=2 batch on the caches a served batch actually gets,
    then run a 4-token block.  Returns (prefill logits, block logits, caches)."""
    cache = _make_cache(model, [0, PAD])
    ids = mx.array([ROW0, [0] * PAD + ROW1], dtype=mx.int32)
    prefill = model(ids, cache=cache).logits
    block = model(mx.array([CONT0, CONT1], dtype=mx.int32), cache=cache).logits
    mx.eval(prefill, block)
    return prefill, block, cache


def _alone(model):
    """The same row 1, unpadded, on its own."""
    cache = _make_cache(model, [0])
    prefill = model(mx.array([ROW1], dtype=mx.int32), cache=cache).logits
    block = model(mx.array([CONT1], dtype=mx.int32), cache=cache).logits
    mx.eval(prefill, block)
    return prefill, block, cache


def _worst(a, b):
    return mx.abs(a - b).max().item()


# --------------------------------------------------------------------------
# 1. the served invariant: the padded row is the row


def test_padded_row_prefill_matches_the_row_alone(model, force_gather):
    b_pre, _, _ = _batched(model)
    s_pre, _, _ = _alone(model)
    worst = _worst(b_pre[1, PAD:], s_pre[0])
    assert worst <= _ROW_TOL, (
        f"row 1 prefill: max abs diff {worst:.3e} against the same prompt run "
        f"alone -- the left padding is being attended (or is displacing real "
        f"tokens out of the top-k)"
    )


def test_padded_row_verify_block_matches_the_row_alone(model):
    """No ``force_gather`` here on purpose: a 4-token block is ``L <= 8``, so it
    takes the gathered path at the STOCK gate.  This is the served speculative
    verify block, at any context length."""
    _, b_blk, _ = _batched(model)
    _, s_blk, _ = _alone(model)
    worst = _worst(b_blk[1], s_blk[0])
    assert worst <= _ROW_TOL, (
        f"row 1 S=4 verify block: max abs diff {worst:.3e} against the same "
        f"block run alone -- the gathered path attended prompt padding"
    )


def test_padded_row_dense_path_also_matches(model, monkeypatch):
    """The dense ``L > 1`` path masks padding out of the attention, so it never
    attended a pad column -- but with padding marked valid the SELECTION was
    still wrong, and real tokens were dropped from the top-k in favour of pads
    that were then masked away.  It must agree too."""
    for layer in model.model.layers:
        if not layer.is_linear:
            monkeypatch.setattr(
                layer.self_attn, "use_gathered_attention", False, raising=False
            )
    b_pre, b_blk, _ = _batched(model)
    s_pre, s_blk, _ = _alone(model)
    assert _worst(b_pre[1, PAD:], s_pre[0]) <= _ROW_TOL
    assert _worst(b_blk[1], s_blk[0]) <= _ROW_TOL


def test_padded_row_decode_step_matches_the_row_alone(model):
    """One S=1 decode step, the path a server spends its time in.

    At ``L == 1`` the dense path DOES mask padding out of the attention
    (``mkeys``), so this step never attended a pad column even before the fix --
    but pad columns had already taken top-k slots away from real tokens, so the
    row still answered differently from the same row alone.  Both effects have
    to be gone."""
    b_cache = _make_cache(model, [0, PAD])
    model(mx.array([ROW0, [0] * PAD + ROW1], dtype=mx.int32), cache=b_cache)
    model(mx.array([CONT0, CONT1], dtype=mx.int32), cache=b_cache)
    b_step = model(mx.array([[7], [7]], dtype=mx.int32), cache=b_cache).logits

    s_cache = _make_cache(model, [0])
    model(mx.array([ROW1], dtype=mx.int32), cache=s_cache)
    model(mx.array([CONT1], dtype=mx.int32), cache=s_cache)
    s_step = model(mx.array([[7]], dtype=mx.int32), cache=s_cache).logits

    worst = _worst(b_step[1], s_step[0])
    assert worst <= _ROW_TOL, (
        f"row 1 decode step: max abs diff {worst:.3e} against the same stream "
        f"alone -- padding is still displacing real tokens out of the top-k"
    )


# --------------------------------------------------------------------------
# 2. the mechanism, asserted directly


def test_indexer_cache_marks_padding_invalid(model, force_gather):
    """The packed validity channel (``2 * head_dim``) of the indexer's own KV
    cache must be False over row 1's pad prefix and True over its real tokens."""
    _, _, cache = _batched(model)
    idx_cache = cache[1][1]  # CacheList(MLA latent KV, indexer KV)
    valid = idx_cache.keys[:, 0, : idx_cache._idx, -1] > 0
    row1 = valid[1].tolist()
    assert row1[:PAD] == [False] * PAD, (
        f"indexer cache marks row 1's {PAD} padding columns {row1[:PAD]} -- "
        f"padding is a pooling candidate"
    )
    assert all(row1[PAD:]), "real tokens must stay valid"
    assert all(valid[0].tolist()), "the unpadded row must be entirely valid"


def test_topk_never_selects_a_padding_column(model, force_gather):
    """No selected index for the padded row may point into its pad prefix -- in
    the prefill block or in the S=4 verify block.  This is the assertion the
    gathered path depends on, because it applies no mask of its own."""
    seen = []
    original = glm5.Glm5NextIndexer.__call__

    def spy(self, x, qr, mask, cache=None):
        out = original(self, x, qr, mask, cache=cache)
        if out is not None:
            seen.append(mx.array(out))
        return out

    glm5.Glm5NextIndexer.__call__ = spy
    try:
        _batched(model)
    finally:
        glm5.Glm5NextIndexer.__call__ = original

    assert seen, "the indexer never produced a selection -- fixture is inert"
    for call, topk in enumerate(seen):
        row1 = topk[1, 0]  # [S, width]
        for q in range(row1.shape[0]):
            bad = sorted({v for v in row1[q].tolist() if 0 <= v < PAD})
            assert not bad, (
                f"call {call} query {q}: selected padding columns {bad} "
                f"(row 1's real tokens start at buffer {PAD})"
            )


def test_padding_that_grows_after_the_write_is_still_excluded(model):
    """Defense in depth: the packed validity channel is the WRITE-time record.

    With the derivation above correct this guard is redundant for every state
    the tree currently produces -- ``BatchKVCache.extend`` right-justifies a
    joining row into ZEROS, whose validity channel reads 0 anyway.  It is here
    because the ``L > 1`` gathered path has no attention mask to fall back on,
    so "a padding column is never a candidate" should be a property of the
    cache, not of whoever wrote it.  The state below is that disagreement made
    explicit: a prefix written as valid, on a row whose declared left padding
    later says it is not.
    """
    cache = _make_cache(model, [0, 0])
    model(mx.array([ROW0, ROW0], dtype=mx.int32), cache=cache)
    grown = mx.array([0, 5])
    for sub in cache[1].caches:  # CacheList(MLA latent KV, indexer KV)
        sub.left_padding = grown
        sub.offset = sub.offset - grown

    seen = []
    original = glm5.Glm5NextIndexer.__call__

    def spy(self, x, qr, mask, cache=None):
        out = original(self, x, qr, mask, cache=cache)
        if out is not None:
            seen.append(mx.array(out))
        return out

    glm5.Glm5NextIndexer.__call__ = spy
    try:
        model(mx.array([CONT0, CONT1], dtype=mx.int32), cache=cache)
    finally:
        glm5.Glm5NextIndexer.__call__ = original

    assert seen, "the indexer never produced a selection -- fixture is inert"
    for call, topk in enumerate(seen):
        row1 = topk[1, 0]
        for q in range(row1.shape[0]):
            bad = sorted({v for v in row1[q].tolist() if 0 <= v < 5})
            assert not bad, (
                f"call {call} query {q}: selected columns {bad} that the cache's "
                f"live left_padding declares to be padding"
            )


def test_row_valid_is_none_without_a_per_row_cache():
    """The helper must answer ``None`` for a scalar-offset cache, so the
    single-stream path keeps its all-ones default and is not touched."""
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    assert glm5._batch_row_valid(None, 1, 0, 4) is None
    assert glm5._batch_row_valid(KVCache(), 1, 0, 4) is None
    got = glm5._batch_row_valid(BatchKVCache([0, 3]), 2, 0, 4)
    assert got.tolist() == [[True] * 4, [False, False, False, True]]


# --------------------------------------------------------------------------
# 3. regression: the unpadded single-row path is byte-identical


# sha256 over the float32 bytes of a B=1 unpadded forward (prefill + a 4-token
# block) on both cache flavours.  Pinned so a change to the validity derivation
# that touched the single-stream path would show up as a different digest rather
# than as a tolerance.  CPU/float32 fixed point; see the skip below.
_B1_DIGEST = "62aec0cd8acacd9d9370f6050c047bf477cd84c1290bfafe8faac2dfa744c001"


def _b1_digest(model, cache):
    pre = model(mx.array([ROW1], dtype=mx.int32), cache=cache).logits
    blk = model(mx.array([CONT1], dtype=mx.int32), cache=cache).logits
    mx.eval(pre, blk)
    h = hashlib.sha256()
    for a in (pre, blk):
        h.update(bytes(memoryview(a.astype(mx.float32))))
    return h.hexdigest()


@pytest.mark.skipif(
    mx.default_device() != mx.cpu,
    reason="the digest is a CPU/float32 fixed point (MLX_DEFAULT_DEVICE=cpu)",
)
def test_unpadded_single_row_is_byte_identical(model):
    scalar = _b1_digest(model, model.make_cache())
    batched = _b1_digest(model, _make_cache(model, [0]))
    assert scalar == _B1_DIGEST, (
        f"single-stream (scalar-offset KVCache) B=1 output changed: {scalar}"
    )
    assert batched == _B1_DIGEST, (
        f"unpadded B=1 on a BatchKVCache changed: {batched}"
    )
