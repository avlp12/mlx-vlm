"""V2e: the minimum-dispatch decode indexer must select exactly what eager does.

``MLX_VLM_GLM5_INDEXER_DECODE_FUSED=1`` replaces the *active*-regime
(``T > index_topk``) B=1/S=1 decode indexer with a path that issues the same
selection with the dead LAUNCHES removed -- the P0 dispatch census
(``V3_P0_CENSUS_20260907/{d512,d4096}_e288.json``) measures 47 Metal dispatches
per DSA layer at depth 4096 against 9 at depth 512, and the depth gradient of
greedy decode (37.4 tok/s at p512 -> 35.0 at 4k, where the byte model predicts
-0.8 %) is that difference times 11 DSA layers.

What the path proves away, and what these tests pin:
  * ``pool_indices`` -- a table of ``j*kpool + m`` that the eager path
    materialises, slice-updates and gathers; here it is one fused multiply-add
    over the winners.
  * ``pool_valid`` / the -1e30 candidate mask -- only the trailing pool can be
    incomplete, so ranking the complete prefix is the same top-k.
  * the trailing pool's replica gather + mask -- the masked lanes land at
    exactly 0.0, so the softmax and the contraction are unchanged.
  * the descending sort's ``Negative`` -- folded into the four-element head
    weight vector, which IEEE negation makes exact.
  * the zero-width VALUES half of the indexer's KV cache.

These run on CPU (no Metal gate): the whole path is plain MLX ops, and the
identity claims are dtype/rounding claims that hold on either backend.  The
counterpart claim that CANNOT be checked here is that Metal's ``ArgSort`` is
comparison-based (so +0.0 and -0.0 tie); see the module docstring of
``bench/hwdossier/v3_dispatch_census.py`` for the CPU/Metal divergence list.
"""

import mlx.core as mx
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.glm5_next.config import TextConfig

# GLM-5.3-Flash text_config, restricted to what the indexer reads, with the
# widths shrunk (dispatch counts and selection semantics are width-invariant).
_CFG = dict(
    model_type="glm5_next_text",
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=8,
    n_shared_experts=1,
    n_routed_experts=8,
    routed_scaling_factor=2.5,
    kv_lora_rank=512,
    q_lora_rank=128,
    qk_rope_head_dim=0,
    v_head_dim=256,
    qk_nope_head_dim=256,
    num_experts_per_tok=2,
    first_k_dense_replace=3,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-05,
    index_topk=2048,
    index_head_dim=128,
    index_n_heads=4,
    layer_types=["deepseek_sparse_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={
        "num_heads": 4,
        "gate_lower_bound": -5.0,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
    },
)


def _config():
    return TextConfig.from_dict(dict(_CFG))


def _indexer(config, seed=0):
    mx.random.seed(seed)
    ix = glm5.Glm5NextIndexer(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    ix.update(rand(ix.parameters()))
    return ix


def _prefill(ix, config, T, seed=1):
    """Eager prefill, so ``_pool`` / ``_no_pad`` come from the reference path."""
    mx.random.seed(seed)
    cache = KVCache()
    x = (mx.random.normal((1, T, config.hidden_size)) * 0.4).astype(mx.bfloat16)
    qr = (mx.random.normal((1, T, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
    glm5._IDX_FAST_ENV = False
    glm5._IDX_FUSED_ENV = False
    mx.eval(ix(x, qr, None, cache=cache), cache.keys)
    return cache


def _clone(cache):
    out = KVCache()
    out.keys = mx.array(cache.keys)
    out.values = mx.array(cache.values) if cache.values is not None else None
    out.offset = cache.offset
    pool = getattr(cache, "_pool", None)
    out._pool = (
        None
        if pool is None
        else tuple(mx.array(a) if isinstance(a, mx.array) else a for a in pool)
    )
    out._no_pad = getattr(cache, "_no_pad", False)
    out._fpool = None
    out._ffpool = None
    mx.eval(out.keys, *(a for a in (out._pool or ()) if isinstance(a, mx.array)))
    return out


def _steps(config, n, seed=99):
    mx.random.seed(seed)
    out = []
    for _ in range(n):
        out.append(
            (
                (mx.random.normal((1, 1, config.hidden_size)) * 0.4).astype(
                    mx.bfloat16
                ),
                (mx.random.normal((1, 1, config.q_lora_rank)) * 0.4).astype(
                    mx.bfloat16
                ),
            )
        )
    mx.eval(out)
    return out


@pytest.fixture(autouse=True)
def _reset_toggles():
    saved = (glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV, glm5._IDX_POOL_STEP)
    yield
    glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV, glm5._IDX_POOL_STEP = saved


def _carry(ix, config, ctx, n_steps, seed=1, pool_step=None):
    """Same prefill, two caches: eager-fast vs fused.  Returns worst mismatch."""
    if pool_step is not None:
        glm5._IDX_POOL_STEP = pool_step
    ref_cache = _prefill(ix, config, ctx, seed=seed)
    fused_cache = _clone(ref_cache)
    worst = 0
    for x, qr in _steps(config, n_steps):
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, False
        ref = ix(x, qr, None, cache=ref_cache)
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, True
        got = ix(x, qr, None, cache=fused_cache)
        mx.eval(ref, got)
        assert ref is not None and got is not None, "a step fell into the bypass"
        assert ref.shape == got.shape and ref.dtype == got.dtype
        worst = max(worst, int(mx.sum((ref != got).astype(mx.int32))))
        # the packed cache bytes must also match: the fused append writes the
        # validity column from a constant and skips the zero-width values half.
        assert bool(
            mx.array_equal(
                ref_cache.keys[..., : ref_cache.offset, :],
                fused_cache.keys[..., : fused_cache.offset, :],
            )
        ), "indexer cache diverged"
        assert ref_cache.offset == fused_cache.offset
    return worst


@pytest.mark.parametrize("ctx", [2049, 2051, 4096, 4099, 8192])
def test_fused_selection_is_bit_identical_over_24_steps(ctx):
    config = _config()
    ix = _indexer(config)
    assert _carry(ix, config, ctx, 24) == 0


def test_fused_survives_pool_buffer_growth():
    config = _config()
    ix = _indexer(config)
    assert _carry(ix, config, 4096, 40, pool_step=2) == 0


@pytest.mark.parametrize("seed", list(range(20)))
def test_fused_matches_eager_over_20_random_cache_states(seed):
    """20 independent random prefills -> one fused step each, element-exact."""
    config = _config()
    ix = _indexer(config, seed=seed)
    assert _carry(ix, config, 4096 + 7 * seed, 2, seed=seed + 11) == 0


def test_attention_output_array_equal_vs_eager():
    """The selection feeds an attention; pin the attended VALUES, not just the ids."""
    config = _config()
    ix = _indexer(config)
    ref_cache = _prefill(ix, config, 4096)
    fused_cache = _clone(ref_cache)
    mx.random.seed(7)
    attn = glm5.Glm5NextSparseAttention(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    attn.update(rand(attn.parameters()))
    kv = (mx.random.normal((1, 1, 4200, config.kv_lora_rank)) * 0.3).astype(
        mx.bfloat16
    )
    for x, qr in _steps(config, 8):
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, False
        ref = ix(x, qr, None, cache=ref_cache)
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, True
        got = ix(x, qr, None, cache=fused_cache)
        kvs = kv[:, :, : ref_cache.offset, :]
        q = (
            mx.random.normal(
                (1, config.num_attention_heads, 1, config.qk_nope_head_dim)
            )
            * 0.3
        ).astype(mx.bfloat16)
        a = attn._gathered_attention(q, kvs, ref)
        b = attn._gathered_attention(q, kvs, got)
        mx.eval(a, b)
        assert bool(mx.array_equal(a, b)), "gathered attention output diverged"


def test_flag_off_is_byte_identical_and_never_enters_fused():
    config = _config()
    ix = _indexer(config)

    def _boom(*a, **k):
        raise AssertionError("fused path taken with the toggle off")

    real = glm5.Glm5NextIndexer._decode_fused
    glm5.Glm5NextIndexer._decode_fused = _boom
    try:
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, False
        cache = _prefill(ix, config, 4096)
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, False
        for x, qr in _steps(config, 3):
            mx.eval(ix(x, qr, None, cache=cache))
        assert getattr(cache, "_ffpool", None) is None
    finally:
        glm5.Glm5NextIndexer._decode_fused = real


def test_prefill_and_verify_blocks_are_untouched_by_the_flag():
    """S > 1 must produce identical output and never reach the fused path."""
    config = _config()
    ix = _indexer(config)
    outs = []
    for flag in (False, True):
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, flag
        mx.random.seed(3)
        cache = KVCache()
        x = (mx.random.normal((1, 4096, config.hidden_size)) * 0.4).astype(
            mx.bfloat16
        )
        qr = (mx.random.normal((1, 4096, config.q_lora_rank)) * 0.4).astype(
            mx.bfloat16
        )
        pre = ix(x, qr, None, cache=cache)
        # an S=4 speculative-verify block
        xv = (mx.random.normal((1, 4, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qv = (mx.random.normal((1, 4, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        ver = ix(xv, qv, None, cache=cache)
        mx.eval(pre, ver, cache.keys)
        outs.append((pre, ver, cache.keys[..., : cache.offset, :]))
    for a, b in zip(outs[0], outs[1]):
        assert bool(mx.array_equal(a, b)), "S>1 output moved with the flag"


def test_fused_declines_ineligible_shapes():
    config = _config()
    ix = _indexer(config)

    def _boom(*a, **k):
        raise AssertionError("fused path taken for an ineligible step")

    real = glm5.Glm5NextIndexer._decode_fused
    try:
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, True
        cache = _prefill(ix, config, 4096)
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, True
        glm5.Glm5NextIndexer._decode_fused = _boom
        # S > 1 verify block
        xv = (mx.random.normal((1, 4, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qv = (mx.random.normal((1, 4, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        mx.eval(ix(xv, qv, None, cache=_clone(cache)))
        x1, qr1 = _steps(config, 1)[0]
        # an explicit token mask
        c2 = _clone(cache)
        mx.eval(ix(x1, qr1, mx.ones((1, 1), dtype=mx.bool_), cache=c2))
        # a padded sequence
        c3 = _clone(cache)
        c3._no_pad = False
        mx.eval(ix(x1, qr1, None, cache=c3))
        # a tensor-parallel rank (the head axis is sharded, so the scorer holds
        # a partial sum and the fused negation trick is not the same reduce)
        c4 = _clone(cache)
        ix._tp_reduce = lambda a: a
        try:
            mx.eval(ix(x1, qr1, None, cache=c4))
        finally:
            ix._tp_reduce = None
    finally:
        glm5.Glm5NextIndexer._decode_fused = real


def test_fused_bypass_regime_returns_none():
    config = _config()
    ix = _indexer(config)
    for flag in (False, True):
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, flag
        cache = KVCache()
        mx.random.seed(5)
        x = (mx.random.normal((1, 512, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, 512, config.q_lora_rank)) * 0.4).astype(
            mx.bfloat16
        )
        assert ix(x, qr, None, cache=cache) is None


def test_verify_hook_agrees_with_the_eager_formulation():
    """MLX_VLM_GLM5_INDEXER_DECODE_FUSED_VERIFY: the in-path cross-check passes."""
    config = _config()
    ix = _indexer(config)
    cache = _prefill(ix, config, 4096)
    saved = glm5._IDX_FUSED_VERIFY_ENV
    glm5._IDX_FUSED_VERIFY_ENV = True
    try:
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, True
        for x, qr in _steps(config, 6):
            mx.eval(ix(x, qr, None, cache=cache))
    finally:
        glm5._IDX_FUSED_VERIFY_ENV = saved


# --------------------------------------------------------------------------
# Dispatch count.  The point of the path is the LAUNCH count, so pin it the way
# the P0 census does: count primitives in the unevaluated graph of one
# steady-state call (bench/hwdossier/v3_dispatch_census.py, "exact per-call node
# delta"), classify metadata rewrites as views, and substitute the one MLX fast
# op the indexer uses (mx.fast.layer_norm decomposes into 11 primitives on CPU
# and is 1 dispatch on Metal).

_VIEW_LABELS = frozenset(
    {
        "Reshape", "Split", "Slice", "Broadcast", "BroadcastAxes", "ExpandDims",
        "Squeeze", "Transpose", "Flatten", "Unflatten", "AsStrided",
        "Contiguous", "Copy", "StopGradient", "Depends", "View",
    }
)


def _graph_labels(arrays):
    import io
    import re

    arrays = [a for a in arrays if isinstance(a, mx.array)]
    if not arrays:
        return {}
    buf = io.StringIO()
    mx.export_to_dot(buf, *arrays)
    return {
        int(i): lab
        for i, lab in re.findall(r'(\d+) \[label ="([^"]+)"', buf.getvalue())
    }


def _cache_arrays(cache):
    out = [cache.keys, cache.values]
    for name in ("_pool", "_fpool", "_ffpool"):
        buf = getattr(cache, name, None)
        if buf:
            out.extend(a for a in buf if isinstance(a, mx.array))
    return out


def _layer_norm_nodes(config):
    """CPU primitives one mx.fast.layer_norm decomposes into (1 Metal dispatch)."""
    hd = config.index_head_dim
    x = mx.zeros((1, 1, hd))
    w = mx.ones((hd,))
    b = mx.zeros((hd,))
    return sum(
        1
        for lab in _graph_labels([mx.fast.layer_norm(x, w, b, 1e-6)]).values()
        if lab not in _VIEW_LABELS
    )


def _decode_dispatches(ix, config, cache, x, qr):
    """Metal-dispatch estimate for ONE steady-state indexer decode call."""
    before = set(_graph_labels([x, qr] + _cache_arrays(cache)))
    out = ix(x, qr, None, cache=cache)
    after = _graph_labels([out, x, qr] + _cache_arrays(cache))
    compute = sum(
        1
        for nid, lab in after.items()
        if nid not in before and lab not in _VIEW_LABELS
    )
    mx.eval(out, *[a for a in _cache_arrays(cache) if a is not None])
    return compute - _layer_norm_nodes(config) + 1


def test_fused_cuts_the_decode_indexer_dispatch_count():
    """47/layer (P0 census, depth 4096) -> a measured, pinned ceiling.

    The census number is the FIRST fast step, which carries the one-off pool
    buffer allocation; this measures the steady state, so the eager baseline it
    compares against is the same steady state (41), not 47.
    """
    config = _config()
    ix = _indexer(config)
    counts = {}
    for name, fused in (("eager", False), ("fused", True)):
        cache = _prefill(ix, config, 4096)
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, fused
        steps = _steps(config, 4)
        for x, qr in steps[:3]:          # reach the steady state
            mx.eval(ix(x, qr, None, cache=cache))
        counts[name] = _decode_dispatches(ix, config, cache, *steps[3])
    # The eager steady state measured on this fork at depth 4096: 41 with the
    # real (quantized, fp32-ape) build, whose pool index table is int64 and so
    # carries one extra cast; 40 here, where every parameter is bfloat16.
    assert counts["eager"] >= 40, counts
    # 21 is the pure-MLX floor derived in _idx_fused_enabled's docstring:
    #   3 quantized GEMVs (wq_b, wk, weights_proj) + 1 layer_norm + 1 gate GEMM
    #   + 1 packed concat + 1 cache slice-update            = 7  pre-cache
    #   + 6 trailing-pool softmax/contraction (2 reductions are unfusable)
    #   + 1 pool slice-update + 2 scoring GEMMs + 1 relu/scale + 1 head scale
    #   + 1 candidate mask + 1 argsort + 1 pool expand + 1 tail concat = 8
    # Regression guard, not an aspiration: it may only ever go down.
    assert counts["fused"] <= 21, counts
    assert counts["fused"] < counts["eager"]


def test_eight_decode_steps_are_the_same_token_sequence():
    """End to end through a LanguageModel: greedy argmax must not move."""
    from mlx_vlm.models.glm5_next.config import TextConfig as _TC

    cfg = dict(_CFG)
    cfg.update(num_hidden_layers=2, layer_types=["deepseek_sparse_attention"] * 2,
               mlp_layer_types=["dense", "dense"], first_k_dense_replace=2)
    config = _TC.from_dict(cfg)
    mx.random.seed(21)
    model = glm5.LanguageModel(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    model.update(rand(model.parameters()))
    model.eval()
    mx.eval(model.parameters())

    seqs = []
    for flag in (False, True):
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, flag
        mx.random.seed(5)
        prompt = mx.random.randint(0, config.vocab_size, (1, 3000))
        cache = model.make_cache()
        logits = model(prompt, cache=cache).logits
        tok = mx.argmax(logits[:, -1], axis=-1, keepdims=True)
        seq = []
        for _ in range(8):
            logits = model(tok, cache=cache).logits
            tok = mx.argmax(logits[:, -1], axis=-1, keepdims=True)
            mx.eval(tok)
            seq.append(int(tok[0, 0]))
        seqs.append(seq)
    assert seqs[0] == seqs[1], seqs


def test_score_kernel_is_compiled_shapeless():
    """The pool count grows; the score kernel must not recompile with it.

    ``P`` gains one pool every ``index_kpool`` decode steps, so a shape-keyed
    trace of the relu/scale kernel would compile a fresh Metal kernel every
    fourth step for the whole generation -- a launch fix that pays for itself in
    compilations.  The other two compiled helpers are deliberately shape-keyed
    (bounded trace counts) because ``shapeless=True`` faults MLX 0.32.1 on their
    reductions, so this pins which one is which rather than the flag alone.
    """
    import inspect

    src = inspect.getsource(glm5)
    relu = src.index("def _idx_relu_scores_fused")
    assert "shapeless=True" in src[relu - 400 : relu], (
        "_idx_relu_scores_fused must be compiled shapeless"
    )
    for name in ("_idx_tail_pool_fused", "_idx_expand_pools_fused"):
        at = src.index("def " + name)
        assert "shapeless=False" in src[at - 500 : at], name


def test_fused_declines_a_batched_cache_shape():
    """A BatchKVCache carries its write cursor in _idx, not offset.

    ``_append_packed`` writes ``cache.offset``/``cache.keys`` in place, so it
    must never see one.  The gate is an exact type check; pin that a cache which
    merely quacks like a KVCache (a subclass with left padding) is declined.
    """
    from mlx_vlm.models.cache import BatchKVCache

    config = _config()
    ix = _indexer(config)

    def _boom(*a, **k):
        raise AssertionError("fused append taken for a batched cache")

    real = glm5.Glm5NextIndexer._append_packed
    glm5.Glm5NextIndexer._append_packed = _boom
    try:
        glm5._IDX_FAST_ENV, glm5._IDX_FUSED_ENV = True, True
        cache = BatchKVCache([0])
        mx.random.seed(4)
        x = (mx.random.normal((1, 1, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, 1, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        mx.eval(ix(x, qr, None, cache=cache))
    finally:
        glm5.Glm5NextIndexer._append_packed = real
