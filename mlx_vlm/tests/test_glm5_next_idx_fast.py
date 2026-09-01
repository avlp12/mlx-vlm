"""Eager vs fast equivalence for the GLM-5-Next DSA indexer decode path.

``MLX_VLM_GLM5_IDX_FAST=1`` replaces the indexer's *active* decode path (the
regime ``T > index_topk``, where the short-context bypass no longer applies)
with one that does only the work a new token actually creates.

The eager path recomputes, every step and in every DSA layer:
  * ``kv_pos`` / ``visible`` / ``pool_visible`` -- O(T) tensors that, for a
    single unpadded stream, are all-True by construction;
  * ``_visible_tail`` -- an O(T) reduction whose result is then a closed form
    in ``T`` alone;
  * ``mx.concatenate`` of the stable pool prefix -- an O(P) copy of pool state
    that is append-only during decode;
  * the whole index apparatus of ``_pooled_states`` (argmax/arange/clip/three
    gathers/validity reductions) for a tail of at most ``index_kpool`` tokens,
    where every one of those indices is known on the host.

None of that changes the selection, so the fast path is expected to be
*bit-identical*: the returned top-k index tensor must match element for
element across carried steps.  These tests pin that at three context lengths,
pin the pool-buffer growth path, and pin that anything the path is not written
for (prefill, S>1 verify blocks, batching, left-padding, a rollback that
shortened the cache) falls back to the eager path.
"""

import mlx.core as mx
import pytest

import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.glm5_next.config import TextConfig

# GLM-5.3-Flash text_config, restricted to what the indexer reads.
_CFG = dict(
    model_type="glm5_next_text",
    vocab_size=1024,
    hidden_size=4096,
    intermediate_size=12288,
    moe_intermediate_size=2048,
    num_hidden_layers=1,
    num_attention_heads=64,
    num_key_value_heads=64,
    n_shared_experts=1,
    n_routed_experts=288,
    routed_scaling_factor=2.5,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=0,
    v_head_dim=256,
    qk_nope_head_dim=256,
    num_experts_per_tok=8,
    first_k_dense_replace=3,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-05,
    index_topk=2048,
    index_head_dim=128,
    index_n_heads=32,
    layer_types=["deepseek_sparse_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={
        "num_heads": 64,
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


def _prefill(ix, config, T, seed=1, batch=1):
    """Eager prefill, so `_pool` / `_no_pad` are built by the reference path."""
    mx.random.seed(seed)
    cache = KVCache()
    x = (mx.random.normal((batch, T, config.hidden_size)) * 0.4).astype(mx.bfloat16)
    qr = (mx.random.normal((batch, T, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
    glm5._IDX_FAST_ENV = False
    mx.eval(ix(x, qr, None, cache=cache), cache.keys)
    return cache


def _clone(cache):
    out = KVCache()
    out.keys = mx.array(cache.keys)
    out.values = cache.values
    out.offset = cache.offset
    # A prefill that landed in the short-context bypass returns before `_pool`
    # and `_no_pad` are ever written, so the boundary cells clone a cache that
    # has neither.  Carry that state faithfully -- inventing a pool here would
    # hand the fast path an eligibility it does not have in production.
    pool = getattr(cache, "_pool", None)
    out._pool = (
        None
        if pool is None
        else tuple(mx.array(a) if isinstance(a, mx.array) else a for a in pool)
    )
    out._no_pad = getattr(cache, "_no_pad", False)
    out._fpool = None
    if out._pool is None:
        mx.eval(out.keys)
    else:
        mx.eval(out.keys, out._pool[0], out._pool[1], out._pool[2])
    return out


def _steps(config, n, batch=1, seed=99):
    mx.random.seed(seed)
    out = []
    for _ in range(n):
        out.append(
            (
                (mx.random.normal((batch, 1, config.hidden_size)) * 0.4).astype(
                    mx.bfloat16
                ),
                (mx.random.normal((batch, 1, config.q_lora_rank)) * 0.4).astype(
                    mx.bfloat16
                ),
            )
        )
    mx.eval(out)
    return out


@pytest.fixture(autouse=True)
def _reset_toggle():
    saved = (glm5._IDX_FAST_ENV, glm5._IDX_POOL_STEP)
    yield
    glm5._IDX_FAST_ENV, glm5._IDX_POOL_STEP = saved


def _carry(ix, config, ctx, n_steps, seed=1):
    """Run n_steps eager and fast from the same prefill; return worst mismatch."""
    eager_cache = _prefill(ix, config, ctx, seed=seed)
    fast_cache = _clone(eager_cache)
    worst = 0
    for x, qr in _steps(config, n_steps):
        glm5._IDX_FAST_ENV = False
        ref = ix(x, qr, None, cache=eager_cache)
        glm5._IDX_FAST_ENV = True
        got = ix(x, qr, None, cache=fast_cache)
        mx.eval(ref, got)
        assert ref is not None and got is not None
        assert ref.shape == got.shape and ref.dtype == got.dtype
        worst = max(worst, int(mx.sum((ref != got).astype(mx.int32))))
    return worst


@pytest.mark.parametrize("ctx", [4096, 16384])
def test_idx_fast_matches_eager_over_32_decode_steps(ctx):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    # The selection is the same selection, reached with less arithmetic, so this
    # is exact rather than merely close.
    assert _carry(ix, config, ctx, 32) == 0


def test_idx_fast_survives_pool_buffer_growth():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    # Shrink the growth step so the preallocated pool buffers must be grown
    # several times inside the run instead of once every 2048 tokens.
    glm5._IDX_POOL_STEP = 2
    assert _carry(ix, config, 4096, 40) == 0


def test_idx_fast_survives_interleaved_verify_blocks():
    """Speculative decoding interleaves S>1 verify blocks with S=1 decode.

    The fast path owns the pool state in `_fpool` and clears `_pool`; a verify
    block is ineligible, so it must fall back to a full rebuild, restore
    `_pool`, and let the next single-token step re-seed `_fpool` from it.  Run
    the identical mixed sequence eager and fast and require the selections to
    stay bit-identical the whole way through.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    eager_cache = _prefill(ix, config, 4096)
    fast_cache = _clone(eager_cache)

    mx.random.seed(1234)
    plan = []
    for i in range(4):
        for _ in range(3):
            plan.append(1)
        plan.append(4)  # a draft-block verify
    for n in plan:
        x = (mx.random.normal((1, n, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, n, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        mx.eval(x, qr)
        glm5._IDX_FAST_ENV = False
        ref = ix(x, qr, None, cache=eager_cache)
        glm5._IDX_FAST_ENV = True
        got = ix(x, qr, None, cache=fast_cache)
        mx.eval(ref, got)
        assert ref.shape == got.shape
        assert int(mx.sum((ref != got).astype(mx.int32))) == 0, f"diverged at S={n}"


def test_idx_fast_declines_ineligible_shapes():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)

    def _boom(*a, **k):
        raise AssertionError("fast path taken for an ineligible step")

    real = glm5.Glm5NextIndexer._decode_fast
    glm5.Glm5NextIndexer._decode_fast = _boom
    try:
        glm5._IDX_FAST_ENV = True
        # prefill (S = T) -- no pool state yet
        cache = _prefill(ix, config, 4096)
        # S > 1 speculative-verify block
        glm5._IDX_FAST_ENV = True
        mx.random.seed(3)
        x = (mx.random.normal((1, 4, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, 4, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        mx.eval(ix(x, qr, None, cache=_clone(cache)))
        # left-padded decode: a bool mask is supplied
        c2 = _clone(cache)
        x1, qr1 = _steps(config, 1)[0]
        mx.eval(ix(x1, qr1, mx.ones((1, 1), dtype=mx.bool_), cache=c2))
        # padded sequence: _no_pad is False
        c3 = _clone(cache)
        c3._no_pad = False
        mx.eval(ix(x1, qr1, None, cache=c3))
        # rollback: the cache was trimmed, so `_pool` is stale by > 1 token
        c4 = _clone(cache)
        c4._pool = c4._pool[:3] + (c4._pool[3] - 4,)
        mx.eval(ix(x1, qr1, None, cache=c4))
        # batched decode
        cb = _prefill(ix, config, 4096, batch=2)
        xb, qrb = _steps(config, 1, batch=2)[0]
        mx.eval(ix(xb, qrb, None, cache=cb))
    finally:
        glm5.Glm5NextIndexer._decode_fast = real


def test_idx_fast_bypass_regime_returns_none():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    for flag in (False, True):
        glm5._IDX_FAST_ENV = flag
        cache = KVCache()
        mx.random.seed(5)
        x = (mx.random.normal((1, 512, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, 512, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        # T = 512 <= index_topk = 2048 -> short-context bypass, no selection
        assert ix(x, qr, None, cache=cache) is None


def test_toggle_off_uses_eager_path():
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)

    def _boom(*a, **k):
        raise AssertionError("fast path taken with the toggle off")

    real = glm5.Glm5NextIndexer._decode_fast
    glm5.Glm5NextIndexer._decode_fast = _boom
    try:
        glm5._IDX_FAST_ENV = False
        cache = _prefill(ix, config, 4096)
        x, qr = _steps(config, 1)[0]
        mx.eval(ix(x, qr, None, cache=cache))
    finally:
        glm5.Glm5NextIndexer._decode_fast = real


# --------------------------------------------------------------------------
# Boundary-adjacent cells.
#
# Every context above is an exact multiple of `index_kpool` and comfortably
# inside one regime, which means the suite never ran the two off-by-one points
# the code actually has:
#
#   * the short-context bypass is `T <= index_topk` (language.py's
#     `bypass_short` guard), so `T == index_topk` is the last silent step and
#     `T == index_topk + 1` is the first step that has to build pool state from
#     nothing -- the bypass returns *before* `_pool` and `_no_pad` are written;
#   * the incremental tail `_pool_tail` sees `n = ((T - 1) % index_kpool) + 1`,
#     so a prefill length congruent to 0 mod `index_kpool` -- which is all of
#     them above -- pins the first incremental step at `n == 1` forever.  The
#     `n == index_kpool` case (a pool completing) and the "new pool opens"
#     case were both unreached.
#
# These cells land ON those points rather than stepping over them.  They are
# written against `_CFG` rather than the literal 2048/4 so that they follow the
# config if it moves.
# --------------------------------------------------------------------------

_TOPK = _CFG["index_topk"]
_KPOOL = TextConfig.from_dict(dict(_CFG)).index_kpool


def _carry_over_boundary(ix, config, ctx, n_steps, seed=1):
    """`_carry`, but able to walk steps that are still in the bypass regime.

    Returns ``(worst_mismatch, n_bypass_steps, n_active_steps)`` so a cell can
    assert that the run really did contain the transition it is named for --
    a boundary cell that silently stayed on one side would pass for free.
    """
    eager_cache = _prefill(ix, config, ctx, seed=seed)
    fast_cache = _clone(eager_cache)
    worst = n_bypass = n_active = 0
    for x, qr in _steps(config, n_steps):
        glm5._IDX_FAST_ENV = False
        ref = ix(x, qr, None, cache=eager_cache)
        glm5._IDX_FAST_ENV = True
        got = ix(x, qr, None, cache=fast_cache)
        if ref is None or got is None:
            # The two arms must agree about *whether* the indexer speaks at
            # all; a one-step disagreement here would be a silent selection
            # change that a value comparison could never see.
            assert ref is None and got is None, (
                f"arms disagree on the bypass at T={eager_cache.offset}: "
                f"eager={'None' if ref is None else 'array'} "
                f"fast={'None' if got is None else 'array'}"
            )
            n_bypass += 1
            continue
        n_active += 1
        mx.eval(ref, got)
        assert ref.shape == got.shape and ref.dtype == got.dtype
        worst = max(worst, int(mx.sum((ref != got).astype(mx.int32))))
    return worst, n_bypass, n_active


def test_bypass_predicate_is_inclusive_at_index_topk():
    """Pin which side of `index_topk` is silent, and how wide the other side is.

    This is the off-by-one itself, with no decode loop around it: at exactly
    `index_topk` the indexer must return None and leave the cache with no pool
    state, and at `index_topk + 1` it must return a selection of width
    `index_topk + index_kpool - 1` (the always-select tail) and leave `_pool`
    and `_no_pad` behind for the incremental path to pick up.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    seen = {}
    for T in (_TOPK, _TOPK + 1):
        cache = KVCache()
        mx.random.seed(5)
        x = (mx.random.normal((1, T, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, T, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        glm5._IDX_FAST_ENV = False
        out = ix(x, qr, None, cache=cache)
        seen[T] = (out, cache)

    at, cache_at = seen[_TOPK]
    assert at is None, "T == index_topk must still take the short-context bypass"
    assert getattr(cache_at, "_pool", None) is None
    assert getattr(cache_at, "_no_pad", None) is None, (
        "the bypass must return before `_no_pad` is written -- if it starts "
        "writing it, the fast path becomes eligible one step earlier"
    )

    above, cache_above = seen[_TOPK + 1]
    assert above is not None, "T == index_topk + 1 must be the active regime"
    assert above.shape[-1] == _TOPK + (_KPOOL - 1 if _KPOOL > 1 else 0)
    assert getattr(cache_above, "_pool", None) is not None
    assert cache_above._pool[3] == _TOPK + 1


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_idx_fast_matches_eager_across_the_bypass_boundary(delta):
    """Prefill one below / exactly on / one above `index_topk`, then decode.

    At delta <= 0 the first step(s) are still bypassed, so the run contains
    the step where the pool is built from an empty cache -- the branch the
    comment at the bypass calls "rebuilt once when T first exceeds
    index_topk", which no cell reached before.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    worst, n_bypass, n_active = _carry_over_boundary(ix, config, _TOPK + delta, 8)
    assert worst == 0
    # The cell is only meaningful if the transition actually happened in it.
    assert n_bypass == max(0, -delta), (n_bypass, delta)
    assert n_active >= 6


@pytest.mark.parametrize("delta", [1, 2, 3, 4])
def test_idx_fast_matches_eager_at_every_kpool_phase(delta):
    """One cell per residue of the prefill length mod `index_kpool`.

    The prefill length fixes the phase of the incremental tail: `_pool_tail`
    is handed `n = ((T - 1) % index_kpool) + 1`, so delta 1..4 walks the
    *first* fast step's `n` through 2, 3, 4, 1.  delta=3 is the pool that
    completes exactly on that first step (`n == index_kpool`, `pool_valid`
    True); delta=4 (`ctx` an exact multiple of `index_kpool`) is the one that
    opens a new pool on it.  Every other context in this file is congruent to
    0, so `_fpool` was only ever seeded from a `_pool` whose trailing pool was
    complete.

    Honest scope: unlike the bypass cells below, these are breadth rather than
    newly-guarded ground.  A 32-step run from a congruent-to-0 context already
    sweeps every value of `n`; what is new here is only that the seeding
    happens on an incomplete trailing pool.  A mutation that rounds the seeded
    `t_prev` down to a pool boundary in `_pool_buffers` -- the obvious bug of
    this shape -- is a semantic no-op, because `_decode_fast` consumes
    `t_prev` only through `t_prev // index_kpool`.  These cells are kept for
    the cost of 12 steps each, not on a demonstrated catch.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    ctx = _TOPK + delta
    worst, n_bypass, n_active = _carry_over_boundary(ix, config, ctx, 12)
    assert worst == 0
    assert n_bypass == 0 and n_active == 12
    # Say out loud which phase this parametrisation actually exercised, so a
    # config change that collapses the four cells into one is visible.
    assert ((ctx - 1) % _KPOOL) + 1 == ((_TOPK + delta - 1) % _KPOOL) + 1


def test_idx_fast_survives_a_verify_block_that_crosses_the_bypass():
    """The S>1 block that turns the bypass off.

    Speculative decoding does not have to cross `index_topk` one token at a
    time: a width-W verify applied just below the boundary makes the *first*
    active indexer call an S>1 call on a cache that has neither `_pool` nor
    `_no_pad`.  Neither the fast path nor the eager incremental branch is
    eligible there, so both arms must full-rebuild and then agree on every
    single-token step that follows.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    width = 4
    # Land the block so that it starts inside the bypass and ends outside it.
    ctx = _TOPK - width + 2
    eager_cache = _prefill(ix, config, ctx)
    fast_cache = _clone(eager_cache)
    assert getattr(eager_cache, "_pool", None) is None

    mx.random.seed(4321)
    plan = [width] + [1] * 6 + [width] + [1] * 3
    crossed = False
    for n in plan:
        x = (mx.random.normal((1, n, config.hidden_size)) * 0.4).astype(mx.bfloat16)
        qr = (mx.random.normal((1, n, config.q_lora_rank)) * 0.4).astype(mx.bfloat16)
        mx.eval(x, qr)
        glm5._IDX_FAST_ENV = False
        ref = ix(x, qr, None, cache=eager_cache)
        glm5._IDX_FAST_ENV = True
        got = ix(x, qr, None, cache=fast_cache)
        if ref is None or got is None:
            assert ref is None and got is None
            continue
        crossed = True
        mx.eval(ref, got)
        assert ref.shape == got.shape
        assert int(mx.sum((ref != got).astype(mx.int32))) == 0, f"diverged at S={n}"
    assert crossed, "the plan never left the bypass regime"


def test_idx_fast_pool_growth_at_the_bypass_boundary():
    """Pool-buffer growth with the smallest possible seed pool.

    `_pool_buffers` sizes its preallocation from the pool count it inherits,
    and the boundary is where that count is smallest relative to the run
    length.  Shrink the growth step so the buffers must grow repeatedly while
    the run is also crossing pool completions.
    """
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")
    config = _config()
    ix = _indexer(config)
    glm5._IDX_POOL_STEP = 2
    worst, n_bypass, n_active = _carry_over_boundary(ix, config, _TOPK + 1, 40)
    assert worst == 0
    assert n_bypass == 0 and n_active == 40
