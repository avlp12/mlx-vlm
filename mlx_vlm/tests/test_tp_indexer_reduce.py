"""The DSA indexer's head-axis contraction must be reduced before the top-k.

Tensor parallelism splits the indexer's head axis and the scorer *contracts*
it, so under sharding each rank holds a partial sum and the full score only
exists after an all-reduce.  ``shard_dsa`` has installed a ``_tp_reduce`` for
that since the sharding was written, and the model had no call site for it
until 2026-09-01 -- so every TP run ranked its own half of the head scores and
selected a different set of KV blocks on each rank.

It does not hang and it does not read as wrong: the two ranks still agree with
each other (they consume the same summed o_proj output), so the identity check
that mattered -- rank0 == rank1 -- kept passing.  What it breaks is agreement
with the *unsharded* model, which was attributed entirely to one-ULP all_sum
rounding.

``tp/validate.py`` did check that summed partials match the reference and did
not catch it, because it supplied the reduce itself as a capture hook.  These
tests pin the wiring rather than the arithmetic: that the forward CALLS the
reduce, how many times, and that the call count follows from S alone -- which
is what makes the two ranks issue the same number of collectives.
"""

import copy
import math

import mlx.core as mx
import pytest

from mlx_vlm.models.glm5_next.language import (Glm5NextSparseAttention,
                                               Glm5NextIndexer)
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.tp.glm5_next import shard_dsa
from mlx_vlm.tp.validate import TINY, _dsa_cache, _prime_arrays


def _cfg():
    return TextConfig.from_dict(TINY)


def _attn(cfg):
    mx.random.seed(0)
    a = Glm5NextSparseAttention(cfg)
    mx.eval(a.parameters())
    return a


def _run(attn, cfg, primed, S=1):
    """One indexer forward over a primed cache; returns the selected top-k."""
    x = mx.random.normal((1, S, cfg.hidden_size))
    qr = attn.q_a_layernorm(attn.q_a_proj(x))
    mask = mx.ones((1, 1, S, primed + S), dtype=mx.bool_)
    cache = _dsa_cache(cfg, primed, _prime_arrays(cfg, primed))
    return attn.indexer(x, qr, mask, cache=cache[1]), x, qr, mask


# ------------------------------------------------------------------- wiring
def test_unsharded_indexer_has_no_reduce_and_does_not_call_one():
    """TP off must be untouched: the hook exists and is None."""
    cfg = _cfg()
    a = _attn(cfg)
    assert a.indexer._tp_reduce is None
    assert a.indexer._scale_heads == cfg.index_n_heads
    _run(a, cfg, primed=4 * cfg.index_topk)     # must not raise


def test_sharding_installs_the_reduce_and_the_forward_actually_calls_it():
    """The regression.  Before the fix this count was zero, forever."""
    cfg = _cfg()
    primed = 4 * cfg.index_topk
    calls = []

    def reduce_fn(z):
        calls.append(z.shape)
        return z

    a = shard_dsa(_attn(cfg), 0, 2, reduce_fn)
    _run(a, cfg, primed=primed, S=1)
    assert calls, "the sharded indexer never invoked its all-reduce"


def test_reduce_is_called_once_per_query_chunk_so_both_ranks_agree():
    """The count must follow from S alone.

    Rank 0 announces S; if the number of reduces depended on anything else --
    the cache contents, the rank, the pool geometry -- the two ranks would
    issue different numbers of collectives and the next recv would pair with
    the wrong send.
    """
    cfg = _cfg()
    primed = 4 * cfg.index_topk
    for S in (1, 3, 512, 513, 1024, 1500):
        counts = []
        for rank in (0, 1):
            calls = []
            a = shard_dsa(_attn(cfg), rank, 2, lambda z: calls.append(1) or z)
            _run(a, cfg, primed=primed, S=S)
            counts.append(len(calls))
        assert counts[0] == counts[1], f"rank disagreement at S={S}: {counts}"
        assert counts[0] == math.ceil(S / 512) or S <= 512 and counts[0] == 1, (
            f"S={S} gave {counts[0]} reduces")


def test_scale_uses_the_global_head_count_after_sharding():
    """Otherwise summed partials come out sqrt(size) off the reference."""
    cfg = _cfg()
    a = shard_dsa(_attn(cfg), 0, 2, lambda z: z)
    assert a.indexer.n_heads == cfg.index_n_heads // 2
    assert a.indexer._scale_heads == cfg.index_n_heads


# --------------------------------------------------------------- arithmetic
def test_summed_partials_reproduce_the_unsharded_scores_and_topk():
    """With the reduce wired, TP selects the blocks the whole model selects.

    Run rank 0 and rank 1 with a capture hook, add their partials, and compare
    against the unsharded scores captured the same way.  This is the property
    tp/validate.py asserted; the point of repeating it here is that it now runs
    through the same call site production uses.
    """
    cfg = _cfg()
    primed = 4 * cfg.index_topk
    mx.random.seed(7)
    ref = _attn(cfg)
    x = mx.random.normal((1, 1, cfg.hidden_size))
    qr = ref.q_a_layernorm(ref.q_a_proj(x))
    mask = mx.ones((1, 1, 1, primed + 1), dtype=mx.bool_)

    def capture(store):
        def f(z):
            store.append(z)
            return z
        return f

    # ONE prime, shared by every arm.  _prime_arrays draws randomly, so a
    # fresh call per rank gives each rank a different cache and the partials
    # would be sums over different pools.
    prime = _prime_arrays(cfg, primed)

    ref_store = []
    ref.indexer._tp_reduce = capture(ref_store)
    ref.indexer(x, qr, mask, cache=_dsa_cache(cfg, primed, prime)[1])
    ref.indexer._tp_reduce = None
    assert ref_store, "reference capture did not fire"

    partials = []
    for rank in (0, 1):
        store = []
        m = shard_dsa(copy.deepcopy(ref), rank, 2, capture(store))
        qr_r = m.q_a_layernorm(m.q_a_proj(x))
        m.indexer(x, qr_r, mask, cache=_dsa_cache(cfg, primed, prime)[1])
        partials.append(store[0])

    total = partials[0] + partials[1]
    rel = float(mx.max(mx.abs(total - ref_store[0]))
                / mx.maximum(mx.max(mx.abs(ref_store[0])), 1e-6))
    assert rel < 5e-2, f"summed partials differ from the reference by {rel}"
    k = min(4, ref_store[0].shape[-1])
    assert bool(mx.all(mx.argsort(-total, axis=-1)[..., :k]
                       == mx.argsort(-ref_store[0], axis=-1)[..., :k]).item()), \
        "reduced TP top-k must match the unsharded top-k"


def test_a_partial_sum_ranks_differently_from_the_full_sum():
    """Why the missing reduce mattered: ranking half the heads is not ranking
    all of them.  If this ever stops being true the bug was cosmetic."""
    mx.random.seed(3)
    a = mx.random.normal((1, 1, 32))
    b = mx.random.normal((1, 1, 32))
    top_partial = mx.argsort(-a, axis=-1)[..., :4]
    top_full = mx.argsort(-(a + b), axis=-1)[..., :4]
    assert not bool(mx.all(top_partial == top_full).item())


# ------------------------------------------------- fast/slow path count parity
def _count_reduces(attn, cfg, cache, x, mask):
    calls = []
    attn.indexer._tp_reduce = lambda z: (calls.append(1), z)[1]
    qr = attn.q_a_layernorm(attn.q_a_proj(x))
    attn.indexer(x, qr, mask, cache=cache)
    return len(calls)


def test_fast_and_eager_decode_paths_issue_the_same_number_of_reduces(monkeypatch):
    """Count parity is a mirror invariant, not an optimisation detail.

    The indexer has two decode implementations: an incremental fast path gated
    on the cache carrying ``_no_pad`` and a pool, and the eager chunked path.
    A vault-restored cache has neither, so it is forced eager while a live
    cache is not -- meaning the two ranks CAN take different paths on the same
    logical step.  That is survivable only if both paths issue the same number
    of collectives.  When only the eager path had the reduce, they differed by
    one per DSA layer and the ranks deadlocked.
    """
    # MLX_VLM_GLM5_IDX_FAST is opt-in and off by default, so the fast path has
    # no live mileage yet -- which is why this parity bug was latent rather
    # than fatal.  Turn it on here so the test exercises what it claims to.
    import mlx_vlm.models.glm5_next.language as L

    monkeypatch.setenv("MLX_VLM_GLM5_IDX_FAST", "1")
    monkeypatch.setattr(L, "_IDX_FAST_ENV", None, raising=False)

    cfg = _cfg()
    a = _attn(cfg)
    primed = 4 * cfg.index_topk
    x = mx.random.normal((1, 1, cfg.hidden_size))

    def valid_prime():
        """A primed cache whose validity channel is actually valid.

        _prime_arrays fills every channel with normals, so the last one -- the
        validity flag -- comes out negative about half the time and _no_pad is
        False, which silently gates the fast path off.
        """
        lat, packed = _prime_arrays(cfg, primed)
        packed = mx.concatenate(
            [packed[..., :-1], mx.ones(packed.shape[:-1] + (1,))], axis=-1)
        return lat, packed

    # Eager: a cache with no derived state, exactly what a restore produces.
    eager_cache = _dsa_cache(cfg, primed, valid_prime())[1]
    n_eager = _count_reduces(a, cfg, eager_cache, x,
                             mx.ones((1, 1, 1, primed + 1), dtype=mx.bool_))

    # Fast: prime the same cache through one eager forward so _pool/_no_pad
    # exist, then step it again with mask=None, which is the fast path's gate.
    fast_cache = _dsa_cache(cfg, primed, valid_prime())[1]
    _count_reduces(a, cfg, fast_cache, x,
                   mx.ones((1, 1, 1, primed + 1), dtype=mx.bool_))
    entered = []
    real_fast = type(a.indexer)._decode_fast

    def spy(self, *args, **kw):
        entered.append(1)
        return real_fast(self, *args, **kw)

    type(a.indexer)._decode_fast = spy
    try:
        n_fast = _count_reduces(a, cfg, fast_cache, x, None)
    finally:
        type(a.indexer)._decode_fast = real_fast
    assert entered, ("the fast path was never taken, so this test has no "
                     "teeth -- check the gate in Glm5NextIndexer.__call__")

    assert n_eager == n_fast, (
        f"path-dependent collective count: eager={n_eager} fast={n_fast}. "
        "Two ranks taking different indexer paths would deadlock.")
    assert n_eager >= 1, "S=1 must issue exactly one indexer reduce"


def test_fast_path_scale_also_uses_the_global_head_count():
    cfg = _cfg()
    a = shard_dsa(_attn(cfg), 0, 2, lambda z: z)
    src = __import__("inspect").getsource(type(a.indexer)._decode_fast)
    assert "_scale_heads" in src and "self.n_heads**-0.5" not in src


# ----------------------------------------------------- lane-specific gate
def test_tp_raises_the_gather_gate_and_single_box_does_not(monkeypatch):
    """The gate is a per-LANE constant, and 32768 is the single-box answer.

    Measured per-chunk prefill: single-box dense 3924 + 252.8*chunk vs a gather
    plateau of 8670 ms -> crossover ~38k, which is what makes 32768 right there.
    TP splits the attention heads, so dense halves (122.9/chunk) while the
    gather plateau falls only to 6394 -> crossover ~68k.  Leaving 32768 in place
    under TP gives away 13.9% of prefill time to 65k.
    """
    from mlx_vlm.models.glm5_next import language as L
    from mlx_vlm.tp.glm5_next import _apply_tp_gather_default

    monkeypatch.delenv("MLX_VLM_GLM5_GATHER_MIN_CONTEXT", raising=False)
    monkeypatch.setattr(L, "_GATHER_MIN_CONTEXT", 32768, raising=False)
    _apply_tp_gather_default()
    assert L._GATHER_MIN_CONTEXT == 65536


def test_an_explicit_gate_beats_the_lane_default(monkeypatch):
    """An operator who set the variable meant it -- on either lane."""
    from mlx_vlm.models.glm5_next import language as L
    from mlx_vlm.tp.glm5_next import _apply_tp_gather_default

    monkeypatch.setenv("MLX_VLM_GLM5_GATHER_MIN_CONTEXT", "16384")
    monkeypatch.setattr(L, "_GATHER_MIN_CONTEXT", 16384, raising=False)
    _apply_tp_gather_default()
    assert L._GATHER_MIN_CONTEXT == 16384
