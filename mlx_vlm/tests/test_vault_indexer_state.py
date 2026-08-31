"""What the Warm Context Vault does and does not carry across a restore.

``KVCacheCloneAdapter.clone`` builds a fresh ``KVCache`` and copies exactly
``keys``, ``values`` and ``offset``.  The DSA layer's cache is
``CacheList(KVCache, KVCache)`` and its second entry is the *indexer* cache,
which carries three derived attributes the model sets on it during a forward:
``_pool`` (the compressed block pool), ``_fpool``, and ``_no_pad``.

None of them survive a capture/restore.  These tests pin what that costs, and
-- more importantly -- whether it changes any answer, because a vault that
returns a differently-behaving cache is a correctness bug and not a
performance note.  Written while investigating a post-restore stall under TP;
they do not explain that stall, and they check a property that matters
single-box too.
"""

import mlx.core as mx
import pytest

from mlx_vlm.context_vault import capture_fragments, restore_fragments
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import Glm5NextSparseAttention
from mlx_vlm.tp.validate import TINY, _dsa_cache, _prime_arrays


def _cfg():
    return TextConfig.from_dict(TINY)


def _attn(cfg):
    mx.random.seed(0)
    a = Glm5NextSparseAttention(cfg)
    mx.eval(a.parameters())
    return a


def _prime_live(cfg, attn, primed):
    """A cache that has actually been through the indexer, so _pool is set."""
    cache = _dsa_cache(cfg, primed, _prime_arrays(cfg, primed))
    x = mx.random.normal((1, 1, cfg.hidden_size))
    qr = attn.q_a_layernorm(attn.q_a_proj(x))
    mask = mx.ones((1, 1, 1, primed + 1), dtype=mx.bool_)
    attn.indexer(x, qr, mask, cache=cache[1])
    return cache


def test_a_live_indexer_cache_carries_derived_state():
    cfg = _cfg()
    a = _attn(cfg)
    cache = _prime_live(cfg, a, primed=4 * cfg.index_topk)
    assert getattr(cache[1], "_pool", None) is not None
    assert getattr(cache[1], "_no_pad", None) is not None


def test_the_vault_drops_that_derived_state():
    """Documented, not desired: the clone adapter copies keys/values/offset."""
    cfg = _cfg()
    a = _attn(cfg)
    primed = 4 * cfg.index_topk
    live = _prime_live(cfg, a, primed)

    frags = capture_fragments([live], primed)
    assert frags is not None, "DSA cache must be capturable at all"
    fresh = _dsa_cache(cfg, 0)
    assert restore_fragments([fresh], frags)

    assert getattr(fresh[1], "_pool", None) is None, (
        "if the pool now survives, delete this test and the slow-path note")
    assert getattr(fresh[1], "_no_pad", None) is None


def test_restore_preserves_the_kv_that_matters():
    cfg = _cfg()
    a = _attn(cfg)
    primed = 4 * cfg.index_topk
    live = _prime_live(cfg, a, primed)
    frags = capture_fragments([live], primed)
    fresh = _dsa_cache(cfg, 0)
    restore_fragments([fresh], frags)
    for i in (0, 1):
        assert int(fresh[i].offset) == int(live[i].offset)
        assert bool(mx.all(fresh[i].keys[..., :primed, :]
                           == live[i].keys[..., :primed, :]).item())


def test_a_restored_cache_selects_the_same_blocks_as_a_live_one():
    """THE gate: dropping derived state must not change any answer.

    The pool is a cache of pooled block states; rebuilding it from the restored
    keys has to reproduce it, or the vault is silently returning a different
    model.  ``_no_pad`` only selects a fast path, so it may cost time -- it may
    not cost agreement.
    """
    cfg = _cfg()
    a = _attn(cfg)
    primed = 4 * cfg.index_topk
    live = _prime_live(cfg, a, primed)

    frags = capture_fragments([live], primed)
    fresh = _dsa_cache(cfg, 0)
    restore_fragments([fresh], frags)

    mx.random.seed(11)
    x = mx.random.normal((1, 1, cfg.hidden_size))
    qr = a.q_a_layernorm(a.q_a_proj(x))
    mask = mx.ones((1, 1, 1, primed + 1), dtype=mx.bool_)

    from copy import deepcopy
    topk_live = a.indexer(x, qr, mask, cache=deepcopy(live)[1])
    topk_restored = a.indexer(x, qr, mask, cache=fresh[1])

    assert topk_live.shape == topk_restored.shape
    assert bool(mx.all(topk_live == topk_restored).item()), (
        "a restored cache selected different KV blocks than the live one -- "
        "the vault is not state-preserving for glm5_next DSA layers")
