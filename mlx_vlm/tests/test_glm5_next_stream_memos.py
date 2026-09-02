"""Per-stream keying of glm5_next's shape-keyed module memos.

Three per-MODULE memos are keyed by shape and so cannot be shared by two
concurrently-constructed streams:

    Glm5NextDecoderLayer._ffn_c       compiled FFN block      (per layer)
    Glm5NextDecoderLayer._attn_pre_c  compiled attn prologue  (per layer)
    <DSA indexer>._ptc                pool-tail constants     (per indexer)

The pool buffers are deliberately NOT in that list -- they live on the cache
(cache._pool / cache._fpool), which is already per-request.

What these tests pin:
  * single-stream behaviour is BIT-IDENTICAL to before, and uses exactly one
    memo entry (the change must be free for everyone who does not need it)
  * two streams at the SAME shape get separate entries and identical results
  * two streams at MIXED shapes get separate entries and identical results --
    this is the case that would silently share a trace without the keying
  * the scope restores on exception
"""
import os
import threading

import mlx.core as mx
import pytest

import mlx_vlm.models.glm5_next.language as glm5

# the sibling test module holds the fixtures; pytest runs these files as
# top-level modules (no package __init__), so import it by plain name
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from test_glm5_next_vis_memo import (  # reuse the established fixtures
    _config,
    _indexers,
    _inputs,
    _model_forward,
    _reset_env,
    _tiny_model,
)


def _layers(model):
    return list(model.layers)


def _decode_forward(model, prefill=32, steps=2):
    """A DECODE-shaped forward: B=1 with S<=8, which is the only shape that
    engages the compiled FFN/attn-prologue path (language.py: `compile_ffn and
    x.shape[0] == 1 and x.shape[1] <= 8`). A seq=48 prefill leaves _ffn_c EMPTY,
    so a test that only prefills asserts nothing about these memos -- the first
    version of this file did exactly that and passed vacuously."""
    from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
    caches = [ArraysCache(size=2) if l.is_linear else CacheList(KVCache(), KVCache())
              for l in model.layers]
    out = model(mx.arange(prefill, dtype=mx.int32)[None], cache=caches)
    mx.eval(out)
    tok = mx.argmax(out[:, -1, :], axis=-1)[:, None].astype(mx.int32)
    for _ in range(steps):
        out = model(tok, cache=caches)
        mx.eval(out)
    return out


def _memo_sizes(model):
    ffn = sum(len(l._ffn_c) for l in _layers(model))
    attn = sum(len(l._attn_pre_c) for l in _layers(model))
    return ffn, attn


# ---------------------------------------------------------------- key plumbing
def test_default_key_and_scope_nesting():
    assert glm5.stream_memo_key() == "default"
    with glm5.stream_memo_scope("a"):
        assert glm5.stream_memo_key() == "a"
        with glm5.stream_memo_scope("b"):
            assert glm5.stream_memo_key() == "b"
        assert glm5.stream_memo_key() == "a"
    assert glm5.stream_memo_key() == "default"


def test_scope_restores_on_exception():
    with pytest.raises(RuntimeError):
        with glm5.stream_memo_scope("boom"):
            raise RuntimeError("x")
    assert glm5.stream_memo_key() == "default"


def test_key_is_thread_local():
    """The driver is single-threaded by design (playbook law 12), but the key
    must not leak between threads if anything ever runs two."""
    seen = {}
    with glm5.stream_memo_scope("main"):
        t = threading.Thread(target=lambda: seen.update(k=glm5.stream_memo_key()))
        t.start()
        t.join(5)
    assert seen["k"] == "default", "a worker inherited the parent's memo key"


# ------------------------------------------------------- single stream is free
def test_single_stream_uses_exactly_one_entry_and_is_unchanged():
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    model = _tiny_model()
    a = _decode_forward(model)
    b = _decode_forward(model)
    assert bool(mx.all(a == b))
    n = len(_layers(model))
    ffn, _attn = _memo_sizes(model)
    # THE PATH MUST ACTUALLY HAVE BEEN TAKEN, or this test proves nothing.
    assert ffn == n, f"compiled FFN memo not populated ({ffn}/{n}) -- test is vacuous"
    for l in _layers(model):
        assert set(l._ffn_c) == {"default"}, set(l._ffn_c)


# --------------------------------------------------- same shape, two stream keys
def test_two_stream_keys_same_shape_are_separate_and_identical():
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    model = _tiny_model()
    with glm5.stream_memo_scope("s0"):
        out0 = _decode_forward(model)
    with glm5.stream_memo_scope("s1"):
        out1 = _decode_forward(model)
    mx.eval(out0, out1)
    assert bool(mx.all(out0 == out1)), "same shape on two keys diverged"
    n = len(_layers(model))
    for l in _layers(model):
        assert set(l._ffn_c) == {"s0", "s1"}, set(l._ffn_c)
    # not vacuous: two keys really did produce two entries per layer
    assert sum(len(l._ffn_c) for l in _layers(model)) == 2 * n


# -------------------------------------------------- MIXED shape, two stream keys
def test_two_stream_keys_mixed_shape_are_separate_and_identical():
    """The case the keying exists for. Two streams at DIFFERENT sequence lengths
    would, unkeyed, contend for one shape-keyed slot per layer."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    model = _tiny_model()
    ref_a = _decode_forward(model, prefill=32, steps=2)
    ref_b = _decode_forward(model, prefill=16, steps=3)
    mx.eval(ref_a, ref_b)

    with glm5.stream_memo_scope("A"):
        got_a = _decode_forward(model, prefill=32, steps=2)
    with glm5.stream_memo_scope("B"):
        got_b = _decode_forward(model, prefill=16, steps=3)
    mx.eval(got_a, got_b)

    assert bool(mx.all(got_a == ref_a)), "mixed-shape stream A diverged from serial"
    assert bool(mx.all(got_b == ref_b)), "mixed-shape stream B diverged from serial"


# ---------------------------------------------------------------- indexer _ptc
def test_indexer_ptc_is_keyed_by_stream():
    """_ptc holds arrays built on whichever stream first reached it, so its key
    must carry the stream."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    config = _config()
    ix = _indexers(config, n=1, seed=3)[0]
    ix._ptc = None
    with glm5.stream_memo_scope("q"):
        assert glm5.stream_memo_key() == "q"
    # the key tuple's first element must be the stream key, so an entry built
    # under one stream cannot satisfy a lookup under another
    ix._ptc = (("q", 4, 2, 0), None, None, None, None)
    with glm5.stream_memo_scope("r"):
        cached = ix._ptc
        assert cached[0][0] == "q"
        assert cached[0] != (glm5.stream_memo_key(), 4, 2, 0), \
            "an entry built under stream q would be reused under stream r"
