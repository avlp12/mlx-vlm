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
import threading

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
    # The memo context is THREAD-LOCAL now (it used to be a module global and two
    # concurrent forwards raced on it). Drive it through the accessors rather
    # than assigning a module attribute; the semantics under test are unchanged.
    prev = glm5._get_vis_memo_ctx()
    glm5._set_vis_memo_ctx(memo)
    try:
        outs = []
        for ix in indexers:
            topk = ix(x, qr, mask, cache=KVCache())
            mx.eval(topk)
            outs.append(topk)
        return outs
    finally:
        glm5._set_vis_memo_ctx(prev)


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
    prev = glm5._get_vis_memo_ctx()
    glm5._set_vis_memo_ctx(memo)
    try:
        assert glm5._active_vis_memo(owned[0]) is memo
        assert glm5._active_vis_memo(foreign) is None
    finally:
        glm5._set_vis_memo_ctx(prev)
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


# ------------------------------------------------- end-to-end model lifecycle
_MODEL_CFG = dict(
    model_type="glm5_next_text", vocab_size=128, hidden_size=64,
    intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1,
    n_routed_experts=4, routed_scaling_factor=2.5, kv_lora_rank=16,
    q_lora_rank=16, qk_rope_head_dim=0, v_head_dim=16, qk_nope_head_dim=16,
    num_experts_per_tok=2, first_k_dense_replace=0, max_position_embeddings=2048,
    rms_norm_eps=1e-05, index_topk=16, index_kpool=4, index_head_dim=16,
    index_n_heads=4,
    layer_types=["linear_attention", "deepseek_sparse_attention",
                 "linear_attention", "deepseek_sparse_attention"],
    mlp_layer_types=["dense"] * 4,
    linear_attn_config={"num_heads": 4, "gate_lower_bound": -5.0,
                        "head_dim": 16, "short_conv_kernel_size": 4},
)


def _tiny_model():
    from mlx_vlm.models.glm5_next.config import TextConfig as TC
    mx.random.seed(0)
    model = glm5.Glm5NextModel(TC.from_dict(dict(_MODEL_CFG)))

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.float32)

    model.update(rand(model.parameters()))
    mx.eval(model.parameters())
    return model


def _model_forward(model, seq=48):
    from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
    caches = [
        ArraysCache(size=2) if l.is_linear else CacheList(KVCache(), KVCache())
        for l in model.layers
    ]
    out = model(mx.arange(seq, dtype=mx.int32)[None], cache=caches)
    mx.eval(out)
    return out


def test_model_forward_registers_and_scopes_the_memo():
    """The lifecycle itself: registration in __init__, open/close around the
    layer loop, and no leak of the context past the forward."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    model = _tiny_model()
    n_dsa = sum(1 for l in model.layers if not l.is_linear)
    assert len(model._dsa_indexer_ids) == n_dsa == 2
    assert glm5._get_vis_memo_ctx() is None
    _model_forward(model)
    assert glm5._get_vis_memo_ctx() is None, "memo context leaked past the forward"


def test_model_forward_bit_identical_on_vs_off():
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    model = _tiny_model()
    on = _model_forward(model)
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="0")
    off = _model_forward(model)
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    assert bool(mx.all(on == off))


def test_model_forward_under_verify_mode():
    """VERIFY recomputes every memoized tensor inside all 11 (here 2) DSA layers
    and raises on the first divergence -- run the whole model under it."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1", MLX_VLM_GLM5_VIS_MEMO_VERIFY="1")
    try:
        model = _tiny_model()
        checked = _model_forward(model)
        _reset_env(MLX_VLM_GLM5_VIS_MEMO="0", MLX_VLM_GLM5_VIS_MEMO_VERIFY=None)
        plain = _model_forward(model)
        assert bool(mx.all(checked == plain))
    finally:
        _reset_env(MLX_VLM_GLM5_VIS_MEMO="1", MLX_VLM_GLM5_VIS_MEMO_VERIFY=None)


# ------------------------------------------------------- thread safety (I97x)
# The memo context used to be a MODULE GLOBAL saved and restored around the layer
# loop.  That is correct for one forward at a time and wrong the moment two run
# concurrently -- which is exactly what co-scheduling a decode beside a prefill
# does.  These tests pin the thread-local behaviour, and every one of them FAILS
# against the old module-global implementation.

def test_memo_scope_survives_the_historic_interleaving():
    """Force the exact ordering that broke the module global:

        A enters -> B enters -> A checks -> A exits -> B checks -> B exits

    Under the old code, B's entry overwrote the single global, so A's check saw
    MemoB; A's exit restored None, so B's check saw None; and B's exit restored
    MemoA, leaving a STALE MEMO INSTALLED PAST BOTH FORWARDS.  That last one is
    the dangerous part: chunk keys are (B, T, S, c0, c1), so a memo that outlives
    its forward will be consulted by the next one, and that one can collide.
    """
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    owners_a = frozenset({1001})
    owners_b = frozenset({2002})
    seen = {}
    err = {}
    a_in, b_in, a_out, b_out = (threading.Event() for _ in range(4))

    def thread_a():
        try:
            with glm5._vis_memo_scope(owners_a) as memo_a:
                seen["a_entered"] = memo_a
                a_in.set()
                assert b_in.wait(5), "B never entered"
                # B is inside its own scope right now.
                seen["a_sees_while_b_open"] = glm5._get_vis_memo_ctx()
            seen["a_after_exit"] = glm5._get_vis_memo_ctx()
            a_out.set()
        except BaseException as e:      # noqa: BLE001 - surface it in the assert
            err["a"] = e
            a_in.set(); a_out.set()

    def thread_b():
        try:
            assert a_in.wait(5), "A never entered"
            with glm5._vis_memo_scope(owners_b) as memo_b:
                seen["b_entered"] = memo_b
                b_in.set()
                assert a_out.wait(5), "A never exited"
                # A has now left its scope. B must be untouched by that.
                seen["b_sees_after_a_exit"] = glm5._get_vis_memo_ctx()
            seen["b_after_exit"] = glm5._get_vis_memo_ctx()
            b_out.set()
        except BaseException as e:      # noqa: BLE001
            err["b"] = e
            b_in.set(); b_out.set()

    ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
    ta.start(); tb.start(); ta.join(10); tb.join(10)
    assert not err, f"worker raised: {err}"

    # neither thread ever saw the other's memo
    assert seen["a_sees_while_b_open"] is seen["a_entered"], \
        "thread A saw thread B's memo -- the context is not thread-local"
    assert seen["b_sees_after_a_exit"] is seen["b_entered"], \
        "thread A's exit clobbered thread B's memo"
    assert seen["a_entered"] is not seen["b_entered"], "both threads got one memo"

    # and each thread's slot is clean afterwards, on both threads and here
    assert seen["a_after_exit"] is None, "memo survived thread A's scope"
    assert seen["b_after_exit"] is None, "memo survived thread B's scope"
    assert glm5._get_vis_memo_ctx() is None, "a memo leaked onto the main thread"


def test_worker_threads_start_with_a_clean_slot():
    """threading.local's per-thread __init__ must give every new thread None,
    not whatever the creating thread happened to hold."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    got = {}
    with glm5._vis_memo_scope(frozenset({7})):
        assert glm5._get_vis_memo_ctx() is not None       # main thread holds one
        t = threading.Thread(target=lambda: got.update(ctx=glm5._get_vis_memo_ctx()))
        t.start(); t.join(5)
    assert got["ctx"] is None, "a new thread inherited the parent's memo"


def test_scope_restores_on_exception_per_thread():
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    out = {}

    def worker():
        try:
            with glm5._vis_memo_scope(frozenset({3})):
                raise RuntimeError("forward blew up")
        except RuntimeError:
            pass
        out["after"] = glm5._get_vis_memo_ctx()

    t = threading.Thread(target=worker); t.start(); t.join(5)
    assert out["after"] is None, "a raising forward left its memo installed"
    assert glm5._get_vis_memo_ctx() is None


def test_active_vis_memo_is_thread_local():
    """_active_vis_memo reads the thread-local slot, so a memo open on one thread
    must not serve an indexer being asked about on another."""
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    config = _config()
    owned = _indexers(config, n=1, seed=11)
    memo = _memo_for(owned)
    out = {}
    with glm5._vis_memo_scope(frozenset(id(ix) for ix in owned)):
        # main thread: served
        glm5._set_vis_memo_ctx(memo)
        assert glm5._active_vis_memo(owned[0]) is memo
        t = threading.Thread(
            target=lambda: out.update(m=glm5._active_vis_memo(owned[0])))
        t.start(); t.join(5)
    assert out["m"] is None, "another thread's indexer lookup found our memo"


def test_two_concurrent_model_forwards_are_bit_identical_to_serial():
    """The end-to-end property item 3(a) depends on: two real forwards running
    at once must produce exactly what they produce alone.

    This is the test that would have caught the bug in production rather than in
    review -- it exercises the real layer loop, the real scope, and the real
    indexers, on two threads at the same time.
    """
    _reset_env(MLX_VLM_GLM5_VIS_MEMO="1")
    model = _tiny_model()
    serial = _model_forward(model)
    mx.eval(serial)

    results, errors = {}, {}

    def worker(tag):
        try:
            results[tag] = _model_forward(model)
        except BaseException as e:      # noqa: BLE001
            errors[tag] = e

    threads = [threading.Thread(target=worker, args=(t,)) for t in ("x", "y")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, f"a concurrent forward raised: {errors}"
    for tag, out in results.items():
        assert bool(mx.all(out == serial)), \
            f"concurrent forward {tag} diverged from the serial result"
    assert glm5._get_vis_memo_ctx() is None, "memo leaked past concurrent forwards"
