"""Fused KDA for the S>1 PREFILL chunk, against the eager path it replaces.

The claim under test is BIT-EXACTNESS, not closeness.  Every arithmetic line in
``fused_kda_prefill`` is copied from ``fused_kda._BLOCK_SOURCE``, which
``test_glm5_next_fused_kda_block.py`` already pins bit-identical to the eager
chain, and the value-axis split this kernel adds is partition-preserving: lane
``l`` still owns key elements ``[NDK*l, NDK*l+NDK)``, the two L2 norms and the
RMS norm still use MLX's row_reduce partition, and no reduction changes operand
order with ``NV``.  So a tolerance would only hide a bug; ``atol = rtol = 0``.

What the eager S>1 baseline actually is matters for reading these tests: its
recurrence was ALREADY a single fused dispatch (``gated_delta_update`` ->
``gated_delta_kernel``, a per-token scan in Metal).  What these tests check is
that folding the glue -- conv window, silu, two fp32 L2 norms, the beta sigmoid,
the hand-rolled gated RMSNorm -- into that scan, and re-cutting the scan's
thread partition across ``NV`` threadgroups, changes nothing.

Device split.  The kernel tests need a real Metal dispatch and are skipped
unless the default device is the GPU (``MLX_DEFAULT_DEVICE`` unset or ``gpu``;
conftest pins it).  The Python-side dispatch tests -- geometry, launch count,
shapes, dtypes, which tensors reach which launch, and every branch of the
eligibility predicate -- run on CPU against a recording fake kernel, so the
half of this file that can fail without a GPU does.
"""
import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.glm5_next.fused_kda_prefill as F
import mlx_vlm.models.glm5_next.language as glm5
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.glm5_next.config import TextConfig

# GLM-5.3-Flash text_config restricted to what the KDA layer reads, with the
# head count cut from 64 to 4 so a 2048-token chunk is a test and not a
# benchmark.  head_dim 128, conv kernel 4 and gate_lower_bound -5.0 are the
# shipped values verbatim: they set NDK, the conv window and the gate branch.
_CFG = dict(
    model_type="glm5_next_text",
    vocab_size=1024,
    hidden_size=512,
    intermediate_size=1024,
    moe_intermediate_size=512,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=8,
    n_shared_experts=1,
    n_routed_experts=8,
    routed_scaling_factor=2.5,
    kv_lora_rank=128,
    q_lora_rank=256,
    qk_rope_head_dim=0,
    v_head_dim=64,
    qk_nope_head_dim=64,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-05,
    index_topk=64,
    index_head_dim=128,
    index_n_heads=8,
    layer_types=["linear_attention"],
    mlp_layer_types=["dense"],
    linear_attn_config={
        "num_heads": 4,
        "gate_lower_bound": -5.0,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
    },
)

H, D, K = 4, 128, 4
HD = H * D

on_gpu = pytest.mark.skipif(
    not mx.metal.is_available()
    or mx.default_device() != mx.gpu
    or F._kernel("fused") is None,
    reason="needs a Metal GPU as the default device",
)


def _config():
    return TextConfig.from_dict(dict(_CFG))


def _layer(config, seed=0, quantize=True):
    mx.random.seed(seed)
    layer = glm5.Glm5NextLinearAttention(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    layer.update(rand(layer.parameters()))
    layer.conv1d.weight = (mx.random.normal(layer.conv1d.weight.shape) * 0.5).astype(
        mx.bfloat16
    )
    # A_log / dt_bias are kept in fp32 by the converter's cast_predicate.
    layer.forget_gate.A_log = (mx.random.normal((H,)) * 0.5).astype(mx.float32)
    layer.forget_gate.dt_bias = (mx.random.normal((H * D,)) * 0.5).astype(mx.float32)
    layer.o_norm.weight = (mx.ones((D,)) + 0.02 * mx.random.normal((D,))).astype(
        mx.bfloat16
    )
    if quantize:  # the live build quantises the KDA projections to 8-bit, group 64
        nn.quantize(layer, group_size=64, bits=8)
    mx.eval(layer.parameters())
    return layer


def _cache(batch=1, seed=1):
    mx.random.seed(seed)
    cache = ArraysCache(size=2)
    # A prefill chunk past the first never sees a zero-initialised cache, and the
    # zero case would hide a state-propagation bug, so warm both.
    cache[0] = (mx.random.normal((batch, K - 1, 3 * HD)) * 0.3).astype(mx.bfloat16)
    cache[1] = (mx.random.normal((batch, H, D, D)) * 0.05).astype(mx.float32)
    mx.eval(cache[0], cache[1])
    return cache


def _clone(cache):
    out = ArraysCache(size=2)
    out[0], out[1] = cache[0], cache[1]
    return out


def _run(layer, x, cache, *, on, nv=None, mask=None):
    """One forward with the prefill kernel forced on or off, everything else equal."""
    prev_env, prev_nv = glm5._FUSED_KDA_PREFILL_ENV, glm5._FUSED_KDA_PREFILL_NV
    prev_geom, prev_ready = layer._fused_kda_prefill_geom, layer._fused_kda_prefill
    try:
        glm5._FUSED_KDA_PREFILL_ENV = bool(on)
        glm5._FUSED_KDA_PREFILL_NV = "" if nv is None else str(nv)
        layer._fused_kda_prefill = None       # re-probe under the new geometry
        layer._fused_kda_prefill_geom = None
        y = layer(x, mask=mask, cache=cache)
        mx.eval(y, cache[0], cache[1])
        return y
    finally:
        glm5._FUSED_KDA_PREFILL_ENV = prev_env
        glm5._FUSED_KDA_PREFILL_NV = prev_nv
        layer._fused_kda_prefill_geom = prev_geom
        layer._fused_kda_prefill = prev_ready


# ===========================================================================
# Kernel parity.  GPU only.
# ===========================================================================
# S values, and why each one is here:
#   2    the narrowest chunk, and S < K-1 so the conv window tail MIXES cached
#        rows with new tokens -- the epilogue's circular-buffer indexing is the
#        only thing that makes that case correct without a special case;
#   5    S > K-1 by a margin that is not a multiple of anything;
#   64   past the point where the t-loop's register allocation is what it will
#        be at 8192, but still cheap;
#   300  not a multiple of K-1, TY, or NV -- catches an index that assumed one;
#   2048 a quarter of the shipped 8192 prefill chunk, i.e. the same order of
#        loop trip count, at 1/16 of the heads.
_S_VALUES = [2, 5, 64, 300, 2048]


@on_gpu
@pytest.mark.parametrize("S", _S_VALUES)
@pytest.mark.parametrize("nv", [None, 1])
def test_prefill_is_bit_identical_to_eager(S, nv):
    """Output, recurrent state and conv window, all exact, at both geometries.

    ``nv=None`` is the default value-split geometry (NV = D/TY = 4 here);
    ``nv=1`` is the maximally fused one-launch arm.  Both must agree with eager
    AND therefore with each other.
    """
    config = _config()
    layer = _layer(config, seed=S)
    mx.random.seed(1000 + S)
    x = (mx.random.normal((1, S, config.hidden_size)) * 0.5).astype(mx.bfloat16)
    c_eager, c_fused = _cache(seed=S), _cache(seed=S)

    y_e = _run(layer, x, c_eager, on=False)
    y_f = _run(layer, x, c_fused, on=True, nv=nv)

    assert y_f.shape == y_e.shape == (1, S, config.hidden_size)
    assert mx.array_equal(y_f, y_e).item(), f"output differs at S={S}, nv={nv}"
    assert mx.array_equal(c_fused[1], c_eager[1]).item(), "final state differs"
    assert mx.array_equal(c_fused[0], c_eager[0]).item(), "conv window differs"
    # ArraysCache has no ``.offset`` (it is not a KVCache): the KDA layer keeps
    # no step counter of its own, only the conv window and recurrent state
    # slots returned by ``.state``. Compare the whole state list bit-exactly
    # so a slot added to ArraysCache later is covered by this test too.
    assert all(
        mx.array_equal(a, b).item()
        for a, b in zip(c_fused.state, c_eager.state)
    ), "full cache state differs"


@on_gpu
def test_state_carries_across_chunks():
    """Two 64-token chunks through one cache must match EAGER over the SAME
    two chunks -- not the fused kernel run whole in one call.

    Fused-split-vs-fused-whole bit-equality is the wrong contract: the eager
    KDA recurrence itself is not bit-exact across a chunk split on Metal (only
    on CPU do the two decompositions agree bitwise at these shapes -- see the
    ``ON_GPU`` branch of ``test_glm5_chunked_spec_prefill.py::_assert_row_matches``
    and its ``CACHE_DRIFT_TOL = 2e-06``, worst measured 9.54e-07 on an M3 Ultra,
    mlx 0.32.1.dev20260902, splitting this same KDA scan at a chunk boundary).
    Demanding fused-split == fused-whole bit-exactly would therefore fail on a
    CORRECT kernel, for the same float non-associativity reason, not a bug.

    So the load-bearing check here is: fused split == eager split, run chunk
    for chunk with fresh same-seed caches, bit-exact (this kernel's actual
    claim -- see the module docstring -- is parity with the eager chain it
    replaces, at whatever chunking the caller uses). Fused-split-vs-fused-whole
    is kept only as a loose secondary guard against a gross chunk-carry bug
    (state or window dropped between chunks), at the same drift bound the
    sibling file established for this exact phenomenon -- confirmed by direct
    measurement (see below) to be exactly 0 on CPU at this config, i.e. this
    tolerance is inert here and only matters on the Metal box this test
    actually runs on.
    """
    config = _config()
    layer = _layer(config, seed=7)
    mx.random.seed(77)
    x = (mx.random.normal((1, 128, config.hidden_size)) * 0.5).astype(mx.bfloat16)
    whole, split, eager_split = _cache(seed=7), _cache(seed=7), _cache(seed=7)

    y_whole = _run(layer, x, whole, on=True)
    y_a = _run(layer, x[:, :64], split, on=True)
    y_b = _run(layer, x[:, 64:], split, on=True)
    y_ea = _run(layer, x[:, :64], eager_split, on=False)
    y_eb = _run(layer, x[:, 64:], eager_split, on=False)

    y_split = mx.concatenate([y_a, y_b], axis=1)
    y_eager_split = mx.concatenate([y_ea, y_eb], axis=1)

    # The module's actual claim: fused matches eager under IDENTICAL chunking,
    # bit-exactly (atol = rtol = 0, per the module docstring).
    assert mx.array_equal(y_split, y_eager_split).item(), (
        "fused split differs from eager split"
    )
    assert all(
        mx.array_equal(a, b).item() for a, b in zip(split.state, eager_split.state)
    ), "fused cache state after two chunks differs from eager cache state"

    # Secondary, loose guard: fused-split vs fused-whole should be close even
    # though it is not the load-bearing contract above. ``KDA_SPLIT_DRIFT_TOL``
    # is the sibling test file's measured Metal bound for splitting this same
    # scan at a chunk boundary (test_glm5_chunked_spec_prefill.py), not a value
    # invented here; measured directly against THIS test's own config on CPU it
    # is 0 (bitwise), consistent with that file's note that CPU does not show
    # the drift Metal does -- so this assertion is a no-op safety net on CPU
    # and only exercises the tolerance on the GPU box.
    KDA_SPLIT_DRIFT_TOL = 2e-06
    assert mx.allclose(
        y_split.astype(mx.float32),
        y_whole.astype(mx.float32),
        atol=KDA_SPLIT_DRIFT_TOL,
        rtol=0,
    ).item(), "fused split vs fused whole exceeded the KDA scan-split drift bound"


@on_gpu
@pytest.mark.parametrize("S", [5, 300])
def test_masked_tokens_match_the_eager_zeroing(S):
    """The eager path zeroes the PRE-conv input of a masked token; so must the
    kernel, before both the conv and the window write."""
    config = _config()
    layer = _layer(config, seed=11)
    mx.random.seed(110 + S)
    x = (mx.random.normal((1, S, config.hidden_size)) * 0.5).astype(mx.bfloat16)
    mask = mx.random.uniform(shape=(1, S)) > 0.25
    c_e, c_f = _cache(seed=11), _cache(seed=11)
    y_e = _run(layer, x, c_e, on=False, mask=mask)
    y_f = _run(layer, x, c_f, on=True, mask=mask)
    assert mx.array_equal(y_f, y_e).item()
    assert mx.array_equal(c_f[1], c_e[1]).item()
    assert mx.array_equal(c_f[0], c_e[0]).item()


@on_gpu
def test_first_token_argmax_is_unchanged_end_to_end():
    """Prefill a prompt through a whole (tiny) LanguageModel and compare the
    first sampled token, not just the layer output.

    Layer-level bit-exactness already implies this, so the test earns its place
    only as the wiring check: that the eligibility predicate actually fires in
    the model's own call path, and that nothing downstream (o_proj shape, cache
    offset) was left inconsistent.
    """
    from mlx_vlm.models.glm5_next.language import LanguageModel

    config = _config()
    mx.random.seed(5)
    model = LanguageModel(config)

    def rand(tree):
        if isinstance(tree, dict):
            return {k: rand(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rand(v) for v in tree]
        return (mx.random.normal(tree.shape) * 0.05).astype(mx.bfloat16)

    model.update(rand(model.parameters()))
    for layer in model.model.layers:
        sa = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
        if isinstance(sa, glm5.Glm5NextLinearAttention):
            sa.forget_gate.A_log = (mx.random.normal((H,)) * 0.5).astype(mx.float32)
            sa.forget_gate.dt_bias = (mx.random.normal((H * D,)) * 0.5).astype(
                mx.float32
            )
    mx.eval(model.parameters())
    prompt = mx.array([[3, 17, 42, 8, 99, 5, 61, 7] * 8])

    outs = []
    for on in (False, True):
        cache = model.make_cache()
        prev = glm5._FUSED_KDA_PREFILL_ENV
        try:
            glm5._FUSED_KDA_PREFILL_ENV = on
            for layer in model.model.layers:
                sa = getattr(layer, "linear_attn", None)
                if isinstance(sa, glm5.Glm5NextLinearAttention):
                    sa._fused_kda_prefill = None
                    sa._fused_kda_prefill_geom = None
            logits = model(prompt, cache=cache).logits[:, -1, :]
            mx.eval(logits)
            outs.append(logits)
        finally:
            glm5._FUSED_KDA_PREFILL_ENV = prev
    assert mx.argmax(outs[0], axis=-1).item() == mx.argmax(outs[1], axis=-1).item()
    assert mx.array_equal(outs[0], outs[1]).item(), "prefill logits are not exact"


# ===========================================================================
# Python-side dispatch.  CPU-runnable: a recording fake kernel stands in for
# the Metal one, so geometry, launch count, shapes, dtypes and which tensors
# reach which launch are all checked without a GPU.
# ===========================================================================
class _FakeKernel:
    """Records the call and returns correctly shaped zeros."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __call__(self, *, inputs, template, grid, threadgroup, output_shapes,
                 output_dtypes):
        self.log.append(
            dict(name=self.name, inputs=inputs, template=dict(template), grid=grid,
                 threadgroup=threadgroup, shapes=list(output_shapes),
                 dtypes=list(output_dtypes))
        )
        return [mx.zeros(s, dtype=d) for s, d in zip(output_shapes, output_dtypes)]


@pytest.fixture
def fake_kernels(monkeypatch):
    log = []
    monkeypatch.setattr(F, "_KERNEL_TRIED", True)
    monkeypatch.setattr(
        F,
        "_KERNELS",
        {k: _FakeKernel(k, log) for k in ("fused", "split", "norm")},
    )
    return log


def _args(B=1, S=7, dt=mx.bfloat16):
    r = lambda *sh: mx.zeros(sh, dt)  # noqa: E731
    return dict(
        q_in=r(B, S, HD), k_in=r(B, S, HD), v_in=r(B, S, HD),
        conv_state=r(B, K - 1, 3 * HD), conv_w=r(3 * HD, K, 1),
        a=r(B, S, HD), b=r(B, S, H),
        A_log=mx.zeros((H,), mx.float32), dt_bias=mx.zeros((H * D,), mx.float32),
        state=mx.zeros((B, H, D, D), mx.float32),
        gate=r(B, S, HD), o_weight=r(D),
    )


def _call(nv, ty=32, **over):
    a = _args(**over)
    return F.fused_kda_prefill(
        a["q_in"], a["k_in"], a["v_in"], a["conv_state"], a["conv_w"], a["a"],
        a["b"], a["A_log"], a["dt_bias"], a["state"], a["gate"], a["o_weight"],
        num_heads=H, head_dim=D, conv_kernel_size=K, lower_bound=-5.0,
        norm_eps=1e-5, nv=nv, ty=ty,
    )


def test_geometry_reproduces_the_eager_recurrence_partition():
    """The default geometry must put ONE value row on each thread, which is
    exactly gated_delta_kernel's partition.  Anything coarser silently divides
    the resident thread count of the recurrence this kernel replaces by NV --
    the failure mode that makes a correct kernel slower than what it replaces."""
    nv, ty = F.prefill_geometry(128)
    assert (nv, ty) == (4, 32)
    assert (128 // nv) // ty == 1, "NDV must be 1 at the default geometry"


@pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 256])
def test_geometry_is_always_a_valid_partition(d):
    """Whatever it returns has to divide: NDV = (D/NV)/TY must be a positive
    integer, or the kernel indexes value rows that do not exist."""
    nv, ty = F.prefill_geometry(d)
    assert nv >= 1 and 1 <= ty <= 32
    assert d % nv == 0 and (d // nv) % ty == 0
    assert (d // nv) // ty >= 1


def test_geometry_honours_an_explicit_nv_and_repairs_an_impossible_one():
    assert F.prefill_geometry(128, nv=1)[0] == 1
    assert F.prefill_geometry(128, nv=2)[0] == 2
    # 5 does not divide 128: fall back to the largest workable NV below it.
    nv, ty = F.prefill_geometry(128, nv=5)
    assert 128 % nv == 0 and (128 // nv) % ty == 0


def test_nv1_is_one_launch_and_returns_the_final_output(fake_kernels):
    y, st, cs = _call(nv=1)
    assert [c["name"] for c in fake_kernels] == ["fused"]
    assert y.shape == (1, 7, HD)          # already normed and gated
    assert st.shape == (1, H, D, D) and cs.shape == (1, K - 1, 3 * HD)
    c = fake_kernels[0]
    assert c["grid"] == (32, 32, 1 * H * 1)
    assert c["template"]["NV"] == 1 and c["template"]["TY"] == 32
    assert c["inputs"][-1] == 7, "S must be a runtime scalar, not a template arg"
    assert "NSTEPS" not in c["template"] and "S" not in c["template"]


def test_nv_gt_1_splits_into_scan_plus_norm(fake_kernels):
    y, st, cs = _call(nv=4)
    assert [c["name"] for c in fake_kernels] == ["split", "norm"]
    scan, norm = fake_kernels
    assert scan["grid"] == (32, 32, 1 * H * 4), "one threadgroup per (row, head, nv)"
    assert scan["shapes"][0] == (1, 7, H, D), "scan emits the PRE-norm output"
    # The scan must not be handed the norm's tensors: they belong to launch two,
    # and passing them would mean NV threadgroups re-read the gate for nothing.
    assert len(scan["inputs"]) == len(F._SCAN_INPUTS_SPLIT)
    assert norm["shapes"][0] == (1, 7, HD)
    assert y.shape == (1, 7, HD)


def test_the_scan_never_templates_the_chunk_width(fake_kernels):
    """One pipeline has to serve every chunk width: prefill chunks are ragged at
    the tail of a prompt, and a templated S would compile a fresh pipeline (and
    pay a cold Metal compile) for every ragged tail."""
    for S in (2, 5, 64, 300, 2048):
        _call(nv=4, S=S)
    templates = {tuple(sorted(c["template"].items())) for c in fake_kernels
                 if c["name"] == "split"}
    assert len(templates) == 1, "chunk width leaked into the template"


def test_state_and_conv_dtypes_are_preserved(fake_kernels):
    y, st, cs = _call(nv=4)
    assert st.dtype == mx.float32, "state accumulates in fp32"
    assert cs.dtype == mx.bfloat16 and y.dtype == mx.bfloat16


def test_mask_is_flattened_to_one_bool_per_row_token(fake_kernels):
    a = _args(B=1, S=7)
    mask = mx.ones((1, 7), dtype=mx.bool_)
    F.fused_kda_prefill_scan(
        a["q_in"], a["k_in"], a["v_in"], a["conv_state"], a["conv_w"], a["a"],
        a["b"], a["A_log"], a["dt_bias"], a["state"], None, None,
        num_heads=H, head_dim=D, conv_kernel_size=K, lower_bound=-5.0,
        norm_eps=1e-5, mask=mask, nv=4, ty=32,
    )
    valid = fake_kernels[0]["inputs"][F._SCAN_INPUTS_SPLIT.index("valid")]
    assert valid.shape == (7,) and valid.dtype == mx.bool_


# ===========================================================================
# The eligibility predicate.  CPU-runnable, and the part that decides whether a
# production forward can reach the kernel at all.
# ===========================================================================
class _Poison:
    """Any attribute access past the point under test is a bug."""

    def __getattr__(self, name):
        raise AssertionError(f"prefill predicate looked past its guard at {name!r}")


def _elig(probe, B, S, mask=None, cache=None, sink=None):
    return glm5.Glm5NextLinearAttention._fused_kda_prefill_eligible(
        probe, B, S, mask, cache, sink, mx.zeros((1, 1, 1), mx.bfloat16)
    )


def test_flag_is_off_by_default_and_refuses_before_touching_anything():
    prev = glm5._FUSED_KDA_PREFILL_ENV
    try:
        glm5._FUSED_KDA_PREFILL_ENV = False
        assert _elig(_Poison(), 1, 2048) is False
    finally:
        glm5._FUSED_KDA_PREFILL_ENV = prev


def test_default_of_the_env_flag_is_off():
    """Read from the environment, not from the cached global: this kernel is not
    on the default serving path until L7-b has measured it."""
    import os

    prev = glm5._FUSED_KDA_PREFILL_ENV
    try:
        glm5._FUSED_KDA_PREFILL_ENV = None
        os.environ.pop("MLX_VLM_GLM5_FUSED_KDA_PREFILL", None)
        assert glm5._fused_kda_prefill_enabled() is False
    finally:
        glm5._FUSED_KDA_PREFILL_ENV = prev


def test_s1_and_a_speculative_sink_are_refused_on_the_guard_alone():
    """S=1 belongs to the decode kernel and a sink needs per-token q/k/v the
    prefill kernel deliberately does not emit.  Both must refuse before reading
    the module, so neither can regress the path that owns it."""
    prev = glm5._FUSED_KDA_PREFILL_ENV
    try:
        glm5._FUSED_KDA_PREFILL_ENV = True
        assert _elig(_Poison(), 1, 1) is False           # width
        assert _elig(_Poison(), 2, 64) is False          # batch cap
        assert _elig(_Poison(), 1, 64, sink=[]) is False  # capture
    finally:
        glm5._FUSED_KDA_PREFILL_ENV = prev


def test_a_non_bool_or_wrong_shaped_mask_falls_back():
    prev = glm5._FUSED_KDA_PREFILL_ENV
    try:
        glm5._FUSED_KDA_PREFILL_ENV = True
        assert _elig(_Poison(), 1, 8, mask=mx.zeros((1, 8), mx.bfloat16)) is False
        assert _elig(_Poison(), 1, 8, mask=mx.ones((1, 4), mx.bool_)) is False
    finally:
        glm5._FUSED_KDA_PREFILL_ENV = prev


def test_a_wrong_shaped_cache_falls_back():
    """The kernel indexes the cache with compile-time extents; a cache that does
    not match them would be read out of bounds rather than refused."""
    config = _config()
    layer = _layer(config, seed=3, quantize=False)
    prev = glm5._FUSED_KDA_PREFILL_ENV
    try:
        glm5._FUSED_KDA_PREFILL_ENV = True
        assert layer._fused_kda_prefill_eligible(
            1, 8, None, None, None, mx.zeros((1, 1, 1), mx.bfloat16)
        ) is False
        bad = ArraysCache(size=2)
        bad[0] = mx.zeros((1, K, 3 * HD), mx.bfloat16)   # K, not K-1
        bad[1] = mx.zeros((1, H, D, D), mx.float32)
        assert layer._fused_kda_prefill_eligible(
            1, 8, None, bad, None, mx.zeros((1, 1, 1), mx.bfloat16)
        ) is False
    finally:
        glm5._FUSED_KDA_PREFILL_ENV = prev


def test_fp16_gate_params_fall_back():
    """A_log / dt_bias are fp32 in the shipped checkpoint and the kernel reads
    them as fp32; a checkpoint that cast them must not be read as float."""
    config = _config()
    layer = _layer(config, seed=4, quantize=False)
    layer.forget_gate.A_log = layer.forget_gate.A_log.astype(mx.bfloat16)
    cache = _cache()
    prev = glm5._FUSED_KDA_PREFILL_ENV
    try:
        glm5._FUSED_KDA_PREFILL_ENV = True
        assert layer._fused_kda_prefill_eligible(
            1, 8, None, cache, None, mx.zeros((1, 1, 1), mx.bfloat16)
        ) is False
    finally:
        glm5._FUSED_KDA_PREFILL_ENV = prev


def test_the_three_kda_predicates_are_disjoint():
    """S=1 -> decode kernel, 2..MAX_WIDTH -> verify block, wider -> prefill.  The
    prefill predicate must not claim a width the block path already serves while
    the block path is enabled, or the verify measurement changes underneath the
    prefill arm."""
    assert glm5._FUSED_KDA_PREFILL_MIN_S >= 2
    assert glm5._FUSED_KDA_PREFILL_MAX_S == 0     # no cap: S is a runtime scalar
    assert glm5._FUSED_KDA_PREFILL_MAX_BATCH == 1
