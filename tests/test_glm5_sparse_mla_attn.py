"""CPU identity tests for the V5a DSA gather/attention path.

CPU-ONLY BY CONSTRUCTION: ``mx.set_default_device(mx.cpu)`` runs at import, before any
array is built, so this file is safe inside a GPU measurement window.  The fused metal
kernel itself cannot be exercised here -- what is checked is (a) that the ``take`` gather
is BIT-EXACT against the take_along_axis form the model uses today, and (b) that the CPU
transcription of the kernel's algorithm reproduces the eager attention to <= 1e-3
relative on the same index set.  Bit-identity for the FUSED path is not expected and not
asserted: fp32 online softmax reassociates and P is rounded to bf16.
"""

import math

import mlx.core as mx

mx.set_default_device(mx.cpu)

import pytest  # noqa: E402

from mlx_vlm.models.glm5_next import sparse_mla_attn as S  # noqa: E402


def _fixture(B=1, L=7, H=6, D=32, Kv=97, K=24, kpool=4, seed=0, dead_row=False):
    """A GLM-5-style selection: ``K/kpool`` pools of ``kpool`` CONSECUTIVE keys.

    The pool structure is the real one (language.py:2183-2189 expands pool_indices by
    index_kpool), and it is what gives the gather its 4 KB runs.  A synthetic uniform
    index would be a different, easier problem -- see the retraction in fused_mla_attn.py.
    """
    mx.random.seed(seed)
    kv = mx.random.normal((Kv, D)).astype(mx.bfloat16)
    q = mx.random.normal((B * L, H, D)).astype(mx.bfloat16)
    npool = K // kpool
    starts = (mx.random.randint(0, max(1, Kv - kpool), (B * L, npool))
              // kpool * kpool)
    idx = (starts[:, :, None] + mx.arange(kpool)[None, None, :]).reshape(B * L, K)
    idx = mx.minimum(idx, Kv - 1).astype(mx.int32)
    # a few tail slots unselected (-1), exactly as index_kpool padding produces
    keep = mx.arange(K)[None, :] < (K - 3)
    idx = mx.where(keep, idx, mx.array(-1, dtype=mx.int32))
    if dead_row:
        idx = mx.concatenate(
            [mx.full((1, K), -1, dtype=mx.int32), idx[1:]], axis=0
        )
    return kv, q, idx


def _eager_gather(kv, idx, Kv):
    """The model's own gather (language.py:2404-2408), shape for shape."""
    G, K = idx.shape
    D = kv.shape[-1]
    clamped = mx.clip(idx, 0, Kv - 1)
    kv4 = kv.reshape(1, 1, Kv, D)
    return mx.take_along_axis(
        mx.broadcast_to(kv4, (1, G, Kv, D)),
        mx.broadcast_to(clamped[None, :, :, None], (1, G, K, D)),
        axis=2,
    )


def test_take_gather_is_bit_exact():
    kv, q, idx = _fixture()
    Kv, D = kv.shape
    eager = _eager_gather(kv, idx, Kv)
    got = S.gather_latents_take(kv.reshape(1, 1, Kv, D),
                                mx.clip(idx, 0, Kv - 1)[None])
    assert got.shape == eager.shape
    assert got.dtype == eager.dtype
    # bit-exact: same bytes, not just close
    assert mx.array_equal(got.view(mx.uint16), eager.view(mx.uint16)).item()


def test_take_gather_bit_exact_batched():
    """B > 1 folds the batch into the row index; the cache slice is non-contiguous there."""
    Kv, D, G, K = 64, 16, 5, 12
    mx.random.seed(3)
    kv = mx.random.normal((2, 1, Kv, D)).astype(mx.bfloat16)
    idx = mx.random.randint(0, Kv, (2, G, K)).astype(mx.int32)
    eager = mx.take_along_axis(
        mx.broadcast_to(kv, (2, G, Kv, D)),
        mx.broadcast_to(idx[:, :, :, None], (2, G, K, D)),
        axis=2,
    )
    got = S.gather_latents_take(kv, idx)
    assert mx.array_equal(got.view(mx.uint16), eager.view(mx.uint16)).item()


def test_index_set_matches_exactly():
    """The fused path must attend to the SAME keys, not merely a similar score."""
    kv, q, idx = _fixture()
    Kv = kv.shape[0]
    clamped = mx.clip(idx, 0, Kv - 1)
    valid = idx >= 0
    for g in range(idx.shape[0]):
        eager_set = {int(v) for v, ok in zip(clamped[g].tolist(), valid[g].tolist()) if ok}
        # the kernel drops a slot iff idx < 0 or idx >= Kv -- same predicate
        kern_set = {int(v) for v in idx[g].tolist() if 0 <= v < Kv}
        assert eager_set == kern_set


@pytest.mark.parametrize("bk", [8, 16, 32])
def test_online_reference_matches_one_pass(bk):
    """THE transcription gate: online tiling vs one-pass, same algorithm, <= 1e-3.

    Run with the bf16 rounding of P switched OFF on both sides, so the two forms are
    algebraically identical and the only thing under test is the online rescale
    recurrence.  A failure here is a bug in the tiling, not a numerical artifact.
    (With the rounding on, the two round against different row maxima and separate by
    ~1 bf16 ulp -- bounded by the next test, which is the honest size of the change.)
    """
    kv, q, idx = _fixture(L=11, H=8, D=64, Kv=200, K=32)
    scale = 1.0 / math.sqrt(kv.shape[-1])
    ref = S.mla_sparse_reference(q, kv, idx, scale, p_bf16=False).astype(mx.float32)
    onl = S.mla_sparse_reference_online(
        q, kv, idx, scale, bk=bk, p_bf16=False).astype(mx.float32)
    denom = mx.maximum(mx.abs(ref).max(), mx.array(1e-6))
    rel = (mx.abs(onl - ref).max() / denom).item()
    assert rel <= 1e-3, f"bk={bk} rel={rel}"


def test_bf16_probability_rounding_is_one_ulp():
    """Size the ONE numerical change the fused path makes, so it is never surprising.

    bfloat16 has an 8-bit significand: 2**-8 = 3.91e-3 relative.  This is why the fused
    path is gated on KL and not on identity.  (The eager path takes the same rounding
    inside MLX's composite; this test bounds it, it does not endorse it.)
    """
    kv, q, idx = _fixture(L=11, H=8, D=64, Kv=200, K=32)
    scale = 1.0 / math.sqrt(kv.shape[-1])
    exact = S.mla_sparse_reference(q, kv, idx, scale, p_bf16=False).astype(mx.float32)
    onl = S.mla_sparse_reference_online(q, kv, idx, scale).astype(mx.float32)
    denom = mx.maximum(mx.abs(exact).max(), mx.array(1e-6))
    rel = (mx.abs(onl - exact).max() / denom).item()
    assert rel <= 2 ** -8, rel      # <= one bfloat16 ulp: sub-ulp on the bf16 output


def test_online_reference_matches_the_model_path():
    """Against the model's own composite (gather -> scores -> softmax -> weighted sum).

    The model path rounds P to bf16 inside mx.softmax on a bf16 score tensor, so this
    comparison is made with the same rounding on both sides; what it checks is that the
    kernel attends to the same keys with the same weights, at <= 1e-3.
    """
    kv, q, idx = _fixture(L=9, H=8, D=64, Kv=150, K=28)
    Kv, D = kv.shape
    scale = 1.0 / math.sqrt(D)
    kg = _eager_gather(kv, idx, Kv)[0].astype(mx.float32)       # [G, K, D]
    s = (q.astype(mx.float32) @ kg.transpose(0, 2, 1)) * scale  # [G, H, K]
    s = mx.where((idx >= 0)[:, None, :], s, mx.array(-3.0e38))
    pu = mx.where(s <= -1.0e37, mx.zeros_like(s), mx.exp(s - mx.max(s, -1, keepdims=True)))
    pu = pu.astype(mx.bfloat16).astype(mx.float32)
    eager = ((pu @ kg) / mx.sum(pu, axis=-1, keepdims=True)).astype(mx.float32)
    onl = S.mla_sparse_reference_online(q, kv, idx, scale).astype(mx.float32)
    denom = mx.maximum(mx.abs(eager).max(), mx.array(1e-6))
    rel = (mx.abs(onl - eager).max() / denom).item()
    assert rel <= 2 ** -8, rel      # sub-ulp against the model's own bf16 composite


def test_dead_row_is_zero_like_the_model():
    """A query with no selected key: _gathered_attention zeroes it (language.py:2415)."""
    kv, q, idx = _fixture(dead_row=True)
    scale = 1.0 / math.sqrt(kv.shape[-1])
    out = S.mla_sparse_reference_online(q, kv, idx, scale)
    assert mx.abs(out[0]).max().item() == 0.0
    assert mx.abs(out[1]).max().item() > 0.0


def test_threadgroup_budget_and_geometry():
    """The tiling must fit 32 KB and cover the whole score tile exactly once."""
    assert S.threadgroup_bytes(512, 64, 16) <= 32768
    assert S.threadgroup_bytes(512, 32, 16) <= 32768
    assert S.threadgroup_bytes(512, 64, 32) > 32768     # why BK is 16, not 32
    for bq, bk in ((64, 16), (32, 16), (64, 8)):
        nsg = (bq // 8) * (bk // 8)
        lpr = bk // 2
        assert 32 % lpr == 0                 # xor-reduction stays inside a simdgroup
        assert nsg * (32 // lpr) == bq       # one softmax pass covers every row
        assert nsg * 32 <= 1024              # Metal threadgroup limit


def test_msl_emits_and_is_self_consistent():
    src = S.emit_msl(512, 64, 16, 0.044)
    assert "[[kernel]] void mla_sparse_probe" in src
    assert "typedef bfloat bfloat16_t" in src
    assert "simdgroup_multiply_accumulate" in src


def test_mode_env(monkeypatch):
    for v, want in (("", "eager"), ("0", "eager"), ("take", "take"), ("fused", "fused")):
        monkeypatch.setenv("MLX_VLM_GLM5_DSA_GATHER_KERNEL", v)
        S._reset_mode_cache()
        assert S.dsa_gather_mode() == want
    monkeypatch.setenv("MLX_VLM_GLM5_DSA_GATHER_KERNEL", "bogus")
    S._reset_mode_cache()
    with pytest.raises(ValueError):
        S.dsa_gather_mode()
    monkeypatch.delenv("MLX_VLM_GLM5_DSA_GATHER_KERNEL")
    S._reset_mode_cache()


# --------------------------------------------------------------------------- #
# end-to-end through the model's own _gathered_attention
# --------------------------------------------------------------------------- #


class _Stub:
    """The three parameterised pieces _gathered_attention touches, nothing else.

    Calling the real ``Glm5NextSparseAttention._gathered_attention`` unbound on this runs
    the patched code path verbatim -- chunk loop, mode branch, masking, epilogue -- without
    building a 320B config.
    """

    def __init__(self, H, dq, dv, dim, hidden):
        from mlx_vlm.models.mla import MultiLinear
        import mlx.nn as nn

        self.embed_q = MultiLinear(dq, dim, H)
        self.unembed_out = MultiLinear(dim, dv, H)
        self.o_proj = nn.Linear(H * dv, hidden, bias=False)
        self.scale = 1.0 / math.sqrt(dq)


def _run_gathered(mode, monkeypatch, seed=5, B=1, L=12, H=4, dq=16, dv=16, dim=32,
                  Kv=80, K=16, hidden=24):
    from mlx_vlm.models.glm5_next.language import Glm5NextSparseAttention

    monkeypatch.setenv("MLX_VLM_GLM5_DSA_GATHER_KERNEL", mode)
    monkeypatch.setenv("MLX_VLM_GLM5_GATHER_Q_CHUNK", "4")   # force >1 chunk
    S._reset_mode_cache()
    import mlx_vlm.models.glm5_next.language as LG
    monkeypatch.setattr(LG, "_GATHER_Q_CHUNK", 4, raising=False)

    mx.random.seed(seed)
    stub = _Stub(H, dq, dv, dim, hidden)
    q = mx.random.normal((B, H, L, dq)).astype(mx.bfloat16)
    kv = mx.random.normal((B, 1, Kv, dim)).astype(mx.bfloat16)
    _, _, idx = _fixture(L=L, H=H, D=dim, Kv=Kv, K=K, seed=seed)
    topk_idx = idx.reshape(B, 1, L, K)
    return Glm5NextSparseAttention._gathered_attention(stub, q, kv, topk_idx)


def test_take_mode_is_bit_identical_end_to_end(monkeypatch):
    """The whole gathered-attention block, eager vs take, byte for byte."""
    a = _run_gathered("eager", monkeypatch)
    b = _run_gathered("take", monkeypatch)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item(), \
        f"max|delta| = {mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()}"
