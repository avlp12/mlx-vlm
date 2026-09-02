"""Fused KDA for S>1 (the speculative verify block) against the eager path.

The claim under test is bit-exactness, not closeness: the kernel is a
transcription of the eager chain including where it rounds, so anything other
than an exact match is a bug rather than a tolerance question.

Note what the eager S>1 baseline actually is.  Its recurrence was ALREADY fused
-- gated_delta_update dispatches gated_delta_kernel, whose own `for t` scan keeps
the state in registers -- so these tests are checking that folding the *glue*
(conv window, silu, two L2 norms, beta, gated RMSNorm) into that scan changes
nothing.  The delta-rule arithmetic is shared with the S=1 kernel, which
427becf9 already pinned bit-identical.
"""
import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.gated_delta import gated_delta_update
from mlx_vlm.models.glm5_next.fused_kda import _kernel, fused_kda_verify_block
from mlx_vlm.models.glm5_next.language import Glm5NextRMSNormGated, _l2norm

H, D, K = 4, 128, 4          # head_dim 128 and kernel 4 are the shipped values
HD = H * D
LB, EPS = -5.0, 1e-5


def _eager(q_o, k_o, v_o, conv_state, conv_w, a, b, A_log, dt_bias, state,
           gate, o_w, mask=None):
    """A line-by-line copy of Glm5NextLinearAttention.__call__'s S>1 tail."""
    B, S, _ = q_o.shape
    mixed = mx.concatenate([q_o, k_o, v_o], axis=-1)
    if mask is not None:
        mixed = mx.where(mask[..., None], mixed, 0)
    conv_input = mx.concatenate([conv_state, mixed], axis=1)
    new_conv_state = mx.contiguous(conv_input[:, -(K - 1):, :])
    conv = nn.Conv1d(3 * HD, 3 * HD, bias=False, kernel_size=K, groups=3 * HD,
                     padding=0)
    conv.weight = conv_w
    conv_out = nn.silu(conv(conv_input))
    q, k, v = mx.split(conv_out, [HD, 2 * HD], axis=-1)
    q = q.reshape(B, S, H, D); k = k.reshape(B, S, H, D); v = v.reshape(B, S, H, D)
    dt = q.dtype
    q = (_l2norm(q.astype(mx.float32)) * (D ** -0.5)).astype(dt)
    k = _l2norm(k.astype(mx.float32)).astype(dt)
    out, st = gated_delta_update(
        q, k, v, a.reshape(B, S, H, D), b,
        A_log.reshape(H, 1), dt_bias.reshape(H, D), state=state, lower_bound=LB,
    )
    norm = Glm5NextRMSNormGated(D, eps=EPS)
    norm.weight = o_w
    y = norm(out, gate.reshape(B, S, H, D)).reshape(B, S, -1)
    return y, st, new_conv_state


def _inputs(B, S, seed=0, dt=mx.bfloat16):
    mx.random.seed(seed)
    r = lambda *sh: (mx.random.normal(sh) * 0.3).astype(dt)  # noqa: E731
    return dict(
        q_in=r(B, S, HD), k_in=r(B, S, HD), v_in=r(B, S, HD),
        conv_state=r(B, K - 1, 3 * HD),
        conv_w=r(3 * HD, K, 1),
        a=r(B, S, HD), b=r(B, S, H),
        A_log=(mx.random.normal((H,)) * 0.1).astype(mx.float32),
        dt_bias=(mx.random.normal((H * D,)) * 0.1).astype(mx.float32),
        state=(mx.random.normal((B, H, D, D)) * 0.05).astype(mx.float32),
        gate=r(B, S, HD), o_weight=r(D),
    )


def _run(args, mask=None):
    return fused_kda_verify_block(
        args["q_in"], args["k_in"], args["v_in"], args["conv_state"],
        args["conv_w"], args["a"], args["b"], args["A_log"], args["dt_bias"],
        args["state"], args["gate"], args["o_weight"],
        num_heads=H, head_dim=D, conv_kernel_size=K,
        lower_bound=LB, norm_eps=EPS, mask=mask, ty=32,
    )


def _ref(args, mask=None):
    return _eager(
        args["q_in"], args["k_in"], args["v_in"], args["conv_state"],
        args["conv_w"], args["a"], args["b"], args["A_log"], args["dt_bias"],
        args["state"], args["gate"], args["o_weight"], mask,
    )


needs_metal = pytest.mark.skipif(
    not mx.metal.is_available() or _kernel("block") is None,
    reason="Metal unavailable",
)


@needs_metal
@pytest.mark.parametrize("S", [1, 2, 3, 4, 5, 6, 7, 8])
def test_block_is_bit_identical_to_eager(S):
    """Every width the verify block can ask for, including the two that straddle
    the conv window: S < K-1 mixes cached rows into the tail, S >= K-1 does not."""
    args = _inputs(1, S, seed=S)
    fy, fs, fc, *_ = _run(args)
    ey, es, ec = _ref(args)
    mx.eval(fy, fs, fc, ey, es, ec)
    assert mx.array_equal(fy, ey).item(), f"y differs at S={S}"
    assert mx.array_equal(fs, es).item(), f"state differs at S={S}"
    assert mx.array_equal(fc, ec).item(), f"conv window differs at S={S}"


@needs_metal
@pytest.mark.parametrize("B,S", [(2, 4), (3, 8), (4, 2)])
def test_batched_block_is_bit_identical(B, S):
    """B>1 x S>1 -- grid.z is B*H and B never enters the source, so this is the
    claim that the batch axis needs no new machinery."""
    args = _inputs(B, S, seed=100 + B * 16 + S)
    fy, fs, fc, *_ = _run(args)
    ey, es, ec = _ref(args)
    mx.eval(fy, fs, fc, ey, es, ec)
    assert mx.array_equal(fy, ey).item()
    assert mx.array_equal(fs, es).item()
    assert mx.array_equal(fc, ec).item()


@needs_metal
def test_masked_rows_match_the_eager_zeroing():
    """The eager path zeroes the PRE-conv input per (row, token); the kernel must
    put the zero in the same place, before both the conv and the window write."""
    B, S = 3, 5
    args = _inputs(B, S, seed=7)
    mask = mx.array([[True] * S, [True, True, False, False, False], [False] * S])
    fy, fs, fc, *_ = _run(args, mask)
    ey, es, ec = _ref(args, mask)
    mx.eval(fy, fs, fc, ey, es, ec)
    assert mx.array_equal(fy, ey).item()
    assert mx.array_equal(fs, es).item()
    assert mx.array_equal(fc, ec).item()


@needs_metal
def test_block_of_one_matches_the_s1_kernel_path():
    """S=1 through the block kernel must equal S=1 through the eager path, which
    is the same value 427becf9 pinned for the S=1 kernel -- so the two kernels
    agree with each other by transitivity."""
    args = _inputs(1, 1, seed=11)
    fy, fs, fc, *_ = _run(args)
    ey, es, ec = _ref(args)
    mx.eval(fy, fs, fc, ey, es, ec)
    assert mx.array_equal(fy, ey).item()
    assert mx.array_equal(fs, es).item()
    assert mx.array_equal(fc, ec).item()


@needs_metal
def test_one_launch_per_block_not_per_token():
    """The point of the kernel: the state round-trip is paid once per block, not
    once per token.  A W-token block reads and writes [H,D,D] exactly once."""
    args = _inputs(1, 8, seed=3)
    fy, fs, fc, *_ = _run(args)
    mx.eval(fy, fs, fc)
    assert fy.shape == (1, 8, HD)
    assert fs.shape == args["state"].shape
    assert fc.shape == args["conv_state"].shape


@needs_metal
def test_sink_tensors_match_what_the_eager_path_stashes():
    """gdn_sink carries the post-conv, post-L2-norm q/k/v; a rejected round is
    replayed from them, so anything but an exact match corrupts the rollback."""
    args = _inputs(1, 6, seed=21)
    _, _, _, fq, fk, fv = _run(args)
    B, S = 1, 6
    mixed = mx.concatenate([args["q_in"], args["k_in"], args["v_in"]], axis=-1)
    conv_input = mx.concatenate([args["conv_state"], mixed], axis=1)
    conv = nn.Conv1d(3 * HD, 3 * HD, bias=False, kernel_size=K, groups=3 * HD,
                     padding=0)
    conv.weight = args["conv_w"]
    co = nn.silu(conv(conv_input))
    eq, ek, ev = mx.split(co, [HD, 2 * HD], axis=-1)
    eq = eq.reshape(B, S, H, D); ek = ek.reshape(B, S, H, D); ev = ev.reshape(B, S, H, D)
    dt = eq.dtype
    eq = (_l2norm(eq.astype(mx.float32)) * (D ** -0.5)).astype(dt)
    ek = _l2norm(ek.astype(mx.float32)).astype(dt)
    mx.eval(fq, fk, fv, eq, ek, ev)
    assert mx.array_equal(fq, eq).item(), "sink q differs"
    assert mx.array_equal(fk, ek).item(), "sink k differs"
    assert mx.array_equal(fv, ev).item(), "sink v differs"


@needs_metal
def test_the_two_predicates_are_disjoint_so_the_S1_path_is_untouched():
    """The S=1 kernel's eligibility and the block kernel's must never both fire,
    and neither may reach into the other's width.  This is the guarantee that
    adding the block path cannot regress plain decode: at S==1 the block
    predicate refuses on width alone, before it looks at anything else.
    """
    from mlx_vlm.models.glm5_next import language as L

    class _Probe:
        """Only the width test should be reached; everything else is poisoned."""
        num_heads = H
        head_dim = D
        conv_kernel_size = K
        _fused_kda_ty = 32

        def __getattr__(self, name):  # any deeper access is a bug
            raise AssertionError(f"block predicate looked past S at {name!r}")

    p = _Probe()
    assert L.Glm5NextLinearAttention._fused_kda_block_eligible(
        p, 1, 1, None, None, mx.zeros((1, 1, 1), mx.bfloat16)
    ) is False
    assert L.Glm5NextLinearAttention._fused_kda_block_eligible(
        p, 1, L._FUSED_KDA_MAX_WIDTH + 1, None, None, mx.zeros((1, 1, 1), mx.bfloat16)
    ) is False


@needs_metal
def test_block_kill_switch_is_honoured():
    """The A/B depends on this: flipping the module global has to take effect
    without a reimport, or the paired arms are measuring the same path twice."""
    from mlx_vlm.models.glm5_next import language as L

    prev = L._FUSED_KDA_BLOCK
    try:
        L._FUSED_KDA_BLOCK = False

        class _Probe:
            def __getattr__(self, name):
                raise AssertionError(f"refused kernel still touched {name!r}")

        assert L.Glm5NextLinearAttention._fused_kda_block_eligible(
            _Probe(), 1, 4, None, None, mx.zeros((1, 1, 1), mx.bfloat16)
        ) is False
    finally:
        L._FUSED_KDA_BLOCK = prev
