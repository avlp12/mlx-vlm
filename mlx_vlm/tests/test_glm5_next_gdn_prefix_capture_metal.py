"""Later-approved Metal parity gate for the opt-in GDN prefix-capture ABI.

This module is intentionally isolated from CPU selector tests.  It launches
the capture kernel and is not part of this desk-side change's verification.
Run it only at the separate Metal approval gate.
"""

import mlx.core as mx
import pytest

from mlx_vlm.models.glm5_next.fused_kda import (
    _kernel,
    fused_kda_verify_block,
    fused_kda_verify_block_capture,
)


H, D, K = 4, 128, 4
HD = H * D
LB, EPS = -5.0, 1e-5


def _inputs(batch, steps, seed):
    mx.random.seed(seed)
    rand = lambda *shape: (mx.random.normal(shape) * 0.3).astype(mx.bfloat16)  # noqa: E731
    return dict(
        q_in=rand(batch, steps, HD),
        k_in=rand(batch, steps, HD),
        v_in=rand(batch, steps, HD),
        conv_state=rand(batch, K - 1, 3 * HD),
        conv_w=rand(3 * HD, K, 1),
        a=rand(batch, steps, HD),
        b=rand(batch, steps, H),
        A_log=(mx.random.normal((H,)) * 0.1).astype(mx.float32),
        dt_bias=(mx.random.normal((HD,)) * 0.1).astype(mx.float32),
        state=(mx.random.normal((batch, H, D, D)) * 0.05).astype(mx.float32),
        gate=rand(batch, steps, HD),
        o_weight=rand(D),
    )


def _run(fn, args, steps, mask):
    return fn(
        args["q_in"][:, :steps], args["k_in"][:, :steps], args["v_in"][:, :steps],
        args["conv_state"], args["conv_w"], args["a"][:, :steps],
        args["b"][:, :steps], args["A_log"], args["dt_bias"], args["state"],
        args["gate"][:, :steps], args["o_weight"],
        num_heads=H, head_dim=D, conv_kernel_size=K, lower_bound=LB,
        norm_eps=EPS, mask=mask[:, :steps], ty=32,
    )


needs_capture_metal = pytest.mark.skipif(
    not mx.metal.is_available() or _kernel("block_capture") is None,
    reason="GDN prefix-capture Metal kernel unavailable",
)


@needs_capture_metal
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("steps", [2, 3, 4, 5, 6, 7, 8])
def test_capture_prefix_state_and_fifo_match_each_default_prefix(batch, steps):
    """Every captured post-token state/window equals a standalone prefix run.

    The nontrivial second row exercises the valid-mask path and proves that a
    masked token's recurrent state and FIFO are captured at the same point as
    the default block kernel, for every supported S=2..8 width.
    """
    args = _inputs(batch, steps, seed=1000 + 100 * batch + steps)
    mask = mx.array(
        [[True] * steps]
        if batch == 1
        else [[True] * steps, [(i % 3) != 1 for i in range(steps)]],
        dtype=mx.bool_,
    )
    captured = _run(fused_kda_verify_block_capture, args, steps, mask)
    plain = _run(fused_kda_verify_block, args, steps, mask)
    mx.eval(*captured, *plain)
    for observed, expected in zip(captured[:6], plain):
        assert mx.array_equal(observed, expected).item()

    prefix_states, prefix_convs = captured[6:]
    for t in range(steps - 1):
        _y, state, conv, *_ = _run(fused_kda_verify_block, args, t + 1, mask)
        mx.eval(state, conv)
        assert mx.array_equal(prefix_states[:, t], state).item(), (batch, steps, t)
        assert mx.array_equal(prefix_convs[:, t], conv).item(), (batch, steps, t)
