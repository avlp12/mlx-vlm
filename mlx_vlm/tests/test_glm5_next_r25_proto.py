"""R25 -- bit-exactness of the EXPERIMENTAL 2-layer persistent KDA kernel.

The prototype (``models/glm5_next/r25_megakernel_proto.py``) runs two consecutive
KDA decode layers in ONE ``mx.fast.metal_kernel`` launch, with a grid-wide
counting barrier on a device atomic between them, and with the head -> threadgroup
map ROTATED between the layers so that every layer-2 threadgroup reads a slice of
layer 1's output that a DIFFERENT threadgroup wrote.

It must be bit-identical to the shipped fused kernel run twice back to back.
These are tiny shapes; the production-shape run is a measurement, not a test.

``FENCE=5`` -- the activation handed over as 32-bit device atomics -- is the only
variant that is correct on applegpu_g15d.  No fence MSL offers makes a plain
device store visible to another threadgroup (logs/sweep3/R25_visprobe.json), so
``test_plain_stores_do_not_cross_threadgroups`` pins that the weaker variants are
*allowed* to fail and does not assert that they do on hardware we have not
measured.

Requires Metal (``mx.fast.metal_kernel`` is GPU-only), so every test here skips
on a CPU-default process.
"""

import mlx.core as mx
import pytest

from mlx_vlm.models.glm5_next.r25_megakernel_proto import (
    new_barrier_counter,
    pack_layers,
    proto_2layer,
    proto_source,
    reference_2layer,
)

_GPU = mx.metal.is_available() and mx.default_device() == mx.gpu
gpu_only = pytest.mark.skipif(not _GPU, reason="mx.fast.metal_kernel needs a Metal GPU")

H, D, K = 4, 32, 4
TY, NTG, ROT = 8, 8, 4
LB, EPS = -5.0, 1e-5


def _layer(seed):
    mx.random.seed(seed)
    n = lambda shape, s=0.3: (mx.random.normal(shape) * s).astype(mx.bfloat16)  # noqa: E731
    d = dict(
        q_in=n((1, 1, H * D)), k_in=n((1, 1, H * D)), v_in=n((1, 1, H * D)),
        conv_state=n((1, K - 1, 3 * H * D)),
        conv_w=n((3 * H * D, K, 1), 0.5),
        a=n((1, 1, H * D)), b=n((1, 1, H)),
        A_log=(mx.random.normal((H,)) * 0.5).astype(mx.float32),
        dt_bias=(mx.random.normal((H * D,)) * 0.5).astype(mx.float32),
        state=(mx.random.normal((1, H, D, D)) * 0.05).astype(mx.float32),
        gate=n((1, 1, H * D)),
        o_weight=(mx.ones((D,)) + 0.02 * mx.random.normal((D,))).astype(mx.bfloat16),
    )
    mx.eval(list(d.values()))
    return d


def _run(seed, fence=5, rot=ROT, ntg=NTG):
    l1, l2 = _layer(seed), _layer(seed + 100)
    kw = dict(num_heads=H, head_dim=D, conv_kernel_size=K,
              lower_bound=LB, norm_eps=EPS)
    ref = reference_2layer(l1, l2, ty=TY, **kw)
    mx.eval(ref)
    got = proto_2layer(l1, pack_layers(l1, l2), bar=new_barrier_counter(),
                       wbuf=mx.zeros((16,), mx.uint32), ntg=ntg, rot=rot,
                       ty=TY, fence=fence, wm=0, **kw)
    mx.eval(got)
    # (y1, st1, cs1, y2, st2, cs2) in the reference's order
    y1, y2, st, cs, dbg = got
    return ref, (y1, st[0], cs[0], y2, st[1], cs[1]), dbg


def test_source_derives_from_the_shipped_kernel():
    """The layer body must be the shipped source verbatim apart from two edits."""
    from mlx_vlm.models.glm5_next.fused_kda import _SOURCE

    src = proto_source(fence=2, wstream=False)
    # every arithmetic landmark of the shipped chain appears twice, once per layer
    for probe in ("mlx_sigmoid_fast", "gated delta rule", "metal::precise::rsqrt",
                  "sq_acc", "quad_sum" if "quad_sum" in _SOURCE else "simd_sum"):
        assert src.count(probe) == 2 * _SOURCE.count(probe), probe
    assert src.count("atomic_fetch_add_explicit(g_bar") == 1


@gpu_only
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bit_exact_against_two_shipped_launches(seed):
    ref, got, _ = _run(seed, fence=5)
    names = ("y1", "state1", "conv_state1", "y2", "state2", "conv_state2")
    for name, r, g in zip(names, ref, got):
        assert mx.array_equal(r, g).item(), name


@gpu_only
def test_barrier_completed_and_no_timeout():
    _, _, dbg = _run(0, fence=5)
    dbg = dbg.tolist()
    assert all(dbg[i * 4 + 1] == 0 for i in range(NTG)), "a threadgroup timed out"
    assert dbg[3] == NTG, "arrival counter should end at exactly NTG"


@gpu_only
def test_no_rotation_is_also_exact():
    """ROT=0 removes the cross-threadgroup read; the arithmetic must not move."""
    ref, got, _ = _run(0, fence=5, rot=0, ntg=H)
    for r, g in zip(ref, got):
        assert mx.array_equal(r, g).item()


@gpu_only
@pytest.mark.parametrize("fence", [0, 1, 2, 3, 4])
def test_plain_stores_may_not_cross_threadgroups(fence):
    """FENCE<5 keeps the cross-barrier hand-off on plain device stores.

    At production shapes on applegpu_g15d every one of these is WRONG in 200/200
    launches.  At these tiny shapes the grid is small enough that it sometimes
    happens to work, so this is not an assertion that they fail -- it is a pin
    that they are not required to succeed, and that ROT=0 (nothing crossing) is
    correct at every fence level.
    """
    ref, got, _ = _run(0, fence=fence, rot=0, ntg=H)
    for r, g in zip(ref, got):
        assert mx.array_equal(r, g).item(), "ROT=0 must be exact at every fence level"
