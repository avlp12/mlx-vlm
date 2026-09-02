"""R25 -- EXPERIMENTAL 2-layer persistent-kernel prototype for the KDA decode step.

NOT wired into the serving path.  Nothing in ``language.py`` imports this module;
it exists so the pre-registered R25 measurement (a *persistent* kernel that
executes TWO consecutive KDA layers in ONE launch, with a device-atomic barrier
between them) can be built and measured against the shipped fused kernel.

WHAT IT IS
----------
``fused_kda.py`` fuses the post-projection chain of ONE KDA layer into one
``mx.fast.metal_kernel`` launch, with ``grid.z = B*H`` threadgroups (one per
(batch row, head)).  This module concatenates **two copies of that exact source**
into a single kernel, separated by a grid-wide counting barrier on a device
atomic, and launches it on ``NTG >= B*H`` threadgroups.

Layer 2 consumes layer 1's output: ``q_in = k_in = v_in = y1``.  So the reference
this is measured against is precisely *two dependent launches of the shipped
kernel* with the same inputs.

WHY THE HEAD ROTATION MATTERS (this is the point of the experiment)
-------------------------------------------------------------------
Threadgroup ``g`` runs head ``g`` in layer 1 and head ``g - ROT`` in layer 2.
With ``ROT != 0`` **every** layer-2 threadgroup reads a slice of ``y1`` that a
*different* threadgroup wrote before the barrier.  Without the rotation the
prototype would have no cross-threadgroup data dependency at all and the memory
model would never be exercised -- the barrier ladder (R25_BARRIER_VERDICT.md,
"what this run does not clear", item 5) left exactly that gap.

CROSS-BARRIER VISIBILITY -- WHAT WAS ARGUED, AND WHAT WAS MEASURED
------------------------------------------------------------------
The write of ``y1`` and the read of ``y1`` are in different threadgroups, so
program order inside a threadgroup buys nothing.  The chain the MSL memory model
*appears* to offer is:

  release (writer)   all threads' stores to y1
                     -> ``threadgroup_barrier(mem_device | mem_threadgroup)``
                     -> ``atomic_thread_fence(mem_device, seq_cst)``
                     -> ``atomic_fetch_add(counter, 1, relaxed)``
  acquire (reader)   spin on the counter -> fence -> ``threadgroup_barrier`` -> loads

**On applegpu_g15d that chain does not work, and the failure is not subtle.**
``logs/sweep3/R25_visprobe.json``: threadgroup ``g`` writes 64 uint32 words, all
80 threadgroups rendezvous on this barrier, threadgroup ``g`` then reads the
words written by threadgroup ``g-ROT``:

  ROT = 0 (each threadgroup reads its own slot)      0 / 25,600 words wrong, every mode
  ROT != 0, plain stores, no device fence            25,600 / 25,600 wrong (100 %)
  ROT != 0, threadgroup_barrier(mem_device)          25,600 / 25,600 wrong (100 %)
  ROT != 0, + device fence in thread 0               10,240-15,360 wrong (40-60 %)
  ROT != 0, + device fence in EVERY thread           10,240-15,360 wrong (40-60 %)
  ROT != 0, + volatile device reads                  10,240-15,360 wrong (40-60 %)
  ROT != 0, ATOMIC stores and ATOMIC loads (relaxed) 0 wrong, every cell

So a device-scope fence *helps* -- it moves the failure from total to partial --
but **no fence MSL offers makes a plain device store visible to another
threadgroup**.  Only 32-bit device atomics are coherent across threadgroups here,
which is consistent with Metal documenting coherence at *dispatch* boundaries and
providing ``threadgroup_barrier`` as a *threadgroup*-scoped construct.

The prototype therefore hands the activation over as 32-bit device atomics
(``r25_xfer_writer`` / ``r25_xfer_reader``): the same bfloat16 bit pattern, in the
low half of a uint32 word, stored and loaded with ``memory_order_relaxed``.  The
ordinary ``y_1`` output is still written by a plain store, so it stays bit-for-bit
what the shipped kernel produces.

``FENCE`` selects how much of the chain is emitted, so all of the above is
reproducible from this module:

  FENCE=0  ``threadgroup_barrier(mem_threadgroup)`` only -- no device ordering
  FENCE=1  ``threadgroup_barrier(mem_device | mem_threadgroup)`` -- what
           ``prep/sweep3/r25_barrier_ladder.py`` used, where nothing crossed
  FENCE=2  + a device-scope fence in thread 0 only
  FENCE=3  + a device-scope fence in every thread
  FENCE=4  + the cross-barrier reads through a ``volatile device`` pointer
  FENCE=5  the atomic hand-off -- **the only variant that is bit-exact**

At ``ROT=0`` every level is bit-exact, because nothing crosses.  At ``ROT!=0``
only ``FENCE=5`` is, in 200/200 launches at production shapes.

FORWARD PROGRESS
----------------
Metal gives no co-residency guarantee.  Every spin is bounded by ``SPINCAP``: a
non-resident peer becomes a *recorded timeout*, never a hang.  The counter is
monotonic (crossing ``b`` completes at ``(b+1)*NTG``), so there is no reset and
no generation variable, hence no relaxed-ordering race in the barrier itself.

The counter buffer is an *input* array that the caller zeroes before the launch
(and never inside a timed region), because MLX's ``init_value`` would zero every
output as well -- three extra fill dispatches, which is precisely the quantity
this experiment is trying to measure.
"""

import re

import mlx.core as mx

from .fused_kda import _HEADER, _SOURCE

# ---------------------------------------------------------------------------
# The COHERENT hand-off.  Measured on applegpu_g15d (logs/sweep3/R25_visprobe.json):
# a plain `device` store made by threadgroup A is NOT visible to a load in
# threadgroup B after a grid-wide barrier -- not with threadgroup_barrier(
# mem_device), not with a device-scope atomic_thread_fence in thread 0, not with
# one in every thread, not through a volatile pointer.  100 % of the words are
# wrong with no fence and 40-60 % are still wrong with every fence MSL offers.
# ATOMIC device stores and loads are coherent: 0 wrong words in every cell.
#
# So the activation that crosses the barrier is written AND read as 32-bit device
# atomics.  These two proxies let the SHIPPED source keep its `y[i] = v` and
# `mq[i]` syntax unchanged, which is what keeps bit-exactness meaningful: the
# stored bit pattern is the identical bfloat16, only the instruction differs.
_XFER_HEADER = """
template <typename U>
struct r25_xfer_reader {          // reads a 16-bit activation out of a uint32 word
  device metal::atomic_uint* p;
  inline U operator[](size_t i) const {
    uint w = metal::atomic_load_explicit(&p[i], metal::memory_order_relaxed);
    return as_type<U>((ushort)(w & 0xffffu));
  }
};

template <typename U>
struct r25_xfer_ref {
  device U* d;
  device metal::atomic_uint* a;
  inline void operator=(U v) const {
    *d = v;                       // the ordinary output, bit-for-bit as shipped
    metal::atomic_store_explicit(a, (uint)as_type<ushort>(v),
                                 metal::memory_order_relaxed);
  }
};

template <typename U>
struct r25_xfer_writer {          // writes both the output buffer and the xfer word
  device U* d;
  device metal::atomic_uint* a;
  inline r25_xfer_ref<U> operator[](size_t i) const { return {d + i, a + i}; }
};
"""


# ---------------------------------------------------------------------------
# Deriving the per-layer body from the SHIPPED source, by two audited edits.
# Nothing else in the arithmetic is touched -- that is what makes bit-exactness
# a meaningful question rather than a tautology.

_TG_DECL_RE = re.compile(r"^  threadgroup float [A-Za-z_]+\[[^\]]+\];.*$\n", re.M)

_BH_LINE = "  const uint bh   = threadgroup_position_in_grid.z;\n"


def _layer_body() -> str:
    """The shipped ``_SOURCE`` with (a) its threadgroup scratch hoisted out and
    (b) its threadgroup -> (batch, head) map made an argument."""
    src = _SOURCE
    src, n = _TG_DECL_RE.subn("", src)
    if n != 7:
        raise RuntimeError(
            "r25 proto: expected 7 threadgroup scratch declarations in the shipped "
            "source, stripped %d -- the shipped kernel changed, re-audit this module" % n)
    if src.count(_BH_LINE) != 1:
        raise RuntimeError("r25 proto: bh line not found verbatim")
    src = src.replace(_BH_LINE, "  const uint bh   = (uint)LBH;\n", 1)
    return src


# A 34-buffer kernel does not build: Metal allows buffer indices 0..30 only.
# Every per-layer tensor is therefore ONE buffer carrying both layers, indexed by
# a compile-time layer stride -- which is what a full-step megakernel has to do
# anyway (45 layers cannot each own a binding).
_LAYER_STRIDES = {
    "conv_state": "(size_t)NB * (size_t)(K - 1) * (size_t)(3 * H * D)",
    "conv_w": "(size_t)(3 * H * D) * (size_t)K",
    "a": "(size_t)NB * (size_t)(H * D)",
    "bvec": "(size_t)NB * (size_t)H",
    "A_log": "(size_t)H",
    "dt_bias": "(size_t)(H * D)",
    "state_in": "(size_t)NB * (size_t)H * (size_t)D * (size_t)D",
    "gate": "(size_t)NB * (size_t)(H * D)",
    "o_w": "(size_t)D",
    "state_out": "(size_t)NB * (size_t)H * (size_t)D * (size_t)D",
    "conv_state_out": "(size_t)NB * (size_t)(K - 1) * (size_t)(3 * H * D)",
}


def _layer_block(layer: int, bh_expr: str, active_expr: str, chain_qkv: bool,
                 volatile_cross: bool = False, atomic_xfer: bool = False) -> str:
    """One layer's work, in its own scope, bound to that layer's slice."""
    li = layer - 1
    lines = []
    if chain_qkv:                       # layer 2 consumes layer 1's output
        if atomic_xfer:
            for n in ("mq", "mk", "mv"):
                lines.append("    r25_xfer_reader<T> %s{g_xfer};" % n)
        else:
            cast = ("volatile device T* %s = (volatile device T*)y_1;"
                    if volatile_cross else "auto %s = y_1;")
            for n in ("mq", "mk", "mv"):
                lines.append("    " + cast % n)
    # layer 1 reads the kernel's own mq/mk/mv arguments -- no alias needed
    for n, stride in _LAYER_STRIDES.items():
        src = "y_all" if n == "y" else n + "_all"
        lines.append("    auto %s = %s + %d * (%s);" % (n, src, li, stride))
    if layer == 1 and atomic_xfer:
        lines.append("    r25_xfer_writer<T> y{y_1, g_xfer};")
    else:
        lines.append("    auto y = %s;" % ("y_2" if layer == 2 else "y_1"))
    return (
        "  // ================================================== LAYER %d\n" % layer
        + "  if (%s) {\n" % active_expr
        + "\n".join(lines)
        + "\n    const uint LBH = (uint)(%s);\n" % bh_expr
        + _layer_body()
        + "  }\n"
    )


_PRELUDE = r"""
  const uint g_gid = threadgroup_position_in_grid.z;
  const uint g_tid = thread_index_in_threadgroup;
  constexpr uint G_NT = 32u * (uint)TY;

  // Scratch, shared by both layers (each layer overwrites it; the grid-wide
  // barrier between them carries mem_threadgroup, so the reuse is ordered).
  threadgroup float sq[D];
  threadgroup float sk[D];
  threadgroup float sv[D];
  threadgroup float sg[D];
  threadgroup float sgate[D];
  threadgroup float sy[D];
  threadgroup float shr[3];
  threadgroup metal::atomic_uint g_xacc;
  threadgroup uint g_spins;
  threadgroup uint g_to;
  if (g_tid == 0u) {
    atomic_store_explicit(&g_xacc, 0u, memory_order_relaxed);
    g_spins = 0u;
    g_to = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The arrival counter lives in an INPUT buffer (see the module docstring for
  // why it is not an output).  MLX declares inputs const; the barrier owns this
  // buffer exclusively for the duration of the launch.
  device metal::atomic_uint* g_bar =
      (device metal::atomic_uint*)(device uint*)bar;
  device metal::atomic_uint* g_xfer =
      (device metal::atomic_uint*)(device uint*)xfer;
"""

_WSTREAM = r"""
  // --------------------------------------------------------- weight phase
  // Stand-in for the real per-layer weight traffic a full-step megakernel would
  // carry across this barrier (o_proj + the routed experts): WM x 16 B per
  // thread of REAL q4 weight bytes, streamed by every threadgroup including the
  // ones that hold no head.  XOR-reduced and written out, so no load can be
  // dead-coded.
  {
    const device uint4* wp =
        (const device uint4*)wbuf + (ulong)g_gid * (ulong)((uint)WM * G_NT);
    uint wacc = 0u;
    for (uint i = g_tid; i < (uint)WM * G_NT; i += G_NT) {
      uint4 v = wp[i];
      wacc ^= v.x; wacc ^= v.y; wacc ^= v.z; wacc ^= v.w;
    }
    atomic_fetch_xor_explicit(&g_xacc, wacc, memory_order_relaxed);
  }
"""

_EPILOGUE = r"""
  // ------------------------------------------------------------- diagnostics
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g_tid == 0u) {
    dbg[g_gid * 4u + 0u] = g_spins;
    dbg[g_gid * 4u + 1u] = g_to;
    dbg[g_gid * 4u + 2u] = atomic_load_explicit(&g_xacc, memory_order_relaxed);
    dbg[g_gid * 4u + 3u] = atomic_load_explicit(g_bar, memory_order_relaxed);
  }
"""

_FENCE_TEXT = "    metal::atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);\n"


def _barrier(bidx: int, fence: int) -> str:
    """One grid-wide crossing.

    FENCE levels, in increasing strength -- the experiment is which one is
    actually required for a cross-threadgroup read to be correct:

      0  threadgroup_barrier(mem_threadgroup) only.  No device ordering at all.
      1  threadgroup_barrier(mem_device | mem_threadgroup).  This is what
         prep/sweep3/r25_barrier_ladder.py used, where nothing crossed.
      2  level 1 + a device-scope atomic_thread_fence in THREAD 0 ONLY.
         This is the obvious-looking chain and it is WRONG: a fence orders the
         issuing thread's own memory operations, and thread 0 did not perform
         the other 1023 threads' stores to y1.
      3  level 1 + a device-scope atomic_thread_fence in EVERY THREAD on both
         sides of the rendezvous.  Every thread releases its own stores; every
         thread acquires before its own loads.
      4  level 3 + the cross-barrier reads issued through a `volatile device`
         pointer, so the compiler cannot reuse a pre-barrier value either.
    """
    tgb = ("threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);"
           if fence >= 1 else
           "threadgroup_barrier(mem_flags::mem_threadgroup);")
    all_fence = _FENCE_TEXT.replace("    ", "  ") if fence >= 3 else ""
    t0_fence = _FENCE_TEXT if fence == 2 else ""
    return (
        "  // ------------------------------ grid-wide barrier %d (bounded spin)\n" % bidx
        + all_fence                       # release: EVERY thread, before the rendezvous
        + "  %s\n" % tgb
        + "  if (g_tid == 0u) {\n"
        + t0_fence
        + "    uint spins = 0u;\n"
        + "    // The counter is NEVER reset.  Launches are strictly dependent, so\n"
        + "    // every threadgroup of launch r sees old in [r*NTG, (r+1)*NTG) and\n"
        + "    // derives the SAME base -- a monotonic barrier that survives being\n"
        + "    // reused by a chain of launches without a host-side zeroing pass.\n"
        + "    const uint old = atomic_fetch_add_explicit(g_bar, 1u,"
          " memory_order_relaxed);\n"
        + "    const uint target = (old / (uint)NTG) * (uint)NTG + (uint)NTG;\n"
        + "    while (atomic_load_explicit(g_bar, memory_order_relaxed) < target) {\n"
        + "      if (++spins >= (uint)SPINCAP) { break; }\n"
        + "    }\n"
        + t0_fence
        + "    g_spins = spins;\n"
        + "    g_to = (atomic_load_explicit(g_bar, memory_order_relaxed) >= target)"
          " ? 0u : 1u;\n"
        + "  }\n"
        + "  %s\n" % tgb
        + all_fence                       # acquire: EVERY thread, before its own loads
    )


def proto_source(fence: int, wstream: bool) -> str:
    """The full 2-layer persistent-kernel source."""
    # Layer 1: threadgroup g owns head g.       active for g < B*H
    # Layer 2: threadgroup g owns head g - ROT. active for ROT <= g < B*H + ROT
    ax = fence >= 5
    l1 = _layer_block(1, "g_gid", "g_gid < (uint)(BH)", chain_qkv=False,
                      atomic_xfer=ax)
    # Layer 2's map is rotated MODULO NTG, so the rotation also works when the
    # grid is exactly B*H (every threadgroup then swaps partner, none idles).
    bh2 = "(g_gid + (uint)NTG - (uint)ROT) % (uint)NTG"
    l2 = _layer_block(2, bh2, "(%s) < (uint)(BH)" % bh2, chain_qkv=True,
                      volatile_cross=(fence == 4), atomic_xfer=ax)
    return (
        _PRELUDE
        + l1
        + (_WSTREAM if wstream else "")
        + _barrier(0, fence)
        + l2
        + _EPILOGUE
    )


_IN = [
    "mq", "mk", "mv",
    "conv_state_all", "conv_w_all", "a_all", "bvec_all", "A_log_all",
    "dt_bias_all", "state_in_all", "gate_all", "o_w_all",
    "lower_bound", "qscale", "norm_eps", "valid", "bar", "wbuf", "xfer",
]
_OUT = ["y_1", "y_2", "state_out_all", "conv_state_out_all", "dbg"]

_CACHE = {}


def proto_kernel(fence: int = 2, wstream: bool = False):
    """Cached ``mx.fast.metal_kernel`` for one (fence, wstream) variant."""
    key = (fence, bool(wstream))
    if key not in _CACHE:
        if not mx.metal.is_available():
            return None
        _CACHE[key] = mx.fast.metal_kernel(
            name="glm5_r25_proto2_f%d_w%d" % (fence, int(bool(wstream))),
            input_names=_IN,
            output_names=_OUT,
            header=_HEADER + _XFER_HEADER,
            source=proto_source(fence, bool(wstream)),
        )
    return _CACHE[key]


def new_barrier_counter() -> mx.array:
    """A zeroed arrival counter.  Size 8 so MLX passes it in the *device* address
    space (its ``constant`` threshold is ``size < 8``), which the atomic cast needs.

    Zeroed on the host and never reset inside the kernel: MLX's ``init_value``
    would zero every OUTPUT too (one ``fill_gpu`` dispatch each), and extra
    dispatches are exactly the quantity this experiment measures.
    """
    c = mx.zeros((8,), dtype=mx.uint32)
    mx.eval(c)
    return c


_PACKED = ("conv_state", "conv_w", "a", "bvec", "A_log", "dt_bias", "state",
           "gate", "o_weight")
_PACK_NAME = {"state": "state_in_all", "o_weight": "o_w_all", "b": "bvec_all"}


def pack_layers(l1, l2):
    """Stack the two layers' tensors on a new leading axis, in the order the
    kernel's layer stride expects.  ``b`` is renamed ``bvec`` to match the shipped
    source's buffer name."""
    out = {}
    for k in _PACKED:
        src = "b" if k == "bvec" else k
        a, b = l1[src], l2[src]
        out[k] = mx.contiguous(mx.stack([a, b], axis=0))
    mx.eval(list(out.values()))
    return out


def proto_2layer(
    l1,
    packed,
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    lower_bound: float,
    norm_eps: float,
    bar: mx.array,
    wbuf: mx.array,
    xfer: mx.array = None,
    ntg: int,
    rot: int,
    ty: int = 32,
    fence: int = 2,
    wm: int = 0,
    spincap: int = 4096,
    mask=None,
):
    """One launch, two KDA layers, B>=1, S=1.

    ``l1`` supplies only the pre-conv q/k/v of layer 1 (layer 2 consumes layer
    1's ``y``); ``packed`` is ``pack_layers(l1, l2)``.

    Returns ``(y1, y2, state_out_all, conv_state_out_all, dbg)`` where the two
    ``*_all`` tensors carry layer 1 at index 0 and layer 2 at index 1.
    """
    H, D, K = num_heads, head_dim, conv_kernel_size
    B = l1["q_in"].shape[0]
    dt = l1["q_in"].dtype
    if ntg < B * H:
        raise ValueError("ntg=%d cannot host B*H=%d heads" % (ntg, B * H))
    if rot % ntg == 0 and rot != 0:
        raise ValueError("rot=%d is a no-op at ntg=%d" % (rot, ntg))
    valid = (mx.ones((B,), dtype=mx.bool_) if mask is None else mask.reshape(B))
    if fence >= 5 and dt.size != 2:
        raise ValueError("the atomic hand-off packs one 16-bit activation per "
                         "uint32 word; dtype %s is %d bytes" % (dt, dt.size))
    if xfer is None:
        xfer = mx.zeros((B * H * D,), dtype=mx.uint32)
        mx.eval(xfer)
    kernel = proto_kernel(fence, wm > 0)
    inputs = [
        l1["q_in"], l1["k_in"], l1["v_in"],
        packed["conv_state"], packed["conv_w"], packed["a"], packed["bvec"],
        packed["A_log"], packed["dt_bias"], packed["state"], packed["gate"],
        packed["o_weight"],
        float(lower_bound), float(head_dim ** -0.5), float(norm_eps), valid,
        bar, wbuf, xfer,
    ]
    return kernel(
        inputs=inputs,
        template=[
            ("T", dt), ("ST", packed["state"].dtype), ("H", H), ("D", D),
            ("K", K), ("TY", ty), ("NB", B), ("NTG", ntg), ("ROT", rot),
            ("BH", B * H), ("SPINCAP", spincap), ("WM", max(int(wm), 1)),
        ],
        grid=(32, ty, ntg),
        threadgroup=(32, ty, 1),
        output_shapes=[
            (B, 1, H * D), (B, 1, H * D),
            packed["state"].shape, packed["conv_state"].shape, (ntg * 4,),
        ],
        output_dtypes=[dt, dt, packed["state"].dtype, dt, mx.uint32],
    )


def reference_2layer(l1, l2, *, num_heads, head_dim, conv_kernel_size,
                     lower_bound, norm_eps, ty: int = 32, mask=None):
    """The thing the prototype must equal, bit for bit: the SHIPPED fused kernel
    run twice, back to back, layer 2 consuming layer 1's ``y``.

    Returns ``(y1, st1, cs1, y2, st2, cs2)``."""
    from .fused_kda import fused_kda_decode_step

    y1, st1, cs1 = fused_kda_decode_step(
        l1["q_in"], l1["k_in"], l1["v_in"], l1["conv_state"], l1["conv_w"],
        l1["a"], l1["b"], l1["A_log"], l1["dt_bias"], l1["state"], l1["gate"],
        l1["o_weight"], num_heads=num_heads, head_dim=head_dim,
        conv_kernel_size=conv_kernel_size, lower_bound=lower_bound,
        norm_eps=norm_eps, mask=mask, ty=ty,
    )
    y2, st2, cs2 = fused_kda_decode_step(
        y1, y1, y1, l2["conv_state"], l2["conv_w"],
        l2["a"], l2["b"], l2["A_log"], l2["dt_bias"], l2["state"], l2["gate"],
        l2["o_weight"], num_heads=num_heads, head_dim=head_dim,
        conv_kernel_size=conv_kernel_size, lower_bound=lower_bound,
        norm_eps=norm_eps, mask=mask, ty=ty,
    )
    return y1, st1, cs1, y2, st2, cs2
