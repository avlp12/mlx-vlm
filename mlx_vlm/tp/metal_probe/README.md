# Metal crossing-cost probes

Standalone Swift, no MLX. Build: `swiftc -O -framework Metal -framework
Foundation -framework QuartzCore decomp.swift -o decomp`.

`decomp.swift` answers the question that decides whether a distributed
collective can live inside a GPU timeline on Apple silicon, by separating the
three things an evented crossing actually pays for. M3 Ultra, N=101,
nelem=262144, min over 60 reps (contention can only inflate a timing):

| | us/op |
|---|---|
| E0 all dispatches in ONE encoder | 15.8 |
| E1 one encoder per dispatch (close/reopen), no events | 15.6 |
| E2 pure MTLSharedEvent ping-pong, no encoders at all | **1.8** |
| E3 dispatch + close + signal + wait, live spinning host | 144.5 |
| **(a) encoder close/reopen** | **-0.2 (free)** |
| **(b) MTLSharedEvent round trip** | **1.8** |
| **(c) GPU pipeline drain/refill forced by a mid-buffer wait** | **127.1** |

The event is not slow and closing an encoder is not slow. What costs 127 us is
that `encodeWaitForEvent` in the middle of a command buffer makes the GPU drain
its pipeline and restart it. It scales with how much work is in flight -- the
added cost is 115.8 / 152.4 / 182.1 us at nelem 65536 / 262144 / 1048576 --
which is the signature of a drain, not of a fixed handshake.

Consequence: a design that fences a host collective with MTLSharedEvent pays
~145 us per crossing and cannot work at decode cadence (101 crossings/step).
A design that never blocks the command buffer -- dispatch a one-thread kernel
that spins on a shared-memory counter, inside the already-open encoder -- pays
~5 us. MLX already implements the second one as its "fast" fence
(`mlx/backend/metal/fence.cpp`, `use_fast`, `MLX_METAL_FAST_SYNCH=1`); its
default fence uses the first (`CommandEncoder::signal_event`/`wait_event` in
`mlx/backend/metal/device.cpp`, both of which call `end_encoding()` first).
