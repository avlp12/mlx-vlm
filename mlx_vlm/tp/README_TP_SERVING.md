# TP=2 serving mode

`MLX_VLM_GLM5_TP_HOSTS=10.0.0.1,10.0.0.2` turns it on. Absent, nothing changes.

    MLX_VLM_GLM5_TP_HOSTS          two or more IPs on the fast link; presence = on
    MLX_VLM_GLM5_TP_WORKER_PYTHON  interpreter for rank 1
    MLX_VLM_GLM5_TP_WORKER_SRC     source tree for rank 1
    MLX_VLM_GLM5_TP_WORKER_MODEL   peer's checkpoint path (defaults to rank 0's)
    MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD  control-plane payload width (8192)

Rank 1 is started for you over ssh. It runs `python -m mlx_vlm.tp.worker`, which
imports nothing from `mlx_vlm.server` — a rank-1 box needs no HTTP stack.

## Measured (3 pairs, alternated order, real-text 512-token prompt)

| | TP=2 | single | ratio |
|---|---|---|---|
| B=1 decode | 38.0 tok/s | 29.9 | **+27.4%** |
| B=8 decode | 161.1 tok/s | 109.0 | **+47.7%** |
| B=1 prefill | 633 tok/s | 378 | **+67.4%** |
| peak memory | 94.5 GiB/box | 183.2 GiB | **48% less per box** |

Spread within each arm is ≤0.5%.

## What TP mode does not do

* **Reproduce single-box tokens.** `all_sum` adds two partial sums where one
  device summed all 4096; at the measured one-ULP top-2 gap the argmax flips.
  Cross-lane token identity is a TP-off property. The TP-mode invariant is
  rank0 == rank1, established by the stage-3/4 driver (byte-identical over 257
  tokens) and inherited by the mirror (integrated output matches the driver's
  byte for byte).
* **Speculative decoding.** The drafter needs `capture_layer_ids`; the mirror
  refuses that rather than desynchronise the ranks. TODO: mirror the capture and
  rollback, or keep the drafter rank-0-only and broadcast only the verify.
* **Vault restore.** Reinstating a checkpoint gives rank 0 a cache rank 1 never
  saw. The control plane has MAKE_CACHE but no RESTORE. TODO.
* **Concurrent generations.** Mode-level TP assumes one live cache; the epoch
  map is there but only one cache is kept.

## Failure behaviour

Transport, worker launch, or sharded load failing logs and returns None, and the
server comes up single-box. Refusing to serve because a second box is
unreachable would be the worse failure.
