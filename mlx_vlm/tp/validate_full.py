"""Whole-stack TP=2 validation in ONE process, two threads in lockstep.

Per-module validation proves each sharded piece sums correctly, but not that a
full layer composes -- mHC reads the hidden state between the attention and MLP
reduces, so a mistake there would not show up module by module.

Two threads each run their own sharded copy; every all-reduce is a barrier plus
a sum of both partials, which is exactly what the fleet does, minus the network.
"""

from __future__ import annotations

import copy
import json
import threading

import mlx.core as mx

from ..models.cache import ArraysCache, CacheList, KVCache
from ..models.glm5_next.config import TextConfig
from ..models.glm5_next.language import Glm5NextModel
from .glm5_next import shard_model
from .validate import TINY, TINY_Q, _quantize


class ThreadGroup:
    """all_sum over N threads: park, exchange, sum in rank order."""

    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slot = [None] * size

    def reducer(self, rank: int):
        def fn(x):
            # Materialize before publishing: MLX streams are thread-local, so a
            # peer cannot evaluate a graph node that belongs to this thread's
            # stream. An evaluated array carries no stream, so it crosses fine.
            # The real fleet has the same property for free -- the collective
            # hands back concrete memory.
            mx.eval(x)
            self.slot[rank] = x
            self.barrier.wait()
            total = self.slot[0]
            for r in range(1, self.size):
                total = total + self.slot[r]      # fixed order -> deterministic
            self.barrier.wait()
            return total

        return fn


def _caches(model):
    out = []
    for layer in model.layers:
        out.append(ArraysCache(size=2) if layer.is_linear
                   else CacheList(KVCache(), KVCache()))
    return out


def validate_full(size: int = 2, seed: int = 0, quant: bool = False,
                  steps: int = 3, verbose: bool = True) -> dict:
    # mx.compile caches a compiled function against the stream it was traced on,
    # and decode (B=1, S=1) takes the compiled FFN path plus the compiled
    # hc_expand -- so a worker thread hits "no Stream(gpu, 0) in current thread".
    # Compilation is semantics-preserving, so turning it off costs speed, not
    # correctness, and this harness only measures correctness.
    mx.disable_compile()
    mx.random.seed(seed)
    cfg = TextConfig.from_dict(TINY_Q if quant else TINY)
    ref = Glm5NextModel(cfg)
    if quant:
        _quantize(ref)
    mx.eval(ref.parameters())

    ids = mx.random.randint(0, cfg.vocab_size, (1, 1))
    mx.eval(ids)

    # reference: unsharded, `steps` decode steps through one cache
    for layer in ref.layers:
        layer.compile_ffn = False
    ref_cache = _caches(ref)
    ref_out = [ref(ids, cache=ref_cache) for _ in range(steps)]
    mx.eval(ref_out)

    group = ThreadGroup(size)
    shards, caches, outs = [], [], [None] * size
    for r in range(size):
        m = copy.deepcopy(ref)
        for layer in m.layers:
            layer.compile_ffn = False
        shard_model(m, r, size, reduce_fn=group.reducer(r))
        # the slices are lazy and bound to THIS thread's stream; materialize
        # them here or the workers cannot evaluate a graph that references them
        mx.eval(m.parameters())
        shards.append(m)
        caches.append(_caches(m))

    err = []

    def work(r):
        try:
            # MLX streams are thread-local: the main thread's default GPU stream
            # does not exist here, so each worker makes its own. Two ranks on
            # two streams is also what the fleet does.
            with mx.stream(mx.new_stream(mx.gpu)):
                o = [shards[r](ids, cache=caches[r]) for _ in range(steps)]
                mx.eval(o)
            outs[r] = o
        except Exception as e:  # noqa: BLE001
            import traceback as _tb
            err.append(f"rank{r}: {e!r}\n" + _tb.format_exc())
            try:
                group.barrier.abort()
            except Exception:
                pass

    ths = [threading.Thread(target=work, args=(r,)) for r in range(size)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=180)
    if any(t.is_alive() for t in ths):
        err.append("threads did not finish -- lockstep deadlock")
    if err:
        out = {"error": err}
        if verbose:
            print(json.dumps(out, indent=1))
        return out

    res = {"quant": quant, "steps": steps}
    for i in range(steps):
        a, b = outs[0][i], ref_out[i]
        res[f"step{i}_rel"] = float(
            mx.max(mx.abs(a - b)) / mx.maximum(mx.max(mx.abs(b)), 1e-6)
        )
        res[f"step{i}_ranks_agree"] = float(mx.max(mx.abs(outs[0][i] - outs[1][i])))
    if verbose:
        print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print("--- fp ---")
    validate_full()
    print("--- quantized ---")
    validate_full(quant=True)
