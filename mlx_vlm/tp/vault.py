"""Per-rank, shard-local context vault for TP=2 serving.

THE WHOLE IDEA IN ONE SENTENCE: each rank saves and restores *its own* half of
the KV/recurrent state, and the only thing that crosses the wire is the name of
the boundary.

Why not ship the state.  The single-box vault stores ~140 MiB of KDA state plus
~28 KB/token of DSA latent; a 32k rung is about 1.0 GiB.  Under TP each rank
already holds exactly the half it computed -- the KDA recurrence is head-local
and the DSA latent is replicated -- so shipping anything would be paying 0.23 s
of wire to move state the receiver could not use and the sender already has.
Announcing a 128-bit name costs one control word group in a collective that was
going to happen anyway.

Why the identity must carry the topology.  A rung is a *shard* of a cache, and a
shard is only meaningful to a process holding the matching half of the weights.
Restoring rank 1's half into a single-box run, or a tp=2 rung into a tp=4 run,
produces a cache that is the right shape and the wrong contents -- the worst
kind of wrong.  :func:`topology_descriptor` is folded into the vault identity so
those stores cannot even name the same rung, and
:class:`~mlx_vlm.context_vault.VaultCheckpoint` carries its origin so a
checkpoint that crosses anyway is refused at restore time rather than believed.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ShardVault",
    "shard_vault",
    "topology_descriptor",
]


def topology_descriptor(report: Optional[dict], model_path: str = "") -> str:
    """Fingerprint of "which half of which model this process is holding".

    Built from the sharding report :func:`mlx_vlm.tp.shard.shard_model` returns,
    so it changes when the rank changes, when the world size changes, and when
    the layer split changes -- the three ways two stores could be shaped alike
    and mean different things.
    """
    r = report or {}
    parts = [
        f"tp={int(r.get('size', 1))}",
        f"rank={int(r.get('rank', 0))}",
        f"kda={int(r.get('kda_layers', 0))}",
        f"dsa={int(r.get('dsa_layers', 0))}",
        f"moe={int(r.get('moe_layers', 0))}",
        f"dense={int(r.get('dense_layers', 0))}",
        f"reduces={int(r.get('all_reduces_per_step', 0))}",
    ]
    if model_path:
        parts.append(f"model={model_path}")
    return "|".join(parts)


class ShardVault:
    """Name-keyed, budgeted, LRU store of this rank's own cache fragments.

    Deliberately *not* the trie-shaped :class:`~mlx_vlm.context_vault.ContextVault`:
    a mirror rank never sees tokens, so it has nothing to walk a trie with.  It
    is handed a name and asked to remember or reproduce.  Rank 0 keeps the
    token-shaped vault and does all the deciding; this is the half that only
    obeys.
    """

    def __init__(self, identity: str, budget_bytes: Optional[int] = None):
        from ..context_vault import vault_budget_bytes

        self.identity = identity
        self.budget = (
            vault_budget_bytes() if budget_bytes is None else int(budget_bytes)
        )
        self._lock = threading.RLock()
        self._rungs: Dict[str, Tuple[int, Any, int, float]] = {}
        self._resident = 0
        self.stores = self.hits = self.misses = self.evictions = 0

    # -- internals --------------------------------------------------------
    def _evict_until(self, headroom: int) -> None:
        if not self.budget:
            return
        while self._rungs and self._resident + headroom > self.budget:
            oldest = min(self._rungs.items(), key=lambda kv: kv[1][3])[0]
            self._resident -= self._rungs.pop(oldest)[2]
            self.evictions += 1

    # -- api --------------------------------------------------------------
    def store(self, name: str, prefix_len: int, caches: Sequence[Any]) -> bool:
        """Remember this rank's cache state as of ``prefix_len`` tokens."""
        from ..context_vault import (capture_fragments, eval_fragments,
                                     fragments_nbytes)

        if not name or prefix_len <= 0:
            return False
        frags = capture_fragments(caches, prefix_len)
        if frags is None:
            return False
        frags = list(frags)
        eval_fragments(frags)
        nbytes = fragments_nbytes(frags)
        with self._lock:
            if name in self._rungs:
                p, f, n, _ = self._rungs[name]
                self._rungs[name] = (p, f, n, time.monotonic())
                return False
            if self.budget and nbytes > self.budget:
                return False
            self._evict_until(nbytes)
            self._rungs[name] = (int(prefix_len), frags, nbytes, time.monotonic())
            self._resident += nbytes
            self.stores += 1
        return True

    def has(self, name: str, prefix_len: int) -> bool:
        with self._lock:
            rec = self._rungs.get(name)
        return rec is not None and rec[0] == int(prefix_len)

    def restore(self, name: str, prefix_len: int, caches: Sequence[Any]) -> bool:
        """Rebuild ``caches`` from the named rung. False when it is not held.

        The ``prefix_len`` check is not redundant with the name: the name is a
        hash of (identity, depth, tokens) so a match already implies the depth,
        but a stored rung is only usable if this process agrees about how many
        tokens it covers, and that agreement is cheap to assert.
        """
        from ..context_vault import restore_fragments

        with self._lock:
            rec = self._rungs.get(name)
            if rec is None or rec[0] != int(prefix_len):
                self.misses += 1
                return False
            self._rungs[name] = (rec[0], rec[1], rec[2], time.monotonic())
            frags = rec[1]
        ok = bool(restore_fragments(caches, frags))
        with self._lock:
            if ok:
                self.hits += 1
            else:
                self.misses += 1
        return ok

    def clear(self) -> None:
        with self._lock:
            self._rungs.clear()
            self._resident = 0

    @property
    def resident_bytes(self) -> int:
        return self._resident

    @property
    def rungs(self) -> int:
        return len(self._rungs)

    def stats_dict(self) -> dict:
        with self._lock:
            return {
                "identity": self.identity,
                "rungs": len(self._rungs),
                "bytes_resident": self._resident,
                "budget": self.budget,
                "stores": self.stores,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


_SHARD_VAULT: Optional[ShardVault] = None
_SHARD_LOCK = threading.Lock()


def shard_vault(identity: str, budget_bytes: Optional[int] = None) -> ShardVault:
    """Process-wide shard vault, rebuilt when the topology identity changes."""
    global _SHARD_VAULT
    with _SHARD_LOCK:
        if _SHARD_VAULT is None or _SHARD_VAULT.identity != identity:
            if _SHARD_VAULT is not None:
                logger.info(
                    "tp vault: topology changed (%s -> %s); dropping %d rungs",
                    _SHARD_VAULT.identity, identity, _SHARD_VAULT.rungs)
            _SHARD_VAULT = ShardVault(identity, budget_bytes)
        return _SHARD_VAULT


def reset_shard_vault() -> None:
    global _SHARD_VAULT
    with _SHARD_LOCK:
        _SHARD_VAULT = None


__all__ += ["reset_shard_vault"]
