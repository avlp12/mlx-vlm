"""Rank-0 side of the shard-local vault: decide locally, announce the name.

Rank 0 keeps the ordinary token-shaped :class:`~mlx_vlm.context_vault.ContextVault`
-- it is the rank that sees tokens, so it is the rank that can walk a prefix
trie and decide which rung is worth restoring.  This wrapper adds exactly one
thing: every store and every restore is announced to rank 1 first, by *name*, so
rank 1 performs the same operation on its own half of the state.

The asymmetry is deliberate.  Rank 1 is a mirror, not a peer: it never decides
anything, and the one fact it contributes -- "I do / do not hold that rung" --
travels back over a swapped-roles all_sum rather than a socket, because a second
transport is exactly what the mirror design exists to avoid.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["MirroredVault"]


class MirroredVault:
    """A :class:`ContextVault` whose rungs exist on both ranks or not at all."""

    def __init__(self, vault, mirror):
        self._vault = vault
        self._mirror = mirror
        self._lock = threading.Lock()
        # The tokens the last lookup was made with, paired with the checkpoint
        # it returned.  ``restore_into`` needs them to name the boundary, and
        # pairing them with the checkpoint object means a caller that
        # interleaves requests gets a refusal (and a cold prefill) rather than
        # a rung named after somebody else's prompt.
        self._pending: Optional[tuple] = None
        self.peer_misses = 0

    def __getattr__(self, name):
        return getattr(self._vault, name)

    # -- naming -----------------------------------------------------------
    def _name(self, tokens: Sequence[int], depth: int) -> str:
        from ..context_vault_wire import boundary_hash

        return boundary_hash(tokens, int(depth), self._vault.identity)

    # -- the three verbs dispatch uses ------------------------------------
    def lookup(self, tokens: Sequence[int]):
        hit = self._vault.lookup(tokens)
        with self._lock:
            self._pending = (list(tokens), hit) if hit is not None else None
        return hit

    def insert(self, tokens: Sequence[int], prefix_len: int, fragments) -> bool:
        """Store rank 0's half, and tell rank 1 to store its own.

        Announced *before* the local insert, while both ranks' caches are still
        exactly at ``prefix_len``: rank 1 is blocked in the control collective
        immediately after the same prefill chunk, so the message finds it in the
        state the rung is supposed to describe.
        """
        if fragments is None:
            return self._vault.insert(tokens, prefix_len, fragments)
        try:
            self._mirror.announce_vault_store(
                self._name(tokens, prefix_len), int(prefix_len))
        except Exception:  # noqa: BLE001 - storing is best-effort on both ranks
            logger.warning("tp vault: store announce failed", exc_info=True)
            return False
        return self._vault.insert(tokens, prefix_len, fragments)

    def restore_into(self, caches: Sequence[Any], checkpoint) -> bool:
        """Restore both halves, or neither.

        Rank 1 goes first and answers; only if it holds the rung does rank 0
        restore its own.  The alternative -- restore locally, ask afterwards --
        would leave rank 0 warm and rank 1 cold in the window between, and the
        sum of those two halves is not a wrong answer that looks wrong.
        """
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None or pending[1] is not checkpoint:
            logger.info("tp vault: restore without a matching lookup; cold prefill")
            return False
        tokens, _ = pending
        depth = int(checkpoint.prefix_len)
        name = self._name(tokens, depth)
        try:
            peer_ok = self._mirror.announce_vault_restore(caches, name, depth)
        except Exception:  # noqa: BLE001
            logger.warning("tp vault: restore announce failed", exc_info=True)
            return False
        if not peer_ok:
            self.peer_misses += 1
            return False
        return bool(self._vault.restore_into(caches, checkpoint))

    # -- passthrough ------------------------------------------------------
    def stats_dict(self) -> dict:
        d = dict(self._vault.stats_dict())
        d["peer_misses"] = self.peer_misses
        return d
