"""Warm Context Vault — process-resident RAM store for computed prefix state.

Motivation
----------
``apc_mode()`` classifies a glm5_next prompt cache as ``"exact"``: the layer
stack mixes ``ArraysCache`` (KDA conv + recurrent state, ``Capability.CHECKPOINT``)
with ``CacheList(KVCache, KVCache)`` (DSA latent + indexer pool). Only ``KVCache``
is block-pageable, so the whole cache degrades to whole-prefix reuse — a stored
131k document plus one new suffix token is a total miss and re-prefills from zero.

The vault converts "exact-only" into partial-prefix reuse for hybrid recurrent
models. A recurrent component cannot be sliced to an arbitrary length after the
fact: a hit at token N needs the state *as of* token N. So the vault snapshots the
complete cache at a ladder of chunk boundaries during prefill, indexes those
boundaries in a compressed radix trie keyed by the token prefix, and on lookup
restores the deepest boundary ``B <= N`` that prefixes the query, leaving only
tokens ``(B, N]`` to re-prefill.

Bit-identity contract
---------------------
Restore-plus-tail is bit-identical to a straight-through prefill **only when the
tail decomposes into the same chunk operations**. Boundaries are therefore
required to be multiples of ``prefill_step_size``; :func:`align_boundaries`
enforces this and the generate hook never clamps a chunk to land on a boundary
that is already aligned. KDA state is fp32 and DSA latents are stored dense, so
the restore itself is exact — no requantization, no lossy compression.

Sizing (measured, GLM-5.3-Flash 320B-A18B q4, see ~/glm53flash/logs/pipeline/)
-----------------------------------------------------------------------------
KDA state is *flat* in sequence length: 4.14 MiB/layer x 34 linear layers =
140.8 MiB, constant. DSA latent grows: 2562 B/tok/layer x 11 attention layers =
28182 B/tok. A checkpoint at length L therefore costs

    140.8 MiB + L * 28182 B

which lands a full 131k context at ~3.58 GB, matching the measured ~3.5 GB. The
flat KDA term is why a boundary ladder is affordable but not free: every extra
boundary re-pays 140.8 MiB regardless of depth.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx

from .apc_adapters import (
    Capability,
    StateFragment,
    _capture_one,
    resolve_adapter,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContextVault",
    "VaultTier",
    "record_session_turn",
    "lookup_session",
    "VaultCheckpoint",
    "VaultStats",
    "align_boundaries",
    "boundary_ladder",
    "capture_fragments",
    "CheckpointLadder",
    "default_boundary_stride",
    "fragments_nbytes",
    "get_vault",
    "reset_vault",
    "restore_fragments",
    "vault_budget_bytes",
    "vault_enabled",
]

_ENV_ENABLE = "MLX_VLM_GLM5_VAULT"
_ENV_BUDGET_GB = "MLX_VLM_GLM5_VAULT_BUDGET_GB"
_ENV_STRIDE = "MLX_VLM_GLM5_VAULT_STRIDE"
_ENV_MAX_LADDER = "MLX_VLM_GLM5_VAULT_MAX_LADDER"
_ENV_SESSION = "MLX_VLM_GLM5_VAULT_SESSION"
_ENV_SESSION_DERIVED_ID = "MLX_VLM_GLM5_VAULT_SESSION_DERIVED_ID"

_DEFAULT_BUDGET_GB = 256.0
_DEFAULT_STRIDE = 8192
_DEFAULT_MAX_LADDER = 8


def _env_truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def vault_enabled() -> bool:
    """True when ``MLX_VLM_GLM5_VAULT`` opts the vault in. Default off."""
    return _env_truthy(_ENV_ENABLE)


def vault_budget_bytes() -> int:
    """Resident budget in bytes (``MLX_VLM_GLM5_VAULT_BUDGET_GB``, default 256)."""
    try:
        gb = float(os.environ.get(_ENV_BUDGET_GB, _DEFAULT_BUDGET_GB))
    except (TypeError, ValueError):
        gb = _DEFAULT_BUDGET_GB
    return int(max(0.0, gb) * (1024**3))


# Why a session capture did not happen, counted by reason. Module-level rather
# than on the vault, because two of the reasons are "there is no vault" and "the
# flag is off" -- a counter that lives on the object cannot record its absence.
#
# THIS EXISTS BECAUSE THE FEATURE WAS INERT LIVE AND SAID NOTHING. On unified
# ff9a3045 the seven live checks found session_inserts stuck at 0 with an empty
# log: capture_session has five early returns and none of them spoke, so from
# outside a disabled feature and a broken one are the same picture. That is the
# same failure the APC default had, built into my own code while fixing theirs.
SESSION_SKIPS: Dict[str, int] = {}
_SKIP_LOCK = threading.Lock()


def record_session_skip(reason: str) -> None:
    with _SKIP_LOCK:
        SESSION_SKIPS[reason] = SESSION_SKIPS.get(reason, 0) + 1


def session_skip_counts() -> Dict[str, int]:
    with _SKIP_LOCK:
        return dict(SESSION_SKIPS)


def reset_session_skips() -> None:
    with _SKIP_LOCK:
        SESSION_SKIPS.clear()


def session_capture_enabled() -> bool:
    """End-of-turn capture, ``MLX_VLM_GLM5_VAULT_SESSION``.  Default OFF.

    Off until it has been validated on a live server: it fires on the RESPONSE
    path, which is the one place in this stack where a mistake reaches the
    client rather than the log.
    """
    return _env_truthy(_ENV_SESSION)


def derived_session_id_allowed() -> bool:
    """Whether a missing session id may be derived from the tokens.  Default OFF.

    ``session_id_for`` hashes the first 64 tokens.  In production those are
    almost always the SYSTEM PROMPT, so every conversation on a server would
    hash to one id, collapse into one eviction group, and deep-first eviction
    would start shedding the deepest rung of unrelated conversations.  Nothing
    becomes incorrect -- the trie still keys on the full prefix -- but the
    graceful-degradation property that justifies deep-first eviction is gone.

    So the id is REQUIRED from the caller in production, and the derivation is
    available only behind this flag, for tests and single-tenant experiments
    where the collapse is harmless and understood.
    """
    return _env_truthy(_ENV_SESSION_DERIVED_ID)


def default_boundary_stride() -> int:
    """Boundary spacing in tokens (``MLX_VLM_GLM5_VAULT_STRIDE``, default 8192)."""
    try:
        stride = int(os.environ.get(_ENV_STRIDE, _DEFAULT_STRIDE))
    except (TypeError, ValueError):
        stride = _DEFAULT_STRIDE
    return max(1, stride)


def _max_ladder() -> int:
    try:
        n = int(os.environ.get(_ENV_MAX_LADDER, _DEFAULT_MAX_LADDER))
    except (TypeError, ValueError):
        n = _DEFAULT_MAX_LADDER
    return max(1, n)


# --------------------------------------------------------------------------
# Fragment capture / restore (thin wrappers over the APC adapter contract)
# --------------------------------------------------------------------------


def capture_fragments(
    caches: Sequence[Any], prefix_len: int, adopt: bool = False
) -> Optional[List[StateFragment]]:
    """Detached snapshot of every cache entry at ``prefix_len``.

    Returns ``None`` if any component lacks a restore contract, so a partial
    ladder is never stored (a half-restored cache is silently wrong, not slow).

    ``adopt`` skips the defensive copy and takes the caller's buffers.  Only for
    a cache the caller will never touch again -- see ``_adopt_tree``.
    """
    frags: List[StateFragment] = []
    for entry in caches:
        adapter = resolve_adapter(entry)
        if adapter.capability == Capability.UNSUPPORTED:
            return None
        frag = _capture_one(adapter, entry, prefix_len, adopt)
        if frag is None:
            return None
        frags.append(frag)
    return frags


def restore_fragments(caches: Sequence[Any], fragments: Sequence[StateFragment]) -> bool:
    """Restore ``fragments`` into fresh ``make_cache()``-shaped ``caches``."""
    if len(caches) != len(fragments):
        return False
    for entry, frag in zip(caches, fragments):
        resolve_adapter(entry).restore(entry, frag)
    return True


def _tree_nbytes(obj: Any) -> int:
    if isinstance(obj, mx.array):
        return int(obj.nbytes)
    if isinstance(obj, (list, tuple)):
        return sum(_tree_nbytes(o) for o in obj)
    if isinstance(obj, dict):
        return sum(_tree_nbytes(v) for v in obj.values())
    if isinstance(obj, StateFragment):
        return _tree_nbytes(obj.payload)
    return 0


def fragments_nbytes(fragments: Sequence[StateFragment]) -> int:
    """Resident byte cost of a captured ladder rung."""
    return sum(_tree_nbytes(f) for f in fragments)


def eval_fragments(fragments: Sequence[StateFragment]) -> None:
    """Force materialization so the vault owns real buffers, not lazy graphs."""
    targets: List[mx.array] = []
    for f in fragments:
        targets.extend(f.eval_targets())
    if targets:
        mx.eval(targets)


# --------------------------------------------------------------------------
# Boundary policy
# --------------------------------------------------------------------------


def align_boundaries(boundaries: Iterable[int], step: Optional[int], total: int) -> List[int]:
    """Keep only boundaries that preserve the chunk decomposition.

    A boundary is admissible when it is a positive multiple of ``step`` and
    strictly below ``total`` (the final token is consumed by the decode step,
    never by chunked prefill). Without this filter the generate loop clamps a
    chunk to land on the boundary, which changes matmul shapes and breaks
    bit-identity against an unaligned baseline.
    """
    if total <= 0:
        return []
    out = set()
    for b in boundaries:
        b = int(b)
        if b <= 0 or b >= total:
            continue
        if step and step > 0 and b % step != 0:
            continue
        out.add(b)
    return sorted(out)


def boundary_ladder(
    total: int,
    stride: Optional[int] = None,
    step: Optional[int] = None,
    max_rungs: Optional[int] = None,
    mode: str = "geometric",
) -> List[int]:
    """Boundary positions for a prompt of ``total`` tokens.

    ``geometric`` (default) places the deepest admissible boundary and then
    halves toward the head. This beats uniform spacing on both axes that matter:
    the deepest rung serves the dominant "same document, new suffix" workload,
    and the halving tail gives log-depth coverage for queries that diverge early
    without re-paying the flat 140.8 MiB KDA cost at every stride.

    At 131072 tokens with stride 8192, geometric yields 5 rungs costing ~7.6 GB;
    uniform-keep-deepest yields 8 rungs clustered in the last 60k costing
    ~22.4 GB for strictly worse early-divergence coverage.

    ``uniform`` keeps every-``stride`` spacing for callers that want dense late
    coverage and have the budget for it.
    """
    stride = stride or default_boundary_stride()
    max_rungs = max_rungs or _max_ladder()
    if total <= stride:
        return []
    if mode == "uniform":
        raw = list(range(stride, total, stride))
        aligned = align_boundaries(raw, step, total)
        return aligned[-max_rungs:] if len(aligned) > max_rungs else aligned

    deepest = ((total - 1) // stride) * stride
    raw: List[int] = []
    cur = deepest
    while cur >= stride and len(raw) < max_rungs:
        raw.append(cur)
        nxt = (cur // 2 // stride) * stride
        if nxt >= cur:
            break
        cur = nxt
    return align_boundaries(raw, step, total)


class CheckpointLadder:
    """Consumes an ascending boundary list across a chunked prefill loop.

    Extracted from the generate loop so the ordering contract is testable
    without a model: ``clamp`` shortens a chunk so it lands exactly on the next
    boundary, and ``reached`` reports boundaries hit exactly (a boundary that a
    caller overshot is dropped, never fired late with the wrong prefix_len).
    """

    __slots__ = ("pending",)

    def __init__(self, boundaries: Iterable[int], total: int):
        self.pending: List[int] = sorted(
            {int(b) for b in boundaries if 0 < int(b) < total}
        )

    def __bool__(self) -> bool:
        return bool(self.pending)

    def clamp(self, processed: int, n_to_process: int) -> int:
        """Shorten ``n_to_process`` so the step ends on the next boundary."""
        if self.pending and processed < self.pending[0] < processed + n_to_process:
            return self.pending[0] - processed
        return n_to_process

    def reached(self, processed: int) -> List[int]:
        """Boundaries satisfied exactly at ``processed`` tokens."""
        out: List[int] = []
        while self.pending and processed >= self.pending[0]:
            b = self.pending.pop(0)
            if b == processed:
                out.append(b)
        return out


# --------------------------------------------------------------------------
# Compressed radix trie over token prefixes
# --------------------------------------------------------------------------


class VaultTier(str, Enum):
    """Which guarantee a rung carries.  These are NOT interchangeable.

    PREFILL -- the shipped tier.  Rungs land on multiples of prefill_step_size
    (:func:`align_boundaries`), so restore-plus-tail is bit-identical to a
    straight-through cold prefill.

    SESSION -- taken when a response finishes, at whatever length that is, after
    a suffix that was decoded one token at a time.  That is already a different
    chunk decomposition from a 2048-token prefill chunk, before the store is
    involved at all, so this tier CANNOT claim cold-prefill equality and does not
    try to.  Its guarantee is "bit-identical to continuing the same session",
    which is exactly the semantics a returning conversation wants.

    The tiers live in separate tries rather than in one trie with a filter, so a
    prefill lookup cannot reach a session rung even if a future caller forgets to
    pass a tier.  Making it structural is the whole point: a session rung served
    to a prefill query would be a fluent wrong answer, not a slow one.
    """

    PREFILL = "prefill"
    SESSION = "session"


@dataclass
class VaultCheckpoint:
    """One ladder rung: the full cache state as of ``prefix_len`` tokens."""

    prefix_len: int
    fragments: List[StateFragment]
    nbytes: int
    created: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    hits: int = 0
    # The vault identity that produced this rung. Carried on the checkpoint,
    # not merely on the store, because a checkpoint is the thing that travels:
    # between vaults in-process, across the peer-tier wire, and -- once TP
    # exists -- between processes holding different halves of the weights.
    # Restoring a shard of somebody else's topology yields a cache of exactly
    # the right shape and entirely the wrong contents.
    origin: str = ""
    # Which guarantee this rung carries; travels with the checkpoint for the
    # same reason ``origin`` does -- a rung can reach restore_into without ever
    # having been looked up in this vault.
    tier: VaultTier = VaultTier.PREFILL
    # Groups the rungs of one conversation so eviction can shed a session's
    # DEEPEST rung first and degrade to the prefill ladder instead of to cold.
    session_id: str = ""
    expires_at: Optional[float] = None

    def expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.monotonic()) >= self.expires_at


class _Node:
    __slots__ = ("edge", "children", "checkpoint", "depth", "parent")

    def __init__(self, edge: Tuple[int, ...], depth: int, parent: Optional["_Node"]):
        self.edge = edge
        self.depth = depth
        self.parent = parent
        self.children: Dict[int, "_Node"] = {}
        self.checkpoint: Optional[VaultCheckpoint] = None


def _common_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class VaultStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0
    tokens_saved: int = 0
    bytes_resident: int = 0
    rejected_unsupported: int = 0
    rejected_foreign: int = 0
    rejected_tier: int = 0
    expired: int = 0
    session_inserts: int = 0
    session_hits: int = 0

    def as_dict(self, budget: int, rungs: int) -> dict:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / self.lookups) if self.lookups else 0.0,
            "inserts": self.inserts,
            "evictions": self.evictions,
            "tokens_saved": self.tokens_saved,
            "rungs_resident": rungs,
            "bytes_resident": self.bytes_resident,
            "gb_resident": self.bytes_resident / (1024**3),
            "budget_bytes": budget,
            "budget_gb": budget / (1024**3),
            "utilization": (self.bytes_resident / budget) if budget else 0.0,
            "rejected_unsupported": self.rejected_unsupported,
            "rejected_foreign": self.rejected_foreign,
            "rejected_tier": self.rejected_tier,
            "expired": self.expired,
            "session_inserts": self.session_inserts,
            "session_hits": self.session_hits,
        }


class ContextVault:
    """Process-resident, budgeted, LRU store of prefix checkpoints.

    Keyed by ``identity`` (model + code + numeric-toggle fingerprint) so a model
    swap or a kernel-affecting branch change cannot serve stale state. Thread-safe
    for the single-process serving lane.
    """

    def __init__(self, identity: str, budget_bytes: Optional[int] = None):
        self.identity = identity
        self.budget = vault_budget_bytes() if budget_bytes is None else int(budget_bytes)
        self._roots = {t: _Node((), 0, None) for t in VaultTier}
        self._lock = threading.RLock()
        self._resident = 0
        self._rungs = 0
        self.stats = VaultStats()
        # Optional cold tier (:mod:`mlx_vlm.vault_disk`).  None -- the default --
        # is the vault exactly as it was before the disk tier existed: every
        # hook below is a no-op, so nothing about RAM behaviour is conditional
        # on a feature that is off.  Set by ``vault_disk.attach_disk_vault``.
        self.disk = None

    # -- internals ------------------------------------------------------

    @property
    def _root(self) -> _Node:
        """The prefill trie.  Kept as a property so every pre-tier caller and
        test that reaches for ``_root`` still means what it used to mean."""
        return self._roots[VaultTier.PREFILL]

    def _walk(
        self, tokens: Sequence[int], tier: VaultTier = VaultTier.PREFILL
    ) -> Tuple[Optional[_Node], int]:
        """Deepest node whose path is a prefix of ``tokens``, plus its depth."""
        node = self._roots[tier]
        best: Optional[_Node] = None
        pos = 0
        if node.checkpoint is not None:
            best = node
        while pos < len(tokens):
            child = node.children.get(tokens[pos])
            if child is None:
                break
            edge = child.edge
            if len(tokens) - pos < len(edge):
                # Partial edge match cannot reach the child's boundary.
                break
            if _common_len(edge, tokens[pos : pos + len(edge)]) != len(edge):
                break
            pos += len(edge)
            node = child
            if node.checkpoint is not None:
                best = node
        return best, (best.depth if best else 0)

    def _insert_path(
        self, tokens: Sequence[int], depth: int, tier: VaultTier = VaultTier.PREFILL
    ) -> _Node:
        """Return the node at ``depth``, splitting edges as needed."""
        node = self._roots[tier]
        pos = 0
        while pos < depth:
            first = tokens[pos]
            child = node.children.get(first)
            if child is None:
                edge = tuple(tokens[pos:depth])
                new = _Node(edge, depth, node)
                node.children[first] = new
                return new
            edge = child.edge
            shared = _common_len(edge, tokens[pos : depth])
            if shared == len(edge):
                pos += shared
                node = child
                continue
            # Split the existing edge at ``shared``.
            mid = _Node(edge[:shared], node.depth + shared, node)
            child.edge = edge[shared:]
            child.parent = mid
            mid.children[child.edge[0]] = child
            node.children[first] = mid
            node = mid
            pos += shared
            if pos == depth:
                return node
            tail = tuple(tokens[pos:depth])
            new = _Node(tail, depth, node)
            node.children[tail[0]] = new
            return new
        return node

    def _prune(self, node: Optional[_Node]) -> None:
        roots = set(id(r) for r in self._roots.values())
        while node is not None and id(node) not in roots:
            if node.checkpoint is not None or node.children:
                return
            parent = node.parent
            if parent is None:
                return
            parent.children.pop(node.edge[0], None)
            node.parent = None
            node = parent

    def _iter_nodes(self) -> Iterable[_Node]:
        stack = list(self._roots.values())
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    def _sweep_expired(self) -> int:
        """Drop TTL-expired rungs.  Returns bytes reclaimed."""
        now = time.monotonic()
        freed = 0
        for n in list(self._iter_nodes()):
            cp = n.checkpoint
            if cp is not None and cp.expired(now):
                freed += cp.nbytes
                self._resident -= cp.nbytes
                self._rungs -= 1
                self.stats.expired += 1
                n.checkpoint = None
                self._prune(n)
        return freed

    def _deepest_of_session(self, session_id: str) -> Optional[_Node]:
        """The deepest surviving rung of one conversation."""
        best: Optional[_Node] = None
        for n in self._iter_nodes():
            cp = n.checkpoint
            if cp is None or cp.session_id != session_id:
                continue
            if best is None or cp.prefix_len > best.checkpoint.prefix_len:
                best = n
        return best

    @staticmethod
    def _tokens_for_node(node: _Node) -> Optional[List[int]]:
        """The token prefix that names ``node``, read back off the trie edges.

        The trie stores each rung's tokens split across the edges on its root
        path, so a checkpoint about to be evicted can still be *named* -- which
        is what the disk tier needs and what a bare ``VaultCheckpoint`` does not
        carry.  Returns None if the reconstruction disagrees with the node's own
        depth, because a mislabelled disk entry would be found by the wrong
        prompt later.
        """
        parts: List[Tuple[int, ...]] = []
        n: Optional[_Node] = node
        while n is not None and n.parent is not None:
            parts.append(n.edge)
            n = n.parent
        toks: List[int] = []
        for edge in reversed(parts):
            toks.extend(int(t) for t in edge)
        return toks if len(toks) == node.depth else None

    def _offload(self, node: _Node, reason: str) -> None:
        """Hand a rung to the disk tier.  Never blocks, never raises.

        Called with ``self._lock`` held and on the generation thread, so the
        contract with :meth:`vault_disk.DiskPrefixVault.save_async` is that it
        only enqueues -- a device write on this thread would trade the stall the
        vault exists to remove for a smaller one.
        """
        disk = getattr(self, "disk", None)
        cp = node.checkpoint
        if disk is None or cp is None:
            return
        try:
            toks = self._tokens_for_node(node)
            if toks is None:
                return
            disk.save_async(toks, cp, reason=reason)
        except Exception:  # noqa: BLE001 - the cold tier must never fail a request
            logger.warning("vault: disk offload failed; the rung is simply lost",
                           exc_info=True)

    def _evict_until(self, headroom: int) -> None:
        if self.budget <= 0:
            return
        if self._resident + headroom > self.budget:
            self._sweep_expired()
        while self._resident + headroom > self.budget:
            victim: Optional[_Node] = None
            for n in self._iter_nodes():
                cp = n.checkpoint
                if cp is None:
                    continue
                if victim is None or cp.last_used < victim.checkpoint.last_used:
                    victim = n
            if victim is None:
                return
            # A session degrades from its deep end: shedding the deepest rung
            # costs the tail, shedding a shallow one costs everything below it.
            # So the LRU scan picks the SESSION, and the session picks the rung.
            # Prefill rungs keep the old behaviour exactly (session_id is "").
            if victim.checkpoint.session_id:
                deepest = self._deepest_of_session(victim.checkpoint.session_id)
                if deepest is not None:
                    victim = deepest
            # Save BEFORE dropping: this is the entry whose 444 s of prefill is
            # about to be thrown away, so it is exactly the one worth 0.55 s of
            # SSD (P2, sweep11).  The write is queued, not performed here.
            self._offload(victim, "evict")
            self._resident -= victim.checkpoint.nbytes
            self._rungs -= 1
            self.stats.evictions += 1
            victim.checkpoint = None
            self._prune(victim)

    # -- public API -----------------------------------------------------

    def lookup(
        self, tokens: Sequence[int], tier: VaultTier = VaultTier.PREFILL
    ) -> Optional[VaultCheckpoint]:
        """Deepest stored rung of ``tier`` that prefixes ``tokens``."""
        with self._lock:
            self.stats.lookups += 1
            self._sweep_expired()
            node, _ = self._walk(tokens, tier)
            if node is None or node.checkpoint is None:
                self.stats.misses += 1
                return None
            cp = node.checkpoint
            cp.last_used = time.monotonic()
            cp.hits += 1
            self.stats.hits += 1
            if tier is VaultTier.SESSION:
                self.stats.session_hits += 1
            self.stats.tokens_saved += cp.prefix_len
            return cp

    def insert(
        self,
        tokens: Sequence[int],
        prefix_len: int,
        fragments: Optional[Sequence[StateFragment]],
        tier: VaultTier = VaultTier.PREFILL,
        session_id: str = "",
        ttl_s: Optional[float] = None,
    ) -> bool:
        """Store the cache state as of ``prefix_len`` tokens of ``tokens``."""
        if fragments is None:
            with self._lock:
                self.stats.rejected_unsupported += 1
            return False
        if prefix_len <= 0 or prefix_len > len(tokens):
            return False
        frags = list(fragments)
        eval_fragments(frags)
        nbytes = fragments_nbytes(frags)
        with self._lock:
            node = self._insert_path(tokens, prefix_len, tier)
            if node.checkpoint is not None:
                # Already stored at this exact depth; refresh recency only.
                node.checkpoint.last_used = time.monotonic()
                return False
            if self.budget and nbytes > self.budget:
                return False
            self._evict_until(nbytes)
            node.checkpoint = VaultCheckpoint(
                prefix_len=prefix_len,
                fragments=frags,
                nbytes=nbytes,
                origin=self.identity,
                tier=tier,
                session_id=session_id,
                expires_at=(time.monotonic() + ttl_s) if ttl_s else None,
            )
            self._resident += nbytes
            self._rungs += 1
            self.stats.inserts += 1
            if tier is VaultTier.SESSION:
                self.stats.session_inserts += 1
            self.stats.bytes_resident = self._resident
            if getattr(getattr(self, "disk", None), "save_on_insert", False):
                # Off by default: eviction-time saving writes each rung at most
                # once and only the rungs that would otherwise be lost, while
                # insert-time saving writes every rung including the ones that
                # are about to be superseded by a deeper one on the same ladder.
                self._offload(node, "insert")
            return True

    def restore_into(
        self,
        caches: Sequence[Any],
        checkpoint: VaultCheckpoint,
        tier: VaultTier = VaultTier.PREFILL,
    ) -> bool:
        """Rebuild ``caches`` from ``checkpoint``, unless it came from elsewhere.

        The refusal is not belt-and-braces over the identity-keyed store: a
        checkpoint can reach this method without ever having been in this vault
        (the peer tier hands one over; a TP rank could be handed the other
        rank's half). Shapes would match and the restore would "work", so the
        check has to be on provenance rather than on structure. Refusing costs
        a cold prefill; accepting costs a fluent wrong answer.
        """
        got = getattr(checkpoint, "tier", VaultTier.PREFILL)
        if got != tier:
            self.stats.rejected_tier += 1
            logger.warning(
                "vault: refusing a %s rung for a %s restore; the two tiers carry "
                "different guarantees and a session rung is not bit-identical to "
                "a cold prefill", got, tier)
            return False
        origin = getattr(checkpoint, "origin", "")
        if origin and origin != self.identity:
            self.stats.rejected_foreign += 1
            logger.warning(
                "vault: refusing a checkpoint from a different topology/build "
                "(rung origin %s..., this vault %s...); falling back to a cold "
                "prefill", origin[:12], self.identity[:12])
            return False
        return restore_fragments(caches, checkpoint.fragments)

    def clear(self) -> None:
        with self._lock:
            self._roots = {t: _Node((), 0, None) for t in VaultTier}
            self._resident = 0
            self._rungs = 0
            self.stats.bytes_resident = 0

    @property
    def resident_bytes(self) -> int:
        return self._resident

    @property
    def rungs(self) -> int:
        return self._rungs

    def stats_dict(self) -> dict:
        with self._lock:
            self.stats.bytes_resident = self._resident
            return self.stats.as_dict(self.budget, self._rungs)


# --------------------------------------------------------------------------
# Process-wide handle, keyed by model + code identity
# --------------------------------------------------------------------------

_VAULT: Optional[ContextVault] = None
_VAULT_LOCK = threading.Lock()


def get_vault(identity: str, budget_bytes: Optional[int] = None) -> ContextVault:
    """Return the process vault, rebuilding it if ``identity`` changed.

    Identity change is the invalidation event: a different model tree, a
    different code revision, or a different numeric-toggle set all produce
    different cache contents for the same tokens, so the old store is dropped
    wholesale rather than risking a stale restore.
    """
    global _VAULT
    with _VAULT_LOCK:
        if _VAULT is None or _VAULT.identity != identity:
            _VAULT = ContextVault(identity, budget_bytes)
        return _VAULT


def reset_vault() -> None:
    global _VAULT
    with _VAULT_LOCK:
        _VAULT = None


# --------------------------------------------------------------------------
# Identity: what makes a stored checkpoint valid to serve
# --------------------------------------------------------------------------

# Env toggles that change kernel selection and therefore cache *contents*.
# Anything that can alter a stored tensor bit-for-bit must be fingerprinted, or
# the vault will happily serve state computed by a different kernel.
_NUMERIC_TOGGLES = (
    "MLX_VLM_GLM5_FUSED_KDA",
    "MLX_VLM_GLM5_FUSED_QPROJ",
    "MLX_VLM_GLM5_QPROJ",
    "MLX_VLM_DFLASH_COMPILE",
    "MLX_VLM_GLM5_SPARSE",
    "MLX_VLM_GLM5_DSA_FUSED",
)


# --------------------------------------------------------------------------
# Sharding topology: which half of which model this process is holding
# --------------------------------------------------------------------------
#
# A cache is only meaningful to a process holding the weights that produced it.
# Under TP each rank holds half of them, so a rung stored by rank 1 describes
# rank 1's half of the state and nothing else -- restoring it into a single-box
# run, or into rank 0, or into a tp=4 run, produces a cache of exactly the right
# shape and entirely the wrong contents. Folding the topology into the identity
# means those stores cannot even agree on a rung *name*, which is the level at
# which the peer tier and the TP mirror both address each other.
#
# Default "tp1" is the single-box case, so an unsharded process keeps the
# identity it has always had modulo this constant, and every sharded process
# differs from it and from its peer.

_TP_TOPOLOGY = "tp1"
_TOPOLOGY_LOCK = threading.Lock()


def set_tp_topology(descriptor: Optional[str]) -> str:
    """Declare this process's sharding topology; returns the value in force.

    Called once by the TP loader with the sharding report. Changing it resets
    the process vault, because every rung already stored was named under the
    old topology and is no longer nameable -- keeping them would be paying
    memory for entries that can never be found again.
    """
    global _TP_TOPOLOGY
    with _TOPOLOGY_LOCK:
        new = descriptor or "tp1"
        if new != _TP_TOPOLOGY:
            logger.info("vault: topology %s -> %s; dropping the store",
                        _TP_TOPOLOGY, new)
            _TP_TOPOLOGY = new
            reset_vault()
        return _TP_TOPOLOGY


def tp_topology() -> str:
    return _TP_TOPOLOGY


__all__ += ["set_tp_topology", "tp_topology"]


def _code_identity() -> str:
    """Revision of the running ``mlx_vlm`` tree, or a mtime fallback."""
    import subprocess
    from pathlib import Path

    pkg = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "-C", str(pkg), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        rev = out.stdout.strip()
        if rev:
            dirty = subprocess.run(
                ["git", "-C", str(pkg), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            return f"{rev[:12]}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        newest = max(p.stat().st_mtime for p in pkg.rglob("*.py"))
        return f"mtime-{int(newest)}"
    except (OSError, ValueError):
        return "unknown"


def vault_identity(model_path: Any, extra: str = "") -> str:
    """Fingerprint of everything that determines cache contents.

    ``model_path`` identifies the weights; ``_code_identity`` the kernels; the
    toggle set the numeric path taken through them. A change in any of the three
    invalidates the whole store (see :func:`get_vault`).
    """
    import hashlib
    from pathlib import Path

    h = hashlib.sha256()
    p = str(model_path)
    h.update(p.encode())
    try:
        cfg = Path(p) / "config.json"
        if cfg.is_file():
            st = cfg.stat()
            h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    except OSError:
        pass
    h.update(_code_identity().encode())
    # Folded in here rather than at the call sites so no caller can forget it.
    h.update(f"topology={tp_topology()}".encode())
    for name in _NUMERIC_TOGGLES:
        h.update(f"{name}={os.environ.get(name, '')}".encode())
    if extra:
        h.update(extra.encode())
    return h.hexdigest()[:32]


__all__ += ["vault_identity"]


def vault_identity_for_model(model: Any) -> str:
    """Identity derived from a loaded model when no path is at hand.

    Uses the shape-and-quantization fingerprint of the config rather than the
    weights themselves: two trees that agree on all of these produce identical
    cache tensors for identical tokens, and any that differ do not.
    """
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    parts = [str(getattr(cfg, "model_type", "?"))]
    for attr in (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "index_head_dim",
        "index_topk",
        "first_k_dense_replace",
    ):
        parts.append(f"{attr}={getattr(text_cfg, attr, None)}")
    quant = getattr(cfg, "quantization", None)
    if isinstance(quant, dict):
        parts.append(f"q={quant.get('bits')}/{quant.get('group_size')}/{quant.get('mode')}")
    path = getattr(cfg, "_name_or_path", None) or getattr(model, "model_path", None) or ""
    return vault_identity(path, extra="|".join(parts))


__all__ += ["vault_identity_for_model"]


# --------------------------------------------------------------------------
# Session tier: end-of-turn capture, keyed by prompt + response
# --------------------------------------------------------------------------


def session_id_for(tokens: Sequence[int], turns: int = 0) -> str:
    """Stable id for the conversation ``tokens`` belongs to.

    Derived from the first 64 tokens, which no turn ever rewrites, so every turn
    of one conversation lands in the same eviction group without the server
    having to thread a session handle through.  A collision costs eviction
    granularity, never correctness: the trie still keys on the full prefix.
    """
    import hashlib

    h = hashlib.sha256()
    for t in list(tokens)[:64]:
        h.update(int(t).to_bytes(8, "little", signed=True))
    return h.hexdigest()[:16]


def prefix_len_from_cache(caches: Sequence[Any]) -> Optional[int]:
    """How many tokens this cache actually holds, asked of the cache itself.

    Not computed from the prompt length plus a token count.  At end of turn the
    last sampled token has NOT been fed back through the model, and the exact
    bookkeeping differs between the padded and unpadded batch paths
    (``_row_real_tokens_processed`` counts prompt columns only and knows nothing
    about decode).  The offset the attention caches carry is the one number that
    is true by construction.

    Returns None -- meaning "do not store" -- when the offsets DISAGREE.  A cache
    whose halves think they hold different numbers of tokens is silently wrong
    rather than slow, and the campaign's rule for that is to refuse.  Components
    that are flat in sequence length (the KDA ``ArraysCache``) carry no offset
    and are ignored.
    """
    offsets = set()
    for entry in caches or ():
        subs = getattr(entry, "caches", None)
        for sub in (subs if subs is not None else [entry]):
            off = getattr(sub, "offset", None)
            if off is not None:
                offsets.add(int(off))
    if len(offsets) != 1:
        return None
    n = offsets.pop()
    return n if n > 0 else None


def record_session_turn(
    vault: Optional[ContextVault],
    tokens: Sequence[int],
    caches: Sequence[Any],
    *,
    completed: bool,
    session_id: str = "",
    ttl_s: Optional[float] = None,
    adopt: bool = True,
    prefix_len: Optional[int] = None,
) -> bool:
    """Store the cache as of the end of a finished turn.  Never raises.

    This is the whole point of the session tier.  The prefill ladder's deepest
    rung sits at the last boundary BELOW the prompt, so a returning conversation
    re-prefills its own last turn plus everything generated since; a rung taken
    here leaves only the new user message.

    ``completed=False`` stores nothing: a truncated response is a prefix no
    future turn will send, so the rung would be pure eviction pressure, and an
    abort mid-token may leave a cache that matches no complete token sequence.

    A fault here is on the RESPONSE path, where raising would surface to the
    client, so every failure is logged and swallowed -- the cost of losing a rung
    is a cold prefill next turn, which is exactly the status quo.
    """
    if vault is None:
        record_session_skip("no_vault")
        # TP disables the vault wholesale (server/generation.py:1286-1290): the
        # request path carries no rungs to rank 1.  The session tier inherits
        # that refusal rather than adding a second, differently-wrong one.
        return False
    if not completed:
        record_session_skip("not_completed")
        return False
    try:
        toks = list(tokens)
        if not toks:
            record_session_skip("empty_tokens")
            return False
        sid = session_id
        if not sid:
            if not derived_session_id_allowed():
                record_session_skip("no_session_id")
                logger.warning(
                    "vault: refusing a session capture with no session_id; the "
                    "caller must pass its conversation id (see "
                    "derived_session_id_allowed for why the token-derived "
                    "fallback is not safe in production)")
                return False
            sid = session_id_for(toks)
        n = prefix_len if prefix_len is not None else prefix_len_from_cache(caches)
        if n is None or n <= 0:
            record_session_skip("no_cache_offset")
            return False
        if n > len(toks):
            record_session_skip("cache_longer_than_key")
            # The cache holds more than the key describes; the key would not
            # identify the state. Refuse rather than store a mislabelled rung.
            logger.warning("vault: session capture skipped, cache holds %d tokens "
                           "but the key has %d", n, len(toks))
            return False
        frags = capture_fragments(caches, n, adopt=adopt)
        if frags is None:
            record_session_skip("uncapturable_cache")
            return False
        return vault.insert(
            toks,
            n,
            frags,
            tier=VaultTier.SESSION,
            session_id=sid,
            ttl_s=ttl_s,
        )
    except Exception:  # noqa: BLE001 - a vault fault must never fail a response
        record_session_skip("exception")
        logger.warning("vault: session capture failed; continuing without it",
                       exc_info=True)
        return False


def lookup_session(
    vault: Optional[ContextVault], tokens: Sequence[int]
) -> Optional[VaultCheckpoint]:
    """Deepest session rung that prefixes ``tokens``; never raises."""
    if vault is None:
        return None
    try:
        return vault.lookup(tokens, tier=VaultTier.SESSION)
    except Exception:  # noqa: BLE001
        logger.warning("vault: session lookup failed; falling back to a cold "
                       "prefill", exc_info=True)
        return None


def restore_session(
    vault: Optional[ContextVault],
    caches: Sequence[Any],
    checkpoint: VaultCheckpoint,
) -> bool:
    """Restore a SESSION rung.  Refuses a prefill rung; never raises."""
    if vault is None:
        return False
    try:
        return vault.restore_into(caches, checkpoint, tier=VaultTier.SESSION)
    except Exception:  # noqa: BLE001
        logger.warning("vault: session restore failed; falling back to a cold "
                       "prefill", exc_info=True)
        return False


__all__ += ["session_id_for", "restore_session", "prefix_len_from_cache",
            "session_skip_counts", "reset_session_skips", "record_session_skip",
            "session_capture_enabled", "derived_session_id_allowed"]
