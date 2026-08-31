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

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx

from .apc_adapters import Capability, StateFragment, resolve_adapter

__all__ = [
    "ContextVault",
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


def capture_fragments(caches: Sequence[Any], prefix_len: int) -> Optional[List[StateFragment]]:
    """Detached snapshot of every cache entry at ``prefix_len``.

    Returns ``None`` if any component lacks a restore contract, so a partial
    ladder is never stored (a half-restored cache is silently wrong, not slow).
    """
    frags: List[StateFragment] = []
    for entry in caches:
        adapter = resolve_adapter(entry)
        if adapter.capability == Capability.UNSUPPORTED:
            return None
        frag = adapter.capture(entry, prefix_len)
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


@dataclass
class VaultCheckpoint:
    """One ladder rung: the full cache state as of ``prefix_len`` tokens."""

    prefix_len: int
    fragments: List[StateFragment]
    nbytes: int
    created: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    hits: int = 0


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
        self._root = _Node((), 0, None)
        self._lock = threading.RLock()
        self._resident = 0
        self._rungs = 0
        self.stats = VaultStats()

    # -- internals ------------------------------------------------------

    def _walk(self, tokens: Sequence[int]) -> Tuple[Optional[_Node], int]:
        """Deepest node whose path is a prefix of ``tokens``, plus its depth."""
        node = self._root
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

    def _insert_path(self, tokens: Sequence[int], depth: int) -> _Node:
        """Return the node at ``depth``, splitting edges as needed."""
        node = self._root
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
        while node is not None and node is not self._root:
            if node.checkpoint is not None or node.children:
                return
            parent = node.parent
            if parent is None:
                return
            parent.children.pop(node.edge[0], None)
            node.parent = None
            node = parent

    def _iter_nodes(self) -> Iterable[_Node]:
        stack = [self._root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    def _evict_until(self, headroom: int) -> None:
        if self.budget <= 0:
            return
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
            self._resident -= victim.checkpoint.nbytes
            self._rungs -= 1
            self.stats.evictions += 1
            victim.checkpoint = None
            self._prune(victim)

    # -- public API -----------------------------------------------------

    def lookup(self, tokens: Sequence[int]) -> Optional[VaultCheckpoint]:
        """Deepest stored checkpoint that prefixes ``tokens``, or ``None``."""
        with self._lock:
            self.stats.lookups += 1
            node, depth = self._walk(tokens)
            if node is None or node.checkpoint is None or depth == 0:
                self.stats.misses += 1
                return None
            cp = node.checkpoint
            cp.last_used = time.monotonic()
            cp.hits += 1
            self.stats.hits += 1
            self.stats.tokens_saved += depth
            return cp

    def insert(
        self,
        tokens: Sequence[int],
        prefix_len: int,
        fragments: Optional[Sequence[StateFragment]],
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
            node = self._insert_path(tokens, prefix_len)
            if node.checkpoint is not None:
                # Already stored at this exact depth; refresh recency only.
                node.checkpoint.last_used = time.monotonic()
                return False
            if self.budget and nbytes > self.budget:
                return False
            self._evict_until(nbytes)
            node.checkpoint = VaultCheckpoint(
                prefix_len=prefix_len, fragments=frags, nbytes=nbytes
            )
            self._resident += nbytes
            self._rungs += 1
            self.stats.inserts += 1
            self.stats.bytes_resident = self._resident
            return True

    def restore_into(self, caches: Sequence[Any], checkpoint: VaultCheckpoint) -> bool:
        return restore_fragments(caches, checkpoint.fragments)

    def clear(self) -> None:
        with self._lock:
            self._root = _Node((), 0, None)
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
    for name in _NUMERIC_TOGGLES:
        h.update(f"{name}={os.environ.get(name, '')}".encode())
    if extra:
        h.update(extra.encode())
    return h.hexdigest()[:32]


__all__ += ["vault_identity"]
