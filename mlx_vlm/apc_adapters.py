"""Pluggable cache-component adapters for Automatic Prefix Caching (issue #1629)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import mlx.core as mx

ADAPTER_SCHEMA_VERSION = 1


class Capability(str, Enum):
    """Reuse capability of a single cache component."""

    PAGEABLE = "pageable"
    WINDOWED = "windowed"
    CHECKPOINT = "checkpoint"
    COMPOSITE = "composite"
    UNSUPPORTED = "unsupported"


def _copy_array(x: mx.array) -> mx.array:
    """Materialize ``x`` into a fresh MLX-owned contiguous buffer (detach)."""
    return mx.contiguous(mx.array(x, dtype=x.dtype))


_ALIAS_TAG = "__vault_alias__"
_ENV_DEDUP = "MLX_VLM_GLM5_VAULT_DEDUP"


def dedup_enabled() -> bool:
    """Whether capture collapses byte-identical sibling arrays. Default ON.

    Bit-identical by construction (see :func:`_bytes_identical`), so it takes the
    playbook's default-ON identity tier.  ``MLX_VLM_GLM5_VAULT_DEDUP=0`` is the
    A/B kill switch and restores the pre-dedup payload shape exactly.

    Deliberately NOT read once and cached: the tests toggle it in-process, and a
    cached read is how ``_adaptive_k_enabled``-style globals become untestable.

    Deliberately NOT added to ``context_vault._NUMERIC_TOGGLES`` either.  That
    tuple invalidates the whole store when a toggle changes the *contents* of a
    cache; this toggle changes only how those contents are stored, and a rung
    captured under either setting restores to the same bytes (pinned by
    ``test_off_switch_reproduces_the_undeduped_payload``).  Folding it in would
    throw away a valid store for no numeric reason.
    """
    return os.environ.get(_ENV_DEDUP, "1").strip().lower() not in ("0", "false", "no", "off")


_UINT_OF_SIZE = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}


def _bytes_identical(a: mx.array, b: mx.array) -> bool:
    """Do ``a`` and ``b`` have identical BIT PATTERNS (not merely equal values)?

    ``mx.array_equal`` is the wrong predicate here and both of its failure modes
    are live on this cache:

      * NaN != NaN, so two byte-identical buffers containing NaN compare unequal
        and the dedup would silently not fire (measured: array_equal False,
        bit-compare True);
      * +0.0 == -0.0, so two buffers that differ in their sign bit compare EQUAL
        and a dedup driven by it would change the restored bytes (measured:
        array_equal True, bit-compare False).

    Reinterpreting through an unsigned integer of the same width makes the test
    exact in both directions, which is what the vault's warm-equals-cold gate
    needs.  Anything unexpected (odd dtype, view refusal) returns False, so the
    failure mode is "did not save memory", never "restored the wrong bytes".
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.size == 0:
        return True
    try:
        u = _UINT_OF_SIZE.get(a.dtype.size)
        if u is None:
            return False
        return bool(mx.array_equal(mx.view(a, u), mx.view(b, u)).item())
    except Exception:  # noqa: BLE001 - a dedup fault must never fail a capture
        return False


def _dedup_seq(items: Sequence[Any], snap) -> List[Any]:
    """Snapshot a tuple/list, replacing a byte-identical repeat with an alias.

    Only siblings inside the same container are compared, which is O(k^2) in a
    container that holds 2 entries in practice and keeps the scan off the rest of
    the tree.  The concrete win this exists for: glm5_next caches the DSA latent
    as BOTH keys and values (models/glm5_next/language.py:1416 passes the same
    array twice into KVCache.update_and_fetch, and models/cache.py:353-354
    allocates two separate buffers for it), so 1024 of every 2562 B/token/layer
    the vault stores is a byte-exact duplicate -- 40.0% of the growing term.
    """
    out: List[Any] = []
    kept: List[tuple] = []  # (position in out, snapshotted array)
    for item in items:
        if isinstance(item, mx.array):
            hit = None
            for pos, prev in kept:
                if _bytes_identical(prev, item):
                    hit = pos
                    break
            if hit is not None:
                out.append((_ALIAS_TAG, hit))
                continue
            c = snap(item)
            kept.append((len(out), c))
            out.append(c)
        else:
            out.append(snap(item))
    return out


def _expand_seq(items: Sequence[Any], snap) -> List[Any]:
    """Inverse of :func:`_dedup_seq`.

    Each alias is materialised as its OWN buffer, never as a second reference to
    the original: ``KVCache.update_and_fetch`` writes into ``self.keys`` and
    ``self.values`` in place (models/cache.py:365-366), so handing it two names
    for one buffer would corrupt the cache on the first decode step after a
    restore.  The wire tier turns tuples into tuples faithfully
    (context_vault_wire.py:106/126), so an alias survives a peer fetch as either
    a tuple or a list; both spellings are accepted.
    """
    out: List[Any] = []
    for item in items:
        if _is_alias(item):
            out.append(_copy_array(out[int(item[1])]))
        else:
            out.append(snap(item))
    return out


def _is_alias(node: Any) -> bool:
    return (
        isinstance(node, (tuple, list))
        and len(node) == 2
        and isinstance(node[0], str)
        and node[0] == _ALIAS_TAG
        and isinstance(node[1], int)
        and not isinstance(node[1], bool)
    )


def _adopt_tree(obj: Any) -> Any:
    """Like :func:`_snapshot_tree` but WITHOUT copying the arrays.

    Only sound when the caller is handing over a cache it will never touch
    again -- the end-of-turn session capture, where the request is finished and
    the cache is about to be dropped anyway.  Deduplication still runs, so an
    adopted DSA layer still stores the latent once.

    Honest limit, because it bounds the saving: ``KVCache.state`` returns
    ``keys[..., :offset, :]`` whenever the buffer is over-allocated (cache.py:338
    grows in steps of 256), and evaluating that slice materialises a new buffer
    regardless of what we do here.  So adoption avoids the copy outright only
    when ``offset`` is a multiple of the step; otherwise it avoids the SECOND
    copy (the snapshot on top of the slice) and no more.  Storing the raw buffer
    plus the offset would avoid it in every case, but that needs a restore path
    that sets ``.offset`` directly instead of going through the ``state`` setter,
    which infers ``offset`` from the array shape -- left for later, deliberately,
    because getting it wrong turns padding into real tokens.
    """
    if isinstance(obj, mx.array):
        return obj
    if _is_alias(obj):
        return obj
    if isinstance(obj, tuple):
        if dedup_enabled():
            return tuple(_dedup_seq(obj, _adopt_tree))
        return tuple(_adopt_tree(o) for o in obj)
    if isinstance(obj, list):
        if dedup_enabled():
            return _dedup_seq(obj, _adopt_tree)
        return [_adopt_tree(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _adopt_tree(v) for k, v in obj.items()}
    return obj


def _snapshot_tree(obj: Any) -> Any:
    """Deep-copy the MLX arrays inside a state tree; pass scalars/None through."""
    if isinstance(obj, mx.array):
        return _copy_array(obj)
    if _is_alias(obj):
        return obj
    if isinstance(obj, tuple):
        if dedup_enabled():
            return tuple(_dedup_seq(obj, _snapshot_tree))
        return tuple(_snapshot_tree(o) for o in obj)
    if isinstance(obj, list):
        if dedup_enabled():
            return _dedup_seq(obj, _snapshot_tree)
        return [_snapshot_tree(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _snapshot_tree(v) for k, v in obj.items()}
    return obj


def _expand_tree(obj: Any) -> Any:
    """Deep-copy a stored payload back into fresh buffers, expanding aliases."""
    if isinstance(obj, mx.array):
        return _copy_array(obj)
    if _is_alias(obj):
        raise ValueError("vault: alias marker outside a sequence container")
    if isinstance(obj, tuple):
        return tuple(_expand_seq(obj, _expand_tree))
    if isinstance(obj, list):
        return _expand_seq(obj, _expand_tree)
    if isinstance(obj, dict):
        return {k: _expand_tree(v) for k, v in obj.items()}
    return obj


def _eval_tree(obj: Any, out: List[mx.array]) -> None:
    if isinstance(obj, mx.array):
        out.append(obj)
    elif isinstance(obj, StateFragment):
        _eval_tree(obj.payload, out)
    elif _is_alias(obj):
        return
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _eval_tree(o, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _eval_tree(v, out)


@dataclass
class StateFragment:
    """Captured, restorable state for one cache component at a boundary."""

    capability: Capability
    prefix_len: int
    payload: Any
    schema: str = f"v{ADAPTER_SCHEMA_VERSION}"

    def eval_targets(self) -> List[mx.array]:
        out: List[mx.array] = []
        _eval_tree(self.payload, out)
        return out


@runtime_checkable
class PrefixStateAdapter(Protocol):
    """Capture/restore/merge contract for one cache-component kind."""

    capability: Capability

    def capture(
        self, cache: Any, prefix_len: int, adopt: bool = False
    ) -> Optional[StateFragment]: ...

    def restore(self, fresh_cache: Any, fragment: StateFragment) -> None: ...

    def merge_rows(
        self, caches: Sequence[Any], prefix_lens: Sequence[int]
    ) -> Optional[Any]: ...

    def serialize(self, fragment: StateFragment) -> Any: ...

    def deserialize(self, tree: Any) -> StateFragment: ...


def _capture_one(adapter: Any, cache: Any, prefix_len: int, adopt: bool):
    """Call ``adapter.capture``, tolerating adapters that predate ``adopt``.

    The default path is byte-for-byte the old two-argument call, so an adapter
    outside this file is unaffected until somebody asks for adoption; if it then
    cannot do it, we fall back to copying rather than failing the capture.
    """
    if not adopt:
        return adapter.capture(cache, prefix_len)
    try:
        return adapter.capture(cache, prefix_len, adopt=True)
    except TypeError:
        return adapter.capture(cache, prefix_len)


def _is_snapshotable(cache: Any) -> bool:
    """True if ``cache`` exposes a restorable snapshot contract."""
    if callable(getattr(cache, "prefix_cache_snapshot", None)):
        return True

    return hasattr(cache, "state") and hasattr(cache, "meta_state")


class CheckpointAdapter:
    """Universal fallback: snapshot ``state`` + ``meta_state`` as an opaque blob."""

    capability = Capability.CHECKPOINT

    def capture(
        self, cache: Any, prefix_len: int, adopt: bool = False
    ) -> Optional[StateFragment]:
        if not _is_snapshotable(cache):
            return None
        snap = getattr(cache, "prefix_cache_snapshot", None)
        raw = (
            snap()
            if callable(snap)
            else {
                "state": cache.state,
                "meta_state": cache.meta_state,
            }
        )
        take = _adopt_tree if adopt else _snapshot_tree
        return StateFragment(self.capability, prefix_len, payload=take(raw))

    def restore(self, fresh_cache: Any, fragment: StateFragment) -> None:
        payload = _expand_tree(fragment.payload)
        restore = getattr(fresh_cache, "prefix_cache_restore", None)
        if callable(restore):
            restore(payload)
        else:
            fresh_cache.state = payload["state"]
            fresh_cache.meta_state = payload["meta_state"]

    def merge_rows(
        self, caches: Sequence[Any], prefix_lens: Sequence[int]
    ) -> Optional[Any]:

        first = caches[0] if caches else None
        merge = getattr(first, "prefix_cache_merge", None)
        return merge(caches, prefix_lens) if callable(merge) else None

    def serialize(self, fragment: StateFragment) -> Any:
        return {
            "capability": fragment.capability.value,
            "prefix_len": fragment.prefix_len,
            "schema": fragment.schema,
            "payload": fragment.payload,
        }

    def deserialize(self, tree: Any) -> StateFragment:
        return StateFragment(
            Capability(tree["capability"]),
            int(tree["prefix_len"]),
            payload=tree["payload"],
            schema=tree.get("schema", "v1"),
        )


class CompositeAdapter:
    """Recurses into ``CacheList`` / tuple caches, one sub-adapter per child."""

    capability = Capability.COMPOSITE

    def _children(self, cache: Any) -> Optional[Sequence[Any]]:
        if isinstance(cache, tuple):
            return list(cache)
        subs = getattr(cache, "caches", None)
        return list(subs) if subs is not None else None

    def capture(
        self, cache: Any, prefix_len: int, adopt: bool = False
    ) -> Optional[StateFragment]:
        children = self._children(cache)
        if children is None:
            return None
        frags = []
        for sub in children:
            adapter = resolve_adapter(sub)
            frag = _capture_one(adapter, sub, prefix_len, adopt)
            if frag is None:
                return None
            frags.append(frag)
        return StateFragment(self.capability, prefix_len, payload=frags)

    def restore(self, fresh_cache: Any, fragment: StateFragment) -> None:
        children = self._children(fresh_cache)
        for sub, frag in zip(children, fragment.payload):
            resolve_adapter(sub).restore(sub, frag)

    def merge_rows(
        self, caches: Sequence[Any], prefix_lens: Sequence[int]
    ) -> Optional[Any]:
        return None

    def serialize(self, fragment: StateFragment) -> Any:
        return {
            "capability": fragment.capability.value,
            "prefix_len": fragment.prefix_len,
            "children": [
                resolve_adapter_by_capability(f.capability).serialize(f)
                for f in fragment.payload
            ],
        }

    def deserialize(self, tree: Any) -> StateFragment:
        children = [
            resolve_adapter_by_capability(Capability(c["capability"])).deserialize(c)
            for c in tree["children"]
        ]
        return StateFragment(self.capability, int(tree["prefix_len"]), payload=children)


_CAPABILITY: Dict[type, Capability] = {}
_ADAPTERS: Dict[Capability, PrefixStateAdapter] = {
    Capability.PAGEABLE: CheckpointAdapter(),
    Capability.WINDOWED: CheckpointAdapter(),
    Capability.CHECKPOINT: CheckpointAdapter(),
    Capability.COMPOSITE: CompositeAdapter(),
}


def register_capability(cls: type, capability: Capability) -> None:
    _CAPABILITY[cls] = capability


def register_default_capabilities() -> None:
    """Register the in-tree cache classes' declared capabilities."""
    from .models import cache as c

    for cls in (c.KVCache, c.QuantizedKVCache, c.ChunkedKVCache):
        register_capability(cls, Capability.PAGEABLE)
    for cls in (c.RotatingKVCache,):
        register_capability(cls, Capability.WINDOWED)
    for cls in (c.ArraysCache, c.PoolingCache, c.StaticPrefixKVCache):
        register_capability(cls, Capability.CHECKPOINT)
    register_capability(c.CacheList, Capability.COMPOSITE)


def resolve_capability(
    cache: Any, overrides: Optional[Dict[type, Capability]] = None
) -> Capability:
    """Resolve the capability of ``cache``."""
    if not _CAPABILITY:
        register_default_capabilities()
    t = type(cache)
    if isinstance(cache, tuple):
        return Capability.COMPOSITE
    if overrides and t in overrides:
        return overrides[t]
    if t in _CAPABILITY:
        return _CAPABILITY[t]
    for base in t.__mro__[1:]:
        if base in _CAPABILITY:
            cap = _CAPABILITY[base]
            return Capability.CHECKPOINT if cap == Capability.PAGEABLE else cap
    if _is_snapshotable(cache):
        return Capability.CHECKPOINT
    return Capability.UNSUPPORTED


def resolve_adapter(
    cache: Any, overrides: Optional[Dict[type, Capability]] = None
) -> PrefixStateAdapter:
    return _ADAPTERS[resolve_capability(cache, overrides)]


def resolve_adapter_by_capability(cap: Capability) -> PrefixStateAdapter:
    return _ADAPTERS[cap]


_APC_EXACT_TYPES: Optional[tuple] = None
_APC_BLOCK_TYPES: Optional[set] = None


def _apc_type_tables():
    global _APC_EXACT_TYPES, _APC_BLOCK_TYPES
    if _APC_EXACT_TYPES is None:
        from .models import cache as c

        _APC_EXACT_TYPES = (
            c.KVCache,
            c.BatchKVCache,
            c.BatchRotatingKVCache,
            c.BatchQuantizedKVCache,
            c.RotatingKVCache,
            c.ChunkedKVCache,
            c.ArraysCache,
        )
        _APC_BLOCK_TYPES = {c.KVCache}
    return _APC_EXACT_TYPES, _APC_BLOCK_TYPES


def apc_block_eligible(cache: Any) -> bool:
    """True if ``cache`` supports block-level (pageable KV) APC reuse."""
    if hasattr(cache, "dequantize_for_apc"):
        return True
    _, block_types = _apc_type_tables()
    return type(cache) in block_types


def apc_exact_eligible(cache: Any) -> bool:
    """True if ``cache`` supports exact whole-prefix snapshot APC reuse."""
    from .models import cache as c

    exact_types, _ = _apc_type_tables()
    if isinstance(cache, exact_types) or hasattr(cache, "dequantize_for_apc"):
        return True
    if isinstance(cache, c.CacheList):
        return all(apc_exact_eligible(s) for s in cache.caches)
    if isinstance(cache, tuple):
        return all(apc_exact_eligible(s) for s in cache)
    return _custom_state_contract(cache)


def apc_mode(caches: Sequence[Any]) -> Optional[str]:
    """APC strategy for a prompt cache: ``"block"``, ``"exact"`` or ``None``."""
    if not caches:
        return None
    if all(apc_block_eligible(c) for c in caches):
        return "block"
    if all(apc_exact_eligible(c) for c in caches):
        return "exact"
    return None


def _apc_array_helpers():
    from .apc import _copy_mlx_array, _pad_kv_for_capacity

    return _copy_mlx_array, _pad_kv_for_capacity


class KVCacheCloneAdapter:
    capability = Capability.PAGEABLE

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        copy, pad = _apc_array_helpers()
        out = type(c)()
        off = int(getattr(c, "offset", 0) or 0)
        if c.keys is not None and c.values is not None and off > 0:
            keys = copy(c.keys[..., :off, :])
            values = copy(c.values[..., :off, :])
            step = int(getattr(c, "step", getattr(type(c), "step", 256)) or 0)
            keys, values = pad(
                keys,
                values,
                offset=off,
                min_capacity_tokens=min_capacity_tokens,
                step=step,
            )
            out.keys, out.values, out.offset = keys, values, off
            eval_targets.extend([keys, values])
        return out

    def merge_rows(self, caches, prefix_lens):
        from .models import cache as lm

        return lm.BatchKVCache.merge(caches)


class RotatingKVCacheCloneAdapter:
    capability = Capability.WINDOWED

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        copy, _ = _apc_array_helpers()
        out = type(c)(max_size=int(c.max_size), keep=int(getattr(c, "keep", 0)))
        out.offset = int(getattr(c, "offset", 0) or 0)
        out._idx = int(getattr(c, "_idx", 0) or 0)
        if c.keys is not None and c.values is not None:
            out.keys, out.values = copy(c.keys), copy(c.values)
            eval_targets.extend([out.keys, out.values])
        return out

    def merge_rows(self, caches, prefix_lens):
        from .models import cache as lm

        return lm.BatchRotatingKVCache.merge(caches)


class ChunkedKVCacheCloneAdapter:
    capability = Capability.PAGEABLE

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        copy, _ = _apc_array_helpers()
        out = type(c)(chunk_size=int(c.chunk_size))
        out.offset = int(getattr(c, "offset", 0) or 0)
        out.start_position = int(getattr(c, "start_position", 0) or 0)
        if c.keys is not None and c.values is not None:
            out.keys, out.values = copy(c.keys), copy(c.values)
            eval_targets.extend([out.keys, out.values])
        return out

    def merge_rows(self, caches, prefix_lens):
        from .models import cache as lm

        return lm.BatchKVCache.merge(caches)


class ArraysCacheCloneAdapter:
    capability = Capability.CHECKPOINT

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        from .models import cache as lm

        copy, _ = _apc_array_helpers()
        out = lm.ArraysCache(len(c.cache))
        out.cache = []
        for state in c.cache:
            if state is None:
                out.cache.append(None)
                continue
            cp = copy(state)
            out.cache.append(cp)
            eval_targets.append(cp)
        if c.left_padding is not None:
            out.left_padding = copy(c.left_padding)
            eval_targets.append(out.left_padding)
        if c.lengths is not None:
            out.lengths = copy(c.lengths)
            eval_targets.append(out.lengths)
        return out

    def merge_rows(self, caches, prefix_lens):
        from .models import cache as lm

        size = len(caches[0].cache)
        out = lm.ArraysCache(size)
        merged: List[Optional[mx.array]] = []
        for i in range(size):
            states = [c.cache[i] for c in caches]
            sample = next((s for s in states if s is not None), None)
            if sample is None:
                merged.append(None)
                continue
            rows = [
                (
                    mx.zeros((1,) + sample.shape[1:], dtype=sample.dtype)
                    if s is None
                    else s[:1]
                )
                for s in states
            ]
            merged.append(mx.concatenate(rows, axis=0))
        out.cache = merged
        return out


_CLONE_RULES: Optional[list] = None


def _clone_rules():
    global _CLONE_RULES
    if _CLONE_RULES is None:
        from .models import cache as lm

        _CLONE_RULES = [
            (lm.KVCache, KVCacheCloneAdapter()),
            (lm.RotatingKVCache, RotatingKVCacheCloneAdapter()),
            (lm.ChunkedKVCache, ChunkedKVCacheCloneAdapter()),
            (lm.ArraysCache, ArraysCacheCloneAdapter()),
        ]
    return _CLONE_RULES


def _custom_state_contract(c) -> bool:
    """True if ``c`` defines its own ``state`` property, not the trivial base one."""
    from .models.cache import _BaseCache

    base_state = _BaseCache.__dict__.get("state")
    for klass in type(c).__mro__:
        if "state" in klass.__dict__:
            return klass.__dict__["state"] is not base_state
    return False


def _state_clone(c, eval_targets):
    """Clone via the state/meta_state contract (from_state), detaching arrays."""
    detached = _snapshot_tree(c.state)
    _eval_tree(detached, eval_targets)
    from_state = getattr(type(c), "from_state", None)
    if callable(from_state):
        return from_state(detached, c.meta_state)
    out = type(c).__new__(type(c))
    out.state = detached
    out.meta_state = c.meta_state
    return out


def clone_cache_entry(c, *, min_capacity_tokens, eval_targets):
    from .models import cache as lm

    if callable(getattr(c, "extract", None)) and callable(
        getattr(c, "is_single_row", None)
    ):
        if not c.is_single_row():
            return None
        if c.empty():
            if isinstance(c, lm.BatchRotatingKVCache):
                return lm.RotatingKVCache(max_size=int(c.max_size))
            return lm.KVCache()
        return clone_cache_entry(
            c.extract(0),
            min_capacity_tokens=min_capacity_tokens,
            eval_targets=eval_targets,
        )
    for typ, adapter in _clone_rules():

        matched = type(c) is typ if typ is lm.KVCache else isinstance(c, typ)
        if matched:
            return adapter.clone(
                c, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
            )
    if isinstance(c, lm.CacheList):
        subs = [
            clone_cache_entry(
                s, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
            )
            for s in c.caches
        ]
        return None if any(s is None for s in subs) else lm.CacheList(*subs)
    if isinstance(c, tuple):
        subs = [
            clone_cache_entry(
                s, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
            )
            for s in c
        ]
        return None if any(s is None for s in subs) else tuple(subs)
    if hasattr(c, "dequantize_for_apc"):
        copy, _ = _apc_array_helpers()
        dk, dv = c.dequantize_for_apc()
        if dk is None or dv is None:
            return lm.KVCache()
        out = lm.KVCache()
        out.keys, out.values, out.offset = copy(dk), copy(dv), dk.shape[-2]
        eval_targets.extend([out.keys, out.values])
        return out
    if _custom_state_contract(c):
        return _state_clone(c, eval_targets)
    return None


def merge_cache_entries(entries, prefix_lens):
    from .models import cache as lm

    if not entries:
        return None
    first = entries[0]
    for typ, adapter in _clone_rules():
        if typ is lm.KVCache:
            ok = all(type(c) is typ for c in entries)
        else:
            ok = all(isinstance(c, typ) for c in entries)
        if ok:
            return adapter.merge_rows(entries, prefix_lens)
    if all(isinstance(c, lm.CacheList) for c in entries):
        merged = [
            merge_cache_entries([e.caches[i] for e in entries], prefix_lens)
            for i in range(len(first.caches))
        ]
        return None if any(m is None for m in merged) else lm.CacheList(*merged)
    if all(isinstance(c, tuple) for c in entries):
        merged = [
            merge_cache_entries([e[i] for e in entries], prefix_lens)
            for i in range(len(first))
        ]
        return None if any(m is None for m in merged) else lm.CacheList(*merged)

    if "merge" in type(first).__dict__ and all(type(c) is type(first) for c in entries):
        return type(first).merge(entries, prefix_lens)
    return None


@dataclass(frozen=True)
class ComponentPlan:
    index: int
    type_name: str
    capability: Capability
    restorable: bool
    reason: Optional[str] = None


@dataclass
class PrefixCachePlan:
    """Per-model description of how each cache entry is captured/restored."""

    components: List[ComponentPlan] = field(default_factory=list)

    @property
    def restorable(self) -> bool:
        return bool(self.components) and all(c.restorable for c in self.components)

    @property
    def capabilities(self) -> List[Capability]:
        return [c.capability for c in self.components]

    def adapter_for(self, index: int) -> PrefixStateAdapter:
        return _ADAPTERS[self.components[index].capability]

    def describe(self) -> str:
        head = (
            f"PrefixCachePlan: {len(self.components)} components, "
            f"restorable={self.restorable}"
        )
        lines = [
            f"  [{c.index}] {c.type_name}: {c.capability.value}"
            + ("" if c.restorable else f"  REJECTED ({c.reason})")
            for c in self.components
        ]
        return "\n".join([head, *lines])


def build_prefix_cache_plan(
    model: Any, overrides: Optional[Dict[type, Capability]] = None
) -> PrefixCachePlan:
    """Build a plan by resolving one adapter per entry ``model.make_cache()``."""
    lm = getattr(model, "language_model", model)
    make_cache = getattr(lm, "make_cache", None) or getattr(model, "make_cache", None)
    if not callable(make_cache):
        return PrefixCachePlan()
    caches = make_cache()
    comps: List[ComponentPlan] = []
    for i, entry in enumerate(caches):
        cap = resolve_capability(entry, overrides)
        restorable = cap != Capability.UNSUPPORTED
        comps.append(
            ComponentPlan(
                index=i,
                type_name=type(entry).__name__,
                capability=cap,
                restorable=restorable,
                reason=None if restorable else "no snapshot/restore contract",
            )
        )
    return PrefixCachePlan(components=comps)
