"""Two-box vault tier: wire format and peer protocol sketch (Stage 3).

Topology
--------
L1 is the local process vault (:mod:`mlx_vlm.context_vault`). L2 is the peer
box's vault, reached over the tbnet ring. Measured transport on this fleet:
``mx.distributed`` ring is env-only over tbnet at 10.9 ms / 64 MiB, and a bulk
131k cache ships in 0.55 s (4.6 GB/s). At those rates a 32k checkpoint
(1.00 GiB) crosses in ~0.23 s against a ~93 s cold prefill, so the peer tier is
worth consulting whenever L1 misses and the peer holds a deeper rung.

Protocol
--------
1. **Digest exchange (cheap, collective).** Neither box ships tokens. Each
   boundary is identified by a rolling hash of its token prefix, so both sides
   name the same rung without transferring it. A digest is
   ``{boundary_hash: (depth, nbytes)}`` -- kilobytes for a full vault.
2. **Targeted fetch (bulk, point-to-point).** On an L1 miss the local box picks
   the deepest peer rung that beats its own match and requests exactly that one.
3. **Ship as one buffer.** :func:`pack_fragments` flattens the whole ladder rung
   into a single contiguous uint8 payload plus a small manifest. One buffer is
   what makes 4.6 GB/s reachable; a per-tensor send loop would pay the 10.9 ms
   ring latency ~1500 times for a 131k rung.
4. **Restore locally.** :func:`unpack_fragments` rebuilds the exact fragment
   tree, which then goes through the ordinary
   :func:`~mlx_vlm.context_vault.restore_fragments` path.

Identity is checked before any fetch: a peer rung computed by different weights,
different code, or a different kernel-toggle set is not interchangeable, so the
digest carries the sender's ``vault_identity`` and a mismatch aborts the fetch.

Status: wire format and digest are implemented and tested offline. The ring
send/recv binding is a sketch -- see :func:`fetch_plan` -- pending a measured
window on both boxes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

from . import harvest_provenance as _harvest_prov
from .apc_adapters import Capability, StateFragment

__all__ = [
    "PeerDigest",
    "boundary_hash",
    "fetch_plan",
    "pack_fragments",
    "plan_fragments",
    "unpack_fragments",
    "vault_digest",
]

_DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "uint8": mx.uint8,
    "int8": mx.int8,
    "uint32": mx.uint32,
    "int32": mx.int32,
}


def boundary_hash(tokens: Sequence[int], depth: int, identity: str) -> str:
    """Stable name for "the state after ``depth`` tokens of ``tokens``".

    Includes the vault identity so two boxes running different code can never
    agree on a rung name and accidentally trade incompatible state.
    """
    h = hashlib.sha256()
    h.update(identity.encode())
    h.update(depth.to_bytes(8, "little"))
    for t in tokens[:depth]:
        h.update(int(t).to_bytes(4, "little", signed=True))
    return h.hexdigest()[:32]


# --------------------------------------------------------------------------
# Flat wire format
# --------------------------------------------------------------------------


def _walk(node: Any, arrays: List[mx.array]) -> Any:
    """Replace every mx.array with an index into ``arrays``; keep the shape."""
    if isinstance(node, mx.array):
        arrays.append(node)
        return {
            "__arr__": len(arrays) - 1,
            "shape": list(node.shape),
            "dtype": str(node.dtype).split(".")[-1],
        }
    if isinstance(node, StateFragment):
        return {
            "__frag__": True,
            "capability": node.capability.value,
            "prefix_len": node.prefix_len,
            "schema": node.schema,
            "payload": _walk(node.payload, arrays),
        }
    if isinstance(node, dict):
        return {"__dict__": {k: _walk(v, arrays) for k, v in node.items()}}
    if isinstance(node, tuple):
        return {"__tuple__": [_walk(v, arrays) for v in node]}
    if isinstance(node, list):
        return {"__list__": [_walk(v, arrays) for v in node]}
    return {"__scalar__": node}


def _rebuild(node: Any, arrays: Sequence[mx.array]) -> Any:
    if "__arr__" in node:
        return arrays[node["__arr__"]]
    if "__frag__" in node:
        return StateFragment(
            Capability(node["capability"]),
            int(node["prefix_len"]),
            payload=_rebuild(node["payload"], arrays),
            schema=node.get("schema", "v1"),
        )
    if "__dict__" in node:
        return {k: _rebuild(v, arrays) for k, v in node["__dict__"].items()}
    if "__tuple__" in node:
        return tuple(_rebuild(v, arrays) for v in node["__tuple__"])
    if "__list__" in node:
        return [_rebuild(v, arrays) for v in node["__list__"]]
    return node["__scalar__"]


def plan_fragments(
    fragments: Sequence[StateFragment],
    harvest_provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[mx.array]]:
    """Manifest for a ladder rung plus the flat uint8 view of every array.

    The manifest is exactly what :func:`pack_fragments` produces; the payload is
    left as a *list* of per-array uint8 views laid out back to back at the
    manifest's offsets, so a caller that streams (the disk vault writes 4 MiB at
    a time) never has to materialise the concatenated buffer. Concatenating the
    list reproduces ``pack_fragments`` byte for byte, which is what makes the
    disk file and the wire payload the same format.
    """
    arrays: List[mx.array] = []
    tree = [_walk(f, arrays) for f in fragments]
    flat = [mx.view(mx.contiguous(a).flatten(), mx.uint8) for a in arrays]
    manifest: Dict[str, Any] = {
        "tree": tree,
        "offsets": [],
        "version": 1,
    }
    # Harvest provenance (L1b-1) rides in the manifest, so the disk header and
    # the peer-wire manifest carry the same fact in the same place -- the two
    # formats are the same bytes by construction and this keeps them the same
    # METADATA too.  Omitted entirely when unknown, so a manifest from a caller
    # that does not know its batch width is byte-identical to what it was.
    prov = _harvest_prov.normalise(harvest_provenance)
    if prov is not None:
        manifest["harvest_provenance"] = prov
    off = 0
    for a, f in zip(arrays, flat):
        n = int(f.size)
        manifest["offsets"].append(
            {"start": off, "nbytes": n, "shape": list(a.shape), "dtype": str(a.dtype).split(".")[-1]}
        )
        off += n
    manifest["total_bytes"] = off
    return manifest, flat


def pack_fragments(
    fragments: Sequence[StateFragment],
    harvest_provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], mx.array]:
    """Flatten a ladder rung into (manifest, one contiguous uint8 payload)."""
    manifest, flat = plan_fragments(fragments, harvest_provenance)
    payload = mx.concatenate(flat) if flat else mx.zeros((0,), dtype=mx.uint8)
    mx.eval(payload)
    return manifest, payload


def unpack_fragments(
    manifest: Dict[str, Any], payload: mx.array
) -> List[StateFragment]:
    """Inverse of :func:`pack_fragments`. Byte-exact."""
    arrays: List[mx.array] = []
    for rec in manifest["offsets"]:
        start, n = int(rec["start"]), int(rec["nbytes"])
        dtype = _DTYPES[rec["dtype"]]
        raw = payload[start : start + n]
        arrays.append(mx.view(raw, dtype).reshape(rec["shape"]))
    return [_rebuild(node, arrays) for node in manifest["tree"]]


# --------------------------------------------------------------------------
# Digest + fetch planning
# --------------------------------------------------------------------------


@dataclass
class PeerDigest:
    """What a peer advertises: rung names it holds, and how deep each is."""

    identity: str
    rungs: Dict[str, int]  # boundary_hash -> depth
    nbytes: Dict[str, int]  # boundary_hash -> payload size

    def deepest_match(self, candidates: Sequence[str]) -> Optional[str]:
        best, best_depth = None, -1
        for c in candidates:
            d = self.rungs.get(c, -1)
            if d > best_depth:
                best, best_depth = c, d
        return best


def vault_digest(vault: Any, identity: str, token_index: Dict[int, Sequence[int]]) -> PeerDigest:
    """Advertise the local vault's rungs.

    ``token_index`` maps a stored depth to the token prefix that produced it;
    the caller owns it because the trie deliberately does not retain full token
    lists for every rung.
    """
    rungs: Dict[str, int] = {}
    sizes: Dict[str, int] = {}
    for depth, toks in token_index.items():
        name = boundary_hash(toks, depth, identity)
        rungs[name] = depth
        sizes[name] = 0
    return PeerDigest(identity=identity, rungs=rungs, nbytes=sizes)


def fetch_plan(
    local_depth: int,
    query_tokens: Sequence[int],
    identity: str,
    peer: PeerDigest,
    candidate_depths: Sequence[int],
    min_gain_tokens: int = 4096,
    transport_gbps: float = 4.6,
    prefill_tok_per_s: float = 250.0,
) -> Optional[Dict[str, Any]]:
    """Decide whether pulling a peer rung beats re-prefilling locally.

    A fetch is only worth it when the extra depth it buys costs less in
    transport than it saves in prefill. With the fleet's measured 4.6 GB/s and
    ~250 tok/s long-prompt prefill, shipping 28182 B/tok of DSA latent to skip
    one token of prefill is favourable by a wide margin -- the guard exists to
    suppress trivial gains, not because the trade is close.
    """
    if peer.identity != identity:
        return None
    names = [boundary_hash(query_tokens, d, identity) for d in candidate_depths]
    best = peer.deepest_match(names)
    if best is None:
        return None
    depth = peer.rungs[best]
    gain = depth - local_depth
    if gain < min_gain_tokens or depth > len(query_tokens):
        return None
    # Flat KDA term + linear DSA term, as measured.
    nbytes = 140.8 * 1024**2 + depth * 28182
    ship_s = nbytes / (transport_gbps * 1e9)
    save_s = gain / prefill_tok_per_s
    if ship_s >= save_s:
        return None
    return {
        "rung": best,
        "depth": depth,
        "gain_tokens": gain,
        "est_ship_s": ship_s,
        "est_prefill_saved_s": save_s,
        "speedup": save_s / ship_s if ship_s else float("inf"),
    }
