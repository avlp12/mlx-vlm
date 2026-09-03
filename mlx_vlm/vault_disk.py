"""Disk-backed prefix vault -- ceiling-map lever C15, "never prefill the same
document twice", extended past the end of RAM.

Why this exists
---------------
:mod:`mlx_vlm.context_vault` keeps prefix checkpoints in unified memory and
evicts by LRU when the budget is reached. On this fleet the budget is the whole
lever: a 131,072-token GLM-5.3-Flash rung costs 0.138 GiB + 28,180 B/token =
3.69 GB, so ~78 of them fill the 299 GB of vault space that is left after the
weights, and rung 79 evicts rung 1 -- whose document is then re-prefilled from
scratch at 295 tok/s, i.e. 444 s.

The measured alternative (``glm53flash/logs/sweep11/P2_VERDICT.md``, probe P2,
2026-09-03, F_NOCACHE so the numbers are device numbers and not page-cache
numbers):

===========================  ============  =========================
target / read shape          GB/s          3.69 GB restore
===========================  ============  =========================
internal NVMe, 64 MiB reads  6.725         0.549 s   (809x prefill)
internal NVMe, 4 MiB reads   6.596         0.560 s   (794x)
internal NVMe, 1 MiB reads   3.202         1.154 s   (385x)
internal, buffered cold      5.496         0.672 s   (661x)
external Crucial X10, 64 MiB 0.819         4.512 s   (99x)
external X10, 28,180 B recs  0.166         22.19 s   (20x, worst rep 99 s)
===========================  ============  =========================

Two design constraints fall straight out of that table and both are enforced
here:

1. **One contiguous blob per entry.** The 28,180 B (one APC token) record shape
   costs the external tier 4.9x median and 22x at its worst rep. Entries are
   therefore written as a single file -- header, token ids, then every array
   back to back -- and read with one sequential pass.
2. **Reads and writes are >= 4 MiB.** 1 MiB granularity halves the internal
   tier (3.2 vs 6.6-6.7 GB/s). :func:`read_chunk_bytes` floors at 4 MiB.

What it does *not* do: it does not benchmark the device at startup. The restore
rate is measured on every restore and exposed (``last_restore_gbps``,
``disk_bytes_read`` / ``disk_read_seconds``), so a slow tier shows up in the
runtime snapshot from the first real restore rather than in a synthetic
self-test that delays the load.

Two-step longest-prefix semantics
---------------------------------
The RAM vault serves the longest strict prefix it holds. A disk entry does NOT
compete in that comparison directly: on a miss (or on a RAM hit shallower than
what the disk holds) the disk entry is restored *into the RAM vault* through the
ordinary :meth:`ContextVault.insert` -- normal byte accounting, normal eviction
-- and the ordinary lookup then serves it. So a disk entry is only ever served
after it has become a RAM entry, and every downstream path is unchanged.

The cost of that design is one restore of latency on the first request that
needs the entry: ~0.55-0.7 s for a 131k rung on the internal SSD (P2 above),
plus the copy into MLX-owned memory, against the 444 s prefill it replaces. The
benefit is that "longest strict prefix" keeps exactly one meaning.

Format (one file per entry, single contiguous blob)
---------------------------------------------------
::

    0                8              16                        tokens_offset
    | MAGIC (8 B)   | header_len   | JSON header ... pad      |
                                                              |
    tokens_offset                       payload_offset        EOF
    | int32 token ids  ... pad         | array bytes, manifest order |

The JSON header carries the identity contract (model identity hash, git head,
tokenizer hash, cache-shape flags), the token count and prompt sha256, and the
per-array ``dtype``/``shape``/``start``/``nbytes`` table -- which is literally
the wire tier's manifest (:func:`mlx_vlm.context_vault_wire.plan_fragments`),
so a disk blob and a peer-tier payload are the same bytes in the same order.

Environment
-----------
``MLX_VLM_VAULT_DISK_DIR``          root directory; unset = feature OFF (default)
``MLX_VLM_VAULT_DISK_MAX_GB``       disk cap, default 200
``MLX_VLM_VAULT_DISK_SAVE_ON_INSERT`` also save when a rung is inserted (default 0;
                                    the default policy is save-on-eviction only)
``MLX_VLM_VAULT_DISK_FSYNC``        fsync before the atomic rename (default 0)
``MLX_VLM_VAULT_DISK_CHUNK_MB``     read/write chunk, default 4, floored at 4
``MLX_VLM_VAULT_DISK_NOCACHE``      F_NOCACHE on every fd (default 1, macOS only)
``MLX_VLM_VAULT_DISK_STRICT_GIT``   refuse an entry from another git head (default 1)

Put the root on the internal NVMe. The external X10 works and is honest about
it in the counters, but it is 8x slower on this box (0.82 vs 6.7 GB/s measured,
and its rated 2.1 GB/s was refuted -- P2b).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import queue
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

from . import harvest_provenance as _harvest_prov
from .apc_adapters import ADAPTER_SCHEMA_VERSION, StateFragment, dedup_enabled
from .context_vault import VaultCheckpoint, VaultTier
from .context_vault_wire import _DTYPES, plan_fragments, unpack_fragments

logger = logging.getLogger(__name__)

__all__ = [
    "DiskPrefixVault",
    "DiskVaultStats",
    "attach_disk_vault",
    "cache_shape_flags",
    "disk_cap_bytes",
    "disk_stats_snapshot",
    "disk_vault_dir",
    "disk_vault_enabled",
    "fsync_enabled",
    "model_identity_hash",
    "nocache_enabled",
    "read_chunk_bytes",
    "reset_disk_stats",
    "save_on_insert_enabled",
    "strict_git_head",
    "tokenizer_identity_hash",
]

MAGIC = b"MLXVAUL1"
FORMAT_VERSION = 1
ENTRY_SUFFIX = ".vault"
ENTRY_PREFIX = "entry_"
PARTIAL_SUFFIX = ".partial"
INDEX_NAME = "index.json"
ALIGN = 4096

# macOS F_NOCACHE. Not exposed as a named constant by Python's fcntl module;
# P2 used the same literal and demonstrated it works (a non-NOCACHE reread of
# the external drive inflated 0.98 -> 6.03 GB/s, a 6.2x fiction).
_F_NOCACHE = 48

_ENV_DIR = "MLX_VLM_VAULT_DISK_DIR"
_ENV_MAX_GB = "MLX_VLM_VAULT_DISK_MAX_GB"
_ENV_SAVE_ON_INSERT = "MLX_VLM_VAULT_DISK_SAVE_ON_INSERT"
_ENV_FSYNC = "MLX_VLM_VAULT_DISK_FSYNC"
_ENV_CHUNK_MB = "MLX_VLM_VAULT_DISK_CHUNK_MB"
_ENV_NOCACHE = "MLX_VLM_VAULT_DISK_NOCACHE"
_ENV_STRICT_GIT = "MLX_VLM_VAULT_DISK_STRICT_GIT"

_DEFAULT_MAX_GB = 200.0
# P2: 1 MiB reads run at 3.202 GB/s on the internal NVMe against 6.596 at
# 4 MiB. Four is the floor, not the default-that-can-be-lowered.
MIN_CHUNK_BYTES = 4 << 20


def _env_truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def disk_vault_dir() -> Optional[Path]:
    """Root for disk entries, or ``None`` (the default) meaning the tier is off."""
    raw = os.environ.get(_ENV_DIR, "").strip()
    return Path(raw).expanduser() if raw else None


def disk_vault_enabled() -> bool:
    return disk_vault_dir() is not None


def disk_cap_bytes() -> int:
    try:
        gb = float(os.environ.get(_ENV_MAX_GB, _DEFAULT_MAX_GB))
    except (TypeError, ValueError):
        gb = _DEFAULT_MAX_GB
    return int(max(0.0, gb) * (1000**3))


def save_on_insert_enabled() -> bool:
    return _env_truthy(_ENV_SAVE_ON_INSERT)


def fsync_enabled() -> bool:
    return _env_truthy(_ENV_FSYNC)


def nocache_enabled() -> bool:
    return _env_truthy(_ENV_NOCACHE, "1")


def strict_git_head() -> bool:
    return _env_truthy(_ENV_STRICT_GIT, "1")


def read_chunk_bytes() -> int:
    """Read/write granularity, never below 4 MiB (P2's measured knee)."""
    try:
        mb = float(os.environ.get(_ENV_CHUNK_MB, "4"))
    except (TypeError, ValueError):
        mb = 4.0
    return max(MIN_CHUNK_BYTES, int(mb * (1 << 20)))


# --------------------------------------------------------------------------
# Identity inputs
# --------------------------------------------------------------------------

# Env toggles that change the SHAPE or CONTENTS of a cache. A blob written under
# a different set is refused rather than restored: shapes would usually still
# match, which is exactly why the check cannot be structural.
_SHAPE_FLAGS = (
    "MLX_VLM_GLM5_FUSED_KDA",
    "MLX_VLM_GLM5_FUSED_QPROJ",
    "MLX_VLM_GLM5_QPROJ",
    "MLX_VLM_GLM5_SPARSE",
    "MLX_VLM_GLM5_DSA_FUSED",
    "MLX_VLM_DFLASH_COMPILE",
    "MLX_VLM_DFLASH_DEFERRED",
    "MLX_VLM_DFLASH_FC_PRETRUNC",
    "MLX_VLM_DFLASH_FIXED_WIDTH",
    "MLX_VLM_DRAFT_KIND",
    "MLX_VLM_DRAFT_MODEL",
)


def cache_shape_flags(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The dflash/spec/kernel toggles a restored cache must agree with.

    ``MLX_VLM_GLM5_VAULT_DEDUP`` is deliberately absent: it changes how a rung is
    *stored* (byte-identical siblings become aliases) and not what it restores to
    -- ``apc_adapters.dedup_enabled`` argues the same case for the RAM store, and
    ``unpack_fragments`` expands aliases regardless of the current setting.
    """
    flags: Dict[str, Any] = {name: os.environ.get(name, "") for name in _SHAPE_FLAGS}
    if extra:
        for k, v in extra.items():
            flags[str(k)] = v
    return flags


def _flags_hash(flags: Dict[str, Any]) -> str:
    blob = json.dumps(flags, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def _sha_of_files(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    seen = 0
    for p in paths:
        try:
            if not p.is_file():
                continue
            h.update(p.name.encode())
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
            seen += 1
        except OSError:
            continue
    return h.hexdigest()[:32] if seen else ""


def model_identity_hash(model_path: Any) -> str:
    """sha256 over ``config.json`` + the safetensors weight index.

    The weight index names every shard and every tensor, so two trees that agree
    on it hold the same weights under the same names; the config pins the shapes
    and the quantization. Returns ``""`` when the path is not a readable local
    tree -- an empty side of the comparison is skipped, never treated as a match.
    """
    try:
        root = Path(str(model_path)).expanduser()
        if not root.is_dir():
            return ""
        return _sha_of_files(
            [
                root / "config.json",
                root / "model.safetensors.index.json",
            ]
        )
    except (OSError, TypeError, ValueError):
        return ""


def tokenizer_identity_hash(model_path: Any) -> str:
    """sha256 over the tokenizer files. Different tokens, different cache."""
    try:
        root = Path(str(model_path)).expanduser()
        if not root.is_dir():
            return ""
        return _sha_of_files(
            [
                root / "tokenizer.json",
                root / "tokenizer_config.json",
                root / "added_tokens.json",
            ]
        )
    except (OSError, TypeError, ValueError):
        return ""


_GIT_HEAD_CACHE: Optional[str] = None


def _git_head() -> str:
    """Revision of the running tree, via the vault's own code-identity probe."""
    global _GIT_HEAD_CACHE
    if _GIT_HEAD_CACHE is None:
        from .context_vault import _code_identity

        try:
            _GIT_HEAD_CACHE = _code_identity()
        except Exception:  # noqa: BLE001 - identity must never fail a load
            _GIT_HEAD_CACHE = "unknown"
    return _GIT_HEAD_CACHE


# --------------------------------------------------------------------------
# Counters
# --------------------------------------------------------------------------


@dataclass
class DiskVaultStats:
    """Counters for the runtime snapshot, alongside ``session_skips``.

    Module-global by default (``STATS``) for the same reason the session skip
    counters are: two of the interesting states are "there is no disk vault" and
    "the directory is unset", and a counter that lives on the object cannot
    record its own absence.
    """

    disk_hits: int = 0
    disk_misses: int = 0
    disk_restores: int = 0
    disk_restore_seconds: float = 0.0
    disk_read_seconds: float = 0.0
    disk_bytes_read: int = 0
    disk_saves: int = 0
    disk_save_seconds: float = 0.0
    disk_bytes_written: int = 0
    disk_save_errors: int = 0
    disk_save_skips: int = 0
    disk_evictions: int = 0
    disk_refusals: Dict[str, int] = field(default_factory=dict)
    last_restore_gbps: float = 0.0
    min_restore_gbps: float = 0.0
    max_restore_gbps: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def record_refusal(self, reason: str) -> None:
        with self._lock:
            self.disk_refusals[reason] = self.disk_refusals.get(reason, 0) + 1

    def record_restore(self, nbytes: int, read_s: float, total_s: float) -> None:
        with self._lock:
            self.disk_restores += 1
            self.disk_bytes_read += int(nbytes)
            self.disk_read_seconds += float(read_s)
            self.disk_restore_seconds += float(total_s)
            gbps = (nbytes / read_s / 1e9) if read_s > 0 else 0.0
            self.last_restore_gbps = gbps
            if self.min_restore_gbps == 0.0 or gbps < self.min_restore_gbps:
                self.min_restore_gbps = gbps
            if gbps > self.max_restore_gbps:
                self.max_restore_gbps = gbps

    def record_save(self, nbytes: int, seconds: float) -> None:
        with self._lock:
            self.disk_saves += 1
            self.disk_bytes_written += int(nbytes)
            self.disk_save_seconds += float(seconds)

    def bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def snapshot(self) -> dict:
        with self._lock:
            read_gbps = (
                self.disk_bytes_read / self.disk_read_seconds / 1e9
                if self.disk_read_seconds > 0
                else 0.0
            )
            return {
                "disk_hits": self.disk_hits,
                "disk_misses": self.disk_misses,
                "disk_restores": self.disk_restores,
                "disk_restore_seconds": round(self.disk_restore_seconds, 6),
                "disk_read_seconds": round(self.disk_read_seconds, 6),
                "disk_bytes_read": self.disk_bytes_read,
                "disk_restore_gbps_mean": round(read_gbps, 4),
                "disk_restore_gbps_last": round(self.last_restore_gbps, 4),
                "disk_restore_gbps_min": round(self.min_restore_gbps, 4),
                "disk_restore_gbps_max": round(self.max_restore_gbps, 4),
                "disk_saves": self.disk_saves,
                "disk_save_seconds": round(self.disk_save_seconds, 6),
                "disk_bytes_written": self.disk_bytes_written,
                "disk_save_errors": self.disk_save_errors,
                "disk_save_skips": self.disk_save_skips,
                "disk_evictions": self.disk_evictions,
                "disk_refusals": dict(self.disk_refusals),
            }

    def reset(self) -> None:
        with self._lock:
            for f_name, default in (
                ("disk_hits", 0),
                ("disk_misses", 0),
                ("disk_restores", 0),
                ("disk_restore_seconds", 0.0),
                ("disk_read_seconds", 0.0),
                ("disk_bytes_read", 0),
                ("disk_saves", 0),
                ("disk_save_seconds", 0.0),
                ("disk_bytes_written", 0),
                ("disk_save_errors", 0),
                ("disk_save_skips", 0),
                ("disk_evictions", 0),
                ("last_restore_gbps", 0.0),
                ("min_restore_gbps", 0.0),
                ("max_restore_gbps", 0.0),
            ):
                setattr(self, f_name, default)
            self.disk_refusals.clear()


STATS = DiskVaultStats()


def disk_stats_snapshot() -> dict:
    """Disk-tier counters for the server's runtime snapshot."""
    snap = STATS.snapshot()
    root = disk_vault_dir()
    snap["enabled"] = root is not None
    snap["dir"] = str(root) if root is not None else None
    return snap


def reset_disk_stats() -> None:
    STATS.reset()


# --------------------------------------------------------------------------
# Low-level file I/O: single blob, >= 4 MiB records, F_NOCACHE
# --------------------------------------------------------------------------


def _set_nocache(fd: int, enable: bool) -> bool:
    """Ask the kernel not to keep this fd's pages in the unified buffer cache.

    macOS only, and advisory: unlike ``O_DIRECT`` it imposes no alignment, so an
    unaligned header read is still legal. Returns whether it was applied, which
    the tests assert on so a silent fallback to buffered I/O -- the thing that
    turned 0.98 GB/s into a fictional 6.03 in P2 -- cannot pass unnoticed.
    """
    if not enable or sys.platform != "darwin":
        return False
    try:
        import fcntl

        fcntl.fcntl(fd, _F_NOCACHE, 1)
        return True
    except (OSError, ValueError, ImportError):  # pragma: no cover - platform dep
        return False


def _align_up(n: int, a: int = ALIGN) -> int:
    return ((n + a - 1) // a) * a


class _BlobWriter:
    """Sequential writer that keeps every record at least ``chunk`` bytes.

    Small pieces (the header, the token block, a short array) are staged in one
    buffer and flushed in ``chunk``-sized records; any run of >= ``chunk`` bytes
    inside a single array is written straight from the array's memory with no
    copy. Record sizes are recorded so a test can prove the 28,180 B per-token
    shape is not what reaches the device.
    """

    def __init__(self, fd: int, chunk: int):
        self.fd = fd
        self.chunk = chunk
        self._stage = bytearray()
        self.offset = 0
        self.record_sizes: List[int] = []

    def _raw_write(self, mv) -> None:
        n = len(mv)
        if not n:
            return
        written = 0
        while written < n:
            chunk = mv[written:]
            try:
                written += os.write(self.fd, chunk)
            finally:
                chunk.release()
        self.record_sizes.append(n)

    def _flush_stage(self) -> None:
        """Emit the staging buffer as one record and hand it to the GC.

        Rebinding rather than ``del self._stage[:n]`` because a bytearray cannot
        be resized while a memoryview of it is exported, and the export is
        exactly what lets the write go out without another copy.
        """
        if not self._stage:
            return
        buf = self._stage
        self._stage = bytearray()
        mv = memoryview(buf)
        try:
            self._raw_write(mv)
        finally:
            mv.release()

    def write(self, data) -> None:
        mv = memoryview(data)
        if mv.format != "B" or not mv.contiguous:
            mv = memoryview(bytes(mv))
        n = len(mv)
        self.offset += n
        pos = 0
        # 1. Top the staging buffer up to exactly one record and flush it, so
        #    the header/token block never forces the payload into a copy.
        if self._stage:
            take = min(self.chunk - len(self._stage), n)
            self._stage += mv[:take]
            pos = take
            if len(self._stage) >= self.chunk:
                self._flush_stage()
        # 2. Whole records straight out of the caller's buffer. A 3.7 GB rung
        #    never exists twice in memory: this is the path it takes.
        while n - pos >= self.chunk:
            end = pos + self.chunk
            piece = mv[pos:end]
            try:
                self._raw_write(piece)
            finally:
                piece.release()
            pos = end
        # 3. The ragged tail waits for the next write, or for finish().
        if pos < n:
            self._stage += mv[pos:]

    def pad_to(self, target: int) -> None:
        if target < self.offset:
            raise ValueError("blob writer cannot rewind")
        if target > self.offset:
            self.write(bytes(target - self.offset))

    def finish(self) -> None:
        self._flush_stage()


def _read_exact(f: io.FileIO, buf: memoryview, chunk: int, sizes: List[int]) -> int:
    """Fill ``buf`` in ``chunk``-sized reads. Returns bytes read."""
    total = 0
    n = len(buf)
    while total < n:
        want = min(chunk, n - total)
        got = f.readinto(buf[total : total + want])
        if not got:
            break
        sizes.append(got)
        total += got
    return total


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------


def _prefix_digests(tokens: Sequence[int], depths: Sequence[int]) -> Dict[int, str]:
    """sha256 of ``tokens[:d]`` for every ``d``, in one pass over the tokens.

    sha256 is incremental, so N candidate depths cost one walk of the prompt and
    N cheap ``copy()`` calls -- not N full hashes. That matters because the
    lookup runs on the request path for every distinct depth in the index.
    """
    out: Dict[int, str] = {}
    h = hashlib.sha256()
    h.update(b"mlx-vlm-vault-prefix-v1")
    pos = 0
    n = len(tokens)
    for d in sorted({int(x) for x in depths}):
        if d < 0 or d > n:
            continue
        while pos < d:
            h.update(int(tokens[pos]).to_bytes(4, "little", signed=True))
            pos += 1
        out[d] = h.copy().hexdigest()
    return out


def prompt_prefix_sha(tokens: Sequence[int], depth: Optional[int] = None) -> str:
    d = len(tokens) if depth is None else int(depth)
    return _prefix_digests(tokens, [d])[d]


def entry_key(prompt_sha: str, identity: str, tier: str, depth: int) -> str:
    h = hashlib.sha256()
    h.update(prompt_sha.encode())
    h.update(b"|")
    h.update(identity.encode())
    h.update(b"|")
    h.update(tier.encode())
    h.update(b"|")
    h.update(int(depth).to_bytes(8, "little"))
    return h.hexdigest()[:32]


def read_header(path: Path) -> Optional[dict]:
    """Parse just the JSON header of an entry file. Cheap; no payload read."""
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != MAGIC:
                return None
            (hlen,) = struct.unpack("<Q", f.read(8))
            if hlen <= 0 or hlen > (64 << 20):
                return None
            raw = f.read(hlen)
            if len(raw) != hlen:
                return None
            return json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _header_refusal(
    header: Optional[dict],
    *,
    identity: str,
    model_identity: str,
    git_head: str,
    tokenizer_hash: str,
    flags_hash: str,
    strict_git: bool,
    tier: Optional[str] = None,
    expect_prompt_sha: Optional[str] = None,
    file_size: Optional[int] = None,
    require_harvest_provenance: bool = True,
) -> Optional[str]:
    """Named reason to refuse, or ``None`` to accept. Never raises.

    Every branch is a *refusal*, never an exception and never a silent accept:
    the cost of refusing is one cold prefill, the cost of accepting the wrong
    state is a fluent wrong answer -- the same trade
    ``ContextVault.restore_into`` already makes for foreign checkpoints.
    """
    if header is None:
        return "header_unreadable"
    if header.get("format") != "mlx-vlm-disk-vault":
        return "header_magic"
    if int(header.get("version", -1)) != FORMAT_VERSION:
        return "header_version"
    if int(header.get("adapter_schema", -1)) != ADAPTER_SCHEMA_VERSION:
        return "adapter_schema_mismatch"
    if header.get("vault_identity") != identity:
        return "identity_mismatch"
    hm = header.get("model_identity") or ""
    if hm and model_identity and hm != model_identity:
        return "model_identity_mismatch"
    if strict_git:
        hg = header.get("git_head") or ""
        if hg and git_head and hg != git_head:
            return "git_head_mismatch"
    ht = header.get("tokenizer_hash") or ""
    if ht and tokenizer_hash and ht != tokenizer_hash:
        return "tokenizer_mismatch"
    if header.get("flags_hash") != flags_hash:
        return "flags_mismatch"
    if require_harvest_provenance and not _harvest_prov.is_complete(
        {f: header.get(f) for f in _harvest_prov.FIELDS}
    ):
        # An entry that cannot say where it came from is refused, not assumed
        # clean.  Blobs written before this field existed land here, and that is
        # the intended outcome: the cost is one cold prefill, and the thing being
        # avoided is a warm start that is bit-different from the cold run with
        # nothing on the entry to say so (L1b-1).  ``MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH=0``
        # relaxes both halves of the policy together.
        return "harvest_provenance_missing"
    if tier is not None and header.get("tier") != tier:
        return "tier_mismatch"
    if expect_prompt_sha is not None and header.get("prompt_sha256") != expect_prompt_sha:
        return "prompt_sha_mismatch"

    arrays = header.get("arrays")
    if not isinstance(arrays, list):
        return "array_table_missing"
    off = 0
    for rec in arrays:
        try:
            dtype = _DTYPES[rec["dtype"]]
            shape = [int(x) for x in rec["shape"]]
            nbytes = int(rec["nbytes"])
            start = int(rec["start"])
        except (KeyError, TypeError, ValueError):
            return "dtype_unknown"
        numel = 1
        for s in shape:
            numel *= int(s)
        if numel * dtype.size != nbytes:
            # A dtype/shape pair that does not account for its own byte count
            # would reinterpret the bytes on restore. Refuse.
            return "dtype_mismatch"
        if start != off:
            return "layout_not_contiguous"
        off += nbytes
    if off != int(header.get("payload_nbytes", -1)):
        return "payload_size_mismatch"

    if file_size is not None:
        want = int(header.get("payload_offset", 0)) + int(header.get("payload_nbytes", 0))
        if file_size != want:
            return "truncated"
    return None


# --------------------------------------------------------------------------
# The vault
# --------------------------------------------------------------------------


class DiskPrefixVault:
    """Contiguous-blob cold tier under a :class:`~mlx_vlm.context_vault.ContextVault`.

    Lifecycle: a rung about to be evicted from RAM is queued to a background
    writer (the generation thread never touches the device); a lookup that the
    RAM trie cannot satisfy, or satisfies less deeply than the disk can, reads
    one blob and re-inserts it through the ordinary ``insert`` path.
    """

    def __init__(
        self,
        root: Any,
        identity: str,
        *,
        model_identity: str = "",
        git_head: Optional[str] = None,
        tokenizer_hash: str = "",
        flags: Optional[Dict[str, Any]] = None,
        cap_bytes: Optional[int] = None,
        chunk_bytes: Optional[int] = None,
        fsync: Optional[bool] = None,
        nocache: Optional[bool] = None,
        save_on_insert: Optional[bool] = None,
        strict_git: Optional[bool] = None,
        stats: Optional[DiskVaultStats] = None,
    ):
        self.dir = Path(root).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.model_identity = model_identity
        self.git_head = _git_head() if git_head is None else git_head
        self.tokenizer_hash = tokenizer_hash
        self.flags = cache_shape_flags(flags if isinstance(flags, dict) else None)
        self.flags_hash = _flags_hash(self.flags)
        self.cap_bytes = disk_cap_bytes() if cap_bytes is None else int(cap_bytes)
        self.chunk = read_chunk_bytes() if chunk_bytes is None else max(1, int(chunk_bytes))
        self.fsync = fsync_enabled() if fsync is None else bool(fsync)
        self.nocache = nocache_enabled() if nocache is None else bool(nocache)
        self.save_on_insert = (
            save_on_insert_enabled() if save_on_insert is None else bool(save_on_insert)
        )
        self.strict_git = strict_git_head() if strict_git is None else bool(strict_git)
        self.stats = stats if stats is not None else STATS

        self._index: Dict[str, dict] = {}
        self._lock = threading.RLock()
        self._in_flight: Dict[str, threading.Event] = {}
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=64)
        self._closed = False
        # Test/observability handles: the sizes of the records that actually
        # reached the device on the most recent save and the most recent read.
        self.last_write_records: List[int] = []
        self.last_read_records: List[int] = []
        self.last_nocache_applied: Optional[bool] = None

        self._sweep_partials()
        self._load_index()

        self._writer = threading.Thread(
            target=self._writer_loop, name="vault-disk-writer", daemon=True
        )
        self._writer.start()

    # -- paths / index --------------------------------------------------

    def _entry_path(self, key: str) -> Path:
        return self.dir / f"{ENTRY_PREFIX}{key}{ENTRY_SUFFIX}"

    @property
    def index_path(self) -> Path:
        return self.dir / INDEX_NAME

    def entry_files(self) -> List[Path]:
        return sorted(p for p in self.dir.glob(f"{ENTRY_PREFIX}*{ENTRY_SUFFIX}"))

    def _sweep_partials(self) -> int:
        n = 0
        for p in self.dir.glob(f"*{PARTIAL_SUFFIX}"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        if n:
            logger.info("vault-disk: removed %d partial file(s) from %s", n, self.dir)
        return n

    def _load_index(self) -> None:
        """Adopt the on-disk index, then reconcile it against the directory.

        The files are the truth: an index record with no file is dropped, and a
        file with no record is adopted by reading its header (which is why the
        header is self-describing). A crash between the rename and the index
        write therefore costs nothing.
        """
        raw: Dict[str, dict] = {}
        try:
            if self.index_path.is_file():
                data = json.loads(self.index_path.read_text())
                if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                    raw = data["entries"]
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("vault-disk: index unreadable at %s; rebuilding from files",
                           self.index_path)
            raw = {}

        index: Dict[str, dict] = {}
        for key, rec in raw.items():
            path = self._entry_path(key)
            if not path.is_file():
                continue
            try:
                rec = dict(rec)
                rec["bytes"] = int(path.stat().st_size)
                index[key] = rec
            except OSError:
                continue

        for path in self.entry_files():
            key = path.stem[len(ENTRY_PREFIX) :]
            if key in index:
                continue
            header = read_header(path)
            if header is None:
                continue
            try:
                index[key] = {
                    "key": key,
                    "prefix_len": int(header.get("prefix_len", 0)),
                    "tier": str(header.get("tier", VaultTier.PREFILL.value)),
                    "prompt_sha256": str(header.get("prompt_sha256", "")),
                    "session_id": str(header.get("session_id", "")),
                    "vault_identity": str(header.get("vault_identity", "")),
                    "bytes": int(path.stat().st_size),
                    "created_at": float(header.get("created_at", time.time())),
                    "last_used": float(header.get("created_at", time.time())),
                }
            except (OSError, TypeError, ValueError):
                continue

        with self._lock:
            self._index = index
        if index:
            logger.info(
                "vault-disk: %d entr%s, %.2f GB under %s",
                len(index),
                "y" if len(index) == 1 else "ies",
                self.disk_bytes / 1e9,
                self.dir,
            )

    def _write_index_locked(self) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        payload = {
            "version": FORMAT_VERSION,
            "identity": self.identity,
            "entries": self._index,
        }
        try:
            tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
            os.replace(tmp, self.index_path)
        except OSError:
            logger.warning("vault-disk: could not persist the index", exc_info=True)

    @property
    def disk_bytes(self) -> int:
        with self._lock:
            return sum(int(r.get("bytes", 0)) for r in self._index.values())

    def records(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._index.values()]

    # -- save -----------------------------------------------------------

    def save_async(
        self,
        tokens: Sequence[int],
        checkpoint: VaultCheckpoint,
        *,
        reason: str = "evict",
    ) -> bool:
        """Queue ``checkpoint`` for the background writer. Never blocks.

        Returns whether it was queued. A full queue is a *skip*, not a wait: the
        caller is the generation thread inside ``_evict_until``, and making it
        wait for a device would be exactly the failure this tier exists to avoid.
        """
        if self._closed:
            return False
        try:
            depth = int(checkpoint.prefix_len)
            toks = list(tokens)[:depth]
            if depth <= 0 or len(toks) != depth:
                self.stats.bump("disk_save_skips")
                return False
            if not _harvest_prov.may_persist(
                getattr(checkpoint, "harvest_provenance", None)
            ):
                # The durability gate, enforced at the queue rather than only at
                # the caller: ``_offload`` is not the only way into this method
                # (a test, a future session-offload hook, a peer sync), and a
                # gate that only one caller honours is not a gate.
                self.stats.record_refusal("harvest_width_not_durable")
                return False
            tier = getattr(checkpoint, "tier", VaultTier.PREFILL)
            tier_s = tier.value if isinstance(tier, VaultTier) else str(tier)
            sha = prompt_prefix_sha(toks, depth)
            key = entry_key(sha, self.identity, tier_s, depth)
            with self._lock:
                if key in self._index or key in self._in_flight:
                    # Already durable (or on its way). Re-writing 3.7 GB to
                    # learn nothing is the one thing worse than not saving.
                    self.stats.bump("disk_save_skips")
                    return False
                ev = threading.Event()
                self._in_flight[key] = ev
            item = (key, sha, toks, checkpoint, tier_s, reason)
            try:
                self._q.put_nowait(item)
            except queue.Full:
                with self._lock:
                    self._in_flight.pop(key, None)
                ev.set()
                self.stats.bump("disk_save_skips")
                return False
            return True
        except Exception:  # noqa: BLE001 - a save fault must never fail a request
            logger.warning("vault-disk: save could not be queued", exc_info=True)
            self.stats.bump("disk_save_errors")
            return False

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait for the writer to drain. Tests and shutdown only."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = bool(self._in_flight)
            if not pending and self._q.empty():
                return True
            time.sleep(0.005)
        return False

    def _writer_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            key = item[0]
            try:
                self._write_entry(*item)
            except Exception:  # noqa: BLE001 - the writer must outlive one bad entry
                self.stats.bump("disk_save_errors")
                logger.warning("vault-disk: save failed for %s", key, exc_info=True)
                try:
                    self._entry_path(key).with_suffix(
                        ENTRY_SUFFIX + PARTIAL_SUFFIX
                    ).unlink()
                except OSError:
                    pass
            finally:
                with self._lock:
                    ev = self._in_flight.pop(key, None)
                if ev is not None:
                    ev.set()
                self._q.task_done()

    def _write_entry(
        self,
        key: str,
        prompt_sha: str,
        tokens: List[int],
        checkpoint: VaultCheckpoint,
        tier_s: str,
        reason: str,
    ) -> None:
        t0 = time.perf_counter()
        provenance = _harvest_prov.normalise(
            getattr(checkpoint, "harvest_provenance", None)
        )
        manifest, flats = plan_fragments(checkpoint.fragments, provenance)
        if flats:
            mx.eval(flats)
        payload_nbytes = int(manifest["total_bytes"])
        tokens_nbytes = 4 * len(tokens)

        base = {
            "format": "mlx-vlm-disk-vault",
            "version": FORMAT_VERSION,
            "key": key,
            "vault_identity": self.identity,
            "model_identity": self.model_identity,
            "git_head": self.git_head,
            "tokenizer_hash": self.tokenizer_hash,
            "flags": self.flags,
            "flags_hash": self.flags_hash,
            "adapter_schema": ADAPTER_SCHEMA_VERSION,
            "dedup": bool(dedup_enabled()),
            "tier": tier_s,
            "session_id": str(getattr(checkpoint, "session_id", "")),
            "origin": str(getattr(checkpoint, "origin", "")),
            "prefix_len": int(checkpoint.prefix_len),
            "token_count": len(tokens),
            "prompt_sha256": prompt_sha,
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "saved_because": reason,
            "chunk_bytes": self.chunk,
            "tokens_dtype": "int32",
            "tokens_nbytes": tokens_nbytes,
            "payload_nbytes": payload_nbytes,
            "arrays": manifest["offsets"],
            "tree": manifest["tree"],
        }
        # L1b-1: the header carries WHERE the rung was harvested alongside WHAT
        # it is.  The design doc already said headers carry tier/session/shape
        # flags because "a session rung served to a prefill query is a fluent
        # wrong answer"; a rung harvested inside a batched prefill and restored
        # into a solo request is the same class of wrong answer, one layer down,
        # and until this field existed nothing on disk could tell them apart.
        # Five flat fields rather than a nested object so a header can be
        # grepped and an index record can copy one of them.
        base.update(provenance or {})
        base["harvest_provenance_complete"] = provenance is not None

        # The header records the offsets it is itself followed by, so serialise
        # until the reserved area stops growing (two passes in practice).
        area = ALIGN
        for _ in range(8):
            header = dict(base)
            header["tokens_offset"] = area
            header["payload_offset"] = _align_up(area + tokens_nbytes)
            blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
            need = _align_up(16 + len(blob) + 64)
            if need <= area:
                break
            area = need
        else:  # pragma: no cover - would need a pathological header
            raise ValueError("vault-disk: header area did not converge")

        tokens_offset = int(header["tokens_offset"])
        payload_offset = int(header["payload_offset"])
        tmp = self.dir / f"{ENTRY_PREFIX}{key}{ENTRY_SUFFIX}{PARTIAL_SUFFIX}"
        final = self._entry_path(key)

        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        applied = _set_nocache(fd, self.nocache)
        self.last_nocache_applied = applied
        w = _BlobWriter(fd, self.chunk)
        try:
            w.write(MAGIC)
            w.write(struct.pack("<Q", len(blob)))
            w.write(blob)
            w.pad_to(tokens_offset)
            if tokens:
                w.write(
                    b"".join(
                        int(t).to_bytes(4, "little", signed=True) for t in tokens
                    )
                )
            w.pad_to(payload_offset)
            for flat in flats:
                w.write(memoryview(flat))
            w.finish()
            if self.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)

        os.replace(tmp, final)
        size = final.stat().st_size
        elapsed = time.perf_counter() - t0
        self.last_write_records = list(w.record_sizes)
        self.stats.record_save(size, elapsed)

        with self._lock:
            self._index[key] = {
                "key": key,
                "prefix_len": int(checkpoint.prefix_len),
                "tier": tier_s,
                "prompt_sha256": prompt_sha,
                "session_id": str(getattr(checkpoint, "session_id", "")),
                "vault_identity": self.identity,
                "bytes": int(size),
                "created_at": float(header["created_at"]),
                "last_used": time.time(),
                "harvest_batch_width": _harvest_prov.batch_width_of(provenance),
            }
            self._enforce_cap_locked(protect=key)
            self._write_index_locked()

        logger.debug(
            "vault-disk: saved %d tokens (%.3f GB) in %.3f s to %s",
            len(tokens), size / 1e9, elapsed, final.name,
        )

    def _enforce_cap_locked(self, protect: Optional[str] = None) -> int:
        """LRU-delete by the index until the tree is under the cap."""
        if self.cap_bytes <= 0:
            return 0
        dropped = 0
        total = sum(int(r.get("bytes", 0)) for r in self._index.values())
        while total > self.cap_bytes:
            victims = [k for k in self._index if k != protect]
            if not victims:
                break
            key = min(victims, key=lambda k: float(self._index[k].get("last_used", 0.0)))
            rec = self._index.pop(key)
            try:
                self._entry_path(key).unlink()
            except OSError:
                pass
            total -= int(rec.get("bytes", 0))
            dropped += 1
            self.stats.bump("disk_evictions")
            logger.info("vault-disk: dropped %s (%d tokens) to stay under the cap",
                        key[:12], rec.get("prefix_len", 0))
        return dropped

    # -- lookup / restore -----------------------------------------------

    def best_record(
        self,
        tokens: Sequence[int],
        tier: VaultTier = VaultTier.PREFILL,
        *,
        min_depth: int = 0,
        strict: bool = True,
    ) -> Optional[dict]:
        """Deepest stored entry whose prompt hash matches a prefix of ``tokens``.

        ``strict`` keeps the RAM semantics: an entry covering the whole prompt
        leaves no suffix to prefill and is not a *prefix* hit.
        """
        tier_s = tier.value if isinstance(tier, VaultTier) else str(tier)
        n = len(tokens)
        with self._lock:
            cands = [
                r
                for r in self._index.values()
                if r.get("tier") == tier_s
                and int(r.get("prefix_len", 0)) > int(min_depth)
                and (int(r.get("prefix_len", 0)) < n if strict else int(r.get("prefix_len", 0)) <= n)
            ]
        if not cands:
            return None
        depths = {int(r["prefix_len"]) for r in cands}
        digests = _prefix_digests(tokens, depths)
        best: Optional[dict] = None
        for r in cands:
            d = int(r["prefix_len"])
            if digests.get(d) != r.get("prompt_sha256"):
                continue
            if best is None or d > int(best["prefix_len"]):
                best = r
        return dict(best) if best else None

    def load_entry(
        self, key: str, *, expect_prompt_sha: Optional[str] = None, tier: Optional[str] = None
    ) -> Optional[Tuple[dict, List[StateFragment], List[int]]]:
        """Read one blob and rebuild (header, fragments, token ids).

        One sequential pass, ``self.chunk``-sized records, F_NOCACHE so the
        3.7 GB does not also land in the page cache it is about to be copied out
        of. A refusal (header mismatch, truncation, unknown dtype) counts a named
        reason and returns ``None``.
        """
        path = self._entry_path(key)
        t0 = time.perf_counter()
        try:
            size = path.stat().st_size
        except OSError:
            self.stats.record_refusal("file_missing")
            return None
        header = read_header(path)
        reason = _header_refusal(
            header,
            identity=self.identity,
            model_identity=self.model_identity,
            git_head=self.git_head,
            tokenizer_hash=self.tokenizer_hash,
            flags_hash=self.flags_hash,
            strict_git=self.strict_git,
            tier=tier,
            expect_prompt_sha=expect_prompt_sha,
            file_size=size,
            # One knob relaxes both halves: a deployment that has opted out of
            # the persist gate has said it does not want provenance enforced,
            # and enforcing it only on the read side would strand every blob it
            # just wrote.
            require_harvest_provenance=(
                _harvest_prov.persist_max_harvest_width() > 0
            ),
        )
        if reason is not None:
            self.stats.record_refusal(reason)
            logger.warning("vault-disk: refusing %s (%s)", path.name, reason)
            return None
        assert header is not None

        payload_offset = int(header["payload_offset"])
        payload_nbytes = int(header["payload_nbytes"])
        tokens_offset = int(header["tokens_offset"])
        tokens_nbytes = int(header["tokens_nbytes"])

        read_records: List[int] = []
        buf = bytearray(payload_nbytes)
        tok_buf = bytearray(tokens_nbytes)
        t_read0 = time.perf_counter()
        try:
            f = io.FileIO(str(path), "r")
        except OSError:
            self.stats.record_refusal("io_error")
            return None
        try:
            self.last_nocache_applied = _set_nocache(f.fileno(), self.nocache)
            if tokens_nbytes:
                f.seek(tokens_offset)
                got = _read_exact(f, memoryview(tok_buf), self.chunk, read_records)
                if got != tokens_nbytes:
                    self.stats.record_refusal("truncated")
                    return None
            f.seek(payload_offset)
            got = _read_exact(f, memoryview(buf), self.chunk, read_records)
            if got != payload_nbytes:
                self.stats.record_refusal("truncated")
                return None
        except OSError:
            self.stats.record_refusal("io_error")
            logger.warning("vault-disk: read failed for %s", path.name, exc_info=True)
            return None
        finally:
            f.close()
        read_s = time.perf_counter() - t_read0

        try:
            payload = mx.array(memoryview(buf))
            mx.eval(payload)
            manifest = {
                "tree": header["tree"],
                "offsets": header["arrays"],
                "total_bytes": payload_nbytes,
                "version": header.get("manifest_version", 1),
            }
            frags = unpack_fragments(manifest, payload)
        except Exception:  # noqa: BLE001 - a corrupt blob must not raise at a caller
            self.stats.record_refusal("rebuild_failed")
            logger.warning("vault-disk: could not rebuild %s", path.name, exc_info=True)
            return None

        token_ids = [
            int.from_bytes(tok_buf[i : i + 4], "little", signed=True)
            for i in range(0, tokens_nbytes, 4)
        ]
        total_s = time.perf_counter() - t0
        self.last_read_records = read_records
        self.stats.record_restore(payload_nbytes + tokens_nbytes, read_s, total_s)
        with self._lock:
            rec = self._index.get(key)
            if rec is not None:
                rec["last_used"] = time.time()
                self._write_index_locked()
        return header, frags, token_ids

    def restore_into_vault(
        self,
        vault: Any,
        tokens: Sequence[int],
        tier: VaultTier = VaultTier.PREFILL,
        *,
        min_depth: int = 0,
        strict: bool = True,
        wait_s: float = 0.0,
    ) -> Optional[VaultCheckpoint]:
        """Promote the deepest usable disk entry into ``vault``; never raises.

        Step one of the documented two-step: after this returns, the entry is an
        ordinary RAM rung with ordinary byte accounting and ordinary eviction,
        and the caller's next ``vault.lookup`` serves it by the same
        longest-strict-prefix rule as everything else.
        """
        try:
            if wait_s > 0:
                self._wait_for_writes(wait_s)
            rec = self.best_record(tokens, tier, min_depth=min_depth, strict=strict)
            if rec is None:
                self.stats.bump("disk_misses")
                return None
            tier_s = tier.value if isinstance(tier, VaultTier) else str(tier)
            loaded = self.load_entry(
                rec["key"], expect_prompt_sha=rec.get("prompt_sha256"), tier=tier_s
            )
            if loaded is None:
                self.stats.bump("disk_misses")
                return None
            header, frags, _token_ids = loaded
            depth = int(header["prefix_len"])
            vault.insert(
                list(tokens),
                depth,
                frags,
                tier=tier,
                session_id=str(header.get("session_id", "")),
                # Carry the provenance back onto the RAM rung.  Without this a
                # restore-then-evict cycle would launder a width-1 entry into an
                # unknown-provenance one and the durability gate would then
                # refuse to re-persist the very blob it just read.
                harvest_provenance={
                    f: header.get(f) for f in _harvest_prov.FIELDS
                },
            )
            cp = vault.lookup(list(tokens), tier=tier)
            if cp is None or int(cp.prefix_len) < depth:
                self.stats.bump("disk_misses")
                return None
            self.stats.bump("disk_hits")
            return cp
        except Exception:  # noqa: BLE001 - a disk fault costs a cold prefill, nothing more
            logger.warning("vault-disk: restore failed; falling back to a cold prefill",
                           exc_info=True)
            self.stats.bump("disk_misses")
            return None

    def _wait_for_writes(self, timeout: float) -> None:
        with self._lock:
            events = list(self._in_flight.values())
        deadline = time.monotonic() + timeout
        for ev in events:
            ev.wait(max(0.0, deadline - time.monotonic()))

    # -- lifecycle -------------------------------------------------------

    def close(self, timeout: float = 30.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush(timeout)
            self._q.put(None)
            self._writer.join(timeout=timeout)
        except Exception:  # noqa: BLE001
            pass

    def stats_dict(self) -> dict:
        d = self.stats.snapshot()
        d.update(
            {
                "enabled": True,
                "dir": str(self.dir),
                "entries": len(self._index),
                "disk_bytes": self.disk_bytes,
                "cap_bytes": self.cap_bytes,
                "chunk_bytes": self.chunk,
                "nocache": self.nocache,
                "fsync": self.fsync,
                "save_on_insert": self.save_on_insert,
            }
        )
        return d


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def attach_disk_vault(
    vault: Any,
    *,
    root: Any = None,
    model_path: Any = None,
    flags: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[DiskPrefixVault]:
    """Give ``vault`` a disk tier. Returns it, or ``None`` when the tier is off.

    A fault here must never fail a model load: the disk tier is an optimisation
    over an optimisation, and the fallback is the RAM vault behaving exactly as
    it did before this module existed.
    """
    if vault is None:
        return None
    root = disk_vault_dir() if root is None else Path(root)
    if root is None:
        return None
    try:
        dv = DiskPrefixVault(
            root,
            getattr(vault, "identity", ""),
            model_identity=model_identity_hash(model_path) if model_path else "",
            tokenizer_hash=tokenizer_identity_hash(model_path) if model_path else "",
            flags=flags,
            **kwargs,
        )
    except Exception:  # noqa: BLE001
        logger.warning("vault-disk: could not open %s; continuing RAM-only", root,
                       exc_info=True)
        return None
    vault.disk = dv
    logger.info(
        "vault-disk: enabled at %s, cap %.0f GB, %d entries resident, "
        "chunk %d MiB, nocache=%s (restore rate is measured per restore, not "
        "benchmarked at startup)",
        dv.dir, dv.cap_bytes / 1e9, len(dv.records()), dv.chunk >> 20, dv.nocache,
    )
    return dv
