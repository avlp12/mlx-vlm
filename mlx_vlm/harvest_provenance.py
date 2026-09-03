"""Where an exact prefix-cache entry was harvested from, and the policies keyed on it.

Motivation (measured, `logs/sweep11/L1b1_apc_poison_RESULT.json`, 2026-09-03,
gesicht, GLM-5.3-Flash-vlm-q4-quasar): the mid-prefill APC exact checkpoint
(``generate/ar.py::_store_apc_exact_checkpoints``) is written **once per APC
lifetime**, on the row's first serve after the seed, and the snapshot it takes is
*bit-different* depending on how wide the prefill batch was at that column.  A
3,091-token entry harvested inside a B=2 prefill made every later SOLO serve of
that prompt return sha ``122e772a``; the same entry harvested at B=1 returned
``5d7c209c``.  Clearing the cache and re-harvesting at B=1 restored the clean
stream, 4/4 serves each way.

The carrier is **batch width at the checkpoint column**, not right padding: the
arm that settles it is an equal-suffix, zero-padding B=2 batch
(``right_pad_per_row=[0, 0]``), whose own rows decode 32/32 correctly and which
*still* poisons the entry to the same sha.  Batch-shaped floating point on the
KDA recurrent snapshot is the mechanism.  So a policy that only declines RIGHT
PADDING does not close it.

Nothing here changes what is harvested.  It records **where from**, exposes that
on the metrics surface and in the server's own prefill log line, and gives two
policies a fact to key on:

* ``MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY`` (default ``0``) -- determinism knob: a
  B=1 request accepts only entries with a known width-1 capture lineage.
* ``MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH`` (default ``1``) -- durability gate: a
  RAM entry dies with the process, a disk entry does not, so by default only a
  width-1 capture lineage is allowed to become durable.

Both are read at call time, not at import, so a test can flip them with
``monkeypatch.setenv`` without reloading the module.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

# Capture fields describe the forward that produced this snapshot. Lineage is
# separate: a B1 forward extending a B2 prefix is still a B1 capture, but cannot
# establish a B1-only history. Old five-field records retain reportable capture
# width and acquire unknown lineage on normalisation, so policy fails closed.
CAPTURE_FIELDS = (
    "harvest_batch_width",
    "harvest_right_pad",
    "harvest_left_pad",
    "harvest_git_head",
    "harvest_at",
)

FIELDS = CAPTURE_FIELDS + ("harvest_lineage_max_width",)

# What ``exact_entries_by_harvest_width`` and the log line report for an entry
# that carries no complete provenance record.
UNKNOWN_WIDTH_KEY = "unknown"

_GIT_HEAD_CACHE: Optional[str] = None


def git_head() -> str:
    """Revision of the running tree, cached, never raising.

    Deliberately the *same* probe ``vault_disk`` uses for its header field, so a
    ``harvest_git_head`` on an APC entry and a ``git_head`` on a vault blob are
    comparable strings (both carry the ``+dirty`` suffix when the tree is).
    """
    global _GIT_HEAD_CACHE
    if _GIT_HEAD_CACHE is None:
        try:
            from .context_vault import _code_identity

            _GIT_HEAD_CACHE = str(_code_identity())
        except Exception:  # noqa: BLE001 - provenance must never fail a store
            _GIT_HEAD_CACHE = "unknown"
    return _GIT_HEAD_CACHE


def make(
    batch_width: int,
    *,
    right_pad: int = 0,
    left_pad: int = 0,
    at: Optional[float] = None,
    prefix_len: int = 0,
    parent: Any = None,
) -> Dict[str, Any]:
    """Record this capture's width and the widest known reused ancestor.

    ``prefix_len == 0`` establishes a cold capture. A warm capture must supply
    its parent's record; absent or legacy ancestry stays unknown (stored as 0).

    ``right_pad`` / ``left_pad`` are **that row's** pads, not the batch's -- the
    poison is a property of the row's own snapshot, and a batch has one pad per
    row.  0 when the batch is unpadded.
    """
    width = max(1, int(batch_width))
    ancestor_width = lineage_width_of(parent) if prefix_len > 0 else width
    return {
        "harvest_batch_width": width,
        # 0 means some reused state has unknown ancestry. It must remain
        # unknown across later solo captures, even though this capture is B1.
        "harvest_lineage_max_width": (
            max(width, ancestor_width) if ancestor_width is not None else 0
        ),
        "harvest_right_pad": max(0, int(right_pad)),
        "harvest_left_pad": max(0, int(left_pad)),
        "harvest_git_head": git_head(),
        "harvest_at": float(at if at is not None else time.time()),
    }


def is_complete(prov: Any) -> bool:
    """True when ``prov`` carries every field of a provenance record."""
    return isinstance(prov, dict) and all(f in prov and prov[f] is not None for f in FIELDS)


def batch_width_of(prov: Any) -> Optional[int]:
    """Actual capture width, including legacy records without lineage."""
    if not isinstance(prov, dict) or not all(
        prov.get(f) is not None for f in CAPTURE_FIELDS
    ):
        return None
    try:
        width = int(prov["harvest_batch_width"])
        return width if width > 0 else None
    except (TypeError, ValueError):
        return None


def lineage_width_of(prov: Any) -> Optional[int]:
    """Maximum width over this capture and every reused ancestor; unknown fails closed.

    Old records can still report their capture width, but cannot establish that
    their own warm prefix was clean. Missing lineage therefore never means B1.
    """
    width = batch_width_of(prov)
    if width is None:
        return None
    try:
        lineage = int(prov.get("harvest_lineage_max_width", 0))
    except (TypeError, ValueError):
        return None
    return lineage if lineage >= width else None


def is_b1_eligible(prov: Any) -> bool:
    return lineage_width_of(prov) == 1


def width_key(prov: Any) -> Any:
    """Key for the ``exact_entries_by_harvest_width`` histogram."""
    w = batch_width_of(prov)
    return UNKNOWN_WIDTH_KEY if w is None else w


def normalise(prov: Any) -> Optional[Dict[str, Any]]:
    """Copy capture metadata; missing or invalid lineage remains unknown."""
    if batch_width_of(prov) is None:
        return None
    return {
        "harvest_batch_width": int(prov["harvest_batch_width"]),
        "harvest_lineage_max_width": lineage_width_of(prov) or 0,
        "harvest_right_pad": int(prov["harvest_right_pad"]),
        "harvest_left_pad": int(prov["harvest_left_pad"]),
        "harvest_git_head": str(prov["harvest_git_head"]),
        "harvest_at": float(prov["harvest_at"]),
    }


# -- policies ---------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


def serve_b1_from_b1_only() -> bool:
    """``MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY`` -- default OFF.

    ON: a request being admitted ALONE accepts only exact entries whose entire
    known capture lineage has width 1.  It falls through to the longest strict prefix among width-1
    entries, and to a cold prefill if there is none -- it does not fall back to
    a wider entry, because "the deepest entry" is exactly the wrong tiebreak
    when depth is what the poison rides in on.

    OFF by default because the cost is throughput (a re-earned prefill on every
    process that batched first) and the measured benefit on the one workload
    that has been looked at is one near-tie token in 32.  Nothing bounds it for
    other prompts -- open item L1b1-a -- which is why the knob exists at all.
    """
    return _env_int("MLX_VLM_APC_SERVE_B1_FROM_B1_ONLY", 0) != 0


def persist_max_harvest_width() -> int:
    """``MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH`` -- default 1.

    An entry is allowed to become DURABLE only when its entire known capture
    lineage has width <= this value; unknown ancestry is never durable.  ``0`` disables the
    gate entirely (persist any width, including unknown) and is the explicit
    opt-out for a deployment that wants disk throughput more than it wants a
    reproducible warm start.

    The name is the coordinator's, kept so the ruling and the code use one word.
    It reads like a floor and behaves like a ceiling on the harvest width, which
    is the same thing said twice: width 1 is the *best* provenance, so "at least
    this good" and "at most this wide" agree.
    """
    return max(0, _env_int("MLX_VLM_VAULT_DISK_PERSIST_MIN_WIDTH", 1))


def may_persist(prov: Any) -> bool:
    """Whether the full capture lineage is eligible for disk writes or reads."""
    cap = persist_max_harvest_width()
    if cap <= 0:
        return True
    w = lineage_width_of(prov)
    if w is None:
        return False
    return w <= cap
