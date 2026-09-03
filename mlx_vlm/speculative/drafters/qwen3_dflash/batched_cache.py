"""One batched context cache for the DFlash drafter, replacing B scalar ones.

``_dflash_rounds_batch`` drafts row by row because the drafter's cache carries a
SCALAR offset and one row's committed context is a different length from the
next row's the moment per-row rollback keeps ragged accepts.  Both facts are
properties of the cache, not of the drafter's arithmetic: ``_hidden``,
``draft_block`` and the candidate selector are already shape-polymorphic in B.

This cache removes both by borrowing the representation the TARGET's batch
caches already use -- per-row ``offset``, per-row ``left_padding``, all padding
kept contiguous at the LEFT -- and adding the one thing the drafter needs on top
of it: a per-round ``pending_lengths``, because a drafter round appends a
DIFFERENT number of context rows per row (each row's committed tokens), where a
target decode step appends exactly one.

Layout invariant, maintained after every update:

    column c of row j is real  <=>  c >= left_padding[j]
    left_padding[j] = physical_length - real_rows[j]
    min(left_padding) == 0                       (no column is pad for every row)

so the cache is exactly as wide as the LONGEST row's context and no wider.

WHY THAT IS THE SAME ANSWER the per-row ``RotatingKVCache`` gives.  The rotating
cache trims ``trim_size = length - max_size + 1`` from the front before each
concat.  Here the trim is applied to the shared physical length and the pad
prefix absorbs it first, so the real rows row ``j`` loses are

    max(0, trim_size - left_padding[j])
      = max(0, (L - max_size + 1) - (L - real_j))
      = max(0, real_j - max_size + 1)

which is the trim the row's OWN scalar cache would have computed from its own
length.  The eviction is therefore per-row exact, not merely close, and it stays
exact once the sliding window saturates.

The DFlash draft block attends the whole resident context non-causally with no
mask (``DFlashAttention.__call__``), so the ORDER of the context rows does not
change the result -- only the SET does.  That is why a concat-only cache here is
interchangeable with the rotating cache's in-place ring for S == 1 rounds: the
retained set is the same ``max_size`` rows either way.
"""

from typing import List, Optional, Sequence

import mlx.core as mx

from ....models.cache import _BaseCache, dynamic_roll


class BatchDFlashKVCache(_BaseCache):
    """Batched, concat-only KV cache for DFlash drafter context rows.

    ``max_size`` is the sliding layer's resident window (``sliding_window - 1``)
    or ``None`` for a full-attention layer.
    """

    # Marks this cache to ``DFlashAttention`` as the batched variant. Checked by
    # attribute rather than isinstance so a subclass or a test double works.
    dflash_batched = True

    def __init__(self, batch_size: int, max_size: Optional[int] = None):
        if int(batch_size) < 1:
            raise ValueError("BatchDFlashKVCache needs a positive batch size.")
        self.keys = None
        self.values = None
        self.max_size = None if max_size is None else int(max_size)
        self._batch = int(batch_size)
        # Host-side bookkeeping on purpose: every quantity here is known to the
        # Python caller (the committed-token counts), so keeping them as ints
        # costs no device sync in the round loop.
        self._offset: List[int] = [0] * self._batch
        self._left: List[int] = [0] * self._batch
        self._length = 0
        self._pending: Optional[List[int]] = None
        self._offset_arr: Optional[mx.array] = None

    # ------------------------------------------------------------------ state
    @property
    def batch_size(self) -> int:
        return self._batch

    @property
    def offset(self) -> mx.array:
        """Per-row RoPE position of the NEXT context row."""
        if self._offset_arr is None:
            self._offset_arr = mx.array(self._offset, dtype=mx.int32)
        return self._offset_arr

    @property
    def left_padding(self) -> List[int]:
        return list(self._left)

    @property
    def context_lengths(self) -> List[int]:
        return [self._length - pad for pad in self._left]

    def _dirty(self) -> None:
        self._offset_arr = None

    # ---------------------------------------------------------- round wiring
    def set_pending_lengths(self, lengths: Sequence[int]) -> None:
        """How many of the next block's rows are REAL, per row.

        Set once per draft round, before the forward. The block itself is
        right-padded (real rows first), which is what lets a single roll move
        every row's new pad into the shared left prefix.
        """
        lengths = [int(v) for v in lengths]
        if len(lengths) != self._batch:
            raise ValueError(
                f"BatchDFlashKVCache has {self._batch} rows but was given "
                f"{len(lengths)} pending lengths."
            )
        if any(v < 0 for v in lengths):
            raise ValueError("Pending context lengths cannot be negative.")
        self._pending = lengths

    def _lengths_for(self, block: int) -> List[int]:
        if self._pending is None:
            return [block] * self._batch
        return [min(v, block) for v in self._pending]

    def consume_context_prefix(self, skip: int) -> None:
        """Account for ``skip`` context rows a caller dropped off the FRONT.

        The scalar-cache half of this is ``cache.offset += skip``
        (``DFlashDraftModel.adopt_pretruncated_context`` and the sliding-layer
        discard in ``DFlashAttention``). Here the pending real-row counts move
        with it, because the rows that were dropped were real rows.
        """
        skip = int(skip)
        if skip <= 0:
            return
        if self._pending is not None:
            if len(set(self._pending)) > 1:
                # A ragged round block wider than the resident window would need
                # a per-row front discard, which this cache does not represent.
                # Loud rather than a silently different context.
                raise RuntimeError(
                    "BatchDFlashKVCache cannot drop a context prefix from a "
                    f"RAGGED block (lengths {self._pending}, skip {skip}). The "
                    "drafter's sliding window is narrower than one draft round."
                )
            self._pending = [max(0, v - skip) for v in self._pending]
        self._offset = [off + skip for off in self._offset]
        self._dirty()

    def query_offset(self, block: int) -> mx.array:
        """Per-row RoPE offset of the proposal block: past this row's context."""
        lengths = self._lengths_for(block)
        return mx.array(
            [off + n for off, n in zip(self._offset, lengths)], dtype=mx.int32
        )

    # ---------------------------------------------------------------- update
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        block = keys.shape[2]
        if keys.shape[0] != self._batch:
            raise ValueError(
                f"BatchDFlashKVCache has {self._batch} rows but was handed "
                f"{keys.shape[0]}."
            )
        lengths = self._lengths_for(block)

        if self.keys is None:
            self.keys, self.values = keys, values
            self._length = block
            left = [block - n for n in lengths]
        else:
            if self.max_size is not None:
                # Mirror RotatingKVCache._update_concat: keep max_size + S - 1 so
                # every proposal position still sees max_size context rows.
                trim = self._length - self.max_size + 1
                if trim > 0:
                    self.keys = self.keys[..., trim:, :]
                    self.values = self.values[..., trim:, :]
                    self._length -= trim
                    self._left = [max(0, pad - trim) for pad in self._left]
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
            left = [pad + block - n for pad, n in zip(self._left, lengths)]
            self._length += block

        roll = [block - n for n in lengths]
        if max(roll) > 0:
            shifts = mx.array(roll, dtype=mx.int32)[:, None]
            self.keys = dynamic_roll(self.keys, shifts, axis=2)
            self.values = dynamic_roll(self.values, shifts, axis=2)

        # Compact: a column that is pad for EVERY row carries nothing.
        drop = min(left)
        if drop > 0:
            self.keys = self.keys[..., drop:, :]
            self.values = self.values[..., drop:, :]
            self._length -= drop
            left = [pad - drop for pad in left]

        self._left = left
        self._offset = [off + n for off, n in zip(self._offset, lengths)]
        self._pending = None
        self._dirty()
        return self.keys, self.values

    def context_mask(self, n_proposal: int):
        """``[B, 1, 1, L + n_proposal]`` bool, or ``None`` when nothing is pad.

        ``None`` matters: with no ragged rows the batched path then calls
        ``scaled_dot_product_attention`` with exactly the argument the row-wise
        path calls it with, so the two cannot differ for a shape-identical
        reason.
        """
        if self.keys is None or max(self._left) == 0:
            return None
        columns = mx.arange(self._length, dtype=mx.int32)[None, :] >= mx.array(
            self._left, dtype=mx.int32
        )[:, None]
        if n_proposal:
            columns = mx.concatenate(
                [columns, mx.ones((self._batch, n_proposal), dtype=mx.bool_)], axis=1
            )
        return columns[:, None, None, :]

    def filter(self, keep) -> None:
        """Keep only ``keep`` (active-slot indices), in order."""
        keep = [int(i) for i in keep]
        if self.keys is not None:
            index = mx.array(keep, dtype=mx.int32)
            self.keys = self.keys[index]
            self.values = self.values[index]
        self._offset = [self._offset[i] for i in keep]
        self._left = [self._left[i] for i in keep]
        if self._pending is not None:
            self._pending = [self._pending[i] for i in keep]
        self._batch = len(keep)
        self._dirty()
        # Dropping rows can leave a column that is pad for every survivor.
        drop = min(self._left) if self._left else 0
        if drop > 0 and self.keys is not None:
            self.keys = self.keys[..., drop:, :]
            self.values = self.values[..., drop:, :]
            self._length -= drop
            self._left = [pad - drop for pad in self._left]

    # ------------------------------------------------------------- plumbing
    def size(self) -> int:
        return self._length

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self._batch,
                    self.max_size if self.max_size is not None else -1,
                    self._length,
                    ",".join(map(str, self._offset)),
                    ",".join(map(str, self._left)),
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        batch, max_size, length, offsets, lefts = v
        self._batch = int(batch)
        self.max_size = None if int(max_size) < 0 else int(max_size)
        self._length = int(length)
        self._offset = [int(x) for x in offsets.split(",")] if offsets else []
        self._left = [int(x) for x in lefts.split(",")] if lefts else []
        self._pending = None
        self._dirty()


__all__ = ["BatchDFlashKVCache"]
