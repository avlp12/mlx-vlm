"""Token-level n-gram index for prompt-lookup speculative decoding.

Pure Python and free of MLX: the whole point of a lookup drafter is that a round
costs no GPU work at all, so the matcher must stay on the CPU and must stay
O(1)-ish per emitted token.

Keys are rolling integer hashes rather than tuples -- a 100k-token context with
three n sizes would otherwise hold ~300k tuple objects.  Hashes collide, so a
hit is always re-verified against the actual tokens before it is proposed; a
collision therefore costs a comparison, never a wrong draft.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

_PRIME = 1000003
_MASK = (1 << 61) - 1


def _hash(tokens: Sequence[int], start: int, n: int) -> int:
    h = 0
    for i in range(start, start + n):
        h = (h * _PRIME + tokens[i] + 1) & _MASK
    return h


class NgramIndex:
    """Suffix -> continuation index over the running context.

    ``append`` is O(n_max - n_min + 1) hash updates; ``propose`` is O(n_max^2)
    in the worst case and touches no more than ``keep`` candidate positions per
    n size.  Neither depends on the context length.
    """

    __slots__ = ("n_min", "n_max", "keep", "_tokens", "_tables")

    def __init__(self, n_min: int = 3, n_max: int = 5, keep: int = 2):
        if not 1 <= n_min <= n_max:
            raise ValueError(f"require 1 <= n_min <= n_max, got {n_min}, {n_max}")
        if keep < 2:
            # keep >= 2 so the suffix's own occurrence can be skipped and a
            # genuine earlier one still found.
            raise ValueError(f"keep must be >= 2, got {keep}")
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.keep = int(keep)
        self._tokens: List[int] = []
        self._tables: Dict[int, Dict[int, Deque[int]]] = {
            n: {} for n in range(self.n_min, self.n_max + 1)
        }

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> List[int]:
        return self._tokens

    def reset(self) -> None:
        self._tokens.clear()
        for table in self._tables.values():
            table.clear()

    def append(self, token: int) -> None:
        """Add one committed token.  Only committed tokens are ever indexed, so
        a rejected speculative tail never has to be rolled back out."""
        self._tokens.append(int(token))
        end = len(self._tokens) - 1
        for n, table in self._tables.items():
            if end + 1 < n:
                continue
            key = _hash(self._tokens, end - n + 1, n)
            slot = table.get(key)
            if slot is None:
                table[key] = deque((end,), maxlen=self.keep)
            else:
                slot.append(end)

    def extend(self, tokens: Sequence[int]) -> None:
        for t in tokens:
            self.append(t)

    def _matches_at(self, pos: int, n: int) -> bool:
        """Verify a hash hit: do the n tokens ending at ``pos`` equal the n
        tokens ending at the current suffix?"""
        end = len(self._tokens) - 1
        a = pos - n + 1
        b = end - n + 1
        if a < 0:
            return False
        return self._tokens[a : pos + 1] == self._tokens[b : end + 1]

    def propose(self, k: int) -> Tuple[List[int], int, int]:
        """Longest-match lookup.

        Returns ``(continuation, n_matched, source_position)``; an empty
        continuation means abstain.  Longest n first, and within an n the most
        recent earlier occurrence -- which is what keeps a drafter that is
        mid-quotation walking forward through the span it is already copying.
        """
        if k <= 0:
            return [], 0, -1
        length = len(self._tokens)
        end = length - 1
        for n in range(self.n_max, self.n_min - 1, -1):
            if length < n:
                continue
            key = _hash(self._tokens, length - n, n)
            slot = self._tables[n].get(key)
            if not slot:
                continue
            for pos in reversed(slot):
                if pos >= end:
                    continue          # the suffix's own occurrence
                if not self._matches_at(pos, n):
                    continue          # hash collision
                nxt = self._tokens[pos + 1 : pos + 1 + k]
                if nxt:
                    return list(nxt), n, pos
        return [], 0, -1


__all__ = ["NgramIndex"]
