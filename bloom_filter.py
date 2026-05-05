"""Bloom filter: probabilistic membership with no false negatives.

A bit array of size m and k hash functions. `add(x)` sets k bits;
`contains(x)` returns True iff all k bits are set. Missing items
(definite no) are exact; present items can yield false positives
with probability roughly (1 - e^(-kn/m))^k.

The two-hash trick (Kirsch & Mitzenmacher 2008) generates the k hash
positions from two independent base hashes, avoiding the need for k
separate hash functions while preserving the FPR bound.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any


class BloomFilter:
    """Space-efficient probabilistic set.

    Construct with `BloomFilter.optimal(expected_n, target_fpr)` to
    auto-size m and k for a target false-positive rate at an expected
    item count.
    """

    def __init__(self, m: int, k: int) -> None:
        if m < 1:
            raise ValueError("m (bit-array size) must be >= 1")
        if k < 1:
            raise ValueError("k (hash count) must be >= 1")
        self._m = m
        self._k = k
        self._bits = bytearray((m + 7) // 8)
        self._n = 0  # inserted-item count (with duplicates)

    @classmethod
    def optimal(cls, expected_n: int, target_fpr: float) -> "BloomFilter":
        """Construct with m, k sized for `target_fpr` at `expected_n` items."""
        if expected_n < 1:
            raise ValueError("expected_n must be >= 1")
        if not 0 < target_fpr < 1:
            raise ValueError("target_fpr must be in (0, 1)")
        m = max(1, math.ceil(-(expected_n * math.log(target_fpr)) / (math.log(2) ** 2)))
        k = max(1, round((m / expected_n) * math.log(2)))
        return cls(m, k)

    def _hashes(self, item: Any) -> tuple[int, int]:
        encoded = repr(item).encode("utf-8")
        # Two independent base hashes via different SHA-256 prefixes.
        h1 = int.from_bytes(hashlib.sha256(b"\x00" + encoded).digest()[:8], "big")
        h2 = int.from_bytes(hashlib.sha256(b"\x01" + encoded).digest()[:8], "big")
        return h1, h2

    def _positions(self, item: Any) -> list[int]:
        h1, h2 = self._hashes(item)
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def _set(self, position: int) -> None:
        self._bits[position >> 3] |= 1 << (position & 7)

    def _get(self, position: int) -> bool:
        return bool(self._bits[position >> 3] & (1 << (position & 7)))

    def add(self, item: Any) -> None:
        for pos in self._positions(item):
            self._set(pos)
        self._n += 1

    def __contains__(self, item: Any) -> bool:
        return all(self._get(pos) for pos in self._positions(item))

    def estimated_fpr(self) -> float:
        """Theoretical false-positive rate given current load."""
        if self._n == 0:
            return 0.0
        return (1 - math.exp(-self._k * self._n / self._m)) ** self._k

    @property
    def m(self) -> int:
        return self._m

    @property
    def k(self) -> int:
        return self._k

    @property
    def n(self) -> int:
        return self._n

    def memory_bytes(self) -> int:
        return len(self._bits)

    def bits_set(self) -> int:
        return sum(bin(b).count("1") for b in self._bits)
