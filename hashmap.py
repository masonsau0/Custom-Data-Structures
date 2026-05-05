"""Open-addressing hash map with linear probing and dynamic resizing.

Average O(1) get / set / delete; resizes when load factor exceeds 0.7
and rehashes all entries. Uses tombstones to mark deleted slots so
probe sequences for live keys remain intact.
"""

from __future__ import annotations

from typing import Any, Iterator

_TOMBSTONE = object()  # sentinel for deleted slots
_EMPTY = object()      # sentinel for never-used slots


class HashMap:
    """Open-addressing hash map with linear probing.

    Operations are average-case O(1) given a low load factor and a
    well-distributed hash. Worst case is O(n) when the table is full of
    collisions or tombstones (the resize keeps that degenerate case
    bounded).
    """

    def __init__(self, initial_capacity: int = 16, max_load_factor: float = 0.7) -> None:
        if initial_capacity < 4:
            raise ValueError("initial_capacity must be >= 4")
        if not 0 < max_load_factor < 1:
            raise ValueError("max_load_factor must be in (0, 1)")
        self._capacity = initial_capacity
        self._max_load = max_load_factor
        self._size = 0
        self._slots: list[Any] = [_EMPTY] * initial_capacity

    def __len__(self) -> int:
        return self._size

    def _index(self, key: Any) -> int:
        return hash(key) & (self._capacity - 1)  # capacity is power of 2

    def _probe(self, key: Any) -> int:
        """Return the slot index where `key` lives or where it should go.

        Walks the probe sequence skipping over tombstones, returning the
        first matching key slot or, failing that, the first empty slot
        (or tombstone, if no empty slot was seen first).
        """
        idx = self._index(key)
        first_tombstone = -1
        for _ in range(self._capacity):
            slot = self._slots[idx]
            if slot is _EMPTY:
                return first_tombstone if first_tombstone != -1 else idx
            if slot is _TOMBSTONE:
                if first_tombstone == -1:
                    first_tombstone = idx
            else:
                k, _v = slot
                if k == key:
                    return idx
            idx = (idx + 1) & (self._capacity - 1)
        return first_tombstone  # table is full of tombstones — caller should resize

    def __setitem__(self, key: Any, value: Any) -> None:
        idx = self._probe(key)
        slot = self._slots[idx]
        if slot is _EMPTY or slot is _TOMBSTONE:
            self._slots[idx] = (key, value)
            self._size += 1
            if self._size / self._capacity > self._max_load:
                self._resize(self._capacity * 2)
        else:
            self._slots[idx] = (key, value)  # overwrite

    def __getitem__(self, key: Any) -> Any:
        idx = self._probe(key)
        slot = self._slots[idx]
        if slot is _EMPTY or slot is _TOMBSTONE:
            raise KeyError(key)
        k, v = slot
        if k != key:
            raise KeyError(key)
        return v

    def __delitem__(self, key: Any) -> None:
        idx = self._probe(key)
        slot = self._slots[idx]
        if slot is _EMPTY or slot is _TOMBSTONE:
            raise KeyError(key)
        k, _v = slot
        if k != key:
            raise KeyError(key)
        self._slots[idx] = _TOMBSTONE
        self._size -= 1

    def __contains__(self, key: Any) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator[Any]:
        for slot in self._slots:
            if slot is not _EMPTY and slot is not _TOMBSTONE:
                yield slot[0]

    def items(self) -> Iterator[tuple[Any, Any]]:
        for slot in self._slots:
            if slot is not _EMPTY and slot is not _TOMBSTONE:
                yield slot

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def _resize(self, new_capacity: int) -> None:
        old_slots = self._slots
        self._capacity = new_capacity
        self._slots = [_EMPTY] * new_capacity
        self._size = 0
        for slot in old_slots:
            if slot is not _EMPTY and slot is not _TOMBSTONE:
                k, v = slot
                self[k] = v

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def load_factor(self) -> float:
        return self._size / self._capacity
