"""LRU cache with O(1) get / put using a doubly-linked list + hash map.

The classic interview design: the hash map gives O(1) key lookup, the
doubly-linked list gives O(1) move-to-front and O(1) eviction at the
tail. Each node sits in both data structures.
"""

from __future__ import annotations

from typing import Any


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key, value) -> None:
        self.key = key
        self.value = value
        self.prev: "_Node | None" = None
        self.next: "_Node | None" = None


class LRUCache:
    """Bounded least-recently-used cache.

    `get(key)` returns the cached value and marks the entry as most
    recently used. `put(key, value)` inserts or updates and evicts the
    LRU entry if capacity is exceeded.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._map: dict[Any, _Node] = {}
        # Sentinel head and tail simplify boundary handling.
        self._head = _Node(None, None)
        self._tail = _Node(None, None)
        self._head.next = self._tail
        self._tail.prev = self._head
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._map)

    def _add_to_front(self, node: _Node) -> None:
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_front(self, node: _Node) -> None:
        self._remove(node)
        self._add_to_front(node)

    def get(self, key) -> Any:
        node = self._map.get(key)
        if node is None:
            self._misses += 1
            raise KeyError(key)
        self._hits += 1
        self._move_to_front(node)
        return node.value

    def put(self, key, value) -> None:
        existing = self._map.get(key)
        if existing is not None:
            existing.value = value
            self._move_to_front(existing)
            return

        node = _Node(key, value)
        self._map[key] = node
        self._add_to_front(node)

        if len(self._map) > self._capacity:
            lru = self._tail.prev
            self._remove(lru)
            del self._map[lru.key]

    def __contains__(self, key) -> bool:
        return key in self._map

    def peek(self, key) -> Any:
        """Return value without affecting recency. Useful for inspection."""
        node = self._map.get(key)
        if node is None:
            raise KeyError(key)
        return node.value

    def keys_in_order(self) -> list:
        """Most-recently-used first."""
        out = []
        node = self._head.next
        while node is not self._tail:
            out.append(node.key)
            node = node.next
        return out

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def stats(self) -> dict[str, int | float]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": len(self._map),
            "capacity": self._capacity,
        }
