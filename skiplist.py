"""Probabilistic skip list with logarithmic average-case operations.

Each node randomly gets a level 1..MAX_LEVEL with P(level >= L) = p^(L-1).
Search walks top-down, skipping forward at each level until the next
node would overshoot the target, then drops down a level. Average
search/insert/delete is O(log n) without the rotation complexity of a
balanced BST.
"""

from __future__ import annotations

import random
from typing import Iterator, Optional


class _Node:
    __slots__ = ("key", "value", "forward")

    def __init__(self, key, value, level: int) -> None:
        self.key = key
        self.value = value
        self.forward: list[Optional["_Node"]] = [None] * level


class SkipList:
    """Sorted associative container backed by a probabilistic skip list.

    Inserts maintain sorted order by `key`. Duplicate keys overwrite.
    """

    def __init__(self, max_level: int = 16, p: float = 0.5, seed: int | None = None) -> None:
        if max_level < 1:
            raise ValueError("max_level must be >= 1")
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")
        self._max_level = max_level
        self._p = p
        self._level = 1
        # Header has the key/value sentinel; only its forward pointers matter.
        self._header: _Node = _Node(None, None, max_level)
        self._size = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._size

    def _random_level(self) -> int:
        level = 1
        while self._rng.random() < self._p and level < self._max_level:
            level += 1
        return level

    def insert(self, key, value) -> None:
        update: list[_Node] = [self._header] * self._max_level
        node = self._header
        for i in reversed(range(self._level)):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node

        candidate = node.forward[0]
        if candidate is not None and candidate.key == key:
            candidate.value = value
            return

        new_level = self._random_level()
        if new_level > self._level:
            for i in range(self._level, new_level):
                update[i] = self._header
            self._level = new_level

        new_node = _Node(key, value, new_level)
        for i in range(new_level):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self._size += 1

    def search(self, key):
        node = self._header
        for i in reversed(range(self._level)):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
        node = node.forward[0]
        if node is not None and node.key == key:
            return node.value
        raise KeyError(key)

    def delete(self, key) -> None:
        update: list[_Node] = [self._header] * self._max_level
        node = self._header
        for i in reversed(range(self._level)):
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
            update[i] = node

        target = node.forward[0]
        if target is None or target.key != key:
            raise KeyError(key)

        for i in range(self._level):
            if update[i].forward[i] is not target:
                break
            update[i].forward[i] = target.forward[i]

        # Trim empty top levels
        while self._level > 1 and self._header.forward[self._level - 1] is None:
            self._level -= 1
        self._size -= 1

    def __contains__(self, key) -> bool:
        try:
            self.search(key)
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator:
        node = self._header.forward[0]
        while node is not None:
            yield node.key
            node = node.forward[0]

    def items(self) -> Iterator[tuple]:
        node = self._header.forward[0]
        while node is not None:
            yield (node.key, node.value)
            node = node.forward[0]

    def height_distribution(self) -> dict[int, int]:
        """Return {level_count: number_of_nodes_with_that_height}."""
        counts: dict[int, int] = {}
        node = self._header.forward[0]
        while node is not None:
            h = len(node.forward)
            counts[h] = counts.get(h, 0) + 1
            node = node.forward[0]
        return counts

    @property
    def level(self) -> int:
        return self._level
