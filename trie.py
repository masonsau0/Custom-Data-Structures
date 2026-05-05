"""Trie (prefix tree) for fast autocomplete / prefix-set membership.

Insert and search are O(L) in the length of the key, independent of
the size of the dictionary. The classic data structure behind
autocomplete, IP routing tables, and spell-check.
"""

from __future__ import annotations

from typing import Iterator


class _Node:
    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: dict[str, "_Node"] = {}
        self.is_word: bool = False


class Trie:
    """Set of strings supporting prefix queries in O(L)."""

    def __init__(self) -> None:
        self._root = _Node()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, word: str) -> None:
        node = self._root
        for ch in word:
            child = node.children.get(ch)
            if child is None:
                child = _Node()
                node.children[ch] = child
            node = child
        if not node.is_word:
            node.is_word = True
            self._size += 1

    def __contains__(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_word

    def starts_with(self, prefix: str) -> bool:
        return self._find(prefix) is not None

    def _find(self, s: str) -> _Node | None:
        node = self._root
        for ch in s:
            child = node.children.get(ch)
            if child is None:
                return None
            node = child
        return node

    def words_with_prefix(self, prefix: str, limit: int = 25) -> list[str]:
        """Return up to `limit` complete words sharing `prefix`."""
        start = self._find(prefix)
        if start is None:
            return []
        out: list[str] = []
        self._collect(start, prefix, out, limit)
        return out

    def _collect(self, node: _Node, current: str, out: list[str], limit: int) -> None:
        if len(out) >= limit:
            return
        if node.is_word:
            out.append(current)
            if len(out) >= limit:
                return
        for ch, child in sorted(node.children.items()):
            self._collect(child, current + ch, out, limit)
            if len(out) >= limit:
                return

    def delete(self, word: str) -> bool:
        """Remove `word`; return True if it was present."""
        if word not in self:
            return False
        self._delete(self._root, word, 0)
        self._size -= 1
        return True

    def _delete(self, node: _Node, word: str, depth: int) -> bool:
        """Recursive helper. Returns True if `node` can be pruned."""
        if depth == len(word):
            node.is_word = False
            return not node.children
        ch = word[depth]
        child = node.children.get(ch)
        if child is None:
            return False
        if self._delete(child, word, depth + 1):
            del node.children[ch]
            return not node.is_word and not node.children
        return False

    def __iter__(self) -> Iterator[str]:
        out: list[str] = []
        self._collect(self._root, "", out, limit=10**9)
        return iter(out)
