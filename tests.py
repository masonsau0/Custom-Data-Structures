"""Correctness tests for every data structure.

Run with::

    python tests.py

Each `test_*` function asserts a specific behavioural property; the
script exits non-zero if any assertion fails. No external test
framework; the goal is to keep the project free of dependencies you'd
need to read source code for.
"""

from __future__ import annotations

import random
import sys
import traceback

from bloom_filter import BloomFilter
from hashmap import HashMap
from lru_cache import LRUCache
from skiplist import SkipList
from trie import Trie


# --- HashMap ----------------------------------------------------------------

def test_hashmap_basic() -> None:
    h = HashMap()
    h["a"] = 1
    h["b"] = 2
    assert len(h) == 2
    assert h["a"] == 1
    assert h["b"] == 2
    h["a"] = 10
    assert h["a"] == 10
    assert len(h) == 2


def test_hashmap_delete_and_reinsert() -> None:
    h = HashMap()
    for i in range(50):
        h[i] = i * i
    del h[25]
    assert 25 not in h
    h[25] = 999
    assert h[25] == 999
    assert len(h) == 50


def test_hashmap_large() -> None:
    h = HashMap()
    for i in range(2000):
        h[f"key_{i}"] = i
    assert len(h) == 2000
    for i in range(2000):
        assert h[f"key_{i}"] == i
    assert h.capacity > 2000  # has resized


def test_hashmap_keyerror() -> None:
    h = HashMap()
    h["x"] = 1
    try:
        _ = h["missing"]
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


# --- SkipList ---------------------------------------------------------------

def test_skiplist_basic() -> None:
    s = SkipList(seed=42)
    for k in [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        s.insert(k, str(k))
    assert s.search(4) == "4"
    assert 9 in s
    assert 100 not in s


def test_skiplist_sorted_iter() -> None:
    s = SkipList(seed=0)
    keys = list(range(200))
    random.Random(7).shuffle(keys)
    for k in keys:
        s.insert(k, k * 2)
    out = list(s)
    assert out == sorted(set(keys))


def test_skiplist_delete() -> None:
    s = SkipList(seed=1)
    for i in range(50):
        s.insert(i, i)
    s.delete(25)
    assert 25 not in s
    try:
        s.delete(25)
        raise AssertionError("expected KeyError on second delete")
    except KeyError:
        pass


# --- LRUCache ---------------------------------------------------------------

def test_lru_basic() -> None:
    c = LRUCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # should evict b (b is LRU now since "a" was just touched)
    assert "b" not in c
    assert "a" in c
    assert "c" in c


def test_lru_overwrite() -> None:
    c = LRUCache(capacity=2)
    c.put("a", 1)
    c.put("a", 2)
    assert c.peek("a") == 2
    assert len(c) == 1


def test_lru_hit_rate() -> None:
    c = LRUCache(capacity=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.get("a")  # hit
    c.get("a")  # hit
    try:
        c.get("z")  # miss
    except KeyError:
        pass
    assert c.stats["hits"] == 2
    assert c.stats["misses"] == 1
    assert abs(c.hit_rate - 2/3) < 1e-9


# --- BloomFilter ------------------------------------------------------------

def test_bloom_no_false_negatives() -> None:
    bf = BloomFilter.optimal(expected_n=1000, target_fpr=0.01)
    items = [f"item_{i}" for i in range(1000)]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf, f"false negative: {item}"


def test_bloom_fpr_within_target() -> None:
    rng = random.Random(0)
    bf = BloomFilter.optimal(expected_n=5000, target_fpr=0.01)
    inserted = {f"in_{i}" for i in range(5000)}
    for x in inserted:
        bf.add(x)
    candidates = [f"out_{rng.random()}" for _ in range(5000)]
    fp = sum(1 for x in candidates if x in bf)
    observed = fp / len(candidates)
    # Allow generous slack — observed FPR should be bounded by ~3x target
    assert observed < 0.03, f"FPR {observed:.4f} exceeds 3x target"


# --- Trie -------------------------------------------------------------------

def test_trie_basic() -> None:
    t = Trie()
    for w in ("apple", "app", "apex", "banana"):
        t.insert(w)
    assert "apple" in t
    assert "app" in t
    assert "appl" not in t
    assert t.starts_with("appl")
    assert not t.starts_with("xyz")


def test_trie_prefix_completion() -> None:
    t = Trie()
    for w in ("car", "card", "care", "careful", "carpet", "dog"):
        t.insert(w)
    out = t.words_with_prefix("car")
    assert set(out) == {"car", "card", "care", "careful", "carpet"}


def test_trie_delete() -> None:
    t = Trie()
    t.insert("hello")
    t.insert("help")
    assert t.delete("hello") is True
    assert "hello" not in t
    assert "help" in t  # delete didn't prune shared path
    assert t.delete("hello") is False  # already gone


# --- runner -----------------------------------------------------------------

def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures: list[tuple[str, str]] = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            failures.append((fn.__name__, traceback.format_exc()))
    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print()
        for name, tb in failures:
            print(f"--- {name} ---")
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
