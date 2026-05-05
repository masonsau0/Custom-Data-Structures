"""Benchmark every data structure against its stdlib counterpart.

Writes `benchmarks.csv` with one row per (structure, n, operation,
implementation) timing measurement. The Streamlit dashboard reads this
to plot complexity curves on log-log axes; the slope of each curve
should match the theoretical complexity (HashMap O(1), SkipList
O(log n), etc.).

Run once locally to refresh the data; the CSV is committed so the
dashboard works on first launch.
"""

from __future__ import annotations

import bisect
import random
import time
from pathlib import Path

import pandas as pd

from bloom_filter import BloomFilter
from hashmap import HashMap
from lru_cache import LRUCache
from skiplist import SkipList
from trie import Trie

SIZES = [100, 500, 1000, 5000, 10_000, 50_000]
INSERT_REPEATS = 3
LOOKUP_REPEATS = 3
RANDOM_SEED = 17


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000  # ms


def _bench(label: str, fn, repeats: int) -> float:
    return min(_time(fn) for _ in range(repeats))


def bench_hashmap(rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for n in SIZES:
        keys = [f"k_{rng.random()}_{i}" for i in range(n)]
        lookup_keys = rng.sample(keys, k=min(n, 1000))

        # Insert
        rows.append({"structure": "HashMap", "implementation": "Custom",
                     "n": n, "operation": "insert",
                     "elapsed_ms": _bench("hm-ins", lambda: _hm_insert(keys), INSERT_REPEATS)})
        rows.append({"structure": "HashMap", "implementation": "stdlib (dict)",
                     "n": n, "operation": "insert",
                     "elapsed_ms": _bench("dict-ins", lambda: _dict_insert(keys), INSERT_REPEATS)})

        # Lookup (after pre-populating)
        hm = HashMap()
        for k in keys:
            hm[k] = 1
        d = {k: 1 for k in keys}
        rows.append({"structure": "HashMap", "implementation": "Custom",
                     "n": n, "operation": "lookup",
                     "elapsed_ms": _bench("hm-look", lambda: _hm_lookup(hm, lookup_keys), LOOKUP_REPEATS)})
        rows.append({"structure": "HashMap", "implementation": "stdlib (dict)",
                     "n": n, "operation": "lookup",
                     "elapsed_ms": _bench("dict-look", lambda: _dict_lookup(d, lookup_keys), LOOKUP_REPEATS)})
    return rows


def _hm_insert(keys):
    hm = HashMap()
    for k in keys:
        hm[k] = 1


def _dict_insert(keys):
    d: dict = {}
    for k in keys:
        d[k] = 1


def _hm_lookup(hm, lookup_keys):
    for k in lookup_keys:
        _ = hm[k]


def _dict_lookup(d, lookup_keys):
    for k in lookup_keys:
        _ = d[k]


def bench_skiplist(rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for n in SIZES:
        keys = list(range(n))
        rng.shuffle(keys)
        lookup_keys = rng.sample(keys, k=min(n, 1000))

        # Insert
        rows.append({"structure": "SkipList", "implementation": "Custom",
                     "n": n, "operation": "insert",
                     "elapsed_ms": _bench("sl-ins", lambda: _sl_insert(keys), INSERT_REPEATS)})
        rows.append({"structure": "SkipList", "implementation": "stdlib (sorted list + bisect)",
                     "n": n, "operation": "insert",
                     "elapsed_ms": _bench("sorted-ins", lambda: _sorted_list_insert(keys), INSERT_REPEATS)})

        # Lookup
        sl = SkipList(seed=42)
        for k in keys:
            sl.insert(k, 1)
        sorted_list = sorted(keys)
        rows.append({"structure": "SkipList", "implementation": "Custom",
                     "n": n, "operation": "lookup",
                     "elapsed_ms": _bench("sl-look", lambda: _sl_lookup(sl, lookup_keys), LOOKUP_REPEATS)})
        rows.append({"structure": "SkipList", "implementation": "stdlib (sorted list + bisect)",
                     "n": n, "operation": "lookup",
                     "elapsed_ms": _bench("sorted-look", lambda: _sorted_list_lookup(sorted_list, lookup_keys), LOOKUP_REPEATS)})
    return rows


def _sl_insert(keys):
    sl = SkipList(seed=42)
    for k in keys:
        sl.insert(k, 1)


def _sorted_list_insert(keys):
    arr: list = []
    for k in keys:
        bisect.insort(arr, k)


def _sl_lookup(sl, lookup_keys):
    for k in lookup_keys:
        sl.search(k)


def _sorted_list_lookup(arr, lookup_keys):
    for k in lookup_keys:
        idx = bisect.bisect_left(arr, k)
        assert arr[idx] == k


def bench_trie(rng: random.Random) -> list[dict]:
    """Insert + prefix-lookup vs `sorted list + binary search`."""
    rows: list[dict] = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for n in SIZES:
        words = ["".join(rng.choices(alphabet, k=rng.randint(3, 12))) for _ in range(n)]
        prefixes = [w[: rng.randint(1, 3)] for w in rng.sample(words, k=min(n, 1000))]

        # Insert
        rows.append({"structure": "Trie", "implementation": "Custom",
                     "n": n, "operation": "insert",
                     "elapsed_ms": _bench("trie-ins", lambda: _trie_insert(words), INSERT_REPEATS)})
        rows.append({"structure": "Trie", "implementation": "stdlib (sorted list)",
                     "n": n, "operation": "insert",
                     "elapsed_ms": _bench("sl-ins", lambda: _list_sort_insert(words), INSERT_REPEATS)})

        # Prefix lookup
        t = Trie()
        for w in words:
            t.insert(w)
        sorted_words = sorted(set(words))
        rows.append({"structure": "Trie", "implementation": "Custom",
                     "n": n, "operation": "prefix_lookup",
                     "elapsed_ms": _bench("trie-prefix", lambda: _trie_prefix(t, prefixes), LOOKUP_REPEATS)})
        rows.append({"structure": "Trie", "implementation": "stdlib (sorted list)",
                     "n": n, "operation": "prefix_lookup",
                     "elapsed_ms": _bench("list-prefix", lambda: _list_prefix(sorted_words, prefixes), LOOKUP_REPEATS)})
    return rows


def _trie_insert(words):
    t = Trie()
    for w in words:
        t.insert(w)


def _list_sort_insert(words):
    sorted(set(words))


def _trie_prefix(t, prefixes):
    for p in prefixes:
        t.starts_with(p)


def _list_prefix(sorted_words, prefixes):
    for p in prefixes:
        # Linear/binary scan for any word starting with p
        idx = bisect.bisect_left(sorted_words, p)
        _ = idx < len(sorted_words) and sorted_words[idx].startswith(p)


def bench_bloom(rng: random.Random) -> list[dict]:
    """FPR vs target for different bit-array sizes."""
    rows: list[dict] = []
    for target_fpr in [0.001, 0.005, 0.01, 0.05, 0.10]:
        n = 10_000
        bf = BloomFilter.optimal(expected_n=n, target_fpr=target_fpr)
        inserted = {f"in_{i}" for i in range(n)}
        for x in inserted:
            bf.add(x)
        candidates = [f"out_{rng.random()}_{i}" for i in range(20_000)]
        fp = sum(1 for x in candidates if x in bf)
        rows.append({
            "structure": "BloomFilter",
            "target_fpr": target_fpr,
            "observed_fpr": fp / len(candidates),
            "m_bits": bf.m,
            "k_hashes": bf.k,
            "memory_bytes": bf.memory_bytes(),
            "n": n,
        })
    return rows


def bench_lru(rng: random.Random) -> list[dict]:
    """Cache hit rate as capacity scales for a hot/cold workload.

    60% of requests target a small "hot" set (500 keys); 40% are
    uniform random over a 50,000-key "cold" tail. Smaller caches catch
    most of the hot set; only large caches start absorbing cold tail.
    """
    rows: list[dict] = []
    hot_size = 500
    keyspace = 50_000
    n_requests = 100_000
    hot_keys = list(range(hot_size))
    cold_keys = list(range(hot_size, keyspace))
    request_stream = [
        rng.choice(hot_keys) if rng.random() < 0.6 else rng.choice(cold_keys)
        for _ in range(n_requests)
    ]

    for cap_pct in [0.5, 1, 2, 5, 10, 25, 50]:
        cap = max(1, int(keyspace * cap_pct / 100))
        c = LRUCache(capacity=cap)
        for k in request_stream:
            try:
                c.get(k)
            except KeyError:
                c.put(k, 1)
        rows.append({
            "structure": "LRUCache",
            "capacity_pct_of_keyspace": cap_pct,
            "capacity": cap,
            "n_requests": n_requests,
            "hits": c.stats["hits"],
            "misses": c.stats["misses"],
            "hit_rate": c.hit_rate,
        })
    return rows


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    print("HashMap vs dict...")
    hm_rows = bench_hashmap(rng)
    print("SkipList vs sorted list + bisect...")
    sl_rows = bench_skiplist(rng)
    print("Trie vs sorted list...")
    trie_rows = bench_trie(rng)

    timing_df = pd.DataFrame(hm_rows + sl_rows + trie_rows)
    out_dir = Path(__file__).parent
    timing_df.to_csv(out_dir / "benchmarks_timing.csv", index=False)
    print(f"  Wrote benchmarks_timing.csv ({len(timing_df)} rows)")

    print("BloomFilter FPR study...")
    bloom_df = pd.DataFrame(bench_bloom(rng))
    bloom_df.to_csv(out_dir / "benchmarks_bloom.csv", index=False)
    print(f"  Wrote benchmarks_bloom.csv ({len(bloom_df)} rows)")

    print("LRUCache capacity sweep...")
    lru_df = pd.DataFrame(bench_lru(rng))
    lru_df.to_csv(out_dir / "benchmarks_lru.csv", index=False)
    print(f"  Wrote benchmarks_lru.csv ({len(lru_df)} rows)")


if __name__ == "__main__":
    main()
