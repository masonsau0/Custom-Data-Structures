# Custom Data Structures

**[Live demo](LIVE_DEMO_URL_PLACEHOLDER)**: runs in the browser, no install required.

Pure-Python from-scratch implementations of five classic data
structures — **HashMap**, **Skip List**, **LRU Cache**, **Bloom
Filter**, and **Trie** — with a 15-test correctness suite, a
benchmarking harness comparing each against its stdlib equivalent, and
an interactive Streamlit dashboard for exercising the structures live
and inspecting the performance curves.

The point isn't to beat CPython's C-implemented `dict` from Python
(you can't); it's to demonstrate the algorithm internals, prove the
theoretical complexity holds in practice, and ship a tested,
documented library.

## What's actually demonstrated

| Concept | Where |
|---|---|
| **Open addressing with linear probing**, tombstone deletes, resize-on-load-factor | `hashmap.py` |
| **Probabilistic balancing** with geometric level distribution | `skiplist.py` |
| **Doubly-linked list + hash map composition** for O(1) get/put | `lru_cache.py` |
| **Bit array with k hash positions** via the Kirsch-Mitzenmacher double-hashing trick | `bloom_filter.py` |
| **Prefix tree** with O(L) operations, autocomplete, and prune-on-delete | `trie.py` |
| **Correctness tests** without a framework dependency (`python tests.py`, 15 assertions) | `tests.py` |
| **Performance benchmarks** vs stdlib equivalents on log-log axes | `benchmarks.py`, dashboard |

## The five structures

### HashMap (`hashmap.py`)
Open-addressing hash table; capacity is power-of-two so `hash & (cap - 1)`
replaces modulo. Linear probing with tombstones preserves probe sequences
through deletes. Resize doubles capacity when load factor exceeds 0.7.

```python
hm = HashMap()
hm["alpha"] = 1
hm["alpha"]            # -> 1
del hm["alpha"]
"alpha" in hm          # -> False
```

### Skip List (`skiplist.py`)
Probabilistic ordered map. Each node randomly gets a level
1..MAX_LEVEL with `P(level >= L) = p^(L-1)` (default p = 0.5). Search
walks top-down, stepping right at each level until the next pointer
would overshoot, then dropping a level. O(log n) expected.

```python
sl = SkipList(seed=42)
sl.insert(3, "three"); sl.insert(1, "one"); sl.insert(4, "four")
list(sl)               # -> [1, 3, 4]  (sorted)
sl.search(3)           # -> "three"
```

### LRU Cache (`lru_cache.py`)
The classic interview design: hash map for O(1) lookup, doubly-linked
list for O(1) move-to-front and O(1) tail eviction. Tracks hit/miss
counters so the dashboard can plot hit rate as capacity scales.

```python
c = LRUCache(capacity=2)
c.put("a", 1); c.put("b", 2)
c.get("a")             # -> 1, "a" is now MRU
c.put("c", 3)          # evicts "b"
```

### Bloom Filter (`bloom_filter.py`)
Bit array of size m, k hash positions per item via the
Kirsch-Mitzenmacher double-hashing technique
`pos_i = (h1 + i*h2) mod m`. `BloomFilter.optimal(expected_n,
target_fpr)` auto-sizes m and k.

```python
bf = BloomFilter.optimal(expected_n=10_000, target_fpr=0.01)
bf.add("alpha"); bf.add("beta")
"alpha" in bf          # -> True (definite)
"gamma" in bf          # -> False or rarely True (FPR ~1%)
```

### Trie (`trie.py`)
Prefix tree for O(L) insert / search / autocomplete, where L is the
length of the key. `delete()` prunes nodes that no longer participate
in any stored word.

```python
t = Trie()
for w in ["car", "card", "care", "careful", "carpet"]: t.insert(w)
t.words_with_prefix("car")    # -> ["car", "card", "care", "careful", "carpet"]
"car" in t                     # -> True
```

## Run it

### Tests

```bash
python tests.py
```

Runs 15 correctness tests across all five structures (single file, no
framework dependencies — `pytest` is overkill here).

### Benchmarks

```bash
pip install -r requirements.txt
python benchmarks.py
```

Times insert and lookup against stdlib equivalents at sizes
100 → 50,000, plus a Bloom-filter FPR study and an LRU hit-rate sweep.
Writes three CSVs the dashboard reads.

### Streamlit dashboard

```bash
streamlit run data_structures_app.py
```

Five tabs (one per structure). Each has an interactive demo at the
top and the relevant benchmark plot at the bottom.

## Repository layout

```
.
├── hashmap.py                 ← Open-addressing hash table
├── skiplist.py                ← Probabilistic balanced ordered map
├── lru_cache.py               ← O(1) LRU cache (hashmap + doubly-linked list)
├── bloom_filter.py            ← Probabilistic membership with tunable FPR
├── trie.py                    ← Prefix tree
├── tests.py                   ← 15 correctness tests, no framework dependency
├── benchmarks.py              ← Timing harness vs stdlib equivalents
├── benchmarks_timing.csv      ← committed timing results (HashMap, SkipList, Trie)
├── benchmarks_bloom.csv       ← committed FPR study results
├── benchmarks_lru.csv         ← committed LRU capacity-sweep results
├── data_structures_app.py     ← Streamlit dashboard (5 tabs)
├── requirements.txt
├── LICENSE
└── README.md
```

## Stack

**Pure Python** for the structures (stdlib only — `random`, `hashlib`,
`bisect`, `math`) · **pandas + Plotly** for benchmark analysis ·
**Streamlit** for the interactive dashboard
