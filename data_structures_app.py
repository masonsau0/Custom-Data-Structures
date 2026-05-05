"""Streamlit dashboard for the Custom Data Structures library.

Six tabs: an overview, plus one per data structure with interactive
demos and the relevant benchmark plot. Reads pre-computed timing /
FPR / hit-rate data from the committed `benchmarks_*.csv` files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bloom_filter import BloomFilter
from hashmap import HashMap
from lru_cache import LRUCache
from skiplist import SkipList
from trie import Trie

ROOT = Path(__file__).parent

st.set_page_config(page_title="Custom Data Structures", layout="wide", page_icon="🧱")


@st.cache_data
def load_timing() -> pd.DataFrame:
    return pd.read_csv(ROOT / "benchmarks_timing.csv")


@st.cache_data
def load_bloom() -> pd.DataFrame:
    return pd.read_csv(ROOT / "benchmarks_bloom.csv")


@st.cache_data
def load_lru() -> pd.DataFrame:
    return pd.read_csv(ROOT / "benchmarks_lru.csv")


# --- Header -----------------------------------------------------------------

st.title("Custom Data Structures")
st.caption(
    "From-scratch pure-Python implementations of HashMap, Skip List, "
    "LRU Cache, Bloom Filter, and Trie, with a 15-test correctness "
    "suite and benchmarks against stdlib equivalents. Each tab below "
    "lets you exercise the structure interactively and inspect the "
    "performance curves."
)

with st.expander("How to use this app", expanded=False):
    st.markdown("""
**What this project shows.**
Five classic data structures implemented from scratch in pure Python,
each with its own correctness tests and a benchmark against the
nearest stdlib equivalent. The point isn't to beat stdlib (CPython's
`dict` is C; you can't outrun it from Python) — it's to demonstrate
the algorithm internals, prove the theoretical complexity holds in
practice, and ship a tested, documented library.

**Quick start (60 seconds).**
1. Click any structure tab on the left.
2. Try the interactive demo at the top of the tab (insert, lookup,
   delete, etc.) and watch the structure's internal state update.
3. Scroll to the bottom for the benchmark chart, which compares the
   custom implementation against its stdlib equivalent on log-log
   axes (slopes match expected complexity classes).

**The five structures at a glance.**
- **HashMap** -- open addressing with linear probing, tombstone
  deletes, and resize-on-load-factor. Average O(1) get/set.
- **Skip List** -- probabilistic balanced search structure. O(log n)
  search/insert/delete without the rotation logic of a self-balancing
  BST.
- **LRU Cache** -- O(1) get/put using a hash map for lookups plus a
  doubly-linked list for ordering. The classic interview design.
- **Bloom Filter** -- bit array + k hash functions. Space-efficient
  set membership with no false negatives and a tunable false-positive
  rate.
- **Trie** -- prefix tree for O(L) autocomplete and prefix-set
  membership.

**Try this.** Open the **Bloom Filter** tab and dial the target FPR
slider. The observed FPR (red bars) should track the target line
within ~10% on a 10,000-item insert with 20,000 negative checks.
That match is the whole point of getting the bit-array sizing math
right.
""")

# --- Tabs -------------------------------------------------------------------
tab_hm, tab_sl, tab_lru, tab_bf, tab_trie = st.tabs(
    ["HashMap", "Skip List", "LRU Cache", "Bloom Filter", "Trie"]
)


# --- HashMap tab ------------------------------------------------------------
with tab_hm:
    st.header("HashMap")
    st.markdown(
        "Open-addressing hash table with linear probing. Resize doubles "
        "capacity when load factor exceeds 0.7; tombstones mark deletes "
        "so probe sequences for live keys remain intact."
    )

    if "hm" not in st.session_state:
        st.session_state.hm = HashMap()

    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.text_input("Key", value="alpha", key="hm_key")
        v = st.text_input("Value", value="1", key="hm_val")
        if st.button("Set", key="hm_set"):
            st.session_state.hm[k] = v
    with c2:
        if st.button("Get", key="hm_get"):
            try:
                st.success(f"`{k}` -> `{st.session_state.hm[k]}`")
            except KeyError:
                st.error(f"`{k}` not in map")
        if st.button("Delete", key="hm_del"):
            try:
                del st.session_state.hm[k]
                st.success(f"deleted `{k}`")
            except KeyError:
                st.error(f"`{k}` not in map")
    with c3:
        if st.button("Reset", key="hm_reset"):
            st.session_state.hm = HashMap()

    hm = st.session_state.hm
    a, b, c = st.columns(3)
    a.metric("Size", len(hm))
    b.metric("Capacity", hm.capacity)
    c.metric("Load factor", f"{hm.load_factor:.2f}")

    if len(hm) > 0:
        st.caption("Current entries:")
        st.dataframe(
            pd.DataFrame(list(hm.items()), columns=["key", "value"]),
            width="stretch", hide_index=True,
        )

    st.subheader("Benchmark vs `dict` (stdlib)")
    timing = load_timing()
    hm_data = timing[timing["structure"] == "HashMap"]
    fig = px.line(
        hm_data, x="n", y="elapsed_ms", color="implementation",
        facet_col="operation", log_x=True, log_y=True, markers=True,
        labels={"n": "Items (n)", "elapsed_ms": "Time (ms, log)"},
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Both implementations are linear in `n` for total insert work "
        "(O(1) amortized per op). The pure-Python version pays a "
        "constant-factor overhead vs CPython's C-implemented `dict`; "
        "what matters is that the slope matches."
    )


# --- Skip List tab ----------------------------------------------------------
with tab_sl:
    st.header("Skip List")
    st.markdown(
        "Probabilistic ordered map. Each insert randomly picks a node "
        "level with P(level >= L) = p^(L-1); search walks top-down, "
        "stepping right at each level until the next pointer would "
        "overshoot, then dropping a level."
    )

    if "sl" not in st.session_state:
        st.session_state.sl = SkipList(seed=42)

    c1, c2 = st.columns(2)
    with c1:
        sl_key = st.number_input("Key (integer)", value=10, step=1, key="sl_key")
        sl_val = st.text_input("Value", value="ten", key="sl_val")
        if st.button("Insert", key="sl_ins"):
            st.session_state.sl.insert(int(sl_key), sl_val)
    with c2:
        if st.button("Delete", key="sl_del"):
            try:
                st.session_state.sl.delete(int(sl_key))
                st.success(f"deleted {sl_key}")
            except KeyError:
                st.error(f"{sl_key} not in list")
        if st.button("Reset", key="sl_reset"):
            st.session_state.sl = SkipList(seed=42)

    sl = st.session_state.sl
    a, b = st.columns(2)
    a.metric("Size", len(sl))
    b.metric("Top level", sl.level)

    if len(sl) > 0:
        st.caption("Sorted entries (skip list maintains order):")
        st.dataframe(
            pd.DataFrame(list(sl.items()), columns=["key", "value"]).head(50),
            width="stretch", hide_index=True,
        )
        height_dist = sl.height_distribution()
        if height_dist:
            st.caption("Node-height distribution (geometric-like):")
            hd_df = pd.DataFrame(
                sorted(height_dist.items()), columns=["level", "node_count"]
            )
            fig = px.bar(hd_df, x="level", y="node_count")
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Benchmark vs sorted list + `bisect` (stdlib)")
    sl_data = load_timing()[load_timing()["structure"] == "SkipList"]
    fig = px.line(
        sl_data, x="n", y="elapsed_ms", color="implementation",
        facet_col="operation", log_x=True, log_y=True, markers=True,
        labels={"n": "Items (n)", "elapsed_ms": "Time (ms, log)"},
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Skip list and bisect-on-sorted-list are both ~O(log n) lookup. "
        "Insert is where they differ: bisect.insort is O(n) due to list "
        "shift; the skip list is O(log n)."
    )


# --- LRU Cache tab ----------------------------------------------------------
with tab_lru:
    st.header("LRU Cache")
    st.markdown(
        "O(1) `get` and `put` via hash map for key lookup plus doubly-"
        "linked list for recency ordering. Most-recently-used at the "
        "head; eviction at the tail when capacity is exceeded."
    )

    if "lru_cap" not in st.session_state:
        st.session_state.lru_cap = 4
        st.session_state.lru = LRUCache(capacity=4)

    cap = st.slider("Capacity", min_value=2, max_value=10,
                    value=st.session_state.lru_cap, key="lru_slider")
    if cap != st.session_state.lru_cap:
        st.session_state.lru_cap = cap
        st.session_state.lru = LRUCache(capacity=cap)

    c1, c2 = st.columns(2)
    with c1:
        lru_key = st.text_input("Key", value="x", key="lru_key")
        lru_val = st.text_input("Value", value="42", key="lru_val")
        if st.button("Put", key="lru_put"):
            st.session_state.lru.put(lru_key, lru_val)
    with c2:
        if st.button("Get", key="lru_get"):
            try:
                v = st.session_state.lru.get(lru_key)
                st.success(f"`{lru_key}` -> `{v}`")
            except KeyError:
                st.warning(f"miss on `{lru_key}`")
        if st.button("Reset", key="lru_reset"):
            st.session_state.lru = LRUCache(capacity=st.session_state.lru_cap)

    lru = st.session_state.lru
    stats = lru.stats
    a, b, c, d = st.columns(4)
    a.metric("Size", stats["size"])
    b.metric("Hits", stats["hits"])
    c.metric("Misses", stats["misses"])
    d.metric("Hit rate", f"{stats['hit_rate']:.0%}")

    keys_in_order = lru.keys_in_order()
    if keys_in_order:
        st.caption("Order (most recent first → least recent last):")
        st.code(" → ".join(str(k) for k in keys_in_order))

    st.subheader("Hit-rate vs capacity (hot / cold workload)")
    lru_df = load_lru()
    fig = px.line(
        lru_df, x="capacity_pct_of_keyspace", y="hit_rate", markers=True,
        labels={
            "capacity_pct_of_keyspace": "Cache capacity (% of keyspace)",
            "hit_rate": "Hit rate",
        },
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Workload: 60 % requests hit a 500-key hot set, 40 % uniform "
        "random over 49,500 cold keys. Tiny caches still catch most of "
        "the hot set; saturation kicks in once capacity exceeds the "
        "hot working set."
    )


# --- Bloom Filter tab -------------------------------------------------------
with tab_bf:
    st.header("Bloom Filter")
    st.markdown(
        "Bit array of size m, k hash functions. `add(x)` sets k bits; "
        "`contains(x)` returns True iff all k bits are set. No false "
        "negatives; tunable false-positive rate."
    )

    target_fpr = st.slider("Target FPR", 0.001, 0.20, 0.01, step=0.001, key="bf_fpr")
    expected_n = st.slider("Expected items", 100, 50_000, 1000, step=100, key="bf_n")

    bf = BloomFilter.optimal(expected_n=expected_n, target_fpr=target_fpr)
    a, b, c, d = st.columns(4)
    a.metric("m (bits)", f"{bf.m:,}")
    b.metric("k (hashes)", bf.k)
    c.metric("Memory", f"{bf.memory_bytes():,} B")
    d.metric("Bits / item", f"{bf.m / expected_n:.1f}")

    if st.button("Add 'hello'", key="bf_add"):
        bf.add("hello")
    if st.button("Check 'hello'", key="bf_check"):
        st.success(f"in filter: {'hello' in bf}")

    st.subheader("Observed vs target FPR (10,000-item study)")
    bloom_df = load_bloom()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bloom_df["target_fpr"], y=bloom_df["observed_fpr"],
        name="Observed", marker_color="firebrick",
    ))
    fig.add_trace(go.Scatter(
        x=bloom_df["target_fpr"], y=bloom_df["target_fpr"],
        name="Target (perfect calibration)", mode="lines+markers",
        line=dict(color="grey", dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Target FPR", yaxis_title="FPR",
        height=380, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Observed FPR tracks the target across two orders of magnitude "
        "(0.001 to 0.10), confirming the Mitzenmacher-Kirsch double-"
        "hashing trick yields the same FPR as k truly independent "
        "hash functions in this regime."
    )


# --- Trie tab ---------------------------------------------------------------
with tab_trie:
    st.header("Trie")
    st.markdown(
        "Prefix tree. Each path from root to a marked node spells a "
        "stored word. Insert and lookup are O(L) in the key length, "
        "independent of dictionary size. Used for autocomplete, IP "
        "routing tables, and spell-check."
    )

    if "trie" not in st.session_state:
        st.session_state.trie = Trie()
        for w in [
            "cat", "car", "card", "cart", "carbon", "care", "careful",
            "dog", "doge", "duck", "duct",
            "fig", "figure", "fish", "fit",
        ]:
            st.session_state.trie.insert(w)

    c1, c2 = st.columns(2)
    with c1:
        word = st.text_input("Word to insert / search", value="cat", key="trie_word")
        if st.button("Insert", key="trie_ins"):
            st.session_state.trie.insert(word)
        if st.button("Delete", key="trie_del"):
            st.success(f"deleted: {st.session_state.trie.delete(word)}")
        if st.button("Reset", key="trie_reset"):
            st.session_state.trie = Trie()
    with c2:
        prefix = st.text_input("Prefix for autocomplete", value="car", key="trie_prefix")
        if st.button("Autocomplete", key="trie_auto"):
            matches = st.session_state.trie.words_with_prefix(prefix, limit=20)
            if matches:
                st.success(f"{len(matches)} matches: " + ", ".join(matches))
            else:
                st.warning("no matches")

    st.metric("Words in trie", len(st.session_state.trie))

    st.subheader("Benchmark vs sorted list + binary search")
    trie_data = load_timing()[load_timing()["structure"] == "Trie"]
    fig = px.line(
        trie_data, x="n", y="elapsed_ms", color="implementation",
        facet_col="operation", log_x=True, log_y=True, markers=True,
        labels={"n": "Items (n)", "elapsed_ms": "Time (ms, log)"},
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Trie prefix lookup is O(L); sorted-list bisect is O(log n + L). "
        "For longer dictionaries the trie wins because the lookup cost "
        "depends only on the prefix length, not the size of the "
        "dictionary."
    )
