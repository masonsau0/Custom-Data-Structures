"""Streamlit dashboard for the Custom Data Structures library.

Five tabs, one per data structure, each with an interactive demo,
metrics that surface internal state, and a benchmark plot. Reads
pre-computed timing / FPR / hit-rate data from the committed
`benchmarks_*.csv` files.
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

with st.expander("How to use this app (read me first)", expanded=False):
    st.markdown("""
### What you're looking at

Five common ways of storing data, each built from scratch in pure
Python and given its own tab. Every tab has the same three sections:

1. **A short explainer**: what this tool is and what it's good for.
2. **An interactive demo**: type in the boxes, click the buttons,
   watch the numbers update.
3. **A speed chart**: how this version compares to the one already
   built into Python.

This project is not trying to be faster than Python's built-in
versions. Python's built-ins are written in C and they will always
win on raw speed. The point here is to show *how the algorithms
work*, prove they behave the way the textbook says they should, and
ship something that's tested and documented.

### How to use a tab in 30 seconds

1. Open any tab.
2. Click the "What this tab does" panel at the top for a plain-
   English description of what every button and field controls.
3. Try the demo: type a key, click Set / Insert / Put. The numbers
   below the buttons update live.
4. Scroll to the bottom for the speed chart.

### The five structures, one line each

| Structure | What it does | Real-world analogy |
|---|---|---|
| **HashMap** | Stores label-and-value pairs and finds them again instantly. | A phone book where every name jumps straight to the right page. |
| **Skip List** | Same as a HashMap, but keeps everything sorted. | A library where books stay in alphabetical order even as you add new ones. |
| **LRU Cache** | A small box that holds N items; the one you ignored longest gets thrown out when a new one arrives. | Your wallet has space for 6 loyalty cards; the oldest unused one goes when you add a new one. |
| **Bloom Filter** | Answers "have we seen this before?" using almost no memory. Sometimes says "maybe" when the real answer is no, but never says "no" when the answer is yes. | A bouncer's "have I seen this face today?" gut check, before they actually look at the list. |
| **Trie** | A word storage tree built so autocomplete is instant, no matter how big the dictionary. | The way your phone suggests words as you type. |

### Try this tour

1. **HashMap**: click Set a few times with different keys. Watch
   "Capacity" jump from 8 to 16 to 32 as you add more items (the
   storage grows automatically when it gets too full).
2. **Skip List**: insert 20 keys, look at the height histogram.
   About half the bars should be at level 1, a quarter at level 2,
   an eighth at level 3, and so on. That random shape is what makes
   skip lists fast.
3. **LRU Cache**: set capacity to 3, click Put for keys a, b, c,
   click Get for "a", then Put for "d". The "Order" line will show
   d, a, c. ("b" got thrown out because "a" was used most recently.)
4. **Bloom Filter**: drag the "Target FPR" slider down to 0.001 and
   watch memory go up. The chart at the bottom shows the filter
   really does hit the target across a wide range.
5. **Trie**: type "car" into the autocomplete box and click. You'll
   instantly get back five pre-loaded "car..." words, no matter how
   many other words are in storage.
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

    with st.expander("What this tab does (plain English)", expanded=False):
        st.markdown("""
**What it is.** A HashMap stores label-and-value pairs and finds
them instantly. Picture a giant array of empty slots. When you save
the value `1` under the label `"alpha"`, the HashMap turns
`"alpha"` into a number, uses that number as the slot position, and
drops `1` in there. To look it up later, it does the same math and
goes straight to that slot. No searching.

**What if two labels land in the same slot?** That's called a
*collision*. This implementation handles it by walking forward to
the next empty slot and using that one instead. Simple and fast.

**Why it leaves "tombstones" when you delete.** Imagine slot 5 is
holding `"alpha"`, and slot 6 is holding `"beta"` because slot 5
was full when we added it. If you delete `"alpha"` and just clear
slot 5, the next person looking for `"beta"` would see an empty
slot at 5 and incorrectly conclude `"beta"` isn't there. So delete
leaves a "tombstone" marker. Lookups skip past tombstones; new
inserts can overwrite them.

**Why it grows automatically.** When slots get more than 70% full,
collisions pile up and lookups slow down. So the HashMap doubles
its storage as soon as it crosses that threshold, and re-files
every existing entry into the new, bigger array.

**Controls.**
- **Key**: the label you want to look something up by (any text).
- **Value**: what you want to store under that label (any text).
- **Set**: save this value under this label. Overwrites if the
  label already exists.
- **Get**: fetch the value for the current label. Shows an error
  if it isn't there.
- **Delete**: remove the label.
- **Reset**: empty everything and start fresh.

**The numbers below the buttons.**
- **Size**: how many label-and-value pairs are stored.
- **Capacity**: how many slots the storage has. Starts small and
  doubles when it gets too full.
- **Load factor**: how full the storage is, as a fraction. When
  this goes above 0.70, the next Set triggers a doubling.

**The chart at the bottom.** Two lines: this HashMap, and Python's
built-in `dict`. The X-axis is how many items we put in; the Y-axis
is how long it took. Both axes use a logarithmic scale (each tick
is 10x the one before), so a straight line means the time per item
stays constant as the data grows. That's exactly what you'd hope
for. Python's `dict` is faster overall (it's written in C), but
the *shape* of the lines matches.
""")

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

    with st.expander("What this tab does (plain English)", expanded=False):
        st.markdown("""
**What it is.** A storage system that keeps your items in sorted
order and lets you find any of them quickly, even when there are
millions.

**The trick.** Imagine a sorted list of names. To find "Smith" you
have to walk from the start, name by name, which is slow. Now
imagine you build a *second*, shorter list above it that only
contains every other name: A, C, E, G... To find Smith you skip
along the express list until you'd overshoot, then drop into the
detailed list to finish. Add a third list above that with every
fourth name, and a fourth above that, and you've got something
that finds anything in just a handful of jumps.

A skip list builds those express lanes randomly. Each new entry
flips a coin: about half land at level 1 only, about a quarter make
it to level 2, an eighth to level 3, and so on. The randomness
keeps the structure roughly balanced without ever needing to
manually rearrange it.

**Why anyone cares.** It does the same job as a fancy "balanced
search tree" but is much easier to write correctly. Real systems
use it: Redis stores its sorted sets this way.

**Controls.**
- **Key (integer)**: a number to insert or delete.
- **Value**: anything you want to attach to that number.
- **Insert**: add or overwrite.
- **Delete**: remove. Errors if it isn't there.
- **Reset**: clear and reseed (so demo runs are reproducible).

**The numbers below the buttons.**
- **Size**: total entries stored.
- **Top level**: the height of the tallest entry currently in the
  list. As you add more entries, this should grow slowly: 100
  entries gives about 7, 1,000 gives about 10, 10,000 gives about
  13.

**The sorted entries table.** Whatever order you inserted in, this
table always shows them in order. That's the whole point of a skip
list (a HashMap can't do this).

**The height histogram.** A bar chart of how many entries exist at
each level. The bars should roughly halve as you go up: half the
entries at level 1, a quarter at level 2, and so on. That random
shape is what makes the structure fast.

**The speed chart.** Insert and lookup times vs Python's standard
"sorted list" approach. Lookup is similar (both fast). Insert is
where the skip list wins: adding to a sorted Python list forces it
to shift every element after the insertion point, so it gets
slower as the list gets bigger; the skip list doesn't have to.
""")

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

    with st.expander("What this tab does (plain English)", expanded=False):
        st.markdown("""
**What it is.** A small storage box with a fixed capacity. When you
try to put one more thing in than it can hold, the *least recently
used* item gets thrown out to make room. Used everywhere a system
needs to remember "recent" things without remembering everything:
your browser's page cache, the operating system's file cache, a
website's "you've already seen this" memory.

**Why "least recently used"?** Because in practice, things you
touched recently are likely to be touched again soon, and things
you haven't touched in a while are probably fine to discard.

**The trick.** We need two operations to be fast: looking up an
item by its label, and "this is the most recent thing now, push the
oldest out." Neither a list nor a hash map can do both well alone,
but combining them gives both at once: the hash map finds the item
instantly, and a chain of items ordered by recency lets us re-order
in one quick step.

**Controls.**
- **Capacity slider**: how many items the cache can hold. Changing
  this clears the cache.
- **Key**: the label.
- **Value**: the data.
- **Put**: save data under a label. If the cache is full, the
  oldest unused item is thrown out first.
- **Get**: look up a label. On hit, that label becomes "most
  recently used" (so it's safer from eviction). On miss, the miss
  counter goes up.
- **Reset**: empty the cache, keep the capacity setting.

**The numbers below the buttons.**
- **Size**: how many items are currently stored.
- **Hits**: how many times Get found what you asked for.
- **Misses**: how many times Get came up empty.
- **Hit rate**: hits divided by total Gets. The single number that
  tells you whether the cache is doing its job.

**The "Order" line.** Keys printed left to right by recency:
leftmost was just used, rightmost is the next to be evicted.

**The chart at the bottom.** A pre-computed experiment. We sent
100,000 lookups against the cache (60% hitting a small "popular"
set of 500 keys, 40% spread across 49,500 less-popular keys) and
varied the cache size from 0.5% of total keys up to 50%. Tiny
caches already grab most of the popular stuff, so the curve climbs
fast at first, then flattens because the extra space mostly goes to
keys that won't be reused soon.
""")

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
        st.caption("Order (most recent first, least recent last):")
        st.code(" -> ".join(str(k) for k in keys_in_order))

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

    with st.expander("What this tab does (plain English)", expanded=False):
        st.markdown("""
**What it is.** A super-compact way to answer "have I seen this
before?" The trade-off: it sometimes says "maybe yes" when the real
answer is no, but it *never* says "no" when the real answer is yes.
Used as a fast filter in front of a slower system. Example: "is
this URL on the malware list?" If the bloom filter says no, you can
safely skip looking it up. If it says maybe, you do the real check.

**How it works (intuition).** Reserve a row of `m` empty slots,
all zero to start. To "add" an item, run it through `k` different
hash functions to get `k` slot positions, and flip those slots from
0 to 1. To check if an item was added, hash it the same `k` ways:
if *any* of those slots is still 0, the item was definitely never
added. If all `k` are 1, the item was probably added.

**Why "probably"?** Two unrelated items might happen to flip the
same slots. That's a *false positive*, and the chance of one is
called the *FPR* (false-positive rate). You set the target FPR,
and the filter computes how many slots `m` and how many hashes `k`
it needs to hit that target.

**A speed-up trick we use.** Computing `k` separate hash functions
is expensive. There's a well-known technique (the
"Kirsch-Mitzenmacher trick") where two strong hashes can stand in
for any number of independent ones, with the same FPR in practice.
We use that.

**Controls.**
- **Target FPR slider**: how often you're willing to get a "maybe"
  when the real answer is no. Lower target = more memory used.
- **Expected items slider**: how many items you plan to insert.
  The filter is sized for this number; insert way more than this
  and the actual FPR drifts above target.
- **Add 'hello' / Check 'hello'**: a tiny demo. Click Add, then
  Check, and you'll see "in filter: True". Without Add first,
  Check usually returns False.

**The numbers below the sliders.**
- **m (bits)**: how many slots the filter has, computed from the
  sliders.
- **k (hashes)**: how many slots each item touches.
- **Memory**: total size of the filter in bytes. Notice how small
  this is even for tens of thousands of items.
- **Bits / item**: m divided by expected items. Useful intuition:
  hitting a 1% FPR takes about 9.6 bits per item, no matter how
  many items you have.

**The chart at the bottom.** A pre-computed experiment. At five
different target FPRs (from 0.1% up to 10%), we add 10,000 items,
then test the filter against 20,000 items it never saw and count
how often it incorrectly says "maybe yes". The bars (observed)
should sit very close to the dashed grey line (target). The match
across that whole range is the proof the math is right.
""")

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

    with st.expander("What this tab does (plain English)", expanded=False):
        st.markdown("""
**What it is.** A tree-shaped storage system optimized for words
that share starts. Every branch is labeled with a single letter; a
word is "stored" by tracing it letter-by-letter from the root and
marking the last letter as "this is the end of a word".

**Why this shape.** Words like "car", "card", "care", and "careful"
all start with `c-a-r`, so they share the same path through the
tree until they diverge. That's a memory win, but more importantly
it makes "give me every word starting with X" instant: walk to the
node for X, then list everything below it.

**The unusual property.** The time it takes to look something up
depends on the *length of the word*, not the number of words
stored. A trie with a million words finds "elephant" in the same
time as a trie with a hundred words.

**Where it's used.**
- *Autocomplete*: phone keyboards, search bars.
- *Spell check*: suggesting "did you mean X?" candidates.
- *Network routers*: matching IP addresses to destinations.

**Pre-loaded demo data.** The trie starts with 15 short words
("cat", "car", "card", "cart", "carbon", "care", "careful", "dog",
"doge", "duck", "duct", "fig", "figure", "fish", "fit") so the
autocomplete button has something to return out of the box.

**Controls.**
- **Word to insert / search**: the word you want to add or remove.
- **Insert**: add the word.
- **Delete**: remove the word. Also cleans up any dead branches the
  deletion left behind.
- **Reset**: empty the tree completely (no pre-load).
- **Prefix for autocomplete**: the start of a word.
- **Autocomplete**: returns up to 20 stored words that begin with
  the prefix.

**The number below the buttons.**
- **Words in trie**: total complete words stored.

**The chart at the bottom.** Two lines: this trie vs Python's
"sorted list" approach. Insert and "find every word starting with
X" times, plotted as the dictionary grows. The trie's curve stays
relatively flat because the cost only depends on the prefix length,
not the dictionary size. The sorted-list curve grows because it has
to do more work as the dictionary gets bigger.
""")

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
