# Deadline-Aware, Self-Certifying IVF Search — Plan

## Context

The DB today does exact top-k by linear scan ([collection.cpp:82-116](src/collection.cpp#L82-L116)). It sits at one fixed
point on the latency/recall curve — perfect recall, O(n·d) latency — with no way to move. `IVFIndex`
([ivf_index.cpp](src/ivf_index.cpp)) is a stub: it re-runs the same brute-force scan and ignores `nprobe`.

We're building an IVF index that does three things a normal IVF doesn't:

1. **Anytime** — takes a latency deadline and returns the best answer found before the clock runs out.
2. **Self-certifying** — proves via a triangle-inequality bound when its approximate answer is in fact the
   exact top-k, and reports per query whether the result is certified exact or best-effort.
3. **Per-query adaptive** — easy queries stop after one cluster, ambiguous queries probe more, without a
   fixed global `nprobe`.

Outcome: `search(query, k, mode)` where mode is a latency contract (`low_latency` / `balanced` /
`high_recall` / `exact`), and every response carries a status saying how much of the data was searched and
whether the result is provably correct. This is a system feature, not a benchmark report — no
algorithm-comparison harness is in scope.

### The one piece of real math

Work in normalized space, where cosine similarity is a dot product and `θ(a,b) = acos(dot(a,b))`.

For cluster `c` with centroid `μ` and **angular radius** `R_c = max over members x of acos(dot(μ, x))`, the
best possible similarity any member of `c` can have with query `q` is `cos(max(0, θ(q, μ) − R_c))`.

**Compute it without ever calling `acos`.** Expanding the angle-difference identity, with
`cq = dot(q, μ)`, `cosR_c = min over members x of dot(μ, x)` and `sinR_c = sqrt(1 − cosR_c²)`:

```
U_c(q) = (cq >= cosR_c) ? 1.0                                   // query sits inside the cone
                        : cq * cosR_c + sqrt(1 − cq²) * sinR_c  // == cos(θq − R_c)
```

This is the same bound, but `acos` is ill-conditioned exactly where clusters are tight (its derivative
blows up as `dot → 1`), and a tight cluster is the common case. Storing `cosR_c`/`sinR_c` keeps everything
in cosine space, removes an `acos` at build and a `cos` per cluster per query, and is better conditioned.
Clamp `cq` and `cosR_c` to `[-1, 1]` and the `sqrt` arguments to `>= 0` before use.

Scan clusters in **decreasing `U_c`** order. After each cluster, if the current k-th best score beats the
`U_c` of the next unscanned cluster, no unscanned vector can beat it — **stop, and the answer is provably
exact.** Because clusters are ordered by `U_c`, that check is O(1).

**The certificate must be conservative.** Floating-point error in the centroid accumulation, the
normalization and the dot products means `U_c` is only approximately the true supremum. The invariant is:

> Numerical uncertainty may cost us early stopping. It must never produce a false certificate.

That fixes the direction of every fudge factor, and both of these are easy to get backwards:

- **Inflate the bound at build**, never at query: store `cosR_c` slack-adjusted downward
  (`cosR_c = clamp(minDot − 1e-9, -1, 1)`), which widens the cone and makes `U_c` a provable over-estimate.
- **Stop on strict `>` with the margin on the bound's side**: `kthScore > U_next + kBoundEpsilon`
  (`kBoundEpsilon = 1e-9`). Writing `U_next − epsilon` would make the loop stop *sooner* and is the unsafe
  direction — the failure it invites is a silent one.

The strict `>` is also what makes the certificate correct under **ties** (see the total order in Phase 1):
`kth >= U_next` proves no unscanned vector *exceeds* the k-th score, but an unscanned vector could *equal*
it and still outrank it on the id tie-break. `>` rules that out, so one comparison buys both properties.
Duplicate vectors in seeded test data hit this case directly; it is not hypothetical.

**A certificate needs a full heap.** `kthScore` only exists once the heap holds `k` candidates. While
fewer than `k` vectors have been scanned there is no bound to compare against and the loop must keep
scanning, whatever `U_next` says.

Idea 3 largely *falls out* of idea 2: the bound-driven loop naturally scans one cluster for an easy
query and many for a boundary query. The explicit adaptive knobs in Phase 3 exist for the case where the
bound is loose (high dimensions) and never fires.

---

## Phase 1 — Make IVF real

Turn `IVFIndex` from a stateless stub into a built index, and cache it on `Collection`.

**[include/vectordb/ivf_index.hpp](include/vectordb/ivf_index.hpp) / [src/ivf_index.cpp](src/ivf_index.cpp)**

- `IVFIndex` gains state: `std::vector<Cluster>` where `Cluster { std::vector<double> centroid; std::vector<std::size_t> members; }`,
  plus flat `std::vector<std::string> ids_` and `std::vector<std::vector<double>> vectors_` holding
  **L2-normalized copies** of every record (records themselves stay untouched — the CLI inserts unnormalized
  vectors, and the bound math needs unit vectors).
- `void build(const std::unordered_map<std::string, VectorRecord>&, std::size_t nlist, unsigned seed = 42)`.
  Default `nlist = ceil(sqrt(n))`, clamped to `[1, n]`.
- **Spherical k-means** (Lloyd's, re-normalizing centroids after each update — the metric is cosine, not L2):
  seeded `std::mt19937` for determinism, fixed iteration cap (~20) plus an early exit when no assignment
  changes. Re-seed any cluster that goes empty from the point farthest from its centroid.
- Skip/guard zero-norm vectors — a zero vector has no direction and cannot be normalized. Reuse the same
  clamping style already in [similarity.cpp](src/similarity.cpp). **Note the deliberate divergence:** the
  index excludes zero records, while `cosineSimilarity` ([similarity.cpp:29-31](src/similarity.cpp#L29-L31))
  *throws* on them, so a collection holding one makes `searchExact` throw and IVF succeed. Document it in the
  `build` comment; parity tests use collections without zero records.
- Keep a plain `search(query, k, nprobe)` for now: rank centroids by similarity, scan the top `nprobe`
  inverted lists, keep top-k. Fixed `nprobe`, no bounds yet.

**[include/vectordb/collection.hpp](include/vectordb/collection.hpp) / [src/collection.cpp](src/collection.cpp)**

- Add `mutable std::shared_ptr<const IVFIndex> index_;` and `mutable bool indexDirty_ = true;`.
  **Use `shared_ptr`, not `unique_ptr`** — `Collection` is copied and returned by value in existing tests
  ([test_synthetic.cpp:20-29](tests/test_synthetic.cpp#L20-L29), `loadFromFile`), and a `unique_ptr` member
  would silently delete the copy constructor and break the build.
- `insert()` / `remove()` set `indexDirty_ = true`.
- `searchIVF` lazily builds on first use.

**One shared total order over results — the single behavioural change to `searchExact`.**
`searchExact` today sorts on `a.score > b.score` with `std::sort` (not stable) over a `std::unordered_map`
whose iteration order is unspecified ([collection.cpp:104-113](src/collection.cpp#L104-L113)), so its tie
ordering is already arbitrary and can shift with insertion order or a rehash. The IVF heap comparator
([ivf_index.cpp:302-308](src/ivf_index.cpp#L302-L308)) already tie-breaks on id. Make both use:

```
primary:   score descending
secondary: id ascending
```

Without this, no result-parity assertion between the two paths is reliable, ties or not. This is the only
edit to `searchExact` in the whole plan, it only reorders vectors that were previously tied, and the
existing 76 tests must keep passing unchanged.

**What "matches `searchExact`" means.** Not byte-identity of scores. `searchExact` computes
`dot/(‖a‖‖b‖)` on raw values; IVF dots pre-normalized copies. Different operation order, so the two differ
in the last ulp — by construction, independent of any bound. The invariant is:

> **identical ids in identical order** under the total order above, with **scores equal to within `1e-12`**.

**Verify:** new `tests/test_ivf.cpp` — build on `synthetic::VectorGenerator::generateClusters` data
([synthetic.hpp:55](include/vectordb/synthetic.hpp#L55)); with `nprobe = nlist`, IVF must match
`searchExact` under that invariant; every vector belongs to exactly one cluster; centroids are unit-norm;
rebuild after `insert` picks up the new record.

---

## Phase 2 — Provable exactness

**Adds to `IVFIndex`:**

- Two new `Cluster` fields, `double cosRadius, sinRadius`, computed at build from the *minimum*
  `dot(μ, x)` over members: apply the `-1e-9` inflation to `cosRadius` once, there, then derive
  `sinRadius = sqrt(max(0, 1 - cosRadius²))` from the inflated value. No `acos` anywhere.
- `double upperBound(double queryCentroidDot, const Cluster& c) const` implementing `U_c` above — it takes
  the dot product rather than the query, since the caller already has it from ranking.
- Rewrite the search loop: compute `U_c` for all clusters, sort descending (tie-break on cluster index, so
  the scan order is deterministic), scan in that order maintaining a bounded min-heap of the top k, and
  after each cluster stop when `heap.size() == k && kthScore > U_next + kBoundEpsilon`.

**New public types** in [ivf_index.hpp](include/vectordb/ivf_index.hpp):

```cpp
enum class SearchStatus { Certified, Exhausted, DeadlineHit, BudgetHit };

struct SearchStats {
    SearchStatus status;
    std::size_t clustersScanned, totalClusters, vectorsScanned, totalVectors;
    double elapsedMs;
};

struct SearchResponse {
    std::vector<SearchResult> results;   // reuses SearchResult from collection.hpp
    SearchStats stats;
};
```

`Certified` = the bound proved it early. `Exhausted` = scanned everything (trivially exact). Both mean the
result is exact; they differ only in how that was established. The status is deliberately **not** named
`Exact`, because `SearchMode::Exact` (Phase 3) most often terminates as `Exhausted` — one word for two
different concepts is a bug waiting to be written into a log line.

**Verify:** on well-separated clusters, a query near a cluster center returns `Certified` with
`clustersScanned == 1`; whenever status is `Certified` or `Exhausted`, results match `searchExact` under the
Phase 1 invariant (same ids, same order, scores within `1e-12`) — this is the property worth testing over
many random queries, since a violated bound is a silent correctness bug. Include a fixture with **duplicate
vectors**, which is where the tie-break and the strict `>` earn their keep.

---

## Phase 3 — Deadline and modes

**`SearchParams`** in [ivf_index.hpp](include/vectordb/ivf_index.hpp):

```cpp
struct SearchParams {
    std::chrono::microseconds deadline{0};  // 0 = no deadline
    std::size_t minProbes = 1;
    std::size_t maxProbes = 0;              // 0 = unlimited
};
```

- Search loop takes a `SearchParams`, snapshots `steady_clock::now()` at entry, and checks the deadline at
  each cluster boundary (and every ~256 vectors inside a large cluster, so one huge cluster can't blow the
  budget). Returns best-so-far with `DeadlineHit`.
- **Ordering is what makes this safe to interrupt**: because clusters are visited best-bound-first, the
  answer only improves monotonically, so cutting the scan short degrades recall gracefully instead of
  returning garbage.
- **Adaptive probe floor** for when the bound never fires: derive `minProbes` from the centroid similarity
  gap `U_1 − U_2`. A large gap means the query sits unambiguously in one cluster (floor stays 1); a small gap
  means it's on a boundary (raise the floor). Keep this a single small documented function so the heuristic
  is easy to tune later.
- **Mode presets** — a `SearchMode` enum (`LowLatency`, `Balanced`, `HighRecall`, `Exact`) and one
  `SearchParams paramsForMode(SearchMode, std::size_t n)` factory. These presets are the entire user-facing
  tuning surface, and they split into two contracts:

  ```
  SearchMode::Exact              guaranteed exact. No deadline, no probe cap.
                                 Terminates as Certified if the bound fires,
                                 otherwise scans every cluster and returns Exhausted.
                                 Never returns DeadlineHit or BudgetHit.

  LowLatency / Balanced /        best-effort. May return Certified or Exhausted when
  HighRecall                     the data makes it free, otherwise DeadlineHit / BudgetHit.
  ```

  So `Exact` is a guarantee about the *answer*, not merely an absence of limits: the early exit is an
  optimization within it, never the thing that defines it.

**Verify:** an absurdly tight deadline (e.g. 1µs) returns `DeadlineHit` with a non-empty best-so-far and
never throws; a generous deadline on the same query returns `Certified` with identical top-1; `SearchMode::Exact`
always agrees with `searchExact` under the Phase 1 invariant, and never yields `DeadlineHit`/`BudgetHit`.

---

## Phase 4 — Surface it

- **[src/collection.cpp](src/collection.cpp)**: `SearchResponse searchIVF(query, k, SearchMode)` becomes the
  primary entry point; keep a `nprobe`-taking overload for tests.
- **[src/main.cpp](src/main.cpp)**: extend the `search` command with an optional trailing
  `--mode <low_latency|balanced|high_recall|exact>` and print the status line, e.g.
  `[certified — scanned 2/32 clusters, 61 vectors, 0.08 ms]`. Parse the flag *before*
  `parseVectorArgs` ([main.cpp:25](src/main.cpp#L25)), which currently consumes every remaining argv as a
  double and would choke on a flag. Update `printUsage` ([main.cpp:11](src/main.cpp#L11)).
- **[CMakeLists.txt](CMakeLists.txt)**: add `tests/test_ivf.cpp` to the `vectordb_test` target
  ([CMakeLists.txt:35-39](CMakeLists.txt#L35-L39)). `src/ivf_index.cpp` is already in the library target.
- Update the "Limitations" section of [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) — brute-force-only is no
  longer true.

**Verify:** `./vectordb search <col> 5 <q...> --mode low_latency` vs `--mode exact` on a seeded collection —
same top-1, different reported cluster counts and statuses.

---

## Out of scope

Index persistence to JSON (rebuilt in-process each CLI run), quantization/PQ, memory budgeting, incremental
re-clustering on drift, and any algorithm-comparison benchmark report.

## End-to-end verification

```bash
cmake --build build
cd build && ctest --output-on-failure      # all 76 existing tests must still pass
./vectordb_test --gtest_filter="IVF*"
```

Plus the CLI check in Phase 4. The load-bearing invariant across the whole plan: **whenever status is
`Certified` or `Exhausted`, IVF must return the same ids in the same order as `searchExact` under the
(score desc, id asc) total order, with scores agreeing to within `1e-12`.** Exact score equality is not
achievable — the two paths compute cosine differently — and asserting it would fail for reasons that have
nothing to do with the bound.
