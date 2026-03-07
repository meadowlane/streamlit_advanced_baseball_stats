# Performance & Scalability Plan: Filter-Validation Framework

## Important Note on Scope
Three files in the user's scope **do not yet exist** (only `__pycache__` artifacts remain):
- `tests/reference_calc.py`
- `tests/filter_verification/conftest.py`, `test_pitcher_hand.py`, `test_game_scope.py`
- `tools/verify_filters.py`

This plan covers **both the existing verification harness** (`tools/verification/`, `stats/filters.py`, `stats/splits.py`) **and design recommendations for the planned files** so they are built efficiently from the start.

---

## Context
The filter-validation framework validates that `apply_filters()` + stat computation produce correct results for each filter type (pitcher_hand, batter_hand, inning, month, home_away, count) and their combinations. It mirrors the existing stat-verification harness (`tools/verify_stats.py` → `tools/verification/engine.py`) but adds a new dimension: the same player's data must be processed under multiple filter conditions. The risk is that a naive implementation fetches and computes data independently for each filter, causing O(N_filters × N_players) network calls and file I/O when the correct structure is O(N_players) fetches with O(N_filters) local operations.

---

## Executive Summary

The existing stat-verification harness has two confirmed, high-impact redundancies: Statcast data is fetched **twice per player** (once in `AppSource`, once in `StatcastSource`) and FanGraphs season data is fetched **once per player** per adapter rather than once per year. These are pure waste with straightforward fixes. Beyond these, the overall computation model is well-designed: vectorized pandas operations throughout, a sound `get_prepared_df_cached()` pattern in `stats/filters.py`, and no nested O(N²) loops in stat computation.

The larger risk is in what doesn't exist yet. If `tools/verify_filters.py` is designed as "call `run_verification()` once per filter type," the Statcast fetch count becomes O(N_filters × N_players) — 6× worse than the already-suboptimal current state. Preventing this structural mistake is the single most important recommendation in this plan. The fixture storage decision for filter verification (raw vs. pre-filtered) is the second most consequential choice.

**Nothing in `stats/filters.py` or `stats/splits.py` is algorithmically problematic.** The computation is already vectorized, the caching is sound, and the filter logic is correct. Minor improvements (combined mask, remove unnecessary `.copy()`) are real but low-priority.

---

## Top Risks (Highest Impact First)

### Risk 1 — Future Design Trap: O(N_filters × N_players) fetch structure in `tools/verify_filters.py`
**Not built yet; prevent before it's written.**
If the planned `verify_filters.py` CLI is designed as an outer loop over filter types that calls the existing `run_verification()` engine per filter, Statcast will be fetched `N_filters × N_players` times instead of `N_players` times. For 6 filter types and 10 players, that's 60 Statcast fetches instead of 10. This is the single structural decision with the most impact on scalability.

### Risk 2 — Statcast double-fetch in existing engine (confirmed)
`AppSource.get_batter_season()` (`app_source.py:114`) and `StatcastSource.get_batter_season()` (`statcast.py:71`) both call `_fetch_statcast_batter(player.mlbam_id, year)` independently. For N players this is 2N Statcast fetches; N would suffice. Same for pitchers via `_fetch_statcast_pitcher`.

### Risk 3 — FanGraphs data fetched N times for N players (confirmed)
`AppSource` calls `_fetch_batting_stats(year)` per player (`app_source.py:103`); `FanGraphsSource` calls `pb.batting_stats(year)` per player. Both import the non-cached private functions, bypassing `@st.cache_data`. For a 50-player run verifying a single year, FG data is fetched ~100 times when once would suffice.

### Risk 4 — Fixture explosion risk for filter verification
If filter-specific fixtures are stored as one file per (player × filter × season) combination, fixture count grows as O(N_players × N_filter_combos × N_seasons). With 6 filter types, 3 seasons, and 20 players, that's 360+ files before testing multi-filter combinations. The correct design stores raw Statcast data per player and derives all filtered results locally in tests.

### Risk 5 — test_stat_verification.py: per-test fixture loads with no sharing
`_run_offline()` is called independently for each of 12 parametrized tests (`test_stat_verification.py:321–364`). Each call performs 4 `fixture_exists()` filesystem checks + JSON loads. With no session-scoped caching, adding more golden players scales poorly. Minor today (12 tests), but each new golden player adds 3 more independent load cycles.

### Risk 6 — `apply_filters()` sequential mask assignments (minor)
Each active filter creates an intermediate DataFrame view by reassigning `df` (`filters.py:407–448`). With 6 active filters, up to 8 reassignments occur. Combining into one boolean mask avoids intermediate allocations. This matters if filter verification calls `apply_filters()` in a tight inner loop across many filter combos.

---

## Specific Recommendations

---

### REC-1: Design `verify_filters.py` with player-outer / filter-inner structure ⚡ DO NOW

**Why it matters:** Prevents the O(N_filters × N_players) fetch trap permanently. The key insight is that filter combinations only affect downstream computation, not upstream data fetching. All filters operate on the same raw Statcast DataFrame.

**Correct structure:**
```python
# tools/verify_filters.py
def run_filter_verification(player_ids, filter_specs, year, player_type):
    for mlbam_id in player_ids:                       # O(N_players) fetches
        raw_df = _fetch_statcast_batter(mlbam_id, year)   # ONCE
        prepared = prepare_df(raw_df)
        for spec in filter_specs:                     # O(N_filters) local ops
            filtered = apply_filters(prepared, spec.as_split_filters())
            computed = _compute_stats(filtered, player_type)
            reference = compute_reference(raw_df, spec)
            compare_and_record(computed, reference, spec, mlbam_id)
```

**Wrong structure (do not do):**
```python
for spec in filter_specs:
    run_verification(player_ids, ...)   # fetches Statcast N_players × again
```

**Expected impact:** Prevents 6× fetch inflation before it exists.
**Complexity / risk:** Low — this is a design decision, not a refactor.
**Helps:** CI and local dev equally.

---

### REC-2: Cache `_fetch_batting_stats` / `_fetch_pitching_stats` with `lru_cache` ⚡ DO NOW

**Why it matters:** FanGraphs season data is the same for all players in a given year. `AppSource` and `FanGraphsSource` both fetch it independently per player, bypassing `@st.cache_data`. For a 10-player run, FG data is fetched ~20 times.

**Fix:** Add `@functools.lru_cache(maxsize=4)` to `_fetch_batting_stats()` and `_fetch_pitching_stats()` in `data/fetcher.py`. The `maxsize=4` guards against holding multiple large DataFrames indefinitely (2 years × 2 player types = 4 slots).

**Affected lines:** `data/fetcher.py` — both private fetch functions.
**Note:** These functions are already imported as private (`_fetch_batting_stats`) by `app_source.py:22` — the cache will be shared because it decorates the function object itself.
**Expected impact:** Reduces FG fetches from O(N_players) to O(1) per year.
**Complexity / risk:** Very low. Only risk is stale data within a process (acceptable — fixtures and offline mode are unaffected).
**Helps:** CLI runs and CI equally.

---

### REC-3: Add a module-level Statcast cache in `engine.py` ⚡ DO NOW

**Why it matters:** `AppSource.get_batter_season()` (`app_source.py:114`) and `StatcastSource.get_batter_season()` (`statcast.py:71`) independently call `_fetch_statcast_batter(mlbam_id, year)`. Each player's raw Statcast data is fetched twice per `run_verification()` call.

**Fix:** Add a simple process-scoped cache to `engine.py` (or `app_source.py`):
```python
# tools/verification/engine.py
_SC_BATTER_CACHE: dict[tuple[int, int], pd.DataFrame] = {}
_SC_PITCHER_CACHE: dict[tuple[int, int], pd.DataFrame] = {}

def _get_statcast_batter(mlbam_id: int, year: int) -> pd.DataFrame:
    key = (mlbam_id, year)
    if key not in _SC_BATTER_CACHE:
        _SC_BATTER_CACHE[key] = _fetch_statcast_batter(mlbam_id, year)
    return _SC_BATTER_CACHE[key]
```

Then thread through to both `AppSource` and `StatcastSource`. Alternatively, pass the raw DataFrame as a parameter to both adapters (cleaner but more invasive).

**Expected impact:** Halves Statcast network calls.
**Complexity / risk:** Low. Cache is per-process and per-verification-run; no thread-safety issues if CLI stays single-threaded.
**Helps:** CLI runs and CI equally.

---

### REC-4: Design `tests/reference_calc.py` as one-pass computation, not per-stat ⚡ DO NOW

**Why it matters:** The reference calculator is called once per (player, filter_combo). If it re-applies the filter or re-parses raw data per stat, it becomes O(N_stats × filter_overhead) instead of O(N_stats + filter_overhead).

**Correct design:**
```python
# tests/reference_calc.py
def compute_reference_stats(raw_df: pd.DataFrame, filters: SplitFilters) -> dict[str, float]:
    """Apply filters once; compute all reference values in one pass."""
    filtered = apply_filters(prepare_df(raw_df), filters)
    return {
        "K%": _ref_k_pct(filtered),
        "BB%": _ref_bb_pct(filtered),
        "Whiff%": _ref_whiff_pct(filtered),
        # ... all stats from one filtered DataFrame
    }
```

Each `_ref_*` function should operate on the pre-filtered DataFrame, not re-filter.

**Expected impact:** Prevents hidden O(N_stats × filter) scaling.
**Complexity / risk:** Low — design choice for a new file.
**Helps:** Test runtime.

---

### REC-5: Store raw Statcast fixtures (not pre-filtered) for filter verification ⚡ DO NOW

**Why it matters:** If filter verification fixtures store pre-computed results for each (filter × player) combination, fixture count grows multiplicatively. Adding a new filter type requires re-recording all combinations. Storing raw Statcast data per player allows all filter combinations to be derived locally in tests.

**Recommended fixture layout:**
```
tests/filter_verification/fixtures/
  statcast_batter_{mlbam_id}_{year}.json   # raw pitch-level data
  statcast_pitcher_{mlbam_id}_{year}.json
```
Tests load raw data, apply filters locally, compute stats, compare to known expected values (hardcoded in test constants, not in fixtures).

**Expected impact:** Prevents O(N_players × N_filter_combos) fixture explosion.
**Complexity / risk:** Medium — raw Statcast fixtures are ~200-500KB each (compressed). Manageable for a golden set of 5-10 players. Use only the minimal columns needed (not the full ~90-column Statcast blob): `game_date`, `events`, `description`, `p_throws`, `stand`, `inning_topbot`, `inning`, `balls`, `strikes`, `game_type`, `launch_speed`, `launch_angle`, `estimated_woba_using_speedangle`.
**Helps:** Long-term maintainability; prevents fixture rot.

---

### REC-6: Session-scoped fixture loading in `tests/filter_verification/conftest.py`

**Why it matters:** For filter verification, the same player's raw data will be used across dozens of test functions (one per filter type × stat). Loading it once per session instead of per-test eliminates repeated disk I/O.

**Design for conftest.py:**
```python
# tests/filter_verification/conftest.py
@pytest.fixture(scope="session")
def raw_statcast_fixtures() -> dict[int, pd.DataFrame]:
    """Load all golden player raw Statcast fixtures once for the session."""
    return {
        mlbam_id: _load_raw_fixture(mlbam_id, year)
        for mlbam_id, year in GOLDEN_FILTER_PLAYERS
    }

@pytest.fixture(scope="session")
def prepared_dfs(raw_statcast_fixtures):
    """Pre-apply prepare_df() once per player."""
    return {k: prepare_df(df) for k, df in raw_statcast_fixtures.items()}
```

Test functions then receive `prepared_dfs` and apply filter combinations locally.

**Expected impact:** Reduces fixture file I/O from O(N_players × N_tests) to O(N_players) per session.
**Complexity / risk:** Low.
**Helps:** CI test runtime.

---

### REC-7: Add `@pytest.fixture(scope="session")` for golden set in `test_stat_verification.py`

**Why it matters:** `_run_offline()` is called independently 12 times (4 batters × 3 test methods), each performing 4 `fixture_exists()` checks + JSON loads. Adding players scales this linearly.

**Fix:** Cache results in a session-scoped fixture in `tests/conftest.py`:
```python
@pytest.fixture(scope="session")
def golden_batter_reports():
    return {
        (mlbam_id, year): _run_offline(mlbam_id, year, "batter")
        for mlbam_id, year, _ in _GOLDEN_BATTERS
    }
```
Test methods receive this dict and look up by key.

**Expected impact:** Reduces disk I/O from 12 independent loads to 4 (one per player).
**Complexity / risk:** Low. Requires modest refactor of 3 test methods.
**Helps:** CI test runtime.

---

### REC-8: Combine filter masks in `apply_filters()` into a single boolean operation — LATER

**Why it matters:** Currently, `apply_filters()` (`filters.py:388–450`) reassigns `df` up to 8 times for 6 active filters. Each reassignment creates an intermediate slice. A single combined mask would reduce this to one allocation.

**Fix:**
```python
def apply_filters(df, filters, pitcher_perspective=False):
    # fast-path (already exists — good)
    if not any([...]):
        return df

    mask = pd.Series(True, index=df.index)
    if filters.inning_min is not None and _col_ok(df, "inning", "inning"):
        mask &= df["inning"] >= filters.inning_min
    if filters.inning_max is not None and _col_ok(df, "inning", "inning"):
        mask &= df["inning"] <= filters.inning_max
    # ... all other filters
    return df[mask]
```

**Expected impact:** Reduces intermediate DataFrame allocations. Meaningful only when `apply_filters()` is called in a tight inner loop (e.g., iterating 50+ filter combinations per player). For current use (one call per player per UI update), impact is negligible.
**Complexity / risk:** Low code change; medium refactor risk (must preserve `_col_ok` semantics and the Streamlit warning path).
**Helps:** Filter verification CLI performance.

---

### REC-9: Remove unnecessary `.copy()` from `_pa_events()` in `stats/splits.py` — LATER

**Why it matters:** `_pa_events()` (`splits.py:~184`) returns a copy of the filtered DataFrame. Since its callers only read from it (they do not mutate), a view suffices.

**Fix:** Return `df[mask]` instead of `df[mask].copy()`.

**Expected impact:** Minor memory reduction (~500KB per player for a full season). Not measurable in current usage.
**Complexity / risk:** Very low.
**Helps:** Memory footprint under high player volume.

---

### REC-10: Pre-check column existence outside groupby loop in `compute_pitch_arsenal()` — LATER

**Why it matters:** `splits.py` checks `if "release_speed" in grp.columns` inside every pitch-type group iteration. The columns are invariant across groups.

**Fix:** Check `has_velo = "release_speed" in work.columns` before the loop.

**Expected impact:** Negligible (typical pitch arsenal has 5-8 types; column check is O(K)).
**Complexity / risk:** Trivial.
**Helps:** Code clarity.

---

## Do Now vs Later

### Do Now (before building the planned files)
| # | Recommendation | Why Now |
|---|---------------|---------|
| 1 | Design `verify_filters.py` with player-outer loop | Prevents structural trap permanently |
| 2 | `lru_cache` on `_fetch_batting_stats/pitching_stats` | Trivial, high payoff, no risk |
| 3 | Module-level Statcast cache in `engine.py` | Halves Statcast fetches now |
| 4 | Design `reference_calc.py` as one-pass | Design decision for new file |
| 5 | Raw Statcast fixture design for filter tests | Architecture decision, hard to reverse |
| 6 | Session-scoped `conftest.py` for filter tests | Foundational for filter test suite |

### Later (after filter framework is working)
| # | Recommendation | When |
|---|---------------|------|
| 7 | Session-scoped golden set fixture in `test_stat_verification.py` | When golden set grows beyond 10 players |
| 8 | Combined mask in `apply_filters()` | When filter verification has >20 filter combos |
| 9 | Remove `.copy()` from `_pa_events()` | Next time splits.py is touched |
| 10 | Pre-check columns in `compute_pitch_arsenal()` | Low priority; cleanup only |
| — | ThreadPoolExecutor for parallel source fetching | When CI run time becomes a bottleneck (>3 min) |

---

## Quick Wins (Low Effort, Immediate Payoff)

1. **`@functools.lru_cache(maxsize=4)` on `_fetch_batting_stats()` and `_fetch_pitching_stats()`** — `data/fetcher.py`, ~2 lines, eliminates the FG over-fetching problem immediately.

2. **Module-level dict cache in `engine.py`** for Statcast DataFrames — ~10 lines, halves Statcast network calls.

3. **Move `_col_ok()` Streamlit import to module level** — `filters.py:339` imports `streamlit.runtime` inside the function body, on every call that reaches the error path. A module-level flag `_IN_STREAMLIT: bool = False` initialized at import time eliminates repeated import resolution in non-Streamlit contexts (verification CLI, tests).

---

## Dangerous Optimizations to Avoid

**DO NOT parallelize fixture writes.** `save_fixture()` in `fixtures.py:75` writes JSON files. Parallelizing `run_verification()` across players risks concurrent writes to the same filename if called with duplicate player IDs. The risk is silent data corruption in fixture files.

**DO NOT reduce the golden player set to speed up CI.** The current 7-player golden set (4 batters, 3 pitchers) provides meaningful coverage. Shrinking it weakens trustworthiness. If CI is slow, parallelize the test runner (`pytest-xdist`) rather than reducing assertions.

**DO NOT replace raw Statcast fixtures with synthetic data for filter tests.** The entire value of fixture-based verification is that it uses real data. Synthetic DataFrames would test the filter logic but not catch real-world edge cases (e.g., players with games straddling April/May boundary, mixed inning_topbot encoding in doubleheaders).

**DO NOT add `lru_cache` to `_fetch_statcast_batter/pitcher()`.** Statcast DataFrames are 5-10MB each. An unconstrained cache across N players would hold gigabytes in memory. The controlled dict cache in `engine.py` (REC-3) is safer because it's explicitly scoped to one verification run.

**DO NOT make fixtures per-filter-combination** (e.g., `statcast_batter_{id}_{year}_ph=R.json`). This path leads to hundreds of small fixture files that must all be re-recorded when the filter logic changes. Store raw data; derive everything locally.

---

## Suggested Benchmark Points (Capture Before Changing Code)

These measurements establish a baseline so improvements can be proven:

1. **Single-player live verification time:**
   ```bash
   time python tools/verify_stats.py --players "Zack Wheeler" --season 2024 --player-type pitcher
   ```
   Expected breakdown: ~1-2s FG fetch, ~3-5s Statcast fetch × 2, ~0.5s other sources.

2. **Multi-player live verification time (5 players, same season):**
   ```bash
   time python tools/verify_stats.py --golden-set --player-type batter --season 2024
   ```
   Track wall-clock time. With REC-2 + REC-3, expect ~40% reduction.

3. **Offline pytest run time:**
   ```bash
   time pytest tests/test_stat_verification.py -v
   ```
   Should be <5s. Track after REC-7.

4. **Number of `_fetch_statcast_batter()` calls for N players:**
   Add a counter (temporary `print` or `unittest.mock.patch` spy) before making any changes. Should be 2N currently; target is N after REC-3.

5. **Number of `pb.batting_stats()` calls for N-player run:**
   Same approach. Should be 2N currently (AppSource + FanGraphsSource); target is 1 after REC-2.

---

## What Is Already Efficient (Do Not Touch)

- **`stats/splits.py` stat computation loop** — Vectorized throughout. Filters are applied once before the stat loop. No per-stat DataFrame scans.
- **`stats/filters.py` fast-path** — `apply_filters()` returns `df` unchanged (no copy) when no filters are active. This is the correct optimization.
- **`get_prepared_df_cached()`** — The `(player_id, season, type)` cache key is appropriate. Caching is correctly scoped.
- **`tools/verification/stat_map.py`** — `STAT_MAP` and `TOLERANCES` are module-level constants. All lookups are O(1).
- **`tools/verification/comparison.py`** — O(N_stats × N_sources) comparisons per player. No nested quadratic loops.
- **`tools/verification/game_scope.py`** — `filter_by_scope()` is a single `.isin()` boolean index. Unavoidably O(N_rows) but minimally so.
- **`tools/verification/normalization.py`** — `normalize_stat()` is O(1) per value. No room for improvement.

---

## Files Affected by Recommendations

| File | Recs | Nature of Change |
|------|------|-----------------|
| `data/fetcher.py` | REC-2 | Add `@functools.lru_cache` to `_fetch_batting_stats`, `_fetch_pitching_stats` |
| `tools/verification/engine.py` | REC-3 | Add module-level Statcast dict cache; thread through to source adapters |
| `tools/verify_filters.py` | REC-1 | Design player-outer / filter-inner loop from the start (new file) |
| `tests/reference_calc.py` | REC-4 | Design one-pass reference computation (new file) |
| `tests/filter_verification/conftest.py` | REC-5, REC-6 | Session-scoped raw fixture loading (new file) |
| `tests/conftest.py` | REC-7 | Add session-scoped golden set fixture |
| `stats/filters.py` | REC-8 | Combined mask in `apply_filters()` (later) |
| `stats/splits.py` | REC-9, REC-10 | Remove `.copy()`, pre-check columns (later) |
