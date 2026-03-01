# MLB Splits App — Filter-Wiring Audit

> Generated: 2026-03-01
> Branch: `claude/audit-filter-dataframe-wiring-6vvyE`
> Scope: Full repo audit of sidebar filter → dataframe → component wiring.

---

## A) Inventory: DataFrame + Lifecycle Map

```
raw_statcast_df
  created:  app.py:978  →  fetch_statcast_fn(mlbam_id, season_a)
  cached:   data/fetcher.py  @st.cache_data(ttl=3600)
  cache key: (mlbam_id: int, season: int)
  columns:  game_date, batter/pitcher, p_throws, stand, inning,
            inning_topbot, launch_speed, events, balls, strikes,
            estimated_woba_using_speedangle, pitch_type, plate_x, plate_z, …

statcast_df  (= "prepared")
  created:  app.py:982-987  →  get_prepared_df_cached(raw_statcast_df, …)
  function: stats/filters.py:prepare_df() (lines 211-227)
  cached:   session_state["_prepared_df_cache"]  (manual memoisation dict)
  cache key: (int(mlbam_id), int(season), str(player_type))
  adds:     _month (from game_date.dt.month), inning coerced to numeric
  note:     cache key does NOT include filter state — correct, because
            filtering is applied AFTER this cache

filtered_df
  created:  app.py:989
            apply_filters(statcast_df, filters, pitcher_perspective=…)
  NOT cached  — recomputed fresh on every Streamlit rerun
  depends:  statcast_df  +  filters object (from session_state["filter_rows"])
  effect:   row subset only; column set unchanged

filtered_df_b  (comparison mode)
  created:  app.py:1005
            apply_filters(statcast_df_b, filters, pitcher_perspective=…)
  mirrors:  filtered_df pattern; same filters object applied to Player B

filtered_mix_df  (pitcher mode only)
  created:  app.py:1262
            _filter_pitch_mix_by_batter_hand(filtered_df, pitch_mix_batter_hand, …)
  parent:   filtered_df  — additional sub-filter on top of active filters

player_stats  (dict)
  created:  app.py:1055-1060
            _compute_stats(filtered_df)  or  _compute_all_pitcher_stats(filtered_df)
            THEN wRC+ and TRAD stats overwritten from player_row (FG season data)
  ⚠️  HYBRID source — Statcast stats from filtered_df, but wRC+/AVG/OBP/SLG/OPS
      from season_df (FG) regardless of active filters

splits_df
  created:  app.py:1346-1402
            get_splits(filtered_df, split_type)  or
            get_pitcher_splits(filtered_df, split_type)
  depends:  filtered_df  ✅

trend_data_a / trend_data_b  (list[dict])
  created:  app.py:1470-1487  via  stats/splits.py:get_trend_stats() (lines 582-643)
  per season: fetch (cached) → prepare (shared cache) → apply_filters(prepared, trend_filters)
  trend_filters = filters  if apply_trend_filters  else  SplitFilters()  (no filters)
  note:     correctly updated on every rerun when apply_trend_filters=True

season_df  (FanGraphs season stats)
  created:  app.py:536-542  →  get_batting_stats() / get_pitching_stats()
  cached:   data/fetcher.py  @st.cache_data(ttl=3600),  key (season, min_pa)
  role:     player list, percentile distributions, wRC+/TRAD stat values
  NOT Statcast data — pitch-level filters cannot apply to this source
```

---

## B) Filter Catalog

| Filter name (UI label) | Where defined | Applied via | Column(s) touched | Effect | Edge cases |
|---|---|---|---|---|---|
| **Inning range** | `stats/filters.py:63-69` `FILTER_REGISTRY["inning"]` | `apply_filters()` line 383-388 | `inning` (numeric) | Row subset: `inning_min ≤ inning ≤ inning_max` | Skipped + `st.warning` if `inning` column absent |
| **Pitcher handedness** | `stats/filters.py:70-75` | `apply_filters()` line 391-393 | `p_throws` | Row subset: `p_throws == "L"` or `"R"` | Excluded in Pitcher mode (only `batter_hand` available instead) |
| **Batter handedness** | `stats/filters.py:76-81` | `apply_filters()` line 395-398 | `stand` | Row subset: `stand == "L"` or `"R"` | Excluded in Batter mode; relabeled "Batter hand" for Pitcher mode |
| **Home / Away** | `stats/filters.py:82-88` | `apply_filters()` line 400-407 | `inning_topbot` | Row subset: `"Bot"` = home bats, `"Top"` = away bats; perspective flipped for Pitcher mode via `pitcher_perspective` flag | Wrong/absent column → skipped |
| **Month** | `stats/filters.py:89-95` | `apply_filters()` line 409-414 | `_month` (preferred) / `game_date` (fallback) | Row subset: specific calendar month | Falls back to on-the-fly `game_date.dt.month` if `_month` absent (pre-prepare_df data path) |
| **Count** | `stats/filters.py:96-102` | `apply_filters()` line 416-424 | `balls`, `strikes` | Row subset: exact balls and/or strikes count; `None` = any | Either or both can be `None`; `"any"` in UI maps to `None` |

Filter → `SplitFilters` translation: `rows_to_split_filters()` at `stats/filters.py:158-208`.
Last-row-wins when two rows set the same field (e.g. two inning rows).

---

## C) Component Dependency Matrix

| UI Section | App.py lines | Source DataFrame | Should respect filters? | Currently respects filters? | Assessment |
|---|---|---|---|---|---|
| **Season Stat Cards — Statcast stats** (wOBA, xwOBA, K%, BB%, HardHit%, Barrel%, pitcher-specific) | 1150-1230 | `player_stats` ← `_compute_stats(filtered_df)` | ✅ Yes | ✅ Yes | Correct |
| **Season Stat Cards — wRC+** | 1057, 1230 | `player_row.get("wRC+")` ← `season_df` (FG) | ❌ Not possible (no pitch-level FG data) | ⚠️ Always full-season | **By design, but UI doesn't annotate it** |
| **Season Stat Cards — TRAD (AVG/OBP/SLG/OPS)** | 1058-1059, 1230 | `player_row.get(stat)` ← `season_df` (FG) | ❌ Not possible | ⚠️ Always full-season | **Same as wRC+ — undisclosed to user** |
| **Pitch Arsenal** | 1262-1289 | `filtered_mix_df` ← `_filter_pitch_mix_by_batter_hand(filtered_df, …)` | ✅ Yes | ✅ Yes | Correct; has its own additional batter-hand sub-filter on top |
| **Pitch Zone Chart — Player A** | 1291-1295 | `filtered_df` | ✅ Yes | ✅ Yes | Correct |
| **Pitch Zone Chart — Player B** | (not rendered) | (none) | ✅ Yes, if comparison | ❌ Missing entirely | **No Player B zone chart in comparison mode** |
| **Percentile Bar Chart — Statcast stats** | 1307-1311 | `percentiles` ← `get_all_percentiles(player_stats, distributions)` where `player_stats` ← `filtered_df` | ✅ Yes | ✅ Yes | Correct |
| **Percentile Bar Chart — wRC+ percentile** | 1307-1311 | `percentiles["wRC+"]` ← `player_stats["wRC+"]` ← FG season | ⚠️ Partial — percentile rank is correct (vs full-season league), but value input is full-season | ⚠️ Value is always unfiltered | **By design; inconsistent appearance** |
| **Split Table** | 1334-1402 | `get_splits(filtered_df, split_type)` / `get_pitcher_splits(filtered_df, split_type)` | ✅ Yes | ✅ Yes | Correct |
| **Trend Dashboard / Custom (apply_trend_filters=False)** | 1492-1526 | `get_trend_stats(…, SplitFilters())` — no filters | Opt-in (No by default) | ⚠️ Opt-in only | **By design; checkbox at line 1444** |
| **Trend Dashboard / Custom (apply_trend_filters=True)** | 1492-1526 | `get_trend_stats(…, filters)` — current filters | ✅ Yes | ✅ Yes | Correct; recomputed every rerun |
| **Percentile distributions (league reference)** | 1062-1076 | `build_league_distributions(season_df)` | ❌ Intentionally full-season | ✅ Correct by design | Documented at 1051-1052: "Distributions use the full-season league (correct reference population)" |
| **Stat column validation** | 1010-1040 | `statcast_df` (unfiltered) | N/A — column existence check | ✅ Correct | Using unfiltered df is correct; filtering removes rows, not columns |

---

## D) Proposed Fixes

### Root cause summary

The codebase is architecturally **sound**. The `filtered_df` is correctly wired to almost all
Statcast-derived views. There is **no systemic filter-bypass bug**.

The real issues are:

| # | Issue | Severity | Root cause |
|---|---|---|---|
| 1 | wRC+ and TRAD stats (AVG/OBP/SLG/OPS) always show full-season FG values even when Statcast filters are active — but no UI indication distinguishes them from filtered stats | UX / Misleading | FG data has no pitch-level granularity. Undisclosed to user. |
| 2 | Trend default is no filters; user must opt in with checkbox | UX | `apply_trend_filters` defaults `False`; `trend_filters = SplitFilters()` at line 1468 |
| 3 | `_trend_ctx` cache key (line 1454) omits filter state — if `apply_trend_filters=True`, filter changes invalidate trend display but don't re-prompt the Load button | Minor UX | `_trend_ctx` only includes `(mlbam_id, season_a, mlbam_id_b, season_b)` |
| 4 | No Player B pitch zone chart in comparison mode | Missing feature | `render_pitch_zone_chart(filtered_df, …)` at line 1295 — only Player A |

---

### 1) Minimal-diff patch plan

**Chunk 1 — Annotate unfiltered FG stats (Issue #1)**

In `app.py`, when filters are active (`active_filter_summary != "No filters (full season data)"`),
add a `st.caption` under the stat grid noting which stats are full-season FG values.

```python
# app.py — after stat cards rendering block (after line ~1230), add:
_fg_stats_shown = [s for s in selected_stats if s in {"wRC+", "AVG", "OBP", "SLG", "OPS"}]
_statcast_filters_active = active_filter_summary != "No filters (full season data)"
if _fg_stats_shown and _statcast_filters_active:
    st.caption(
        f"Note: {', '.join(_fg_stats_shown)} use full-season FanGraphs values "
        "and are unaffected by Statcast filters."
    )
```

No change to data wiring needed — this is a UI transparency fix only.

---

**Chunk 2 — Include filter state in `_trend_ctx` (Issue #3)**

In `app.py` around line 1454, incorporate a filter fingerprint so the Load button reappears
whenever filters change while `apply_trend_filters` is active:

```python
# Before (app.py:1454-1459):
_trend_ctx = (
    int(mlbam_id),
    int(season_a),
    int(mlbam_id_b) if (comparison_mode and mlbam_id_b is not None) else None,
    int(season_b) if comparison_mode else int(season_a),
)

# After:
_filter_fingerprint = (
    active_filter_summary          # str: uniquely encodes filter state
    if apply_trend_filters
    else ""
)
_trend_ctx = (
    int(mlbam_id),
    int(season_a),
    int(mlbam_id_b) if (comparison_mode and mlbam_id_b is not None) else None,
    int(season_b) if comparison_mode else int(season_a),
    _filter_fingerprint,           # NEW
)
```

`active_filter_summary` is already computed before this point (line 891).

---

**Chunk 3 — Add Player B pitch zone chart in comparison mode (Issue #4)**

In `app.py` around line 1291-1295:

```python
# Before:
if FEATURE_PITCH_ZONE:
    from ui.components import render_pitch_zone_chart
    _pitch_zone_role = "pitcher" if player_type == "Pitcher" else "batter"
    render_pitch_zone_chart(filtered_df, role=_pitch_zone_role)

# After:
if FEATURE_PITCH_ZONE:
    from ui.components import render_pitch_zone_chart
    _pitch_zone_role = "pitcher" if player_type == "Pitcher" else "batter"
    if comparison_mode and filtered_df_b is not None:
        _zone_col_a, _zone_col_b = st.columns(2)
        with _zone_col_a:
            st.caption(f"Player A: {selected_name}")
            render_pitch_zone_chart(filtered_df, role=_pitch_zone_role)
        with _zone_col_b:
            st.caption(f"Player B: {selected_name_b}")
            render_pitch_zone_chart(filtered_df_b, role=_pitch_zone_role)
    else:
        render_pitch_zone_chart(filtered_df, role=_pitch_zone_role)
```

---

**Chunk 4 — Trend filter opt-in label (Issue #2, optional)**

Consider renaming the checkbox label from `"Apply current filters to each year"` to
`"Apply sidebar filters to trend"` for clarity. Changing the default from `False` to `True`
is a product decision — opting in by default is more consistent but potentially surprising.

---

### 2) Clean architecture refactor plan

A `DataContext` object would consolidate the data pipeline and make the FG/Statcast split explicit.

**New file: `stats/context.py`**

```python
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class PlayerDataContext:
    """Single source of truth for one player at one (player, season, filters) combo.

    Statcast-derived stats (filtered) and FG season stats are stored separately
    so rendering code can never accidentally mix them without an explicit choice.
    """
    mlbam_id: int
    season: int
    player_type: str
    filters: SplitFilters

    # Data layers
    raw_df: pd.DataFrame          # @st.cache_data fetch result
    prepared_df: pd.DataFrame     # after prepare_df() — session_state cache
    filtered_df: pd.DataFrame     # after apply_filters() — NOT cached

    # Computed from filtered_df
    player_stats: dict[str, float | None]      # Statcast stats only
    sample_sizes: dict[str, int | None]
    distributions: dict[str, np.ndarray]       # league reference from season_df
    percentiles: dict[str, float]
    color_tiers: dict[str, dict[str, str]]

    # Explicitly separated: FG season stats, unaffected by Statcast filters
    fg_stats: dict[str, float | None]          # wRC+, AVG, OBP, SLG, OPS


def build_player_context(
    mlbam_id: int,
    season: int,
    player_type: str,
    filters: SplitFilters,
    season_df: pd.DataFrame,        # FG league reference
    player_row: pd.Series,          # FG player row
    prepared_cache: dict,
    fetch_fn: Callable,
    log_fn: Callable | None = None,
) -> PlayerDataContext:
    raw_df = fetch_fn(mlbam_id, season)
    cache_key = (int(mlbam_id), int(season), str(player_type))
    prepared_df = get_prepared_df_cached(raw_df, prepared_cache, cache_key, log_fn)
    filtered_df = apply_filters(prepared_df, filters, pitcher_perspective=(player_type == "Pitcher"))

    _raw = _compute_all_pitcher_stats(filtered_df) if player_type == "Pitcher" else _compute_stats(filtered_df)

    # FG stats stored explicitly — not mixed into player_stats
    fg_stats = {s: _as_optional_float(player_row.get(s)) for s in ["wRC+", "AVG", "OBP", "SLG", "OPS"]}

    distributions = (
        build_pitcher_league_distributions(season_df)
        if player_type == "Pitcher"
        else build_league_distributions(season_df)
    )
    # … add wRC+/TRAD to distributions for percentile ranking …

    percentiles = get_all_percentiles(_raw, distributions, player_type=player_type)
    color_tiers = get_all_color_tiers(percentiles)

    return PlayerDataContext(
        mlbam_id=mlbam_id, season=season, player_type=player_type, filters=filters,
        raw_df=raw_df, prepared_df=prepared_df, filtered_df=filtered_df,
        player_stats=_raw, sample_sizes=get_sample_sizes(filtered_df),
        distributions=distributions, percentiles=percentiles, color_tiers=color_tiers,
        fg_stats=fg_stats,
    )
```

**Caching strategy:**

| Layer | Mechanism | Cache key |
|---|---|---|
| Raw fetch | `@st.cache_data(ttl=3600)` | `(mlbam_id, season)` |
| Prepared DF | `session_state["_prepared_df_cache"]` | `(mlbam_id, season, player_type)` |
| `PlayerDataContext` | Not cached — recomputed each rerun (cheap) | — |
| Trend per-season | Uses prepared cache; filter applied inline | `(mlbam_id, season, player_type)` |

**Comparison mode under clean arch:**

```python
ctx_a = build_player_context(mlbam_id_a, season_a, player_type, filters, season_df_a, …)
ctx_b = build_player_context(mlbam_id_b, season_b, player_type, filters, season_df_b, …)
# Same `filters` SplitFilters object → applied independently to each player's data
# A and B never share a filtered_df
```

**Migration order (files to edit):**

1. **`stats/context.py`** — New file: `PlayerDataContext` + `build_player_context()`
2. **`app.py` lines 976-1111** — Replace with two `build_player_context()` calls
3. **`app.py` render sites** — Pass `ctx_a.filtered_df`, `ctx_a.player_stats`, `ctx_a.fg_stats` explicitly
4. **`ui/components.py`** — No changes needed; components already accept individual args

---

## E) Acceptance Criteria

### Filter-correctness test checklist

For each filter type (inning, pitcher hand, batter hand, home/away, month, count):

- [ ] Add filter → all Statcast stat cards (wOBA, xwOBA, K%, BB%, HardHit%, Barrel%) change value
- [ ] Add filter → Pitch Arsenal row counts reflect reduced sample
- [ ] Add filter → Pitch Zone Chart renders only pitches matching filter
- [ ] Add filter → Split table `PA` column totals change
- [ ] Add filter → Percentile bar chart for Statcast stats updates
- [ ] Add filter → `wRC+`/`AVG`/`OBP`/`SLG`/`OPS` stat cards do NOT change (expected — annotated caption should appear)
- [ ] Remove filter → all of the above revert to full-season values

**Comparison mode (A/B independence):**

- [ ] Applying a filter updates both Player A AND Player B stat cards independently
- [ ] Changing Player B's season does not affect Player A's filtered data
- [ ] Split tables for A and B both reflect active filters

**Trend:**

- [ ] With `apply_trend_filters=False`: trend data is unaffected by sidebar filters
- [ ] With `apply_trend_filters=True`: changing a filter updates trend chart on next rerun
- [ ] After Chunk 2 fix: changing filters with `apply_trend_filters=True` causes Load button to reappear

**Performance (no regression):**

- [ ] Toggling a filter does NOT re-fetch raw Statcast data (verify via `@st.cache_data` hit)
- [ ] Toggling a filter does NOT re-run `prepare_df` (verify via "cache hit" log)
- [ ] Trend with 10 seasons loads within existing timing

### Lightweight automated tests

```python
# tests/test_filter_wiring.py

import pandas as pd
from stats.filters import SplitFilters, apply_filters, prepare_df
from stats.splits import _compute_stats, get_splits


def _make_statcast_df() -> pd.DataFrame:
    """Minimal synthetic Statcast df covering both hands and innings."""
    return pd.DataFrame({
        "game_date":   ["2024-04-01"] * 100,
        "inning":      [1] * 50 + [7] * 50,
        "p_throws":    ["R"] * 70 + ["L"] * 30,
        "stand":       ["R"] * 60 + ["L"] * 40,
        "inning_topbot": ["Bot"] * 50 + ["Top"] * 50,
        "events":      ["single"] * 10 + [None] * 90,
        "estimated_woba_using_speedangle": [0.4] * 10 + [None] * 90,
        "launch_speed": [95.0] * 10 + [None] * 90,
        "bb_type":     ["fly_ball"] * 5 + [None] * 95,
        "balls":       [0] * 100,
        "strikes":     [0] * 100,
    })


def test_filter_reduces_rows():
    df = prepare_df(_make_statcast_df())
    f = SplitFilters(inning_min=7)
    result = apply_filters(df, f)
    assert len(result) == 50, "Inning filter should keep only late-inning rows"


def test_stats_change_with_filter():
    df = prepare_df(_make_statcast_df())
    stats_unfiltered = _compute_stats(df)
    f = SplitFilters(pitcher_hand="L")
    stats_filtered = _compute_stats(apply_filters(df, f))
    assert stats_unfiltered != stats_filtered


def test_splits_use_filtered_df():
    df = prepare_df(_make_statcast_df())
    f = SplitFilters(inning_max=1)
    filtered = apply_filters(df, f)
    splits = get_splits(filtered, "hand")
    full_splits = get_splits(df, "hand")
    assert splits["PA"].sum() < full_splits["PA"].sum()


def test_filter_does_not_mutate_prepared_df():
    df = prepare_df(_make_statcast_df())
    original_len = len(df)
    apply_filters(df, SplitFilters(inning_min=7))
    assert len(df) == original_len, "apply_filters must not mutate input df"


def test_no_filters_returns_same_object():
    df = prepare_df(_make_statcast_df())
    result = apply_filters(df, SplitFilters())   # all None
    assert result is df, "No-op filter should return original object (fast path)"
```

---

## Summary verdict

| Component | Filter-correct? | Action needed |
|---|---|---|
| Season stat cards (Statcast stats) | ✅ Yes | None |
| Season stat cards (wRC+, TRAD) | ⚠️ By design / undisclosed | Add annotating `st.caption` when filters active (Chunk 1) |
| Pitch Arsenal | ✅ Yes | None |
| Pitch Zone Chart — Player A | ✅ Yes | None |
| Pitch Zone Chart — Player B | ❌ Missing | Add B-side chart in comparison mode (Chunk 3) |
| Percentile bar chart (Statcast) | ✅ Yes | None |
| Percentile bar chart (wRC+) | ⚠️ By design / undisclosed | Covered by Chunk 1 caption |
| Split Table | ✅ Yes | None |
| Trend (opt-out default) | ⚠️ Opt-in | Chunk 2 for cache key; consider default change (Chunk 4) |
| Comparison A/B isolation | ✅ Yes | None — same `filters` applied to independent DFs |
| Prepared DF cache | ✅ Yes | None — cache key correctly excludes filters (filters applied post-cache) |

The codebase has **no systemic filter-bypass**. The observed "filter-stale" behavior is confined
to wRC+/TRAD stats (FG data, architecturally unfilterable) and the Trend view (user opt-in).
The minimal-diff fix is primarily a **UI transparency** issue (label the unfiltered stats),
plus a **cache-key** tightening for the Trend and a **missing Player B** zone chart.
