"""Split calculations from raw Statcast pitch-level data.

Each public function accepts a Statcast DataFrame (as returned by
data.fetcher.get_statcast_batter) and returns a tidy DataFrame with one row
per split group and one column per computed stat.

Stat computation notes
----------------------
- K%, BB%    : derived from plate-appearance outcome events.
- HardHit%   : batted balls with exit velocity >= 95 mph / total batted balls.
- Barrel%    : launch_speed_angle == 6 / total batted balls.
- xwOBA      : mean of estimated_woba_using_speedangle on batted ball events
               (Ks and BBs carry NaN in Statcast; this is approximate).
- wOBA       : computed from linear weights (2024 season constants).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
import os
import time

import pandas as pd
import streamlit as st

from data.fetcher import load_or_fetch_year_df
from stats.filters import SplitFilters, apply_filters, get_prepared_df_cached, prepare_df

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 2024 FanGraphs wOBA linear weights
WOBA_WEIGHTS = {
    "walk": 0.690,
    "hit_by_pitch": 0.722,
    "single": 0.888,
    "double": 1.271,
    "triple": 1.616,
    "home_run": 2.101,
}

# Events where Statcast records a plate-appearance outcome on the final pitch
PA_EVENTS = frozenset(
    [
        "single", "double", "triple", "home_run",
        "walk", "hit_by_pitch",
        "strikeout", "strikeout_double_play",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "force_out",
        "fielders_choice", "fielders_choice_out",
        "sac_fly", "sac_fly_double_play",
        "sac_bunt", "sac_bunt_double_play",
        "catcher_interf", "other_out",
    ]
)

# Events that produce a batted ball (have launch_speed populated)
BATTED_BALL_EVENTS = frozenset(
    [
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "force_out",
        "fielders_choice", "fielders_choice_out",
        "sac_fly", "sac_fly_double_play",
        "sac_bunt", "sac_bunt_double_play",
        "other_out",
    ]
)

# launch_speed_angle value that denotes a barrel
BARREL_CODE = 6

# Hard-hit exit-velocity threshold (mph)
HARD_HIT_MPH = 95

# K events used for K% computation
K_EVENTS = frozenset(["strikeout", "strikeout_double_play"])

# Ordered output columns
SPLIT_COLS = ["Split", "PA", "wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]

MONTH_NAMES = {
    3: "March/April",  # Spring Training overlap sometimes puts March games here
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
}

_DEBUG_TREND_TIMING = os.getenv("DEBUG_TREND_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


StatComputeFn = Callable[[pd.DataFrame, pd.DataFrame, int], float | None]


@dataclass(frozen=True)
class StatSpec:
    """Metadata + compute hook for a single displayed stat."""

    key: str
    label: str
    required_cols: list[str]
    formatter: Literal["pct_1", "decimal_3"]
    compute_fn: StatComputeFn
    glossary_key: str | None = None

def _pa_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows that represent plate-appearance outcomes."""
    return df[df["events"].notna() & df["events"].isin(PA_EVENTS)].copy()


def _batted_ball_events(pa: pd.DataFrame) -> pd.DataFrame:
    """Return PA-ending batted-ball events with non-null launch speed."""
    bb_mask = pa["events"].isin(BATTED_BALL_EVENTS) & pa["launch_speed"].notna()
    return pa[bb_mask]


def _compute_woba(pa: pd.DataFrame) -> float | None:
    """Compute wOBA from PA-level events using 2024 linear weights.

    Denominator: PA - sac_bunts  (approximates AB + BB + SF + HBP - IBB).
    """
    sac_bunts = pa["events"].isin({"sac_bunt", "sac_bunt_double_play"}).sum()
    denominator = len(pa) - sac_bunts
    if denominator == 0:
        return None

    numerator = sum(
        weight * pa["events"].eq(event).sum()
        for event, weight in WOBA_WEIGHTS.items()
    )
    return numerator / denominator


def _compute_k_rate(pa: pd.DataFrame, _bb_df: pd.DataFrame, n_pa: int) -> float | None:
    if n_pa == 0:
        return None
    return pa["events"].isin(K_EVENTS).sum() / n_pa


def _compute_bb_rate(pa: pd.DataFrame, _bb_df: pd.DataFrame, n_pa: int) -> float | None:
    if n_pa == 0:
        return None
    return (pa["events"] == "walk").sum() / n_pa


def _compute_hard_hit_rate(_pa: pd.DataFrame, bb_df: pd.DataFrame, _n_pa: int) -> float | None:
    n_bb_events = len(bb_df)
    if n_bb_events == 0:
        return None
    return (bb_df["launch_speed"] >= HARD_HIT_MPH).sum() / n_bb_events


def _compute_barrel_rate(_pa: pd.DataFrame, bb_df: pd.DataFrame, _n_pa: int) -> float | None:
    n_bb_events = len(bb_df)
    if n_bb_events == 0 or "launch_speed_angle" not in bb_df.columns:
        return None
    return (bb_df["launch_speed_angle"] == BARREL_CODE).sum() / n_bb_events


def _compute_xwoba_value(pa: pd.DataFrame, _bb_df: pd.DataFrame, _n_pa: int) -> float | None:
    xwoba_col = "estimated_woba_using_speedangle"
    if xwoba_col not in pa.columns:
        return None
    xwoba_vals = pa[xwoba_col].dropna()
    return float(xwoba_vals.mean()) if len(xwoba_vals) > 0 else None


def _compute_woba_value(pa: pd.DataFrame, _bb_df: pd.DataFrame, _n_pa: int) -> float | None:
    return _compute_woba(pa)


STAT_REGISTRY: dict[str, StatSpec] = {
    "wOBA": StatSpec(
        key="wOBA",
        label="wOBA",
        required_cols=["events"],
        formatter="decimal_3",
        compute_fn=_compute_woba_value,
        glossary_key="wOBA",
    ),
    "xwOBA": StatSpec(
        key="xwOBA",
        label="xwOBA",
        required_cols=["estimated_woba_using_speedangle"],
        formatter="decimal_3",
        compute_fn=_compute_xwoba_value,
        glossary_key="xwOBA",
    ),
    "K%": StatSpec(
        key="K%",
        label="K%",
        required_cols=["events"],
        formatter="pct_1",
        compute_fn=_compute_k_rate,
        glossary_key="K%",
    ),
    "BB%": StatSpec(
        key="BB%",
        label="BB%",
        required_cols=["events"],
        formatter="pct_1",
        compute_fn=_compute_bb_rate,
        glossary_key="BB%",
    ),
    "HardHit%": StatSpec(
        key="HardHit%",
        label="HardHit%",
        required_cols=["events", "launch_speed"],
        formatter="pct_1",
        compute_fn=_compute_hard_hit_rate,
        glossary_key="HardHit%",
    ),
    "Barrel%": StatSpec(
        key="Barrel%",
        label="Barrel%",
        required_cols=["events", "launch_speed", "launch_speed_angle"],
        formatter="pct_1",
        compute_fn=_compute_barrel_rate,
        glossary_key="Barrel%",
    ),
}


def _format_stat_value(value: float | None, formatter: Literal["pct_1", "decimal_3"]) -> float | None:
    if value is None:
        return None
    if formatter == "pct_1":
        return round(value * 100, 1)
    return round(value, 3)


def _compute_stats(df: pd.DataFrame) -> dict:
    """Compute all 6 core stats from a raw Statcast subset.

    Accepts all pitches for the subset; filters internally to PA events.
    Returns a dict keyed by stat name (None when not computable).
    """
    pa = _pa_events(df)
    n_pa = len(pa)

    if n_pa == 0:
        return {"PA": 0, **{key: None for key in STAT_REGISTRY}}

    bb_df = _batted_ball_events(pa)
    stats: dict[str, float | int | None] = {"PA": n_pa}

    for key, spec in STAT_REGISTRY.items():
        raw_value = spec.compute_fn(pa, bb_df, n_pa)
        stats[key] = _format_stat_value(raw_value, spec.formatter)

    return stats


FiltersSignature = tuple[
    int | None,
    int | None,
    str | None,
    str | None,
    int | None,
    int | None,
    int | None,
]


def _normalize_player_mode(player_type: str) -> str:
    lowered = str(player_type).strip().lower()
    return "pitcher" if lowered.startswith("pitch") else "batter"


def _filters_signature_from_filters(filters: SplitFilters) -> FiltersSignature:
    return (
        filters.inning_min,
        filters.inning_max,
        filters.pitcher_hand,
        filters.home_away,
        filters.month,
        filters.balls,
        filters.strikes,
    )


def _filters_from_signature(signature: FiltersSignature) -> SplitFilters:
    return SplitFilters(
        inning_min=signature[0],
        inning_max=signature[1],
        pitcher_hand=signature[2],  # type: ignore[arg-type]
        home_away=signature[3],  # type: ignore[arg-type]
        month=signature[4],
        balls=signature[5],
        strikes=signature[6],
    )


def _compute_selected_stat(df: pd.DataFrame, stat_key: str) -> dict[str, float | int | None]:
    pa = _pa_events(df)
    n_pa = len(pa)
    if n_pa == 0:
        return {"PA": 0, stat_key: None}

    spec = STAT_REGISTRY.get(stat_key)
    if spec is None:
        raise ValueError(f"Unknown trend stat key {stat_key!r}.")

    bb_df = _batted_ball_events(pa)
    raw_value = spec.compute_fn(pa, bb_df, n_pa)
    return {"PA": n_pa, stat_key: _format_stat_value(raw_value, spec.formatter)}


@st.cache_data(show_spinner=False)
def load_or_fetch_prepared_year_df(player_id: int, year: int, mode: str) -> pd.DataFrame:
    """Load a year of Statcast data and run prepare_df once per (player, year, mode)."""
    raw_df = load_or_fetch_year_df(player_id, year, _normalize_player_mode(mode))
    return prepare_df(raw_df)


@st.cache_data(show_spinner=False)
def compute_year_stat(
    player_id: int,
    year: int,
    mode: str,
    filters_signature: FiltersSignature,
    stat_key: str,
) -> dict[str, float | int | None]:
    """Compute one trend stat (or all stats) + sample sizes for a single season."""
    mode_key = _normalize_player_mode(mode)
    prepared = load_or_fetch_prepared_year_df(player_id, year, mode_key)
    filters = _filters_from_signature(filters_signature)
    filtered = apply_filters(prepared, filters)

    if stat_key == "__all__":
        stats = _compute_stats(filtered)
    else:
        stats = _compute_selected_stat(filtered, stat_key)

    sample_sizes = get_sample_sizes(filtered)
    stats["season"] = int(year)
    stats["n_pitches"] = sample_sizes.get("N_pitches")
    stats["n_bip"] = sample_sizes.get("N_BIP")
    stats["approx_pa"] = sample_sizes.get("approx_PA")
    return stats


def get_sample_sizes(df: pd.DataFrame) -> dict[str, int | None]:
    """Return display-ready sample sizes for a (possibly filtered) Statcast subset.

    Notes
    -----
    ``approx_PA`` is an approximation for pitch-level Statcast data: it counts
    rows where ``events`` is a recognized PA-ending outcome.
    """
    n_pitches = len(df)

    if "events" not in df.columns:
        return {"N_pitches": n_pitches, "N_BIP": None, "approx_PA": None}

    pa = _pa_events(df)
    approx_pa = len(pa)

    if "launch_speed" not in pa.columns:
        return {"N_pitches": n_pitches, "N_BIP": None, "approx_PA": approx_pa}

    bip_mask = pa["events"].isin(BATTED_BALL_EVENTS) & pa["launch_speed"].notna()
    n_bip = int(bip_mask.sum())

    return {"N_pitches": n_pitches, "N_BIP": n_bip, "approx_PA": approx_pa}


# ---------------------------------------------------------------------------
# Public split functions
# ---------------------------------------------------------------------------

def split_by_hand(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 2-row DataFrame: vs RHP and vs LHP splits.

    Uses the `p_throws` column (pitcher handedness).
    """
    rows = []
    for hand, label in [("R", "vs RHP"), ("L", "vs LHP")]:
        subset = df[df["p_throws"] == hand]
        stats = _compute_stats(subset)
        rows.append({"Split": label, **stats})
    return pd.DataFrame(rows)[SPLIT_COLS]


def split_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 2-row DataFrame: Home and Away splits.

    Uses `inning_topbot`: Bot = batter is on home team; Top = away team.
    This is valid because statcast_batter returns data for a single batter,
    so every row belongs to that batter's plate appearance.
    """
    rows = []
    for topbot, label in [("Bot", "Home"), ("Top", "Away")]:
        subset = df[df["inning_topbot"] == topbot]
        stats = _compute_stats(subset)
        rows.append({"Split": label, **stats})
    return pd.DataFrame(rows)[SPLIT_COLS]


def split_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per calendar month in the data.

    Uses `game_date`; months with zero PA are omitted.
    """
    df = df.copy()
    df["_month"] = pd.to_datetime(df["game_date"]).dt.month

    rows = []
    for month_num in sorted(df["_month"].dropna().unique()):
        subset = df[df["_month"] == month_num]
        stats = _compute_stats(subset)
        if stats["PA"] == 0:
            continue
        label = MONTH_NAMES.get(int(month_num), f"Month {int(month_num)}")
        rows.append({"Split": label, **stats})

    return pd.DataFrame(rows)[SPLIT_COLS] if rows else pd.DataFrame(columns=SPLIT_COLS)


def get_splits(df: pd.DataFrame, split_type: str) -> pd.DataFrame:
    """Dispatch to the correct split function by name.

    Parameters
    ----------
    df : raw Statcast batter DataFrame
    split_type : one of "hand", "home_away", "monthly"
    """
    dispatch = {
        "hand": split_by_hand,
        "home_away": split_home_away,
        "monthly": split_by_month,
    }
    if split_type not in dispatch:
        raise ValueError(f"Unknown split_type {split_type!r}. Choose from: {list(dispatch)}")
    return dispatch[split_type](df)


def get_trend_stats(
    mlbam_id: int,
    seasons: list[int],
    player_type: str,
    filters: SplitFilters,
    fetch_fn: Callable[[int, int], pd.DataFrame] | None = None,
    prepare_cache: dict | None = None,
    stat_key: str | None = None,
    debug_timing: bool | None = None,
    progress_cb: Callable[[int, int, int, float], None] | None = None,
) -> list[dict]:
    """Return per-season stat dicts for trend charting.

    Uses cached per-year loading/preparation by default. When *fetch_fn* is
    injected (tests), the function follows the legacy path and preserves
    prepare_cache semantics.
    """
    stat_key_cache = stat_key if stat_key is not None else "__all__"
    mode_key = _normalize_player_mode(player_type)
    filters_signature = _filters_signature_from_filters(filters)
    use_debug_timing = _DEBUG_TREND_TIMING if debug_timing is None else bool(debug_timing)

    results: list[dict] = []
    total_seasons = len(seasons)
    trend_start = time.perf_counter()
    effective_prepare_cache = prepare_cache if prepare_cache is not None else {}

    for idx, season in enumerate(seasons, start=1):
        season_start = time.perf_counter()

        if fetch_fn is None:
            stats = compute_year_stat(
                player_id=int(mlbam_id),
                year=int(season),
                mode=mode_key,
                filters_signature=filters_signature,
                stat_key=stat_key_cache,
            )

            # Keep session-state prepare cache warm for the single-season view.
            if prepare_cache is not None:
                cache_key = (int(mlbam_id), int(season), str(player_type))
                if cache_key not in prepare_cache:
                    prepare_cache[cache_key] = load_or_fetch_prepared_year_df(
                        int(mlbam_id),
                        int(season),
                        mode_key,
                    )
        else:
            raw_df = fetch_fn(mlbam_id, season)
            cache_key = (int(mlbam_id), int(season), str(player_type))
            prepared = get_prepared_df_cached(raw_df, effective_prepare_cache, cache_key)
            filtered = apply_filters(prepared, filters)
            if stat_key_cache == "__all__":
                stats = _compute_stats(filtered)
            else:
                stats = _compute_selected_stat(filtered, stat_key_cache)
            sample_sizes = get_sample_sizes(filtered)
            stats["season"] = int(season)
            stats["n_pitches"] = sample_sizes.get("N_pitches")
            stats["n_bip"] = sample_sizes.get("N_BIP")
            stats["approx_pa"] = sample_sizes.get("approx_PA")

        elapsed = time.perf_counter() - season_start
        if use_debug_timing:
            print(
                f"[trend] player={mlbam_id} season={season} stat={stat_key_cache} "
                f"elapsed={elapsed:.3f}s"
            )

        if progress_cb is not None:
            progress_cb(idx, total_seasons, int(season), elapsed)

        results.append(stats)

    if use_debug_timing:
        total_elapsed = time.perf_counter() - trend_start
        print(
            f"[trend] player={mlbam_id} seasons={total_seasons} stat={stat_key_cache} "
            f"total={total_elapsed:.3f}s"
        )

    return results
