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

import pandas as pd

from stats.filters import SplitFilters, apply_filters, get_prepared_df_cached

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
    fetch_fn: Callable[[int, int], pd.DataFrame],
    prepare_cache: dict,
) -> list[dict]:
    """Return per-season stat dicts for trend charting.

    For each season in *seasons*, fetches raw Statcast data via *fetch_fn*,
    prepares it (shared with the single-season view via *prepare_cache*),
    applies *filters*, then computes all 6 core stats.

    Parameters
    ----------
    mlbam_id : int
        MLBAM player ID.
    seasons : list[int]
        Ordered list of season years to include (e.g. [2019, 2020, ..., 2025]).
    player_type : str
        Player type string (currently always "Batter"); used as the third
        component of the prepare_cache key to match the existing convention.
    filters : SplitFilters
        Active filter configuration. Applied identically within each season.
    fetch_fn : Callable[[int, int], pd.DataFrame]
        Callable with signature ``(mlbam_id, season) -> DataFrame``.
        Production code passes ``get_statcast_batter``; tests pass a stub.
        This injection point also serves as the extension seam for pre-2015
        data sources — any adapter that satisfies the signature and produces a
        DataFrame with compatible columns will work transparently.
    prepare_cache : dict
        The session-state memoisation dict keyed by ``(player_id, season, type)``.
        Shared with the single-season view so prepared DataFrames are reused
        when both views are active in the same session.

    Returns
    -------
    list[dict]
        One dict per season in the same order as *seasons*. Each dict has:
        ``{"season": int, "PA": int, "wOBA": float|None, "xwOBA": float|None,
        "K%": float|None, "BB%": float|None, "HardHit%": float|None,
        "Barrel%": float|None}``.
        Seasons with no data (empty fetch) produce PA=0 and all stats None.
    """
    results: list[dict] = []
    for season in seasons:
        raw_df = fetch_fn(mlbam_id, season)
        cache_key = (int(mlbam_id), int(season), str(player_type))
        prepared = get_prepared_df_cached(raw_df, prepare_cache, cache_key)
        filtered = apply_filters(prepared, filters)
        stats = _compute_stats(filtered)
        sample_sizes = get_sample_sizes(filtered)
        stats["season"] = season
        stats["n_pitches"] = sample_sizes.get("N_pitches")
        stats["n_bip"] = sample_sizes.get("N_BIP")
        stats["approx_pa"] = sample_sizes.get("approx_PA")
        results.append(stats)
    return results
