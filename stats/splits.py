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

# Pitch-level descriptions used for pitcher-only metrics.
CSW_DESCRIPTIONS = frozenset(
    ["called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"]
)
WHIFF_DESCRIPTIONS = frozenset(["swinging_strike", "swinging_strike_blocked"])
SWING_DESCRIPTIONS = frozenset(
    [
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip",
        "foul_bunt",
        "hit_into_play",
        "hit_into_play_no_out",
        "hit_into_play_score",
        "swinging_pitchout",
        "foul_pitchout",
        "missed_bunt",
    ]
)
# Per plan, FirstStrike% uses the same strike-classification set as CSW%.
FIRST_STRIKE_DESCRIPTIONS = CSW_DESCRIPTIONS

# Pitch type display labels used in the pitcher arsenal table.
PITCH_TYPE_NAMES: dict[str, str] = {
    "FA": "Fastball",
    "FF": "Four-Seam Fastball",
    "FT": "Two-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "FS": "Splitter",
    "FO": "Forkball",
    "CH": "Changeup",
    "SC": "Screwball",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "KN": "Knuckleball",
    "EP": "Eephus",
}
MIN_ARSENAL_PITCHES = 25

# Ordered output columns
SPLIT_COLS = ["Split", "PA", "wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%", "GB%"]
PITCHER_SPLIT_COLS = SPLIT_COLS + ["K-BB%", "CSW%", "Whiff%", "FirstStrike%"]
ARSENAL_COLS = ["Pitch", "N", "Usage%", "Velo", "Spin", "CSW%", "Whiff%"]

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


def _compute_gb_rate(_pa: pd.DataFrame, bb_df: pd.DataFrame, _n_pa: int) -> float | None:
    """Return ground-ball rate as a percent of batted-ball events (0-100)."""
    n_bb_events = len(bb_df)
    if n_bb_events == 0 or "bb_type" not in bb_df.columns:
        return None
    gb = (bb_df["bb_type"] == "ground_ball").sum()
    return (gb / n_bb_events) * 100.0


def _compute_gb_rate_fraction(pa: pd.DataFrame, bb_df: pd.DataFrame, n_pa: int) -> float | None:
    """Return ground-ball rate as a 0-1 fraction for pct formatter reuse."""
    gb_rate = _compute_gb_rate(pa, bb_df, n_pa)
    if gb_rate is None:
        return None
    return gb_rate / 100.0


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
    "GB%": StatSpec(
        key="GB%",
        label="GB%",
        required_cols=["events", "launch_speed"],
        formatter="pct_1",
        compute_fn=_compute_gb_rate_fraction,
        glossary_key=None,
    ),
}


def _format_stat_value(value: float | None, formatter: Literal["pct_1", "decimal_3"]) -> float | None:
    if value is None:
        return None
    if formatter == "pct_1":
        return round(value * 100, 1)
    return round(value, 3)


def get_stat_registry(player_type: str) -> dict[str, StatSpec]:
    """Return the active stat registry for the requested player type."""
    if str(player_type).strip().lower() == "pitcher":
        return STAT_REGISTRY
    return STAT_REGISTRY


def _compute_stats(df: pd.DataFrame, player_type: str = "Batter") -> dict:
    """Compute all 6 core stats from a raw Statcast subset.

    Accepts all pitches for the subset; filters internally to PA events.
    Returns a dict keyed by stat name (None when not computable).
    """
    pa = _pa_events(df)
    n_pa = len(pa)

    stat_registry = get_stat_registry(player_type)

    if n_pa == 0:
        return {"PA": 0, **{key: None for key in stat_registry}}

    bb_df = _batted_ball_events(pa)
    stats: dict[str, float | int | None] = {"PA": n_pa}

    for key, spec in stat_registry.items():
        raw_value = spec.compute_fn(pa, bb_df, n_pa)
        stats[key] = _format_stat_value(raw_value, spec.formatter)

    return stats


def _compute_pitch_level_stats(df_all_pitches: pd.DataFrame) -> dict[str, float | None]:
    """Compute pitcher-only pitch-level metrics from all pitch rows."""
    if df_all_pitches.empty or "description" not in df_all_pitches.columns:
        return {"CSW%": None, "Whiff%": None, "FirstStrike%": None}

    descriptions = df_all_pitches["description"]
    n_pitches = len(df_all_pitches)
    if n_pitches == 0:
        return {"CSW%": None, "Whiff%": None, "FirstStrike%": None}

    csw = descriptions.isin(CSW_DESCRIPTIONS).sum()
    csw_rate = round((csw / n_pitches) * 100.0, 1)

    swings = descriptions.isin(SWING_DESCRIPTIONS).sum()
    whiffs = descriptions.isin(WHIFF_DESCRIPTIONS).sum()
    whiff_rate = round((whiffs / swings) * 100.0, 1) if swings > 0 else None

    if "balls" not in df_all_pitches.columns or "strikes" not in df_all_pitches.columns:
        first_strike_rate = None
    else:
        first_pitch = df_all_pitches[
            (df_all_pitches["balls"] == 0) & (df_all_pitches["strikes"] == 0)
        ]
        if first_pitch.empty:
            first_strike_rate = None
        else:
            first_strikes = first_pitch["description"].isin(FIRST_STRIKE_DESCRIPTIONS).sum()
            first_strike_rate = round((first_strikes / len(first_pitch)) * 100.0, 1)

    return {"CSW%": csw_rate, "Whiff%": whiff_rate, "FirstStrike%": first_strike_rate}


def _compute_all_pitcher_stats(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Return the union of PA-level and pitch-level pitcher metrics."""
    base = _compute_stats(df, player_type="Pitcher")
    k_pct = base.get("K%")
    bb_pct = base.get("BB%")
    k_minus_bb = None
    if k_pct is not None and bb_pct is not None:
        k_minus_bb = round(float(k_pct) - float(bb_pct), 1)
    return {**base, "K-BB%": k_minus_bb, **_compute_pitch_level_stats(df)}


def compute_pitch_arsenal(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-pitch-type arsenal metrics for a pitcher-season data subset.

    Requires ``pitch_type`` and ``description`` columns. Returns an empty
    DataFrame when either column is unavailable or no pitch types meet the
    minimum sample threshold.
    """
    if "pitch_type" not in df.columns or "description" not in df.columns:
        return pd.DataFrame(columns=ARSENAL_COLS)

    work = df[df["pitch_type"].notna() & (df["pitch_type"] != "UN")].copy()
    if work.empty:
        return pd.DataFrame(columns=ARSENAL_COLS)

    rows: list[dict[str, float | int | str | None]] = []
    for pitch_type, grp in work.groupby("pitch_type", sort=False):
        n_pitches = int(len(grp))
        if n_pitches < MIN_ARSENAL_PITCHES:
            continue

        descriptions = grp["description"]
        csw = int(descriptions.isin(CSW_DESCRIPTIONS).sum())
        csw_pct = round((csw / n_pitches) * 100.0, 1)

        swings = int(descriptions.isin(SWING_DESCRIPTIONS).sum())
        whiffs = int(descriptions.isin(WHIFF_DESCRIPTIONS).sum())
        whiff_pct: float | None = None
        if swings > 0:
            whiff_pct = round((whiffs / swings) * 100.0, 1)

        velo = grp["release_speed"].mean() if "release_speed" in grp.columns else float("nan")
        spin = grp["release_spin_rate"].mean() if "release_spin_rate" in grp.columns else float("nan")

        rows.append(
            {
                "pitch_type": str(pitch_type),
                "Pitch": PITCH_TYPE_NAMES.get(str(pitch_type), str(pitch_type)),
                "N": n_pitches,
                "Velo": velo,
                "Spin": spin,
                "CSW%": csw_pct,
                "Whiff%": whiff_pct,
            }
        )

    if not rows:
        return pd.DataFrame(columns=ARSENAL_COLS)

    arsenal = pd.DataFrame(rows)
    total = float(arsenal["N"].sum())
    arsenal["Usage%"] = round((arsenal["N"] / total) * 100.0, 1)
    arsenal = arsenal.sort_values(by="Usage%", ascending=False, kind="stable")
    return arsenal[ARSENAL_COLS].reset_index(drop=True)


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
        stats = _compute_stats(subset, player_type="Batter")
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
        stats = _compute_stats(subset, player_type="Batter")
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
        stats = _compute_stats(subset, player_type="Batter")
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


def split_by_batter_hand(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 2-row DataFrame: vs LHB and vs RHB splits."""
    rows = []
    for hand, label in [("L", "vs LHB"), ("R", "vs RHB")]:
        subset = df[df["stand"] == hand]
        stats = _compute_all_pitcher_stats(subset)
        rows.append({"Split": label, **stats})
    return pd.DataFrame(rows)[PITCHER_SPLIT_COLS]


def split_home_away_pitcher(df: pd.DataFrame) -> pd.DataFrame:
    """Return pitcher home/away splits with inverted top/bottom mapping."""
    rows = []
    for topbot, label in [("Top", "Home"), ("Bot", "Away")]:
        subset = df[df["inning_topbot"] == topbot]
        stats = _compute_all_pitcher_stats(subset)
        rows.append({"Split": label, **stats})
    return pd.DataFrame(rows)[PITCHER_SPLIT_COLS]


def split_by_month_pitcher(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per month using pitcher-oriented stat computation."""
    df = df.copy()
    df["_month"] = pd.to_datetime(df["game_date"]).dt.month

    rows = []
    for month_num in sorted(df["_month"].dropna().unique()):
        subset = df[df["_month"] == month_num]
        stats = _compute_all_pitcher_stats(subset)
        if stats["PA"] == 0:
            continue
        label = MONTH_NAMES.get(int(month_num), f"Month {int(month_num)}")
        rows.append({"Split": label, **stats})

    return pd.DataFrame(rows)[PITCHER_SPLIT_COLS] if rows else pd.DataFrame(columns=PITCHER_SPLIT_COLS)


def get_pitcher_splits(df: pd.DataFrame, split_type: str) -> pd.DataFrame:
    """Dispatch to the correct pitcher split function by name."""
    dispatch = {
        "hand": split_by_batter_hand,
        "home_away": split_home_away_pitcher,
        "monthly": split_by_month_pitcher,
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
        filtered = apply_filters(prepared, filters, pitcher_perspective=(player_type == "Pitcher"))
        if str(player_type).strip().lower() == "pitcher":
            stats = _compute_all_pitcher_stats(filtered)
        else:
            stats = _compute_stats(filtered, player_type=player_type)
        sample_sizes = get_sample_sizes(filtered)
        stats["season"] = season
        stats["n_pitches"] = sample_sizes.get("N_pitches")
        stats["n_bip"] = sample_sizes.get("N_BIP")
        stats["approx_pa"] = sample_sizes.get("approx_PA")
        results.append(stats)
    return results
