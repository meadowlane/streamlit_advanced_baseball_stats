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

import pandas as pd

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

def _pa_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows that represent plate-appearance outcomes."""
    return df[df["events"].notna() & df["events"].isin(PA_EVENTS)].copy()


def _compute_stats(df: pd.DataFrame) -> dict:
    """Compute all 6 core stats from a raw Statcast subset.

    Accepts all pitches for the subset; filters internally to PA events.
    Returns a dict keyed by stat name (None when not computable).
    """
    pa = _pa_events(df)
    n_pa = len(pa)

    if n_pa == 0:
        return {
            "PA": 0,
            "K%": None, "BB%": None,
            "HardHit%": None, "Barrel%": None,
            "xwOBA": None, "wOBA": None,
        }

    # --- K% and BB% ---
    k_events = {"strikeout", "strikeout_double_play"}
    n_k = pa["events"].isin(k_events).sum()
    n_bb = (pa["events"] == "walk").sum()
    k_pct = n_k / n_pa
    bb_pct = n_bb / n_pa

    # --- Batted balls (launch_speed present) ---
    bb_mask = pa["events"].isin(BATTED_BALL_EVENTS) & pa["launch_speed"].notna()
    bb_df = pa[bb_mask]
    n_bb_events = len(bb_df)

    # --- HardHit% ---
    if n_bb_events > 0:
        hard_hit_pct = (bb_df["launch_speed"] >= HARD_HIT_MPH).sum() / n_bb_events
    else:
        hard_hit_pct = None

    # --- Barrel% (launch_speed_angle == 6) ---
    if n_bb_events > 0 and "launch_speed_angle" in bb_df.columns:
        barrel_pct = (bb_df["launch_speed_angle"] == BARREL_CODE).sum() / n_bb_events
    else:
        barrel_pct = None

    # --- xwOBA (mean over batted-ball events with estimated value) ---
    xwoba_col = "estimated_woba_using_speedangle"
    if xwoba_col in pa.columns:
        xwoba_vals = pa[xwoba_col].dropna()
        xwoba = float(xwoba_vals.mean()) if len(xwoba_vals) > 0 else None
    else:
        xwoba = None

    # --- wOBA from linear weights ---
    woba = _compute_woba(pa)

    def _pct(val: float | None, decimals: int = 1) -> float | None:
        return round(val * 100, decimals) if val is not None else None

    def _round(val: float | None, decimals: int = 3) -> float | None:
        return round(val, decimals) if val is not None else None

    return {
        "PA": n_pa,
        "K%": _pct(k_pct),
        "BB%": _pct(bb_pct),
        "HardHit%": _pct(hard_hit_pct),
        "Barrel%": _pct(barrel_pct),
        "xwOBA": _round(xwoba),
        "wOBA": _round(woba),
    }


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
