"""Independent reference stat calculations for filter verification.

This module MUST NOT import from stats/, data/, or tools/.
It independently defines all constants and computation functions so that
tests can verify production code against a truly independent implementation.

The constants and formulas here are derived from the same source definitions
(Statcast documentation, FanGraphs glossary) but typed independently to
catch any drift between documentation and production code.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Constants — independently defined (do NOT copy from production)
# ---------------------------------------------------------------------------

#: Events where Statcast records a plate-appearance outcome on the final pitch.
PA_EVENTS = frozenset(
    [
        "single",
        "double",
        "triple",
        "home_run",
        "walk",
        "intent_walk",
        "hit_by_pitch",
        "strikeout",
        "strikeout_double_play",
        "field_out",
        "field_error",
        "grounded_into_double_play",
        "double_play",
        "triple_play",
        "force_out",
        "fielders_choice",
        "fielders_choice_out",
        "sac_fly",
        "sac_fly_double_play",
        "sac_bunt",
        "sac_bunt_double_play",
        "catcher_interf",
        "other_out",
    ]
)

#: Events that produce a batted ball (have launch_speed populated).
BATTED_BALL_EVENTS = frozenset(
    [
        "single",
        "double",
        "triple",
        "home_run",
        "field_out",
        "grounded_into_double_play",
        "double_play",
        "triple_play",
        "force_out",
        "fielders_choice",
        "fielders_choice_out",
        "sac_fly",
        "sac_fly_double_play",
        "sac_bunt",
        "sac_bunt_double_play",
        "other_out",
    ]
)

#: Strikeout event codes.
K_EVENTS = frozenset(["strikeout", "strikeout_double_play"])

#: Walk event codes (includes intentional walks).
BB_EVENTS = frozenset(["walk", "intent_walk"])

#: 2024 FanGraphs wOBA linear weights.
WOBA_WEIGHTS: dict[str, float] = {
    "walk": 0.690,
    "hit_by_pitch": 0.722,
    "single": 0.888,
    "double": 1.271,
    "triple": 1.616,
    "home_run": 2.101,
}

#: Sac bunt events excluded from wOBA denominator.
SAC_BUNT_EVENTS = frozenset(["sac_bunt", "sac_bunt_double_play"])

#: Hard-hit exit velocity threshold (mph).
HARD_HIT_MPH = 95.0

#: launch_speed_angle code for barrels.
BARREL_CODE = 6

# ---------------------------------------------------------------------------
# Game type codes
# ---------------------------------------------------------------------------

REGULAR_GAME_TYPES = frozenset(["R"])
POSTSEASON_GAME_TYPES = frozenset(["F", "D", "L", "W", "C", "P"])
SPRING_GAME_TYPES = frozenset(["S", "E", "A"])

# ---------------------------------------------------------------------------
# Filter functions — independent implementations
# ---------------------------------------------------------------------------


def filter_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Filter by game_type column.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``game_type`` column.
    scope : str
        One of ``"regular"``, ``"postseason"``, ``"all"``.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    if "game_type" not in df.columns:
        return df.copy()
    if scope == "regular":
        return df[df["game_type"].isin(REGULAR_GAME_TYPES)].copy()
    elif scope == "postseason":
        return df[df["game_type"].isin(POSTSEASON_GAME_TYPES)].copy()
    return df.copy()


def filter_pitcher_hand(df: pd.DataFrame, hand: str) -> pd.DataFrame:
    """Filter to pitches thrown by a pitcher of the given handedness."""
    return df[df["p_throws"] == hand].copy()


def filter_batter_hand(df: pd.DataFrame, hand: str) -> pd.DataFrame:
    """Filter to pitches where the batter has the given handedness."""
    return df[df["stand"] == hand].copy()


def filter_home_away(
    df: pd.DataFrame, side: str, pitcher_perspective: bool = False
) -> pd.DataFrame:
    """Filter to home or away plate appearances.

    For batters (pitcher_perspective=False):
        home → inning_topbot == "Bot"  (batter's team is home, batting in bottom half)
        away → inning_topbot == "Top"
    For pitchers (pitcher_perspective=True):
        home → inning_topbot == "Top"  (pitcher is home, opponents bat in top half)
        away → inning_topbot == "Bot"
    """
    if pitcher_perspective:
        topbot = "Top" if side == "home" else "Bot"
    else:
        topbot = "Bot" if side == "home" else "Top"
    return df[df["inning_topbot"] == topbot].copy()


def filter_month(df: pd.DataFrame, month: int) -> pd.DataFrame:
    """Filter to pitches from games in the given calendar month."""
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    return df[dates.dt.month == month].copy()


def filter_inning(
    df: pd.DataFrame,
    inning_min: int | None = None,
    inning_max: int | None = None,
) -> pd.DataFrame:
    """Filter to pitches within the given inning range (inclusive)."""
    result = df
    if inning_min is not None:
        result = result[result["inning"] >= inning_min]
    if inning_max is not None:
        result = result[result["inning"] <= inning_max]
    return result.copy()


def filter_count(
    df: pd.DataFrame,
    balls: int | None = None,
    strikes: int | None = None,
) -> pd.DataFrame:
    """Filter to pitches at the given ball/strike count."""
    mask = pd.Series(True, index=df.index)
    if balls is not None:
        mask &= df["balls"] == balls
    if strikes is not None:
        mask &= df["strikes"] == strikes
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Stat computation — independent implementations
# ---------------------------------------------------------------------------


def compute_pa(df: pd.DataFrame) -> int:
    """Count plate appearances from pitch-level data."""
    if "events" not in df.columns:
        return 0
    return int(df["events"].isin(PA_EVENTS).sum())


def _pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows that represent PA outcomes."""
    return df[df["events"].notna() & df["events"].isin(PA_EVENTS)]


def _batted_ball_rows(pa_df: pd.DataFrame) -> pd.DataFrame:
    """Return PA-ending batted-ball rows with non-null launch speed."""
    mask = pa_df["events"].isin(BATTED_BALL_EVENTS)
    if "launch_speed" in pa_df.columns:
        mask = mask & pa_df["launch_speed"].notna()
    return pa_df[mask]


def compute_woba(pa_df: pd.DataFrame) -> float | None:
    """Compute wOBA from PA-level events using 2024 linear weights.

    Denominator: PA minus sac bunts (approximates AB + BB + SF + HBP - IBB).
    """
    sac_bunts = pa_df["events"].isin(SAC_BUNT_EVENTS).sum()
    denominator = len(pa_df) - sac_bunts
    if denominator == 0:
        return None
    numerator = sum(
        weight * pa_df["events"].eq(event).sum()
        for event, weight in WOBA_WEIGHTS.items()
    )
    return numerator / denominator


def compute_stats(df: pd.DataFrame) -> dict[str, int | float | None]:
    """Compute batter stats dict from pitch-level DataFrame.

    Returns a dict with keys: PA, wOBA, xwOBA, K%, BB%, HardHit%, Barrel%, GB%.
    Rate stats are returned as 0-100 percentages (K%, BB%, etc.) or 0-1 scale
    (wOBA, xwOBA), matching the production ``_format_stat_value`` convention:
    - pct_1 stats: multiplied by 100 and rounded to 1 decimal
    - decimal_3 stats: rounded to 3 decimals
    """
    pa_df = _pa_rows(df)
    n_pa = len(pa_df)

    if n_pa == 0:
        return {
            "PA": 0,
            "wOBA": None,
            "xwOBA": None,
            "K%": None,
            "BB%": None,
            "HardHit%": None,
            "Barrel%": None,
            "GB%": None,
        }

    bb_df = _batted_ball_rows(pa_df)
    n_bb = len(bb_df)

    # wOBA (decimal_3 format)
    raw_woba = compute_woba(pa_df)
    woba = round(raw_woba, 3) if raw_woba is not None else None

    # xwOBA (decimal_3 format)
    xwoba = None
    xwoba_col = "estimated_woba_using_speedangle"
    if xwoba_col in pa_df.columns:
        xwoba_vals = pa_df[xwoba_col].dropna()
        if len(xwoba_vals) > 0:
            xwoba = round(float(xwoba_vals.mean()), 3)

    # K% (pct_1 format: fraction * 100, rounded to 1 decimal)
    k_count = int(pa_df["events"].isin(K_EVENTS).sum())
    k_pct = round((k_count / n_pa) * 100.0, 1)

    # BB% (pct_1 format)
    bb_count = int(pa_df["events"].isin(BB_EVENTS).sum())
    bb_pct = round((bb_count / n_pa) * 100.0, 1)

    # HardHit% (pct_1 format)
    hard_hit_pct = None
    if n_bb > 0 and "launch_speed" in bb_df.columns:
        hard_hits = (bb_df["launch_speed"] >= HARD_HIT_MPH).sum()
        hard_hit_pct = round((hard_hits / n_bb) * 100.0, 1)

    # Barrel% (pct_1 format)
    barrel_pct = None
    if n_bb > 0 and "launch_speed_angle" in bb_df.columns:
        barrels = (bb_df["launch_speed_angle"] == BARREL_CODE).sum()
        barrel_pct = round((barrels / n_bb) * 100.0, 1)

    # GB% (pct_1 format — computed as fraction then * 100)
    gb_pct = None
    if n_bb > 0 and "bb_type" in bb_df.columns:
        gb_count = (bb_df["bb_type"] == "ground_ball").sum()
        # Production computes GB% as (gb/n_bb)*100 then divides by 100 for
        # the fraction, then _format_stat_value multiplies by 100 again.
        # Net: (gb/n_bb) * 100.0, rounded to 1 decimal.
        gb_pct = round((gb_count / n_bb) * 100.0, 1)

    return {
        "PA": n_pa,
        "wOBA": woba,
        "xwOBA": xwoba,
        "K%": k_pct,
        "BB%": bb_pct,
        "HardHit%": hard_hit_pct,
        "Barrel%": barrel_pct,
        "GB%": gb_pct,
    }
