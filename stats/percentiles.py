"""Percentile engine — rank a player's stats against league-wide distributions.

Usage
-----
    from data.fetcher import get_batting_stats
    from stats.percentiles import build_league_distributions, get_all_percentiles

    season_df = get_batting_stats(2024)
    distributions = build_league_distributions(season_df)
    percentiles = get_all_percentiles({"wOBA": 0.380, "K%": 18.0, ...}, distributions)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stats where a LOWER value is better (percentile is inverted)
LOWER_IS_BETTER: frozenset[str] = frozenset(["K%"])

# FanGraphs stores these as proportions (0–1); splits.py outputs them as
# percentages (0–100). We convert distributions to the percentage scale so
# both sources are comparable.
PROPORTION_STATS: frozenset[str] = frozenset(["K%", "BB%", "HardHit%", "Barrel%"])

CORE_STATS: list[str] = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]

# Color tiers (mirrors Baseball Savant convention).
# List of (min_percentile_inclusive, name, hex).  Evaluated top-down.
COLOR_TIERS: list[tuple[float, str, str]] = [
    (90.0, "red",    "#C0392B"),
    (70.0, "orange", "#E67E22"),
    (50.0, "yellow", "#F1C40F"),
    (30.0, "blue",   "#2980B9"),
    (0.0,  "gray",   "#95A5A6"),
]


# ---------------------------------------------------------------------------
# Distribution building
# ---------------------------------------------------------------------------

def build_league_distributions(season_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return per-stat arrays of all qualified batters' values for a season.

    Proportion stats (K%, BB%, HardHit%, Barrel%) are multiplied by 100 so
    they share the same scale as splits.py output.
    """
    distributions: dict[str, np.ndarray] = {}
    for stat in CORE_STATS:
        if stat not in season_df.columns:
            continue
        values = season_df[stat].dropna().to_numpy(dtype=float)
        if stat in PROPORTION_STATS:
            values = values * 100.0
        distributions[stat] = values
    return distributions


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------

def compute_percentile(
    value: float,
    distribution: np.ndarray,
    higher_is_better: bool = True,
) -> float:
    """Return the percentile rank (0–100) of *value* within *distribution*.

    Method: proportion of the distribution strictly beaten by *value*.
    Returns np.nan for NaN input or an empty distribution.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    arr = distribution[~np.isnan(distribution)]
    if len(arr) == 0:
        return np.nan

    if higher_is_better:
        pct = (arr < value).sum() / len(arr) * 100.0
    else:
        pct = (arr > value).sum() / len(arr) * 100.0

    return round(float(pct), 1)


def get_percentile(stat: str, value: float, distributions: dict[str, np.ndarray]) -> float:
    """Convenience wrapper: look up direction from LOWER_IS_BETTER and compute."""
    if stat not in distributions:
        return np.nan
    higher_is_better = stat not in LOWER_IS_BETTER
    return compute_percentile(value, distributions[stat], higher_is_better)


def get_all_percentiles(
    player_stats: dict[str, float | None],
    distributions: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return {stat: percentile} for every stat in *player_stats*.

    Stats missing from distributions or with None/NaN values get np.nan.
    """
    return {
        stat: get_percentile(stat, value, distributions)  # type: ignore[arg-type]
        if value is not None
        else np.nan
        for stat, value in player_stats.items()
    }


# ---------------------------------------------------------------------------
# Color tier
# ---------------------------------------------------------------------------

def get_color_tier(percentile: float) -> dict[str, str]:
    """Map a percentile (0–100) to a color name and hex string.

    Returns the gray tier for NaN or out-of-range values.
    """
    if percentile is None or (isinstance(percentile, float) and np.isnan(percentile)):
        return {"name": "gray", "hex": "#95A5A6"}

    for min_pct, name, hex_color in COLOR_TIERS:
        if percentile >= min_pct:
            return {"name": name, "hex": hex_color}

    return {"name": "gray", "hex": "#95A5A6"}


def get_all_color_tiers(percentiles: dict[str, float]) -> dict[str, dict[str, str]]:
    """Return {stat: color_tier} for a dict of percentiles."""
    return {stat: get_color_tier(pct) for stat, pct in percentiles.items()}
