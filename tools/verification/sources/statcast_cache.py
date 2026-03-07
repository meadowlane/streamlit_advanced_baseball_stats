"""Shared Statcast cache for a single verification run."""

from __future__ import annotations

import pandas as pd

from data.fetcher import _fetch_statcast_batter, _fetch_statcast_pitcher

_BATTER_CACHE: dict[tuple[int, int], pd.DataFrame] = {}
_PITCHER_CACHE: dict[tuple[int, int], pd.DataFrame] = {}


def clear_verification_statcast_cache() -> None:
    """Drop all cached Statcast DataFrames for the current verification run."""
    _BATTER_CACHE.clear()
    _PITCHER_CACHE.clear()


def get_cached_batter_statcast(mlbam_id: int, year: int) -> pd.DataFrame:
    """Return cached raw Statcast data for one batter season."""
    key = (int(mlbam_id), int(year))
    if key not in _BATTER_CACHE:
        _BATTER_CACHE[key] = _fetch_statcast_batter(*key)
    return _BATTER_CACHE[key]


def get_cached_pitcher_statcast(mlbam_id: int, year: int) -> pd.DataFrame:
    """Return cached raw Statcast data for one pitcher season."""
    key = (int(mlbam_id), int(year))
    if key not in _PITCHER_CACHE:
        _PITCHER_CACHE[key] = _fetch_statcast_pitcher(*key)
    return _PITCHER_CACHE[key]
