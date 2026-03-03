"""Statcast source adapter — consistency re-fetch and re-compute.

This adapter re-fetches raw Statcast data via pybaseball and recomputes
Statcast-derived stats using the **same functions** as the app
(``stats.splits._compute_stats``, ``_compute_all_pitcher_stats``).

Design note — NOT independent
------------------------------
Because this adapter uses the same computation code as the app, it is **not**
an independent verification source (``is_independent = False``).  It validates:

1. The raw Statcast data round-trips correctly (no filtering bugs).
2. The computation results are stable between calls.
3. The FanGraphs wOBA column matches the in-app Statcast computation.

Any discrepancy between AppSource and StatcastSource for the *same* player
indicates a data-ordering or caching issue, not a formula bug.  The comparison
engine excludes StatcastSource from PASS/FAIL verdict counts; it is shown in
the report as an informational cross-check only.

Game type / regular-season filtering
-------------------------------------
When ``game_type="regular"`` (the default), the Statcast DataFrame is filtered
to rows where ``game_type == 'R'`` **before** stats are computed.  This ensures
PA counts match FanGraphs and the MLB Stats API (both regular-season only).
The raw fetch covers ``{year}-03-01`` through ``{year}-11-30`` to capture the
full regular season plus playoffs; filtering happens in memory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pybaseball as pb  # type: ignore[import-untyped]  # noqa: E402

from data.fetcher import _fetch_statcast_batter, _fetch_statcast_pitcher  # noqa: E402
from stats.splits import (  # noqa: E402
    _compute_stats,
    _compute_all_pitcher_stats,
    WOBA_WEIGHTS,
    PA_EVENTS,
    BATTED_BALL_EVENTS,
    K_EVENTS,
    CSW_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
    SWING_DESCRIPTIONS,
    FIRST_STRIKE_DESCRIPTIONS,
    HARD_HIT_MPH,
    BARREL_CODE,
)

from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError  # noqa: E402

# Fastball pitch type codes used for FBv calculation
_FB_CODES = frozenset(["FF", "FA", "FT", "SI"])


def _filter_regular_season(df: Any) -> Any:
    """Filter a Statcast DataFrame to regular-season rows (game_type == 'R').

    If the DataFrame has no ``game_type`` column (older Statcast schemas), the
    unfiltered DataFrame is returned with a note — callers should be aware that
    PA may include non-regular-season games in that case.
    """
    if df is None or df.empty:
        return df
    if "game_type" not in df.columns:
        # Column absent — cannot filter; return as-is
        return df
    return df[df["game_type"] == "R"].copy()


class StatcastSource(BaseSource):
    """Re-fetches Statcast data and recomputes all Statcast-derived stats.

    This adapter is **not independent** (``is_independent = False``).  See
    module docstring for details.
    """

    #: Not independent — reuses the same code path as AppSource.
    is_independent: bool = False

    @property
    def source_name(self) -> str:
        return "statcast"

    def get_batter_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        offline: bool = False,
        game_type: str = "regular",
    ) -> dict[str, Any]:
        if offline:
            raise SourceError("StatcastSource: offline mode requires fixture")

        try:
            df = _fetch_statcast_batter(player.mlbam_id, year)
        except Exception as exc:
            raise SourceError(f"Statcast batter fetch failed for {player.mlbam_id}/{year}: {exc}") from exc

        if df.empty:
            raise SourceError(f"No Statcast data for batter {player.mlbam_id} in {year}")

        if game_type == "regular":
            df = _filter_regular_season(df)
            if df.empty:
                raise SourceError(
                    f"No regular-season Statcast data for batter {player.mlbam_id} in {year}"
                )

        computed = _compute_stats(df, player_type="Batter")

        # Also expose FBv from raw data (mean release_speed on FB pitch types)
        result: dict[str, Any] = dict(computed)
        fbv = self._compute_fbv(df)
        if fbv is not None:
            result["FBv"] = fbv

        return result

    def get_pitcher_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        offline: bool = False,
        game_type: str = "regular",
    ) -> dict[str, Any]:
        if offline:
            raise SourceError("StatcastSource: offline mode requires fixture")

        try:
            df = _fetch_statcast_pitcher(player.mlbam_id, year)
        except Exception as exc:
            raise SourceError(f"Statcast pitcher fetch failed for {player.mlbam_id}/{year}: {exc}") from exc

        if df.empty:
            raise SourceError(f"No Statcast data for pitcher {player.mlbam_id} in {year}")

        if game_type == "regular":
            df = _filter_regular_season(df)
            if df.empty:
                raise SourceError(
                    f"No regular-season Statcast data for pitcher {player.mlbam_id} in {year}"
                )

        computed = _compute_all_pitcher_stats(df)
        result: dict[str, Any] = dict(computed)

        fbv = self._compute_fbv(df)
        if fbv is not None:
            result["FBv"] = fbv

        return result

    @staticmethod
    def _compute_fbv(df: Any) -> float | None:
        """Return mean fastball velocity from pitch-level Statcast data."""
        try:
            import pandas as pd
            if "pitch_type" not in df.columns or "release_speed" not in df.columns:
                return None
            fb_df = df[df["pitch_type"].isin(_FB_CODES) & df["release_speed"].notna()]
            if fb_df.empty:
                return None
            return round(float(fb_df["release_speed"].mean()), 1)
        except Exception:
            return None
