"""App source adapter — wraps our own data pipeline.

Calls the non-cached fetch/compute functions directly (bypassing Streamlit
``@st.cache_data`` wrappers) so they can run in a pure-Python test context.

This adapter is the *reference* against which all external sources are compared.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make sure the project root is on the path so `data` and `stats` are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.fetcher import (  # noqa: E402
    _fetch_batting_stats,
    _fetch_pitching_stats,
    get_player_row,
)
from stats.splits import _compute_stats, _compute_all_pitcher_stats  # noqa: E402

from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError  # noqa: E402
from tools.verification.sources.statcast_cache import (  # noqa: E402
    get_cached_batter_statcast,
    get_cached_pitcher_statcast,
)
from tools.verification.game_scope import (  # noqa: E402
    filter_by_scope,
    pa_breakdown_by_game_type,
    SOURCE_SCOPE_SUPPORT,
)


# Pitcher FG passthrough columns we want to expose alongside the Statcast stats.
_FG_PITCHER_PASSTHROUGHS = [
    "ERA",
    "FIP",
    "xFIP",
    "SIERA",
    "xERA",
    "W",
    "L",
    "IP",
    "wOBA",
    "xwOBA",
    "K%",
    "BB%",
    "GB%",
    "FB%",
    "HardHit%",
    "Barrel%",
    "FBv",
    "Stuff+",
    "Location+",
    "Pitching+",
    "CSW%",
    "Whiff%",
    "F-Strike%",
]

# Batter FG passthrough columns
_FG_BATTER_PASSTHROUGHS = [
    "wRC+",
    "wOBA",
    "xwOBA",
    "K%",
    "BB%",
    "HardHit%",
    "Barrel%",
    "GB%",
    "FB%",
    "AVG",
    "OBP",
    "SLG",
    "OPS",
    "PA",
    "H",
    "HR",
    "BB",
    "SO",
    "HBP",
]


def _safe_get(row: Any, col: str) -> Any:
    """Safely extract a column from a pandas Series or dict-like."""
    try:
        val = row[col]
        # pandas NA → None
        import pandas as pd  # local import to avoid top-level pandas dependency issues

        if pd.isna(val):
            return None
        return val
    except (KeyError, TypeError, ValueError):
        return None


class AppSource(BaseSource):
    """Retrieves stats by running the app's own data pipeline.

    This is the "ground truth" — values we compare all external sources against.
    It calls the same functions the Streamlit app uses, but via the non-cached
    private variants so no Streamlit runtime is required.
    """

    @property
    def source_name(self) -> str:
        return "app"

    @property
    def supported_scopes(self) -> frozenset[str]:
        return SOURCE_SCOPE_SUPPORT["app"]

    def get_batter_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        if offline:
            raise SourceError(
                "AppSource does not support --offline mode: it always calls the live pipeline. "
                "Use fixtures for offline runs."
            )

        # 1. FanGraphs season data (for FG passthroughs + player identification)
        fg_df = _fetch_batting_stats(year, min_pa=10)
        if fg_df.empty:
            raise SourceError(f"No FanGraphs batting data for {year}")

        player_row = get_player_row(fg_df, player.name)
        if player_row is None:
            raise SourceError(
                f"Player '{player.name}' not found in FanGraphs batting data for {year}"
            )

        # 2. Statcast pitch-level data (full season, all game types)
        sc_df_full = get_cached_batter_statcast(player.mlbam_id, year)

        # Record PA breakdown by game_type BEFORE filtering — used for diagnostics
        _pa_breakdown = pa_breakdown_by_game_type(sc_df_full)

        # Filter to the requested scope
        sc_df = filter_by_scope(sc_df_full, game_type)

        # Compute stats from scope-filtered data
        computed = _compute_stats(sc_df, player_type="Batter")

        # 3. Merge: Statcast-computed stats override FG where both exist, except for
        #    pure FG passthroughs (wRC+, AVG, OBP, SLG, OPS).
        result: dict[str, Any] = {}
        for col in _FG_BATTER_PASSTHROUGHS:
            val = _safe_get(player_row, col)
            if val is not None:
                result[col] = val

        # Statcast-computed stats win for the 6 core metrics and K-BB%
        for key, val in computed.items():
            result[key] = val

        # Rename F-Strike% if present
        if "F-Strike%" in result:
            result["FirstStrike%"] = result.pop("F-Strike%")

        # Attach diagnostic metadata (underscore prefix → not treated as a stat)
        result["_pa_by_game_type"] = _pa_breakdown

        return result

    def get_pitcher_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        if offline:
            raise SourceError("AppSource does not support --offline mode.")

        # 1. FanGraphs pitching season data
        fg_df = _fetch_pitching_stats(year, min_ip=5)
        if fg_df.empty:
            raise SourceError(f"No FanGraphs pitching data for {year}")

        player_row = get_player_row(fg_df, player.name)
        if player_row is None:
            raise SourceError(
                f"Pitcher '{player.name}' not found in FanGraphs pitching data for {year}"
            )

        # 2. Statcast pitch-level data (full season, all game types)
        sc_df_full = get_cached_pitcher_statcast(player.mlbam_id, year)

        # Filter to the requested scope before computing
        sc_df = filter_by_scope(sc_df_full, game_type)

        computed = _compute_all_pitcher_stats(sc_df)

        # 3. FG passthroughs (ERA, FIP, xFIP, SIERA, xERA, FBv, Stuff+ etc.)
        result: dict[str, Any] = {}
        for col in _FG_PITCHER_PASSTHROUGHS:
            val = _safe_get(player_row, col)
            if val is not None:
                result[col] = val

        # Statcast-computed values override for pitch-level stats
        for key, val in computed.items():
            if val is not None:
                result[key] = val

        # Rename F-Strike% if present
        if "F-Strike%" in result:
            result["FirstStrike%"] = result.pop("F-Strike%")

        return result
