"""FanGraphs source adapter — extracts stats from the shared FanGraphs season pull.

This adapter shares the season-level fetch path with the app source so one
yearly table can be reused across players during verification. It still
validates the app's field-mapping and post-processing logic because it
extracts and normalizes the row independently.

Scale convention
----------------
FanGraphs stores K%, BB%, HardHit%, Barrel%, GB%, FB% as 0-1 fractions.
This adapter converts them to 0-100 before returning, matching the app's
display convention.  All :mod:`tools.verification.normalization` conversions
are applied here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
from data.fetcher import _fetch_batting_stats, _fetch_pitching_stats  # noqa: E402

from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError  # noqa: E402
from tools.verification.normalization import (  # noqa: E402
    normalize_float,
    normalize_count,
    normalize_ip,
    normalize_avg,
)

# FG 0-1 fraction columns → multiply by 100 to get 0-100 scale
_FG_FRACTION_COLS = frozenset(
    ["K%", "BB%", "HardHit%", "Barrel%", "GB%", "FB%", "CSW%", "Whiff%", "F-Strike%"]
)

# Column aliases: our canonical key → FG column names to try
_BATTER_COL_MAP: dict[str, list[str]] = {
    "PA": ["PA"],
    "H": ["H"],
    "HR": ["HR"],
    "BB": ["BB"],
    "SO": ["SO"],
    "HBP": ["HBP"],
    "AVG": ["AVG"],
    "OBP": ["OBP"],
    "SLG": ["SLG"],
    "OPS": ["OPS"],
    "wOBA": ["wOBA"],
    "xwOBA": ["xwOBA"],
    "wRC+": ["wRC+", "wRC_plus", "wrc_plus"],
    "K%": ["K%"],
    "BB%": ["BB%"],
    "HardHit%": ["HardHit%"],
    "Barrel%": ["Barrel%"],
    "GB%": ["GB%"],
    "FB%": ["FB%"],
}

_PITCHER_COL_MAP: dict[str, list[str]] = {
    "W": ["W"],
    "L": ["L"],
    "SO": ["SO"],
    "BB": ["BB"],
    "HR": ["HR"],
    "IP": ["IP"],
    "ERA": ["ERA"],
    "FIP": ["FIP"],
    "xFIP": ["xFIP"],
    "SIERA": ["SIERA"],
    "xERA": ["xERA"],
    "wOBA": ["wOBA"],
    "xwOBA": ["xwOBA"],
    "K%": ["K%"],
    "BB%": ["BB%"],
    "K-BB%": ["K-BB%"],
    "GB%": ["GB%"],
    "FB%": ["FB%"],
    "HardHit%": ["HardHit%"],
    "Barrel%": ["Barrel%"],
    "FBv": ["FBv"],
    "Stuff+": ["Stuff+"],
    "Location+": ["Location+"],
    "Pitching+": ["Pitching+"],
    "CSW%": ["CSW%"],
    "Whiff%": ["Whiff%"],
    "FirstStrike%": ["F-Strike%"],
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        actual = lower_map.get(cand.lower())
        if actual is not None:
            return str(actual)
    return None


def _extract_row(
    df: pd.DataFrame,
    player: PlayerIdentity,
) -> pd.Series | None:
    """Find the player row, preferring fg_id match then name match."""
    if player.fg_id is not None and "IDfg" in df.columns:
        mask = pd.to_numeric(df["IDfg"], errors="coerce") == player.fg_id
        matches = df[mask]
        if not matches.empty:
            return matches.iloc[0]

    # Name fallback (case-insensitive)
    if "Name" in df.columns:
        name_lower = player.name.lower()
        mask = df["Name"].str.lower() == name_lower
        matches = df[mask]
        if not matches.empty:
            return matches.iloc[0]

    return None


def _build_stat_dict(
    row: pd.Series,
    col_map: dict[str, list[str]],
    df: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for our_key, candidates in col_map.items():
        col = _find_col(df, candidates)
        if col is None:
            continue
        raw = row[col]
        try:
            if pd.isna(raw):
                continue
        except (TypeError, ValueError):
            pass

        if our_key in ("W", "L", "SO", "BB", "HR", "HBP", "PA", "H"):
            val: Any = normalize_count(raw)
        elif our_key == "IP":
            val = normalize_ip(raw)
        elif our_key in ("AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA",
                          "ERA", "FIP", "xFIP", "SIERA", "xERA"):
            val = normalize_avg(raw)
        elif our_key in _FG_FRACTION_COLS or our_key in ("K%", "BB%", "K-BB%",
                                                           "HardHit%", "Barrel%",
                                                           "GB%", "FB%", "CSW%",
                                                           "Whiff%", "FirstStrike%"):
            f = normalize_float(raw)
            if f is not None:
                # FG stores these as 0-1 fractions
                val = round(f * 100.0, 4) if f < 1.5 else f
            else:
                val = None
        elif our_key in ("wRC+", "Stuff+", "Location+", "Pitching+"):
            f = normalize_float(raw)
            val = None if f is None else int(round(f))
        else:
            val = normalize_float(raw)

        if val is not None:
            result[our_key] = val

    return result


class FanGraphsSource(BaseSource):
    """Fetches season stats from FanGraphs via pybaseball."""

    @property
    def source_name(self) -> str:
        return "fangraphs"

    def get_batter_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        if game_type != "regular":
            raise SourceError(
                f"FanGraphsSource only supports 'regular' scope; "
                f"requested {game_type!r} — SKIP"
            )
        if offline:
            raise SourceError("FanGraphsSource: offline mode requires fixture — no data available")

        df = _fetch_batting_stats(year, min_pa=10)
        if df.empty:
            raise SourceError(f"FanGraphs returned empty batting data for {year}")

        row = _extract_row(df, player)
        if row is None:
            raise SourceError(
                f"Player '{player.name}' (fg_id={player.fg_id}) not found in FG batting {year}"
            )

        return _build_stat_dict(row, _BATTER_COL_MAP, df)

    def get_pitcher_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        if game_type != "regular":
            raise SourceError(
                f"FanGraphsSource only supports 'regular' scope; "
                f"requested {game_type!r} — SKIP"
            )
        if offline:
            raise SourceError("FanGraphsSource: offline mode requires fixture — no data available")

        df = _fetch_pitching_stats(year, min_ip=5)
        if df.empty:
            raise SourceError(f"FanGraphs returned empty pitching data for {year}")

        row = _extract_row(df, player)
        if row is None:
            raise SourceError(
                f"Pitcher '{player.name}' (fg_id={player.fg_id}) not found in FG pitching {year}"
            )

        return _build_stat_dict(row, _PITCHER_COL_MAP, df)
