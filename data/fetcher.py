"""Data fetching layer — wraps pybaseball with Streamlit caching."""

import streamlit as st
import pybaseball as pb
import pandas as pd
from pybaseball import playerid_reverse_lookup

# Cache TTLs
_TTL_STATS = 3600    # 1 h — season stats refresh once an hour during the season
_TTL_IDS   = 86400  # 24 h — player ID mappings are stable across a season

# FanGraphs column names for our 6 core stats
CORE_STAT_COLS = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]

# Statcast columns used for split calculations in Phase 2
STATCAST_KEEP_COLS = [
    "game_date",
    "batter",
    "pitcher",
    "p_throws",          # pitcher handedness: L or R
    "home_team",
    "away_team",
    "inning_topbot",     # Top = visiting batter, Bot = home batter
    "launch_speed",      # exit velocity
    "launch_angle",
    "launch_speed_angle",    # 6 = barrel (Baseball Savant classification)
    "estimated_woba_using_speedangle",  # xwOBA per event
    "events",            # plate appearance result
    "description",
    "stand",             # batter handedness
]


# ---------------------------------------------------------------------------
# Pure fetch functions (no cache — called by cached wrappers; testable directly)
# ---------------------------------------------------------------------------

def _fetch_batting_stats(season: int, min_pa: int = 50) -> pd.DataFrame:
    """Return FanGraphs season batting stats for all qualified batters."""
    pb.cache.enable()
    df = pb.batting_stats(season, qual=min_pa)
    return df


def _fetch_statcast_batter(player_mlbam_id: int, season: int) -> pd.DataFrame:
    """Return raw Statcast pitch-level events for one batter for a full season."""
    pb.cache.enable()
    start = f"{season}-03-01"
    end = f"{season}-11-30"
    df = pb.statcast_batter(start, end, player_mlbam_id)
    # Keep only columns that exist in the response (Statcast schema can vary by year)
    cols = [c for c in STATCAST_KEEP_COLS if c in df.columns]
    return df[cols].copy()


def _lookup_player(last_name: str, first_name: str = "") -> pd.DataFrame:
    """Return player ID table from pybaseball name lookup."""
    return pb.playerid_lookup(last_name, first_name)


# ---------------------------------------------------------------------------
# Cached wrappers (used by the Streamlit app)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_TTL_STATS, show_spinner=False)
def get_batting_stats(season: int, min_pa: int = 50) -> pd.DataFrame:
    """Cached FanGraphs season batting stats."""
    return _fetch_batting_stats(season, min_pa)


@st.cache_data(ttl=_TTL_STATS, show_spinner=False)
def get_statcast_batter(player_mlbam_id: int, season: int) -> pd.DataFrame:
    """Cached raw Statcast events for one batter."""
    return _fetch_statcast_batter(player_mlbam_id, season)


@st.cache_data(ttl=_TTL_IDS, show_spinner=False)
def lookup_player(last_name: str, first_name: str = "") -> pd.DataFrame:
    """Cached player ID lookup. Results cached for 24 h (roster data is stable)."""
    return _lookup_player(last_name, first_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_player_row(batting_df: pd.DataFrame, name: str) -> pd.Series | None:
    """Return the first matching row for a player name (case-insensitive)."""
    mask = batting_df["Name"].str.lower() == name.lower()
    matches = batting_df[mask]
    return matches.iloc[0] if not matches.empty else None


@st.cache_data(ttl=_TTL_IDS, show_spinner=False)
def get_mlbam_id(fg_id: int) -> int | None:
    """Convert a FanGraphs player ID to an MLBAM ID. Cached for 24 h."""
    result = playerid_reverse_lookup([fg_id], key_type="fangraphs")
    if result.empty or result["key_mlbam"].isna().all():
        return None
    return int(result["key_mlbam"].iloc[0])


def assert_core_stats_present(df: pd.DataFrame) -> None:
    """Raise ValueError if any core stat column is missing from *df*."""
    missing = [c for c in CORE_STAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Core stat columns missing from DataFrame: {missing}")
