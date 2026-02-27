"""Data fetching layer — wraps pybaseball with Streamlit caching."""

import os
from pathlib import Path

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
    "inning",            # inning number (1–9+)
    "inning_topbot",     # Top = visiting batter, Bot = home batter
    "launch_speed",      # exit velocity
    "launch_angle",
    "launch_speed_angle",    # 6 = barrel (Baseball Savant classification)
    "estimated_woba_using_speedangle",  # xwOBA per event
    "events",            # plate appearance result
    "description",
    "stand",             # batter handedness
    "balls",             # ball count at time of pitch (0–3)
    "strikes",           # strike count at time of pitch (0–2)
]

_DISK_CACHE_ROOT = Path(
    os.getenv(
        "CLAUDE_BASEBALL_CACHE_DIR",
        str(Path(__file__).resolve().parents[1] / ".cache"),
    )
)
_STATCAST_YEAR_CACHE_DIR = _DISK_CACHE_ROOT / "statcast_year"


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


def _fetch_statcast_pitcher(player_mlbam_id: int, season: int) -> pd.DataFrame:
    """Return raw Statcast pitch-level events for one pitcher for a full season."""
    pb.cache.enable()
    start = f"{season}-03-01"
    end = f"{season}-11-30"
    df = pb.statcast_pitcher(start, end, player_mlbam_id)
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


def _normalize_mode(mode: str) -> str:
    lowered = str(mode).strip().lower()
    return "pitcher" if lowered.startswith("pitch") else "batter"


def _statcast_year_cache_path(player_mlbam_id: int, season: int, mode: str) -> Path:
    mode_key = _normalize_mode(mode)
    return _STATCAST_YEAR_CACHE_DIR / f"{mode_key}_{int(player_mlbam_id)}_{int(season)}.pkl"


def _load_year_df_from_disk(player_mlbam_id: int, season: int, mode: str) -> pd.DataFrame | None:
    cache_path = _statcast_year_cache_path(player_mlbam_id, season, mode)
    if not cache_path.exists():
        return None

    try:
        cached = pd.read_pickle(cache_path)
    except Exception:
        return None

    return cached.copy() if isinstance(cached, pd.DataFrame) else None


def _save_year_df_to_disk(player_mlbam_id: int, season: int, mode: str, df: pd.DataFrame) -> None:
    cache_path = _statcast_year_cache_path(player_mlbam_id, season, mode)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    except Exception:
        # Disk caching is a performance enhancement; failures should not block app flow.
        return


@st.cache_data(ttl=_TTL_STATS, show_spinner=False)
def load_or_fetch_year_df(player_mlbam_id: int, season: int, mode: str = "batter") -> pd.DataFrame:
    """Return one season of Statcast data using disk+Streamlit caching."""
    mode_key = _normalize_mode(mode)
    disk_cached = _load_year_df_from_disk(player_mlbam_id, season, mode_key)
    if disk_cached is not None:
        return disk_cached

    fetcher = _fetch_statcast_pitcher if mode_key == "pitcher" else _fetch_statcast_batter
    fetched = fetcher(player_mlbam_id, season)
    _save_year_df_to_disk(player_mlbam_id, season, mode_key, fetched)
    return fetched


def get_statcast_batter(player_mlbam_id: int, season: int) -> pd.DataFrame:
    """Cached raw Statcast events for one batter."""
    return load_or_fetch_year_df(player_mlbam_id, season, mode="batter")


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
def get_mlbam_id(fg_id: int, player_name: str | None = None) -> int | None:
    """Convert a FanGraphs player ID to an MLBAM ID. Cached for 24 h.

    Falls back to a live name-based lookup when the static Chadwick register
    doesn't contain the player (e.g. rookies who debuted mid-season).
    """
    result = playerid_reverse_lookup([fg_id], key_type="fangraphs")
    if not result.empty and not result["key_mlbam"].isna().all():
        return int(result["key_mlbam"].iloc[0])

    # Static register miss — try live name lookup as fallback.
    if player_name:
        parts = player_name.strip().split(" ", 1)
        first = parts[0]
        last  = parts[1] if len(parts) > 1 else ""
        name_result = _lookup_player(last, first)
        if not name_result.empty and "key_mlbam" in name_result.columns:
            valid = name_result["key_mlbam"].dropna()
            if not valid.empty:
                return int(valid.iloc[0])

    return None


def assert_core_stats_present(df: pd.DataFrame) -> None:
    """Raise ValueError if any core stat column is missing from *df*."""
    missing = [c for c in CORE_STAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Core stat columns missing from DataFrame: {missing}")
