"""Data fetching layer — wraps pybaseball with Streamlit caching."""

import datetime as dt

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
PITCHER_KEEP_COLS = STATCAST_KEEP_COLS + [
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "bb_type",
    "plate_x",   # zone
    "plate_z",   # zone
    "sz_top",    # zone
    "sz_bot",    # zone
    "pfx_x",     # movement (Slice 5)
    "pfx_z",     # movement (Slice 5)
]
BATTER_KEEP_COLS = STATCAST_KEEP_COLS + [
    "pitch_type",
    "release_speed",
    "plate_x",   # zone
    "plate_z",   # zone
    "sz_top",    # zone
    "sz_bot",    # zone
    "pfx_x",     # movement (Slice 5)
    "pfx_z",     # movement (Slice 5)
]
_BATTING_BASE_COLS = ["IDfg", "Name", "Team", "PA", "Season"]
_PITCHING_BASE_COLS = ["IDfg", "Name", "Team", "TBF", "Season"]
_FG_WRC_PLUS_COLS = ["season", "key_mlbam", "IDfg", "Name", "Team", "name_team_key", "wRC+"]


def _last_completed_season_year() -> int:
    """Return the most recent season that should have stable full-year stats."""
    return dt.date.today().year - 1


def _empty_batting_stats_df(season: int, reason: str | None = None) -> pd.DataFrame:
    cols = _BATTING_BASE_COLS + CORE_STAT_COLS
    df = pd.DataFrame(columns=cols)
    df.attrs["season"] = int(season)
    if reason:
        df.attrs["warning"] = reason
    return df


def _empty_pitching_stats_df(season: int, reason: str | None = None) -> pd.DataFrame:
    cols = _PITCHING_BASE_COLS + CORE_STAT_COLS
    df = pd.DataFrame(columns=cols)
    df.attrs["season"] = int(season)
    if reason:
        df.attrs["warning"] = reason
    return df


# ---------------------------------------------------------------------------
# Pure fetch functions (no cache — called by cached wrappers; testable directly)
# ---------------------------------------------------------------------------

def _fetch_batting_stats(season: int, min_pa: int = 50) -> pd.DataFrame:
    """Return FanGraphs season batting stats for all qualified batters."""
    season_int = int(season)
    if season_int > _last_completed_season_year():
        return _empty_batting_stats_df(
            season_int,
            reason=f"No batting stats available for {season_int} yet.",
        )

    pb.cache.enable()
    try:
        return _ensure_traditional_slash_stats(pb.batting_stats(season_int, qual=min_pa))
    except Exception:
        return _empty_batting_stats_df(
            season_int,
            reason=f"No batting stats available for {season_int} yet.",
        )


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_vals = pd.to_numeric(numer, errors="coerce")
    denom_vals = pd.to_numeric(denom, errors="coerce")
    return numer_vals.where(denom_vals > 0, float("nan")) / denom_vals.where(denom_vals > 0, float("nan"))


def _find_col_case_insensitive(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lowered:
            return str(lowered[key])
    return None


def _get_numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    col = _find_col_case_insensitive(df, candidates)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _ensure_traditional_slash_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure AVG/OBP/SLG/OPS exist in a FanGraphs-style batting DataFrame."""
    if df.empty:
        out = df.copy()
        for stat in ("AVG", "OBP", "SLG", "OPS"):
            if stat not in out.columns:
                out[stat] = pd.Series(dtype="float64")
        return out

    out = df.copy()

    # Map existing columns first (case-insensitive).
    for stat in ("AVG", "OBP", "SLG", "OPS"):
        found = _find_col_case_insensitive(out, [stat])
        if found is not None:
            out[stat] = pd.to_numeric(out[found], errors="coerce")
        elif stat not in out.columns:
            out[stat] = pd.Series(float("nan"), index=out.index, dtype="float64")

    hits = _get_numeric_series(out, ["H", "Hits"])
    at_bats = _get_numeric_series(out, ["AB", "At Bats", "at_bats"])
    walks = _get_numeric_series(out, ["BB", "Walks"])
    doubles = _get_numeric_series(out, ["2B", "Doubles"])
    triples = _get_numeric_series(out, ["3B", "Triples"])
    homers = _get_numeric_series(out, ["HR", "Home Runs", "home_runs"])

    # Per requirement: if HBP/SF are missing, treat them as 0 for OBP computation.
    hbp = _get_numeric_series(out, ["HBP", "Hit By Pitch", "hit_by_pitch"])
    if hbp is None:
        hbp = pd.Series(0.0, index=out.index)
    sf = _get_numeric_series(out, ["SF", "Sacrifice Fly", "sacrifice_fly"])
    if sf is None:
        sf = pd.Series(0.0, index=out.index)

    if out["AVG"].isna().any() and hits is not None and at_bats is not None:
        computed_avg = _safe_divide(hits, at_bats)
        out["AVG"] = out["AVG"].fillna(computed_avg)

    if out["OBP"].isna().any() and hits is not None and at_bats is not None and walks is not None:
        obp_numer = hits + walks + hbp
        obp_denom = at_bats + walks + hbp + sf
        computed_obp = _safe_divide(obp_numer, obp_denom)
        out["OBP"] = out["OBP"].fillna(computed_obp)

    if (
        out["SLG"].isna().any()
        and hits is not None
        and at_bats is not None
        and doubles is not None
        and triples is not None
        and homers is not None
    ):
        singles = hits - doubles - triples - homers
        total_bases = singles + (2.0 * doubles) + (3.0 * triples) + (4.0 * homers)
        computed_slg = _safe_divide(total_bases, at_bats)
        out["SLG"] = out["SLG"].fillna(computed_slg)

    if out["OPS"].isna().any():
        obp_vals = pd.to_numeric(out["OBP"], errors="coerce")
        slg_vals = pd.to_numeric(out["SLG"], errors="coerce")
        computed_ops = obp_vals + slg_vals
        out["OPS"] = pd.to_numeric(out["OPS"], errors="coerce").fillna(computed_ops)

    return out


def _normalize_name_team_key(name: object, team: object) -> str:
    name_str = str(name).strip().lower() if name is not None else ""
    team_str = str(team).strip().upper() if team is not None else ""
    return f"{name_str}|{team_str}"


def _empty_fg_wrc_plus_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_FG_WRC_PLUS_COLS)


def _fetch_fg_batting_wrc_plus(season: int) -> pd.DataFrame:
    """Return a slim FanGraphs batting table with season/player keys + wRC+."""
    season_int = int(season)
    if season_int > _last_completed_season_year():
        return _empty_fg_wrc_plus_df()

    pb.cache.enable()
    try:
        df = pb.batting_stats(season_int)
    except Exception:
        return _empty_fg_wrc_plus_df()

    if df.empty:
        return _empty_fg_wrc_plus_df()

    wrc_col = None
    for candidate in ("wRC+", "wRC_plus", "wrc_plus", "wrcplus"):
        if candidate in df.columns:
            wrc_col = candidate
            break
    if wrc_col is None:
        return _empty_fg_wrc_plus_df()

    out = pd.DataFrame(index=df.index)
    out["season"] = season_int
    out["key_mlbam"] = (
        pd.to_numeric(df["key_mlbam"], errors="coerce").astype("Int64")
        if "key_mlbam" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    out["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce") if "IDfg" in df.columns else pd.Series(pd.NA, index=df.index)
    out["Name"] = df["Name"].astype(str) if "Name" in df.columns else pd.Series("", index=df.index, dtype="string")
    out["Team"] = df["Team"].astype(str) if "Team" in df.columns else pd.Series("", index=df.index, dtype="string")
    out["name_team_key"] = [
        _normalize_name_team_key(name, team) for name, team in zip(out["Name"], out["Team"], strict=False)
    ]
    out["wRC+"] = pd.to_numeric(df[wrc_col], errors="coerce")
    return out[_FG_WRC_PLUS_COLS].copy()


def _fetch_statcast_batter(player_mlbam_id: int, season: int) -> pd.DataFrame:
    """Return raw Statcast pitch-level events for one batter for a full season."""
    pb.cache.enable()
    start = f"{season}-03-01"
    end = f"{season}-11-30"
    df = pb.statcast_batter(start, end, player_mlbam_id)
    # Keep only columns that exist in the response (Statcast schema can vary by year)
    cols = [c for c in BATTER_KEEP_COLS if c in df.columns]
    return df[cols].copy()


def _fetch_pitching_stats(season: int, min_ip: int = 20) -> pd.DataFrame:
    """Return FanGraphs season pitching stats for all qualified pitchers."""
    season_int = int(season)
    if season_int > _last_completed_season_year():
        return _empty_pitching_stats_df(
            season_int,
            reason=f"No pitching stats available for {season_int} yet.",
        )

    pb.cache.enable()
    try:
        return pb.pitching_stats(season_int, qual=min_ip)
    except Exception:
        return _empty_pitching_stats_df(
            season_int,
            reason=f"No pitching stats available for {season_int} yet.",
        )


def _fetch_statcast_pitcher(player_mlbam_id: int, season: int) -> pd.DataFrame:
    """Return raw Statcast pitch-level events for one pitcher for a full season."""
    pb.cache.enable()
    start = f"{season}-03-01"
    end = f"{season}-11-30"
    df = pb.statcast_pitcher(start, end, player_mlbam_id)
    cols = [c for c in PITCHER_KEEP_COLS if c in df.columns]
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
def get_fg_batting_wrc_plus(season: int) -> pd.DataFrame:
    """Cached slim FanGraphs batting table with season/player keys + wRC+."""
    return _fetch_fg_batting_wrc_plus(season)


@st.cache_data(ttl=_TTL_STATS, show_spinner=False)
def get_statcast_batter(player_mlbam_id: int, season: int) -> pd.DataFrame:
    """Cached raw Statcast events for one batter."""
    return _fetch_statcast_batter(player_mlbam_id, season)


@st.cache_data(ttl=_TTL_STATS, show_spinner=False)
def get_pitching_stats(season: int, min_ip: int = 20) -> pd.DataFrame:
    """Cached FanGraphs season pitching stats."""
    return _fetch_pitching_stats(season, min_ip)


@st.cache_data(ttl=_TTL_STATS, show_spinner=False)
def get_statcast_pitcher(player_mlbam_id: int, season: int) -> pd.DataFrame:
    """Cached raw Statcast events for one pitcher."""
    return _fetch_statcast_pitcher(player_mlbam_id, season)


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
