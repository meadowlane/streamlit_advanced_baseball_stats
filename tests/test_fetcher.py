"""Unit tests for data/fetcher.py — all network calls are mocked."""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from data.fetcher import (
    CORE_STAT_COLS,
    STATCAST_KEEP_COLS,
    _fetch_batting_stats,
    _fetch_statcast_batter,
    _lookup_player,
    get_player_row,
    assert_core_stats_present,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_batting_df(names=("Aaron Judge", "Shohei Ohtani")) -> pd.DataFrame:
    """Minimal FanGraphs-shaped DataFrame with all 6 core stat columns."""
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "IDfg": 1000 + i,
            "Name": name,
            "Team": "NYY" if i == 0 else "LAD",
            "Season": 2024,
            "PA": 600,
            "wOBA": 0.400 + i * 0.010,
            "xwOBA": 0.395 + i * 0.010,
            "K%": 0.22 - i * 0.01,
            "BB%": 0.12 + i * 0.01,
            "HardHit%": 55.0 + i,
            "Barrel%": 20.0 + i,
        })
    return pd.DataFrame(rows)


def _make_statcast_df(n=100) -> pd.DataFrame:
    """Minimal Statcast-shaped DataFrame covering STATCAST_KEEP_COLS."""
    import numpy as np

    rng = pd.date_range("2024-04-01", periods=n, freq="D")
    data = {
        "game_date": rng,
        "batter": [660271] * n,
        "pitcher": [123456] * n,
        "p_throws": ["R"] * (n // 2) + ["L"] * (n - n // 2),
        "home_team": ["LAD"] * n,
        "away_team": ["NYY"] * n,
        "inning_topbot": ["Bot"] * n,
        "launch_speed": [98.0 if i % 3 == 0 else 88.0 for i in range(n)],
        "launch_angle": [15.0] * n,
        "barrel": [1 if i % 5 == 0 else 0 for i in range(n)],
        "estimated_woba_using_speedangle": [0.400] * n,
        "events": ["single"] * n,
        "description": ["hit_into_play"] * n,
        "stand": ["L"] * n,
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# _fetch_batting_stats
# ---------------------------------------------------------------------------

class TestFetchBattingStats:
    @patch("data.fetcher.pb.batting_stats")
    @patch("data.fetcher.pb.cache.enable")
    def test_returns_dataframe(self, mock_cache, mock_batting):
        mock_batting.return_value = _make_batting_df()
        df = _fetch_batting_stats(2024)
        assert isinstance(df, pd.DataFrame)

    @patch("data.fetcher.pb.batting_stats")
    @patch("data.fetcher.pb.cache.enable")
    def test_calls_pybaseball_with_correct_season(self, mock_cache, mock_batting):
        mock_batting.return_value = _make_batting_df()
        _fetch_batting_stats(2023, min_pa=100)
        mock_batting.assert_called_once_with(2023, qual=100)

    @patch("data.fetcher.pb.batting_stats")
    @patch("data.fetcher.pb.cache.enable")
    def test_all_core_stats_present(self, mock_cache, mock_batting):
        mock_batting.return_value = _make_batting_df()
        df = _fetch_batting_stats(2024)
        for col in CORE_STAT_COLS:
            assert col in df.columns, f"Missing core stat: {col}"

    @patch("data.fetcher.pb.batting_stats")
    @patch("data.fetcher.pb.cache.enable")
    def test_default_min_pa_is_50(self, mock_cache, mock_batting):
        mock_batting.return_value = _make_batting_df()
        _fetch_batting_stats(2024)
        mock_batting.assert_called_once_with(2024, qual=50)


# ---------------------------------------------------------------------------
# _fetch_statcast_batter
# ---------------------------------------------------------------------------

class TestFetchStatcastBatter:
    @patch("data.fetcher.pb.statcast_batter")
    @patch("data.fetcher.pb.cache.enable")
    def test_returns_dataframe(self, mock_cache, mock_statcast):
        mock_statcast.return_value = _make_statcast_df()
        df = _fetch_statcast_batter(660271, 2024)
        assert isinstance(df, pd.DataFrame)

    @patch("data.fetcher.pb.statcast_batter")
    @patch("data.fetcher.pb.cache.enable")
    def test_date_range_covers_full_season(self, mock_cache, mock_statcast):
        mock_statcast.return_value = _make_statcast_df()
        _fetch_statcast_batter(660271, 2024)
        mock_statcast.assert_called_once_with("2024-03-01", "2024-11-30", 660271)

    @patch("data.fetcher.pb.statcast_batter")
    @patch("data.fetcher.pb.cache.enable")
    def test_only_keep_cols_returned(self, mock_cache, mock_statcast):
        full_df = _make_statcast_df()
        full_df["irrelevant_col"] = "noise"
        mock_statcast.return_value = full_df
        df = _fetch_statcast_batter(660271, 2024)
        assert "irrelevant_col" not in df.columns

    @patch("data.fetcher.pb.statcast_batter")
    @patch("data.fetcher.pb.cache.enable")
    def test_missing_statcast_cols_handled_gracefully(self, mock_cache, mock_statcast):
        """If Statcast response omits some columns, fetcher should not crash."""
        sparse_df = _make_statcast_df()[["game_date", "batter", "launch_speed"]]
        mock_statcast.return_value = sparse_df
        df = _fetch_statcast_batter(660271, 2024)
        assert list(df.columns) == ["game_date", "batter", "launch_speed"]


# ---------------------------------------------------------------------------
# _lookup_player
# ---------------------------------------------------------------------------

class TestLookupPlayer:
    @patch("data.fetcher.pb.playerid_lookup")
    def test_passes_names_to_pybaseball(self, mock_lookup):
        mock_lookup.return_value = pd.DataFrame(
            [{"name_last": "ohtani", "name_first": "shohei", "key_mlbam": 660271}]
        )
        _lookup_player("ohtani", "shohei")
        mock_lookup.assert_called_once_with("ohtani", "shohei")

    @patch("data.fetcher.pb.playerid_lookup")
    def test_last_name_only(self, mock_lookup):
        mock_lookup.return_value = pd.DataFrame()
        _lookup_player("judge")
        mock_lookup.assert_called_once_with("judge", "")


# ---------------------------------------------------------------------------
# get_player_row
# ---------------------------------------------------------------------------

class TestGetPlayerRow:
    def test_returns_matching_row(self):
        df = _make_batting_df()
        row = get_player_row(df, "Aaron Judge")
        assert row is not None
        assert row["Name"] == "Aaron Judge"

    def test_case_insensitive(self):
        df = _make_batting_df()
        row = get_player_row(df, "aaron judge")
        assert row is not None

    def test_returns_none_when_not_found(self):
        df = _make_batting_df()
        row = get_player_row(df, "Babe Ruth")
        assert row is None


# ---------------------------------------------------------------------------
# assert_core_stats_present
# ---------------------------------------------------------------------------

class TestAssertCoreStatsPresent:
    def test_passes_when_all_cols_present(self):
        df = _make_batting_df()
        assert_core_stats_present(df)  # should not raise

    def test_raises_when_col_missing(self):
        df = _make_batting_df().drop(columns=["xwOBA"])
        with pytest.raises(ValueError, match="xwOBA"):
            assert_core_stats_present(df)
