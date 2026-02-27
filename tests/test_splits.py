"""Unit tests for stats/splits.py — uses in-memory fixture DataFrames."""

import pandas as pd
import numpy as np
import pytest

from stats.splits import (
    _compute_all_pitcher_stats,
    _pa_events,
    _compute_stats,
    _compute_gb_rate,
    _compute_pitch_level_stats,
    _compute_woba,
    BATTED_BALL_EVENTS,
    STAT_REGISTRY,
    StatSpec,
    get_sample_sizes,
    split_by_batter_hand,
    split_by_hand,
    split_home_away_pitcher,
    split_home_away,
    split_by_month_pitcher,
    split_by_month,
    get_pitcher_splits,
    get_splits,
    get_trend_stats,
    PITCHER_SPLIT_COLS,
    SPLIT_COLS,
    PA_EVENTS,
    BARREL_CODE,
)
from stats.filters import SplitFilters, prepare_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 30,
    p_throws: str | list = "R",
    stand: str | list = "R",
    inning_topbot: str | list = "Bot",
    month: int | list = 4,
) -> pd.DataFrame:
    """Build a minimal Statcast-shaped DataFrame for testing.

    By default all PAs are vs RHP, at home, in April.
    Spread of events: 6 SO, 3 BB, 1 HR, 1 3B, 2 2B, 5 1B, rest field_out.
    """
    events_cycle = [
        "strikeout", "strikeout", "walk", "home_run", "single",
        "field_out", "double", "strikeout", "walk", "single",
        "field_out", "field_out", "triple", "strikeout", "single",
        "double", "walk", "field_out", "strikeout", "single",
        "field_out", "field_out", "strikeout", "single", "field_out",
        "field_out", "field_out", "field_out", "field_out", "field_out",
    ]
    # Pad or trim to exactly n
    events = (events_cycle * ((n // len(events_cycle)) + 1))[:n]

    # Pitch rows: only the last pitch of each PA has events set; rest are None
    # For simplicity every row IS a PA outcome (all rows have events set)
    dates = pd.date_range(f"2024-0{month if isinstance(month, int) else 4}-01", periods=n).strftime("%Y-%m-%d")

    def _expand(val, n):
        if isinstance(val, list):
            return (val * ((n // len(val)) + 1))[:n]
        return [val] * n

    launch_speeds = [
        98.5 if e in {"single", "double", "triple", "home_run", "field_out"} else np.nan
        for e in events
    ]
    launch_speed_angles = [
        BARREL_CODE if s and s >= 98 else (4.0 if s else np.nan)
        for s in launch_speeds
    ]

    return pd.DataFrame(
        {
            "game_date": _expand(dates.tolist(), n) if isinstance(month, int) else [
                f"2024-0{m}-01" for m in _expand(month, n)
            ],
            "p_throws": _expand(p_throws, n),
            "stand": _expand(stand, n),
            "inning_topbot": _expand(inning_topbot, n),
            "events": events,
            "description": ["called_strike"] * n,
            "balls": [0] * n,
            "strikes": [0] * n,
            "launch_speed": launch_speeds,
            "launch_speed_angle": [float(x) for x in launch_speed_angles],
            "bb_type": [
                "ground_ball" if e in {"single", "double", "triple", "home_run", "field_out"} else np.nan
                for e in events
            ],
            "estimated_woba_using_speedangle": [
                0.500 if e in {"single", "double", "triple", "home_run"} else
                (0.050 if e == "field_out" else np.nan)
                for e in events
            ],
        }
    )


def _empty_df() -> pd.DataFrame:
    """A DataFrame with the right schema but zero rows."""
    return pd.DataFrame(
        columns=[
            "game_date", "p_throws", "stand", "inning_topbot", "events",
            "description", "balls", "strikes", "bb_type",
            "launch_speed", "launch_speed_angle",
            "estimated_woba_using_speedangle",
        ]
    )


# ---------------------------------------------------------------------------
# _pa_events
# ---------------------------------------------------------------------------

class TestPaEvents:
    def test_returns_only_pa_rows(self):
        df = _make_df(10)
        # Inject non-PA rows (pitch not resulting in event)
        df.loc[0, "events"] = None
        pa = _pa_events(df)
        assert pa["events"].notna().all()
        assert len(pa) == 9

    def test_all_events_in_pa_set(self):
        df = _make_df(10)
        pa = _pa_events(df)
        assert pa["events"].isin(PA_EVENTS).all()

    def test_empty_input_returns_empty(self):
        pa = _pa_events(pd.DataFrame(columns=["events"]))
        assert len(pa) == 0


# ---------------------------------------------------------------------------
# _compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_returns_all_keys(self):
        stats = _compute_stats(_make_df())
        expected_keys = {"PA", "K%", "BB%", "HardHit%", "Barrel%", "GB%", "xwOBA", "wOBA"}
        assert expected_keys == set(stats.keys())

    def test_pa_count_correct(self):
        stats = _compute_stats(_make_df(30))
        assert stats["PA"] == 30

    def test_k_pct_correct(self):
        # 6 strikeouts in 30 PA → 20%
        stats = _compute_stats(_make_df(30))
        assert stats["K%"] == pytest.approx(20.0, abs=0.1)

    def test_bb_pct_correct(self):
        # 3 walks in 30 PA → 10%
        stats = _compute_stats(_make_df(30))
        assert stats["BB%"] == pytest.approx(10.0, abs=0.1)

    def test_hard_hit_pct_positive(self):
        stats = _compute_stats(_make_df(30))
        assert stats["HardHit%"] is not None
        assert stats["HardHit%"] > 0

    def test_barrel_pct_positive(self):
        stats = _compute_stats(_make_df(30))
        assert stats["Barrel%"] is not None
        assert stats["Barrel%"] > 0

    def test_xwoba_in_range(self):
        stats = _compute_stats(_make_df(30))
        assert stats["xwOBA"] is not None
        assert 0.0 <= stats["xwOBA"] <= 2.0

    def test_empty_df_returns_all_none(self):
        stats = _compute_stats(_empty_df())
        assert stats["PA"] == 0
        for key in ["K%", "BB%", "HardHit%", "Barrel%", "GB%", "xwOBA", "wOBA"]:
            assert stats[key] is None

    def test_missing_launch_speed_angle_column(self):
        df = _make_df(20)
        df = df.drop(columns=["launch_speed_angle"])
        stats = _compute_stats(df)
        # Barrel% should be None, not raise
        assert stats["Barrel%"] is None

    def test_missing_xwoba_column(self):
        df = _make_df(20)
        df = df.drop(columns=["estimated_woba_using_speedangle"])
        stats = _compute_stats(df)
        assert stats["xwOBA"] is None

    def test_gb_none_when_bb_type_missing(self):
        df = _make_df(20).drop(columns=["bb_type"])
        stats = _compute_stats(df)
        assert stats["GB%"] is None


class TestComputeGbRate:
    def test_gb_percent_correct(self):
        pa = pd.DataFrame({"events": ["single", "field_out", "walk"]})
        bb_df = pd.DataFrame({"bb_type": ["ground_ball", "fly_ball"]})
        gb = _compute_gb_rate(pa, bb_df, 3)
        assert gb == pytest.approx(50.0, abs=0.1)

    def test_gb_percent_none_when_bb_type_missing(self):
        pa = pd.DataFrame({"events": ["single", "field_out"]})
        bb_df = pd.DataFrame({"launch_speed": [90.0, 95.0]})
        assert _compute_gb_rate(pa, bb_df, 2) is None

    def test_gb_percent_none_when_no_bip(self):
        pa = pd.DataFrame({"events": ["walk", "strikeout"]})
        bb_df = pd.DataFrame(columns=["bb_type"])
        assert _compute_gb_rate(pa, bb_df, 2) is None


class TestComputePitchLevelStats:
    def test_csw_correct(self):
        df = pd.DataFrame(
            {
                "description": ["called_strike", "ball", "foul_tip", "in_play"],
                "balls": [0, 0, 0, 1],
                "strikes": [0, 1, 2, 1],
            }
        )
        stats = _compute_pitch_level_stats(df)
        assert stats["CSW%"] == pytest.approx(50.0, abs=0.1)

    def test_whiff_none_on_zero_swings(self):
        df = pd.DataFrame(
            {
                "description": ["called_strike", "ball", "blocked_ball"],
                "balls": [0, 1, 2],
                "strikes": [0, 0, 1],
            }
        )
        stats = _compute_pitch_level_stats(df)
        assert stats["Whiff%"] is None

    def test_first_strike_correct(self):
        df = pd.DataFrame(
            {
                "description": ["called_strike", "ball", "foul_tip", "ball"],
                "balls": [0, 0, 1, 1],
                "strikes": [0, 0, 1, 2],
            }
        )
        stats = _compute_pitch_level_stats(df)
        assert stats["FirstStrike%"] == pytest.approx(50.0, abs=0.1)

    def test_first_strike_none_if_no_zero_zero_rows(self):
        df = pd.DataFrame(
            {
                "description": ["called_strike", "swinging_strike"],
                "balls": [1, 2],
                "strikes": [1, 2],
            }
        )
        stats = _compute_pitch_level_stats(df)
        assert stats["FirstStrike%"] is None


class TestPitcherDerivedStats:
    def test_k_minus_bb_equals_k_minus_bb(self):
        stats = _compute_all_pitcher_stats(_make_df(30))
        assert stats["K%"] is not None
        assert stats["BB%"] is not None
        assert stats["K-BB%"] == pytest.approx(stats["K%"] - stats["BB%"], abs=0.1)


# ---------------------------------------------------------------------------
# _compute_woba
# ---------------------------------------------------------------------------

class TestComputeWoba:
    def test_all_singles(self):
        df = pd.DataFrame({"events": ["single"] * 10})
        woba = _compute_woba(df)
        # 10 singles / 10 PA → weight 0.888
        assert woba == pytest.approx(0.888, abs=0.001)

    def test_all_strikeouts(self):
        df = pd.DataFrame({"events": ["strikeout"] * 10})
        woba = _compute_woba(df)
        assert woba == pytest.approx(0.0, abs=0.001)

    def test_sac_bunts_excluded_from_denominator(self):
        df = pd.DataFrame({"events": ["single", "sac_bunt", "strikeout"]})
        # denominator = 3 - 1 = 2; numerator = 0.888
        woba = _compute_woba(df)
        assert woba == pytest.approx(0.888 / 2, abs=0.001)

    def test_empty_returns_none(self):
        assert _compute_woba(pd.DataFrame({"events": []})) is None

    def test_only_sac_bunts_returns_none(self):
        df = pd.DataFrame({"events": ["sac_bunt", "sac_bunt"]})
        assert _compute_woba(df) is None


# ---------------------------------------------------------------------------
# StatSpec / registry
# ---------------------------------------------------------------------------

class TestStatRegistry:
    def test_registry_has_current_seven_stats(self):
        assert set(STAT_REGISTRY.keys()) == {
            "wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%", "GB%"
        }
        assert all(isinstance(spec, StatSpec) for spec in STAT_REGISTRY.values())

    def test_k_percent_compute_fn_on_synthetic_pa(self):
        pa = pd.DataFrame(
            {
                "events": ["strikeout", "walk", "single", "field_out"],
                "launch_speed": [np.nan, np.nan, 100.0, 90.0],
                "launch_speed_angle": [np.nan, np.nan, 6.0, 4.0],
                "estimated_woba_using_speedangle": [np.nan, np.nan, 0.5, 0.05],
            }
        )
        bb_df = pa[pa["events"].isin(BATTED_BALL_EVENTS) & pa["launch_speed"].notna()]
        raw_k = STAT_REGISTRY["K%"].compute_fn(pa, bb_df, len(pa))
        assert raw_k == pytest.approx(0.25, abs=0.001)

    def test_woba_compute_fn_on_synthetic_pa(self):
        pa = pd.DataFrame({"events": ["single", "walk", "strikeout"]})
        raw_woba = STAT_REGISTRY["wOBA"].compute_fn(pa, pd.DataFrame(), len(pa))
        assert raw_woba == pytest.approx((0.888 + 0.690) / 3, abs=0.001)


# ---------------------------------------------------------------------------
# get_sample_sizes
# ---------------------------------------------------------------------------

class TestGetSampleSizes:
    def test_returns_pitch_pa_and_bip_counts(self):
        counts = get_sample_sizes(_make_df(30))
        assert counts["N_pitches"] == 30
        assert counts["approx_PA"] == 30
        assert counts["N_BIP"] is not None
        assert counts["N_BIP"] > 0

    def test_empty_df_safe(self):
        counts = get_sample_sizes(_empty_df())
        assert counts["N_pitches"] == 0
        assert counts["approx_PA"] == 0
        assert counts["N_BIP"] == 0

    def test_missing_events_column(self):
        df = pd.DataFrame({"launch_speed": [95.0, 100.0]})
        counts = get_sample_sizes(df)
        assert counts["N_pitches"] == 2
        assert counts["approx_PA"] is None
        assert counts["N_BIP"] is None

    def test_missing_launch_speed_column(self):
        df = pd.DataFrame({"events": ["single", "walk", None]})
        counts = get_sample_sizes(df)
        assert counts["N_pitches"] == 3
        assert counts["approx_PA"] == 2
        assert counts["N_BIP"] is None


# ---------------------------------------------------------------------------
# split_by_hand
# ---------------------------------------------------------------------------

class TestSplitByHand:
    def test_returns_two_rows(self):
        df = _make_df(30, p_throws=["R", "L"])
        result = split_by_hand(df)
        assert len(result) == 2
        assert set(result["Split"]) == {"vs RHP", "vs LHP"}

    def test_columns_match_spec(self):
        result = split_by_hand(_make_df(30, p_throws=["R", "L"]))
        assert list(result.columns) == SPLIT_COLS

    def test_righty_pa_count(self):
        # 20 vs R, 10 vs L
        throws = ["R"] * 20 + ["L"] * 10
        df = _make_df(30, p_throws=throws)
        result = split_by_hand(df)
        rh_row = result[result["Split"] == "vs RHP"].iloc[0]
        assert rh_row["PA"] == 20

    def test_zero_pa_group_included(self):
        # All RHP — LHP row should still appear with PA=0
        df = _make_df(20, p_throws="R")
        result = split_by_hand(df)
        lh_row = result[result["Split"] == "vs LHP"].iloc[0]
        assert lh_row["PA"] == 0


# ---------------------------------------------------------------------------
# split_home_away
# ---------------------------------------------------------------------------

class TestSplitHomeAway:
    def test_returns_two_rows(self):
        df = _make_df(30, inning_topbot=["Bot", "Top"])
        result = split_home_away(df)
        assert len(result) == 2
        assert set(result["Split"]) == {"Home", "Away"}

    def test_columns_match_spec(self):
        result = split_home_away(_make_df(30, inning_topbot=["Bot", "Top"]))
        assert list(result.columns) == SPLIT_COLS

    def test_home_pa_count(self):
        topbot = ["Bot"] * 18 + ["Top"] * 12
        df = _make_df(30, inning_topbot=topbot)
        result = split_home_away(df)
        home_row = result[result["Split"] == "Home"].iloc[0]
        assert home_row["PA"] == 18


# ---------------------------------------------------------------------------
# split_by_month
# ---------------------------------------------------------------------------

class TestSplitByMonth:
    def test_one_row_per_month(self):
        df = _make_df(30, month=[4, 5, 6])
        result = split_by_month(df)
        assert len(result) == 3
        assert set(result["Split"]) == {"April", "May", "June"}

    def test_columns_match_spec(self):
        result = split_by_month(_make_df(30, month=[4, 5, 6]))
        assert list(result.columns) == SPLIT_COLS

    def test_month_ordering(self):
        df = _make_df(30, month=[6, 4, 5])  # shuffled months
        result = split_by_month(df)
        assert list(result["Split"]) == ["April", "May", "June"]

    def test_empty_df_returns_empty(self):
        df = _empty_df()
        result = split_by_month(df)
        assert len(result) == 0
        assert list(result.columns) == SPLIT_COLS


# ---------------------------------------------------------------------------
# get_splits dispatcher
# ---------------------------------------------------------------------------

class TestGetSplits:
    def test_dispatches_hand(self):
        df = _make_df(30, p_throws=["R", "L"])
        result = get_splits(df, "hand")
        assert set(result["Split"]) == {"vs RHP", "vs LHP"}

    def test_dispatches_home_away(self):
        df = _make_df(30, inning_topbot=["Bot", "Top"])
        result = get_splits(df, "home_away")
        assert set(result["Split"]) == {"Home", "Away"}

    def test_dispatches_monthly(self):
        df = _make_df(30, month=[4, 5])
        result = get_splits(df, "monthly")
        assert set(result["Split"]) == {"April", "May"}

    def test_invalid_split_type_raises(self):
        with pytest.raises(ValueError, match="Unknown split_type"):
            get_splits(_make_df(), "invalid")


class TestPitcherSplits:
    def test_split_by_batter_hand_uses_stand_and_labels(self):
        df = _make_df(30, p_throws="R", stand=["L"] * 12 + ["R"] * 18)
        result = split_by_batter_hand(df)
        assert set(result["Split"]) == {"vs LHB", "vs RHB"}
        assert "CSW%" in result.columns
        assert "Whiff%" in result.columns
        assert "FirstStrike%" in result.columns
        assert list(result.columns) == PITCHER_SPLIT_COLS
        lhb = result[result["Split"] == "vs LHB"].iloc[0]
        rhb = result[result["Split"] == "vs RHB"].iloc[0]
        assert lhb["PA"] == 12
        assert rhb["PA"] == 18

    def test_split_home_away_pitcher_mapping_correct(self):
        topbot = ["Top"] * 11 + ["Bot"] * 19
        df = _make_df(30, inning_topbot=topbot, stand="R")
        result = split_home_away_pitcher(df)
        home = result[result["Split"] == "Home"].iloc[0]
        away = result[result["Split"] == "Away"].iloc[0]
        assert home["PA"] == 11
        assert away["PA"] == 19
        assert "CSW%" in result.columns
        assert "Whiff%" in result.columns
        assert "FirstStrike%" in result.columns

    def test_get_pitcher_splits_dispatch(self):
        df = _make_df(30, stand=["L", "R"])
        hand = get_pitcher_splits(df, "hand")
        home_away = get_pitcher_splits(df, "home_away")
        monthly = get_pitcher_splits(df, "monthly")
        assert set(hand["Split"]) == {"vs LHB", "vs RHB"}
        assert set(home_away["Split"]) == {"Home", "Away"}
        assert "April" in set(monthly["Split"])


# ---------------------------------------------------------------------------
# get_trend_stats
# ---------------------------------------------------------------------------

class TestGetTrendStats:
    """Tests for get_trend_stats — uses injected fetch_fn stubs."""

    _STAT_KEYS = {"PA", "wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%", "GB%"}

    def _stub(self, df: "pd.DataFrame"):
        """Return a fetch_fn that always returns the given DataFrame."""
        def fetch_fn(mlbam_id: int, season: int) -> pd.DataFrame:
            return df.copy()
        return fetch_fn

    def test_returns_one_dict_per_season(self):
        stub = self._stub(_make_df(30))
        result = get_trend_stats(
            mlbam_id=1, seasons=[2022, 2023, 2024],
            player_type="Batter", filters=SplitFilters(),
            fetch_fn=stub, prepare_cache={},
        )
        assert len(result) == 3

    def test_season_key_present_in_each_row(self):
        stub = self._stub(_make_df(30))
        result = get_trend_stats(
            mlbam_id=1, seasons=[2022, 2023],
            player_type="Batter", filters=SplitFilters(),
            fetch_fn=stub, prepare_cache={},
        )
        assert result[0]["season"] == 2022
        assert result[1]["season"] == 2023

    def test_all_stat_keys_present_in_each_row(self):
        stub = self._stub(_make_df(30))
        result = get_trend_stats(
            mlbam_id=1, seasons=[2024, 2025],
            player_type="Batter", filters=SplitFilters(),
            fetch_fn=stub, prepare_cache={},
        )
        for row in result:
            assert self._STAT_KEYS.issubset(row.keys()), (
                f"Missing keys: {self._STAT_KEYS - row.keys()}"
            )

    def test_empty_df_produces_none_stats(self):
        stub = self._stub(_empty_df())
        result = get_trend_stats(
            mlbam_id=1, seasons=[2024],
            player_type="Batter", filters=SplitFilters(),
            fetch_fn=stub, prepare_cache={},
        )
        assert len(result) == 1
        row = result[0]
        assert row["PA"] == 0
        for stat in ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%", "GB%"]:
            assert row[stat] is None, f"Expected {stat} to be None for empty df"

    def test_empty_seasons_list_returns_empty(self):
        stub = self._stub(_make_df(30))
        result = get_trend_stats(
            mlbam_id=1, seasons=[],
            player_type="Batter", filters=SplitFilters(),
            fetch_fn=stub, prepare_cache={},
        )
        assert result == []

    def test_filters_applied_per_season(self):
        # Half RHP, half LHP — pitcher_hand filter should halve PA
        df = _make_df(30, p_throws=["R", "L"])  # alternating: 15 R, 15 L
        stub = self._stub(df)
        filters = SplitFilters(pitcher_hand="R")
        result = get_trend_stats(
            mlbam_id=1, seasons=[2024],
            player_type="Batter", filters=filters,
            fetch_fn=stub, prepare_cache={},
        )
        assert result[0]["PA"] == 15

    def test_prepare_cache_populated_after_call(self):
        stub = self._stub(_make_df(30))
        cache: dict = {}
        get_trend_stats(
            mlbam_id=7, seasons=[2023, 2024],
            player_type="Batter", filters=SplitFilters(),
            fetch_fn=stub, prepare_cache=cache,
        )
        assert (7, 2023, "Batter") in cache
        assert (7, 2024, "Batter") in cache

    def test_prepare_cache_hit_reuses_prepared_df(self):
        # Pre-populate cache with a known prepared df (30 PA)
        known_df = prepare_df(_make_df(30))
        cache = {(1, 2024, "Batter"): known_df}

        # Stub returns a different df (10 PA); cache hit should win
        different_stub = self._stub(_make_df(10))
        result = get_trend_stats(
            mlbam_id=1, seasons=[2024],
            player_type="Batter", filters=SplitFilters(),
            fetch_fn=different_stub, prepare_cache=cache,
        )
        # PA should be 30 (from cached df), not 10 (from stub)
        assert result[0]["PA"] == 30
