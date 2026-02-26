"""Unit tests for stats/splits.py — uses in-memory fixture DataFrames."""

import pandas as pd
import numpy as np
import pytest

from stats.splits import (
    _pa_events,
    _compute_stats,
    _compute_woba,
    split_by_hand,
    split_home_away,
    split_by_month,
    get_splits,
    SPLIT_COLS,
    PA_EVENTS,
    BARREL_CODE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 30,
    p_throws: str | list = "R",
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
            "inning_topbot": _expand(inning_topbot, n),
            "events": events,
            "launch_speed": launch_speeds,
            "launch_speed_angle": [float(x) for x in launch_speed_angles],
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
            "game_date", "p_throws", "inning_topbot", "events",
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
        expected_keys = {"PA", "K%", "BB%", "HardHit%", "Barrel%", "xwOBA", "wOBA"}
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
        for key in ["K%", "BB%", "HardHit%", "Barrel%", "xwOBA", "wOBA"]:
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
