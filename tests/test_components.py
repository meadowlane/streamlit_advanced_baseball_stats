"""Unit tests for ui/components.py — covers pure helper functions only.

Streamlit rendering functions (stat_card, percentile_bar_chart, etc.)
require a live runtime and are exercised via manual smoke test in Phase 7.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ui.components import (
    format_stat_value,
    format_percentile,
    build_chart_df,
    _build_trend_tidy_df,
    _build_single_stat_chart,
    _filter_real_data_rows,
    _stat_formatter,
    _stats_share_scale,
    _add_trend_traces,
    _ORDERED_STATS,
    _SPLIT_TABLE_FORMAT,
    _SPLIT_TABLE_HELP,
    _ARSENAL_TABLE_HELP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STAT_VALUES = {
    "wOBA": 0.380,
    "xwOBA": 0.365,
    "K%": 18.5,
    "BB%": 11.2,
    "HardHit%": 52.3,
    "Barrel%": 9.8,
}

_PERCENTILES = {
    "wOBA": 82.0,
    "xwOBA": 74.0,
    "K%": 68.0,
    "BB%": 71.0,
    "HardHit%": 65.0,
    "Barrel%": 60.0,
}

_COLOR_TIERS = {
    "wOBA":     {"name": "orange", "hex": "#E67E22"},
    "xwOBA":    {"name": "orange", "hex": "#E67E22"},
    "K%":       {"name": "yellow", "hex": "#F1C40F"},
    "BB%":      {"name": "orange", "hex": "#E67E22"},
    "HardHit%": {"name": "yellow", "hex": "#F1C40F"},
    "Barrel%":  {"name": "yellow", "hex": "#F1C40F"},
}


# ---------------------------------------------------------------------------
# format_stat_value
# ---------------------------------------------------------------------------

class TestFormatStatValue:
    def test_woba_three_decimals(self):
        assert format_stat_value("wOBA", 0.380) == ".380"

    def test_xwoba_three_decimals(self):
        assert format_stat_value("xwOBA", 0.365) == ".365"

    def test_k_pct_one_decimal_with_suffix(self):
        assert format_stat_value("K%", 18.5) == "18.5%"

    def test_bb_pct_one_decimal_with_suffix(self):
        assert format_stat_value("BB%", 11.2) == "11.2%"

    def test_hardhit_one_decimal_with_suffix(self):
        assert format_stat_value("HardHit%", 52.3) == "52.3%"

    def test_barrel_one_decimal_with_suffix(self):
        assert format_stat_value("Barrel%", 9.8) == "9.8%"

    def test_none_returns_dash(self):
        assert format_stat_value("wOBA", None) == "—"

    def test_nan_returns_dash(self):
        assert format_stat_value("K%", float("nan")) == "—"

    def test_unknown_stat_falls_back_to_three_decimals(self):
        result = format_stat_value("FIP", 3.21)
        assert "3.210" in result


# ---------------------------------------------------------------------------
# format_percentile
# ---------------------------------------------------------------------------

class TestFormatPercentile:
    @pytest.mark.parametrize("pct,expected", [
        (87.4, "87th"),
        (90.0, "90th"),
        (0.0,  "0th"),
        (100.0, "100th"),
        (50.6, "51th"),  # rounds to nearest int
    ])
    def test_formats_correctly(self, pct, expected):
        assert format_percentile(pct) == expected

    def test_nan_returns_dash(self):
        assert format_percentile(float("nan")) == "—"

    def test_none_returns_dash(self):
        assert format_percentile(None) == "—"  # type: ignore


# ---------------------------------------------------------------------------
# build_chart_df
# ---------------------------------------------------------------------------

class TestBuildChartDf:
    def _df(self):
        return build_chart_df(_PERCENTILES, _COLOR_TIERS, _STAT_VALUES)

    def test_returns_dataframe(self):
        assert isinstance(self._df(), pd.DataFrame)

    def test_has_one_row_per_ordered_stat(self):
        df = self._df()
        assert list(df["stat"]) == _ORDERED_STATS

    def test_required_columns_present(self):
        df = self._df()
        for col in ["stat", "percentile", "color", "label"]:
            assert col in df.columns

    def test_percentile_values_are_numeric(self):
        df = self._df()
        assert pd.api.types.is_numeric_dtype(df["percentile"])

    def test_nan_percentile_becomes_zero(self):
        percentiles = {**_PERCENTILES, "Barrel%": float("nan")}
        df = build_chart_df(percentiles, _COLOR_TIERS, _STAT_VALUES)
        barrel_row = df[df["stat"] == "Barrel%"].iloc[0]
        assert barrel_row["percentile"] == 0.0

    def test_label_contains_value_string(self):
        df = self._df()
        woba_row = df[df["stat"] == "wOBA"].iloc[0]
        assert ".380" in woba_row["label"]

    def test_label_contains_percentile_string(self):
        df = self._df()
        woba_row = df[df["stat"] == "wOBA"].iloc[0]
        assert "82th" in woba_row["label"]

    def test_color_hex_starts_with_hash(self):
        df = self._df()
        assert df["color"].str.startswith("#").all()

    def test_missing_stat_value_shows_dash_in_label(self):
        values = {**_STAT_VALUES, "Barrel%": None}
        df = build_chart_df(_PERCENTILES, _COLOR_TIERS, values)
        barrel_row = df[df["stat"] == "Barrel%"].iloc[0]
        assert "—" in barrel_row["label"]


# ---------------------------------------------------------------------------
# _build_trend_tidy_df
# ---------------------------------------------------------------------------

def test_build_trend_tidy_df_includes_stat_key_schema():
    trend_data = [
        {
            "season": 2024,
            "xwOBA": 0.365,
            "n_pitches": 412,
            "approx_pa": 104,
            "n_bip": 71,
        }
    ]

    df = _build_trend_tidy_df(trend_data, ["xwOBA"])

    required = {"year", "stat_key", "value", "n_pitches", "approx_pa", "n_bip"}
    assert required.issubset(df.columns)
    assert df.iloc[0]["stat_key"] == "xwOBA"


def test_filter_real_data_rows_keeps_only_non_null_with_pitches():
    df = pd.DataFrame(
        [
            {"year": 2019, "value": None, "n_pitches": 0},
            {"year": 2020, "value": 0.312, "n_pitches": 300},
            {"year": 2021, "value": 0.325, "n_pitches": 0},
            {"year": 2022, "value": 0.340, "n_pitches": 420},
        ]
    )
    filtered = _filter_real_data_rows(df)
    assert filtered["year"].tolist() == [2020, 2022]


def test_stat_formatter_uses_registry_for_decimal_stats():
    assert _stat_formatter("wOBA") == "decimal_3"


def test_stat_formatter_pitcher_only_fallback_is_pct():
    assert _stat_formatter("CSW%") == "pct_1"
    assert _stat_formatter("K-BB%") == "pct_1"


def test_stats_share_scale_detects_mixed_scales():
    assert _stats_share_scale(["wOBA", "K%"]) is False
    assert _stats_share_scale(["K%", "BB%"]) is True


def test_add_trend_traces_creates_open_marker_trace_for_low_sample():
    fig = go.Figure()
    stat_df = pd.DataFrame(
        [
            {"year": 2022, "value": 25.0, "approx_pa": 120, "n_pitches": 500},
            {"year": 2023, "value": 23.5, "approx_pa": 40, "n_pitches": 180},
        ]
    )
    has_low = _add_trend_traces(
        fig=fig,
        stat_df=stat_df,
        player_label="Player A",
        stat_key="K%",
        color="#4FC3F7",
        dash="solid",
        player_type="Batter",
    )
    assert has_low is True
    assert len(fig.data) == 2
    assert fig.data[0].connectgaps is False
    assert fig.data[1].mode == "markers"
    assert fig.data[1].marker.symbol == "circle-open"
    assert fig.data[1].showlegend is False


def test_build_single_stat_chart_contains_low_sample_open_marker_trace():
    stat_df_a = pd.DataFrame(
        [
            {"year": 2022, "value": 24.0, "approx_pa": 90, "n_pitches": 420},
            {"year": 2023, "value": 21.5, "approx_pa": 35, "n_pitches": 175},
        ]
    )
    fig = _build_single_stat_chart(
        stat_df_a=stat_df_a,
        stat_df_b=pd.DataFrame(columns=stat_df_a.columns),
        stat_key="K%",
        player_label_a="Player A",
        player_label_b=None,
        year_range=(2022, 2023),
        player_type="Batter",
    )
    assert any(getattr(trace.marker, "symbol", None) == "circle-open" for trace in fig.data)


def test_split_table_column_help_is_wired():
    required = {
        "PA",
        "wOBA",
        "wOBA Allowed",
        "xwOBA",
        "xwOBA Allowed",
        "K%",
        "BB%",
        "K-BB%",
        "CSW%",
        "Whiff%",
        "FirstStrike%",
        "HardHit%",
        "Barrel%",
        "GB%",
    }
    assert required.issubset(_SPLIT_TABLE_HELP.keys())
    assert required.issubset(_SPLIT_TABLE_FORMAT.keys())
    for col in required:
        help_text = _SPLIT_TABLE_FORMAT[col].get("help")
        assert isinstance(help_text, str) and len(help_text) > 0


def test_arsenal_table_help_keys_present():
    required = {"Pitch", "N", "Usage%", "Velo", "Spin", "CSW%", "Whiff%"}
    assert set(_ARSENAL_TABLE_HELP.keys()) == required
    for col in required:
        assert isinstance(_ARSENAL_TABLE_HELP[col], str)
        assert len(_ARSENAL_TABLE_HELP[col]) > 0
