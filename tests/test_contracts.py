"""Contract tests for filter wiring into derived outputs and cached trend paths."""

from __future__ import annotations

import pandas as pd
import pytest

from stats.filters import SplitFilters, apply_filters, prepare_df
from stats.splits import compute_pitch_arsenal, get_splits, get_trend_stats


@pytest.fixture
def varied_statcast_df(statcast_df_factory) -> pd.DataFrame:
    """Deterministic Statcast-like data with variation across key filter dimensions."""
    return statcast_df_factory(
        n=240,
        p_throws=["R", "L", "R", "L", "R", "L", "R", "L"],
        stand=["L", "R", "R", "L", "L", "R", "L", "R"],
        inning_topbot=["Bot", "Top", "Bot", "Top", "Bot", "Top", "Top", "Bot"],
        month=[4, 4, 4, 5, 5, 6, 7, 7],
        include_pitch_cols=True,
    )


@pytest.fixture
def varied_prepared_statcast_df(varied_statcast_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_df(varied_statcast_df)


def _pa_for_split(splits_df: pd.DataFrame, split_label: str) -> int:
    row = splits_df.loc[splits_df["Split"] == split_label, "PA"]
    assert not row.empty, f"Missing split row: {split_label}"
    return int(row.iloc[0])


def test_month_filter_changes_monthly_split_rows(varied_prepared_statcast_df: pd.DataFrame):
    all_months = get_splits(varied_prepared_statcast_df, "monthly")
    june_filtered = apply_filters(varied_prepared_statcast_df, SplitFilters(month=6))
    june_only = get_splits(june_filtered, "monthly")

    assert set(all_months["Split"]) == {"April", "May", "June", "July"}
    assert june_only["Split"].tolist() == ["June"]
    assert int(june_only["PA"].sum()) == 30
    assert int(all_months["PA"].sum()) == len(varied_prepared_statcast_df)


def test_inning_and_handedness_filters_change_hand_split_pas(
    varied_prepared_statcast_df: pd.DataFrame,
):
    vs_right = apply_filters(
        varied_prepared_statcast_df,
        SplitFilters(inning_min=2, inning_max=7, batter_hand="L", pitcher_hand="R"),
    )
    vs_left = apply_filters(
        varied_prepared_statcast_df,
        SplitFilters(inning_min=2, inning_max=7, batter_hand="L", pitcher_hand="L"),
    )

    right_splits = get_splits(vs_right, "hand")
    left_splits = get_splits(vs_left, "hand")

    assert _pa_for_split(right_splits, "vs RHP") > 0
    assert _pa_for_split(right_splits, "vs LHP") == 0
    assert _pa_for_split(left_splits, "vs RHP") == 0
    assert _pa_for_split(left_splits, "vs LHP") > 0


def test_batter_hand_filter_changes_pitch_arsenal(varied_prepared_statcast_df: pd.DataFrame):
    month_april = SplitFilters(month=4)
    left_april = apply_filters(
        apply_filters(varied_prepared_statcast_df, month_april),
        SplitFilters(batter_hand="L"),
    )
    right_april = apply_filters(
        apply_filters(varied_prepared_statcast_df, month_april),
        SplitFilters(batter_hand="R"),
    )

    left_arsenal = compute_pitch_arsenal(left_april)
    right_arsenal = compute_pitch_arsenal(right_april)

    assert set(left_arsenal["Pitch"]) == {"Four-Seam Fastball"}
    assert set(right_arsenal["Pitch"]) == {"Slider", "Changeup"}


def test_trend_cache_reuse_still_respects_filter_context(
    varied_statcast_df: pd.DataFrame,
    make_trend_stub,
):
    fetch_stub = make_trend_stub(varied_statcast_df)
    prepare_cache: dict[tuple[int, int, str], pd.DataFrame] = {}

    april = get_trend_stats(
        mlbam_id=42,
        seasons=[2024],
        player_type="Batter",
        filters=SplitFilters(month=4),
        fetch_fn=fetch_stub,
        prepare_cache=prepare_cache,
    )[0]
    june = get_trend_stats(
        mlbam_id=42,
        seasons=[2024],
        player_type="Batter",
        filters=SplitFilters(month=6),
        fetch_fn=fetch_stub,
        prepare_cache=prepare_cache,
    )[0]

    assert april["PA"] == 90
    assert june["PA"] == 30
    assert april["PA"] != june["PA"]
    assert set(prepare_cache.keys()) == {(42, 2024, "Batter")}
