"""Phase 2 integration tests for the core data pipeline and cache invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats.filters import SplitFilters, apply_filters, get_prepared_df_cached, prepare_df
from stats.percentiles import build_league_distributions, get_all_percentiles
from stats.splits import _compute_stats, compute_pitch_arsenal, get_splits, get_trend_stats


def test_prepare_df_outputs_downstream_columns_and_types(statcast_df_factory) -> None:
    raw = statcast_df_factory(n=36, include_pitch_cols=True)
    prepared = prepare_df(raw)

    required = {
        "game_date",
        "inning",
        "_month",
        "events",
        "p_throws",
        "stand",
        "inning_topbot",
        "pitch_type",
    }
    assert required.issubset(prepared.columns)
    assert len(prepared) == len(raw)
    assert pd.api.types.is_datetime64_any_dtype(prepared["game_date"])
    assert pd.api.types.is_numeric_dtype(prepared["inning"])
    assert prepared["_month"].dropna().between(1, 12).all()


def test_prepare_df_is_deterministic_for_same_input(statcast_df_factory) -> None:
    raw = statcast_df_factory(n=40, month=[4, 5, 6], include_pitch_cols=True)
    first = prepare_df(raw)
    second = prepare_df(raw)
    pd.testing.assert_frame_equal(first, second)


def test_prepare_df_handles_empty_input(empty_statcast_df) -> None:
    prepared = prepare_df(empty_statcast_df)
    assert prepared.empty
    assert "_month" in prepared.columns

    filtered = apply_filters(prepared, SplitFilters(month=4))
    splits = get_splits(filtered, "monthly")
    assert filtered.empty
    assert splits.empty


def test_full_pipeline_filtered_pas_align_between_stats_and_splits(
    statcast_df_factory,
    season_df_factory,
) -> None:
    raw = statcast_df_factory(
        n=120,
        p_throws=["R", "L"],
        stand=["L", "R"],
        inning_topbot=["Bot", "Top"],
        month=[4, 5, 6],
        include_pitch_cols=True,
    )
    cache: dict[tuple[int, int, str], pd.DataFrame] = {}
    prepared = get_prepared_df_cached(raw, cache, (7, 2024, "Batter"))
    filtered = apply_filters(
        prepared,
        SplitFilters(inning_min=3, inning_max=7, pitcher_hand="R", month=5),
    )

    stats = _compute_stats(filtered)
    splits = get_splits(filtered, "hand")
    distributions = build_league_distributions(season_df_factory(n=200, seed=7))
    percentiles = get_all_percentiles(stats, distributions)

    assert stats["PA"] == len(filtered)
    assert int(splits["PA"].sum()) == len(filtered)
    for pct in percentiles.values():
        assert np.isnan(pct) or (0.0 <= pct <= 100.0)


def test_pitch_arsenal_updates_when_inning_filter_changes(statcast_df_factory) -> None:
    raw = statcast_df_factory(n=270, include_pitch_cols=True)
    raw.loc[raw["inning"] <= 3, "pitch_type"] = "FF"
    raw.loc[raw["inning"] >= 4, "pitch_type"] = "SL"
    raw["description"] = "called_strike"
    prepared = prepare_df(raw)

    all_mix = compute_pitch_arsenal(prepared)
    early_mix = compute_pitch_arsenal(apply_filters(prepared, SplitFilters(inning_max=3)))
    late_mix = compute_pitch_arsenal(apply_filters(prepared, SplitFilters(inning_min=4)))

    assert set(all_mix["Pitch"]) == {"Four-Seam Fastball", "Slider"}
    assert early_mix["Pitch"].tolist() == ["Four-Seam Fastball"]
    assert late_mix["Pitch"].tolist() == ["Slider"]
    assert int(all_mix["N"].sum()) == 270
    assert int(early_mix["N"].sum()) == 90
    assert int(late_mix["N"].sum()) == 180


def test_trend_results_change_with_filters_even_when_prepared_df_is_cached(
    statcast_df_factory,
    make_trend_stub,
) -> None:
    raw = statcast_df_factory(n=180, month=[4, 4, 4, 5, 5, 6], include_pitch_cols=True)
    fetch_stub = make_trend_stub(raw)
    cache: dict[tuple[int, int, str], pd.DataFrame] = {}
    cache_key = (99, 2024, "Batter")

    april = get_trend_stats(
        mlbam_id=99,
        seasons=[2024],
        player_type="Batter",
        filters=SplitFilters(month=4),
        fetch_fn=fetch_stub,
        prepare_cache=cache,
    )[0]
    first_prepared = cache[cache_key]
    june = get_trend_stats(
        mlbam_id=99,
        seasons=[2024],
        player_type="Batter",
        filters=SplitFilters(month=6),
        fetch_fn=fetch_stub,
        prepare_cache=cache,
    )[0]

    assert cache[cache_key] is first_prepared
    assert april["PA"] == 90
    assert june["PA"] == 30
    assert april["PA"] != june["PA"]
