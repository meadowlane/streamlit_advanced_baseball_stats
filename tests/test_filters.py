"""Unit tests for stats/filters.py."""

import pandas as pd
import pytest

from stats.filters import (
    FILTER_REGISTRY,
    FilterSpec,
    SplitFilters,
    apply_filters,
    get_prepared_df_cached,
    prepare_df,
    rows_to_split_filters,
    summarize_filter_rows,
)


# ===========================================================================
# Fixtures / helpers
# ===========================================================================

def _make_df() -> pd.DataFrame:
    """Tiny synthetic pitch-level DataFrame with an 'inning' column."""
    return pd.DataFrame(
        {
            "inning": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "events": ["single", "out", "homer", "out", "double",
                       "out", "triple", "out", "walk"],
        }
    )


def _make_full_df() -> pd.DataFrame:
    """Six-row synthetic pitch-level DataFrame covering all filter columns.

    Layout (0-indexed rows):
      row  inning  p_throws  topbot  game_date    balls  strikes
        0      1       R      Top   2024-04-10      0       0
        1      2       R      Bot   2024-04-15      1       1
        2      3       L      Top   2024-05-01      2       2
        3      4       L      Bot   2024-05-20      3       0
        4      5       R      Bot   2024-06-05      0       1
        5      6       L      Top   2024-07-10      1       2
    """
    return pd.DataFrame(
        {
            "inning":        [1, 2, 3, 4, 5, 6],
            "p_throws":      ["R", "R", "L", "L", "R", "L"],
            "inning_topbot": ["Top", "Bot", "Top", "Bot", "Bot", "Top"],
            "game_date":     [
                "2024-04-10", "2024-04-15",
                "2024-05-01", "2024-05-20",
                "2024-06-05", "2024-07-10",
            ],
            "balls":   [0, 1, 2, 3, 0, 1],
            "strikes": [0, 1, 2, 0, 1, 2],
            "events":  ["out", "single", "homer", "walk", "out", "double"],
        }
    )


def _row(filter_type: str, params: dict, row_id: str = "r1") -> dict:
    """Build a filter-row dict as the UI session state would."""
    return {"id": row_id, "filter_type": filter_type, "params": params}


# ===========================================================================
# TestFilterSpec
# ===========================================================================

class TestFilterSpec:
    def test_is_dataclass(self):
        spec = FilterSpec(
            key="inning",
            label="Inning range",
            required_cols=["inning"],
            default_params={"min": 1, "max": 9},
        )
        assert spec.key == "inning"
        assert spec.label == "Inning range"
        assert spec.required_cols == ["inning"]
        assert spec.default_params == {"min": 1, "max": 9}


# ===========================================================================
# TestFilterRegistry
# ===========================================================================

class TestFilterRegistry:
    def test_all_six_keys_present(self):
        assert set(FILTER_REGISTRY.keys()) == {
            "inning", "pitcher_hand", "batter_hand", "home_away", "month", "count"
        }

    def test_inning_spec(self):
        s = FILTER_REGISTRY["inning"]
        assert s.required_cols == ["inning"]
        assert s.default_params == {"min": 1, "max": 9}

    def test_pitcher_hand_spec(self):
        s = FILTER_REGISTRY["pitcher_hand"]
        assert "p_throws" in s.required_cols
        assert s.default_params["hand"] in ("L", "R")

    def test_batter_hand_spec(self):
        s = FILTER_REGISTRY["batter_hand"]
        assert "stand" in s.required_cols
        assert s.default_params["hand"] in ("L", "R")

    def test_home_away_spec(self):
        s = FILTER_REGISTRY["home_away"]
        assert "inning_topbot" in s.required_cols
        assert s.default_params["side"] in ("home", "away")

    def test_month_spec(self):
        s = FILTER_REGISTRY["month"]
        assert "game_date" in s.required_cols
        assert isinstance(s.default_params["month"], int)
        assert 1 <= s.default_params["month"] <= 12

    def test_count_spec(self):
        s = FILTER_REGISTRY["count"]
        assert "balls" in s.required_cols
        assert "strikes" in s.required_cols
        assert "balls" in s.default_params
        assert "strikes" in s.default_params

    def test_every_key_is_a_filterspec(self):
        for key, spec in FILTER_REGISTRY.items():
            assert isinstance(spec, FilterSpec), f"{key} is not a FilterSpec"
            assert spec.key == key


# ===========================================================================
# TestSplitFilters
# ===========================================================================

class TestSplitFilters:
    def test_default_construction(self):
        f = SplitFilters()
        assert f is not None

    def test_inning_min_defaults_to_none(self):
        assert SplitFilters().inning_min is None

    def test_inning_max_defaults_to_none(self):
        assert SplitFilters().inning_max is None

    def test_fields_settable(self):
        f = SplitFilters(inning_min=1, inning_max=6)
        assert f.inning_min == 1
        assert f.inning_max == 6

    def test_partial_construction(self):
        f = SplitFilters(inning_min=7)
        assert f.inning_min == 7
        assert f.inning_max is None

    def test_new_fields_default_to_none(self):
        f = SplitFilters()
        assert f.pitcher_hand is None
        assert f.batter_hand is None
        assert f.home_away is None
        assert f.month is None
        assert f.balls is None
        assert f.strikes is None

    def test_new_fields_settable(self):
        f = SplitFilters(
            pitcher_hand="L",
            batter_hand="R",
            home_away="home",
            month=5,
            balls=2,
            strikes=1,
        )
        assert f.pitcher_hand == "L"
        assert f.batter_hand == "R"
        assert f.home_away == "home"
        assert f.month == 5
        assert f.balls == 2
        assert f.strikes == 1


# ===========================================================================
# TestRowsToSplitFilters
# ===========================================================================

class TestRowsToSplitFilters:
    def test_empty_list_gives_default_split_filters(self):
        result = rows_to_split_filters([])
        assert result == SplitFilters()

    # --- individual filter types ---

    def test_inning_row(self):
        rows = [_row("inning", {"min": 3, "max": 7})]
        result = rows_to_split_filters(rows)
        assert result.inning_min == 3
        assert result.inning_max == 7

    def test_pitcher_hand_row_right(self):
        rows = [_row("pitcher_hand", {"hand": "R"})]
        result = rows_to_split_filters(rows)
        assert result.pitcher_hand == "R"

    def test_pitcher_hand_row_left(self):
        rows = [_row("pitcher_hand", {"hand": "L"})]
        result = rows_to_split_filters(rows)
        assert result.pitcher_hand == "L"

    def test_batter_hand_row_right(self):
        rows = [_row("batter_hand", {"hand": "R"})]
        result = rows_to_split_filters(rows)
        assert result.batter_hand == "R"

    def test_batter_hand_row_left(self):
        rows = [_row("batter_hand", {"hand": "L"})]
        result = rows_to_split_filters(rows)
        assert result.batter_hand == "L"

    def test_home_away_row_home(self):
        rows = [_row("home_away", {"side": "home"})]
        result = rows_to_split_filters(rows)
        assert result.home_away == "home"

    def test_home_away_row_away(self):
        rows = [_row("home_away", {"side": "away"})]
        result = rows_to_split_filters(rows)
        assert result.home_away == "away"

    def test_month_row(self):
        rows = [_row("month", {"month": 5})]
        result = rows_to_split_filters(rows)
        assert result.month == 5

    def test_count_row_both_specified(self):
        rows = [_row("count", {"balls": 2, "strikes": 1})]
        result = rows_to_split_filters(rows)
        assert result.balls == 2
        assert result.strikes == 1

    def test_count_row_only_balls(self):
        rows = [_row("count", {"balls": 3, "strikes": None})]
        result = rows_to_split_filters(rows)
        assert result.balls == 3
        assert result.strikes is None

    def test_count_row_only_strikes(self):
        rows = [_row("count", {"balls": None, "strikes": 2})]
        result = rows_to_split_filters(rows)
        assert result.balls is None
        assert result.strikes == 2

    # --- last-writer-wins (count field override) ---

    def test_count_later_row_resets_balls_to_none(self):
        """A later count row with balls=None must clear an earlier balls value."""
        rows = [
            _row("count", {"balls": 2, "strikes": 1}, row_id="r1"),
            _row("count", {"balls": None, "strikes": 1}, row_id="r2"),
        ]
        result = rows_to_split_filters(rows)
        assert result.balls is None
        assert result.strikes == 1

    def test_count_later_row_resets_strikes_to_none(self):
        """A later count row with strikes=None must clear an earlier strikes value."""
        rows = [
            _row("count", {"balls": 1, "strikes": 2}, row_id="r1"),
            _row("count", {"balls": 1, "strikes": None}, row_id="r2"),
        ]
        result = rows_to_split_filters(rows)
        assert result.balls == 1
        assert result.strikes is None

    # --- malformed rows ---

    def test_row_missing_filter_type_is_skipped(self):
        """Rows with no 'filter_type' key are silently skipped."""
        rows = [
            {"id": "r1", "params": {"hand": "R"}},          # no filter_type
            _row("pitcher_hand", {"hand": "L"}, row_id="r2"),
        ]
        result = rows_to_split_filters(rows)
        assert result.pitcher_hand == "L"

    # --- last-writer-wins ---

    def test_two_inning_rows_last_wins(self):
        rows = [
            _row("inning", {"min": 1, "max": 6}, row_id="r1"),
            _row("inning", {"min": 5, "max": 9}, row_id="r2"),
        ]
        result = rows_to_split_filters(rows)
        assert result.inning_min == 5
        assert result.inning_max == 9

    def test_two_pitcher_hand_rows_last_wins(self):
        rows = [
            _row("pitcher_hand", {"hand": "R"}, row_id="r1"),
            _row("pitcher_hand", {"hand": "L"}, row_id="r2"),
        ]
        result = rows_to_split_filters(rows)
        assert result.pitcher_hand == "L"

    # --- unknown filter type ---

    def test_unknown_filter_type_ignored(self):
        rows = [_row("xyzzy", {"foo": "bar"})]
        result = rows_to_split_filters(rows)
        assert result == SplitFilters()

    def test_mixed_known_and_unknown_types(self):
        rows = [
            _row("month", {"month": 6}),
            _row("xyzzy", {"foo": "bar"}),
        ]
        result = rows_to_split_filters(rows)
        assert result.month == 6
        assert result.pitcher_hand is None
        assert result.batter_hand is None

    # --- unrelated fields stay None ---

    def test_inning_row_leaves_other_fields_none(self):
        rows = [_row("inning", {"min": 1, "max": 9})]
        result = rows_to_split_filters(rows)
        assert result.pitcher_hand is None
        assert result.batter_hand is None
        assert result.home_away is None
        assert result.month is None
        assert result.balls is None
        assert result.strikes is None


# ===========================================================================
# TestPrepareDf
# ===========================================================================

class TestPrepareDf:
    def test_derives_month_and_coerces_game_date_and_inning(self):
        df = pd.DataFrame(
            {
                "game_date": ["2024-04-10", "not-a-date", None],
                "inning": ["1", "2", "x"],
            }
        )
        prepared = prepare_df(df)

        assert "_month" in prepared.columns
        assert prepared["_month"].iloc[0] == 4
        assert pd.isna(prepared["_month"].iloc[1])
        assert pd.isna(prepared["_month"].iloc[2])

        assert pd.api.types.is_datetime64_any_dtype(prepared["game_date"])
        assert prepared["inning"].iloc[0] == 1
        assert prepared["inning"].iloc[1] == 2
        assert pd.isna(prepared["inning"].iloc[2])

    def test_get_prepared_df_cached_hit_and_miss(self):
        df = pd.DataFrame({"game_date": ["2024-04-10"], "inning": ["1"]})
        cache: dict[tuple[int, int, str], pd.DataFrame] = {}
        logs: list[str] = []
        key = (123, 2025, "Batter")

        first = get_prepared_df_cached(df, cache, key, log_fn=logs.append)
        second = get_prepared_df_cached(df, cache, key, log_fn=logs.append)

        assert first is second
        assert logs[0].startswith("[prepare_df] cache miss")
        assert logs[1].startswith("[prepare_df] cache hit")

    def test_get_prepared_df_cached_new_key_misses(self):
        df = pd.DataFrame({"game_date": ["2024-04-10"], "inning": ["1"]})
        cache: dict[tuple[int, int, str], pd.DataFrame] = {}
        logs: list[str] = []

        get_prepared_df_cached(df, cache, (123, 2025, "Batter"), log_fn=logs.append)
        get_prepared_df_cached(df, cache, (123, 2024, "Batter"), log_fn=logs.append)

        assert sum("cache miss" in msg for msg in logs) == 2


# ===========================================================================
# TestSummarizeFilterRows
# ===========================================================================

class TestSummarizeFilterRows:
    def test_empty_rows_shows_full_season_message(self):
        assert summarize_filter_rows([]) == "No filters (full season data)"

    def test_summary_uses_registry_labels(self):
        rows = [
            _row("inning", {"min": 5, "max": 9}, row_id="r1"),
            _row("pitcher_hand", {"hand": "L"}, row_id="r2"),
            _row("count", {"balls": 1, "strikes": 2}, row_id="r3"),
        ]
        summary = summarize_filter_rows(rows)
        assert summary == (
            "Inning range: 5-9, Pitcher handedness: L, Count: 1-2"
        )

    def test_summary_includes_batter_hand(self):
        rows = [_row("batter_hand", {"hand": "R"}, row_id="r1")]
        assert summarize_filter_rows(rows) == "Batter handedness: R"


# ===========================================================================
# TestApplyFilters — inning (existing behaviour preserved)
# ===========================================================================

class TestApplyFilters:
    def test_no_op_when_no_filters(self):
        """apply_filters with no active fields returns the identical object."""
        df = _make_df()
        result = apply_filters(df, SplitFilters())
        assert result is df  # same object — no copy made

    def test_inning_min_only(self):
        """inning_min keeps rows >= threshold."""
        result = apply_filters(_make_df(), SplitFilters(inning_min=7))
        assert list(result["inning"]) == [7, 8, 9]

    def test_inning_max_only(self):
        """inning_max keeps rows <= threshold."""
        result = apply_filters(_make_df(), SplitFilters(inning_max=3))
        assert list(result["inning"]) == [1, 2, 3]

    def test_inning_between(self):
        """Both min and max are applied together (inclusive)."""
        result = apply_filters(_make_df(), SplitFilters(inning_min=4, inning_max=6))
        assert list(result["inning"]) == [4, 5, 6]

    def test_missing_inning_col_raises_when_filter_active(self):
        """ValueError raised outside Streamlit when inning col is absent and filter is set."""
        df_no_inning = pd.DataFrame({"events": ["out", "homer"]})
        with pytest.raises(ValueError, match="inning"):
            apply_filters(df_no_inning, SplitFilters(inning_min=1))

    def test_missing_inning_col_noop_when_no_filter(self):
        """No error when inning col is absent but no inning filter is active."""
        df_no_inning = pd.DataFrame({"events": ["out", "homer"]})
        result = apply_filters(df_no_inning, SplitFilters())
        assert result is df_no_inning


# ===========================================================================
# TestApplyFiltersPitcherHand
# ===========================================================================

class TestApplyFiltersPitcherHand:
    def test_right_handed_only(self):
        # rows 0, 1, 4 have p_throws=="R" → innings 1, 2, 5
        result = apply_filters(_make_full_df(), SplitFilters(pitcher_hand="R"))
        assert result["inning"].tolist() == [1, 2, 5]

    def test_left_handed_only(self):
        # rows 2, 3, 5 have p_throws=="L" → innings 3, 4, 6
        result = apply_filters(_make_full_df(), SplitFilters(pitcher_hand="L"))
        assert result["inning"].tolist() == [3, 4, 6]

    def test_missing_col_raises(self):
        df = pd.DataFrame({"inning": [1, 2]})
        with pytest.raises(ValueError, match="p_throws"):
            apply_filters(df, SplitFilters(pitcher_hand="R"))

class TestApplyFiltersBatterHand:
    def test_batter_hand_filter_right(self):
        df = _make_full_df().copy()
        df["stand"] = ["R", "L", "R", "L", "R", "L"]
        result = apply_filters(df, SplitFilters(batter_hand="R"))
        assert result["inning"].tolist() == [1, 3, 5]

    def test_batter_hand_filter_left(self):
        df = _make_full_df().copy()
        df["stand"] = ["R", "L", "R", "L", "R", "L"]
        result = apply_filters(df, SplitFilters(batter_hand="L"))
        assert result["inning"].tolist() == [2, 4, 6]


# ===========================================================================
# TestApplyFiltersHomeAway
# ===========================================================================

class TestApplyFiltersHomeAway:
    def test_home_keeps_bot_half(self):
        # inning_topbot=="Bot" → rows 1, 3, 4 → innings 2, 4, 5
        result = apply_filters(_make_full_df(), SplitFilters(home_away="home"))
        assert result["inning"].tolist() == [2, 4, 5]

    def test_away_keeps_top_half(self):
        # inning_topbot=="Top" → rows 0, 2, 5 → innings 1, 3, 6
        result = apply_filters(_make_full_df(), SplitFilters(home_away="away"))
        assert result["inning"].tolist() == [1, 3, 6]

    def test_missing_col_raises(self):
        df = pd.DataFrame({"inning": [1, 2]})
        with pytest.raises(ValueError, match="inning_topbot"):
            apply_filters(df, SplitFilters(home_away="home"))

    def test_home_away_pitcher_perspective_inverts_mapping(self):
        home = apply_filters(
            _make_full_df(),
            SplitFilters(home_away="home"),
            pitcher_perspective=True,
        )
        away = apply_filters(
            _make_full_df(),
            SplitFilters(home_away="away"),
            pitcher_perspective=True,
        )
        assert home["inning"].tolist() == [1, 3, 6]
        assert away["inning"].tolist() == [2, 4, 5]

    def test_home_away_batter_default_mapping_unchanged(self):
        home = apply_filters(_make_full_df(), SplitFilters(home_away="home"))
        away = apply_filters(_make_full_df(), SplitFilters(home_away="away"))
        assert home["inning"].tolist() == [2, 4, 5]
        assert away["inning"].tolist() == [1, 3, 6]


# ===========================================================================
# TestApplyFiltersMonth
# ===========================================================================

class TestApplyFiltersMonth:
    def test_april(self):
        # game_date months: Apr, Apr, May, May, Jun, Jul
        # month=4 → rows 0, 1 → innings 1, 2
        result = apply_filters(_make_full_df(), SplitFilters(month=4))
        assert result["inning"].tolist() == [1, 2]

    def test_may(self):
        # month=5 → rows 2, 3 → innings 3, 4
        result = apply_filters(_make_full_df(), SplitFilters(month=5))
        assert result["inning"].tolist() == [3, 4]

    def test_month_with_datetime_dtype(self):
        """game_date as actual datetime objects (not strings) should also work."""
        df = _make_full_df()
        df["game_date"] = pd.to_datetime(df["game_date"])
        result = apply_filters(df, SplitFilters(month=6))
        assert result["inning"].tolist() == [5]

    def test_missing_col_raises(self):
        df = pd.DataFrame({"inning": [1, 2]})
        with pytest.raises(ValueError, match="game_date"):
            apply_filters(df, SplitFilters(month=4))

    def test_month_filter_works_with_prepared_month_column(self):
        df = _make_full_df().drop(columns=["game_date"])
        df["_month"] = [4, 4, 5, 5, 6, 7]
        result = apply_filters(df, SplitFilters(month=4))
        assert result["inning"].tolist() == [1, 2]


# ===========================================================================
# TestApplyFiltersBalls
# ===========================================================================

class TestApplyFiltersBalls:
    def test_zero_balls(self):
        # balls==0 → rows 0, 4 → innings 1, 5
        result = apply_filters(_make_full_df(), SplitFilters(balls=0))
        assert result["inning"].tolist() == [1, 5]

    def test_one_ball(self):
        # balls==1 → rows 1, 5 → innings 2, 6
        result = apply_filters(_make_full_df(), SplitFilters(balls=1))
        assert result["inning"].tolist() == [2, 6]

    def test_missing_col_raises(self):
        df = pd.DataFrame({"inning": [1, 2]})
        with pytest.raises(ValueError, match="balls"):
            apply_filters(df, SplitFilters(balls=0))


# ===========================================================================
# TestApplyFiltersStrikes
# ===========================================================================

class TestApplyFiltersStrikes:
    def test_two_strikes(self):
        # strikes==2 → rows 2, 5 → innings 3, 6
        result = apply_filters(_make_full_df(), SplitFilters(strikes=2))
        assert result["inning"].tolist() == [3, 6]

    def test_zero_strikes(self):
        # strikes==0 → rows 0, 3 → innings 1, 4
        result = apply_filters(_make_full_df(), SplitFilters(strikes=0))
        assert result["inning"].tolist() == [1, 4]

    def test_missing_col_raises(self):
        df = pd.DataFrame({"inning": [1, 2]})
        with pytest.raises(ValueError, match="strikes"):
            apply_filters(df, SplitFilters(strikes=0))


# ===========================================================================
# TestApplyFiltersAndCombination
# ===========================================================================

class TestApplyFiltersAndCombination:
    def test_pitcher_hand_and_home_away(self):
        # pitcher_hand="R"  → rows 0, 1, 4 (innings 1, 2, 5)
        # home_away="home"  → rows 1, 3, 4 (innings 2, 4, 5)
        # intersection      → rows 1, 4 (innings 2, 5)
        result = apply_filters(
            _make_full_df(),
            SplitFilters(pitcher_hand="R", home_away="home"),
        )
        assert result["inning"].tolist() == [2, 5]

    def test_balls_and_strikes(self):
        # balls==1 → rows 1, 5 (innings 2, 6)
        # strikes==1 → rows 1, 4 (innings 2, 5)
        # intersection → row 1 (inning 2)
        result = apply_filters(_make_full_df(), SplitFilters(balls=1, strikes=1))
        assert result["inning"].tolist() == [2]

    def test_inning_and_pitcher_hand(self):
        # inning_max=3 → rows 0, 1, 2 (innings 1, 2, 3)
        # pitcher_hand="R" → rows 0, 1 (innings 1, 2)
        result = apply_filters(
            _make_full_df(),
            SplitFilters(inning_max=3, pitcher_hand="R"),
        )
        assert result["inning"].tolist() == [1, 2]

    def test_month_and_home_away(self):
        # month=4 → rows 0, 1 (innings 1, 2)
        # home_away="away" (Top) → rows 0, 2, 5 (innings 1, 3, 6)
        # intersection → row 0 (inning 1)
        result = apply_filters(
            _make_full_df(),
            SplitFilters(month=4, home_away="away"),
        )
        assert result["inning"].tolist() == [1]

    def test_no_matching_rows_returns_empty(self):
        # month=9 is not in _make_full_df (Jun is max month == 7)
        result = apply_filters(_make_full_df(), SplitFilters(month=9))
        assert result.empty

    def test_all_filters_active_and_combine(self):
        # Target row 1: inning=2, p_throws=R, topbot=Bot, game_date=Apr, balls=1, strikes=1
        result = apply_filters(
            _make_full_df(),
            SplitFilters(
                inning_min=1,
                inning_max=3,
                pitcher_hand="R",
                home_away="home",
                month=4,
                balls=1,
                strikes=1,
            ),
        )
        assert result["inning"].tolist() == [2]
