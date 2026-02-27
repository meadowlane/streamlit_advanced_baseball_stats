"""Unit tests for the rule-based NL query parser."""

from __future__ import annotations

from stats.nl_query import parse_nl_query


PLAYER_NAMES = [
    "Gunnar Henderson",
    "Dylan Beavers",
    "Aaron Judge",
    "Pete Alonso",
    "Mike Trout",
    "Mike Tauchman",
]

VALID_SEASONS = {2025, 2024, 2023, 2022}


def _parse(query: str) -> dict:
    return parse_nl_query(
        query,
        PLAYER_NAMES,
        valid_seasons=VALID_SEASONS,
    )


def _filter_params(parsed: dict, filter_type: str):
    for row in parsed["filter_rows"]:
        if row["filter_type"] == filter_type:
            return row["params"]
    return None


def test_simple_player_only():
    parsed = _parse("Gunnar Henderson")
    assert parsed["player_a"] == "Gunnar Henderson"
    assert parsed["comparison_mode"] is False
    assert parsed["filter_rows"] == []


def test_comparison_vs_pattern():
    parsed = _parse("Gunnar Henderson vs Dylan Beavers")
    assert parsed["comparison_mode"] is True
    assert parsed["player_a"] == "Gunnar Henderson"
    assert parsed["player_b"] == "Dylan Beavers"


def test_comparison_compare_and_pattern():
    parsed = _parse("compare Gunnar Henderson and Dylan Beavers")
    assert parsed["comparison_mode"] is True
    assert parsed["player_a"] == "Gunnar Henderson"
    assert parsed["player_b"] == "Dylan Beavers"


def test_ambiguous_comparison_keeps_mode_enabled():
    parsed = _parse("compare Mike and Dylan Beavers")
    assert parsed["comparison_mode"] is True
    assert parsed["player_a"] in {"Mike Trout", "Mike Tauchman"}
    assert parsed["player_b"] == "Dylan Beavers"
    assert any("matched multiple players" in warning.lower() for warning in parsed["warnings"])


def test_parses_season_year():
    parsed = _parse("Gunnar Henderson 2024")
    assert parsed["season"] == 2024


def test_invalid_season_is_dropped_with_warning():
    parsed = _parse("Gunnar Henderson 2019")
    assert parsed["season"] is None
    assert any("season 2019" in warning.lower() for warning in parsed["warnings"])


def test_stats_detection_in_core_order():
    parsed = _parse("Gunnar Henderson barrel% xwOBA wOBA")
    assert parsed["selected_stats"] == ["wOBA", "xwOBA", "Barrel%"]


def test_no_stats_mentioned_keeps_stats_empty():
    parsed = _parse("Gunnar Henderson inning 7+")
    assert parsed["selected_stats"] == []


def test_pitcher_hand_left_detection():
    parsed = _parse("Gunnar Henderson vs LHP")
    assert _filter_params(parsed, "pitcher_hand") == {"hand": "L"}


def test_pitcher_hand_right_detection():
    parsed = _parse("Gunnar Henderson vs righties")
    assert _filter_params(parsed, "pitcher_hand") == {"hand": "R"}


def test_inning_plus_detection():
    parsed = _parse("Gunnar Henderson inning 7+")
    assert _filter_params(parsed, "inning") == {"min": 7, "max": 9}


def test_inning_range_detection():
    parsed = _parse("Gunnar Henderson innings 7-9")
    assert _filter_params(parsed, "inning") == {"min": 7, "max": 9}


def test_home_filter_detection():
    parsed = _parse("Gunnar Henderson at home")
    assert _filter_params(parsed, "home_away") == {"side": "home"}


def test_away_filter_detection():
    parsed = _parse("Gunnar Henderson on the road")
    assert _filter_params(parsed, "home_away") == {"side": "away"}


def test_month_filter_detection():
    parsed = _parse("Gunnar Henderson May")
    assert _filter_params(parsed, "month") == {"month": 5}


def test_count_pair_detection():
    parsed = _parse("Gunnar Henderson 2-1 count")
    assert _filter_params(parsed, "count") == {"balls": 2, "strikes": 1}


def test_full_count_detection():
    parsed = _parse("Gunnar Henderson full count")
    assert _filter_params(parsed, "count") == {"balls": 3, "strikes": 2}


def test_balls_any_strikes_detection():
    parsed = _parse("Gunnar Henderson 3 balls any strikes")
    assert _filter_params(parsed, "count") == {"balls": 3, "strikes": None}


def test_last_writer_wins_for_same_filter_type():
    parsed = _parse("Gunnar Henderson vs LHP vs RHP")
    assert _filter_params(parsed, "pitcher_hand") == {"hand": "R"}


def test_multiple_filters_combined():
    parsed = _parse("Gunnar Henderson inning 7+ at home May vs LHP 2-1 count")
    assert parsed["comparison_mode"] is False
    assert _filter_params(parsed, "inning") == {"min": 7, "max": 9}
    assert _filter_params(parsed, "home_away") == {"side": "home"}
    assert _filter_params(parsed, "month") == {"month": 5}
    assert _filter_params(parsed, "pitcher_hand") == {"hand": "L"}
    assert _filter_params(parsed, "count") == {"balls": 2, "strikes": 1}


def test_filter_phrase_vs_lhp_does_not_trigger_comparison():
    parsed = _parse("Gunnar Henderson inning 7-9 vs LHP")
    assert parsed["comparison_mode"] is False
    assert parsed["player_a"] == "Gunnar Henderson"


def test_regression_pete_alonzo_vs_gunnar_in_2024():
    parsed = _parse("Pete Alonzo vs Gunnar in 2024")
    assert parsed["season"] == 2024
    assert parsed["cleaned_query"] == "Pete Alonzo vs Gunnar"
    assert parsed["comparison_mode"] is True
    assert parsed["player_a_fragment"] == "Pete Alonzo"
    assert parsed["player_b_fragment"] == "Gunnar"
    assert " in" not in parsed["player_b_fragment"].lower()


def test_regression_pete_alonso_vs_gunnar_henderson_2024():
    parsed = _parse("Pete Alonso vs Gunnar Henderson 2024")
    assert parsed["season"] == 2024
    assert parsed["comparison_mode"] is True
    assert parsed["player_a_fragment"] == "Pete Alonso"
    assert parsed["player_b_fragment"] == "Gunnar Henderson"
    assert parsed["player_a"] == "Pete Alonso"
    assert parsed["player_b"] == "Gunnar Henderson"


def test_regression_compare_pete_alonso_and_gunnar_henderson_in_2024():
    parsed = _parse("compare Pete Alonso and Gunnar Henderson in 2024")
    assert parsed["season"] == 2024
    assert parsed["cleaned_query"] == "compare Pete Alonso and Gunnar Henderson"
    assert parsed["comparison_mode"] is True
    assert parsed["player_a_fragment"] == "Pete Alonso"
    assert parsed["player_b_fragment"] == "Gunnar Henderson"


def test_comparison_kept_when_b_unresolved():
    parsed = parse_nl_query(
        "Pete Alonso vs Gunnar in 2024",
        ["Pete Alonso", "Aaron Judge"],
        valid_seasons=VALID_SEASONS,
    )
    assert parsed["comparison_mode"] is True
    assert parsed["player_a"] == "Pete Alonso"
    assert parsed["player_b"] is None
    assert any("could not resolve player b" in warning.lower() for warning in parsed["warnings"])
