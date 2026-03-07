import ast
from functools import lru_cache
from pathlib import Path

import pytest

from stats.filters import SplitFilters
from ui.components import _pitch_zone_hand_conflict_note


@lru_cache(maxsize=1)
def _load_app_filter_helpers():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_path))

    wanted_functions = {
        "_pitch_mix_batter_hand_ui_state",
        "_build_pitcher_group_filter_caption",
        "_headline_filter_note",
        "_split_filter_warning",
        "_count_filter_note",
        "_trend_filter_note",
    }
    wanted_assignments = {"_BATTER_STAT_LABELS"}

    selected = []
    found_functions = set()
    found_assignments = set()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected.append(node)
            found_functions.add(node.name)
        elif isinstance(node, ast.Assign):
            target_names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if target_names & wanted_assignments:
                selected.append(node)
                found_assignments.update(target_names & wanted_assignments)

    assert found_functions == wanted_functions
    assert found_assignments == wanted_assignments

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"SplitFilters": SplitFilters}
    exec(compile(module, filename=str(app_path), mode="exec"), namespace)
    return namespace


def test_pitch_mix_sidebar_hand_hides_local_control():
    namespace = _load_app_filter_helpers()

    show_control, note = namespace["_pitch_mix_batter_hand_ui_state"](
        SplitFilters(batter_hand="L")
    )

    assert show_control is False
    assert note == "Batter hand controlled by sidebar filter (vs LHB)."


def test_pitch_mix_local_control_stays_available_without_sidebar_hand_filter():
    namespace = _load_app_filter_helpers()

    show_control, note = namespace["_pitch_mix_batter_hand_ui_state"](SplitFilters())

    assert show_control is True
    assert note is None


def test_pitcher_group_caption_names_season_and_filter_stats():
    namespace = _load_app_filter_helpers()

    caption = namespace["_build_pitcher_group_filter_caption"](
        ["ERA", "wOBA", "xwOBA"],
        {"ERA", "FIP", "xFIP", "SIERA", "xERA"},
        {"wOBA": "wOBA Allowed", "xwOBA": "xwOBA Allowed"},
    )

    assert caption == "Season-level: ERA. Filter-affected: wOBA Allowed, xwOBA Allowed."


def test_headline_filter_note_only_appears_with_active_filters():
    namespace = _load_app_filter_helpers()

    assert namespace["_headline_filter_note"](SplitFilters()) is None
    assert (
        namespace["_headline_filter_note"](SplitFilters(month=4))
        == "Headline stats are full-season; filters apply to cards below."
    )


@pytest.mark.parametrize(
    ("split_type", "filters", "expected"),
    [
        (
            "hand",
            SplitFilters(batter_hand="R"),
            "Active Batter hand filter makes this split degenerate; all rows already reflect vs RHB.",
        ),
        (
            "home_away",
            SplitFilters(home_away="home"),
            "Active Home / Away filter makes this split degenerate; all rows already reflect Home.",
        ),
        (
            "monthly",
            SplitFilters(month=4),
            "Active Month filter makes this split degenerate; all rows already reflect Apr.",
        ),
    ],
)
def test_split_filter_warning_detects_degenerate_split(split_type, filters, expected):
    namespace = _load_app_filter_helpers()

    assert namespace["_split_filter_warning"](split_type, filters) == expected


def test_count_filter_note_only_appears_when_count_filter_is_active():
    namespace = _load_app_filter_helpers()

    assert namespace["_count_filter_note"](SplitFilters()) is None
    assert (
        namespace["_count_filter_note"](SplitFilters(balls=3, strikes=2))
        == "Count filter active: rate stats (K%, BB%, wOBA, etc.) reflect per-pitch frequency in this count, not traditional PA rates."
    )


def test_trend_filter_note_only_appears_for_unapplied_active_filters():
    namespace = _load_app_filter_helpers()

    assert namespace["_trend_filter_note"](SplitFilters(), False) is None
    assert namespace["_trend_filter_note"](SplitFilters(month=5), True) is None
    assert (
        namespace["_trend_filter_note"](SplitFilters(month=5), False)
        == "Trend data currently shows full-season stats. Enable 'Apply current filters to each year' to reflect sidebar filters."
    )


def test_batter_wrc_plus_label_marks_season_scope():
    namespace = _load_app_filter_helpers()

    assert namespace["_BATTER_STAT_LABELS"]["wRC+"] == "wRC+ (season)"


def test_pitch_zone_conflict_note_detects_sidebar_batter_hand_conflict():
    note = _pitch_zone_hand_conflict_note(
        "pitcher",
        SplitFilters(batter_hand="L"),
        "vs RHB",
    )

    assert (
        note
        == "Pitch Location Batter hand filter conflicts with the sidebar batter hand filter (vs LHB). Clear one filter to see data."
    )


def test_pitch_zone_conflict_note_allows_matching_filter():
    note = _pitch_zone_hand_conflict_note(
        "batter",
        SplitFilters(pitcher_hand="R"),
        "vs RHP",
    )

    assert note is None
