import ast
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from stats.splits import _compute_traditional_stats


@lru_cache(maxsize=1)
def _load_app_traditional_stat_helpers():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_path))

    wanted_functions = {"_as_optional_float", "_populate_batter_traditional_stats"}
    wanted_assignments = {"TRADITIONAL_STATS"}

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
    namespace = {
        "pd": pd,
        "_compute_traditional_stats": _compute_traditional_stats,
    }
    exec(compile(module, filename=str(app_path), mode="exec"), namespace)
    return namespace


def test_no_filters_keep_fangraphs_traditional_stats():
    namespace = _load_app_traditional_stat_helpers()
    helper = namespace["_populate_batter_traditional_stats"]

    player_row = pd.Series(
        {
            "AVG": 0.301,
            "OBP": 0.412,
            "SLG": 0.588,
            "OPS": 1.000,
            "HR": 41,
            "RBI": 108,
            "wRC+": 171,
        }
    )
    filtered_df = pd.DataFrame({"events": ["home_run", "home_run", "walk", "strikeout"]})

    result = helper({}, player_row, filtered_df, False)

    for stat in namespace["TRADITIONAL_STATS"]:
        assert result[stat] == pytest.approx(float(player_row[stat]), abs=0.001)
    assert result["wRC+"] == pytest.approx(171.0, abs=0.001)


def test_active_filters_use_filtered_traditional_stats_but_keep_season_only_values():
    namespace = _load_app_traditional_stat_helpers()
    helper = namespace["_populate_batter_traditional_stats"]

    player_row = pd.Series(
        {
            "AVG": 0.250,
            "OBP": 0.340,
            "SLG": 0.430,
            "OPS": 0.770,
            "HR": 22,
            "RBI": 91,
            "wRC+": 128,
        }
    )
    filtered_df = pd.DataFrame(
        {
            "events": [
                "home_run",
                "single",
                "walk",
                "strikeout",
                "field_out",
            ]
        }
    )

    expected = _compute_traditional_stats(filtered_df)
    result = helper({}, player_row, filtered_df, True)

    for stat in ["AVG", "OBP", "SLG", "OPS", "HR"]:
        assert result[stat] == expected[stat]
    assert result["AVG"] != pytest.approx(float(player_row["AVG"]), abs=0.001)
    assert result["RBI"] == pytest.approx(91.0, abs=0.001)
    assert result["wRC+"] == pytest.approx(128.0, abs=0.001)


def test_app_reuses_same_batter_traditional_helper_for_player_a_and_b():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))

    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_populate_batter_traditional_stats"
    ]

    assert len(helper_calls) == 2
