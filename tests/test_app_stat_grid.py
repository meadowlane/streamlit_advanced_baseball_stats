import ast
import math
from pathlib import Path


class _DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyStreamlit:
    def info(self, _message):
        return None

    def columns(self, count):
        return [_DummyColumn() for _ in range(count)]

    def empty(self):
        return None


def _load_stat_grid_functions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_path))

    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_chunk_stats",
            "_render_player_stat_grid",
        }:
            selected.append(node)

    assert (
        len(selected) == 2
    ), "Expected to find both _chunk_stats and _render_player_stat_grid in app.py"

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"_GRID_COLS_PER_ROW": 3}
    exec(compile(module, filename=str(app_path), mode="exec"), namespace)
    return namespace


def test_render_player_stat_grid_missing_percentile_uses_nan_default_without_nameerror():
    namespace = _load_stat_grid_functions()
    st = _DummyStreamlit()
    calls = []

    def fake_stat_card(**kwargs):
        calls.append(kwargs)

    namespace["st"] = st
    namespace["stat_card"] = fake_stat_card

    render_fn = namespace["_render_player_stat_grid"]
    render_fn(
        stat_values={"K%": 18.2},
        percentiles={},
        color_tiers={},
        stats_order=["K%"],
    )

    assert len(calls) == 1
    assert math.isnan(calls[0]["percentile"])
