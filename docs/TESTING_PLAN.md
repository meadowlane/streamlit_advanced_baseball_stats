# MLB Splits App — Testing Master Plan

## Executive Summary

1. **Strong foundation, critical gaps:** 289 tests, ~2,700 lines across 8 modules; core business logic (stats/, data/) is comprehensively unit-tested. Zero CI/CD, no coverage tooling, no type checking, no linting.
2. **Three new test files deliver the highest ROI:** `tests/conftest.py` (shared fixtures), `tests/test_pipeline.py` (integration), `tests/test_contracts.py` (schema invariants). These fill the most dangerous untested territory.
3. **The critical correctness risk is the PROPORTION_STATS scale:** K%/BB%/HardHit%/Barrel% are stored as 0–1 in FanGraphs but 0–100 in Statcast. A silent mismatch produces wrong percentile rankings. No test currently validates this cross-module invariant.
4. **Pitcher `player_type` direction inversion is the second-highest risk:** High K% is good for pitchers (bad for batters). `get_all_percentiles(player_type="Pitcher")` must invert the direction. No integration test validates the full path.
5. **`apply_filters` fast-path and `pitcher_perspective` inversion are critical behavioral contracts** — both are tested at unit level but not in the full pipeline. Add explicit invariant assertions.
6. **New dependencies needed:** `pytest-cov>=5.0` (coverage), `ruff>=0.4.0` (lint+format), `mypy>=1.10` (type check). No VCR cassettes, no Playwright, no Hypothesis — the existing `@patch` strategy is correct.
7. **`pyproject.toml` is the single most impactful file to create:** consolidates pytest config, coverage settings, ruff, and mypy into one place with zero migration cost.
8. **GitHub Actions CI in `.github/workflows/ci.yml`** runs three parallel jobs (lint, typecheck, tests+coverage) on every PR. Coverage gate starts at measured baseline; tightened each phase.
9. **Streamlit-specific E2E testing is deliberately minimal:** use AST extraction (already pioneered in `test_app_stat_grid.py`) for app.py helpers. No Playwright/Selenium — AGENTS.md prohibits complex infra.
10. **4-phase roadmap, each a PR-sized chunk:** Phase 0 scaffolding → Phase 1 coverage baseline + contracts → Phase 2 integration pipeline tests → Phase 3 smoke + extended E2E.

---

## Current State Audit

| Module | LOC | Test Lines | Est. Coverage | Complexity |
|---|---|---|---|---|
| `stats/filters.py` | 427 | 661 | ~90% | Medium |
| `stats/splits.py` | 644 | 667 | ~85% | Medium |
| `stats/percentiles.py` | 193 | 344 | ~92% | Simple |
| `stats/nl_query.py` | 672 | 229 | ~65% | Complex |
| `data/fetcher.py` | 374 | 459 | ~70% | Medium |
| `ui/components.py` | 2177 | 292 | ~35% | High |
| `ui/glossary.py` | 379 | 0 | ~0% | Simple |
| `app.py` | 1528 | 71 (AST) | ~3% | High |

**Missing infrastructure:** No CI/CD, no coverage config, no mypy, no ruff, no `conftest.py`, no `pyproject.toml`, no integration tests, no E2E tests.

---

## Section 1 — Test Pyramid & Scope

### Layer 1: Unit Tests (existing — fill gaps)
- **Validates:** Individual pure functions in isolation
- **Mock vs real:** All pybaseball calls mocked with `@patch`; synthetic DataFrames in-memory
- **Current:** 289 tests, 0.90 s
- **Target:** 340+ tests, < 5 s
- **Gaps to fill:** `ui/glossary.py` (0 tests), `stats/nl_query.py` edge cases (regex coverage, all month variants, filter_type key validity), `ui/components.py` module-level dict completeness

### Layer 2: Integration Tests (new — `tests/test_pipeline.py`)
- **Validates:** Full pipeline: `fetch_fn stub → prepare_df → apply_filters → _compute_stats → build_league_distributions → get_all_percentiles`
- **Mock vs real:** `fetch_fn` is always a stub; all other code runs real (no mocking inside pipeline)
- **Target:** 15–20 tests, < 10 s
- **Key scenarios:** Batter pipeline end-to-end, pitcher perspective inversion, proportion scale consistency, trend cache sharing, filter propagation across seasons

### Layer 3: Contract Tests (new — `tests/test_contracts.py`)
- **Validates:** Shape/key contracts at module boundaries — "what A produces, B must accept"
- **Mock vs real:** Pure unit tests, no network; uses shared conftest fixtures
- **Target:** 10–15 tests, < 1 s
- **Key contracts:** `_compute_stats` output keys == `STAT_REGISTRY.keys() | {"PA"}`, `get_splits` column set == `SPLIT_COLS`, `WOBA_WEIGHTS` covers exactly 6 credited events and is monotonically increasing, `nl_query` filter_type values are always valid `FILTER_REGISTRY` keys

### Layer 4: Smoke Tests (new — `tests/test_smoke.py`)
- **Validates:** Module import safety; module-level dict completeness; architectural boundary (stats/ must never import streamlit)
- **Mock vs real:** Uses AST parsing — no imports required
- **Target:** 10–12 tests, < 3 s

### Layer 5: Performance Benchmarks (deferred — Phase 3+)
- **Validates:** `apply_filters` and `_compute_stats` on 50,000-row DataFrames < 200 ms
- **File:** `tests/test_benchmarks.py` with `@pytest.mark.slow`
- **CI:** Nightly only, never on PR

**Runtime budget (PR gate):** Total ≤ 30 s for Layers 1–4 combined

---

## Section 2 — Targeted Coverage Map

### Top Critical User-Visible Flows

| Flow | Subsystems | Coverage Gap |
|---|---|---|
| Single player stat view | fetcher → prepare_df → apply_filters → _compute_stats → get_all_percentiles → stat_card | No test crosses the full chain |
| Split table (vs L/R/Home/Away) | apply_filters → get_splits → split_table | No test chains filter + split_type together |
| Trend dashboard | get_trend_stats (all 3 components) → _build_trend_tidy_df → _build_single_stat_chart | Cache sharing across seasons untested |
| NL query → session state | parse_nl_query → app.py key writes | filter_type key validity contract untested |
| Pitcher comparison (cross-season) | two fetches → build_pitcher_league_distributions → get_all_percentiles(player_type="Pitcher") | Pitcher K% direction untested end-to-end |

### Must-Test Invariants

```python
# INV-1: apply_filters fast-path — no filters = same object
result = apply_filters(df, SplitFilters())
assert result is df

# INV-2: pitcher_perspective inversion — selects opposite rows
home_batter = apply_filters(df, SplitFilters(home_away="home"), pitcher_perspective=False)
home_pitcher = apply_filters(df, SplitFilters(home_away="home"), pitcher_perspective=True)
assert set(home_batter.index).isdisjoint(set(home_pitcher.index))

# INV-3: compute_percentile is monotonically non-decreasing
dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
pcts = [compute_percentile(v, dist, higher_is_better=True) for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
assert pcts == sorted(pcts)

# INV-4: NaN/None → gray color tier
assert get_color_tier(float("nan"))["name"] == "gray"
assert get_color_tier(None)["name"] == "gray"

# INV-5: PROPORTION_STATS scale consistency (most critical cross-module invariant)
stats = _compute_stats(statcast_df_factory(n=100))   # must be 0-100 scale
dists = build_league_distributions(season_df_factory()) # must be 0-100 scale
assert stats["K%"] > 1.0                             # NOT 0-1 proportion
assert dists["K%"].max() > 1.0                       # NOT 0-1 proportion

# INV-6: split_by_hand always returns exactly 2 rows with correct labels
result = split_by_hand(make_statcast_df(n))
assert len(result) == 2
assert set(result["Split"]) == {"vs RHP", "vs LHP"}

# INV-7: prepare_cache hit returns identical object
cache = {}
key = (player_id, 2024, "Batter")
first = get_prepared_df_cached(raw, cache, key)
second = get_prepared_df_cached(raw, cache, key)
assert first is second

# INV-8: nl_query filter_type values are always FILTER_REGISTRY keys
for row in parse_nl_query(query, names, ...)["filter_rows"]:
    assert row["filter_type"] in FILTER_REGISTRY

# INV-9: WOBA_WEIGHTS covers exactly credited events and is monotonically increasing
assert set(WOBA_WEIGHTS.keys()) == {"walk", "hit_by_pitch", "single", "double", "triple", "home_run"}
assert list(WOBA_WEIGHTS.values()) == sorted(WOBA_WEIGHTS.values())

# INV-10: pitcher K% direction inverted vs batter K%
pitcher_pcts = get_all_percentiles({"K%": 30.0}, dists, player_type="Pitcher")
batter_pcts = get_all_percentiles({"K%": 30.0}, dists, player_type="Batter")
assert pitcher_pcts["K%"] > batter_pcts["K%"]  # high K% is good for pitchers
```

---

## Section 3 — Tooling & Structure

### New Dependencies (add to `requirements.txt`)

| Package | Version | Justification |
|---|---|---|
| `pytest-cov` | `>=5.0` | **Required** — coverage measurement; zero new behavior |
| `ruff` | `>=0.4.0` | Replaces flake8 + isort + black in one tool; sub-1s on this codebase |
| `mypy` | `>=1.10` | Existing type annotations are thorough; mypy locks them in; `pandas-stubs` already installed in venv |

**Do NOT add:** `VCR.py` (existing `@patch` is correct), `pytest-xdist` (suite is fast), `Playwright/Selenium` (too heavy per AGENTS.md), `Hypothesis` (deferred), `pytest-benchmark` (deferred).

### Final Test Directory Structure

```
tests/
    __init__.py                   # existing
    conftest.py                   # NEW — shared fixtures
    test_filters.py               # existing — no changes
    test_splits.py                # existing — no changes
    test_percentiles.py           # existing — no changes
    test_nl_query.py              # existing + 6-8 edge cases
    test_fetcher.py               # existing — no changes
    test_components.py            # existing + glossary dict tests
    test_placeholder.py           # existing — no changes
    test_app_stat_grid.py         # existing + _chunk_stats invariants
    test_pipeline.py              # NEW — integration tests (Phase 2)
    test_contracts.py             # NEW — schema contracts (Phase 1)
    test_smoke.py                 # NEW — import safety (Phase 1)
```

### `tests/conftest.py` Fixture Signatures

```python
@pytest.fixture
def statcast_df_factory():
    """Returns factory function for synthetic Statcast-shaped DataFrames.

    def _factory(
        n: int = 30,
        p_throws: str | list = "R",        # pitcher handedness
        stand: str | list = "R",            # batter stance
        inning_topbot: str | list = "Bot",  # Bot=home bats, Top=away bats
        month: int | list = 4,              # calendar month(s)
        include_pitch_cols: bool = False,   # add pitch_type, release_speed etc.
    ) -> pd.DataFrame
    """

@pytest.fixture
def fg_batting_df_factory():
    """Returns factory function for FanGraphs-shaped season DataFrames.

    def _factory(
        names: tuple = ("Aaron Judge", "Shohei Ohtani"),
        season: int = 2024,
        proportion_scale: bool = True,   # True = 0-1 (FG), False = 0-100
    ) -> pd.DataFrame
    """

@pytest.fixture
def season_df_factory():
    """Returns factory function for percentile-testing season DataFrames.

    def _factory(n: int = 100, seed: int = 42) -> pd.DataFrame
    Proportion stats on 0-1 scale (FG convention); seeded for determinism.
    """

@pytest.fixture
def empty_statcast_df():
    """Zero-row Statcast DataFrame with correct column schema."""

@pytest.fixture
def make_trend_stub():
    """Factory that returns a get_trend_stats-compatible fetch_fn stub.

    Usage: stub = make_trend_stub(statcast_df_factory(n=50))
    def stub(mlbam_id: int, season: int) -> pd.DataFrame
    """
```

**Decision:** Existing test files keep their private `_make_*` helpers — do not refactor them. New conftest fixtures are used only in new test files. Incremental migration is a future cleanup PR.

### pytest Marks Strategy

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "unit: pure function tests with no I/O",
    "integration: multi-module pipeline tests",
    "contract: schema and interface contract tests",
    "smoke: import-safety and module-level validation",
    "slow: performance/benchmark tests (nightly only)",
]
addopts = "--tb=short"
testpaths = ["tests"]
```

- PR CI: `pytest -m "not slow"` — runs all except benchmarks
- Nightly: `pytest` — runs everything
- Local fast: `pytest -m "unit"` — run only pure unit tests

### Snapshot Strategy

**For Plotly figures:** Do NOT use snapshot files — Plotly JSON is verbose and changes with every library upgrade. Assert structural properties instead:
```python
fig = _build_single_stat_chart(...)
assert len(fig.data) >= 1
assert fig.data[0].connectgaps is False
assert fig.layout.xaxis.title.text is not None
```
This is already the pattern in `test_components.py` — keep it.

**For DataFrames:** Assert column sets and dtypes, not raw values:
```python
result = get_splits(df, "hand")
assert list(result.columns) == SPLIT_COLS
assert pd.api.types.is_numeric_dtype(result["PA"])
```

**For glossary content:** Inline assertions against known prose (not snapshot files):
```python
assert "wOBA" in STAT_DEFINITIONS
assert "short" in STAT_DEFINITIONS["wOBA"]
```

---

## Section 4 — CI/CD Plan

### `.github/workflows/ci.yml` (complete)

```yaml
name: CI

on:
  push:
    branches: [main, "feature/**"]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 6 * * *"   # nightly 6 AM UTC

env:
  PYTHON_VERSION: "3.13"

jobs:
  lint:
    name: Lint (ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install "ruff>=0.4.0"
      - run: ruff check .
      - run: ruff format --check .

  typecheck:
    name: Type check (mypy)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt
      - run: mypy stats/ data/ ui/ --ignore-missing-imports --no-strict-optional

  test:
    name: Tests + Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt
      - name: Run test suite (not slow)
        run: |
          python -m pytest tests/ \
            -m "not slow" \
            --cov=stats --cov=data --cov=ui \
            --cov-report=term-missing \
            --cov-report=xml:coverage.xml \
            --cov-fail-under=80 \
            -q
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-report
          path: coverage.xml

  nightly:
    name: Nightly full suite
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt
      - run: |
          python -m pytest tests/ \
            --cov=stats --cov=data --cov=ui --cov=app \
            --cov-report=term-missing \
            --cov-fail-under=80 \
            -v
```

### Quality Gates

| Gate | PR | Nightly |
|---|---|---|
| `ruff check` (no errors) | Required | Required |
| `ruff format --check` (no changes) | Required | Required |
| `mypy` on `stats/ data/ ui/` | Required | Required |
| `pytest -m "not slow"` passes | Required | Required |
| Coverage ≥ 80% (`stats/` + `data/` + `ui/`) | Required | Required |
| Coverage ≥ 80% (including `app.py`) | Not gated | Reported |
| `pytest` (all marks) passes | Not run | Required |

**Coverage gate trajectory:** Phase 0 → measure baseline → Phase 1 → gate at (baseline - 0)% → Phase 2 → raise to 82% → Phase 3 → raise to 85%

**Note:** Do NOT include `app.py` in the PR coverage gate — it is 1528 lines of Streamlit-entangled code with ~3% testable coverage. Include it in nightly reporting only.

---

## Section 5 — Phased Roadmap

### Phase 0: Infrastructure Scaffolding (1 PR)

**Objective:** Get CI green, tooling in place, shared fixtures available. Zero new test logic.

**Files to add/edit:**
- `pyproject.toml` — NEW (pytest, coverage, ruff, mypy config)
- `requirements.txt` — EDIT (add `pytest-cov>=5.0`, `ruff>=0.4.0`, `mypy>=1.10`)
- `tests/conftest.py` — NEW (5 shared fixture factories)
- `.github/workflows/ci.yml` — NEW (3-job CI + nightly schedule)

**Critical pseudo-diff — `pyproject.toml` (create from scratch):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: pure function tests",
    "integration: multi-module pipeline tests",
    "contract: schema and interface contract tests",
    "smoke: import-safety tests",
    "slow: performance/benchmark tests (nightly only)",
]
addopts = "--tb=short"

[tool.coverage.run]
source = ["stats", "data", "ui"]
omit = ["tests/*", "app.py"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP"]

[tool.mypy]
python_version = "3.13"
ignore_missing_imports = true
no_strict_optional = true
warn_unused_ignores = true
```

**Acceptance criteria:**
- All 3 CI jobs green on main branch
- All 289 existing tests still pass
- `ruff check .` exits 0 (fix any issues before merging)
- `mypy stats/ data/ ui/ --ignore-missing-imports` exits 0 (document suppressions)
- `conftest.py` exists; no existing test broken by its presence

---

### Phase 1: Coverage Baseline + Quality Gates (1 PR)

**Objective:** Measure actual coverage, fill the most impactful unit test gaps, add contract and smoke tests. Tighten coverage gate.

**Files to add/edit:**
- `tests/test_smoke.py` — NEW (~12 tests)
- `tests/test_contracts.py` — NEW (~15 tests)
- `tests/test_nl_query.py` — EDIT (add 6–8 edge cases)
- `tests/test_app_stat_grid.py` — EDIT (add `_chunk_stats` invariant tests)

**Key test additions:**

`tests/test_contracts.py`:
```python
pytestmark = pytest.mark.contract

def test_compute_stats_keys_match_stat_registry_plus_pa(statcast_df_factory):
    result = _compute_stats(statcast_df_factory(n=30))
    assert set(result.keys()) == {"PA"} | set(STAT_REGISTRY.keys())

def test_get_splits_columns_equal_split_cols(statcast_df_factory):
    # for split_type in ["hand", "home_away", "monthly"]
    result = get_splits(statcast_df_factory(n=30, p_throws=["R","L"]), "hand")
    assert list(result.columns) == SPLIT_COLS  # SPLIT_COLS must be exported

def test_nl_query_filter_types_are_valid_registry_keys():
    result = parse_nl_query("Gunnar Henderson inning 7+ at home vs LHP May 2-1 count", ...)
    for row in result["filter_rows"]:
        assert row["filter_type"] in FILTER_REGISTRY

def test_woba_weights_monotonically_increasing():
    weights = list(WOBA_WEIGHTS.values())
    assert weights == sorted(weights)

def test_woba_weights_covers_exactly_credited_events():
    assert set(WOBA_WEIGHTS.keys()) == {"walk","hit_by_pitch","single","double","triple","home_run"}
```

`tests/test_smoke.py`:
```python
def test_stats_modules_have_no_streamlit_import():
    """AST-based: stats/ modules must never import streamlit (architectural boundary)."""
    for py_file in Path("stats").glob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "streamlit"

def test_glossary_stat_definitions_has_required_keys():
    from ui.glossary import STAT_DEFINITIONS
    required = {"wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"}
    assert required.issubset(STAT_DEFINITIONS.keys())

def test_split_table_format_and_help_complete():
    from ui.components import _SPLIT_TABLE_FORMAT, _SPLIT_TABLE_HELP
    required = {"PA", "wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"}
    assert required.issubset(_SPLIT_TABLE_FORMAT.keys())

def test_filter_registry_self_consistent():
    from stats.filters import FILTER_REGISTRY
    for key, spec in FILTER_REGISTRY.items():
        assert spec.key == key
        assert len(spec.required_cols) > 0
```

**`test_contracts.py` depends on exporting constants from `stats/splits.py`:** `SPLIT_COLS`, `PITCHER_SPLIT_COLS`, `ARSENAL_COLS` (add these as module-level `list[str]` constants if they don't already exist — check the file).

**Acceptance criteria:**
- Baseline coverage measured and documented in PR description
- Coverage gate set at measured baseline (rounded down to nearest 5%)
- All new tests pass
- ruff and mypy clean on new files
- Total test count ≥ 315

---

### Phase 2: Integration Pipeline Tests (1 PR)

**Objective:** Add `tests/test_pipeline.py` covering the critical path end-to-end with synthetic data. This is the most important testing investment.

**Files to add:**
- `tests/test_pipeline.py` — NEW (~20 tests)

**Key test classes:**

```python
pytestmark = pytest.mark.integration

class TestBatterFullPipeline:
    def test_full_pipeline_end_to_end(self, statcast_df_factory, season_df_factory):
        raw = statcast_df_factory(n=100)
        cache = {}
        prepared = get_prepared_df_cached(raw, cache, (1, 2024, "Batter"))
        filtered = apply_filters(prepared, SplitFilters())
        stats = _compute_stats(filtered)
        dists = build_league_distributions(season_df_factory(n=200))
        percentiles = get_all_percentiles(stats, dists)
        for pct in percentiles.values():
            assert np.isnan(pct) or (0.0 <= pct <= 100.0)

    def test_k_pct_scale_consistent_batter(self, statcast_df_factory, season_df_factory):
        stats = _compute_stats(statcast_df_factory(n=100))
        dists = build_league_distributions(season_df_factory(n=200))
        assert stats["K%"] > 1.0             # 0-100 scale
        assert dists["K%"].max() > 1.0       # 0-100 scale
        pct = get_all_percentiles({"K%": stats["K%"]}, dists)
        assert 0.0 <= pct["K%"] <= 100.0

    def test_pitcher_perspective_inverts_home_away(self, statcast_df_factory):
        raw = statcast_df_factory(n=40, inning_topbot=["Bot"]*20 + ["Top"]*20)
        prepared = get_prepared_df_cached(raw, {}, (1, 2024, "Pitcher"))
        batter_home = apply_filters(prepared, SplitFilters(home_away="home"), pitcher_perspective=False)
        pitcher_home = apply_filters(prepared, SplitFilters(home_away="home"), pitcher_perspective=True)
        assert set(batter_home.index).isdisjoint(set(pitcher_home.index))

    def test_prepare_cache_hit_returns_same_object(self, statcast_df_factory):
        raw = statcast_df_factory(n=50)
        cache = {}
        key = (7, 2024, "Batter")
        assert get_prepared_df_cached(raw, cache, key) is get_prepared_df_cached(raw, cache, key)


class TestProportionScaleConsistency:
    def test_proportion_stats_0_100_after_compute(self, statcast_df_factory):
        stats = _compute_stats(statcast_df_factory(n=100))
        for stat in ["K%", "BB%", "HardHit%", "Barrel%"]:
            if stats[stat] is not None and stats[stat] != 0.0:
                assert stats[stat] > 1.0, f"{stat}={stats[stat]} looks like 0-1 scale"

    def test_pitcher_k_pct_is_higher_is_better(self):
        dists = {"K%": np.array([10.0, 20.0, 30.0])}
        high = get_all_percentiles({"K%": 30.0}, dists, player_type="Pitcher")
        low  = get_all_percentiles({"K%": 10.0}, dists, player_type="Pitcher")
        assert high["K%"] > low["K%"]

    def test_batter_k_pct_is_lower_is_better(self):
        dists = {"K%": np.array([10.0, 20.0, 30.0])}
        low  = get_all_percentiles({"K%": 10.0}, dists, player_type="Batter")
        high = get_all_percentiles({"K%": 30.0}, dists, player_type="Batter")
        assert low["K%"] > high["K%"]


class TestTrendPipeline:
    def test_trend_populates_cache_for_all_seasons(self, statcast_df_factory, make_trend_stub):
        df = statcast_df_factory(n=30)
        cache = {}
        get_trend_stats(1, [2022, 2023, 2024], "Batter", SplitFilters(), make_trend_stub(df), cache)
        for season in [2022, 2023, 2024]:
            assert (1, season, "Batter") in cache

    def test_trend_filter_propagates_to_all_seasons(self, statcast_df_factory, make_trend_stub):
        df = statcast_df_factory(n=60, p_throws=["R", "L"])
        result = get_trend_stats(1, [2022, 2023, 2024], "Batter",
                                  SplitFilters(pitcher_hand="R"), make_trend_stub(df), {})
        for row in result:
            assert row["PA"] == 30
```

**Acceptance criteria:**
- All 20 integration tests pass
- `TestProportionScaleConsistency` tests pass (most critical)
- Pitcher K% direction test passes
- Coverage gate passes (should now exceed 82%)
- No existing tests broken

---

### Phase 3: Smoke Tests + E2E Extensions (1 PR)

**Objective:** Extend import safety, add more `app.py` extractable helpers, enforce architectural boundaries. Optionally add 2–3 more Streamlit-mocked integration tests.

**Files to add/edit:**
- `tests/test_smoke.py` — EDIT (add app.py import test with mocked streamlit)
- `tests/test_app_stat_grid.py` — EDIT (extract more app.py pure helpers)

**Key additions:**

Verify `stats/` modules can be imported in an environment where `streamlit` is missing:
```python
def test_stats_modules_importable_without_streamlit(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "streamlit", None)  # block import
    import importlib
    for mod in ["stats.filters", "stats.splits", "stats.percentiles", "stats.nl_query"]:
        importlib.import_module(mod)  # must not raise
```

Extend `_load_stat_grid_functions()` in `test_app_stat_grid.py` to also extract:
- `_delta_text(stat, a_val, b_val) -> str` — pure string formatter
- `_delta_value(a_val, b_val) -> float | None` — pure numeric diff
- `_sample_size_text(sample_sizes, player_type) -> str` — pure label formatter

**Acceptance criteria:**
- Total test count ≥ 340
- `test_stats_modules_have_no_streamlit_import` passes (architectural boundary enforced)
- Coverage gate at 85% for `stats/` + `data/`
- All 3 CI jobs green; nightly job green

---

## Section 6 — Immediate Next Actions (Week 1)

**Day 1 — Probe the tooling baseline:**
```bash
cd "/Users/nate/Claude Baseball"
.venv/bin/pip install pytest-cov ruff mypy
.venv/bin/ruff check .              # count errors; fix before PR
.venv/bin/mypy stats/ data/ ui/ --ignore-missing-imports --no-strict-optional
```

**Day 2 — Create `pyproject.toml` and update `requirements.txt`:**
Create `pyproject.toml` with full content from Phase 0 pseudo-diff. Add `pytest-cov>=5.0`, `ruff>=0.4.0`, `mypy>=1.10` to `requirements.txt`.

**Day 3 — Measure baseline coverage:**
```bash
.venv/bin/python -m pytest tests/ \
    --cov=stats --cov=data --cov=ui \
    --cov-report=term-missing -q
```
Record output. Set `--cov-fail-under` to measured overall % rounded down to nearest 5.

**Day 4 — Create `tests/conftest.py`:**
Implement 5 fixture factories (copy implementations from existing test files). Verify all 289 tests still pass.

**Day 5 — Create `.github/workflows/ci.yml` + `tests/test_contracts.py` + `tests/test_smoke.py`:**
Create CI workflow. Write 15–20 new tests across contract + smoke files. Open Phase 0 + Phase 1 PRs.

**Week 1 Definition of Done:**
- [ ] `pyproject.toml` with pytest/coverage/ruff/mypy config
- [ ] `requirements.txt` includes `pytest-cov`, `ruff`, `mypy`
- [ ] `tests/conftest.py` with 5 fixture factories
- [ ] `.github/workflows/ci.yml` with 3 jobs (lint, typecheck, test) + nightly
- [ ] All 3 CI jobs green on main
- [ ] `tests/test_contracts.py` — all passing
- [ ] `tests/test_smoke.py` — all passing
- [ ] Baseline coverage measured and documented
- [ ] Total test count ≥ 315

---

## PR Checklist Template

Use this on every PR that touches test infrastructure or adds new tests:

```markdown
## Testing PR Checklist

### Scope
- [ ] PR addresses one phase or one logical chunk of the testing roadmap
- [ ] No unrelated refactors included

### Tests
- [ ] New tests follow naming convention: `test_<what>_<condition>_<expected>`
- [ ] New test files have a `pytestmark = pytest.mark.<layer>` at module level
- [ ] New test files import only from `conftest.py` fixtures (no copy-paste of `_make_*`)
- [ ] Edge cases covered: empty DataFrame, None input, NaN propagation
- [ ] No `time.sleep()`, no flaky network calls, no hardcoded paths outside project root

### Invariants
- [ ] If this PR changes filter logic: INV-1 (fast-path) and INV-2 (pitcher_perspective) still pass
- [ ] If this PR changes stat computation: INV-5 (PROPORTION_STATS scale) still passes
- [ ] If this PR changes percentile logic: INV-3 (monotonicity), INV-10 (pitcher K% direction) still pass
- [ ] If this PR changes nl_query: INV-8 (filter_type in FILTER_REGISTRY) still passes

### CI
- [ ] `ruff check .` exits 0
- [ ] `mypy stats/ data/ ui/ --ignore-missing-imports` exits 0
- [ ] `pytest tests/ -m "not slow" --cov=stats --cov=data --cov=ui --cov-fail-under=<gate>` passes
- [ ] Coverage did not decrease from previous baseline
- [ ] All 3 GitHub Actions CI jobs green

### Documentation
- [ ] New fixtures documented with docstring explaining factory signature
- [ ] Any new `# type: ignore` comments include the error code and reason
- [ ] PR description states measured coverage before and after
```

---

## Critical File Reference

| File | Role | Phase |
|---|---|---|
| `pyproject.toml` | Foundation for all tooling | Phase 0 — create |
| `requirements.txt` | Add `pytest-cov`, `ruff`, `mypy` | Phase 0 — edit |
| `tests/conftest.py` | Shared fixture factories | Phase 0 — create |
| `.github/workflows/ci.yml` | 3-job CI + nightly | Phase 0 — create |
| `tests/test_contracts.py` | Schema/key invariants | Phase 1 — create |
| `tests/test_smoke.py` | Import safety + arch boundary | Phase 1 — create |
| `tests/test_nl_query.py` | +6–8 edge cases | Phase 1 — edit |
| `tests/test_app_stat_grid.py` | +`_chunk_stats` invariants | Phase 1 — edit |
| `tests/test_pipeline.py` | Full pipeline integration | Phase 2 — create |
| `stats/splits.py` | Export `SPLIT_COLS`, `PITCHER_SPLIT_COLS`, `ARSENAL_COLS` | Phase 1 — minor edit |
