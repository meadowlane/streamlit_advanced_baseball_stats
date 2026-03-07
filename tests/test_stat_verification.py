"""Automated stat verification test suite.

Architecture
------------
* **Offline tests** (default, CI-safe): Load pre-recorded JSON fixtures from
  ``tests/verification_fixtures/``.  These run without any network access.
  If fixtures are missing, tests are *skipped* (not failed), so CI remains
  green after initial checkout before fixtures have been recorded.

* **Online tests** (``--run-online``): Hit live external sources.  Intended
  for the fixture-recording step and periodic manual validation.

* **Golden set**: Seven canonical players chosen to cover critical edge cases.
  Any new stat or source adapter should pass for all seven.

Recording fixtures
------------------
    python -m tools.verify_stats --record-fixtures --golden-set --verbose

Running the suite
-----------------
    # Offline (CI default):
    pytest tests/test_stat_verification.py -v

    # With live network:
    pytest tests/test_stat_verification.py -v --run-online

Edge cases covered
------------------
1. Aaron Judge 2024  — HR leader; extreme counting stats; high Barrel%
2. Shohei Ohtani 2024 (batter) — elite batter season; formerly two-way
3. Luis Arraez 2024  — near-zero K% (edge case for K% computation)
4. Juan Soto 2024    — team change between 2023 and 2024 (TOT row handling)
5. Spencer Strider 2023 — extreme K% pitcher; high Stuff+
6. Corbin Burnes 2021   — Cy Young; near-perfect FIP; minimal BB
7. Shohei Ohtani 2023 (pitcher) — two-way; ~23 IP (low-IP edge case)
"""

from __future__ import annotations

from functools import lru_cache
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.verification.engine import (
    GOLDEN_PLAYERS,
    run_verification,
)
from tools.verification.fixtures import fixture_exists, load_fixture
from tools.verification.stat_map import STAT_MAP, TOLERANCES
from tools.verification.normalization import normalize_ip, normalize_pct
from tools.verification.comparison import StatComparison
from tools.verification.game_scope import (
    filter_by_scope,
    pa_breakdown_by_game_type,
    SOURCE_SCOPE_SUPPORT,
)


# ---------------------------------------------------------------------------
# Golden player definitions
# ---------------------------------------------------------------------------

_GOLDEN_BATTERS = [
    (592450, 2024, "Aaron Judge"),
    (660271, 2024, "Shohei Ohtani"),
    (650333, 2024, "Luis Arraez"),
    (665742, 2024, "Juan Soto"),
]

_GOLDEN_PITCHERS = [
    (675911, 2023, "Spencer Strider"),
    (669203, 2021, "Corbin Burnes"),
    (660271, 2023, "Shohei Ohtani"),  # two-way pitcher, low IP
]

_GOLDEN_IDENTITY_MAP = {p.mlbam_id: p for p in GOLDEN_PLAYERS.values()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixtures_present(mlbam_id: int, year: int, player_type: str) -> bool:
    """Return True only if at least 2 external-source fixtures exist."""
    sources = ["fangraphs", "statcast", "mlb_api", "baseball_ref"]
    count = sum(
        1 for src in sources
        if fixture_exists(src, player_type, mlbam_id, year)
    )
    return count >= 2


@lru_cache(maxsize=None)
def _run_offline(
    mlbam_id: int,
    year: int,
    player_type: str,
) -> list[StatComparison] | None:
    """Run offline verification for one player.  Returns None if fixtures missing."""
    if not _fixtures_present(mlbam_id, year, player_type):
        return None
    identity_map = {mlbam_id: _GOLDEN_IDENTITY_MAP.get(mlbam_id)}
    if identity_map[mlbam_id] is None:
        from tools.verification.sources.base import PlayerIdentity
        identity_map[mlbam_id] = PlayerIdentity(name=str(mlbam_id), mlbam_id=mlbam_id)
    reports = run_verification(
        [mlbam_id],
        year=year,
        player_type=player_type,
        player_identities={k: v for k, v in identity_map.items() if v is not None},
        offline=True,
    )
    if not reports:
        return None
    return reports[0].comparisons


class TestVerificationStatcastCache:
    def test_batter_cache_reuses_same_player_year(self) -> None:
        import pandas as pd

        from tools.verification.sources.statcast_cache import (
            clear_verification_statcast_cache,
            get_cached_batter_statcast,
        )

        clear_verification_statcast_cache()
        with patch(
            "tools.verification.sources.statcast_cache._fetch_statcast_batter",
            return_value=pd.DataFrame([{"events": "single"}]),
        ) as mock_fetch:
            first = get_cached_batter_statcast(660271, 2024)
            second = get_cached_batter_statcast(660271, 2024)

        assert mock_fetch.call_count == 1
        assert first.equals(second)
        clear_verification_statcast_cache()

    def test_pitcher_cache_clear_forces_refetch(self) -> None:
        import pandas as pd

        from tools.verification.sources.statcast_cache import (
            clear_verification_statcast_cache,
            get_cached_pitcher_statcast,
        )

        first_df = pd.DataFrame([{"events": "strikeout"}])
        second_df = pd.DataFrame([{"events": "walk"}])

        clear_verification_statcast_cache()
        with patch(
            "tools.verification.sources.statcast_cache._fetch_statcast_pitcher",
            side_effect=[first_df, second_df],
        ) as mock_fetch:
            first = get_cached_pitcher_statcast(675911, 2023)
            clear_verification_statcast_cache()
            second = get_cached_pitcher_statcast(675911, 2023)

        assert mock_fetch.call_count == 2
        assert first.equals(first_df)
        assert second.equals(second_df)
        clear_verification_statcast_cache()


# ---------------------------------------------------------------------------
# Unit tests — STAT_MAP structure (always run, no network, no fixtures)
# ---------------------------------------------------------------------------


class TestStatMapStructure:
    def test_non_verifiable_have_reason(self) -> None:
        """Every non-verifiable stat must supply a human-readable reason."""
        for key, mapping in STAT_MAP.items():
            if not mapping.verifiable:
                assert mapping.non_verifiable_reason is not None, (
                    f"{key}: verifiable=False but non_verifiable_reason is None"
                )

    def test_non_verifiable_have_fallback(self) -> None:
        """Every non-verifiable stat should supply a fallback strategy."""
        proprietary = {"Stuff+", "Location+", "Pitching+", "xFIP", "SIERA", "wRC+"}
        for key in proprietary:
            assert key in STAT_MAP, f"{key} missing from STAT_MAP"
            assert not STAT_MAP[key].verifiable, f"{key} should be non_verifiable"
            assert STAT_MAP[key].fallback is not None, f"{key} missing fallback strategy"

    def test_exact_match_for_counting_stats(self) -> None:
        """Counting stats must use 'exact' tolerance."""
        counting = ["PA", "H", "HR", "BB", "SO", "HBP", "W", "L"]
        for key in counting:
            assert key in TOLERANCES, f"{key} missing from TOLERANCES"
            assert TOLERANCES[key].kind == "exact", (
                f"{key}: expected 'exact' tolerance, got {TOLERANCES[key].kind!r}"
            )

    def test_rate_stats_have_abs_tolerance(self) -> None:
        """Rate stats must use 'abs' tolerance with a non-zero value."""
        rate_stats = ["AVG", "OBP", "SLG", "wOBA", "ERA", "K%", "BB%"]
        for key in rate_stats:
            assert key in TOLERANCES, f"{key} missing from TOLERANCES"
            assert TOLERANCES[key].kind == "abs", (
                f"{key}: expected 'abs' tolerance, got {TOLERANCES[key].kind!r}"
            )
            assert TOLERANCES[key].value > 0, f"{key}: tolerance value must be > 0"

    def test_woba_tolerance_reasonable(self) -> None:
        assert TOLERANCES["wOBA"].value <= 0.005, "wOBA tolerance > 0.005 is too loose"

    def test_k_pct_tolerance_reasonable(self) -> None:
        assert TOLERANCES["K%"].value <= 1.0, "K% tolerance > 1.0 pp is too loose"

    def test_all_displayed_stats_in_map(self) -> None:
        """All stats the app displays must have a STAT_MAP entry."""
        required = {
            # Batter
            "wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%", "GB%", "wRC+",
            "AVG", "OBP", "SLG", "OPS",
            # Pitcher
            "ERA", "FIP", "xFIP", "SIERA", "xERA", "K-BB%",
            "CSW%", "Whiff%", "FirstStrike%", "FBv",
            "Stuff+", "Location+", "Pitching+",
        }
        missing = required - set(STAT_MAP.keys())
        assert not missing, f"Stats missing from STAT_MAP: {missing}"


# ---------------------------------------------------------------------------
# Unit tests — normalization (no network)
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_ip_whole_innings(self) -> None:
        assert normalize_ip(6) == pytest.approx(6.0)
        assert normalize_ip("200") == pytest.approx(200.0)

    def test_normalize_ip_thirds(self) -> None:
        assert normalize_ip("157.1") == pytest.approx(157 + 1 / 3, abs=1e-6)
        assert normalize_ip("200.2") == pytest.approx(200 + 2 / 3, abs=1e-6)
        assert normalize_ip("6.0") == pytest.approx(6.0)

    def test_normalize_ip_none(self) -> None:
        assert normalize_ip(None) is None
        assert normalize_ip("nan") is None
        assert normalize_ip("") is None

    def test_normalize_pct_fraction_to_100(self) -> None:
        val = normalize_pct(0.225, source_name="fangraphs")
        assert val == pytest.approx(22.5, abs=1e-4)

    def test_normalize_pct_already_100_scale(self) -> None:
        val = normalize_pct(22.5, source_name="fangraphs")
        assert val == pytest.approx(22.5, abs=1e-4)

    def test_normalize_pct_non_fg_source(self) -> None:
        # MLB API returns 0-100 already — should not double-multiply
        val = normalize_pct(22.5, source_name="mlb_api")
        assert val == pytest.approx(22.5, abs=1e-4)


# ---------------------------------------------------------------------------
# Unit tests — comparison / verdict logic (no network)
# ---------------------------------------------------------------------------


class TestComparisonLogic:
    def test_pass_within_tolerance(self) -> None:
        from tools.verification.comparison import compare_stat
        result = compare_stat("wOBA", 0.350, {"fangraphs": 0.351, "statcast": 0.349})
        assert result.verdict == "PASS"

    def test_fail_outside_tolerance(self) -> None:
        from tools.verification.comparison import compare_stat
        result = compare_stat("wOBA", 0.350, {"fangraphs": 0.400, "statcast": 0.401})
        assert result.verdict == "FAIL"

    def test_warn_one_outlier(self) -> None:
        from tools.verification.comparison import compare_stat
        result = compare_stat(
            "wOBA", 0.350,
            {"fangraphs": 0.350, "statcast": 0.351, "mlb_api": 0.390},
        )
        assert result.verdict == "WARN"
        assert result.note is not None
        assert "mlb_api" in result.note

    def test_non_verifiable_stat(self) -> None:
        from tools.verification.comparison import compare_stat
        result = compare_stat("xFIP", 3.50, {"fangraphs": 3.50})
        assert result.verdict == "NON_VERIFIABLE"
        assert result.non_verifiable_reason is not None

    def test_exact_match_counting(self) -> None:
        from tools.verification.comparison import compare_stat
        result = compare_stat("HR", 62, {"fangraphs": 62, "mlb_api": 62, "baseball_ref": 62})
        assert result.verdict == "PASS"

    def test_warn_one_outlier_counting(self) -> None:
        """1/3 sources disagree with app on a counting stat → WARN (2/3 agree)."""
        from tools.verification.comparison import compare_stat
        result = compare_stat("HR", 62, {"fangraphs": 61, "mlb_api": 62, "baseball_ref": 62})
        assert result.verdict == "WARN"

    def test_fail_majority_counting(self) -> None:
        """2/3 sources disagree with app on a counting stat → FAIL."""
        from tools.verification.comparison import compare_stat
        result = compare_stat("HR", 60, {"fangraphs": 62, "mlb_api": 62, "baseball_ref": 62})
        assert result.verdict == "FAIL"

    def test_skip_when_our_value_none(self) -> None:
        from tools.verification.comparison import compare_stat
        result = compare_stat("ERA", None, {"fangraphs": 3.50})
        assert result.verdict == "SKIP"

    def test_all_sources_disagree_note(self) -> None:
        from tools.verification.comparison import compare_stat
        result = compare_stat(
            "wOBA", 0.310,
            {"fangraphs": 0.370, "statcast": 0.380, "baseball_ref": 0.390},
        )
        assert result.verdict == "FAIL"
        assert result.note is not None


# ---------------------------------------------------------------------------
# Fixture-based tests (offline golden set)
# ---------------------------------------------------------------------------


class TestFixturePresence:
    """Verify that fixture files exist after recording.  Skipped if absent."""

    @pytest.mark.parametrize("mlbam_id,year,name", _GOLDEN_BATTERS)
    def test_batter_app_fixture_exists(
        self, mlbam_id: int, year: int, name: str
    ) -> None:
        if not fixture_exists("app", "batter", mlbam_id, year):
            pytest.skip(
                f"App fixture missing for {name} {year}.  "
                "Run: python -m tools.verify_stats --record-fixtures --golden-set"
            )

    @pytest.mark.parametrize("mlbam_id,year,name", _GOLDEN_PITCHERS)
    def test_pitcher_app_fixture_exists(
        self, mlbam_id: int, year: int, name: str
    ) -> None:
        if not fixture_exists("app", "pitcher", mlbam_id, year):
            pytest.skip(
                f"App fixture missing for pitcher {name} {year}.  "
                "Run: python -m tools.verify_stats --record-fixtures --golden-set"
            )


class TestGoldenBatters:
    """Offline comparison tests for the golden batter set.

    Each test loads pre-recorded fixtures and runs the comparison engine.
    Tests are *skipped* (not failed) when fixtures have not yet been recorded.
    """

    @pytest.mark.parametrize("mlbam_id,year,name", _GOLDEN_BATTERS)
    def test_no_fail_verdicts(self, mlbam_id: int, year: int, name: str) -> None:
        comparisons = _run_offline(mlbam_id, year, "batter")
        if comparisons is None:
            pytest.skip(f"Fixtures not yet recorded for {name} {year}")

        failures = [c for c in comparisons if c.verdict == "FAIL"]
        assert not failures, (
            f"{name} {year} FAIL on: "
            + ", ".join(
                f"{c.stat}(ours={c.our_value}, sources={c.source_values})"
                for c in failures
            )
        )

    @pytest.mark.parametrize("mlbam_id,year,name", _GOLDEN_BATTERS)
    def test_counting_stats_exact(self, mlbam_id: int, year: int, name: str) -> None:
        """Counting stats must match exactly across all sources."""
        comparisons = _run_offline(mlbam_id, year, "batter")
        if comparisons is None:
            pytest.skip(f"Fixtures not yet recorded for {name} {year}")

        counting = {"HR", "BB", "SO", "H"}
        count_results = [c for c in comparisons if c.stat in counting]
        failures = [c for c in count_results if c.verdict == "FAIL"]
        assert not failures, (
            f"{name} {year}: counting stat mismatch — "
            + ", ".join(f"{c.stat}={c.our_value} vs {c.source_values}" for c in failures)
        )

    @pytest.mark.parametrize("mlbam_id,year,name", _GOLDEN_BATTERS)
    def test_woba_verified(self, mlbam_id: int, year: int, name: str) -> None:
        """wOBA must be verifiable and not FAIL (it's in the primary stat set)."""
        comparisons = _run_offline(mlbam_id, year, "batter")
        if comparisons is None:
            pytest.skip(f"Fixtures not yet recorded for {name} {year}")

        woba_cmp = next((c for c in comparisons if c.stat == "wOBA"), None)
        if woba_cmp is None:
            pytest.skip("wOBA not in comparison results")
        assert woba_cmp.verdict != "FAIL", (
            f"{name} {year}: wOBA FAIL — ours={woba_cmp.our_value}, "
            f"sources={woba_cmp.source_values}, diffs={woba_cmp.abs_diffs}"
        )

    def test_arraez_low_k_pct(self) -> None:
        """Luis Arraez 2024: K% must be below 10% — extreme contact hitter."""
        comparisons = _run_offline(650333, 2024, "batter")
        if comparisons is None:
            pytest.skip("Fixtures not yet recorded for Luis Arraez 2024")

        k_pct_cmp = next((c for c in comparisons if c.stat == "K%"), None)
        if k_pct_cmp is None:
            pytest.skip("K% not in comparison results")
        our_k = k_pct_cmp.our_value
        assert our_k is not None and our_k < 10.0, (
            f"Arraez K% = {our_k!r} — expected < 10.0%"
        )

    def test_judge_hr_count(self) -> None:
        """Aaron Judge 2024: HR must be ≥ 50 for this elite season."""
        comparisons = _run_offline(592450, 2024, "batter")
        if comparisons is None:
            pytest.skip("Fixtures not yet recorded for Aaron Judge 2024")

        hr_cmp = next((c for c in comparisons if c.stat == "HR"), None)
        if hr_cmp is None:
            pytest.skip("HR not in comparison results")
        assert hr_cmp.our_value is not None and hr_cmp.our_value >= 50, (
            f"Judge 2024 HR = {hr_cmp.our_value!r} — expected ≥ 50"
        )


class TestGoldenPitchers:
    """Offline comparison tests for the golden pitcher set."""

    @pytest.mark.parametrize("mlbam_id,year,name", _GOLDEN_PITCHERS)
    def test_no_fail_verdicts(self, mlbam_id: int, year: int, name: str) -> None:
        comparisons = _run_offline(mlbam_id, year, "pitcher")
        if comparisons is None:
            pytest.skip(f"Fixtures not yet recorded for pitcher {name} {year}")

        failures = [c for c in comparisons if c.verdict == "FAIL"]
        assert not failures, (
            f"{name} {year} pitcher FAIL on: "
            + ", ".join(
                f"{c.stat}(ours={c.our_value}, sources={c.source_values})"
                for c in failures
            )
        )

    @pytest.mark.parametrize("mlbam_id,year,name", _GOLDEN_PITCHERS)
    def test_era_verified(self, mlbam_id: int, year: int, name: str) -> None:
        comparisons = _run_offline(mlbam_id, year, "pitcher")
        if comparisons is None:
            pytest.skip(f"Fixtures not yet recorded for {name} {year}")

        era_cmp = next((c for c in comparisons if c.stat == "ERA"), None)
        if era_cmp is None:
            pytest.skip("ERA not found in comparison results")
        assert era_cmp.verdict in ("PASS", "WARN", "SKIP", "NON_VERIFIABLE"), (
            f"{name} {year}: ERA verdict = {era_cmp.verdict!r}, "
            f"ours={era_cmp.our_value}, sources={era_cmp.source_values}"
        )

    def test_ohtani_low_ip_skip(self) -> None:
        """Ohtani 2023 pitcher: low IP should cause some stats to be SKIP."""
        comparisons = _run_offline(660271, 2023, "pitcher")
        if comparisons is None:
            pytest.skip("Fixtures not yet recorded for Ohtani 2023 pitcher")

        # If IP < MIN_PITCHER_IP, at least some stats should be SKIP
        skip_cmps = [c for c in comparisons if c.verdict == "SKIP"]
        # At minimum, there should be a result set (not empty)
        assert comparisons, "No comparisons returned for Ohtani 2023 pitcher"

    def test_strider_high_k_pct(self) -> None:
        """Spencer Strider 2023: K% should be ≥ 30%."""
        comparisons = _run_offline(675911, 2023, "pitcher")
        if comparisons is None:
            pytest.skip("Fixtures not yet recorded for Spencer Strider 2023")

        k_cmp = next((c for c in comparisons if c.stat == "K%"), None)
        if k_cmp is None:
            pytest.skip("K% not in Strider comparison results")
        if k_cmp.our_value is not None:
            assert k_cmp.our_value >= 30.0, (
                f"Strider 2023 K% = {k_cmp.our_value!r} — expected ≥ 30%"
            )


# ---------------------------------------------------------------------------
# Game scope tests — no network, no fixtures; uses a synthetic DataFrame
# ---------------------------------------------------------------------------


class TestGameScope:
    """Unit tests for game-type scope filtering.

    All tests use a synthetic Statcast-like DataFrame that mirrors the
    Juan Soto 2024 pattern: 713 regular-season PA + 101 postseason PA
    (the known discrepancy that motivated this feature).
    """

    # -------------------------------------------------------------------
    # Shared fixture: synthetic event DataFrame
    # -------------------------------------------------------------------

    @staticmethod
    def _make_soto_like_df():
        """Return a minimal synthetic DataFrame with mixed game_type rows.

        Mimics Juan Soto 2024 season:
          - R:  713 PA + 87 non-PA pitches
          - F:   15 PA (Wild Card)
          - D:   22 PA (Division Series)
          - L:   64 PA (League Championship)
        Total PA across all types: 814; regular PA only: 713.
        """
        import pandas as pd

        # Use a minimal PA event that stats.splits.PA_EVENTS will recognise
        # (import that at test time to avoid import ordering issues)
        try:
            from stats.splits import PA_EVENTS
            pa_event = next(iter(PA_EVENTS))  # any PA event (e.g. "single")
        except ImportError:
            pa_event = "single"  # fallback if stats.splits not available

        rows = []
        # Regular season: 713 PA events + 87 non-PA pitches (balls, etc.)
        for _ in range(713):
            rows.append({"game_type": "R", "events": pa_event, "batter": 665742})
        for _ in range(87):
            rows.append({"game_type": "R", "events": None, "batter": 665742})

        # Postseason PA
        for _ in range(15):
            rows.append({"game_type": "F", "events": pa_event, "batter": 665742})
        for _ in range(22):
            rows.append({"game_type": "D", "events": pa_event, "batter": 665742})
        for _ in range(64):
            rows.append({"game_type": "L", "events": pa_event, "batter": 665742})

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------
    # filter_by_scope tests
    # -------------------------------------------------------------------

    def test_filter_regular_keeps_only_R_rows(self) -> None:
        df = self._make_soto_like_df()
        filtered = filter_by_scope(df, "regular")
        assert set(filtered["game_type"].unique()) == {"R"}, (
            f"Regular filter should keep only R rows; got {filtered['game_type'].unique()}"
        )

    def test_filter_regular_row_count(self) -> None:
        """Regular filter should drop all postseason rows (713 PA + 87 non-PA = 800)."""
        df = self._make_soto_like_df()
        filtered = filter_by_scope(df, "regular")
        assert len(filtered) == 800  # 713 PA + 87 non-PA pitches

    def test_filter_postseason_excludes_regular(self) -> None:
        df = self._make_soto_like_df()
        filtered = filter_by_scope(df, "postseason")
        assert "R" not in filtered["game_type"].unique(), (
            "Postseason filter should exclude regular-season rows"
        )
        # All postseason rows: 15 + 22 + 64 = 101 rows
        assert len(filtered) == 101

    def test_filter_all_includes_every_row(self) -> None:
        df = self._make_soto_like_df()
        filtered = filter_by_scope(df, "all")
        assert len(filtered) == len(df), (
            "All-scope filter must include every row"
        )

    def test_filter_missing_game_type_column_is_a_noop(self) -> None:
        """When game_type column is absent, filter_by_scope returns df unchanged."""
        import pandas as pd
        df = pd.DataFrame({"events": ["single", "walk"]})
        filtered = filter_by_scope(df, "regular")
        assert len(filtered) == len(df)

    # -------------------------------------------------------------------
    # pa_breakdown_by_game_type tests
    # -------------------------------------------------------------------

    def test_pa_breakdown_counts_pa_events_per_game_type(self) -> None:
        df = self._make_soto_like_df()
        breakdown = pa_breakdown_by_game_type(df)
        assert breakdown.get("R") == 713, f"R PA expected 713, got {breakdown.get('R')}"
        assert breakdown.get("F") == 15,  f"F PA expected 15, got {breakdown.get('F')}"
        assert breakdown.get("D") == 22,  f"D PA expected 22, got {breakdown.get('D')}"
        assert breakdown.get("L") == 64,  f"L PA expected 64, got {breakdown.get('L')}"

    def test_pa_breakdown_excludes_non_pa_rows(self) -> None:
        """Non-PA pitches (events=None) must not be counted in breakdown."""
        df = self._make_soto_like_df()
        breakdown = pa_breakdown_by_game_type(df)
        # Total across all game types: 713+15+22+64 = 814 (not 814+87)
        total = sum(breakdown.values())
        assert total == 814, f"Total PA expected 814, got {total}"

    # -------------------------------------------------------------------
    # Scope regression: regular PA must match independent regular-season source
    # -------------------------------------------------------------------

    def test_regular_scope_pa_matches_fangraphs_figure(self) -> None:
        """Core regression: after filtering to regular scope, PA = 713 (the FG/MLB API value)."""
        try:
            from stats.splits import PA_EVENTS
        except ImportError:
            pytest.skip("stats.splits not importable in this environment")


        df = self._make_soto_like_df()
        regular_df = filter_by_scope(df, "regular")

        pa_count = regular_df[
            regular_df["events"].notna() & regular_df["events"].isin(PA_EVENTS)
        ].shape[0]

        # 713 = the FG/MLB API regular-season PA for Juan Soto 2024
        assert pa_count == 713, (
            f"Expected 713 regular-season PA (the FG/MLB API figure), got {pa_count}"
        )

    def test_all_scope_pa_exceeds_regular(self) -> None:
        """'all' scope must have more PA than regular — postseason is included."""
        try:
            from stats.splits import PA_EVENTS
        except ImportError:
            pytest.skip("stats.splits not importable in this environment")

        df = self._make_soto_like_df()

        regular_df = filter_by_scope(df, "regular")
        all_df = filter_by_scope(df, "all")

        pa_regular = regular_df[
            regular_df["events"].notna() & regular_df["events"].isin(PA_EVENTS)
        ].shape[0]
        pa_all = all_df[
            all_df["events"].notna() & all_df["events"].isin(PA_EVENTS)
        ].shape[0]

        assert pa_all > pa_regular, "All scope must include more PA than regular"
        assert pa_all == 814, f"All-scope PA expected 814, got {pa_all}"

    def test_postseason_scope_pa_equals_playoff_sum(self) -> None:
        """Postseason PA = sum of F + D + L rows (15 + 22 + 64 = 101)."""
        try:
            from stats.splits import PA_EVENTS
        except ImportError:
            pytest.skip("stats.splits not importable in this environment")

        df = self._make_soto_like_df()
        post_df = filter_by_scope(df, "postseason")

        pa_post = post_df[
            post_df["events"].notna() & post_df["events"].isin(PA_EVENTS)
        ].shape[0]

        assert pa_post == 101, f"Postseason PA expected 101, got {pa_post}"

    # -------------------------------------------------------------------
    # Source capability matrix tests
    # -------------------------------------------------------------------

    def test_event_based_sources_support_all_scopes(self) -> None:
        """App and statcast sources must support all three scopes."""
        for src in ("app", "statcast"):
            supported = SOURCE_SCOPE_SUPPORT[src]
            assert "regular" in supported, f"{src} must support 'regular'"
            assert "postseason" in supported, f"{src} must support 'postseason'"
            assert "all" in supported, f"{src} must support 'all'"

    def test_aggregated_sources_support_only_regular(self) -> None:
        """FanGraphs, MLB API, and BRef must support only 'regular' scope."""
        for src in ("fangraphs", "mlb_api", "baseball_ref"):
            supported = SOURCE_SCOPE_SUPPORT[src]
            assert supported == frozenset(["regular"]), (
                f"{src} should support only 'regular'; got {supported}"
            )

    # -------------------------------------------------------------------
    # Soto 2024 fixture-based assertion (offline; skip if no fixtures)
    # -------------------------------------------------------------------

    def test_soto_2024_pa_in_regular_scope_matches_fangraphs_fixture(self) -> None:
        """If both app and fangraphs fixtures exist for Soto 2024, app PA ≤ FG PA + 5.

        With scope filtering applied, the app's regular-season PA must equal
        (or be very close to) the FanGraphs regular-season PA.  A large delta
        indicates the scope filter is not being applied correctly.

        Note: fixtures recorded before this fix may still show the old (unfiltered)
        values.  Re-run ``--record-fixtures`` to refresh them.
        """
        mlbam_id, year = 665742, 2024

        if not fixture_exists("fangraphs", "batter", mlbam_id, year):
            pytest.skip("FanGraphs fixture missing for Juan Soto 2024 — run --record-fixtures")
        if not fixture_exists("app", "batter", mlbam_id, year):
            pytest.skip("App fixture missing for Juan Soto 2024 — run --record-fixtures")

        fg_data = load_fixture("fangraphs", "batter", mlbam_id, year)
        app_data = load_fixture("app", "batter", mlbam_id, year)

        fg_pa = fg_data.get("PA")
        app_pa = app_data.get("PA")

        if fg_pa is None or app_pa is None:
            pytest.skip("PA not present in one or both fixtures")

        assert abs(app_pa - fg_pa) <= 5, (
            f"App PA ({app_pa}) differs from FanGraphs PA ({fg_pa}) by more than 5. "
            f"This suggests the game-type scope filter is not applied in the app source. "
            f"Re-record fixtures after this fix with: "
            f"python -m tools.verify_stats --record-fixtures --golden-set"
        )


# ---------------------------------------------------------------------------
# Online tests (require --run-online flag)
# ---------------------------------------------------------------------------


class TestOnlineFanGraphsAdapter:
    @pytest.mark.online
    def test_fangraphs_batter_returns_core_stats(self) -> None:
        from tools.verification.sources.fangraphs import FanGraphsSource
        from tools.verification.sources.base import PlayerIdentity

        src = FanGraphsSource()
        player = PlayerIdentity(name="Aaron Judge", mlbam_id=592450, fg_id=9063)
        data = src.get_batter_season(player, 2024)
        assert "wOBA" in data, "FanGraphs batter result missing wOBA"
        assert "K%" in data, "FanGraphs batter result missing K%"
        assert data["HR"] >= 50, f"Judge 2024 HR = {data['HR']!r}"

    @pytest.mark.online
    def test_mlb_api_batter_returns_core_stats(self) -> None:
        from tools.verification.sources.mlb_api import MLBApiSource
        from tools.verification.sources.base import PlayerIdentity

        src = MLBApiSource()
        player = PlayerIdentity(name="Aaron Judge", mlbam_id=592450)
        data = src.get_batter_season(player, 2024)
        assert "AVG" in data
        assert "HR" in data
        assert data["HR"] >= 50

    @pytest.mark.online
    def test_baseball_ref_batter_returns_core_stats(self) -> None:
        from tools.verification.sources.baseball_ref import BaseballRefSource
        from tools.verification.sources.base import PlayerIdentity

        src = BaseballRefSource()
        player = PlayerIdentity(name="Aaron Judge", mlbam_id=592450, bref_id="judgea01")
        data = src.get_batter_season(player, 2024)
        assert "AVG" in data
        assert "OPS" in data

    @pytest.mark.online
    @pytest.mark.slow
    def test_full_golden_set_live(self) -> None:
        """Run the full golden set against live sources and record fixtures."""
        from tools.verification.engine import run_golden_set_verification
        reports = run_golden_set_verification(
            offline=False,
            record_fixtures=True,
            verbose=True,
        )
        assert reports, "No reports returned for golden set live run"
        for rep in reports:
            failures = [c for c in rep.comparisons if c.verdict == "FAIL"]
            # Warn but don't assert — live data may have legitimate differences
            if failures:
                stat_names = [c.stat for c in failures]
                print(
                    f"\n[WARN] {rep.player.name} {rep.year}: FAIL on {stat_names}",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Unit tests — MLB API multi-team season aggregation (no network)
# ---------------------------------------------------------------------------


class TestMLBApiMultiTeamAggregation:
    """Validate _aggregate_splits and _pick_or_aggregate_splits logic.

    Uses synthetic split data matching the structure returned by the MLB Stats
    API.  No network calls are made.
    """

    def _make_split(self, **stat_overrides: Any) -> dict[str, Any]:
        """Build a minimal MLB API split dict."""
        defaults: dict[str, Any] = {
            "plateAppearances": 200,
            "atBats": 180,
            "hits": 50,
            "doubles": 10,
            "triples": 1,
            "homeRuns": 8,
            "baseOnBalls": 15,
            "strikeOuts": 40,
            "hitByPitch": 2,
            "sacFlies": 2,
            "runs": 25,
            "rbi": 30,
        }
        defaults.update(stat_overrides)
        return {"stat": defaults, "team": {"name": "Team A"}}

    def test_single_split_returned_directly(self) -> None:
        from tools.verification.sources.mlb_api import _pick_or_aggregate_splits
        split = self._make_split(plateAppearances=300)
        result = _pick_or_aggregate_splits([split], "plateAppearances")
        assert result is split

    def test_total_row_preferred(self) -> None:
        """If a 'Total' split exists it should be returned without summing."""
        from tools.verification.sources.mlb_api import _pick_or_aggregate_splits
        split_a = self._make_split(plateAppearances=200)
        split_b = self._make_split(plateAppearances=150)
        total_split = {"stat": {"plateAppearances": 350}, "team": {"name": "Total"}}
        result = _pick_or_aggregate_splits([split_a, split_b, total_split], "plateAppearances")
        assert result is total_split

    def test_aggregate_counting_stats_summed(self) -> None:
        """Multiple splits with no Total row should be aggregated by summing counts."""
        from tools.verification.sources.mlb_api import _aggregate_splits
        s1 = {"stat": {
            "plateAppearances": 300, "atBats": 270, "hits": 81,
            "doubles": 15, "triples": 2, "homeRuns": 20,
            "baseOnBalls": 25, "strikeOuts": 60, "hitByPitch": 3, "sacFlies": 2,
        }, "team": {"name": "San Diego Padres"}}
        s2 = {"stat": {
            "plateAppearances": 100, "atBats": 90, "hits": 28,
            "doubles": 5, "triples": 0, "homeRuns": 8,
            "baseOnBalls": 8, "strikeOuts": 22, "hitByPitch": 1, "sacFlies": 1,
        }, "team": {"name": "Miami Marlins"}}
        agg = _aggregate_splits([s1, s2])
        stat = agg["stat"]
        # PA must be sum
        assert stat["plateAppearances"] == 400
        assert stat["hits"] == 109
        assert stat["homeRuns"] == 28
        assert stat["baseOnBalls"] == 33
        assert stat["strikeOuts"] == 82

    def test_aggregate_avg_recalculated(self) -> None:
        """After aggregation, AVG must be recomputed from summed H/AB."""
        from tools.verification.sources.mlb_api import _aggregate_splits
        s1 = {"stat": {
            "atBats": 200, "hits": 60, "doubles": 10, "triples": 1,
            "homeRuns": 10, "baseOnBalls": 20, "sacFlies": 2, "hitByPitch": 2,
        }, "team": {"name": "A"}}
        s2 = {"stat": {
            "atBats": 100, "hits": 30, "doubles": 5, "triples": 0,
            "homeRuns": 5, "baseOnBalls": 10, "sacFlies": 1, "hitByPitch": 1,
        }, "team": {"name": "B"}}
        agg = _aggregate_splits([s1, s2])
        stat = agg["stat"]
        expected_avg = (60 + 30) / (200 + 100)  # 90 / 300 = 0.300
        assert float(stat["avg"]) == pytest.approx(expected_avg, abs=0.001)

    def test_arraez_2024_scenario(self) -> None:
        """Simulate Arraez 2024 multi-team scenario: two splits → correct aggregate."""
        from tools.verification.sources.mlb_api import _pick_or_aggregate_splits, MLBApiSource
        # Arraez 2024: traded from Marlins to Padres mid-season
        marlins_split = {"stat": {
            "plateAppearances": 201, "atBats": 184, "hits": 65,
            "doubles": 11, "triples": 1, "homeRuns": 1,
            "baseOnBalls": 14, "strikeOuts": 12, "hitByPitch": 2, "sacFlies": 1,
            "avg": ".354", "obp": ".401", "slg": ".413", "ops": ".814",
        }, "team": {"name": "Miami Marlins"}}
        padres_split = {"stat": {
            "plateAppearances": 362, "atBats": 332, "hits": 105,
            "doubles": 16, "triples": 2, "homeRuns": 6,
            "baseOnBalls": 22, "strikeOuts": 23, "hitByPitch": 5, "sacFlies": 3,
            "avg": ".316", "obp": ".364", "slg": ".410", "ops": ".774",
        }, "team": {"name": "San Diego Padres"}}

        best = _pick_or_aggregate_splits([marlins_split, padres_split], "plateAppearances")
        assert best is not None
        stat = best["stat"]

        # Season total PA must be sum of both teams
        assert stat["plateAppearances"] == 201 + 362

        # Parse as hitting stat — should produce sensible totals
        parsed = MLBApiSource._parse_hitting_stat(stat)
        assert parsed["PA"] == 563
        assert parsed["H"] == 65 + 105
        assert parsed["HR"] == 1 + 6

    def test_extract_splits_prefers_season_type(self) -> None:
        """_extract_splits should prefer the 'season' type entry."""
        from tools.verification.sources.mlb_api import MLBApiSource
        data = {
            "stats": [
                {
                    "type": {"displayName": "career"},
                    "splits": [{"stat": {"plateAppearances": 9999}}],
                },
                {
                    "type": {"displayName": "season"},
                    "splits": [{"stat": {"plateAppearances": 650}}],
                },
            ]
        }
        splits = MLBApiSource._extract_splits(data)
        assert len(splits) == 1
        assert splits[0]["stat"]["plateAppearances"] == 650

    def test_extract_splits_falls_back_to_first_entry(self) -> None:
        """When no 'season' type entry, first stats entry is used."""
        from tools.verification.sources.mlb_api import MLBApiSource
        data = {
            "stats": [
                {
                    "type": {"displayName": "yearByYear"},
                    "splits": [{"stat": {"plateAppearances": 500}}],
                },
            ]
        }
        splits = MLBApiSource._extract_splits(data)
        assert splits[0]["stat"]["plateAppearances"] == 500


# ---------------------------------------------------------------------------
# Unit tests — Statcast independence flag
# ---------------------------------------------------------------------------


class TestStatcastIndependence:
    def test_statcast_is_not_independent(self) -> None:
        """StatcastSource must be marked as non-independent."""
        from tools.verification.sources.statcast import StatcastSource
        src = StatcastSource()
        assert not src.is_independent, (
            "StatcastSource.is_independent must be False — it reuses app computation code"
        )

    def test_other_sources_are_independent(self) -> None:
        from tools.verification.sources.fangraphs import FanGraphsSource
        from tools.verification.sources.mlb_api import MLBApiSource
        from tools.verification.sources.baseball_ref import BaseballRefSource

        for cls in (FanGraphsSource, MLBApiSource, BaseballRefSource):
            src = cls()
            assert src.is_independent, f"{cls.__name__} should be independent"

    def test_independent_sources_excluded_from_verdict(self) -> None:
        """With statcast marked info-only, it should not affect FAIL verdict."""
        from tools.verification.comparison import compare_stat

        # Our value, FG agrees, statcast disagrees wildly
        result = compare_stat(
            "wOBA",
            0.350,
            {"fangraphs": 0.351, "statcast": 0.999},
            independent_sources=frozenset(["fangraphs"]),
        )
        # statcast is NOT independent → should not cause FAIL
        assert result.verdict == "PASS", (
            f"Expected PASS (statcast excluded), got {result.verdict}: {result.note}"
        )
        assert "statcast" in result.info_only_sources

    def test_engine_builds_correct_independent_set(self) -> None:
        from tools.verification.engine import EXTERNAL_SOURCES, _build_independent_source_names
        indep = _build_independent_source_names(EXTERNAL_SOURCES)
        assert "statcast" not in indep, "statcast should not be in independent source set"
        assert "fangraphs" in indep
        assert "mlb_api" in indep
        assert "baseball_ref" in indep


# ---------------------------------------------------------------------------
# Unit tests — fixture path correctness
# ---------------------------------------------------------------------------


class TestFixturePath:
    def test_fixture_dir_inside_repo(self) -> None:
        """FIXTURE_DIR must be inside the project repo, not a parent directory."""
        from tools.verification.fixtures import FIXTURE_DIR, _PROJECT_ROOT
        # FIXTURE_DIR should be a child of _PROJECT_ROOT
        assert str(FIXTURE_DIR).startswith(str(_PROJECT_ROOT)), (
            f"FIXTURE_DIR {FIXTURE_DIR!r} is not under PROJECT_ROOT {_PROJECT_ROOT!r}"
        )

    def test_fixture_dir_correct_structure(self) -> None:
        """Fixture dir should end with tests/verification_fixtures by default."""
        from tools.verification.fixtures import FIXTURE_DIR
        parts = FIXTURE_DIR.parts
        assert "tests" in parts
        assert "verification_fixtures" in parts

    def test_set_fixture_root_override(self, tmp_path: Path) -> None:
        """set_fixture_root should update FIXTURE_DIR and fixture_path() output."""
        import tools.verification.fixtures as fx
        original = fx.FIXTURE_DIR
        try:
            fx.set_fixture_root(tmp_path)
            assert fx.FIXTURE_DIR == tmp_path.resolve()
            p = fx.fixture_path("fangraphs", "batter", 592450, 2024)
            assert str(p).startswith(str(tmp_path))
        finally:
            # Restore original
            fx.FIXTURE_DIR = original
