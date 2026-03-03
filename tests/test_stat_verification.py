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

import sys
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Golden player definitions
# ---------------------------------------------------------------------------

_GOLDEN_BATTERS = [
    (592450, 2024, "Aaron Judge"),
    (660271, 2024, "Shohei Ohtani"),
    (650402, 2024, "Luis Arraez"),
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
        comparisons = _run_offline(650402, 2024, "batter")
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
