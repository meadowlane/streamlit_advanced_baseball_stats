"""Filter verification: game-type scope (regular vs postseason vs all).

This is the highest-value filter to verify because failing to apply scope
filtering is the #1 bug class — postseason PAs inflate regular-season totals,
causing PA, K%, BB%, and all rate stats to be wrong.

Verification strategy:
1. Structural invariants: regular + postseason = all (exact partition)
2. App vs Reference: production filter_by_scope must match reference filter_scope
3. External comparison: regular-scope PA should match FanGraphs PA (when fixtures exist)
4. Regression: specific known cases (e.g., Soto 2024 ~713 PA, not ~814)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on path for production code imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.reference_calc import (
    PA_EVENTS,
    REGULAR_GAME_TYPES,
    POSTSEASON_GAME_TYPES,
    compute_pa,
    compute_reference_stats,
    compute_stats,
    filter_scope,
)
from tests.filter_verification.conftest import (
    SEED_BATTERS,
    SEED_PLAYERS,
    load_raw_fixture,
    load_summary_fixture,
    make_synthetic_statcast_df,
)

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------

RATE_TOL_DECIMAL3 = 0.001  # wOBA, xwOBA
RATE_TOL_PCT1 = 0.1  # K%, BB%, HardHit%, Barrel%, GB%
EXTERNAL_PA_TOL = 2  # FG may differ by 1-2 due to late corrections


# ===================================================================
# 1. Structural Invariants (reference-only, no production imports)
# ===================================================================


class TestGameScopeInvariantsSynthetic:
    """Structural invariants using synthetic data (always available)."""

    def test_regular_plus_postseason_equals_all(self, synthetic_df: pd.DataFrame) -> None:
        """PA(regular) + PA(postseason) must equal PA(all)."""
        pa_all = compute_pa(synthetic_df)
        pa_reg = compute_pa(filter_scope(synthetic_df, "regular"))
        pa_post = compute_pa(filter_scope(synthetic_df, "postseason"))

        assert pa_reg + pa_post == pa_all, (
            f"Partition violated: regular ({pa_reg}) + postseason ({pa_post}) "
            f"= {pa_reg + pa_post} != all ({pa_all})"
        )

    def test_regular_less_than_all_when_postseason_exists(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """When postseason data exists, regular PA must be strictly less than all."""
        pa_all = compute_pa(synthetic_df)
        pa_reg = compute_pa(filter_scope(synthetic_df, "regular"))
        pa_post = compute_pa(filter_scope(synthetic_df, "postseason"))

        assert pa_post > 0, "Synthetic data should have postseason rows"
        assert pa_reg < pa_all, (
            f"regular PA ({pa_reg}) should be < all PA ({pa_all}) "
            f"when postseason PA ({pa_post}) > 0"
        )

    def test_scope_partitions_are_disjoint(self, synthetic_df: pd.DataFrame) -> None:
        """Regular and postseason rows must not overlap."""
        df_reg = filter_scope(synthetic_df, "regular")
        df_post = filter_scope(synthetic_df, "postseason")

        reg_idx = set(df_reg.index)
        post_idx = set(df_post.index)
        overlap = reg_idx & post_idx

        assert len(overlap) == 0, (
            f"{len(overlap)} rows appear in both regular and postseason"
        )

    def test_filtered_counts_never_increase(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Filtered PA must be <= unfiltered PA for any scope."""
        pa_all = compute_pa(synthetic_df)

        for scope in ("regular", "postseason"):
            pa_filtered = compute_pa(filter_scope(synthetic_df, scope))
            assert pa_filtered <= pa_all, (
                f"scope={scope}: filtered PA ({pa_filtered}) > all PA ({pa_all})"
            )

    def test_rates_recompute_correctly_from_filtered_totals(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Rate stats on filtered data must be self-consistent (not leaked from unfiltered)."""
        df_reg = filter_scope(synthetic_df, "regular")
        stats_reg = compute_stats(df_reg)

        # If PA > 0, K% and BB% must be between 0 and 100
        if stats_reg["PA"] > 0:
            assert 0.0 <= stats_reg["K%"] <= 100.0
            assert 0.0 <= stats_reg["BB%"] <= 100.0
            if stats_reg["HardHit%"] is not None:
                assert 0.0 <= stats_reg["HardHit%"] <= 100.0

    def test_known_synthetic_counts(
        self, synthetic_df: pd.DataFrame, synthetic_df_counts: dict
    ) -> None:
        """Verify synthetic data has the expected structure."""
        assert compute_pa(synthetic_df) == synthetic_df_counts["total_pa"]
        assert compute_pa(filter_scope(synthetic_df, "regular")) == synthetic_df_counts["regular_pa"]
        assert compute_pa(filter_scope(synthetic_df, "postseason")) == synthetic_df_counts["postseason_pa"]


# ===================================================================
# 2. App vs Reference (production code must match reference calc)
# ===================================================================


class TestGameScopeAppVsReferenceSynthetic:
    """Production filter_by_scope must produce identical results to reference filter_scope."""

    def test_regular_scope_pa_matches(self, synthetic_df: pd.DataFrame) -> None:
        """Production and reference must agree on regular-scope PA."""
        from tools.verification.game_scope import filter_by_scope

        app_df = filter_by_scope(synthetic_df, "regular")
        ref_df = filter_scope(synthetic_df, "regular")

        app_pa = compute_pa(app_df)
        ref_pa = compute_pa(ref_df)

        assert app_pa == ref_pa, (
            f"PA mismatch: app={app_pa}, ref={ref_pa}. "
            "Likely scope filter implementation differs."
        )

    def test_postseason_scope_pa_matches(self, synthetic_df: pd.DataFrame) -> None:
        """Production and reference must agree on postseason-scope PA."""
        from tools.verification.game_scope import filter_by_scope

        app_df = filter_by_scope(synthetic_df, "postseason")
        ref_df = filter_scope(synthetic_df, "postseason")

        assert compute_pa(app_df) == compute_pa(ref_df)

    def test_all_scope_pa_matches(self, synthetic_df: pd.DataFrame) -> None:
        """'all' scope should return all rows from both implementations."""
        from tools.verification.game_scope import filter_by_scope

        app_df = filter_by_scope(synthetic_df, "all")
        ref_df = filter_scope(synthetic_df, "all")

        assert compute_pa(app_df) == compute_pa(ref_df)

    def test_regular_scope_stats_match(self, synthetic_df: pd.DataFrame) -> None:
        """All stats must match between app and reference for regular scope."""
        from tools.verification.game_scope import filter_by_scope
        from stats.splits import _compute_stats

        app_df = filter_by_scope(synthetic_df, "regular")

        app_stats = _compute_stats(app_df, player_type="Batter")
        ref_stats = compute_reference_stats(synthetic_df, scope="regular")

        # PA must be exact
        assert app_stats["PA"] == ref_stats["PA"], (
            f"PA: app={app_stats['PA']}, ref={ref_stats['PA']}"
        )

        # wOBA, xwOBA: decimal_3 tolerance
        for stat in ("wOBA", "xwOBA"):
            app_val = app_stats.get(stat)
            ref_val = ref_stats.get(stat)
            if app_val is not None and ref_val is not None:
                assert abs(app_val - ref_val) <= RATE_TOL_DECIMAL3, (
                    f"{stat}: app={app_val}, ref={ref_val}, "
                    f"diff={abs(app_val - ref_val)}"
                )

        # Percentage stats: pct_1 tolerance
        for stat in ("K%", "BB%", "HardHit%", "Barrel%", "GB%"):
            app_val = app_stats.get(stat)
            ref_val = ref_stats.get(stat)
            if app_val is not None and ref_val is not None:
                assert abs(app_val - ref_val) <= RATE_TOL_PCT1, (
                    f"{stat}: app={app_val}, ref={ref_val}, "
                    f"diff={abs(app_val - ref_val)}"
                )


# ===================================================================
# 3. Real fixture tests (skipped when fixtures not recorded)
# ===================================================================


class TestGameScopeRealFixtures:
    """Tests using recorded Statcast fixtures — skipped if fixtures missing."""

    @pytest.mark.parametrize("player_type,mlbam_id,year,name", SEED_BATTERS)
    def test_partition_sum_real_data(
        self, player_type: str, mlbam_id: int, year: int, name: str
    ) -> None:
        """Regular + postseason PA must equal all PA in real data."""
        df_all = load_raw_fixture(player_type, mlbam_id, year, "all")

        pa_all = compute_pa(df_all)
        pa_reg = compute_pa(filter_scope(df_all, "regular"))
        pa_post = compute_pa(filter_scope(df_all, "postseason"))

        assert pa_reg + pa_post == pa_all, (
            f"{name}: regular ({pa_reg}) + postseason ({pa_post}) "
            f"= {pa_reg + pa_post} != all ({pa_all})"
        )

    @pytest.mark.parametrize("player_type,mlbam_id,year,name", SEED_BATTERS)
    def test_app_vs_reference_real_data(
        self, player_type: str, mlbam_id: int, year: int, name: str
    ) -> None:
        """App and reference agree on real fixture data."""
        from tools.verification.game_scope import filter_by_scope
        from stats.splits import _compute_stats

        df_all = load_raw_fixture(player_type, mlbam_id, year, "all")

        app_df = filter_by_scope(df_all, "regular")

        app_stats = _compute_stats(app_df, player_type="Batter")
        ref_stats = compute_reference_stats(df_all, scope="regular")

        assert app_stats["PA"] == ref_stats["PA"], (
            f"{name}: app PA={app_stats['PA']}, ref PA={ref_stats['PA']}. "
            "Cause: scope_mismatch — production filter_by_scope differs from reference."
        )

    @pytest.mark.parametrize("player_type,mlbam_id,year,name", SEED_BATTERS)
    def test_regular_pa_matches_fangraphs(
        self, player_type: str, mlbam_id: int, year: int, name: str
    ) -> None:
        """Regular-scope PA must match FanGraphs PA within tolerance."""
        df_all = load_raw_fixture(player_type, mlbam_id, year, "all")
        fg = load_summary_fixture("fangraphs", player_type, mlbam_id, year, "full")
        if fg is None:
            pytest.skip(f"No FG fixture for {name}")

        ref_pa = compute_pa(filter_scope(df_all, "regular"))
        fg_pa = fg.get("PA")

        if fg_pa is None:
            pytest.skip(f"FG fixture for {name} has no PA field")

        assert abs(ref_pa - fg_pa) <= EXTERNAL_PA_TOL, (
            f"{name}: ref regular PA={ref_pa} vs FG PA={fg_pa} "
            f"(diff={abs(ref_pa - fg_pa)}). "
            "If ref >> FG, likely scope filter not applied (postseason included). "
            "If within 1-2, likely late-season correction in FG data."
        )


# ===================================================================
# 4. Regression tests
# ===================================================================


class TestGameScopeRegression:
    """Specific regression cases for known scope bugs."""

    def test_unfiltered_all_includes_postseason(self, synthetic_df: pd.DataFrame) -> None:
        """Without scope filtering, PA should include postseason — this is the bug scenario."""
        # If someone computes stats without filtering, they get inflated PA
        pa_all = compute_pa(synthetic_df)
        pa_reg = compute_pa(filter_scope(synthetic_df, "regular"))

        # This test documents the expected difference
        assert pa_all > pa_reg, (
            "Without scope filtering, PA should be larger than regular-only PA. "
            "If equal, synthetic data may lack postseason rows."
        )

    def test_game_type_column_required(self) -> None:
        """When game_type column is missing, filter_scope returns all data (documented behavior)."""
        df = pd.DataFrame({
            "events": ["single", "strikeout", "walk"],
            "batter": [1, 1, 1],
        })
        # Reference should return all rows when game_type is absent
        filtered = filter_scope(df, "regular")
        assert len(filtered) == len(df)

    def test_scope_filter_preserves_all_columns(self, synthetic_df: pd.DataFrame) -> None:
        """Scope filtering must not drop or add columns."""
        original_cols = set(synthetic_df.columns)
        filtered = filter_scope(synthetic_df, "regular")
        assert set(filtered.columns) == original_cols
