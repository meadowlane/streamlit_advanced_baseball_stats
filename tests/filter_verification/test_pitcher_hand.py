"""Filter verification: pitcher handedness (L/R) split.

This filter is well-supported by external sources (FanGraphs publishes vs-LHP
and vs-RHP splits) and exercises the core handedness filter logic.

Verification strategy:
1. Structural invariants: PA(vs L) + PA(vs R) = PA(all), disjoint sets
2. App vs Reference: production apply_filters(pitcher_hand=) must match
   reference filter_pitcher_hand()
3. External: FanGraphs vs-L/R split PA (when fixtures exist)
4. Consistency: split_by_hand() output must match individual filter application
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.reference_calc import (
    compute_pa,
    compute_stats,
    filter_pitcher_hand,
    filter_scope,
)
from tests.filter_verification.conftest import (
    SEED_BATTERS,
    load_raw_fixture,
    load_summary_fixture,
    make_synthetic_statcast_df,
)

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------

RATE_TOL_DECIMAL3 = 0.001
RATE_TOL_PCT1 = 0.1
EXTERNAL_PA_TOL = 2


# ===================================================================
# 1. Structural Invariants (synthetic data)
# ===================================================================


class TestPitcherHandInvariantsSynthetic:
    """Structural invariants using synthetic data."""

    def test_lr_partition_equals_whole(self, synthetic_df: pd.DataFrame) -> None:
        """PA(vs L) + PA(vs R) must equal PA(all)."""
        # Use regular scope only to match how the app works
        df = synthetic_df  # all pitches
        pa_all = compute_pa(df)
        pa_l = compute_pa(filter_pitcher_hand(df, "L"))
        pa_r = compute_pa(filter_pitcher_hand(df, "R"))

        assert pa_l + pa_r == pa_all, (
            f"Partition violated: vs L ({pa_l}) + vs R ({pa_r}) "
            f"= {pa_l + pa_r} != all ({pa_all})"
        )

    def test_filtered_pa_less_than_total(self, synthetic_df: pd.DataFrame) -> None:
        """Each hand split must have fewer PA than the total."""
        pa_all = compute_pa(synthetic_df)

        for hand in ("L", "R"):
            pa_hand = compute_pa(filter_pitcher_hand(synthetic_df, hand))
            assert pa_hand < pa_all, (
                f"vs {hand}HP: PA={pa_hand} not < total {pa_all}"
            )
            assert pa_hand > 0, f"vs {hand}HP: PA=0, expected some data"

    def test_lr_rows_disjoint(self, synthetic_df: pd.DataFrame) -> None:
        """L and R filtered DataFrames must have no overlapping rows."""
        df_l = filter_pitcher_hand(synthetic_df, "L")
        df_r = filter_pitcher_hand(synthetic_df, "R")

        overlap = set(df_l.index) & set(df_r.index)
        assert len(overlap) == 0, f"{len(overlap)} rows in both L and R"

    def test_filtered_counts_never_increase(self, synthetic_df: pd.DataFrame) -> None:
        """Filtered PA must be <= unfiltered PA."""
        pa_all = compute_pa(synthetic_df)

        for hand in ("L", "R"):
            pa_hand = compute_pa(filter_pitcher_hand(synthetic_df, hand))
            assert pa_hand <= pa_all

    def test_rates_are_valid(self, synthetic_df: pd.DataFrame) -> None:
        """Rate stats on filtered data must be in valid range."""
        for hand in ("L", "R"):
            df_hand = filter_pitcher_hand(synthetic_df, hand)
            stats = compute_stats(df_hand)

            if stats["PA"] > 0:
                assert 0.0 <= stats["K%"] <= 100.0
                assert 0.0 <= stats["BB%"] <= 100.0
                if stats["wOBA"] is not None:
                    assert 0.0 <= stats["wOBA"] <= 2.5

    def test_known_synthetic_counts(
        self, synthetic_df: pd.DataFrame, synthetic_df_counts: dict
    ) -> None:
        """Verify synthetic hand split has the expected structure."""
        pa_l = compute_pa(filter_pitcher_hand(synthetic_df, "L"))
        pa_r = compute_pa(filter_pitcher_hand(synthetic_df, "R"))

        assert pa_l == synthetic_df_counts["vs_lhp_pa"]
        assert pa_r == synthetic_df_counts["vs_rhp_pa"]


# ===================================================================
# 2. App vs Reference (production code must match reference calc)
# ===================================================================


class TestPitcherHandAppVsReferenceSynthetic:
    """Production apply_filters(pitcher_hand=) must match reference."""

    @pytest.mark.parametrize("hand", ["L", "R"])
    def test_pa_matches(self, synthetic_df: pd.DataFrame, hand: str) -> None:
        """Production and reference must agree on filtered PA."""
        from stats.filters import SplitFilters, apply_filters

        app_filtered = apply_filters(synthetic_df, SplitFilters(pitcher_hand=hand))
        ref_filtered = filter_pitcher_hand(synthetic_df, hand)

        app_pa = compute_pa(app_filtered)
        ref_pa = compute_pa(ref_filtered)

        assert app_pa == ref_pa, (
            f"vs {hand}HP PA mismatch: app={app_pa}, ref={ref_pa}. "
            "Cause: filter_predicate_mismatch — production uses wrong column or value."
        )

    @pytest.mark.parametrize("hand", ["L", "R"])
    def test_stats_match(self, synthetic_df: pd.DataFrame, hand: str) -> None:
        """All stats must match between app and reference."""
        from stats.filters import SplitFilters, apply_filters
        from stats.splits import _compute_stats

        app_filtered = apply_filters(synthetic_df, SplitFilters(pitcher_hand=hand))
        ref_filtered = filter_pitcher_hand(synthetic_df, hand)

        app_stats = _compute_stats(app_filtered, player_type="Batter")
        ref_stats = compute_stats(ref_filtered)

        # PA exact
        assert app_stats["PA"] == ref_stats["PA"]

        # wOBA, xwOBA
        for stat in ("wOBA", "xwOBA"):
            app_val = app_stats.get(stat)
            ref_val = ref_stats.get(stat)
            if app_val is not None and ref_val is not None:
                assert abs(app_val - ref_val) <= RATE_TOL_DECIMAL3, (
                    f"vs {hand}HP {stat}: app={app_val}, ref={ref_val}"
                )

        # Percentage stats
        for stat in ("K%", "BB%", "HardHit%", "Barrel%", "GB%"):
            app_val = app_stats.get(stat)
            ref_val = ref_stats.get(stat)
            if app_val is not None and ref_val is not None:
                assert abs(app_val - ref_val) <= RATE_TOL_PCT1, (
                    f"vs {hand}HP {stat}: app={app_val}, ref={ref_val}"
                )

    def test_split_by_hand_matches_filtered(self, synthetic_df: pd.DataFrame) -> None:
        """split_by_hand() must produce identical stats to manual filter+compute."""
        from stats.splits import split_by_hand, _compute_stats
        from stats.filters import SplitFilters, apply_filters

        split_result = split_by_hand(synthetic_df)

        for hand, label in [("R", "vs RHP"), ("L", "vs LHP")]:
            split_rows = split_result[split_result["Split"] == label]
            assert len(split_rows) == 1, f"Expected 1 row for {label}, got {len(split_rows)}"
            split_row = split_rows.iloc[0]

            filtered = apply_filters(synthetic_df, SplitFilters(pitcher_hand=hand))
            manual_stats = _compute_stats(filtered, player_type="Batter")

            assert split_row["PA"] == manual_stats["PA"], (
                f"{label}: split_by_hand PA={split_row['PA']} "
                f"!= filter PA={manual_stats['PA']}"
            )


# ===================================================================
# 3. Real fixture tests (skipped when fixtures not recorded)
# ===================================================================


class TestPitcherHandRealFixtures:
    """Tests using recorded Statcast fixtures."""

    @pytest.mark.parametrize("player_type,mlbam_id,year,name", SEED_BATTERS)
    def test_lr_partition_real(
        self, player_type: str, mlbam_id: int, year: int, name: str
    ) -> None:
        """L + R PA must equal total PA in real data (regular scope)."""
        df = load_raw_fixture(player_type, mlbam_id, year, "regular")

        pa_all = compute_pa(df)
        pa_l = compute_pa(filter_pitcher_hand(df, "L"))
        pa_r = compute_pa(filter_pitcher_hand(df, "R"))

        assert pa_l + pa_r == pa_all, (
            f"{name}: vs L ({pa_l}) + vs R ({pa_r}) = {pa_l + pa_r} != all ({pa_all})"
        )

    @pytest.mark.parametrize("player_type,mlbam_id,year,name", SEED_BATTERS)
    @pytest.mark.parametrize("hand", ["L", "R"])
    def test_app_vs_reference_real(
        self, player_type: str, mlbam_id: int, year: int, name: str, hand: str
    ) -> None:
        """App and reference agree on handedness filter for real data."""
        from stats.filters import SplitFilters, apply_filters
        from stats.splits import _compute_stats

        df = load_raw_fixture(player_type, mlbam_id, year, "regular")

        app_filtered = apply_filters(df, SplitFilters(pitcher_hand=hand))
        ref_filtered = filter_pitcher_hand(df, hand)

        app_stats = _compute_stats(app_filtered, player_type="Batter")
        ref_stats = compute_stats(ref_filtered)

        assert app_stats["PA"] == ref_stats["PA"], (
            f"{name} vs {hand}HP: app PA={app_stats['PA']}, ref PA={ref_stats['PA']}"
        )

    @pytest.mark.parametrize("player_type,mlbam_id,year,name", SEED_BATTERS)
    @pytest.mark.parametrize("hand,split_key", [("L", "vsL"), ("R", "vsR")])
    def test_pa_matches_fangraphs_split(
        self,
        player_type: str,
        mlbam_id: int,
        year: int,
        name: str,
        hand: str,
        split_key: str,
    ) -> None:
        """Filtered PA should match FanGraphs split PA."""
        df = load_raw_fixture(player_type, mlbam_id, year, "regular")
        fg = load_summary_fixture("fangraphs", player_type, mlbam_id, year, split_key)
        if fg is None:
            pytest.skip(f"No FG split fixture for {name} {split_key}")

        ref_pa = compute_pa(filter_pitcher_hand(df, hand))
        fg_pa = fg.get("PA")
        if fg_pa is None:
            pytest.skip(f"FG fixture for {name} {split_key} has no PA field")

        assert abs(ref_pa - fg_pa) <= EXTERNAL_PA_TOL, (
            f"{name} vs {hand}HP: ref PA={ref_pa} vs FG PA={fg_pa} "
            f"(diff={abs(ref_pa - fg_pa)})"
        )


# ===================================================================
# 4. Regression / Edge Cases
# ===================================================================


class TestPitcherHandRegression:
    """Edge cases and regression tests."""

    def test_filter_does_not_swap_lr(self, synthetic_df: pd.DataFrame) -> None:
        """Ensure L filter returns L pitchers, not R (swap bug)."""
        df_l = filter_pitcher_hand(synthetic_df, "L")
        if len(df_l) > 0:
            assert (df_l["p_throws"] == "L").all(), (
                "filter_pitcher_hand('L') returned rows with p_throws != 'L'"
            )

        df_r = filter_pitcher_hand(synthetic_df, "R")
        if len(df_r) > 0:
            assert (df_r["p_throws"] == "R").all(), (
                "filter_pitcher_hand('R') returned rows with p_throws != 'R'"
            )

    def test_filter_uses_p_throws_not_stand(self, synthetic_df: pd.DataFrame) -> None:
        """Pitcher hand filter must use p_throws, not stand (column swap bug)."""
        from stats.filters import SplitFilters, apply_filters

        # Apply pitcher_hand filter via production code
        app_l = apply_filters(synthetic_df, SplitFilters(pitcher_hand="L"))

        # Verify it's filtering on p_throws
        if len(app_l) > 0:
            assert (app_l["p_throws"] == "L").all(), (
                "Production pitcher_hand filter appears to use wrong column. "
                "All rows should have p_throws='L'."
            )

    def test_combined_scope_and_hand(self, synthetic_df: pd.DataFrame) -> None:
        """Scope + hand filter must compose correctly."""
        df_reg = filter_scope(synthetic_df, "regular")
        pa_reg = compute_pa(df_reg)

        pa_reg_l = compute_pa(filter_pitcher_hand(df_reg, "L"))
        pa_reg_r = compute_pa(filter_pitcher_hand(df_reg, "R"))

        assert pa_reg_l + pa_reg_r == pa_reg, (
            f"Within regular scope: vs L ({pa_reg_l}) + vs R ({pa_reg_r}) "
            f"= {pa_reg_l + pa_reg_r} != regular total ({pa_reg})"
        )

    def test_empty_hand_filter(self) -> None:
        """Filtering by a hand value that doesn't exist should return 0 PA."""
        df = pd.DataFrame({
            "events": ["single", "strikeout"],
            "p_throws": ["R", "R"],
            "launch_speed": [95.0, float("nan")],
            "launch_speed_angle": [6.0, float("nan")],
            "bb_type": ["fly_ball", None],
            "estimated_woba_using_speedangle": [0.5, float("nan")],
        })
        result = filter_pitcher_hand(df, "L")
        assert compute_pa(result) == 0
