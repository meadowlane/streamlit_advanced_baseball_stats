"""Unit tests for stats/percentiles.py."""

import numpy as np
import pandas as pd
import pytest

from stats.percentiles import (
    CORE_STATS,
    LOWER_IS_BETTER,
    PROPORTION_STATS,
    build_league_distributions,
    compute_percentile,
    get_percentile,
    get_all_percentiles,
    get_color_tier,
    get_all_color_tiers,
    COLOR_TIERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_season_df(n: int = 100) -> pd.DataFrame:
    """Minimal FanGraphs-shaped DataFrame with all 6 core stats as proportions."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Name": [f"Player {i}" for i in range(n)],
            "wOBA":     rng.uniform(0.200, 0.450, n),
            "xwOBA":    rng.uniform(0.200, 0.450, n),
            "K%":       rng.uniform(0.050, 0.420, n),   # proportion
            "BB%":      rng.uniform(0.030, 0.180, n),   # proportion
            "HardHit%": rng.uniform(0.100, 0.650, n),   # proportion
            "Barrel%":  rng.uniform(0.000, 0.250, n),   # proportion
        }
    )


_UNIFORM_10 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


# ---------------------------------------------------------------------------
# build_league_distributions
# ---------------------------------------------------------------------------

class TestBuildLeagueDistributions:
    def test_returns_all_core_stats(self):
        dists = build_league_distributions(_make_season_df())
        for stat in CORE_STATS:
            assert stat in dists

    def test_proportion_stats_scaled_to_100(self):
        df = _make_season_df(50)
        dists = build_league_distributions(df)
        for stat in PROPORTION_STATS:
            # Original FG values are 0–1; distributions should be 0–100
            assert dists[stat].max() <= 100.0 + 1e-9
            assert dists[stat].max() > 1.0, f"{stat} not scaled up"

    def test_non_proportion_stats_unchanged(self):
        df = _make_season_df(50)
        dists = build_league_distributions(df)
        for stat in ["wOBA", "xwOBA"]:
            assert dists[stat].max() < 2.0  # still on 0–1 scale

    def test_nans_dropped(self):
        df = _make_season_df(10)
        df.loc[0, "wOBA"] = np.nan
        dists = build_league_distributions(df)
        assert not np.isnan(dists["wOBA"]).any()

    def test_missing_column_skipped(self):
        df = _make_season_df().drop(columns=["xwOBA"])
        dists = build_league_distributions(df)
        assert "xwOBA" not in dists

    def test_distribution_length_matches_non_nan_rows(self):
        df = _make_season_df(20)
        df.loc[[0, 1, 2], "wOBA"] = np.nan  # 3 NaN rows
        dists = build_league_distributions(df)
        assert len(dists["wOBA"]) == 17


# ---------------------------------------------------------------------------
# compute_percentile
# ---------------------------------------------------------------------------

class TestComputePercentile:
    def test_median_value_near_50(self):
        pct = compute_percentile(5.5, _UNIFORM_10, higher_is_better=True)
        assert pct == pytest.approx(50.0, abs=1.0)

    def test_max_value_is_100(self):
        pct = compute_percentile(10.0, _UNIFORM_10, higher_is_better=True)
        assert pct == pytest.approx(90.0, abs=0.1)  # 9 out of 10 below

    def test_min_value_is_0(self):
        pct = compute_percentile(1.0, _UNIFORM_10, higher_is_better=True)
        assert pct == pytest.approx(0.0, abs=0.1)

    def test_lower_is_better_inverts(self):
        # value=1 (best K%) should rank ~90th percentile
        pct = compute_percentile(1.0, _UNIFORM_10, higher_is_better=False)
        assert pct == pytest.approx(90.0, abs=0.1)  # 9 out of 10 have higher value

    def test_nan_value_returns_nan(self):
        assert np.isnan(compute_percentile(np.nan, _UNIFORM_10))

    def test_none_value_returns_nan(self):
        assert np.isnan(compute_percentile(None, _UNIFORM_10))  # type: ignore

    def test_empty_distribution_returns_nan(self):
        assert np.isnan(compute_percentile(5.0, np.array([])))

    def test_all_nan_distribution_returns_nan(self):
        assert np.isnan(compute_percentile(5.0, np.array([np.nan, np.nan])))

    def test_value_above_all_returns_near_100(self):
        pct = compute_percentile(999.0, _UNIFORM_10, higher_is_better=True)
        assert pct == pytest.approx(100.0, abs=0.1)

    def test_value_below_all_returns_0(self):
        pct = compute_percentile(-999.0, _UNIFORM_10, higher_is_better=True)
        assert pct == pytest.approx(0.0, abs=0.1)

    def test_return_value_in_range(self):
        rng = np.random.default_rng(0)
        dist = rng.uniform(0, 1, 500)
        for v in rng.uniform(0, 1, 20):
            pct = compute_percentile(float(v), dist)
            assert 0.0 <= pct <= 100.0


# ---------------------------------------------------------------------------
# get_percentile  (direction-aware wrapper)
# ---------------------------------------------------------------------------

class TestGetPercentile:
    def setup_method(self):
        self.dists = build_league_distributions(_make_season_df(200))

    def test_woba_higher_is_better(self):
        # A very high wOBA should rank near 100th
        pct = get_percentile("wOBA", 0.449, self.dists)
        assert pct > 80.0

    def test_k_pct_lower_is_better(self):
        assert "K%" in LOWER_IS_BETTER
        # K% value of 5.0 (5%) is excellent — should rank high
        pct_low_k = get_percentile("K%", 5.0, self.dists)
        # K% value of 42.0 (42%) is terrible — should rank low
        pct_high_k = get_percentile("K%", 42.0, self.dists)
        assert pct_low_k > pct_high_k

    def test_missing_stat_returns_nan(self):
        assert np.isnan(get_percentile("FIP", 3.5, self.dists))


# ---------------------------------------------------------------------------
# get_all_percentiles
# ---------------------------------------------------------------------------

class TestGetAllPercentiles:
    def setup_method(self):
        self.dists = build_league_distributions(_make_season_df(200))

    def test_returns_dict_with_all_input_stats(self):
        player = {"wOBA": 0.350, "K%": 20.0, "BB%": 10.0}
        result = get_all_percentiles(player, self.dists)
        assert set(result.keys()) == set(player.keys())

    def test_none_value_becomes_nan(self):
        result = get_all_percentiles({"wOBA": None}, self.dists)
        assert np.isnan(result["wOBA"])

    def test_all_six_stats(self):
        player = {
            "wOBA": 0.350, "xwOBA": 0.340,
            "K%": 22.0, "BB%": 10.0,
            "HardHit%": 45.0, "Barrel%": 8.0,
        }
        result = get_all_percentiles(player, self.dists)
        assert len(result) == 6
        for stat, pct in result.items():
            assert 0.0 <= pct <= 100.0, f"{stat}: {pct} out of range"


# ---------------------------------------------------------------------------
# get_color_tier
# ---------------------------------------------------------------------------

class TestGetColorTier:
    @pytest.mark.parametrize("pct,expected_name", [
        (95.0, "red"),
        (90.0, "red"),
        (85.0, "orange"),
        (70.0, "orange"),
        (60.0, "yellow"),
        (50.0, "yellow"),
        (40.0, "blue"),
        (30.0, "blue"),
        (20.0, "gray"),
        (0.0,  "gray"),
    ])
    def test_tier_boundaries(self, pct, expected_name):
        assert get_color_tier(pct)["name"] == expected_name

    def test_nan_returns_gray(self):
        assert get_color_tier(np.nan)["name"] == "gray"

    def test_none_returns_gray(self):
        assert get_color_tier(None)["name"] == "gray"  # type: ignore

    def test_returns_hex(self):
        tier = get_color_tier(95.0)
        assert "hex" in tier
        assert tier["hex"].startswith("#")

    def test_all_tiers_reachable(self):
        """Every tier in COLOR_TIERS should be returned by at least one value."""
        expected_names = {name for _, name, _ in COLOR_TIERS}
        found = {get_color_tier(pct)["name"] for pct in [95, 75, 55, 35, 10]}
        assert found == expected_names


# ---------------------------------------------------------------------------
# get_all_color_tiers
# ---------------------------------------------------------------------------

class TestGetAllColorTiers:
    def test_keys_match_input(self):
        percentiles = {"wOBA": 80.0, "K%": 20.0, "BB%": np.nan}
        result = get_all_color_tiers(percentiles)
        assert set(result.keys()) == set(percentiles.keys())

    def test_values_are_dicts_with_name_and_hex(self):
        result = get_all_color_tiers({"wOBA": 90.0})
        assert "name" in result["wOBA"]
        assert "hex" in result["wOBA"]
