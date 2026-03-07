"""Fixture loading and synthetic data helpers for filter verification tests.

This module provides:
1. ``load_raw_fixture`` — loads real Statcast parquet fixtures (for online/recorded tests)
2. ``load_summary_fixture`` — loads external source JSON fixtures
3. ``make_synthetic_statcast_df`` — builds a deterministic synthetic DataFrame
   with known properties for offline unit testing
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_ROOT = _PROJECT_ROOT / "tests" / "verification_fixtures"
_FILTER_FIXTURE_ROOT = _FIXTURE_ROOT / "filter_validation"
_LEGACY_FILTER_FIXTURE_ROOT = _PROJECT_ROOT / "tests" / "filter_fixtures"


def _raw_fixture_candidates(player_type: str, mlbam_id: int, year: int) -> list[Path]:
    file_name = f"{player_type}_{mlbam_id}_{year}_all.parquet"
    return [
        _FILTER_FIXTURE_ROOT / "raw" / file_name,
        _FIXTURE_ROOT / "raw" / file_name,
        _LEGACY_FILTER_FIXTURE_ROOT / "raw" / file_name,
    ]


def _summary_fixture_candidates(
    source: str,
    player_type: str,
    mlbam_id: int,
    year: int,
    split: str,
) -> list[Path]:
    base_name = f"{player_type}_{mlbam_id}_{year}"
    sources = [source]

    # Handedness split fixtures are recorded from Baseball Reference when
    # FanGraphs split fixtures are unavailable.
    if source == "fangraphs" and split in {"vsL", "vsR"}:
        sources.append("baseball_ref")

    paths: list[Path] = []
    for source_name in sources:
        paths.extend(
            [
                _FILTER_FIXTURE_ROOT / "summaries" / source_name / f"{base_name}_{split}.json",
                _FIXTURE_ROOT / source_name / f"{base_name}_{split}.json",
                _LEGACY_FILTER_FIXTURE_ROOT / "summaries" / source_name / f"{base_name}_{split}.json",
            ]
        )
        if split == "full":
            paths.extend(
                [
                    _FILTER_FIXTURE_ROOT / "summaries" / source_name / f"{base_name}.json",
                    _FIXTURE_ROOT / source_name / f"{base_name}.json",
                    _LEGACY_FILTER_FIXTURE_ROOT / "summaries" / source_name / f"{base_name}.json",
                ]
            )
    return paths

# ---------------------------------------------------------------------------
# Seed player registry
# ---------------------------------------------------------------------------

SEED_BATTERS = [
    ("batter", 592450, 2024, "Aaron Judge"),
    ("batter", 665742, 2024, "Juan Soto"),
    ("batter", 660271, 2024, "Shohei Ohtani"),
]

SEED_PITCHERS = [
    ("pitcher", 675911, 2023, "Spencer Strider"),
    ("pitcher", 669203, 2021, "Corbin Burnes"),
]

SEED_PLAYERS = SEED_BATTERS + SEED_PITCHERS


# ---------------------------------------------------------------------------
# Real fixture loaders (for recorded/online tests)
# ---------------------------------------------------------------------------


def load_raw_fixture(
    player_type: str, mlbam_id: int, year: int, scope: str = "all"
) -> pd.DataFrame:
    """Load a raw Statcast parquet fixture.

    Skips the test if the fixture file doesn't exist.
    Optionally applies scope filtering via reference_calc (NOT production code).
    """
    candidates = _raw_fixture_candidates(player_type, mlbam_id, year)
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        checked = ", ".join(str(p.relative_to(_PROJECT_ROOT)) for p in candidates)
        pytest.skip(
            f"Raw fixture missing for {player_type}_{mlbam_id}_{year}. "
            f"Checked: {checked}"
        )
    df = pd.read_parquet(path)
    if scope != "all":
        from tests.reference_calc import filter_scope

        df = filter_scope(df, scope)
    return df


def load_summary_fixture(
    source: str, player_type: str, mlbam_id: int, year: int, split: str
) -> dict | None:
    """Load an external source summary fixture.

    Returns None if the fixture doesn't exist (caller should SKIP).
    """
    for path in _summary_fixture_candidates(source, player_type, mlbam_id, year, split):
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        # Preferred explicit split fixture.
        if split != "full" and isinstance(data, dict):
            if "stats" in data and isinstance(data["stats"], dict):
                return data["stats"]
            if "splits" in data and isinstance(data["splits"], dict):
                split_stats = data["splits"].get(split)
                if isinstance(split_stats, dict):
                    return split_stats

        # Full-season fixture format.
        return data.get("stats", data)
    return None


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def make_synthetic_statcast_df(
    n_regular: int = 200,
    n_postseason: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a deterministic synthetic Statcast DataFrame with known properties.

    The DataFrame has:
    - ``n_regular`` rows with game_type="R" and ``n_postseason`` rows with game_type="D"
    - Mix of L/R pitcher hands and L/R batter hands (roughly 30% L, 70% R)
    - Home/Away split via inning_topbot (roughly 50/50)
    - Multiple months (April through September)
    - Various pitch counts (balls 0-3, strikes 0-2)
    - Diverse events from PA_EVENTS
    - Batted ball data (launch_speed, launch_speed_angle, bb_type) for batted ball events
    - pitch_type and description columns

    All values are deterministic given the seed, so tests produce reproducible results.
    """
    rng = np.random.default_rng(seed)
    n_total = n_regular + n_postseason

    # Game types: regular then postseason
    game_types = ["R"] * n_regular + ["D"] * n_postseason

    # Events cycle (deterministic mix of outcomes)
    events_pool = [
        "single", "single", "single", "single",          # 4 singles
        "double", "double",                                # 2 doubles
        "triple",                                          # 1 triple
        "home_run", "home_run",                            # 2 HRs
        "walk", "walk", "walk",                            # 3 walks
        "intent_walk",                                     # 1 IBB
        "hit_by_pitch",                                    # 1 HBP
        "strikeout", "strikeout", "strikeout", "strikeout", "strikeout",  # 5 Ks
        "field_out", "field_out", "field_out", "field_out", "field_out",  # 5 FOs
        "field_out", "field_out",                          # 2 more FOs
        "grounded_into_double_play",                       # 1 GIDP
        "force_out",                                       # 1 FC out
        "sac_fly",                                         # 1 SF
    ]
    # Repeat to fill n_total
    events = (events_pool * ((n_total // len(events_pool)) + 1))[:n_total]

    # Descriptions for each pitch (the last pitch of the PA)
    batted_ball_set = {
        "single", "double", "triple", "home_run", "field_out",
        "grounded_into_double_play", "force_out", "sac_fly",
    }
    descriptions = []
    for ev in events:
        if ev in batted_ball_set:
            descriptions.append("hit_into_play")
        elif ev in ("strikeout", "strikeout_double_play"):
            descriptions.append(rng.choice(["swinging_strike", "called_strike"]))
        elif ev in ("walk", "intent_walk"):
            descriptions.append("ball")
        elif ev == "hit_by_pitch":
            descriptions.append("hit_by_pitch")
        else:
            descriptions.append("hit_into_play")

    # Pitcher hand: ~30% L, ~70% R (deterministic pattern)
    p_throws = []
    for i in range(n_total):
        p_throws.append("L" if i % 10 < 3 else "R")

    # Batter hand: ~40% L, ~60% R
    stand = []
    for i in range(n_total):
        stand.append("L" if i % 5 < 2 else "R")

    # Home/away: alternating pattern (roughly 50/50)
    inning_topbot = []
    for i in range(n_total):
        inning_topbot.append("Bot" if i % 2 == 0 else "Top")

    # Months: spread across April-September
    months = [4, 5, 6, 7, 8, 9]
    game_dates = []
    for i in range(n_total):
        m = months[i % len(months)]
        day = (i % 28) + 1
        year = 2024
        game_dates.append(f"{year}-{m:02d}-{day:02d}")

    # Innings: 1-9
    innings = [(i % 9) + 1 for i in range(n_total)]

    # Counts
    balls_list = [i % 4 for i in range(n_total)]
    strikes_list = [i % 3 for i in range(n_total)]

    # Batted ball data
    launch_speed = []
    launch_speed_angle = []
    bb_type = []
    xwoba = []
    for i, ev in enumerate(events):
        if ev in batted_ball_set:
            speed = 98.0 if (i % 5 == 0) else 88.0
            launch_speed.append(speed)
            launch_speed_angle.append(6.0 if speed >= 98.0 else 4.0)
            bb_type.append("ground_ball" if i % 3 == 0 else "fly_ball")
            xwoba.append(0.500 if ev in ("single", "double", "triple", "home_run") else 0.050)
        else:
            launch_speed.append(float("nan"))
            launch_speed_angle.append(float("nan"))
            bb_type.append(np.nan)
            xwoba.append(float("nan"))

    # Pitch types
    pitch_types = ["FF", "SL", "CH", "CU", "SI"]
    pitch_type_list = [pitch_types[i % len(pitch_types)] for i in range(n_total)]

    df = pd.DataFrame(
        {
            "game_date": game_dates,
            "game_type": game_types,
            "batter": [660271] * n_total,
            "pitcher": [123456] * n_total,
            "p_throws": p_throws,
            "stand": stand,
            "home_team": ["LAD"] * n_total,
            "away_team": ["NYY"] * n_total,
            "inning": innings,
            "inning_topbot": inning_topbot,
            "events": events,
            "description": descriptions,
            "balls": balls_list,
            "strikes": strikes_list,
            "launch_speed": launch_speed,
            "launch_angle": [15.0] * n_total,
            "launch_speed_angle": launch_speed_angle,
            "bb_type": bb_type,
            "estimated_woba_using_speedangle": xwoba,
            "pitch_type": pitch_type_list,
            "release_speed": [94.0] * n_total,
        }
    )

    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A deterministic synthetic Statcast DataFrame for filter testing."""
    return make_synthetic_statcast_df()


@pytest.fixture
def synthetic_df_counts() -> dict[str, int]:
    """Pre-computed counts from the synthetic DataFrame for assertions.

    These are independently computed from the make_synthetic_statcast_df
    construction logic — NOT by running compute_pa on the DataFrame.
    """
    # The events_pool has 30 entries, all of which are in PA_EVENTS.
    # n_total = 250 rows, each row has a PA event.
    # So total PA = 250.
    # Regular PA = 200, postseason PA = 50.
    #
    # Pitcher hand: i%10 < 3 → L, otherwise R
    # For 250 rows: 75 L, 175 R
    #
    # Batter hand: i%5 < 2 → L, otherwise R
    # For 250 rows: 100 L, 150 R
    #
    # Home/away: i%2==0 → Bot (home), i%2==1 → Top (away)
    # For 250 rows: 125 Bot, 125 Top
    return {
        "total_pa": 250,
        "regular_pa": 200,
        "postseason_pa": 50,
        "vs_lhp_pa": 75,
        "vs_rhp_pa": 175,
        "vs_lhb_pa": 100,
        "vs_rhb_pa": 150,
        "home_pa": 125,
        "away_pa": 125,
    }
