"""Shared test fixtures for Phase 0 scaffolding.

These fixtures are added for future test modules and are not yet consumed by
existing test files.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest


def _expand(value: object, n: int) -> list[object]:
    if isinstance(value, list):
        if not value:
            return [None] * n
        return (value * ((n // len(value)) + 1))[:n]
    return [value] * n


def _month_to_date(month: object) -> str:
    month_int = int(month)
    month_int = min(max(month_int, 1), 12)
    return f"2024-{month_int:02d}-01"


@pytest.fixture
def statcast_df_factory() -> Callable[..., pd.DataFrame]:
    """Return a factory for synthetic Statcast-shaped DataFrames.

    Signature:
        _factory(
            n: int = 30,
            p_throws: str | list = "R",
            stand: str | list = "R",
            inning_topbot: str | list = "Bot",
            month: int | list = 4,
            include_pitch_cols: bool = False,
        ) -> pd.DataFrame
    """

    events_cycle = [
        "strikeout",
        "strikeout",
        "walk",
        "home_run",
        "single",
        "field_out",
        "double",
        "strikeout",
        "walk",
        "single",
        "field_out",
        "field_out",
        "triple",
        "strikeout",
        "single",
        "double",
        "walk",
        "field_out",
        "strikeout",
        "single",
        "field_out",
        "field_out",
        "strikeout",
        "single",
        "field_out",
        "field_out",
        "field_out",
        "field_out",
        "field_out",
        "field_out",
    ]
    batted_ball_events = {
        "single",
        "double",
        "triple",
        "home_run",
        "field_out",
        "grounded_into_double_play",
        "double_play",
        "triple_play",
        "force_out",
        "fielders_choice",
        "fielders_choice_out",
        "sac_fly",
        "sac_fly_double_play",
        "sac_bunt",
        "sac_bunt_double_play",
        "other_out",
    }

    def _factory(
        n: int = 30,
        p_throws: str | list = "R",
        stand: str | list = "R",
        inning_topbot: str | list = "Bot",
        month: int | list = 4,
        include_pitch_cols: bool = False,
    ) -> pd.DataFrame:
        events = (events_cycle * ((n // len(events_cycle)) + 1))[:n]

        months = _expand(month, n)
        game_dates = [_month_to_date(m) for m in months]

        launch_speed: list[float] = []
        launch_speed_angle: list[float] = []
        bb_type: list[object] = []
        xwoba: list[float] = []
        descriptions: list[str] = []
        balls: list[int] = []
        strikes: list[int] = []
        for i, event in enumerate(events):
            is_batted_ball = event in batted_ball_events
            if is_batted_ball:
                speed = 98.0 if (i % 5 == 0) else 88.0
                launch_speed.append(speed)
                launch_speed_angle.append(6.0 if speed >= 98.0 else 4.0)
                bb_type.append("ground_ball" if i % 2 == 0 else "fly_ball")
                xwoba.append(
                    0.500 if event in {"single", "double", "triple", "home_run"} else 0.050
                )
                descriptions.append("hit_into_play")
            else:
                launch_speed.append(float("nan"))
                launch_speed_angle.append(float("nan"))
                bb_type.append(np.nan)
                xwoba.append(float("nan"))
                if event == "walk":
                    descriptions.append("ball")
                else:
                    descriptions.append("swinging_strike")
            balls.append(i % 4)
            strikes.append(i % 3)

        df = pd.DataFrame(
            {
                "game_date": game_dates,
                "batter": [660271] * n,
                "pitcher": [123456] * n,
                "p_throws": _expand(p_throws, n),
                "stand": _expand(stand, n),
                "home_team": ["LAD"] * n,
                "away_team": ["NYY"] * n,
                "inning": [(i % 9) + 1 for i in range(n)],
                "inning_topbot": _expand(inning_topbot, n),
                "events": events,
                "description": descriptions,
                "balls": balls,
                "strikes": strikes,
                "launch_speed": launch_speed,
                "launch_angle": [15.0] * n,
                "launch_speed_angle": launch_speed_angle,
                "bb_type": bb_type,
                "estimated_woba_using_speedangle": xwoba,
            }
        )

        if include_pitch_cols:
            pitch_types = ["FF", "SL", "CH", "CU"]
            df["pitch_type"] = [pitch_types[i % len(pitch_types)] for i in range(n)]
            df["release_speed"] = [95.0 if pt == "FF" else 84.0 for pt in df["pitch_type"]]
            df["release_spin_rate"] = [2400.0 if pt == "FF" else 2600.0 for pt in df["pitch_type"]]
            df["plate_x"] = np.linspace(-0.8, 0.8, n)
            df["plate_z"] = np.linspace(1.8, 3.4, n)
            df["sz_top"] = [3.4] * n
            df["sz_bot"] = [1.5] * n
            df["pfx_x"] = np.linspace(-10.0, 10.0, n)
            df["pfx_z"] = np.linspace(-10.0, 10.0, n)

        return df

    return _factory


@pytest.fixture
def fg_batting_df_factory() -> Callable[..., pd.DataFrame]:
    """Return a factory for FanGraphs-shaped batting season DataFrames.

    Signature:
        _factory(
            names: tuple = ("Aaron Judge", "Shohei Ohtani"),
            season: int = 2024,
            proportion_scale: bool = True,
        ) -> pd.DataFrame
    """

    def _factory(
        names: tuple[str, ...] = ("Aaron Judge", "Shohei Ohtani"),
        season: int = 2024,
        proportion_scale: bool = True,
    ) -> pd.DataFrame:
        teams = ("NYY", "LAD", "ATL", "PHI", "SDP", "BAL")
        rows: list[dict[str, object]] = []
        for i, name in enumerate(names):
            k_rate = 0.220 - (i * 0.005)
            bb_rate = 0.120 + (i * 0.005)
            hard_hit = 0.520 + (i * 0.010)
            barrel = 0.150 + (i * 0.005)
            if not proportion_scale:
                k_rate *= 100.0
                bb_rate *= 100.0
                hard_hit *= 100.0
                barrel *= 100.0

            rows.append(
                {
                    "IDfg": 1000 + i,
                    "Name": name,
                    "Team": teams[i % len(teams)],
                    "Season": int(season),
                    "PA": 600 - (i * 20),
                    "wOBA": 0.350 + (i * 0.010),
                    "xwOBA": 0.345 + (i * 0.010),
                    "K%": k_rate,
                    "BB%": bb_rate,
                    "HardHit%": hard_hit,
                    "Barrel%": barrel,
                }
            )
        return pd.DataFrame(rows)

    return _factory


@pytest.fixture
def season_df_factory() -> Callable[..., pd.DataFrame]:
    """Return a factory for deterministic percentile-testing season DataFrames.

    Signature:
        _factory(n: int = 100, seed: int = 42) -> pd.DataFrame
    Proportion stats are produced on a 0-1 scale (FanGraphs convention).
    """

    def _factory(n: int = 100, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "Name": [f"Player {i}" for i in range(n)],
                "Season": [2024] * n,
                "wOBA": rng.uniform(0.200, 0.450, n),
                "xwOBA": rng.uniform(0.200, 0.450, n),
                "K%": rng.uniform(0.050, 0.420, n),
                "BB%": rng.uniform(0.030, 0.180, n),
                "HardHit%": rng.uniform(0.100, 0.650, n),
                "Barrel%": rng.uniform(0.000, 0.250, n),
            }
        )

    return _factory


@pytest.fixture
def empty_statcast_df() -> pd.DataFrame:
    """Zero-row Statcast DataFrame with the expected core schema."""

    columns = [
        "game_date",
        "batter",
        "pitcher",
        "p_throws",
        "stand",
        "home_team",
        "away_team",
        "inning",
        "inning_topbot",
        "events",
        "description",
        "balls",
        "strikes",
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "bb_type",
        "estimated_woba_using_speedangle",
    ]
    return pd.DataFrame(columns=columns)


@pytest.fixture
def make_trend_stub() -> Callable[[pd.DataFrame], Callable[[int, int], pd.DataFrame]]:
    """Return a factory that builds get_trend_stats-compatible fetch stubs.

    Usage:
        stub = make_trend_stub(statcast_df_factory(n=50))
        def stub(mlbam_id: int, season: int) -> pd.DataFrame
    """

    def _factory(df: pd.DataFrame) -> Callable[[int, int], pd.DataFrame]:
        def _stub(_mlbam_id: int, _season: int) -> pd.DataFrame:
            return df.copy()

        return _stub

    return _factory
