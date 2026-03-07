"""Statcast source adapter — shared raw fetch, independent re-compute.

This adapter reuses the verification run's cached raw Statcast data and
recomputes Statcast-derived stats using the same functions as the app.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stats.splits import _compute_all_pitcher_stats, _compute_stats  # noqa: E402
from tools.verification.game_scope import SOURCE_SCOPE_SUPPORT, filter_by_scope  # noqa: E402
from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError  # noqa: E402
from tools.verification.sources.statcast_cache import (  # noqa: E402
    get_cached_batter_statcast,
    get_cached_pitcher_statcast,
)

_FB_CODES = frozenset(["FF", "FA", "FT", "SI"])


class StatcastSource(BaseSource):
    """Recomputes Statcast-derived stats from cached raw Statcast data."""

    is_independent: bool = False

    @property
    def source_name(self) -> str:
        return "statcast"

    @property
    def supported_scopes(self) -> frozenset[str]:
        return SOURCE_SCOPE_SUPPORT["statcast"]

    def get_batter_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        if offline:
            raise SourceError("StatcastSource: offline mode requires fixture")

        try:
            df_full = get_cached_batter_statcast(player.mlbam_id, year)
        except Exception as exc:
            raise SourceError(
                f"Statcast batter fetch failed for {player.mlbam_id}/{year}: {exc}"
            ) from exc

        if df_full.empty:
            raise SourceError(f"No Statcast data for batter {player.mlbam_id} in {year}")

        df = filter_by_scope(df_full, game_type)
        if df.empty:
            raise SourceError(
                f"No Statcast data for batter {player.mlbam_id} in {year} "
                f"for scope={game_type!r}"
            )

        result: dict[str, Any] = dict(_compute_stats(df, player_type="Batter"))
        fbv = self._compute_fbv(df)
        if fbv is not None:
            result["FBv"] = fbv
        return result

    def get_pitcher_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        if offline:
            raise SourceError("StatcastSource: offline mode requires fixture")

        try:
            df_full = get_cached_pitcher_statcast(player.mlbam_id, year)
        except Exception as exc:
            raise SourceError(
                f"Statcast pitcher fetch failed for {player.mlbam_id}/{year}: {exc}"
            ) from exc

        if df_full.empty:
            raise SourceError(f"No Statcast data for pitcher {player.mlbam_id} in {year}")

        df = filter_by_scope(df_full, game_type)
        if df.empty:
            raise SourceError(
                f"No Statcast data for pitcher {player.mlbam_id} in {year} "
                f"for scope={game_type!r}"
            )

        result: dict[str, Any] = dict(_compute_all_pitcher_stats(df))
        fbv = self._compute_fbv(df)
        if fbv is not None:
            result["FBv"] = fbv
        return result

    @staticmethod
    def _compute_fbv(df: Any) -> float | None:
        try:
            if "pitch_type" not in df.columns or "release_speed" not in df.columns:
                return None
            fb_df = df[df["pitch_type"].isin(_FB_CODES) & df["release_speed"].notna()]
            if fb_df.empty:
                return None
            return round(float(fb_df["release_speed"].mean()), 1)
        except Exception:
            return None
