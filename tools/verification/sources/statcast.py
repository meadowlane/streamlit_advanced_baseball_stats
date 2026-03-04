"""Statcast source adapter — independent re-fetch and re-compute.

This adapter re-fetches raw Statcast data via pybaseball and recomputes
Statcast-derived stats using the **same functions** as the app
(``stats.splits._compute_stats``, ``_compute_all_pitcher_stats``).

Design note
-----------
Because this adapter uses the same computation code as the app, it is a
**consistency check**, not fully independent verification.  It validates that:
1. The raw Statcast data round-trips correctly (no filtering bugs).
2. The computation results are stable between calls.
3. The FanGraphs wOBA column matches the in-app Statcast computation.

Any discrepancy between AppSource and StatcastSource for the *same* player
indicates a data-ordering or caching issue, not a formula bug.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from data.fetcher import _fetch_statcast_batter, _fetch_statcast_pitcher  # noqa: E402
from stats.splits import (  # noqa: E402
    _compute_stats,
    _compute_all_pitcher_stats,
)

from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError  # noqa: E402
from tools.verification.game_scope import filter_by_scope, SOURCE_SCOPE_SUPPORT  # noqa: E402

# Fastball pitch type codes used for FBv calculation
_FB_CODES = frozenset(["FF", "FA", "FT", "SI"])


class StatcastSource(BaseSource):
    """Re-fetches Statcast data and recomputes all Statcast-derived stats.

    Calling ``get_batter_season`` or ``get_pitcher_season`` on this adapter
    exercises the exact same computation path as the app.  Any differences
    from ``AppSource`` are bugs in the pipeline (caching, field ordering, etc.).
    """

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
            df_full = _fetch_statcast_batter(player.mlbam_id, year)
        except Exception as exc:
            raise SourceError(f"Statcast batter fetch failed for {player.mlbam_id}/{year}: {exc}") from exc

        if df_full.empty:
            raise SourceError(f"No Statcast data for batter {player.mlbam_id} in {year}")

        # Filter to the requested scope before computing
        df = filter_by_scope(df_full, game_type)
        if df.empty:
            raise SourceError(
                f"No Statcast data for batter {player.mlbam_id} in {year} "
                f"for scope={game_type!r}"
            )

        computed = _compute_stats(df, player_type="Batter")

        # FBv from scope-filtered data
        result: dict[str, Any] = dict(computed)
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
            df_full = _fetch_statcast_pitcher(player.mlbam_id, year)
        except Exception as exc:
            raise SourceError(f"Statcast pitcher fetch failed for {player.mlbam_id}/{year}: {exc}") from exc

        if df_full.empty:
            raise SourceError(f"No Statcast data for pitcher {player.mlbam_id} in {year}")

        df = filter_by_scope(df_full, game_type)
        if df.empty:
            raise SourceError(
                f"No Statcast data for pitcher {player.mlbam_id} in {year} "
                f"for scope={game_type!r}"
            )

        computed = _compute_all_pitcher_stats(df)
        result: dict[str, Any] = dict(computed)

        fbv = self._compute_fbv(df)
        if fbv is not None:
            result["FBv"] = fbv

        return result

    @staticmethod
    def _compute_fbv(df: Any) -> float | None:
        """Return mean fastball velocity from pitch-level Statcast data."""
        try:
            if "pitch_type" not in df.columns or "release_speed" not in df.columns:
                return None
            fb_df = df[df["pitch_type"].isin(_FB_CODES) & df["release_speed"].notna()]
            if fb_df.empty:
                return None
            return round(float(fb_df["release_speed"].mean()), 1)
        except Exception:
            return None
