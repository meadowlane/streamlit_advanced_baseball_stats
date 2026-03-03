"""MLB Stats API source adapter.

Uses the official, free MLB Stats API (statsapi.mlb.com) — no authentication
required.  Provides an independent source for traditional stats (AVG, OBP, SLG,
OPS, HR, BB, SO, PA, ERA, W, L, IP).

API endpoints used
------------------
Batter:  GET https://statsapi.mlb.com/api/v1/people/{mlbam_id}/stats
              ?stats=season&season={year}&group=hitting&sportId=1&gameType=R

Pitcher: GET https://statsapi.mlb.com/api/v1/people/{mlbam_id}/stats
              ?stats=season&season={year}&group=pitching&sportId=1&gameType=R

Multi-team players
------------------
The API returns one split entry per team.  We select the entry with the
*highest* plateAppearances / battersFaced value, which should be the combined
(total) row or the team where the player spent the majority of the season.
Some seasons may produce separate per-team entries — this is documented in the
report as a "multi-team" note.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import requests  # noqa: E402
except ImportError as exc:
    raise ImportError(
        "The 'requests' package is required for MLBApiSource.  "
        "Add it to requirements.txt: requests>=2.31.0"
    ) from exc

from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError  # noqa: E402
from tools.verification.normalization import normalize_avg, normalize_count, normalize_ip  # noqa: E402

_BASE_URL = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 15  # seconds


def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Make a GET request and return parsed JSON."""
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError as exc:
        raise SourceError(f"MLB API connection error: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise SourceError("MLB API request timed out") from exc
    except requests.exceptions.HTTPError as exc:
        raise SourceError(f"MLB API HTTP error {exc.response.status_code}") from exc
    except Exception as exc:
        raise SourceError(f"MLB API unexpected error: {exc}") from exc


def _pick_best_split(splits: list[dict[str, Any]], pa_key: str) -> dict[str, Any] | None:
    """Return the split entry with the highest plate appearances."""
    if not splits:
        return None
    # Prefer a split labelled "Total" or "2TM" / "3TM"
    for split in splits:
        team_name = split.get("team", {}).get("name", "").lower()
        if "total" in team_name or team_name in ("2tm", "3tm"):
            return split
    # Fall back to highest PA
    def _pa(s: dict[str, Any]) -> int:
        stat = s.get("stat", {})
        return (
            normalize_count(stat.get(pa_key)) or 0
        )
    return max(splits, key=_pa)


class MLBApiSource(BaseSource):
    """Fetches stats from the official MLB Stats API."""

    @property
    def source_name(self) -> str:
        return "mlb_api"

    def get_batter_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        if game_type != "regular":
            raise SourceError(
                f"MLBApiSource only supports 'regular' scope; "
                f"requested {game_type!r} — SKIP"
            )
        if offline:
            raise SourceError("MLBApiSource: offline mode requires fixture")

        url = f"{_BASE_URL}/people/{player.mlbam_id}/stats"
        data = _get(url, {
            "stats": "season",
            "season": year,
            "group": "hitting",
            "sportId": 1,
            "gameType": "R",
        })

        splits = self._extract_splits(data)
        best = _pick_best_split(splits, "plateAppearances")
        if best is None:
            raise SourceError(
                f"MLB API: no hitting splits for {player.name} in {year}"
            )

        stat = best.get("stat", {})
        return self._parse_hitting_stat(stat)

    def get_pitcher_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        if game_type != "regular":
            raise SourceError(
                f"MLBApiSource only supports 'regular' scope; "
                f"requested {game_type!r} — SKIP"
            )
        if offline:
            raise SourceError("MLBApiSource: offline mode requires fixture")

        url = f"{_BASE_URL}/people/{player.mlbam_id}/stats"
        data = _get(url, {
            "stats": "season",
            "season": year,
            "group": "pitching",
            "sportId": 1,
            "gameType": "R",
        })

        splits = self._extract_splits(data)
        best = _pick_best_split(splits, "battersFaced")
        if best is None:
            raise SourceError(
                f"MLB API: no pitching splits for {player.name} in {year}"
            )

        stat = best.get("stat", {})
        return self._parse_pitching_stat(stat)

    @staticmethod
    def _extract_splits(data: dict[str, Any]) -> list[dict[str, Any]]:
        stats_list = data.get("stats", [])
        if not stats_list:
            return []
        return stats_list[0].get("splits", [])

    @staticmethod
    def _parse_hitting_stat(stat: dict[str, Any]) -> dict[str, Any]:
        """Map MLB API hitting stat dict → our canonical keys."""
        result: dict[str, Any] = {}

        _add_count(result, stat, "PA", "plateAppearances")
        _add_count(result, stat, "H", "hits")
        _add_count(result, stat, "HR", "homeRuns")
        _add_count(result, stat, "BB", "baseOnBalls")
        _add_count(result, stat, "SO", "strikeOuts")
        _add_count(result, stat, "HBP", "hitByPitch")

        _add_avg(result, stat, "AVG", "avg")
        _add_avg(result, stat, "OBP", "obp")
        _add_avg(result, stat, "SLG", "slg")
        _add_avg(result, stat, "OPS", "ops")

        # Derive K% and BB% from counts if PA available
        pa = result.get("PA")
        so = result.get("SO")
        bb = result.get("BB")
        if pa and pa > 0:
            if so is not None:
                result["K%"] = round(so / pa * 100.0, 1)
            if bb is not None:
                result["BB%"] = round(bb / pa * 100.0, 1)

        return result

    @staticmethod
    def _parse_pitching_stat(stat: dict[str, Any]) -> dict[str, Any]:
        """Map MLB API pitching stat dict → our canonical keys."""
        result: dict[str, Any] = {}

        _add_count(result, stat, "W", "wins")
        _add_count(result, stat, "L", "losses")
        _add_count(result, stat, "SO", "strikeOuts")
        _add_count(result, stat, "BB", "baseOnBalls")
        _add_count(result, stat, "HR", "homeRuns")
        _add_count(result, stat, "HBP", "hitBatsmen")

        _add_avg(result, stat, "ERA", "era")

        ip_raw = stat.get("inningsPitched")
        if ip_raw is not None:
            ip = normalize_ip(ip_raw)
            if ip is not None:
                result["IP"] = ip

        # Derive K%, BB%, K-BB% from counts if battersFaced is available
        bf = normalize_count(stat.get("battersFaced"))
        so = result.get("SO")
        bb = result.get("BB")
        if bf and bf > 0:
            if so is not None:
                result["K%"] = round(so / bf * 100.0, 1)
            if bb is not None:
                result["BB%"] = round(bb / bf * 100.0, 1)
        if result.get("K%") is not None and result.get("BB%") is not None:
            result["K-BB%"] = round(result["K%"] - result["BB%"], 1)

        return result


def _add_count(
    result: dict[str, Any],
    stat: dict[str, Any],
    our_key: str,
    api_key: str,
) -> None:
    raw = stat.get(api_key)
    val = normalize_count(raw)
    if val is not None:
        result[our_key] = val


def _add_avg(
    result: dict[str, Any],
    stat: dict[str, Any],
    our_key: str,
    api_key: str,
) -> None:
    raw = stat.get(api_key)
    val = normalize_avg(raw)
    if val is not None:
        result[our_key] = val
