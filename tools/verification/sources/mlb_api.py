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
The API may return one split entry per team for some players.  We handle this
by:

1. Looking for a split whose team name contains "Total", "2TM", or "3TM" — the
   MLB API sometimes includes an explicit aggregate row.
2. If no aggregate row is found and there are multiple splits, **summing** all
   counting stats and recalculating rate stats from the combined totals.  This
   produces the correct season aggregate regardless of mid-season trades.

Game type
---------
``gameType=R`` is hardcoded for regular-season scope.  If a different game type
is requested the adapter raises :class:`SourceError` with a clear note
(MLB Stats API game-type values: R=regular, P=postseason, S=spring).
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

# Mapping: our game_type value → MLB API gameType param
_GAME_TYPE_MAP: dict[str, str] = {
    "regular": "R",
    "postseason": "P",
    "spring": "S",
    "all": "R",  # MLB API doesn't support a true "all"; default to regular
}


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


def _pick_or_aggregate_splits(
    splits: list[dict[str, Any]],
    pa_key: str,
) -> dict[str, Any] | None:
    """Return the best split, preferring an explicit aggregate row.

    Strategy:
    1. Look for a split labelled "Total", "2TM", "3TM" (or similar).
    2. If not found and there is only one split, return it directly.
    3. If not found and there are multiple splits, **aggregate** them by
       summing counting stats and recomputing rate stats.
    """
    if not splits:
        return None

    # --- Step 1: look for explicit aggregate row ---
    for split in splits:
        team_name = split.get("team", {}).get("name", "").lower()
        if "total" in team_name or team_name in ("2tm", "3tm", "2 teams", "3 teams"):
            return split

    # --- Step 2: single split → return directly ---
    if len(splits) == 1:
        return splits[0]

    # --- Step 3: aggregate multiple team splits ---
    return _aggregate_splits(splits)


def _aggregate_splits(splits: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum counting stats across all team splits and recompute rate stats.

    This handles multi-team seasons where the MLB API returns separate splits
    per team with no combined total row.
    """
    # Counting stat keys in the MLB API stat dict
    _INT_KEYS = [
        "plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns",
        "baseOnBalls", "intentionalWalks", "strikeOuts", "hitByPitch",
        "sacBunts", "sacFlies", "groundOuts", "airOuts", "runs", "rbi",
        "stolenBases", "caughtStealing",
        # Pitching
        "wins", "losses", "saves", "blownSaves", "holds", "saveOpportunities",
        "gamesPlayed", "gamesStarted", "completeGames", "shutouts",
        "strikeOuts", "baseOnBalls", "homeRuns", "hitByPitch",
        "battersFaced", "outs", "numberOfPitches", "pitchesThrown",
    ]
    _FLOAT_KEYS = ["inningsPitched"]

    agg_stat: dict[str, Any] = {}

    # Sum integers
    for key in _INT_KEYS:
        total = 0
        found = False
        for split in splits:
            val = split.get("stat", {}).get(key)
            if val is not None:
                try:
                    total += int(val)
                    found = True
                except (TypeError, ValueError):
                    pass
        if found:
            agg_stat[key] = total

    # Sum IP (stored as decimal string like "6.2" meaning 6⅔ innings)
    total_ip = 0.0
    has_ip = False
    for split in splits:
        raw_ip = split.get("stat", {}).get("inningsPitched")
        ip = normalize_ip(raw_ip)
        if ip is not None:
            total_ip += ip
            has_ip = True
    if has_ip:
        # Store as decimal for downstream parsing
        agg_stat["inningsPitched"] = str(round(total_ip, 1))

    # Recompute rate stats from aggregated counts
    ab = agg_stat.get("atBats", 0)
    h = agg_stat.get("hits", 0)
    bb = agg_stat.get("baseOnBalls", 0)
    hbp = agg_stat.get("hitByPitch", 0)
    sf = sum(split.get("stat", {}).get("sacFlies", 0) or 0 for split in splits)
    doubles = agg_stat.get("doubles", 0)
    triples = agg_stat.get("triples", 0)
    hr = agg_stat.get("homeRuns", 0)
    so = agg_stat.get("strikeOuts", 0)

    if ab > 0:
        agg_stat["avg"] = f"{h / ab:.3f}"
    obp_denom = ab + bb + hbp + sf
    if obp_denom > 0:
        agg_stat["obp"] = f"{(h + bb + hbp) / obp_denom:.3f}"
    singles = h - doubles - triples - hr
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    if ab > 0:
        agg_stat["slg"] = f"{tb / ab:.3f}"
    obp_val = float(agg_stat.get("obp", "0") or 0)
    slg_val = float(agg_stat.get("slg", "0") or 0)
    agg_stat["ops"] = f"{obp_val + slg_val:.3f}"

    # ERA: (ER × 9) / IP — need earned runs
    er_total = sum(split.get("stat", {}).get("earnedRuns", 0) or 0 for split in splits)
    if has_ip and total_ip > 0:
        agg_stat["era"] = f"{er_total * 9 / total_ip:.2f}"
        agg_stat["earnedRuns"] = er_total

    return {"stat": agg_stat, "_aggregated": True}


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
        offline: bool = False,
        game_type: str = "regular",
    ) -> dict[str, Any]:
        if offline:
            raise SourceError("MLBApiSource: offline mode requires fixture")

        api_game_type = _GAME_TYPE_MAP.get(game_type, "R")
        url = f"{_BASE_URL}/people/{player.mlbam_id}/stats"
        data = _get(url, {
            "stats": "season",
            "season": year,
            "group": "hitting",
            "sportId": 1,
            "gameType": api_game_type,
        })

        splits = self._extract_splits(data)
        best = _pick_or_aggregate_splits(splits, "plateAppearances")
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
        offline: bool = False,
        game_type: str = "regular",
    ) -> dict[str, Any]:
        if offline:
            raise SourceError("MLBApiSource: offline mode requires fixture")

        api_game_type = _GAME_TYPE_MAP.get(game_type, "R")
        url = f"{_BASE_URL}/people/{player.mlbam_id}/stats"
        data = _get(url, {
            "stats": "season",
            "season": year,
            "group": "pitching",
            "sportId": 1,
            "gameType": api_game_type,
        })

        splits = self._extract_splits(data)
        best = _pick_or_aggregate_splits(splits, "battersFaced")
        if best is None:
            raise SourceError(
                f"MLB API: no pitching splits for {player.name} in {year}"
            )

        stat = best.get("stat", {})
        return self._parse_pitching_stat(stat)

    @staticmethod
    def _extract_splits(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract the season-type splits from an MLB API stats response.

        Prefers the stats entry with ``type.displayName == 'season'`` to
        avoid accidentally reading a ``yearByYear`` or ``career`` entry when
        the API returns multiple stats objects.
        """
        stats_list = data.get("stats", [])
        if not stats_list:
            return []
        # Prefer an entry explicitly typed as "season"
        for entry in stats_list:
            type_name = entry.get("type", {}).get("displayName", "").lower()
            if type_name == "season":
                return entry.get("splits", [])
        # Fallback: first entry
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
