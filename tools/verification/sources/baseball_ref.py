"""Baseball Reference source adapter — fetches via pybaseball.

Uses ``pybaseball.batting_stats_bref`` and ``pybaseball.pitching_stats_bref``
to provide an independent third source for traditional and some advanced stats.

Coverage
--------
- Batters: AVG, OBP, SLG, OPS, PA, H, HR, BB, SO, HBP, wOBA (approx)
- Pitchers: ERA, FIP, W, L, SO, BB, HR, IP, K%, BB%

Limitations
-----------
* BRef uses different K-rate and BB-rate denominators for pitchers
  (BF vs PA) — normalised before comparison.
* BRef wOBA uses FanGraphs linear weights but applied to BRef's own
  component counts — may differ from FG wOBA by ±0.005.
* GB%, FB%, HardHit%, Barrel%, xwOBA not available from BRef.
* Multi-team players: BRef publishes a "TOT" row for traded players.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import pybaseball as pb  # type: ignore[import-untyped]  # noqa: E402

from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError  # noqa: E402
from tools.verification.normalization import (  # noqa: E402
    normalize_count,
    normalize_float,
    normalize_avg,
    normalize_ip,
    normalize_player_name,
)


def _find_player_row(
    df: pd.DataFrame,
    player: PlayerIdentity,
) -> pd.Series | None:
    """Locate a player in a BRef DataFrame.

    BRef uses the 'Name' column.  For traded players, it emits a 'TOT' row
    with ``Tm == 'TOT'``; we prefer that when multiple rows exist.
    """
    if "Name" not in df.columns:
        return None

    target_norm = normalize_player_name(player.name)
    name_col_norm = df["Name"].apply(normalize_player_name)
    matches = df[name_col_norm == target_norm]

    if matches.empty:
        return None
    if len(matches) == 1:
        return matches.iloc[0]

    # Multiple rows (traded player) — prefer TOT
    tm_col = None
    for col in ("Tm", "Team", "tm"):
        if col in matches.columns:
            tm_col = col
            break
    if tm_col:
        tot_rows = matches[matches[tm_col].str.upper() == "TOT"]
        if not tot_rows.empty:
            return tot_rows.iloc[0]

    # Fall back to the row with the most PA / IP
    for pa_col in ("PA", "IP", "TBF", "BF"):
        if pa_col in matches.columns:
            numeric = pd.to_numeric(matches[pa_col], errors="coerce")
            idx = numeric.idxmax()
            return matches.loc[idx]

    return matches.iloc[0]


def _safe_pct(
    numerator: int | None,
    denominator: int | None,
) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return round(numerator / denominator * 100.0, 1)


class BaseballRefSource(BaseSource):
    """Fetches stats from Baseball Reference via pybaseball."""

    @property
    def source_name(self) -> str:
        return "baseball_ref"

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
                f"BaseballRefSource only supports 'regular' scope; "
                f"requested {game_type!r} — SKIP"
            )
        if offline:
            raise SourceError("BaseballRefSource: offline mode requires fixture")

        pb.cache.enable()
        try:
            df = pb.batting_stats_bref(year)
        except Exception as exc:
            raise SourceError(f"BRef batting_stats_bref({year}) failed: {exc}") from exc

        if df is None or df.empty:
            raise SourceError(f"BRef returned empty batting data for {year}")

        row = _find_player_row(df, player)
        if row is None:
            raise SourceError(
                f"Player '{player.name}' not found in BRef batting data for {year}"
            )

        result: dict[str, Any] = {}

        def _add(our_key: str, *bref_cols: str, transform: Any = None) -> None:
            for col in bref_cols:
                if col in df.columns:
                    raw = row[col]
                    try:
                        if pd.isna(raw):
                            return
                    except (TypeError, ValueError):
                        pass
                    if transform is not None:
                        val = transform(raw)
                    else:
                        val = raw
                    if val is not None:
                        result[our_key] = val
                    return

        _add("PA", "PA", transform=normalize_count)
        _add("H", "H", transform=normalize_count)
        _add("HR", "HR", transform=normalize_count)
        _add("BB", "BB", transform=normalize_count)
        _add("SO", "SO", transform=normalize_count)
        _add("HBP", "HBP", transform=normalize_count)
        _add("AVG", "BA", "AVG", transform=normalize_avg)
        _add("OBP", "OBP", transform=normalize_avg)
        _add("SLG", "SLG", transform=normalize_avg)
        _add("OPS", "OPS", transform=normalize_avg)

        # wOBA — BRef publishes this in some tables
        _add("wOBA", "wOBA", transform=normalize_avg)

        # Derive K% and BB%
        pa = result.get("PA")
        so = result.get("SO")
        bb = result.get("BB")
        k_pct = _safe_pct(so, pa)
        bb_pct = _safe_pct(bb, pa)
        if k_pct is not None:
            result["K%"] = k_pct
        if bb_pct is not None:
            result["BB%"] = bb_pct

        return result

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
                f"BaseballRefSource only supports 'regular' scope; "
                f"requested {game_type!r} — SKIP"
            )
        if offline:
            raise SourceError("BaseballRefSource: offline mode requires fixture")

        pb.cache.enable()
        try:
            df = pb.pitching_stats_bref(year)
        except Exception as exc:
            raise SourceError(f"BRef pitching_stats_bref({year}) failed: {exc}") from exc

        if df is None or df.empty:
            raise SourceError(f"BRef returned empty pitching data for {year}")

        row = _find_player_row(df, player)
        if row is None:
            raise SourceError(
                f"Pitcher '{player.name}' not found in BRef pitching data for {year}"
            )

        result: dict[str, Any] = {}

        def _add(our_key: str, *bref_cols: str, transform: Any = None) -> None:
            for col in bref_cols:
                if col in df.columns:
                    raw = row[col]
                    try:
                        if pd.isna(raw):
                            return
                    except (TypeError, ValueError):
                        pass
                    if transform is not None:
                        val = transform(raw)
                    else:
                        val = raw
                    if val is not None:
                        result[our_key] = val
                    return

        _add("W", "W", transform=normalize_count)
        _add("L", "L", transform=normalize_count)
        _add("SO", "SO", transform=normalize_count)
        _add("BB", "BB", transform=normalize_count)
        _add("HR", "HR", transform=normalize_count)
        _add("HBP", "HBP", transform=normalize_count)
        _add("IP", "IP", transform=normalize_ip)
        _add("ERA", "ERA", transform=normalize_avg)
        _add("FIP", "FIP", transform=normalize_avg)
        _add("wOBA", "wOBA", transform=normalize_avg)

        # Derive K%, BB%, K-BB% — BRef denominator is BF (batters faced)
        bf = normalize_count(row["BF"]) if "BF" in df.columns else None
        so = result.get("SO")
        bb = result.get("BB")
        k_pct = _safe_pct(so, bf)
        bb_pct = _safe_pct(bb, bf)
        if k_pct is not None:
            result["K%"] = k_pct
        if bb_pct is not None:
            result["BB%"] = bb_pct
        if k_pct is not None and bb_pct is not None:
            result["K-BB%"] = round(k_pct - bb_pct, 1)

        # GB% — BRef publishes GB% in some years
        for gb_col in ("GB%", "GB"):
            if gb_col in df.columns:
                raw = row[gb_col]
                try:
                    if not pd.isna(raw):
                        val = normalize_float(raw)
                        if val is not None:
                            # BRef stores as integer 0-100 or 0-1 fraction
                            result["GB%"] = val if val > 1.5 else round(val * 100.0, 1)
                        break
                except (TypeError, ValueError):
                    pass

        return result
