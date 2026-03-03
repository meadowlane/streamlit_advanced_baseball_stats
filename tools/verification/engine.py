"""Orchestration engine — ties together fetch, fixture, compare, and report.

Usage
-----
From code::

    from tools.verification.engine import run_verification
    reports = run_verification([592450, 660271], year=2024, player_type="batter")

From CLI::

    python -m tools.verify_stats --year 2024 --player-type batter --sample 20

Design
------
For each player the engine:
1. Fetches app stats via :class:`AppSource` (live or from fixtures).
2. Fetches external source stats via :class:`FanGraphsSource`,
   :class:`StatcastSource`, :class:`MLBApiSource`, :class:`BaseballRefSource`.
3. Normalises all values to a common scale.
4. Calls :func:`compare_all_stats` with tolerance rules.
5. Returns :class:`PlayerReport` objects for the reporting layer.

Independent sources
-------------------
``StatcastSource`` reuses the app's own computation code and is therefore **not
independent** (``is_independent = False``).  The engine automatically builds
``independent_source_names`` from the ``is_independent`` attribute on each
source and passes it to the comparison layer.  This ensures:

- PASS/FAIL verdicts are based only on truly independent external sources.
- StatcastSource results appear in the report as an informational cross-check.

Fixture strategy
----------------
* ``record_fixtures=True`` — fetches live and writes JSON to
  ``tests/verification_fixtures/{source}/{player_type}_{mlbam_id}_{year}.json``.
* ``offline=True``         — loads all data from fixtures; raises
  :class:`FixtureNotFoundError` if a fixture is absent.
* Default (neither flag)   — tries live fetch; if any source fails, continues
  without that source (and marks affected comparisons SKIP).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError
from tools.verification.sources.app_source import AppSource
from tools.verification.sources.fangraphs import FanGraphsSource
from tools.verification.sources.statcast import StatcastSource
from tools.verification.sources.mlb_api import MLBApiSource
from tools.verification.sources.baseball_ref import BaseballRefSource
from tools.verification.comparison import compare_all_stats
from tools.verification.reporting import PlayerReport
from tools.verification.fixtures import (
    save_fixture,
    load_fixture,
    fixture_exists,
    FixtureNotFoundError,
)
from tools.verification.normalization import normalize_stat


# ---------------------------------------------------------------------------
# Player registry for the golden test set
# ---------------------------------------------------------------------------

#: Pre-defined player identities for the golden verification set.
#: Run ``python -m tools.verify_stats --record-fixtures`` to create fixtures.
GOLDEN_PLAYERS: dict[str, PlayerIdentity] = {
    # ---------- Batters ----------
    "Aaron Judge 2024": PlayerIdentity(
        name="Aaron Judge",
        mlbam_id=592450,
        fg_id=9063,
        bref_id="judgea01",
    ),
    "Shohei Ohtani 2024": PlayerIdentity(
        name="Shohei Ohtani",
        mlbam_id=660271,
        fg_id=19755,
        bref_id="ohtansh01",
    ),
    "Luis Arraez 2024": PlayerIdentity(
        name="Luis Arraez",
        mlbam_id=650402,
        fg_id=18403,
        bref_id="arraelu01",
    ),
    "Juan Soto 2024": PlayerIdentity(
        name="Juan Soto",
        mlbam_id=665742,
        fg_id=20123,
        bref_id="sotoju01",
    ),
    # ---------- Pitchers ----------
    "Spencer Strider 2023": PlayerIdentity(
        name="Spencer Strider",
        mlbam_id=675911,
        fg_id=25016,
        bref_id="strisp01",
    ),
    "Corbin Burnes 2021": PlayerIdentity(
        name="Corbin Burnes",
        mlbam_id=669203,
        fg_id=26055,
        bref_id="burneco01",
    ),
    "Shohei Ohtani 2023 P": PlayerIdentity(
        name="Shohei Ohtani",
        mlbam_id=660271,
        fg_id=19755,
        bref_id="ohtansh01",
    ),
}


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

#: All external sources consulted by the engine (not including AppSource).
#: These instances are used as prototypes; ``run_verification`` may pass
#: ``game_type`` as a kwarg to each source's fetch method.
EXTERNAL_SOURCES: list[BaseSource] = [
    FanGraphsSource(),
    StatcastSource(),
    MLBApiSource(),
    BaseballRefSource(),
]


def _build_independent_source_names(sources: list[BaseSource]) -> frozenset[str]:
    """Return the set of source names that are flagged as independent."""
    return frozenset(s.source_name for s in sources if s.is_independent)


# ---------------------------------------------------------------------------
# Core orchestration helpers
# ---------------------------------------------------------------------------


def _fetch_with_fixture_support(
    source: BaseSource,
    player: PlayerIdentity,
    year: int,
    player_type: str,
    *,
    offline: bool,
    record: bool,
    game_type: str = "regular",
) -> dict[str, Any] | None:
    """Fetch stats from *source*, using/writing fixtures as instructed.

    Returns ``None`` when the source is unavailable and we should skip it.
    """
    src = source.source_name

    # --- Offline mode: load from fixture ---
    if offline:
        try:
            return load_fixture(src, player_type, player.mlbam_id, year)
        except FixtureNotFoundError as exc:
            print(f"  [WARN] {exc}", file=sys.stderr)
            return None

    # --- Live fetch ---
    try:
        kwargs: dict[str, Any] = {"game_type": game_type}
        if player_type == "batter":
            data = source.get_batter_season(player, year, **kwargs)
        else:
            data = source.get_pitcher_season(player, year, **kwargs)
    except TypeError:
        # Source doesn't accept game_type kwarg — call without it
        try:
            if player_type == "batter":
                data = source.get_batter_season(player, year)
            else:
                data = source.get_pitcher_season(player, year)
        except SourceError as exc:
            print(f"  [WARN] {src}: {exc}", file=sys.stderr)
            return None
        except Exception as exc:
            print(f"  [WARN] {src} unexpected error: {exc}", file=sys.stderr)
            return None
    except SourceError as exc:
        print(f"  [WARN] {src}: {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [WARN] {src} unexpected error: {exc}", file=sys.stderr)
        return None

    # --- Record fixture ---
    if record:
        path = save_fixture(src, player_type, player.mlbam_id, year, data)
        print(f"  [RECORDED] {path}", file=sys.stderr)

    return data


def _fetch_app_stats(
    player: PlayerIdentity,
    year: int,
    player_type: str,
    *,
    offline: bool,
    record: bool,
    game_type: str = "regular",
) -> dict[str, Any] | None:
    """Fetch the app's own stats (with fixture support for the app source)."""
    src_name = "app"

    if offline:
        try:
            return load_fixture(src_name, player_type, player.mlbam_id, year)
        except FixtureNotFoundError as exc:
            print(f"  [WARN] App fixture missing: {exc}", file=sys.stderr)
            return None

    app = AppSource()
    try:
        kwargs: dict[str, Any] = {"game_type": game_type}
        if player_type == "batter":
            data = app.get_batter_season(player, year, **kwargs)
        else:
            data = app.get_pitcher_season(player, year, **kwargs)
    except TypeError:
        # AppSource doesn't yet accept game_type — call without it
        try:
            if player_type == "batter":
                data = app.get_batter_season(player, year)
            else:
                data = app.get_pitcher_season(player, year)
        except SourceError as exc:
            print(f"  [WARN] AppSource: {exc}", file=sys.stderr)
            return None
    except SourceError as exc:
        print(f"  [WARN] AppSource: {exc}", file=sys.stderr)
        return None

    if record:
        path = save_fixture(src_name, player_type, player.mlbam_id, year, data)
        print(f"  [RECORDED] {path}", file=sys.stderr)

    return data


def _normalize_source_dict(
    raw: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    """Apply per-stat normalization to all values in a source dict."""
    result: dict[str, Any] = {}
    for key, val in raw.items():
        norm = normalize_stat(key, val, source_name)
        if norm is not None:
            result[key] = norm
    return result


def _extract_sample_notes(stats: dict[str, Any]) -> dict[str, int | float | None]:
    """Pull sample-size metadata from a stat dict."""
    notes: dict[str, int | float | None] = {}
    for key in ("PA", "IP", "N_pitches", "N_BIP", "approx_PA"):
        val = stats.get(key)
        if val is not None:
            notes[key] = val
    return notes


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def run_verification(
    player_ids: list[int],
    year: int,
    player_type: Literal["batter", "pitcher"],
    *,
    player_identities: dict[int, PlayerIdentity] | None = None,
    stats_to_check: list[str] | None = None,
    offline: bool = False,
    record_fixtures: bool = False,
    verbose: bool = False,
    game_type: str = "regular",
) -> list[PlayerReport]:
    """Verify stats for a list of MLBAM player IDs.

    Parameters
    ----------
    player_ids:
        MLBAM IDs to verify.
    year:
        Season year.
    player_type:
        ``"batter"`` or ``"pitcher"``.
    player_identities:
        Optional pre-built ``{mlbam_id: PlayerIdentity}`` map.  When omitted,
        minimal identities are constructed from the MLBAM ID alone (name lookup
        is skipped — name-based matching on FG/BRef will fail without it).
    stats_to_check:
        Subset of canonical stat keys.  ``None`` = all found.
    offline:
        Load all data from recorded fixtures; raise if missing.
    record_fixtures:
        Fetch live and write fixtures.  Implies online mode.
    verbose:
        Print progress to stderr.
    game_type:
        Season scope: ``"regular"`` (default) or ``"all"``.
        FanGraphs and BRef always return regular-season only.
        Statcast data is filtered to ``game_type == 'R'`` when this is
        ``"regular"``.  MLB API uses ``gameType=R``.
        If ``"all"`` is requested, Statcast data is not filtered and a note
        is added to the report.
    """
    independent_names = _build_independent_source_names(EXTERNAL_SOURCES)

    reports: list[PlayerReport] = []

    for mlbam_id in player_ids:
        # Resolve or build a minimal PlayerIdentity
        if player_identities and mlbam_id in player_identities:
            player = player_identities[mlbam_id]
        else:
            # Minimal identity — source adapters using name/fg_id may fail
            player = PlayerIdentity(name=str(mlbam_id), mlbam_id=mlbam_id)

        if verbose:
            print(f"\n[{player.name} / {year} / {player_type} / game_type={game_type}]",
                  file=sys.stderr)

        # 1. App stats
        app_raw = _fetch_app_stats(
            player, year, player_type,
            offline=offline, record=record_fixtures, game_type=game_type,
        )
        if app_raw is None:
            print(f"  [SKIP] Could not fetch app stats for {player.name}", file=sys.stderr)
            continue
        app_stats = _normalize_source_dict(app_raw, "app")
        sample_notes = _extract_sample_notes(app_raw)

        # 2. External sources
        source_dicts: dict[str, dict[str, Any]] = {}
        for src in EXTERNAL_SOURCES:
            raw = _fetch_with_fixture_support(
                src, player, year, player_type,
                offline=offline, record=record_fixtures, game_type=game_type,
            )
            if raw is not None:
                source_dicts[src.source_name] = _normalize_source_dict(raw, src.source_name)

        if not source_dicts:
            print(
                f"  [WARN] No external sources returned data for {player.name}; skipping.",
                file=sys.stderr,
            )
            continue

        # 3. Compare — only independent sources affect verdicts
        sample_pa = sample_notes.get("PA") or sample_notes.get("approx_PA")
        sample_ip_raw = sample_notes.get("IP")
        sample_ip: float | None = None
        if sample_ip_raw is not None:
            try:
                sample_ip = float(sample_ip_raw)
            except (TypeError, ValueError):
                pass

        comparisons = compare_all_stats(
            our_stats=app_stats,
            source_dicts=source_dicts,
            stats_to_check=stats_to_check,
            sample_pa=int(sample_pa) if sample_pa is not None else None,
            sample_ip=sample_ip,
            player_type=player_type,
            independent_sources=independent_names,
        )

        report = PlayerReport(
            player=player,
            year=year,
            player_type=player_type,
            comparisons=comparisons,
            sample_notes=sample_notes,
            game_type=game_type,
        )
        reports.append(report)

        if verbose:
            fails = [c for c in comparisons if c.verdict == "FAIL"]
            warns = [c for c in comparisons if c.verdict == "WARN"]
            non_ver = [c for c in comparisons if c.verdict == "NON_VERIFIABLE"]
            print(
                f"  → FAIL: {len(fails)}  WARN: {len(warns)}  "
                f"NON_VER: {len(non_ver)}  Total: {len(comparisons)}",
                file=sys.stderr,
            )

    return reports


# ---------------------------------------------------------------------------
# Convenience: verify the golden set
# ---------------------------------------------------------------------------


def run_golden_set_verification(
    *,
    offline: bool = True,
    record_fixtures: bool = False,
    verbose: bool = False,
    game_type: str = "regular",
) -> list[PlayerReport]:
    """Run verification for the built-in golden player set.

    This is what ``pytest tests/test_stat_verification.py`` calls.
    """
    batter_players = {
        592450: GOLDEN_PLAYERS["Aaron Judge 2024"],
        660271: GOLDEN_PLAYERS["Shohei Ohtani 2024"],
        650402: GOLDEN_PLAYERS["Luis Arraez 2024"],
        665742: GOLDEN_PLAYERS["Juan Soto 2024"],
    }
    pitcher_players_2023 = {
        675911: GOLDEN_PLAYERS["Spencer Strider 2023"],
    }
    pitcher_players_2021 = {
        669203: GOLDEN_PLAYERS["Corbin Burnes 2021"],
    }

    all_reports: list[PlayerReport] = []

    all_reports.extend(
        run_verification(
            list(batter_players.keys()),
            year=2024,
            player_type="batter",
            player_identities=batter_players,
            offline=offline,
            record_fixtures=record_fixtures,
            verbose=verbose,
            game_type=game_type,
        )
    )
    all_reports.extend(
        run_verification(
            list(pitcher_players_2023.keys()),
            year=2023,
            player_type="pitcher",
            player_identities=pitcher_players_2023,
            offline=offline,
            record_fixtures=record_fixtures,
            verbose=verbose,
            game_type=game_type,
        )
    )
    all_reports.extend(
        run_verification(
            list(pitcher_players_2021.keys()),
            year=2021,
            player_type="pitcher",
            player_identities=pitcher_players_2021,
            offline=offline,
            record_fixtures=record_fixtures,
            verbose=verbose,
            game_type=game_type,
        )
    )

    return all_reports
