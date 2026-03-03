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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.verification.sources.base import BaseSource, PlayerIdentity, SourceError
from tools.verification.sources.app_source import AppSource
from tools.verification.sources.fangraphs import FanGraphsSource
from tools.verification.sources.statcast import StatcastSource
from tools.verification.sources.mlb_api import MLBApiSource
from tools.verification.sources.baseball_ref import BaseballRefSource
from tools.verification.comparison import compare_all_stats, StatComparison
from tools.verification.reporting import PlayerReport
from tools.verification.fixtures import (
    save_fixture,
    load_fixture,
    fixture_exists,
    FixtureNotFoundError,
)
from tools.verification.normalization import normalize_stat
from tools.verification.game_scope import SOURCE_SCOPE_SUPPORT, REGULAR_GAME_TYPES


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
EXTERNAL_SOURCES: list[BaseSource] = [
    FanGraphsSource(),
    StatcastSource(),
    MLBApiSource(),
    BaseballRefSource(),
]


# ---------------------------------------------------------------------------
# Core orchestration helpers
# ---------------------------------------------------------------------------


def _fetch_with_fixture_support(
    source: BaseSource,
    player: PlayerIdentity,
    year: int,
    player_type: str,
    *,
    game_type: str = "regular",
    offline: bool,
    record: bool,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Fetch stats from *source*, using/writing fixtures as instructed.

    Returns ``None`` when the source is unavailable or does not support the
    requested scope — in which case it should be treated as SKIP, not FAIL.
    """
    src = source.source_name

    # --- Scope capability check ---
    supported = SOURCE_SCOPE_SUPPORT.get(src, frozenset(["regular"]))
    if game_type not in supported:
        if verbose:
            print(
                f"  [SKIP] {src}: does not support scope={game_type!r} "
                f"(supports: {sorted(supported)}) — comparison skipped",
                file=sys.stderr,
            )
        return None

    # --- Offline mode: load from fixture ---
    if offline:
        try:
            return load_fixture(src, player_type, player.mlbam_id, year)
        except FixtureNotFoundError as exc:
            print(f"  [WARN] {exc}", file=sys.stderr)
            return None

    # --- Live fetch ---
    try:
        if player_type == "batter":
            data = source.get_batter_season(player, year, game_type=game_type)
        else:
            data = source.get_pitcher_season(player, year, game_type=game_type)
    except SourceError as exc:
        # A source that doesn't support the scope raises SourceError — treat as SKIP
        msg = str(exc)
        level = "[SKIP]" if "SKIP" in msg else "[WARN]"
        print(f"  {level} {src}: {msg}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [ERROR] {src} unexpected error: {exc}", file=sys.stderr)
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
    game_type: str = "regular",
    offline: bool,
    record: bool,
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
        if player_type == "batter":
            data = app.get_batter_season(player, year, game_type=game_type)
        else:
            data = app.get_pitcher_season(player, year, game_type=game_type)
    except SourceError as exc:
        print(f"  [WARN] AppSource: {exc}", file=sys.stderr)
        return None

    if record:
        # Strip internal metadata keys before saving fixture
        fixture_data = {k: v for k, v in data.items() if not k.startswith("_")}
        path = save_fixture(src_name, player_type, player.mlbam_id, year, fixture_data)
        print(f"  [RECORDED] {path}", file=sys.stderr)

    return data


def _normalize_source_dict(
    raw: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    """Apply per-stat normalization to all values in a source dict.

    Keys prefixed with ``_`` are internal metadata (e.g. ``_pa_by_game_type``)
    and are skipped — they are not stats and must not be compared.
    """
    result: dict[str, Any] = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue  # skip internal metadata
        norm = normalize_stat(key, val, source_name)
        if norm is not None:
            result[key] = norm
    return result


def _apply_scope_mismatch_verdicts(
    comparisons: list[StatComparison],
    pa_by_game_type: dict[str, int],
) -> list[StatComparison]:
    """Downgrade PA/counting FAIL → SCOPE_MISMATCH when non-regular games explain it.

    When the unfiltered Statcast data contains non-regular game rows (postseason,
    spring), and that extra count approximates the discrepancy vs. regular-season
    sources, the mismatch is clearly a scoping issue — not a computation bug.

    This function is only applied when ``game_type == "regular"`` but some
    non-regular rows were still found (e.g. ``game_type`` column missing from
    the raw data, preventing proper filtering).
    """
    # Count PA from non-regular game types
    extra_pa = sum(
        count for gt, count in pa_by_game_type.items()
        if gt not in REGULAR_GAME_TYPES
    )
    if extra_pa == 0:
        return comparisons

    non_regular = {
        gt: count for gt, count in pa_by_game_type.items()
        if gt not in REGULAR_GAME_TYPES
    }
    breakdown_str = "  ".join(
        f"{gt}={n}" for gt, n in sorted(non_regular.items())
    )

    # Stats directly affected by game-scope: counting stats and PA-derived rates
    scope_sensitive = frozenset(["PA", "H", "HR", "BB", "SO", "HBP"])

    for cmp in comparisons:
        if cmp.verdict != "FAIL":
            continue
        if cmp.stat not in scope_sensitive:
            continue
        if cmp.our_value is None:
            continue
        # Check if the discrepancy vs. any source is within tolerance of extra_pa
        for src_val in cmp.source_values.values():
            if src_val is None:
                continue
            try:
                delta = abs(float(cmp.our_value) - float(src_val))
            except (TypeError, ValueError):
                continue
            # Allow ±5 rounding wiggle-room
            if delta <= extra_pa + 5:
                cmp.verdict = "SCOPE_MISMATCH"
                cmp.note = (
                    f"Mismatch likely due to non-regular games in source data "
                    f"(game_type filter may not have applied). "
                    f"Non-regular PA: +{extra_pa}  ({breakdown_str})"
                )
                break

    return comparisons


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
    game_type: str = "regular",
    offline: bool = False,
    record_fixtures: bool = False,
    verbose: bool = False,
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
    game_type:
        ``"regular"`` (default), ``"postseason"``, or ``"all"``.
        Sources that don't support the requested scope are silently skipped
        (treated as SKIP, not FAIL).
    offline:
        Load all data from recorded fixtures; raise if missing.
    record_fixtures:
        Fetch live and write fixtures.  Implies online mode.
    verbose:
        Print progress to stderr.
    """
    reports: list[PlayerReport] = []

    if verbose:
        print(f"[Scope] Game type: {game_type}", file=sys.stderr)

    for mlbam_id in player_ids:
        # Resolve or build a minimal PlayerIdentity
        if player_identities and mlbam_id in player_identities:
            player = player_identities[mlbam_id]
        else:
            # Minimal identity — source adapters using name/fg_id may fail
            player = PlayerIdentity(name=str(mlbam_id), mlbam_id=mlbam_id)

        if verbose:
            print(f"\n[{player.name} / {year} / {player_type} / {game_type}]", file=sys.stderr)

        # 1. App stats
        app_raw = _fetch_app_stats(
            player, year, player_type,
            game_type=game_type,
            offline=offline, record=record_fixtures,
        )
        if app_raw is None:
            print(f"  [SKIP] Could not fetch app stats for {player.name}", file=sys.stderr)
            continue

        # Extract diagnostic metadata before normalizing
        pa_by_game_type: dict[str, int] = app_raw.get("_pa_by_game_type", {})

        app_stats = _normalize_source_dict(app_raw, "app")
        sample_notes = _extract_sample_notes(app_raw)

        # 2. External sources (only those that support the requested scope)
        source_dicts: dict[str, dict[str, Any]] = {}
        for src in EXTERNAL_SOURCES:
            raw = _fetch_with_fixture_support(
                src, player, year, player_type,
                game_type=game_type,
                offline=offline, record=record_fixtures,
                verbose=verbose,
            )
            if raw is not None:
                source_dicts[src.source_name] = _normalize_source_dict(raw, src.source_name)

        if not source_dicts:
            if verbose:
                print(
                    f"  [WARN] No external sources returned data for {player.name} "
                    f"(scope={game_type!r}); building report with app-only data.",
                    file=sys.stderr,
                )

        # 3. Compare
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
        )

        # 4. Scope-mismatch post-processing:
        #    If any non-regular game rows were present in the raw Statcast data
        #    (e.g. game_type column was absent so filter couldn't apply), downgrade
        #    PA/counting FAILs to SCOPE_MISMATCH with an explanatory note.
        if game_type == "regular" and pa_by_game_type:
            comparisons = _apply_scope_mismatch_verdicts(comparisons, pa_by_game_type)

        report = PlayerReport(
            player=player,
            year=year,
            player_type=player_type,
            game_type=game_type,
            comparisons=comparisons,
            sample_notes=sample_notes,
            pa_by_game_type=pa_by_game_type,
        )
        reports.append(report)

        if verbose:
            fails = [c for c in comparisons if c.verdict == "FAIL"]
            warns = [c for c in comparisons if c.verdict == "WARN"]
            scope_mm = [c for c in comparisons if c.verdict == "SCOPE_MISMATCH"]
            print(
                f"  → FAIL: {len(fails)}  WARN: {len(warns)}  "
                f"SCOPE_MISMATCH: {len(scope_mm)}  "
                f"Total: {len(comparisons)}",
                file=sys.stderr,
            )
            if pa_by_game_type:
                breakdown = "  ".join(
                    f"{gt}={n}" for gt, n in sorted(pa_by_game_type.items())
                )
                print(f"  PA by game_type: {breakdown}", file=sys.stderr)

    return reports


# ---------------------------------------------------------------------------
# Convenience: verify the golden set
# ---------------------------------------------------------------------------


def run_golden_set_verification(
    *,
    game_type: str = "regular",
    offline: bool = True,
    record_fixtures: bool = False,
    verbose: bool = False,
) -> list[PlayerReport]:
    """Run verification for the built-in golden player set.

    This is what ``pytest tests/test_stat_verification.py`` calls.

    Parameters
    ----------
    game_type:
        ``"regular"`` (default), ``"postseason"``, or ``"all"``.
        Sources that don't support the scope are silently skipped.
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
            game_type=game_type,
            offline=offline,
            record_fixtures=record_fixtures,
            verbose=verbose,
        )
    )
    all_reports.extend(
        run_verification(
            list(pitcher_players_2023.keys()),
            year=2023,
            player_type="pitcher",
            player_identities=pitcher_players_2023,
            game_type=game_type,
            offline=offline,
            record_fixtures=record_fixtures,
            verbose=verbose,
        )
    )
    all_reports.extend(
        run_verification(
            list(pitcher_players_2021.keys()),
            year=2021,
            player_type="pitcher",
            player_identities=pitcher_players_2021,
            game_type=game_type,
            offline=offline,
            record_fixtures=record_fixtures,
            verbose=verbose,
        )
    )

    return all_reports
