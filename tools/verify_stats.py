"""CLI entry point for the stat verification harness.

Usage examples
--------------
# Verify a sample of 20 batters for 2024 (live, text report):
    python -m tools.verify_stats --year 2024 --player-type batter --sample 20

# Record fixtures for the golden set:
    python -m tools.verify_stats --record-fixtures --golden-set

# Run the golden set offline (uses recorded fixtures):
    python -m tools.verify_stats --offline --golden-set --output html \\
        --output-path reports/golden_2024.html

# Verify specific players by MLBAM ID:
    python -m tools.verify_stats --year 2024 --player-type batter \\
        --player-ids 592450 660271 --output csv

# Verify a specific stat subset:
    python -m tools.verify_stats --year 2023 --player-type pitcher \\
        --player-ids 675911 --stats ERA FIP K% BB% --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.verification.engine import (  # noqa: E402
    GOLDEN_PLAYERS,
    run_golden_set_verification,
    run_verification,
    EXTERNAL_SOURCES,
)
from tools.verification.reporting import write_report  # noqa: E402
from tools.verification.sources.base import PlayerIdentity  # noqa: E402

_GAME_TYPE_HELP = (
    "Game-type scope for Statcast-based computation. "
    "'regular' (default): regular season only — matches FanGraphs/MLB API. "
    "'postseason': postseason games only (FG/BRef/MLB API will be SKIP). "
    "'all': include every game type including spring training."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.verify_stats",
        description="Cross-source stat verification harness for the baseball stats app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- What to verify ---
    who = parser.add_argument_group("What to verify")
    who.add_argument(
        "--golden-set",
        action="store_true",
        help="Verify the built-in golden player set (overrides --player-ids and --sample).",
    )
    who.add_argument(
        "--player-ids",
        nargs="+",
        type=int,
        metavar="MLBAM_ID",
        help="One or more MLBAM player IDs to verify.",
    )
    who.add_argument(
        "--player-names",
        nargs="+",
        metavar="NAME",
        help="Player display names (must match FanGraphs 'Name' column exactly).",
    )
    who.add_argument(
        "--players",
        nargs="+",
        metavar="NAME",
        help=(
            "Look up players by display name without needing --player-ids.  "
            "Names are matched against FanGraphs data.  "
            "Shortcut: --players 'Juan Soto' 'Aaron Judge'."
        ),
    )
    who.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Season year to verify (default: 2024).",
    )
    who.add_argument(
        "--player-type",
        choices=["batter", "pitcher"],
        default="batter",
        help="Player type to verify (default: batter).",
    )
    who.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Randomly sample N player IDs from FanGraphs season data.",
    )
    who.add_argument(
        "--stats",
        nargs="+",
        metavar="STAT",
        help=(
            "Limit verification to these stat keys (e.g. --stats wOBA K%% ERA FIP).  "
            "Default: all verifiable stats."
        ),
    )
    who.add_argument(
        "--game-type",
        choices=["regular", "postseason", "all"],
        default="regular",
        dest="game_type",
        help=_GAME_TYPE_HELP,
    )

    # --- Mode flags ---
    mode = parser.add_argument_group("Mode")
    mode.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Load all data from recorded fixtures.  "
            "Raises an error if any fixture is missing."
        ),
    )
    mode.add_argument(
        "--record-fixtures",
        action="store_true",
        help=(
            "Fetch live data and write fixtures to "
            "tests/verification_fixtures/.  "
            "Cannot be combined with --offline."
        ),
    )

    # --- Output ---
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output",
        choices=["text", "csv", "json", "html"],
        default="text",
        help="Report format (default: text).",
    )
    out.add_argument(
        "--output-path",
        metavar="FILE",
        help=(
            "Write report to this file.  "
            "When omitted, text is printed to stdout; other formats also print to stdout."
        ),
    )
    out.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and per-player status to stderr.",
    )

    return parser


def _resolve_player_ids(
    args: argparse.Namespace,
) -> tuple[list[int], dict[int, PlayerIdentity]]:
    """Return (mlbam_id_list, identity_map) from CLI arguments."""
    ids: list[int] = []
    identities: dict[int, PlayerIdentity] = {}

    if args.players:
        # Name-only lookup: resolve MLBAM IDs from FanGraphs data
        ids, identities = _lookup_players_by_name(
            args.players, args.year, args.player_type
        )

    elif args.player_ids:
        ids = list(args.player_ids)
        if args.player_names:
            for mlbam_id, name in zip(ids, args.player_names):
                identities[mlbam_id] = PlayerIdentity(name=name, mlbam_id=mlbam_id)
        else:
            for mlbam_id in ids:
                identities[mlbam_id] = PlayerIdentity(name=str(mlbam_id), mlbam_id=mlbam_id)

    elif args.sample:
        ids, identities = _sample_players(args.year, args.player_type, args.sample)

    return ids, identities


def _lookup_players_by_name(
    names: list[str],
    year: int,
    player_type: str,
) -> tuple[list[int], dict[int, PlayerIdentity]]:
    """Resolve player names → MLBAM IDs by pulling FanGraphs season data.

    Performs a case-insensitive name match against the FanGraphs roster.
    For each match, attempts to cross-reference the FanGraphs ID with an MLBAM ID
    via ``pybaseball.playerid_reverse_lookup``.
    """
    import pybaseball as pb

    pb.cache.enable()
    try:
        if player_type == "batter":
            df = pb.batting_stats(year, qual=1)
        else:
            df = pb.pitching_stats(year, qual=1)
    except Exception as exc:
        print(f"[ERROR] Could not fetch FanGraphs data for name lookup: {exc}", file=sys.stderr)
        return [], {}

    if df is None or df.empty:
        return [], {}

    ids: list[int] = []
    identities: dict[int, PlayerIdentity] = {}

    for target_name in names:
        name_lower = target_name.lower()
        if "Name" not in df.columns:
            print(f"[WARN] FanGraphs data has no 'Name' column", file=sys.stderr)
            continue
        matches = df[df["Name"].str.lower() == name_lower]
        if matches.empty:
            print(f"[WARN] Player '{target_name}' not found in FanGraphs {year} data.", file=sys.stderr)
            continue

        row = matches.iloc[0]
        name = str(row.get("Name", target_name))
        fg_id_raw = row.get("IDfg")
        fg_id = int(fg_id_raw) if fg_id_raw is not None else None

        # Try to find MLBAM ID
        mlbam_id: int | None = None
        mlbam_raw = row.get("key_mlbam")
        if mlbam_raw is not None:
            try:
                mlbam_id = int(mlbam_raw)
            except (TypeError, ValueError):
                pass

        if mlbam_id is None and fg_id is not None:
            try:
                from pybaseball import playerid_reverse_lookup
                id_map = playerid_reverse_lookup([fg_id], key_type="fangraphs")
                if not id_map.empty:
                    mlbam_raw = id_map["key_mlbam"].iloc[0]
                    mlbam_id = int(mlbam_raw)
            except Exception as exc:
                print(f"[WARN] Could not resolve MLBAM ID for '{name}': {exc}", file=sys.stderr)

        if mlbam_id is None:
            print(f"[WARN] Could not resolve MLBAM ID for '{target_name}' — skipping.", file=sys.stderr)
            continue

        ids.append(mlbam_id)
        identities[mlbam_id] = PlayerIdentity(name=name, mlbam_id=mlbam_id, fg_id=fg_id)

    return ids, identities


def _sample_players(
    year: int,
    player_type: str,
    n: int,
) -> tuple[list[int], dict[int, PlayerIdentity]]:
    """Pull a random sample of player IDs from FanGraphs season data."""
    import pybaseball as pb

    pb.cache.enable()
    try:
        if player_type == "batter":
            df = pb.batting_stats(year, qual=100)
        else:
            df = pb.pitching_stats(year, qual=20)
    except Exception as exc:
        print(f"[ERROR] Could not fetch FanGraphs data to build sample: {exc}", file=sys.stderr)
        return [], {}

    if df is None or df.empty:
        return [], {}

    # Need MLBAM IDs
    if "key_mlbam" not in df.columns:
        try:
            from pybaseball import playerid_reverse_lookup
            fg_ids = df["IDfg"].dropna().astype(int).tolist()
            id_map = playerid_reverse_lookup(fg_ids, key_type="fangraphs")
        except Exception:
            id_map = None
    else:
        id_map = df

    sampled = df.sample(min(n, len(df)), random_state=42)
    ids: list[int] = []
    identities: dict[int, PlayerIdentity] = {}

    for _, row in sampled.iterrows():
        name = str(row.get("Name", "Unknown"))
        fg_id_raw = row.get("IDfg")
        fg_id = int(fg_id_raw) if fg_id_raw is not None else None
        mlbam_raw = row.get("key_mlbam")
        if mlbam_raw is None and id_map is not None:
            try:
                match = id_map[id_map["IDfg"] == fg_id]
                mlbam_raw = match["key_mlbam"].iloc[0] if not match.empty else None
            except Exception:
                mlbam_raw = None
        if mlbam_raw is None:
            continue
        try:
            mlbam_id = int(mlbam_raw)
        except (TypeError, ValueError):
            continue
        ids.append(mlbam_id)
        identities[mlbam_id] = PlayerIdentity(name=name, mlbam_id=mlbam_id, fg_id=fg_id)

    return ids, identities


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate
    if args.offline and args.record_fixtures:
        parser.error("--offline and --record-fixtures are mutually exclusive.")

    if not args.golden_set and not args.player_ids and not args.sample and not args.players:
        parser.error(
            "Provide --golden-set, --player-ids, --players, or --sample to select players."
        )

    # --- Run ---
    if args.golden_set:
        if args.verbose:
            print(
                f"[Golden set] Running built-in golden player verification "
                f"(game-type scope: {args.game_type}).",
                file=sys.stderr,
            )
        reports = run_golden_set_verification(
            game_type=args.game_type,
            offline=args.offline,
            record_fixtures=args.record_fixtures,
            verbose=args.verbose,
        )
    else:
        ids, identities = _resolve_player_ids(args)
        if not ids:
            print("[ERROR] No player IDs resolved.  Exiting.", file=sys.stderr)
            return 1

        reports = run_verification(
            player_ids=ids,
            year=args.year,
            player_type=args.player_type,
            player_identities=identities,
            stats_to_check=args.stats,
            game_type=args.game_type,
            offline=args.offline,
            record_fixtures=args.record_fixtures,
            verbose=args.verbose,
        )

    if not reports:
        print("[ERROR] No reports generated.  Check player IDs and data availability.",
              file=sys.stderr)
        return 1

    # --- Report ---
    output = write_report(
        reports,
        output_format=args.output,
        output_path=args.output_path,
    )

    if args.output_path:
        print(f"[OK] Report written to: {args.output_path}", file=sys.stderr)
    else:
        print(output)

    # Exit code: 1 if any FAIL, else 0
    from tools.verification.reporting import verdict_counts
    counts = verdict_counts(reports)
    return 1 if counts.get("FAIL", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
