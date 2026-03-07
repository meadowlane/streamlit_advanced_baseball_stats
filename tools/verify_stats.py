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

# Override fixture root:
    python -m tools.verify_stats --golden-set --offline \\
        --fixture-root /path/to/my/fixtures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.verification.engine import (  # noqa: E402
    run_golden_set_verification,
    run_verification,
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

    # --- Scope ---
    scope = parser.add_argument_group("Scope")
    scope.add_argument(
        "--game-type",
        choices=["regular", "all"],
        default="regular",
        help=(
            "Season scope for stat comparisons (default: regular).  "
            "'regular' = regular season only; filters Statcast data to game_type='R'.  "
            "FanGraphs and Baseball-Reference always report regular season only.  "
            "MLB Stats API uses gameType=R by default."
        ),
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
            "Fetch live data and write fixtures to the fixture root.  "
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
    out.add_argument(
        "--fixture-root",
        metavar="DIR",
        help=(
            "Override the fixture directory root "
            "(default: tests/verification_fixtures/ inside the repo).  "
            "Affects both --record-fixtures writes and --offline reads."
        ),
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
            print("[WARN] FanGraphs data has no 'Name' column", file=sys.stderr)
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
    """Pull a random sample of player IDs from FanGraphs season data.

    Resolution order:
    1. FanGraphs batting_stats / pitching_stats via pybaseball (preferred).
    2. MLB Stats API leaders endpoint fallback (if FG fails or produces no IDs).
    3. Built-in golden player set as last resort (prints a WARN).
    """
    ids, identities = _sample_from_fangraphs(year, player_type, n)
    if ids:
        return ids, identities

    print(
        "[WARN] FanGraphs sample failed; trying MLB Stats API leaders ...",
        file=sys.stderr,
    )
    ids, identities = _sample_from_mlb_api(year, player_type, n)
    if ids:
        return ids, identities

    print(
        "[WARN] MLB Stats API sample also failed; "
        f"falling back to built-in golden set for {player_type}s.",
        file=sys.stderr,
    )
    return _sample_from_golden_set(player_type, n)


def _sample_from_fangraphs(
    year: int,
    player_type: str,
    n: int,
) -> tuple[list[int], dict[int, PlayerIdentity]]:
    """Try to get a sample from FanGraphs via pybaseball.  Returns ([], {}) on failure."""
    try:
        import pybaseball as pb
        pb.cache.enable()
        if player_type == "batter":
            df = pb.batting_stats(year, qual=100)
        else:
            df = pb.pitching_stats(year, qual=20)
    except Exception as exc:
        print(f"  [WARN] FanGraphs fetch failed: {exc}", file=sys.stderr)
        return [], {}

    if df is None or df.empty:
        return [], {}

    # Build MLBAM ID lookup
    id_map = None
    if "key_mlbam" not in df.columns:
        try:
            from pybaseball import playerid_reverse_lookup
            fg_ids = df["IDfg"].dropna().astype(int).tolist()
            id_map = playerid_reverse_lookup(fg_ids, key_type="fangraphs")
            # playerid_reverse_lookup result uses 'key_fangraphs' not 'IDfg'
        except Exception:
            id_map = None
    # else: key_mlbam is already in df, we can read it directly from each row

    sampled = df.sample(min(n, len(df)), random_state=42)
    ids: list[int] = []
    identities: dict[int, PlayerIdentity] = {}

    for _, row in sampled.iterrows():
        name = str(row.get("Name", "Unknown"))
        fg_id_raw = row.get("IDfg")
        fg_id = int(fg_id_raw) if fg_id_raw is not None else None

        # Try direct column first
        mlbam_raw = row.get("key_mlbam")

        # If not present, look up via the reverse-lookup table
        if mlbam_raw is None and id_map is not None and fg_id is not None:
            try:
                # playerid_reverse_lookup uses 'key_fangraphs' as the FG ID column
                match = id_map[id_map["key_fangraphs"] == fg_id]
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


def _sample_from_mlb_api(
    year: int,
    player_type: str,
    n: int,
) -> tuple[list[int], dict[int, PlayerIdentity]]:
    """Fetch a sample of player MLBAM IDs from the MLB Stats API leaders endpoint."""
    try:
        import requests
        group = "hitting" if player_type == "batter" else "pitching"
        pa_stat = "plateAppearances" if player_type == "batter" else "inningsPitched"
        url = "https://statsapi.mlb.com/api/v1/stats/leaders"
        params = {
            "leaderCategories": pa_stat,
            "season": year,
            "leaderGameTypes": "R",
            "sportId": 1,
            "limit": max(n * 3, 60),  # fetch extra so we have room to deduplicate
            "statGroup": group,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [WARN] MLB API leaders fetch failed: {exc}", file=sys.stderr)
        return [], {}

    leaders = []
    for category in data.get("leagueLeaders", []):
        leaders.extend(category.get("leaders", []))

    ids: list[int] = []
    identities: dict[int, PlayerIdentity] = {}

    import random
    rng = random.Random(42)
    rng.shuffle(leaders)

    for entry in leaders:
        if len(ids) >= n:
            break
        person = entry.get("person", {})
        mlbam_id_raw = person.get("id")
        name = person.get("fullName", "Unknown")
        if mlbam_id_raw is None:
            continue
        try:
            mlbam_id = int(mlbam_id_raw)
        except (TypeError, ValueError):
            continue
        if mlbam_id in identities:
            continue
        ids.append(mlbam_id)
        identities[mlbam_id] = PlayerIdentity(name=name, mlbam_id=mlbam_id)

    return ids, identities


def _sample_from_golden_set(
    player_type: str,
    n: int,
) -> tuple[list[int], dict[int, PlayerIdentity]]:
    """Return players from the built-in golden set matching player_type."""
    batter_keys = {"Aaron Judge 2024", "Shohei Ohtani 2024", "Luis Arraez 2024", "Juan Soto 2024"}
    pitcher_keys = {"Spencer Strider 2023", "Corbin Burnes 2021", "Shohei Ohtani 2023 P"}

    key_set = batter_keys if player_type == "batter" else pitcher_keys
    candidates = [(k, v) for k, v in GOLDEN_PLAYERS.items() if k in key_set]

    ids: list[int] = []
    identities: dict[int, PlayerIdentity] = {}
    for _, player in candidates[:n]:
        if player.mlbam_id not in identities:
            ids.append(player.mlbam_id)
            identities[player.mlbam_id] = player
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

    # --- Apply fixture root override ---
    if args.fixture_root:
        from tools.verification.fixtures import set_fixture_root
        set_fixture_root(args.fixture_root)
        if args.verbose:
            print(f"[Fixtures] Root set to: {args.fixture_root}", file=sys.stderr)

    # --- Run ---
    game_type = args.game_type

    if args.verbose:
        print(f"[Config] game_type={game_type}", file=sys.stderr)

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
            game_type=game_type,
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
            game_type=game_type,
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
    # WARN / SKIP / NON_VERIFIABLE do NOT trigger a non-zero exit
    from tools.verification.reporting import verdict_counts
    counts = verdict_counts(reports)
    return 1 if counts.get("FAIL", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
