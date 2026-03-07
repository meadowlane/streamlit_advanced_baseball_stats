"""CLI entry point for filter verification.

Usage
-----
Record fixtures (requires network)::

    python -m tools.verify_filters --record-fixtures
    python -m tools.verify_filters --record-fixtures --player-id 665742 --year 2024

Run offline (default in CI)::

    python -m tools.verify_filters --offline
    python -m tools.verify_filters --offline --filter game_scope
    python -m tools.verify_filters --offline --filter pitcher_hand

Run all filter verification tests via pytest::

    pytest tests/filter_verification/ -v
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

FIXTURE_DIR = _PROJECT_ROOT / "tests" / "filter_fixtures"
RAW_DIR = FIXTURE_DIR / "raw"
SUMMARY_DIR = FIXTURE_DIR / "summaries"

SEED_PLAYERS = [
    {"player_type": "batter", "mlbam_id": 592450, "year": 2024, "name": "Aaron Judge"},
    {"player_type": "batter", "mlbam_id": 665742, "year": 2024, "name": "Juan Soto"},
    {"player_type": "batter", "mlbam_id": 660271, "year": 2024, "name": "Shohei Ohtani"},
    {"player_type": "pitcher", "mlbam_id": 675911, "year": 2023, "name": "Spencer Strider"},
    {"player_type": "pitcher", "mlbam_id": 669203, "year": 2021, "name": "Corbin Burnes"},
]


def _record_raw_fixture(player: dict) -> None:
    """Fetch raw Statcast data and save as parquet fixture."""
    try:
        import pybaseball as pb
    except ImportError:
        print("ERROR: pybaseball is required for recording fixtures.")
        print("  pip install pybaseball")
        sys.exit(1)

    pb.cache.enable()

    player_type = player["player_type"]
    mlbam_id = player["mlbam_id"]
    year = player["year"]
    name = player["name"]

    print(f"Fetching {name} ({player_type}, {mlbam_id}, {year})...")

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    if player_type == "batter":
        df = pb.statcast_batter(start_date, end_date, mlbam_id)
    else:
        df = pb.statcast_pitcher(start_date, end_date, mlbam_id)

    out_path = RAW_DIR / f"{player_type}_{mlbam_id}_{year}_all.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")


def _record_summary_fixture(player: dict) -> None:
    """Fetch FanGraphs season summary and save as JSON fixture."""
    try:
        import pybaseball as pb
    except ImportError:
        return

    player_type = player["player_type"]
    mlbam_id = player["mlbam_id"]
    year = player["year"]
    name = player["name"]

    print(f"Fetching FG summary for {name}...")

    try:
        if player_type == "batter":
            fg_df = pb.batting_stats(year, year, qual=0)
        else:
            fg_df = pb.pitching_stats(year, year, qual=0)

        # Try to find the player by name (approximate match)
        name_parts = name.split()
        last_name = name_parts[-1] if name_parts else name
        matches = fg_df[fg_df["Name"].str.contains(last_name, case=False, na=False)]

        if matches.empty:
            print(f"  WARNING: Could not find {name} in FG data")
            return

        row = matches.iloc[0]
        stats = {}
        for col in ["PA", "wOBA", "K%", "BB%"]:
            if col in row.index:
                val = row[col]
                if hasattr(val, "item"):
                    val = val.item()
                stats[col] = val

        summary = {
            "source": "fangraphs",
            "player_type": player_type,
            "mlbam_id": mlbam_id,
            "year": year,
            "scope": "regular",
            "split": "full",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "stats": stats,
        }

        out_dir = SUMMARY_DIR / "fangraphs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{player_type}_{mlbam_id}_{year}_full.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {out_path}")

    except Exception as e:
        print(f"  WARNING: FG fetch failed for {name}: {e}")


def record_fixtures(player_id: int | None = None, year: int | None = None) -> None:
    """Record all fixtures for the seed player set (or a specific player)."""
    players = SEED_PLAYERS
    if player_id is not None:
        players = [p for p in players if p["mlbam_id"] == player_id]
        if year is not None:
            players = [p for p in players if p["year"] == year]
        if not players:
            print(f"No seed player found with ID={player_id}, year={year}")
            sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    for player in players:
        _record_raw_fixture(player)
        _record_summary_fixture(player)

    print("\nFixture recording complete.")


def run_tests(filter_name: str = "all") -> int:
    """Run filter verification tests via pytest."""
    import pytest as _pytest

    test_dir = str(_PROJECT_ROOT / "tests" / "filter_verification")

    args = [test_dir, "-v", "--tb=short"]

    if filter_name != "all":
        # Run only the specific filter test module
        test_file = str(
            _PROJECT_ROOT / "tests" / "filter_verification" / f"test_{filter_name}.py"
        )
        args = [test_file, "-v", "--tb=short"]

    return _pytest.main(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter verification harness for baseball stats app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.verify_filters --record-fixtures
  python -m tools.verify_filters --offline --filter game_scope
  python -m tools.verify_filters --offline --all
        """,
    )
    parser.add_argument(
        "--record-fixtures",
        action="store_true",
        help="Fetch live data and record fixture files.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run tests using only recorded fixtures (default).",
    )
    parser.add_argument(
        "--filter",
        choices=["game_scope", "pitcher_hand", "all"],
        default="all",
        help="Which filter test to run (default: all).",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        default=None,
        help="MLBAM player ID (for --record-fixtures).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Season year (for --record-fixtures).",
    )

    args = parser.parse_args()

    if args.record_fixtures:
        record_fixtures(player_id=args.player_id, year=args.year)
    else:
        sys.exit(run_tests(filter_name=args.filter))


if __name__ == "__main__":
    main()
