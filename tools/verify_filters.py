"""CLI entry point for filter verification with strict real-validation gating.

Usage
-----
Record fixtures (requires network)::

    python -m tools.verify_filters --record-fixtures
    python -m tools.verify_filters --record-fixtures --player-id 665742 --year 2024

Run offline (strict by default)::

    python -m tools.verify_filters --offline
    python -m tools.verify_filters --offline --filter game_scope
    python -m tools.verify_filters --offline --filter pitcher_hand

Run all filter verification tests directly via pytest::

    pytest tests/filter_verification/ -q
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_FIXTURE_ROOT = _PROJECT_ROOT / "tests" / "verification_fixtures"
_FILTER_FIXTURE_ROOT = _FIXTURE_ROOT / "filter_validation"
RAW_DIR = _FILTER_FIXTURE_ROOT / "raw"
SUMMARY_DIR = _FILTER_FIXTURE_ROOT / "summaries"

SEED_PLAYERS = [
    {"player_type": "batter", "mlbam_id": 592450, "year": 2024, "name": "Aaron Judge"},
    {"player_type": "batter", "mlbam_id": 665742, "year": 2024, "name": "Juan Soto"},
    {"player_type": "batter", "mlbam_id": 660271, "year": 2024, "name": "Shohei Ohtani"},
    {"player_type": "pitcher", "mlbam_id": 675911, "year": 2023, "name": "Spencer Strider"},
    {"player_type": "pitcher", "mlbam_id": 669203, "year": 2021, "name": "Corbin Burnes"},
]

_KNOWN_BBREF_IDS = {
    592450: "judgeaa01",
    665742: "sotoju01",
    660271: "ohtansh01",
}


def _raw_fixture_path(player: dict[str, Any]) -> Path:
    return RAW_DIR / (
        f"{player['player_type']}_{player['mlbam_id']}_{player['year']}_all.parquet"
    )


def _summary_fixture_path(source: str, player: dict[str, Any], split: str) -> Path:
    base_name = f"{player['player_type']}_{player['mlbam_id']}_{player['year']}"
    return SUMMARY_DIR / source / f"{base_name}_{split}.json"


def _summary_fixture_compat_path(source: str, player: dict[str, Any]) -> Path:
    base_name = f"{player['player_type']}_{player['mlbam_id']}_{player['year']}"
    return SUMMARY_DIR / source / f"{base_name}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _record_raw_fixture(player: dict[str, Any]) -> None:
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

    print(f"Fetching Statcast raw fixture for {name} ({player_type}, {mlbam_id}, {year})...")

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    if player_type == "batter":
        df = pb.statcast_batter(start_date, end_date, mlbam_id)
    else:
        df = pb.statcast_pitcher(start_date, end_date, mlbam_id)

    out_path = _raw_fixture_path(player)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} rows)")


def _match_player_row(df, name: str):
    """Find the best matching player row from a FanGraphs season dataframe."""
    if "Name" not in df.columns:
        return None

    exact = df[df["Name"].str.lower() == name.lower()]
    if not exact.empty:
        return exact.iloc[0]

    name_parts = name.split()
    last_name = name_parts[-1] if name_parts else name
    fallback = df[df["Name"].str.contains(last_name, case=False, na=False)]
    if not fallback.empty:
        return fallback.iloc[0]

    return None


def _record_fangraphs_full_summary(player: dict[str, Any], pb) -> None:
    """Fetch FanGraphs season summary and save as full split fixture."""
    player_type = player["player_type"]
    mlbam_id = player["mlbam_id"]
    year = player["year"]
    name = player["name"]

    print(f"Fetching FanGraphs full-season summary for {name}...")

    try:
        if player_type == "batter":
            fg_df = pb.batting_stats(year, year, qual=0)
        else:
            fg_df = pb.pitching_stats(year, year, qual=0)
    except Exception as exc:
        print(f"  WARNING: FanGraphs fetch failed for {name}: {exc}")
        return

    if fg_df is None or fg_df.empty:
        print(f"  WARNING: FanGraphs returned empty data for {name}")
        return

    row = _match_player_row(fg_df, name)
    if row is None:
        print(f"  WARNING: Could not find {name} in FanGraphs data")
        return

    stats = {}
    for col in ["PA", "wOBA", "K%", "BB%"]:
        if col not in row.index:
            continue
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

    full_path = _summary_fixture_path("fangraphs", player, "full")
    compat_path = _summary_fixture_compat_path("fangraphs", player)
    _write_json(full_path, summary)
    _write_json(compat_path, summary)
    print(f"  Saved: {full_path}")


def _lookup_bbref_id(player: dict[str, Any], pb) -> str | None:
    """Resolve Baseball Reference ID for a player."""
    mlbam_id = int(player["mlbam_id"])
    if mlbam_id in _KNOWN_BBREF_IDS:
        return _KNOWN_BBREF_IDS[mlbam_id]

    try:
        lookup = pb.playerid_reverse_lookup([str(mlbam_id)], key_type="mlbam")
    except Exception as exc:
        print(f"  WARNING: Could not resolve bbref id for {mlbam_id}: {exc}")
        return None

    if lookup is None or lookup.empty or "key_bbref" not in lookup.columns:
        return None

    series = lookup["key_bbref"].dropna()
    if series.empty:
        return None

    value = str(series.iloc[0]).strip()
    return value or None


def _extract_split_pa(split_df, split_label: str) -> int | None:
    """Extract PA for a named split label from pybaseball.get_splits output."""
    if split_df is None or getattr(split_df, "empty", True):
        return None
    if "PA" not in split_df.columns:
        return None

    target = split_label.strip().lower()
    candidates: list[tuple[Any, str]] = []

    for idx in split_df.index:
        if isinstance(idx, tuple):
            split_type = str(idx[0]).strip().lower() if len(idx) > 1 else ""
            split_name = str(idx[-1]).strip().lower()
        else:
            split_type = ""
            split_name = str(idx).strip().lower()
        if split_name == target:
            candidates.append((idx, split_type))

    if not candidates:
        return None

    preferred_idx = next(
        (idx for idx, split_type in candidates if "platoon" in split_type),
        candidates[0][0],
    )
    row = split_df.loc[preferred_idx]
    if hasattr(row, "ndim") and getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]

    pa = row.get("PA") if hasattr(row, "get") else row["PA"]
    if pa is None:
        return None
    if hasattr(pa, "item"):
        pa = pa.item()
    try:
        return int(pa)
    except (TypeError, ValueError):
        return int(float(pa))


def _record_baseball_ref_handed_split_summaries(player: dict[str, Any], pb) -> None:
    """Record vsL/vsR PA split fixtures from Baseball Reference."""
    if player["player_type"] != "batter":
        return

    name = player["name"]
    year = int(player["year"])
    bbref_id = _lookup_bbref_id(player, pb)
    if not bbref_id:
        print(f"  WARNING: No bbref id for {name}; skipping split fixtures.")
        return

    print(f"Fetching Baseball Reference handedness splits for {name} ({bbref_id})...")
    try:
        split_df = pb.get_splits(bbref_id, year=year)
    except Exception as exc:
        print(f"  WARNING: Baseball Reference split fetch failed for {name}: {exc}")
        return

    for split_key, split_label in (("vsL", "vs LHP"), ("vsR", "vs RHP")):
        pa = _extract_split_pa(split_df, split_label)
        if pa is None:
            print(
                f"  WARNING: Could not find split PA for {name} split={split_label}; "
                "expected for pitcher-hand validation."
            )
            continue

        summary = {
            "source": "baseball_ref",
            "player_type": player["player_type"],
            "mlbam_id": player["mlbam_id"],
            "year": player["year"],
            "scope": "regular",
            "split": split_key,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "stats": {"PA": pa},
        }
        out_path = _summary_fixture_path("baseball_ref", player, split_key)
        _write_json(out_path, summary)
        print(f"  Saved: {out_path}")


def _record_summary_fixture(player: dict[str, Any]) -> None:
    """Fetch external summary fixtures used by filter verification tests."""
    try:
        import pybaseball as pb
    except ImportError:
        print("WARNING: pybaseball not installed; skipping summary fixture recording.")
        return

    _record_fangraphs_full_summary(player, pb)
    _record_baseball_ref_handed_split_summaries(player, pb)


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


class FilterValidationPlugin:
    """Collects synthetic/real/external coverage stats for a pytest run."""

    def __init__(self) -> None:
        self.synthetic_nodeids: set[str] = set()
        self.real_nodeids: set[str] = set()
        self.external_nodeids: set[str] = set()
        self._node_outcomes: dict[str, tuple[str, str | None]] = {}

    @property
    def tracked_nodeids(self) -> set[str]:
        return self.synthetic_nodeids | self.real_nodeids

    def pytest_collection_modifyitems(self, session, config, items) -> None:
        for item in items:
            nodeid = item.nodeid
            if "tests/filter_verification/" not in nodeid:
                continue
            if "RealFixtures" in nodeid:
                self.real_nodeids.add(nodeid)
                if "matches_fangraphs" in nodeid:
                    self.external_nodeids.add(nodeid)
            else:
                self.synthetic_nodeids.add(nodeid)

    def pytest_runtest_logreport(self, report) -> None:
        nodeid = report.nodeid
        if nodeid not in self.tracked_nodeids:
            return

        if report.when == "setup" and report.skipped:
            self._record_outcome(nodeid, "skipped", self._extract_skip_reason(report))
            return

        if report.when != "call":
            return

        if report.passed:
            self._record_outcome(nodeid, "passed", None)
        elif report.failed:
            self._record_outcome(nodeid, "failed", None)
        elif report.skipped:
            self._record_outcome(nodeid, "skipped", self._extract_skip_reason(report))

    def _record_outcome(self, nodeid: str, status: str, reason: str | None) -> None:
        previous = self._node_outcomes.get(nodeid)
        if previous is None or (previous[0] == "skipped" and status in {"passed", "failed"}):
            self._node_outcomes[nodeid] = (status, reason)

    @staticmethod
    def _extract_skip_reason(report) -> str:
        longrepr = report.longrepr
        if isinstance(longrepr, tuple) and len(longrepr) >= 3:
            reason = str(longrepr[2])
        else:
            reason = str(longrepr)
        reason = reason.strip()
        if reason.startswith("Skipped:"):
            reason = reason[len("Skipped:"):].strip()
        return reason

    def _counts_for(self, nodeids: set[str]) -> dict[str, int]:
        counts = {"selected": len(nodeids), "passed": 0, "failed": 0, "skipped": 0}
        for nodeid in nodeids:
            status, _reason = self._node_outcomes.get(nodeid, ("", None))
            if status in counts:
                counts[status] += 1
        return counts

    def build_summary(self) -> dict[str, Any]:
        real_skip_reasons: Counter[str] = Counter()
        for nodeid in self.real_nodeids:
            status, reason = self._node_outcomes.get(nodeid, ("", None))
            if status == "skipped":
                real_skip_reasons[reason or "unknown skip reason"] += 1

        return {
            "synthetic": self._counts_for(self.synthetic_nodeids),
            "real": self._counts_for(self.real_nodeids),
            "external": self._counts_for(self.external_nodeids),
            "real_skip_reasons": dict(real_skip_reasons),
        }


def _print_summary(summary: dict[str, Any]) -> None:
    synthetic = summary["synthetic"]
    real = summary["real"]
    external = summary["external"]

    print("\nValidation coverage summary:")
    print(
        "  Synthetic/internal tests: "
        f"selected={synthetic['selected']} passed={synthetic['passed']} "
        f"failed={synthetic['failed']} skipped={synthetic['skipped']}"
    )
    print(
        "  Real-fixture tests:       "
        f"selected={real['selected']} passed={real['passed']} "
        f"failed={real['failed']} skipped={real['skipped']}"
    )
    print(
        "  External comparisons:     "
        f"selected={external['selected']} passed={external['passed']} "
        f"failed={external['failed']} skipped={external['skipped']}"
    )

    reasons = summary.get("real_skip_reasons", {})
    if reasons:
        print("  Real skip reasons:")
        for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    - ({count}) {reason}")


def _gate_failures(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    real = summary["real"]
    external = summary["external"]

    if real["selected"] == 0:
        failures.append("No real-fixture tests were collected for this filter selection.")
        return failures

    if real["passed"] == 0 and real["failed"] == 0:
        failures.append("No real-fixture tests executed (synthetic/internal only).")

    if real["skipped"] > 0:
        failures.append(
            f"{real['skipped']} real-fixture tests were skipped; fixture coverage is incomplete."
        )

    if external["selected"] > 0:
        if external["passed"] == 0 and external["failed"] == 0:
            failures.append("No external comparison tests executed.")
        if external["skipped"] > 0:
            failures.append(
                f"{external['skipped']} external comparison tests were skipped."
            )

    return failures


def run_tests(filter_name: str = "all") -> int:
    """Run filter verification tests via pytest, then enforce real-validation gate."""
    import pytest as _pytest

    test_dir = str(_PROJECT_ROOT / "tests" / "filter_verification")
    args = [test_dir, "-q", "--tb=short"]

    if filter_name != "all":
        test_file = str(
            _PROJECT_ROOT / "tests" / "filter_verification" / f"test_{filter_name}.py"
        )
        args = [test_file, "-q", "--tb=short"]

    plugin = FilterValidationPlugin()
    exit_code = _pytest.main(args, plugins=[plugin])
    summary = plugin.build_summary()
    _print_summary(summary)

    if exit_code != 0:
        return exit_code

    failures = _gate_failures(summary)
    if failures:
        print("\nREAL VALIDATION GATE: FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 2

    print("\nREAL VALIDATION GATE: PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter verification harness for baseball stats app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.verify_filters --record-fixtures
  python -m tools.verify_filters --offline --filter game_scope
  python -m tools.verify_filters --offline --filter all
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
        help="Run tests using only recorded fixtures (strict real-validation gate enabled).",
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
        return

    sys.exit(run_tests(filter_name=args.filter))


if __name__ == "__main__":
    main()
