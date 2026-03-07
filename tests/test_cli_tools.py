from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import tools.verify_filters as verify_filters
import tools.verify_stats as verify_stats
from tools.verification.comparison import StatComparison
from tools.verification.reporting import (
    PlayerReport,
    summary_rows,
    top_discrepancies,
    verdict_counts,
    write_report,
)
from tools.verification.sources.base import PlayerIdentity


def _sample_reports() -> list[PlayerReport]:
    player = PlayerIdentity(name="Aaron Judge", mlbam_id=592450, fg_id=9063)
    comparisons = [
        StatComparison(
            stat="HR",
            our_value=58,
            source_values={"fangraphs": 58, "mlb_api": 57},
            abs_diffs={"fangraphs": 0.0, "mlb_api": 1.0},
            rel_diffs={"fangraphs": 0.0, "mlb_api": 1 / 57},
            verdict="WARN",
            note="One source is one homer behind.",
        ),
        StatComparison(
            stat="PA",
            our_value=704,
            source_values={"fangraphs": 704, "mlb_api": 704},
            abs_diffs={"fangraphs": 0.0, "mlb_api": 0.0},
            rel_diffs={"fangraphs": 0.0, "mlb_api": 0.0},
            verdict="PASS",
        ),
        StatComparison(
            stat="AVG",
            our_value=0.323,
            source_values={"fangraphs": 0.311, "mlb_api": 0.312},
            abs_diffs={"fangraphs": 0.012, "mlb_api": 0.011},
            rel_diffs={"fangraphs": 0.038585, "mlb_api": 0.035256},
            verdict="FAIL",
            note="App calculation drifted from both external sources.",
        ),
        StatComparison(
            stat="wOBA",
            our_value=0.455,
            source_values={"statcast": 0.455},
            abs_diffs={"statcast": 0.0},
            rel_diffs={"statcast": 0.0},
            verdict="SCOPE_MISMATCH",
            note="Non-regular games explain the extra PA.",
        ),
    ]
    return [
        PlayerReport(
            player=player,
            year=2024,
            player_type="batter",
            game_type="regular",
            comparisons=comparisons,
            sample_notes={"PA": 704, "N_pitches": 2601},
            pa_by_game_type={"R": 704, "F": 14},
        )
    ]


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_reporting_formats_render_and_write(tmp_path: Path) -> None:
    reports = _sample_reports()

    rows = summary_rows(reports)
    assert len(rows) == 4
    assert rows[0]["player"] == "Aaron Judge"

    discrepancies = top_discrepancies(reports, n=2)
    assert [row["stat"] for row in discrepancies] == ["HR", "AVG"]

    counts = verdict_counts(reports)
    assert counts["PASS"] == 1
    assert counts["FAIL"] == 1
    assert counts["WARN"] == 1
    assert counts["SCOPE_MISMATCH"] == 1

    text_report = write_report(reports, "text")
    assert "BASEBALL STATS VERIFICATION REPORT" in text_report
    assert "Aaron Judge" in text_report

    csv_report = write_report(reports, "csv")
    assert "player,mlbam_id,year,player_type,stat" in csv_report
    assert "AVG" in csv_report

    json_report = write_report(reports, "json")
    payload = json.loads(json_report)
    assert payload["summary"]["FAIL"] == 1
    assert payload["players"][0]["comparisons"][0]["stat"] == "HR"

    html_path = tmp_path / "report.html"
    html_report = write_report(reports, "html", output_path=html_path)
    assert "<html" in html_report
    assert html_path.read_text(encoding="utf-8") == html_report


def test_verify_stats_sample_from_mlb_api(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_requests = SimpleNamespace(
        get=lambda url, params, timeout: _FakeResponse(
            {
                "leagueLeaders": [
                    {
                        "leaders": [
                            {"person": {"id": "1", "fullName": "Player One"}},
                            {"person": {"id": "2", "fullName": "Player Two"}},
                            {"person": {"id": "1", "fullName": "Duplicate"}},
                            {"person": {"id": None, "fullName": "Missing"}},
                            {"person": {"id": "bad", "fullName": "Invalid"}},
                        ]
                    }
                ]
            }
        )
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    ids, identities = verify_stats._sample_from_mlb_api(2024, "batter", 3)

    assert set(ids) == {1, 2}
    assert len(ids) == 2
    assert identities[1].mlbam_id == 1
    assert identities[2].mlbam_id == 2


def test_verify_stats_resolution_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = verify_stats._build_parser()
    parsed = parser.parse_args(["--golden-set", "--game-type", "postseason"])
    assert parsed.game_type == "postseason"

    monkeypatch.setattr(
        verify_stats,
        "_lookup_players_by_name",
        lambda names, year, player_type: (
            [99],
            {99: PlayerIdentity(name=names[0], mlbam_id=99)},
        ),
    )
    args = SimpleNamespace(
        players=["Custom Player"],
        year=2024,
        player_type="batter",
        player_ids=None,
        player_names=None,
        sample=None,
    )
    ids, identities = verify_stats._resolve_player_ids(args)
    assert ids == [99]
    assert identities[99].name == "Custom Player"

    monkeypatch.setattr(verify_stats, "_sample_from_fangraphs", lambda *args: ([], {}))
    monkeypatch.setattr(verify_stats, "_sample_from_mlb_api", lambda *args: ([], {}))
    monkeypatch.setattr(
        verify_stats,
        "_sample_from_golden_set",
        lambda player_type, n: (
            [592450],
            {592450: PlayerIdentity(name="Aaron Judge", mlbam_id=592450)},
        ),
    )

    ids, identities = verify_stats._sample_players(2024, "batter", 1)
    captured = capsys.readouterr()
    assert ids == [592450]
    assert identities[592450].name == "Aaron Judge"
    assert "FanGraphs sample failed" in captured.err
    assert "falling back to built-in golden set" in captured.err


def test_verify_stats_main_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reports = _sample_reports()
    fixture_roots: list[str] = []

    import tools.verification.fixtures as fixtures

    monkeypatch.setattr(fixtures, "set_fixture_root", fixture_roots.append)
    monkeypatch.setattr(
        verify_stats, "run_golden_set_verification", lambda **kwargs: reports
    )
    monkeypatch.setattr(verify_stats, "write_report", lambda *args, **kwargs: "REPORT")

    exit_code = verify_stats.main(
        ["--golden-set", "--fixture-root", str(tmp_path), "--verbose"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert fixture_roots == [str(tmp_path)]
    assert "REPORT" in captured.out
    assert "[Fixtures] Root set to:" in captured.err

    captured_run: dict[str, object] = {}

    def _fake_run_verification(**kwargs: object) -> list[PlayerReport]:
        captured_run.update(kwargs)
        return reports

    output_path = tmp_path / "verification.txt"
    monkeypatch.setattr(verify_stats, "run_verification", _fake_run_verification)
    monkeypatch.setattr(verify_stats, "write_report", write_report)

    exit_code = verify_stats.main(
        [
            "--player-ids",
            "12345",
            "--player-names",
            "Named Player",
            "--output-path",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured_run["player_ids"] == [12345]
    identities = captured_run["player_identities"]
    assert isinstance(identities, dict)
    assert identities[12345].name == "Named Player"
    assert output_path.exists()
    assert "Report written" in captured.err


def test_verify_filters_recording_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    player = {
        "player_type": "batter",
        "mlbam_id": 592450,
        "year": 2024,
        "name": "Aaron Judge",
    }
    monkeypatch.setattr(verify_filters, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(verify_filters, "SUMMARY_DIR", tmp_path / "summaries")

    fake_pb = SimpleNamespace(
        cache=SimpleNamespace(enable=lambda: None),
        statcast_batter=lambda start, end, mlbam_id: pd.DataFrame(
            [{"events": "single", "game_type": "R"}]
        ),
        batting_stats=lambda start, end, qual=0: pd.DataFrame(
            [{"Name": "Aaron Judge", "PA": 704, "wOBA": 0.455, "K%": 24.5, "BB%": 18.9}]
        ),
        get_splits=lambda bbref_id, year: pd.DataFrame(
            {"PA": [120, 584]},
            index=pd.MultiIndex.from_tuples(
                [("Platoon Splits", "vs LHP"), ("Platoon Splits", "vs RHP")]
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "pybaseball", fake_pb)

    verify_filters._record_raw_fixture(player)
    verify_filters._record_fangraphs_full_summary(player, fake_pb)
    verify_filters._record_baseball_ref_handed_split_summaries(player, fake_pb)

    raw_path = verify_filters._raw_fixture_path(player)
    full_summary_path = verify_filters._summary_fixture_path(
        "fangraphs", player, "full"
    )
    compat_path = verify_filters._summary_fixture_compat_path("fangraphs", player)
    vs_l_path = verify_filters._summary_fixture_path("baseball_ref", player, "vsL")
    vs_r_path = verify_filters._summary_fixture_path("baseball_ref", player, "vsR")

    assert raw_path.exists()
    assert full_summary_path.exists()
    assert compat_path.exists()
    assert json.loads(full_summary_path.read_text())["stats"]["PA"] == 704
    assert json.loads(vs_l_path.read_text())["stats"]["PA"] == 120
    assert json.loads(vs_r_path.read_text())["stats"]["PA"] == 584


def test_verify_filters_helper_logic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    df = pd.DataFrame([{"Name": "Aaron Judge"}, {"Name": "Juan Soto"}])
    assert verify_filters._match_player_row(df, "Aaron Judge")["Name"] == "Aaron Judge"
    assert verify_filters._match_player_row(df, "Soto")["Name"] == "Juan Soto"
    assert (
        verify_filters._match_player_row(pd.DataFrame([{"Other": "x"}]), "Aaron")
        is None
    )

    custom_player = {
        "player_type": "batter",
        "mlbam_id": 123,
        "year": 2024,
        "name": "Lookup Test",
    }
    fake_pb = SimpleNamespace(
        playerid_reverse_lookup=lambda ids, key_type: pd.DataFrame(
            {"key_bbref": ["lookupaa01"]}
        )
    )
    assert verify_filters._lookup_bbref_id(custom_player, fake_pb) == "lookupaa01"

    split_df = pd.DataFrame(
        {"PA": [33, "44"]},
        index=pd.MultiIndex.from_tuples(
            [("Other", "vs LHP"), ("Platoon Splits", "vs LHP")]
        ),
    )
    assert verify_filters._extract_split_pa(split_df, "vs LHP") == 44

    out_path = tmp_path / "nested" / "payload.json"
    verify_filters._write_json(out_path, {"ok": True})
    assert json.loads(out_path.read_text()) == {"ok": True}


def test_verify_filters_plugin_run_tests_and_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plugin = verify_filters.FilterValidationPlugin()
    synthetic_nodeid = (
        "tests/filter_verification/test_game_scope.py::TestSynthetic::test_case"
    )
    real_nodeid = (
        "tests/filter_verification/test_game_scope.py::"
        "TestGameScopeRealFixtures::test_matches_fangraphs[0]"
    )
    plugin.pytest_collection_modifyitems(
        None,
        None,
        [SimpleNamespace(nodeid=synthetic_nodeid), SimpleNamespace(nodeid=real_nodeid)],
    )
    plugin.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid=synthetic_nodeid,
            when="call",
            passed=True,
            failed=False,
            skipped=False,
            longrepr="",
        )
    )
    plugin.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid=real_nodeid,
            when="setup",
            passed=False,
            failed=False,
            skipped=True,
            longrepr=("", "", "Skipped: missing fixture"),
        )
    )
    summary = plugin.build_summary()
    assert summary["synthetic"]["passed"] == 1
    assert summary["real"]["skipped"] == 1
    assert summary["external"]["selected"] == 1
    assert verify_filters._gate_failures(summary)

    monkeypatch.setattr(pytest, "main", lambda args, plugins: 0)
    monkeypatch.setattr(
        verify_filters.FilterValidationPlugin,
        "build_summary",
        lambda self: {
            "synthetic": {"selected": 2, "passed": 2, "failed": 0, "skipped": 0},
            "real": {"selected": 2, "passed": 2, "failed": 0, "skipped": 0},
            "external": {"selected": 1, "passed": 1, "failed": 0, "skipped": 0},
            "real_skip_reasons": {},
        },
    )
    assert verify_filters.run_tests("game_scope") == 0

    monkeypatch.setattr(
        verify_filters.FilterValidationPlugin,
        "build_summary",
        lambda self: {
            "synthetic": {"selected": 2, "passed": 2, "failed": 0, "skipped": 0},
            "real": {"selected": 2, "passed": 0, "failed": 0, "skipped": 2},
            "external": {"selected": 1, "passed": 0, "failed": 0, "skipped": 1},
            "real_skip_reasons": {"missing fixture": 2},
        },
    )
    assert verify_filters.run_tests("all") == 2
    captured = capsys.readouterr()
    assert "REAL VALIDATION GATE: FAILED" in captured.out

    recorded: dict[str, int | None] = {}

    def _record(player_id: int | None = None, year: int | None = None) -> None:
        recorded["player_id"] = player_id
        recorded["year"] = year

    monkeypatch.setattr(verify_filters, "record_fixtures", _record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_filters",
            "--record-fixtures",
            "--player-id",
            "592450",
            "--year",
            "2024",
        ],
    )
    verify_filters.main()
    assert recorded == {"player_id": 592450, "year": 2024}

    monkeypatch.setattr(verify_filters, "run_tests", lambda filter_name: 3)
    monkeypatch.setattr(
        sys, "argv", ["verify_filters", "--offline", "--filter", "pitcher_hand"]
    )
    with pytest.raises(SystemExit) as exc_info:
        verify_filters.main()
    assert exc_info.value.code == 3
