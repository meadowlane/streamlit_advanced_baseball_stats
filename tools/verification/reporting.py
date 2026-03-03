"""Report generation for the stat verification harness.

Produces four output formats:
- ``text``  — coloured terminal table (default for CLI)
- ``csv``   — flat CSV file, one row per stat+player
- ``json``  — structured JSON for programmatic processing
- ``html``  — self-contained HTML with summary + per-player drilldowns
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from tools.verification.comparison import StatComparison
from tools.verification.sources.base import PlayerIdentity


# ---------------------------------------------------------------------------
# PlayerReport
# ---------------------------------------------------------------------------


@dataclass
class PlayerReport:
    """Collects all comparison results for one player-season."""

    player: PlayerIdentity
    year: int
    player_type: str
    comparisons: list[StatComparison] = field(default_factory=list)
    sample_notes: dict[str, int | float | None] = field(default_factory=dict)
    game_type: str = "regular"
    # e.g. {"PA": 650, "BIP": 480, "N_pitches": 2200}


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _verdict_emoji(verdict: str) -> str:
    return {
        "PASS": "✓",
        "FAIL": "✗",
        "WARN": "~",
        "SKIP": "-",
        "NON_VERIFIABLE": "?",
    }.get(verdict, "?")


def summary_rows(reports: list[PlayerReport]) -> list[dict[str, Any]]:
    """Return one dict per StatComparison across all reports.

    Suitable for CSV or JSON serialisation.
    """
    rows: list[dict[str, Any]] = []
    for rep in reports:
        for cmp in rep.comparisons:
            # Build per-source columns
            src_vals = {}
            abs_ds = {}
            for src, val in cmp.source_values.items():
                src_vals[f"src_{src}"] = val
            for src, val in cmp.abs_diffs.items():
                abs_ds[f"absdiff_{src}"] = val

            row: dict[str, Any] = {
                "player": rep.player.name,
                "mlbam_id": rep.player.mlbam_id,
                "year": rep.year,
                "player_type": rep.player_type,
                "stat": cmp.stat,
                "our_value": cmp.our_value,
                **src_vals,
                **abs_ds,
                "verdict": cmp.verdict,
                "note": cmp.note or "",
                "non_verifiable_reason": cmp.non_verifiable_reason or "",
            }
            # Sample context
            row.update({f"sample_{k}": v for k, v in rep.sample_notes.items()})
            rows.append(row)
    return rows


def top_discrepancies(
    reports: list[PlayerReport],
    n: int = 10,
    include_verdicts: frozenset[str] = frozenset(["FAIL", "WARN"]),
) -> list[dict[str, Any]]:
    """Return the *n* worst discrepancies sorted by maximum absolute diff."""
    rows = [
        r for r in summary_rows(reports)
        if r["verdict"] in include_verdicts
    ]
    # Sort by max abs diff across all sources
    def _max_abs(r: dict[str, Any]) -> float:
        diffs = [v for k, v in r.items() if k.startswith("absdiff_") and v is not None]
        return max(diffs) if diffs else 0.0

    return sorted(rows, key=_max_abs, reverse=True)[:n]


def verdict_counts(reports: list[PlayerReport]) -> dict[str, int]:
    counts: dict[str, int] = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0, "NON_VERIFIABLE": 0}
    for rep in reports:
        for cmp in rep.comparisons:
            counts[cmp.verdict] = counts.get(cmp.verdict, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def _text_report(reports: list[PlayerReport]) -> str:
    lines: list[str] = []
    counts = verdict_counts(reports)
    total = sum(counts.values())
    pass_pct = 100.0 * counts["PASS"] / total if total else 0.0

    # Derive game_type from reports (may vary per set)
    game_types = sorted({r.game_type for r in reports})
    game_type_str = ", ".join(game_types) if game_types else "regular"

    lines.append("=" * 70)
    lines.append("  BASEBALL STATS VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append(f"  Season scope: {game_type_str}")
    lines.append(
        f"  Players: {len(reports)}  |  "
        f"Total checks: {total}  |  "
        f"PASS: {counts['PASS']} ({pass_pct:.1f}%)  |  "
        f"FAIL: {counts['FAIL']}  |  "
        f"WARN: {counts['WARN']}  |  "
        f"SKIP: {counts['SKIP']}  |  "
        f"NON_VER: {counts['NON_VERIFIABLE']}"
    )
    lines.append(
        "  Note: PASS/FAIL based on independent sources only.  "
        "StatcastSource is shown as [info] — not counted toward verdicts."
    )
    lines.append("=" * 70)
    lines.append("")

    # Top discrepancies
    top = top_discrepancies(reports, n=10)
    if top:
        lines.append("TOP DISCREPANCIES")
        lines.append("-" * 70)
        lines.append(f"  {'Player':<20} {'Stat':<14} {'Ours':>8} {'MaxAbsDiff':>10} {'Verdict':<8}")
        lines.append(f"  {'-'*20} {'-'*14} {'-'*8} {'-'*10} {'-'*8}")
        for r in top:
            diffs = {k: v for k, v in r.items() if k.startswith("absdiff_") and v is not None}
            max_d = max(diffs.values()) if diffs else 0.0
            lines.append(
                f"  {r['player']:<20} {r['stat']:<14} {_fmt(r['our_value']):>8} "
                f"{max_d:>10.4f} {r['verdict']:<8}"
            )
        lines.append("")

    # Per-player drilldown
    for rep in reports:
        lines.append(f"{'─'*70}")
        lines.append(
            f"  {rep.player.name}  |  {rep.year}  |  {rep.player_type.upper()}  "
            f"|  scope={rep.game_type}"
        )
        sample_str = ", ".join(f"{k}={v}" for k, v in rep.sample_notes.items() if v is not None)
        if sample_str:
            lines.append(f"  Sample: {sample_str}")
        lines.append("")
        lines.append(f"  {'Stat':<14} {'Ours':>8} {'Sources':<44} {'V':>4}  Note")
        lines.append(f"  {'-'*14} {'-'*8} {'-'*44} {'-'*4}  {'-'*20}")
        for cmp in rep.comparisons:
            info_set = set(getattr(cmp, "info_only_sources", []))
            src_parts = []
            for k, v in cmp.source_values.items():
                if v is not None:
                    tag = "[i]" if k in info_set else ""
                    src_parts.append(f"{k}{tag}={_fmt(v)}")
            src_str = "  ".join(src_parts)
            note_str = (cmp.note or cmp.non_verifiable_reason or "")[:40]
            lines.append(
                f"  {cmp.stat:<14} {_fmt(cmp.our_value):>8} {src_str:<44} "
                f"{_verdict_emoji(cmp.verdict):>4}  {note_str}"
            )
        lines.append("")

    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------


def _csv_report(reports: list[PlayerReport]) -> str:
    rows = summary_rows(reports)
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def _json_report(reports: list[PlayerReport]) -> str:
    data = {
        "summary": verdict_counts(reports),
        "top_discrepancies": top_discrepancies(reports),
        "players": [
            {
                "name": rep.player.name,
                "mlbam_id": rep.player.mlbam_id,
                "year": rep.year,
                "player_type": rep.player_type,
                "sample_notes": rep.sample_notes,
                "comparisons": [
                    {
                        "stat": cmp.stat,
                        "our_value": cmp.our_value,
                        "source_values": cmp.source_values,
                        "abs_diffs": cmp.abs_diffs,
                        "verdict": cmp.verdict,
                        "note": cmp.note,
                        "non_verifiable_reason": cmp.non_verifiable_reason,
                        "fallback": cmp.fallback,
                    }
                    for cmp in rep.comparisons
                ],
            }
            for rep in reports
        ],
    }
    return json.dumps(data, indent=2, default=lambda o: None)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_VERDICT_STYLE = {
    "PASS": "background:#d4edda;color:#155724",
    "FAIL": "background:#f8d7da;color:#721c24",
    "WARN": "background:#fff3cd;color:#856404",
    "SKIP": "background:#e2e3e5;color:#383d41",
    "NON_VERIFIABLE": "background:#d1ecf1;color:#0c5460",
}


def _html_report(reports: list[PlayerReport]) -> str:
    counts = verdict_counts(reports)
    total = sum(counts.values())
    pass_pct = 100.0 * counts["PASS"] / total if total else 0.0
    top = top_discrepancies(reports, n=10)

    rows_html = ""
    for r in top:
        diffs = {k: v for k, v in r.items() if k.startswith("absdiff_") and v is not None}
        max_d = max(diffs.values()) if diffs else 0.0
        style = _VERDICT_STYLE.get(r["verdict"], "")
        rows_html += (
            f"<tr style='{style}'>"
            f"<td>{r['player']}</td><td>{r['stat']}</td>"
            f"<td>{_fmt(r['our_value'])}</td><td>{max_d:.4f}</td>"
            f"<td>{r['verdict']}</td>"
            f"<td>{(r.get('note') or '')[:60]}</td>"
            f"</tr>"
        )

    player_sections = ""
    for rep in reports:
        cmp_rows = ""
        for cmp in rep.comparisons:
            style = _VERDICT_STYLE.get(cmp.verdict, "")
            src_cells = " | ".join(
                f"{k}: {_fmt(v)}" for k, v in cmp.source_values.items() if v is not None
            )
            note = (cmp.note or cmp.non_verifiable_reason or "")[:80]
            cmp_rows += (
                f"<tr style='{style}'>"
                f"<td>{cmp.stat}</td>"
                f"<td>{_fmt(cmp.our_value)}</td>"
                f"<td style='font-size:0.85em'>{src_cells}</td>"
                f"<td>{cmp.verdict}</td>"
                f"<td style='font-size:0.85em'>{note}</td>"
                f"</tr>"
            )
        sample_str = ", ".join(
            f"{k}={v}" for k, v in rep.sample_notes.items() if v is not None
        )
        player_sections += f"""
        <h3>{rep.player.name} — {rep.year} ({rep.player_type})</h3>
        <p style='color:#555'>{sample_str}</p>
        <table border='1' cellpadding='4' cellspacing='0'
               style='border-collapse:collapse;width:100%;font-family:monospace'>
          <tr style='background:#343a40;color:white'>
            <th>Stat</th><th>Our Value</th><th>Source Values</th>
            <th>Verdict</th><th>Note</th>
          </tr>
          {cmp_rows}
        </table><br/>"""

    return f"""<!DOCTYPE html>
<html lang='en'>
<head><meta charset='UTF-8'>
<title>Baseball Stats Verification Report</title>
<style>
  body {{font-family: sans-serif; margin: 2em; color: #212529;}}
  h1, h2, h3 {{color: #343a40;}}
  .badge {{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:bold;margin:2px}}
  .badge-pass {{background:#d4edda;color:#155724}}
  .badge-fail {{background:#f8d7da;color:#721c24}}
  .badge-warn {{background:#fff3cd;color:#856404}}
  .badge-skip {{background:#e2e3e5;color:#383d41}}
  .badge-nv   {{background:#d1ecf1;color:#0c5460}}
</style>
</head>
<body>
  <h1>Baseball Stats Verification Report</h1>
  <p>
    <span class='badge badge-pass'>PASS {counts['PASS']}</span>
    <span class='badge badge-fail'>FAIL {counts['FAIL']}</span>
    <span class='badge badge-warn'>WARN {counts['WARN']}</span>
    <span class='badge badge-skip'>SKIP {counts['SKIP']}</span>
    <span class='badge badge-nv'>NON_VER {counts['NON_VERIFIABLE']}</span>
    &nbsp; Pass rate: <strong>{pass_pct:.1f}%</strong> of {total} checks
  </p>

  <h2>Top Discrepancies</h2>
  <table border='1' cellpadding='4' cellspacing='0'
         style='border-collapse:collapse;width:100%;font-family:monospace'>
    <tr style='background:#343a40;color:white'>
      <th>Player</th><th>Stat</th><th>Our Value</th>
      <th>Max AbsDiff</th><th>Verdict</th><th>Note</th>
    </tr>
    {rows_html}
  </table><br/>

  <h2>Per-Player Details</h2>
  {player_sections}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def write_report(
    reports: list[PlayerReport],
    output_format: Literal["text", "csv", "json", "html"] = "text",
    output_path: str | Path | None = None,
) -> str:
    """Generate a report and optionally write it to a file.

    Returns the rendered report string.
    """
    if output_format == "text":
        content = _text_report(reports)
    elif output_format == "csv":
        content = _csv_report(reports)
    elif output_format == "json":
        content = _json_report(reports)
    elif output_format == "html":
        content = _html_report(reports)
    else:
        raise ValueError(f"Unknown output format: {output_format!r}")

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(content, encoding="utf-8")

    return content
