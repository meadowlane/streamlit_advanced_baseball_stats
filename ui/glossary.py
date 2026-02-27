"""Glossary definitions and percentile explainer for mlb-splits.

All content lives in module-level constants so it can be tested independently
of a Streamlit runtime. The render_glossary() function wires it into the UI.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Stat definitions
# ---------------------------------------------------------------------------

STAT_DEFINITIONS: dict[str, dict[str, str | None]] = {
    "wOBA": {
        "full_name": "Weighted On-Base Average",
        "definition": (
            "A single-number offensive metric that weights each type of plate appearance "
            "outcome by how much it actually contributes to scoring runs. A walk is worth "
            "less than a single, which is worth less than a home run — wOBA reflects "
            "those differences. League average is typically around .315–.325."
        ),
        "context": "Excellent: .400+ · Above avg: .360+ · Average: .320 · Below avg: .300−",
        "direction": "Higher is better",
        "direction_pitcher": "Lower is better",
        "denominator": "Plate appearances (batters) / batters faced outcomes (pitchers).",
    },
    "xwOBA": {
        "full_name": "Expected Weighted On-Base Average",
        "definition": (
            "The wOBA a batter *deserved* based solely on exit velocity and launch angle "
            "of each batted ball, plus actual strikeout and walk rates. Because it strips "
            "out batted-ball luck and defense, xwOBA is a better predictor of future "
            "performance than wOBA. A large gap between wOBA and xwOBA suggests the "
            "batter has been unusually lucky (wOBA > xwOBA) or unlucky (xwOBA > wOBA)."
        ),
        "context": "Same scale as wOBA. Sourced from Baseball Savant (Statcast).",
        "direction": "Higher is better",
        "direction_pitcher": "Lower is better",
        "denominator": "Plate appearances (batters) / batters faced outcomes (pitchers).",
    },
    "K%": {
        "full_name": "Strikeout Rate",
        "definition": (
            "The percentage of plate appearances that end in a strikeout. High strikeout "
            "rates reduce a batter's ability to put the ball in play and create offense. "
            "League average hovers around 22–24%. Elite contact hitters often post K% "
            "below 12%; free-swingers can exceed 35%."
        ),
        "context": "Excellent: <12% · Above avg: <18% · Average: ~23% · Below avg: >28%",
        "direction": "Lower is better",
        "direction_pitcher": "Higher is better",
        "denominator": "Plate appearances (batters) / batters faced (pitchers).",
    },
    "BB%": {
        "full_name": "Walk Rate",
        "definition": (
            "The percentage of plate appearances that end in a walk (base on balls). "
            "Walking shows plate discipline and drives up pitch counts. League average "
            "is roughly 8–9%. Elite on-base threats routinely exceed 13–15%."
        ),
        "context": "Excellent: >14% · Above avg: >10% · Average: ~8.5% · Below avg: <6%",
        "direction": "Higher is better",
        "direction_pitcher": "Lower is better",
        "denominator": "Plate appearances (batters) / batters faced (pitchers).",
    },
    "K-BB%": {
        "full_name": "Strikeout Minus Walk Rate",
        "definition": (
            "A quick pitcher command/dominance indicator computed as strikeout rate "
            "minus walk rate (K% - BB%). It captures how often a pitcher gets outs "
            "via strikeout while avoiding free passes."
        ),
        "context": "Higher is better for pitchers; strong starters are often in low-to-mid teens or better.",
        "direction": "Higher is better",
        "direction_pitcher": None,
        "denominator": "Batters faced (derived from K% and BB% over the same sample).",
    },
    "GB%": {
        "full_name": "Ground-Ball Rate",
        "definition": (
            "The share of tracked balls in play that are ground balls. For pitchers, "
            "more ground balls generally suppress damage; for batters, this is often "
            "more neutral/context-dependent because too many grounders can cap power."
        ),
        "context": "Typical MLB pitcher range is roughly 40–48%; 50%+ is strong for pitchers.",
        "direction": "Context-dependent",
        "direction_pitcher": "Higher is better",
        "denominator": "Tracked balls in play (batted-ball events).",
    },
    "CSW%": {
        "full_name": "Called Strikes Plus Whiffs",
        "definition": (
            "The percentage of all pitches that end as a called strike or a swing-and-miss. "
            "Denominator: total pitches in the selected sample."
        ),
        "context": "Around 28–30% is solid; low-30s is often excellent.",
        "direction": "Higher is better",
        "direction_pitcher": None,
        "denominator": "Total pitches.",
    },
    "Whiff%": {
        "full_name": "Whiff Rate",
        "definition": (
            "How often a pitcher gets a swing-and-miss when hitters swing. "
            "Denominator: swings (not all pitches)."
        ),
        "context": "About 25–30% is good; 35%+ is typically elite.",
        "direction": "Higher is better",
        "direction_pitcher": None,
        "denominator": "Swings.",
    },
    "FirstStrike%": {
        "full_name": "First-Pitch Strike Rate",
        "definition": (
            "The share of 0-0 pitches that are strikes by called strike/whiff classification. "
            "Denominator: pitches thrown in 0 balls, 0 strikes counts."
        ),
        "context": "League average is often near 60%; mid-60s is strong.",
        "direction": "Higher is better",
        "direction_pitcher": None,
        "denominator": "0-0 first pitches.",
    },
    "HardHit%": {
        "full_name": "Hard Hit Rate",
        "definition": (
            "The percentage of batted balls hit with an exit velocity of 95 mph or "
            "higher (Statcast threshold). Hard contact correlates strongly with run "
            "production and is largely within the batter's control, unlike BABIP. "
            "League average is approximately 37–40%."
        ),
        "context": "Excellent: >50% · Above avg: >43% · Average: ~38% · Below avg: <32%",
        "direction": "Higher is better",
        "direction_pitcher": "Lower is better",
        "denominator": "Batted-ball events.",
    },
    "Barrel%": {
        "full_name": "Barrel Rate",
        "definition": (
            "The percentage of batted balls classified as 'barrels' — the optimal "
            "combination of exit velocity and launch angle that historically produces "
            "a batting average above .500 and slugging above 1.500. Barrel% is the "
            "best single indicator of raw power output. League average is around 7–8%."
        ),
        "context": "Excellent: >15% · Above avg: >10% · Average: ~7.5% · Below avg: <5%",
        "direction": "Higher is better",
        "direction_pitcher": "Lower is better",
        "denominator": "Batted-ball events.",
    },
    "Velo": {
        "full_name": "Velocity",
        "definition": (
            "Pitch speed in miles per hour. In arsenal views this is typically average "
            "velocity for each pitch type over the selected sample."
        ),
        "context": "Higher velocity can increase margin for error, but effectiveness depends on shape/command.",
        "direction": "Context-dependent",
        "direction_pitcher": None,
        "denominator": "Individual pitches (averaged over the selected sample).",
    },
    "Spin": {
        "full_name": "Spin Rate",
        "definition": (
            "Rate of pitch rotation (RPM). Spin influences movement profile and pitch shape; "
            "its value depends on pitch type and movement efficiency."
        ),
        "context": "Not universally higher-is-better; pitch-type context matters.",
        "direction": "Context-dependent",
        "direction_pitcher": None,
        "denominator": "Individual pitches (averaged over the selected sample).",
    },
    "Usage%": {
        "full_name": "Pitch Usage Rate",
        "definition": (
            "How often a pitch type is thrown relative to all pitches in the selected sample."
        ),
        "context": "Descriptive mix stat, not a direct quality metric by itself.",
        "direction": "Context-dependent",
        "direction_pitcher": None,
        "denominator": "Total pitches.",
    },
    "PA": {
        "full_name": "Plate Appearances / Batters Faced",
        "definition": (
            "Counting stat for opportunities in the sample. For batters this is plate "
            "appearances; for pitchers this reflects batters faced."
        ),
        "context": "Higher values mean larger samples, not automatically better performance.",
        "direction": "Context-dependent",
        "direction_pitcher": None,
        "denominator": "Not a rate stat (raw count).",
    },
}

# ---------------------------------------------------------------------------
# Percentile explainer content
# ---------------------------------------------------------------------------

PERCENTILE_EXPLAINER = """
**What does a percentile mean here?**

Each percentile badge shows how a player's stat compares to MLB players in the
same role (batters vs pitchers) from the selected season leaderboard.

- **90th percentile** → better than 90% of qualifiers
- **50th percentile** → exactly league average
- **10th percentile** → below 90% of qualifiers

Direction depends on role:

- **Batters:** K% is inverted (lower is better).
- **Pitchers:** wOBA, xwOBA, BB%, HardHit%, and Barrel% are inverted (lower is better),
  while **K% is higher-is-better**.

**Percentile benchmarks are based on season-level league distributions.**
When viewing splits (vs LHP/RHP, Home/Away, Monthly), percentiles show
where that split value falls relative to the *overall* season distribution
— a useful reference even though true split distributions aren't available.
"""

# ---------------------------------------------------------------------------
# Render function
# ---------------------------------------------------------------------------


def _direction_icon(direction: str) -> str:
    if direction.startswith("Higher"):
        return "↑"
    if direction.startswith("Lower"):
        return "↓"
    return "↔"


def _compact_denominator_text(denominator: str) -> str:
    text = denominator.strip()
    if "Plate appearances (batters)" in text and "batters faced" in text:
        return "Denom: PA (hitters) / BF (pitchers)"
    if "Batters faced" in text:
        return "Denom: BF"
    if "Tracked balls in play" in text:
        return "Denom: balls in play"
    if "Batted-ball events" in text:
        return "Denom: batted-ball events"
    if "Total pitches" in text:
        return "Denom: total pitches"
    if "Swings" in text:
        return "Denom: swings"
    if "0-0 first pitches" in text:
        return "Denom: 0-0 first pitches"
    if "Not a rate stat" in text:
        return "Denom: count stat"
    if "Individual pitches" in text:
        return "Denom: pitches (avg by pitch type)"
    return f"Denom: {text}"


def _compact_direction_text(
    batter_direction: str,
    pitcher_direction: str | None,
    player_type: str | None = None,
) -> str:
    role = (player_type or "").strip().lower()
    batter_icon = _direction_icon(batter_direction)
    if pitcher_direction and pitcher_direction != batter_direction:
        pitcher_icon = _direction_icon(pitcher_direction)
        return f"{batter_icon} hitter / {pitcher_icon} pitcher"
    if role == "pitcher" and pitcher_direction:
        return f"{_direction_icon(pitcher_direction)} pitcher"
    if role == "batter":
        return f"{batter_icon} hitter"
    return f"{batter_icon} both roles"


def _compact_context_text(context: str) -> str:
    text = context.strip()
    if "Average:" in text:
        avg_value = text.split("Average:", 1)[1].split("·", 1)[0].strip()
        return f"Avg {avg_value}"
    if "League average" in text:
        return text.split(".", 1)[0].strip()
    return text.split("·", 1)[0].strip()


def _compact_meaning_text(definition: str, max_len: int = 96) -> str:
    text = " ".join(definition.strip().split())
    if not text:
        return ""
    sentence = text.split(". ", 1)[0].strip()
    if not sentence.endswith("."):
        sentence += "."
    if len(sentence) <= max_len:
        return sentence
    return sentence[: max_len - 1].rstrip() + "…"


def _compact_better_text(
    batter_direction: str,
    pitcher_direction: str | None,
) -> str:
    batter_icon = _direction_icon(batter_direction)
    if batter_direction == "Context-dependent":
        return "info"
    if pitcher_direction and pitcher_direction != batter_direction:
        pitcher_icon = _direction_icon(pitcher_direction)
        return f"{batter_icon} hitter / {pitcher_icon} pitcher"
    if pitcher_direction and "pitcher" in pitcher_direction.lower():
        return f"{_direction_icon(pitcher_direction)} pitcher"
    return f"{batter_icon} both"


def render_glossary(mode: Literal["full", "compact"] = "full", player_type: str | None = None) -> None:
    """Render the full glossary and percentile explainer inside a Streamlit expander."""
    if mode == "compact":
        st.caption("Color scale: 90-100 Elite | 70-89 Above avg | 50-69 Avg | 30-49 Below avg | 0-29 Well below")
        rows = []
        for stat, info in STAT_DEFINITIONS.items():
            batter_direction = str(info["direction"])
            pitcher_direction = str(info["direction_pitcher"]) if info["direction_pitcher"] else None
            meaning = _compact_meaning_text(str(info["definition"]))
            context = _compact_context_text(str(info["context"]))
            if context:
                meaning = f"{meaning} {context}."
            rows.append(
                {
                    "Stat": stat,
                    "Meaning": meaning,
                    "Denom": _compact_denominator_text(str(info["denominator"])),
                    "Better": _compact_better_text(batter_direction, pitcher_direction),
                }
            )

        compact_df = pd.DataFrame(rows, columns=["Stat", "Meaning", "Denom", "Better"])
        st.dataframe(
            compact_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Stat": st.column_config.TextColumn("Stat", width="small"),
                "Meaning": st.column_config.TextColumn("Meaning", width="large"),
                "Denom": st.column_config.TextColumn("Denom", width="small"),
                "Better": st.column_config.TextColumn("Better", width="small"),
            },
        )
        return

    with st.expander("Glossary & How to Read Percentiles", expanded=False):
        # ---- Color scale key ----
        st.markdown("#### Percentile Color Scale")
        st.markdown(
            "90–100 Elite | 70–89 Above avg | 50–69 Avg | 30–49 Below avg | 0–29 Well below"
        )
        st.markdown("")

        # ---- Percentile explainer ----
        st.markdown(PERCENTILE_EXPLAINER)
        st.divider()

        # ---- Stat definitions ----
        st.markdown("#### Stat Definitions")
        for stat, info in STAT_DEFINITIONS.items():
            direction_value = str(info.get("direction", "Context-dependent"))
            direction_icon = _direction_icon(direction_value)
            st.markdown(
                f"**{stat} — {info['full_name']}** &nbsp;"
                f'<span style="font-size:11px;color:#888;">'
                f"{direction_icon} {direction_value}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(info["definition"])
            st.caption(f"Context: {info['context']}")
            st.caption(f"Denominator: {info['denominator']}")
            if info.get("direction_pitcher"):
                st.caption(f"Pitcher direction: {info['direction_pitcher']}")
            st.markdown("")
