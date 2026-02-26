"""Glossary definitions and percentile explainer for mlb-splits.

All content lives in module-level constants so it can be tested independently
of a Streamlit runtime. The render_glossary() function wires it into the UI.
"""

from __future__ import annotations

import streamlit as st

from stats.percentiles import COLOR_TIERS

# ---------------------------------------------------------------------------
# Stat definitions
# ---------------------------------------------------------------------------

STAT_DEFINITIONS: dict[str, dict[str, str]] = {
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
    },
}

# ---------------------------------------------------------------------------
# Percentile explainer content
# ---------------------------------------------------------------------------

PERCENTILE_EXPLAINER = """
**What does a percentile mean here?**

Each percentile badge shows how a player's stat compares to all MLB batters
who logged at least 50 plate appearances in the selected season.

- **90th percentile** → better than 90% of qualifiers
- **50th percentile** → exactly league average
- **10th percentile** → below 90% of qualifiers

**For K% (Strikeout Rate)**, the scale is *inverted* — a low K% is good,
so a player with a 10% K% in a league where the average is 23% will rank
in the **90th percentile**, not the 10th.

**Percentile benchmarks are based on season-level league distributions.**
When viewing splits (vs LHP/RHP, Home/Away, Monthly), percentiles show
where that split value falls relative to the *overall* season distribution
— a useful reference even though true split distributions aren't available.
"""

# ---------------------------------------------------------------------------
# Render function
# ---------------------------------------------------------------------------


def render_glossary() -> None:
    """Render the full glossary and percentile explainer inside a Streamlit expander."""
    with st.expander("Glossary & How to Read Percentiles", expanded=False):
        # ---- Color scale key ----
        st.markdown("#### Percentile Color Scale")
        tier_labels = [
            (name.capitalize(), hex_color, label)
            for (_, name, hex_color), label in zip(
                COLOR_TIERS,
                [
                    "90th–100th percentile — Elite",
                    "70th–89th percentile — Above average",
                    "50th–69th percentile — Average",
                    "30th–49th percentile — Below average",
                    "0th–29th percentile — Well below average",
                ],
            )
        ]

        badge_html = " &nbsp; ".join(
            f'<span style="background:{hex_};color:#fff;padding:3px 12px;'
            f'border-radius:99px;font-size:12px;font-weight:700;">'
            f'{name}: {label}</span>'
            for name, hex_, label in tier_labels
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("")

        # ---- Percentile explainer ----
        st.markdown(PERCENTILE_EXPLAINER)
        st.divider()

        # ---- Stat definitions ----
        st.markdown("#### Stat Definitions")
        for stat, info in STAT_DEFINITIONS.items():
            direction_icon = "↑" if info["direction"] == "Higher is better" else "↓"
            st.markdown(
                f"**{stat} — {info['full_name']}** &nbsp;"
                f'<span style="font-size:11px;color:#888;">'
                f"{direction_icon} {info['direction']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(info["definition"])
            st.caption(f"Context: {info['context']}")
            st.markdown("")
