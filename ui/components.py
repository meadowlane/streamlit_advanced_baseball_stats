"""Compatibility facade for reusable UI components."""

from .pitch_charts import (
    _pitch_zone_hand_conflict_note,
    render_pitch_movement_chart,
    render_pitch_zone_chart,
)
from .stat_panels import (
    _ARSENAL_TABLE_HELP,
    _ORDERED_STATS,
    _SPLIT_TABLE_FORMAT,
    _SPLIT_TABLE_HELP,
    build_chart_df,
    format_percentile,
    format_stat_value,
    percentile_bar_chart,
    player_header,
    render_pitch_arsenal,
    split_table,
    stat_card,
    stat_cards_row,
)
from .trend_charts import (
    _add_trend_traces,
    _build_single_stat_chart,
    _build_trend_tidy_df,
    _filter_real_data_rows,
    _stat_formatter,
    _stats_share_scale,
    render_trend_custom,
    render_trend_dashboard,
    render_trend_section,
)
