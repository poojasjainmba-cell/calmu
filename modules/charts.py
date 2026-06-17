from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CALMU_COLORS = {
    "blue": "#2938D5",
    "lime": "#EDFF81",
    "navy": "#1E2944",
    "royal": "#3D59D9",
    "white": "#FFFFFF",
    "sky": "#ABCCE3",
    "red": "#CD1141",
    "burgundy": "#8F0028",
    "green": "#1A5347",
    "sage": "#ABBEB3",
    "slate": "#657874",
    "mist": "#C4CDD3",
}

CHART_SEQUENCE = [
    CALMU_COLORS["blue"],
    CALMU_COLORS["green"],
    CALMU_COLORS["royal"],
    CALMU_COLORS["red"],
    CALMU_COLORS["sky"],
    CALMU_COLORS["sage"],
    CALMU_COLORS["burgundy"],
    CALMU_COLORS["slate"],
    CALMU_COLORS["lime"],
]


def configure_plotly_theme() -> None:
    px.defaults.color_discrete_sequence = CHART_SEQUENCE
    px.defaults.color_continuous_scale = [CALMU_COLORS["sky"], CALMU_COLORS["royal"], CALMU_COLORS["navy"]]


def empty_chart() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=320,
        annotations=[
            {
                "text": "No confirmed data available",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "font": {"color": CALMU_COLORS["slate"], "size": 16},
            }
        ],
    )
    return fig


def enrollment_progress(actual: float, goal: float) -> go.Figure:
    goal_value = max(float(goal or 0), 1)
    actual_value = float(actual or 0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=actual_value,
            number={"font": {"color": CALMU_COLORS["navy"]}},
            gauge={
                "axis": {"range": [0, goal_value]},
                "bar": {"color": CALMU_COLORS["blue"]},
                "bgcolor": "#F3F7FA",
                "bordercolor": CALMU_COLORS["mist"],
                "steps": [{"range": [0, goal_value], "color": "#EEF4F8"}],
                "threshold": {"line": {"color": CALMU_COLORS["green"], "width": 4}, "value": goal_value},
            },
        )
    )
    fig.update_layout(template="plotly_white", height=300, margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def funnel_chart(funnel: pd.DataFrame) -> go.Figure:
    if funnel.empty:
        return empty_chart()
    fig = go.Figure(go.Funnel(y=funnel["stage"], x=funnel["count"], marker={"color": CHART_SEQUENCE[: len(funnel)]}))
    fig.update_layout(template="plotly_white", height=360, margin={"l": 24, "r": 24, "t": 24, "b": 24})
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str = "", color: str | None = None, horizontal: bool = False) -> go.Figure:
    if df.empty or x not in df.columns or y not in df.columns:
        return empty_chart()
    work = df.copy()
    work[y] = pd.to_numeric(work[y], errors="coerce").fillna(0)
    if horizontal:
        fig = px.bar(work.sort_values(y).tail(15), x=y, y=x, orientation="h", color=color, title=title)
    else:
        fig = px.bar(work.sort_values(y, ascending=False).head(15), x=x, y=y, color=color, title=title)
    fig.update_layout(template="plotly_white", height=380, title_font_color=CALMU_COLORS["navy"], margin={"l": 24, "r": 24, "t": 54, "b": 80})
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    if df.empty or x not in df.columns or y not in df.columns:
        return empty_chart()
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_traces(line={"color": CALMU_COLORS["blue"], "width": 3})
    fig.update_layout(template="plotly_white", height=340, title_font_color=CALMU_COLORS["navy"], margin={"l": 24, "r": 24, "t": 54, "b": 44})
    return fig


def donut_chart(df: pd.DataFrame, names: str, values: str, title: str = "") -> go.Figure:
    if df.empty or names not in df.columns or values not in df.columns:
        return empty_chart()
    fig = px.pie(df, names=names, values=values, hole=0.48, title=title)
    fig.update_layout(template="plotly_white", height=360, title_font_color=CALMU_COLORS["navy"], margin={"l": 24, "r": 24, "t": 54, "b": 24})
    return fig

