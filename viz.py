"""
viz.py
Plotly figures for the Market Strategy Copilot dashboard, built from the real
aggregates in data/metrics/*.json. Colors follow a validated, CVD-aware palette;
identity is never color-alone (direct value labels + legends).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go

METRICS_DIR = Path(__file__).resolve().parent / "data" / "metrics"

# Palette (validated default, light mode)
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
# Polarity encoding for sentiment (two hues + neutral gray midpoint)
SENTIMENT_COLORS = {"negative": "#e34948", "neutral": "#8d99ae", "positive": "#1baf7a"}
EMOTION_ORDER = ["neutral", "joy", "surprise", "anger", "fear", "sadness", "disgust"]

_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=INK, size=13),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def load_metrics(name: str) -> Dict[str, Any]:
    return json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def campaign_sentiment_fig(data: Dict[str, Any]) -> go.Figure:
    """Grouped bar: share of sentiment per brand (normalised, since counts differ)."""
    brands = list(data["sentiment"].keys())
    fig = go.Figure()
    for sent in ("negative", "neutral", "positive"):
        shares = []
        for b in brands:
            counts = data["sentiment"][b]
            total = sum(counts.values()) or 1
            shares.append(round(100 * counts.get(sent, 0) / total))
        fig.add_bar(
            name=sent.title(),
            x=brands,
            y=shares,
            marker_color=SENTIMENT_COLORS[sent],
            marker_line=dict(color=SURFACE, width=2),
            text=[f"{s}%" for s in shares],
            textposition="outside",
            textfont=dict(color=MUTED, size=11),
        )
    fig.update_layout(barmode="group", yaxis_title="share of coverage", **_LAYOUT)
    fig.update_yaxes(showgrid=True, gridcolor="#ececec", range=[0, 100], ticksuffix="%")
    fig.update_xaxes(showgrid=False)
    return fig


def campaign_emotion_fig(data: Dict[str, Any]) -> go.Figure:
    """Heatmap: emotion counts per brand (sequential single hue = magnitude)."""
    brands = list(data["emotion"].keys())
    z = [[data["emotion"][b].get(e, 0) for e in EMOTION_ORDER] for b in brands]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[e.title() for e in EMOTION_ORDER],
            y=brands,
            colorscale=[[0, "#eef4fb"], [1, BLUE]],
            text=z,
            texttemplate="%{text}",
            textfont=dict(size=11, color=INK),
            colorbar=dict(title="items", thickness=10),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(**_LAYOUT)
    return fig


def churn_metrics_fig(data: Dict[str, Any]) -> go.Figure:
    """Single-series bar of model metrics; weak ones (recall, AUC) flagged."""
    m = data["metrics"]
    names = list(m.keys())
    vals = [m[k] for k in names]
    weak = {"Recall", "AUC"}
    colors = [("#e34948" if k in weak else BLUE) for k in names]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=vals,
            marker_color=colors,
            marker_line=dict(color=SURFACE, width=2),
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont=dict(color=MUTED, size=12),
        )
    )
    fig.update_layout(showlegend=False, yaxis_title="score", **_LAYOUT)
    fig.update_yaxes(showgrid=True, gridcolor="#ececec", range=[0, 1])
    fig.update_xaxes(showgrid=False)
    return fig
