"""Gemeinsame Diagrammbausteine: aktuelles Jahr vor dem Band 1991–2025, Histogramm mit Markierung, Trendlinie."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go

GRAU = "#9aa0a6"
BLAU = "#1f6fd1"
GRUEN = "#2e9d5b"
ROT = "#d9534f"
ORANGE = "#e8892b"


def tag_zu_datum(jahr: int, tag: int) -> dt.date:
    return dt.date(2001, 1, 1) + dt.timedelta(days=int(tag)) if jahr is None else \
        dt.date(jahr, 1, 1) + dt.timedelta(days=int(tag) + (1 if (jahr % 4 == 0 and tag >= 59) else 0))


def band_fig(matrix: pd.DataFrame, aktuell: pd.Series | np.ndarray | None, jahr: int, titel: str, yachse: str,
             x0_tag: int = 0, hlines: list[tuple[float, str]] | None = None, height: int = 360,
             hist_jahre: list[int] | None = None) -> go.Figure:
    """matrix: Jahre × Tage; aktuell: Werte des laufenden Jahres ab Tag x0_tag."""
    m = matrix.loc[[y for y in (hist_jahre or matrix.index) if y in matrix.index and y != jahr]]
    cols = list(m.columns)
    xs = [tag_zu_datum(jahr, x0_tag + c) for c in cols]
    q = np.nanpercentile(m.to_numpy(float), [10, 25, 50, 75, 90], axis=0)
    f = go.Figure()
    f.add_scatter(x=xs + xs[::-1], y=list(q[4]) + list(q[0])[::-1], fill="toself", fillcolor="rgba(150,150,150,0.20)",
                  line=dict(width=0), name="10–90 % Historie")
    f.add_scatter(x=xs + xs[::-1], y=list(q[3]) + list(q[1])[::-1], fill="toself", fillcolor="rgba(150,150,150,0.30)",
                  line=dict(width=0), name="25–75 % Historie")
    f.add_scatter(x=xs, y=q[2], line=dict(color=GRAU, dash="dot"), name="Median Historie")
    if aktuell is not None:
        a = np.asarray(aktuell, float)
        f.add_scatter(x=xs[: len(a)], y=a, line=dict(color=ROT, width=2.5), name=str(jahr))
    for y, label in hlines or []:
        f.add_hline(y=y, line_dash="dash", line_color="black", annotation_text=label)
    f.update_layout(title=titel, height=height, margin=dict(t=50, b=10), yaxis_title=yachse,
                    legend=dict(orientation="h", y=-0.15), xaxis=dict(tickformat="%d.%m."))
    return f


def hist_fig(werte: pd.Series, aktuell: float | None, jahr: int | None, xachse: str, height: int = 280) -> go.Figure:
    f = go.Figure()
    f.add_histogram(x=werte.dropna(), marker_color=GRAU, opacity=0.7, name="Jahre", nbinsx=15)
    if aktuell is not None and not np.isnan(aktuell):
        f.add_vline(x=aktuell, line_color=ROT, line_width=3, annotation_text=str(jahr))
    f.update_layout(height=height, margin=dict(t=20, b=10), xaxis_title=xachse, yaxis_title="Anzahl Jahre",
                    showlegend=False)
    return f


def trend_fig(serie: pd.Series, tr: dict, titel: str, einheit: str, markiere: int | None = None,
              height: int = 320) -> go.Figure:
    s = serie.dropna()
    farben = [ROT if (markiere is not None and y == markiere) else BLAU for y in s.index]
    f = go.Figure()
    f.add_bar(x=s.index, y=s.values, marker_color=farben, opacity=0.75, name=titel)
    if "slope" in tr:
        x = np.array([s.index.min(), s.index.max()], float)
        f.add_scatter(x=x, y=tr["intercept"] + tr["slope"] * x, line=dict(color="black", width=2),
                      name=f"Theil-Sen {tr['steigung_jahrzehnt']:+.1f} {einheit}/10 J.")
    rng = (s.min(), s.max())
    pad = (rng[1] - rng[0]) * 0.15 or 1
    f.update_layout(title=titel, height=height, margin=dict(t=50, b=10), yaxis_title=einheit,
                    yaxis=dict(range=[rng[0] - pad, rng[1] + pad]), legend=dict(orientation="h", y=-0.2))
    return f


def ampel_farbe(p: float) -> str:
    if p is None or np.isnan(p):
        return GRAU
    return GRUEN if p < 50 else (ORANGE if p < 80 else ROT)
