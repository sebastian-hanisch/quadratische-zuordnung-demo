"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import qmh_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def log_axis(fig, values, title: str, axis: str = "y", row=None, col=None):
    v = np.asarray([x for x in values if x is not None and x > 0 and math.isfinite(x)], float)
    lo, hi = (math.floor(math.log10(v.min())) - 0.2, math.ceil(math.log10(v.max())) + 0.1) if len(v) else (-1.0, 1.0)
    update = fig.update_yaxes if axis == "y" else fig.update_xaxes
    kw = dict(title_text=title, type="log", dtick=1, exponentformat="power", showexponent="all", range=[lo, max(hi, lo + 1.0)])
    if row is not None:
        kw.update(row=row, col=col)
    update(**kw)
    return fig


# ----------------------------------------------------------------------------- Experiment 1: die Halle

def build_board(inst, p, title: str) -> go.Figure:
    """Plätze als graue Quadrate, Bauteile als Punkte mit Namen auf ihrem Platz; Netze als Linien (Dicke = Fluss)."""
    xy = {slot: (slot % inst.cols, -(slot // inst.cols)) for slot in range(inst.n)}
    pos = [xy[int(p[i])] for i in range(inst.n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[xy[s][0] for s in xy], y=[xy[s][1] for s in xy], mode="markers", marker=dict(symbol="square", size=58, color="rgba(150,150,150,0.18)", line=dict(width=1, color="rgba(150,150,150,0.5)")),
                             hoverinfo="skip", showlegend=False))
    weights = sorted({int(inst.F[a, b]) for a in range(inst.n) for b in range(a + 1, inst.n) if inst.F[a, b]})
    for w in weights:
        xs, ys = [], []
        for a in range(inst.n):
            for b in range(a + 1, inst.n):
                if inst.F[a, b] == w:
                    xs += [pos[a][0], pos[b][0], None]
                    ys += [pos[a][1], pos[b][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1.2 + 1.8 * w, color="rgba(76,120,168,0.55)"), name=f"Fluss {w}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[q[0] for q in pos], y=[q[1] for q in pos], mode="markers+text", text=list(inst.names), textposition="middle center", showlegend=False,
                             marker=dict(size=34, color=C.COLORS["best"]), textfont=dict(color="white", size=12), hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False, range=[-0.7, inst.cols - 0.3])
    fig.update_yaxes(visible=False, range=[-(inst.rows - 0.3), 0.7])
    return _base(fig, height=max(260, 150 * inst.rows), legend_below=True)


# ----------------------------------------------------------------------------- Experiment 2: alle Anordnungen

def build_hist(ls) -> go.Figure:
    """Zahl der Anordnungen je Kostenwert; das Optimum ist grün, der Mittelwert eine Linie."""
    colors = [C.COLORS["best"] if v == ls.opt else C.COLORS["other"] for v in ls.cost_values]
    fig = go.Figure(go.Bar(x=ls.cost_values, y=ls.cost_counts, marker_color=colors, hovertemplate="Kosten %{x}: %{y} Anordnungen<extra></extra>", showlegend=False))
    fig.add_vline(x=ls.mean_cost, line=dict(color=C.COLORS["line"], dash="dot"), annotation_text="Mittel", annotation_position="top right")
    fig.update_layout(title=f"Kosten aller {ls.perms:,} Anordnungen".replace(",", " "))
    fig.update_xaxes(title_text="Kosten (Summe Fluss · Abstand)")
    fig.update_yaxes(title_text="Zahl der Anordnungen")
    return _base(fig, legend_below=False)


# ----------------------------------------------------------------------------- Experiment 3 und 4: Tausch, Abstieg, Einzugsgebiete

def build_deltas(deltas, names) -> go.Figure:
    """Δ aller möglichen Tausche der Startanordnung, nach Δ sortiert; verbessernde Tausche (Δ < 0) sind grün."""
    rows = sorted(deltas, key=lambda t: t[2])
    fig = go.Figure(go.Bar(x=[f"{names[a]}↔{names[b]}" for a, b, _ in rows], y=[d for _, _, d in rows], marker_color=[C.COLORS["best"] if d < 0 else C.COLORS["other"] for _, _, d in rows],
                           hovertemplate="%{x}: Δ = %{y}<extra></extra>", showlegend=False))
    fig.update_layout(title="Änderung der Kosten durch jeden möglichen Tausch")
    fig.update_yaxes(title_text="Δ der Kosten (negativ = besser)")
    fig.update_xaxes(tickangle=-60)
    return _base(fig, legend_below=False)


def build_descent(costs, opt: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(range(len(costs))), y=costs, mode="lines+markers", line=dict(color=C.COLORS["line"], width=3), marker=dict(size=9), name="Kosten", hovertemplate="Schritt %{x}: Kosten %{y}<extra></extra>"))
    fig.add_hline(y=opt, line=dict(color=C.COLORS["best"], dash="dash"), annotation_text="Optimum", annotation_position="bottom right")
    fig.update_layout(title="Steilster Abstieg vom gewählten Start")
    fig.update_xaxes(title_text="Tausch", dtick=1)
    fig.update_yaxes(title_text="Kosten")
    return _base(fig, legend_below=False)


def build_basins(ls) -> go.Figure:
    """Jedes schwache lokale Optimum: Kosten gegen die Zahl der Starts, deren Abstieg dort endet (grün = globales Optimum)."""
    is_opt = ls.local_cost == ls.opt
    fig = go.Figure()
    for flag, name, color in ((True, "globales Optimum", C.COLORS["best"]), (False, "lokales Optimum", C.COLORS["local"])):
        m = is_opt == flag
        fig.add_trace(go.Scatter(x=ls.local_cost[m], y=ls.local_basin[m], mode="markers", name=name, marker=dict(size=9, color=color, opacity=0.75), hovertemplate="Kosten %{x}: %{y} Starts<extra></extra>"))
    fig.update_layout(title="Wohin die Abstiege führen")
    fig.update_xaxes(title_text="Kosten des lokalen Optimums", dtick=1)
    log_axis(fig, ls.local_basin.tolist(), "Zahl der Starts, die dort enden")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 5: Wachstum

def build_growth(rows) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Lokale Optima je Bahn (Median, Spanne)", "Erfolg eines Abstiegs (Median, Spanne)"))
    xs = [r["n"] for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["local_orbits"] for r in rows], mode="lines+markers", name="lokale Optima je Bahn", line=dict(color=C.COLORS["local"], width=3), marker=dict(size=9),
                             error_y=dict(type="data", symmetric=False, array=[r["local_max"] - r["local_orbits"] for r in rows], arrayminus=[r["local_orbits"] - r["local_min"] for r in rows], visible=True)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["success"] for r in rows], mode="lines+markers", name="Erfolg eines Starts (%)", line=dict(color=C.COLORS["best"], width=3), marker=dict(size=9),
                             error_y=dict(type="data", symmetric=False, array=[100 * (r["success_max"] - r["success"]) for r in rows], arrayminus=[100 * (r["success"] - r["success_min"]) for r in rows], visible=True)), row=1, col=2)
    fig.update_xaxes(title_text="Bauteile n", tickvals=xs)
    log_axis(fig, [r["local_orbits"] for r in rows] + [r["local_min"] for r in rows] + [r["local_max"] for r in rows], "lokale Optima je Bahn", row=1, col=1)
    fig.update_yaxes(title_text="Anteil der Starts (%)", range=[0, 100], row=1, col=2)
    return _base(fig, legend_below=True)
