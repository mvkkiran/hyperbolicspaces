"""Plotly figures for the app."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from data import Hierarchy, short, tree_layout
from hyperbolic import geodesic_points

PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2",
    "#FF9DA6", "#9D755D", "#BAB0AC", "#72B7B2", "#EECA3B",
]


def _colors(h: Hierarchy, group_depth: int) -> tuple[list[str], list[str]]:
    lab, _, names = h.labels_at_depth(group_depth)
    cols = [PALETTE[l % len(PALETTE)] if l >= 0 else "#777777" for l in lab]
    legend = [names[l] if l >= 0 else "upper levels" for l in lab]
    return cols, legend


def tree_figure(h: Hierarchy, group_depth: int = 2, highlight: list[int] | None = None) -> go.Figure:
    pos = tree_layout(h)
    cols, legend = _colors(h, group_depth)
    fig = go.Figure()
    ex, ey = [], []
    for u, v in h.graph.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#999", width=1),
                             hoverinfo="skip", showlegend=False))
    xs = [pos[i][0] for i in range(h.n)]
    ys = [pos[i][1] for i in range(h.n)]
    sizes = [16 if highlight and i in highlight else 10 for i in range(h.n)]
    lines = [("#000" if highlight and i in highlight else "#fff") for i in range(h.n)]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[short(s) for s in h.statements], textposition="bottom center",
        textfont=dict(size=9),
        marker=dict(size=sizes, color=cols, line=dict(width=1.5, color=lines)),
        hovertext=h.statements, hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def embedding_figure(
    h: Hierarchy, emb: np.ndarray, geometry: str, title: str,
    group_depth: int = 2, highlight: list[int] | None = None, show_labels: bool = True,
    background: tuple | None = None,
) -> go.Figure:
    """2-D scatter of an embedding with tree edges drawn as geodesics."""
    cols, legend = _colors(h, group_depth)
    fig = go.Figure()

    if background is not None:  # decision regions: (xs, z, n_classes)
        xs, z, k = background
        colorscale = [[i / max(k - 1, 1), PALETTE[i % len(PALETTE)]] for i in range(k)]
        fig.add_trace(go.Heatmap(x=xs, y=xs, z=z, colorscale=colorscale, zmin=0, zmax=k - 1,
                                 opacity=0.25, showscale=False, hoverinfo="skip"))

    if geometry == "hyperbolic":
        t = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines",
                                 line=dict(color="#444", width=1.5, dash="dot"),
                                 hoverinfo="skip", showlegend=False, name="boundary (∞)"))

    ex, ey = [], []
    for u, v in h.graph.edges():
        if geometry == "hyperbolic":
            pts = geodesic_points(emb[u], emb[v])
            ex += list(pts[:, 0]) + [None]
            ey += list(pts[:, 1]) + [None]
        else:
            ex += [emb[u, 0], emb[v, 0], None]
            ey += [emb[u, 1], emb[v, 1], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#888", width=1),
                             hoverinfo="skip", showlegend=False))

    seen = set()
    for i in range(h.n):
        name = legend[i]
        is_hl = bool(highlight) and i in highlight
        fig.add_trace(go.Scatter(
            x=[emb[i, 0]], y=[emb[i, 1]], mode="markers+text" if show_labels else "markers",
            text=[short(h.statements[i])] if show_labels else None,
            textposition="top center", textfont=dict(size=8),
            marker=dict(size=14 if is_hl else 9, color=cols[i],
                        line=dict(width=2 if is_hl else 1, color="#000" if is_hl else "#fff")),
            hovertext=f"{h.statements[i]}<br>depth {h.depth[i]} · |x| = {np.linalg.norm(emb[i]):.3f}",
            hoverinfo="text", name=name, legendgroup=name, showlegend=name not in seen,
        ))
        seen.add(name)

    if geometry == "hyperbolic":
        rng = dict(range=[-1.05, 1.05])
    else:
        m = float(np.abs(emb).max()) * 1.15 + 1e-9
        rng = dict(range=[-m, m])
    fig.update_layout(
        title=title, height=520, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(**rng, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(**rng, showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, font=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def distance_scatter(D_emb: np.ndarray, dist: np.ndarray, scale: float, title: str) -> go.Figure:
    """Tree distance vs embedding distance for every pair (ideal = diagonal)."""
    iu = np.triu_indices(dist.shape[0], 1)
    x, y = dist[iu], D_emb[iu] * scale
    jitter = (np.random.default_rng(0).random(len(x)) - 0.5) * 0.3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + jitter, y=y, mode="markers",
                             marker=dict(size=5, opacity=0.5, color="#4C78A8"), name="pairs"))
    m = max(x.max(), y.max())
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines",
                             line=dict(color="#E45756", dash="dash"), name="perfect"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="tree distance (hops)", yaxis_title="embedding distance (rescaled)",
                      showlegend=False)
    return fig


def depth_radius_figure(h: Hierarchy, emb: np.ndarray, title: str) -> go.Figure:
    r = np.linalg.norm(emb, axis=1)
    fig = go.Figure(go.Scatter(
        x=h.depth + (np.random.default_rng(1).random(h.n) - 0.5) * 0.2, y=r, mode="markers",
        marker=dict(size=8, color="#54A24B"), hovertext=h.statements, hoverinfo="text"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="depth in hierarchy", yaxis_title="distance from origin")
    return fig


def sweep_figure(rows: list[dict], metric: str) -> go.Figure:
    fig = go.Figure()
    for geom, color in (("euclidean", "#E45756"), ("hyperbolic", "#4C78A8")):
        pts = [r for r in rows if r["geometry"] == geom]
        fig.add_trace(go.Scatter(x=[r["dim"] for r in pts], y=[r[metric] for r in pts],
                                 mode="lines+markers", name=geom, line=dict(color=color)))
    fig.update_layout(title=metric, height=300, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="embedding dimension", xaxis_type="log",
                      xaxis=dict(tickvals=[2, 3, 5, 10, 20, 50]))
    return fig


def loss_figure(histories: dict[str, list[float]]) -> go.Figure:
    fig = go.Figure()
    for name, hist in histories.items():
        fig.add_trace(go.Scatter(y=hist, mode="lines", name=name))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="epoch", yaxis_title="loss", yaxis_type="log")
    return fig
