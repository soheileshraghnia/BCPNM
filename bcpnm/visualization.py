"""Plots for networks, trajectories, attractors and phase behaviour (Matplotlib).

Colours: for ternary models, blue = +1, grey = 0, red = -1 (a diverging scheme with a
neutral midpoint); for Boolean and ordinal models, a single-hue blue ramp. Every
function returns the Matplotlib ``Axes`` so figures can be customized and saved.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .core import PNMGraph

POS, NEU, NEG = "#2a78d6", "#a9a79f", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
RAMP = ["#e8f1fc", "#86b6ef", "#2a78d6", "#104281", "#0d366b"]


def _ax(ax, figsize=(6, 4)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


LIGHT = "#ecebe6"


def state_colors(values: Sequence[float]) -> Dict[float, str]:
    """Colours per state value: diverging for (-1, 0, +1), light-to-dark otherwise."""
    vals = [float(v) for v in values]
    if vals == [-1.0, 0.0, 1.0]:
        return {-1.0: NEG, 0.0: NEU, 1.0: POS}
    if len(vals) == 2:
        return {vals[0]: LIGHT, vals[1]: POS}
    ramp = [LIGHT] + RAMP[1:]
    return {v: ramp[min(k, len(ramp) - 1)] for k, v in enumerate(vals)}


def _pad_limits(ax, pos, pad=0.22):
    xy = np.array(list(pos.values()), dtype=float)
    span = np.maximum(xy.max(0) - xy.min(0), 1e-9)
    lo, hi = xy.min(0) - pad * span - 0.12, xy.max(0) + pad * span + 0.12
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])


def plot_network(graph: PNMGraph, state: Optional[Sequence[float]] = None, ax=None, pos=None,
                 weight_attr: str = "weight", seed: int = 1, node_size: int = 1100, font_size: int = 9,
                 labels: Optional[Dict[str, str]] = None):
    """Draw the network; arrows show influence, line width |weight|, colour the sign
    (dark = positive, red dashed = negative). Nodes are coloured by ``state`` if given."""
    import networkx as nx

    ax = _ax(ax, (6, 5))
    G = graph.to_networkx()
    pos = pos or nx.spring_layout(G, seed=seed)
    if state is not None:
        cols = state_colors(next(iter(graph.nodes.values())).states.values)
        nc = [cols.get(float(v), NEU) for v in state]
    else:
        nc = ["#ffffff"] * graph.N
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, edgecolors=INK, linewidths=1.0, node_size=node_size)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=font_size)
    for u, v, d in G.edges(data=True):
        w = d.get(weight_attr, 1.0)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, width=0.6 + 1.8 * min(abs(w), 2) / 2,
                               edge_color=INK if w >= 0 else NEG, style="solid" if w >= 0 else "dashed",
                               arrows=True, arrowsize=12, node_size=node_size,
                               connectionstyle="arc3,rad=0.08")
    _pad_limits(ax, pos)
    ax.set_axis_off()
    return ax


def plot_trajectory(traj, run: int = 0, ax=None, labels: Optional[Dict[float, str]] = None):
    """Raster of node states over time for one run."""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    ax = _ax(ax, (7, 0.4 * len(traj.names) + 1.2))
    S = traj.states[:, run, :].T
    vals = np.unique(traj.states).astype(float)
    cols = state_colors(list(vals))
    cmap = ListedColormap([cols[v] for v in vals])
    norm = BoundaryNorm(np.concatenate([vals - 0.5, [vals[-1] + 0.5]]), cmap.N)
    ax.imshow(S, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(np.arange(-0.5, len(traj.names), 1), minor=True)
    ax.grid(which="minor", axis="y", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_yticks(range(len(traj.names)), traj.names)
    ax.set_xlabel("Update step")
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for sp in ax.spines.values():
        sp.set_visible(False)
    if labels:
        from matplotlib.patches import Patch

        ax.legend(handles=[Patch(color=cols[v], label=labels.get(v, str(v))) for v in vals],
                  loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=len(vals), frameon=False)
    return ax


def plot_probabilities(traj, value: float, nodes: Optional[Sequence[str]] = None, ax=None, shade=None):
    """Proportion of runs in ``value`` over time, one line per node."""
    ax = _ax(ax)
    nodes = nodes or traj.names
    styles = ["-", "-", "--", ":", "-.", (0, (5, 1)), (0, (1, 1))]
    greys = [POS, INK, "#52514e", MUTED, "#b7b5ad", "#6a6964", "#3a3936"]
    if shade:
        ax.axvspan(*shade, color="#e8f1fc", lw=0)
    for k, n in enumerate(nodes):
        ax.plot(traj.prob(value, n), color=greys[k % len(greys)], ls=styles[k % len(styles)], lw=1.6, label=n)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Update step"); ax.set_ylabel(f"P(state = {value:+g})" if value else "P(state = 0)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return ax


def plot_basins(result, ax=None, labels: Optional[Sequence[str]] = None, title: str = ""):
    """Horizontal bars of basin fractions (or stationary basin mass) per attractor."""
    ax = _ax(ax, (6, 0.45 * len(result.attractors) + 1.2))
    fr = [a.basin_fraction for a in result.attractors]
    lab = labels or [a.label or f"A{k}" for k, a in enumerate(result.attractors)]
    y = np.arange(len(fr))[::-1]
    ax.barh(y, fr, color=POS, height=0.55)
    for yy, f in zip(y, fr):
        ax.text(f + 0.01, yy, f"{f:.1%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, lab)
    ax.set_xlim(0, max(fr) * 1.18 if fr else 1)
    ax.set_xlabel("Basin share")
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return ax


def plot_state_transition_graph(graph: PNMGraph, scheme: str = "synchronous", ax=None, max_states: int = 300,
                                result=None, seed: int = 2):
    """State transition graph of a small deterministic network, attractor states highlighted."""
    import networkx as nx

    from .attractors import async_transition_graph, find_attractors
    from .dynamics import successors

    S = graph.state_space_size()
    if S > max_states:
        raise ValueError(f"{S} states is too many to draw; reduce the network or raise max_states.")
    ax = _ax(ax, (6, 6))
    X = graph.all_states()
    if scheme == "synchronous":
        nxt = graph.encode(successors(graph, X))
        G = nx.DiGraph([(i, int(j)) for i, j in enumerate(nxt)])
    else:
        M = async_transition_graph(graph).tocoo()
        G = nx.DiGraph(list(zip(M.row.tolist(), M.col.tolist())))
        G.add_nodes_from(range(S))
    result = result or find_attractors(graph, scheme=scheme)
    att_codes = {int(c) for a in result for c in graph.encode(a.states)}
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=MUTED, width=0.6, arrowsize=6, node_size=40)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[60 if n in att_codes else 18 for n in G],
                           node_color=[POS if n in att_codes else NEU for n in G])
    for a in result:  # label each attractor next to its first state
        if a.label:
            x, y = pos[int(graph.encode(a.states[:1])[0])]
            ax.annotate(a.label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8,
                        color=INK, fontweight="bold")
    _pad_limits(ax, pos, pad=0.08)
    ax.set_axis_off()
    return ax


def plot_hysteresis(sweep, column: int = 0, ax=None, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    """Upward and downward continuation branches of a mean-field sweep."""
    ax = _ax(ax)
    up = sweep.up[:, column] if column == 0 else 1 - sweep.up[:, 1]
    dn = sweep.down[:, column] if column == 0 else 1 - sweep.down[:, 1]
    ax.fill_between(sweep.values, up, dn, color="#e8f1fc", lw=0)
    ax.plot(sweep.values, up, color=POS, lw=1.8, label="increasing")
    ax.plot(sweep.values, dn, color=INK, lw=1.4, ls="--", label="decreasing")
    ax.set_xlabel(xlabel or sweep.parameter)
    ax.set_ylabel(ylabel or ("mean orientation m" if column == 0 else "non-disclosure share S0"))
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return ax


def plot_phase_diagram(M: np.ndarray, betas, deltas, ax=None):
    """Heat map of |m*| over (beta, delta) from :func:`bcpnm.equilibrium.phase_diagram`."""
    from matplotlib.colors import LinearSegmentedColormap

    ax = _ax(ax)
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fe", "#86b6ef", "#2a78d6", "#0d366b"])
    im = ax.pcolormesh(betas, deltas, M, cmap=cmap, vmin=0, vmax=1, shading="auto")
    ax.contour(betas, deltas, M, levels=[0.5], colors="white", linewidths=1.0)
    ax.set_xlabel("concentration β"); ax.set_ylabel("zero-state stability δ")
    ax.figure.colorbar(im, ax=ax, label="|m*|")
    return ax


def plot_derrida(curve: Dict[int, float], ax=None):
    """Derrida plot with the diagonal as reference."""
    ax = _ax(ax, (4, 4))
    d = np.array(list(curve)); v = np.array(list(curve.values()))
    ax.plot(d, d, color=MUTED, lw=1, ls="--", label="no change")
    ax.plot(d, v, "o-", color=POS, lw=1.8, ms=5, label="after one update")
    ax.set_xlabel("initial Hamming distance"); ax.set_ylabel("distance after one update")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return ax
