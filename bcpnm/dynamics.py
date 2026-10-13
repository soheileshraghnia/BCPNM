"""Simulation of network dynamics.

Update schemes
--------------
``"synchronous"``
    All nodes update at once from the previous state (Kauffman, 1969).
``"asynchronous"``
    General asynchronous updating: at each step one node, chosen uniformly at random,
    updates (Garg et al., 2008). With symmetric Blume–Capel rules this is Gibbs sampling.
``"sequential"``
    Every node updates once per step in a fresh random order (a "sweep").

All simulations are vectorized over replicate runs and fully determined by ``seed``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import PNMGraph
from .utils import make_rng

SCHEMES = ("synchronous", "asynchronous", "sequential")


def _node_update(graph: PNMGraph, X: np.ndarray, i: int, rows, rng, cache) -> np.ndarray:
    name = graph.names[i]
    rule = graph.rules[name]
    pidx = cache.setdefault(("p", i), [graph.index(p) for p in rule.inputs])
    vals = cache.setdefault(("v", i), graph.nodes[name].states.array)
    sub = X[rows] if rows is not None else X
    parents = sub[:, pidx] if pidx else np.empty((sub.shape[0], 0))
    return rule.sample(parents, sub[:, i], vals, rng)


def step(graph: PNMGraph, X: np.ndarray, scheme: str = "synchronous", rng=None, _cache=None) -> np.ndarray:
    """Advance replicate states ``X`` (R x N) by one update step."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    rng = make_rng(rng)
    cache = {} if _cache is None else _cache
    X = np.asarray(X, dtype=float)
    R, N = X.shape
    if scheme == "synchronous":
        new = X.copy()
        for i in range(N):
            new[:, i] = _node_update(graph, X, i, None, rng, cache)
        return new
    if scheme == "asynchronous":
        new = X.copy()
        chosen = rng.integers(0, N, size=R)
        for i in range(N):
            rows = np.flatnonzero(chosen == i)
            if rows.size:
                new[rows, i] = _node_update(graph, X, i, rows, rng, cache)
        return new
    # sequential sweep in random order per replicate
    new = X.copy()
    order = np.argsort(rng.random((R, N)), axis=1)
    for k in range(N):
        for i in range(N):
            rows = np.flatnonzero(order[:, k] == i)
            if rows.size:
                new[rows, i] = _node_update(graph, new, i, rows, rng, cache)
    return new


@dataclass
class Intervention:
    """A perturbation that is active during ``start <= t < end`` (``end=None``: until the end)."""

    perturbation: object
    start: int = 0
    end: Optional[int] = None
    name: str = ""

    def active(self, t: int) -> bool:
        return t >= self.start and (self.end is None or t < self.end)


@dataclass
class Trajectory:
    """Simulated states with shape ``(T + 1, R, N)``: time x replicate run x node."""

    states: np.ndarray
    names: List[str]
    scheme: str
    seed: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.states.shape[0] - 1

    @property
    def runs(self) -> int:
        return self.states.shape[1]

    def node(self, name: str) -> np.ndarray:
        """States of one node, shape ``(T + 1, R)``."""
        return self.states[:, :, self.names.index(name)]

    def prob(self, value: float, node: Optional[str] = None) -> np.ndarray:
        """Proportion of runs in ``value`` at each time: shape ``(T + 1,)`` for one node,
        or ``(T + 1, N)`` for all nodes."""
        if node is not None:
            return (self.node(node) == value).mean(axis=1)
        return (self.states == value).mean(axis=1)

    def occupancy(self, value: float) -> np.ndarray:
        """Proportion of nodes in ``value`` at each time, averaged over runs."""
        return (self.states == value).mean(axis=(1, 2))

    def final(self) -> np.ndarray:
        return self.states[-1]

    def to_long(self):
        """Long-format ``pandas.DataFrame`` with columns time, run, node, state."""
        import pandas as pd

        T1, R, N = self.states.shape
        t, r, n = np.meshgrid(np.arange(T1), np.arange(R), np.arange(N), indexing="ij")
        return pd.DataFrame({"time": t.ravel(), "run": r.ravel(),
                             "node": np.array(self.names)[n.ravel()], "state": self.states.ravel()})

    def to_panel(self, run: int = 0) -> np.ndarray:
        """States of one run as a ``(T + 1) x N`` array (e.g. for inference)."""
        return self.states[:, run, :]


def simulate(graph: PNMGraph, x0: Optional[np.ndarray] = None, steps: int = 100, runs: int = 1,
             scheme: str = "synchronous", seed: Optional[int] = None,
             interventions: Sequence[Intervention] = (), record_every: int = 1) -> Trajectory:
    """Simulate ``runs`` replicate trajectories of length ``steps``.

    Parameters
    ----------
    graph : PNMGraph
    x0 : array, optional
        Initial state: a vector (N,) used for every run, an array (runs x N), or
        ``None`` for node ``initial`` values (random where not set), drawn per run.
    steps : int
    runs : int
    scheme : {"synchronous", "asynchronous", "sequential"}
    seed : int, optional
        Seed of the random stream; the same seed reproduces the same trajectories.
    interventions : sequence of :class:`Intervention`
        Time-windowed perturbations (see :mod:`bcpnm.perturbation`).
    record_every : int
        Keep every ``k``-th state to save memory.
    """
    graph.validate()
    rng = make_rng(seed)
    if x0 is None:
        X = np.vstack([graph.initial_state(rng) for _ in range(runs)]).astype(float)
    else:
        x0 = np.asarray(x0, dtype=float)
        X = np.tile(x0, (runs, 1)) if x0.ndim == 1 else x0.copy()
    if X.shape != (runs, graph.N):
        raise ValueError(f"x0 must have shape ({runs}, {graph.N}).")
    variants: Dict[Tuple[int, ...], Tuple[PNMGraph, dict]] = {(): (graph, {})}

    def graph_at(t):
        key = tuple(k for k, iv in enumerate(interventions) if iv.active(t))
        if key not in variants:
            g = graph
            for k in key:
                g = interventions[k].perturbation.apply(g)
            variants[key] = (g, {})
        return variants[key]

    hist = [X.copy()]
    for t in range(steps):
        g, cache = graph_at(t)
        X = step(g, X, scheme, rng, cache)
        # clamped nodes take their value immediately at the start of an intervention
        if (t + 1) % record_every == 0:
            hist.append(X.copy())
    states = np.asarray(hist, dtype=np.int8 if np.all(np.abs(hist[0]) < 127) else float)
    return Trajectory(states, graph.names, scheme, seed if isinstance(seed, (int, np.integer)) else None,
                      {"steps": steps, "runs": runs, "record_every": record_every,
                       "interventions": [iv.name or repr(iv.perturbation) for iv in interventions]})


def successors(graph: PNMGraph, X: np.ndarray) -> np.ndarray:
    """Deterministic synchronous successors of states ``X`` (requires deterministic rules)."""
    if not graph.deterministic:
        raise ValueError("successors() requires deterministic rules; use deterministic_limit().")
    return step(graph, X, "synchronous", rng=0)
