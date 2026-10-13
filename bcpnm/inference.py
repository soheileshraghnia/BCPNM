"""Learning update rules and parameters from data.

* :func:`reveal` - REVEAL-style search for the smallest input sets that determine each
  node's next state (Liang, Fuhrman, & Somogyi, 1998), returning lookup-table rules.
* :func:`fit_transition_model` - penalized maximum likelihood for the directed
  Blume–Capel transition rule (multinomial logit per node) from time series.
* :func:`fit_equilibrium` - nodewise pseudo-likelihood for the symmetric equilibrium
  model from cross-sectional data (cf. Waldorp et al., 2026), symmetrized afterwards.
* :func:`log_score` - mean one-step log predictive probability, for held-out comparison.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .core import PNMGraph, StateSpace
from .rules import SoftmaxRule, TableRule

ArrayList = Union[np.ndarray, Sequence[np.ndarray]]


def _pairs(data: ArrayList) -> Tuple[np.ndarray, np.ndarray]:
    """Stack (x_t, x_{t+1}) pairs from one panel (T x N), a list of panels, or a
    Trajectory-like (T x R x N) array."""
    if isinstance(data, np.ndarray) and data.ndim == 3:
        data = [data[:, r, :] for r in range(data.shape[1])]
    if isinstance(data, np.ndarray):
        data = [data]
    X = np.concatenate([np.asarray(d, float)[:-1] for d in data])
    Y = np.concatenate([np.asarray(d, float)[1:] for d in data])
    return X, Y


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _cond_entropy(y: np.ndarray, Xs: np.ndarray) -> float:
    if Xs.shape[1] == 0:
        return _entropy(np.unique(y, return_counts=True)[1])
    keys, inv = np.unique(Xs, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    H, n = 0.0, y.size
    for k in range(keys.shape[0]):
        yk = y[inv == k]
        H += yk.size / n * _entropy(np.unique(yk, return_counts=True)[1])
    return H


@dataclass
class RevealResult:
    """Inferred input sets, information-theoretic determination, in-sample accuracy and
    lookup-table rules per node; ``to_graph(template)`` builds a deterministic model."""

    parents: Dict[str, Tuple[str, ...]]
    determination: Dict[str, float]      # share of the entropy of x_i(t+1) explained, I/H
    accuracy: Dict[str, float]            # in-sample accuracy of the lookup table
    tables: Dict[str, TableRule]

    def to_graph(self, template: PNMGraph, name: str = "inferred") -> PNMGraph:
        g = template.copy()
        g.name = name
        for n, r in self.tables.items():
            g.set_rule(n, r)
        return g


def reveal(data: ArrayList, names: Sequence[str], max_parents: int = 3, criterion: str = "bic",
           min_determination: float = 0.95) -> RevealResult:
    """REVEAL-style inference of input sets and lookup tables from synchronous time series.

    Parameters
    ----------
    data : array (T x N), list of arrays, or array (T x R x N)
    names : node names in column order
    max_parents : int
        Largest input set searched.
    criterion : {"reveal", "bic"}
        ``"reveal"``: the smallest set whose mutual information with the next state is at
        least ``min_determination`` of that state's entropy (exact for noise-free data).
        ``"bic"``: minimizes ``n * H(y | inputs) * ln 2 + 0.5 * n_params * ln n`` and is
        preferable for noisy data.
    """
    X, Y = _pairs(data)
    n, N = X.shape
    parents, det, acc, tables = {}, {}, {}, {}
    for i, name in enumerate(names):
        y = Y[:, i]
        Hy = _entropy(np.unique(y, return_counts=True)[1])
        ky = np.unique(y).size
        best = None
        for k in range(0, max_parents + 1):
            for S in itertools.combinations(range(N), k):
                Hc = _cond_entropy(y, X[:, list(S)])
                frac = 1.0 if Hy == 0 else (Hy - Hc) / Hy
                if criterion == "reveal":
                    score = (k, -frac)
                    ok = frac >= min_determination
                    if ok and (best is None or score < best[0]):
                        best = (score, S, frac)
                else:
                    ncfg = np.unique(X[:, list(S)], axis=0).shape[0] if k else 1
                    bic = n * Hc * np.log(2) + 0.5 * ncfg * max(ky - 1, 1) * np.log(n)
                    if best is None or bic < best[0]:
                        best = (bic, S, frac)
            if criterion == "reveal" and best is not None:
                break
        if best is None:  # nothing reached the determination threshold
            best = (None, tuple(range(min(max_parents, N))), float("nan"))
        S = best[1]
        pn = tuple(names[j] for j in S)
        table = {}
        Xs = X[:, list(S)]
        if S:
            keys, inv = np.unique(Xs, axis=0, return_inverse=True)
            inv = np.asarray(inv).ravel()
            hit = 0
            for kk, key in enumerate(keys):
                vals, cnt = np.unique(y[inv == kk], return_counts=True)
                table[tuple(int(v) for v in key)] = float(vals[np.argmax(cnt)])
                hit += cnt.max()
        else:
            vals, cnt = np.unique(y, return_counts=True)
            hit = cnt.max()
        parents[name], det[name], acc[name] = pn, float(best[2]), float(hit / n)
        tables[name] = TableRule(pn, table, default=float(np.unique(y)[np.argmax(np.unique(y, return_counts=True)[1])]))
    return RevealResult(parents, det, acc, tables)


# ----------------------------------------------------------------------------- Blume–Capel likelihoods
def _design(Xpred: np.ndarray, current: Optional[np.ndarray], values: np.ndarray, i: int,
            use_lam: bool, use_rho: bool, exclude_self: bool = True):
    """Feature tensor Phi (n, S, P) for the softmax utility of node i and parameter names."""
    n, N = Xpred.shape
    others = [j for j in range(N) if not (exclude_self and j == i)]
    S = values.size
    feats, names = [], []
    v = values[None, :]
    for j in others:
        feats.append(v * Xpred[:, [j]]); names.append(("w", j))
    if use_lam:
        for j in others:
            feats.append((v == 0) * (Xpred[:, [j]] == 0)); names.append(("lam", j))
    feats.append(np.broadcast_to(v, (n, S))); names.append(("h", i))
    feats.append(np.broadcast_to(-v ** 2, (n, S))); names.append(("delta", i))
    if use_rho:
        feats.append(v == current[:, None]); names.append(("rho", i))
    return np.stack(feats, axis=2).astype(float), names


def _fit_softmax(Phi: np.ndarray, yidx: np.ndarray, penalize: np.ndarray, alpha: float):
    n, S, P = Phi.shape

    def f(theta):
        U = Phi @ theta
        U -= U.max(axis=1, keepdims=True)
        lse = np.log(np.exp(U).sum(axis=1))
        ll = U[np.arange(n), yidx] - lse
        pr = np.exp(U - lse[:, None])
        grad_ll = Phi[np.arange(n), yidx] - np.einsum("ns,nsp->np", pr, Phi)
        obj = -ll.sum() / n + 0.5 * alpha * (penalize * theta ** 2).sum()
        g = -grad_ll.sum(axis=0) / n + alpha * penalize * theta
        return obj, g

    res = minimize(f, np.zeros(P), jac=True, method="L-BFGS-B")
    return res.x, res


@dataclass
class TransitionFit:
    """Estimates of the directed Blume–Capel rule (on the scale beta * parameter):
    ``W[i, j]`` and ``Lam[i, j]`` are influences of j on i; ``to_graph()`` builds the model."""

    names: List[str]
    W: np.ndarray              # W[i, j]: influence of j on i
    Lam: np.ndarray            # directed zero–zero terms
    h: np.ndarray
    delta: np.ndarray
    rho: np.ndarray
    converged: bool
    meta: dict = field(default_factory=dict)

    def to_graph(self, states: StateSpace = None, name: str = "fitted", beta: float = 1.0,
                 threshold: float = 0.0) -> PNMGraph:
        """Graph with :class:`SoftmaxRule` per node; edges with |w| and |lam| at or below
        ``threshold`` are dropped."""
        states = states or StateSpace.ternary()
        g = PNMGraph(name, meta={"fitted": True})
        for n in self.names:
            g.add_node(n, states)
        for i, n in enumerate(self.names):
            w = {self.names[j]: self.W[i, j] for j in range(len(self.names))
                 if j != i and (abs(self.W[i, j]) > threshold or abs(self.Lam[i, j]) > threshold)}
            lam = {self.names[j]: self.Lam[i, j] for j in range(len(self.names)) if self.names[j] in w}
            g.set_rule(n, SoftmaxRule(w, lam, h=self.h[i], delta=self.delta[i], rho=self.rho[i], beta=beta))
        return g


def fit_transition_model(data: ArrayList, names: Sequence[str], values: Sequence[float] = (-1, 0, 1),
                         lam: bool = True, rho: bool = True, alpha: float = 1e-3) -> TransitionFit:
    """Fit the directed Blume–Capel transition rule to synchronous time series.

    The concentration :math:`\\beta` is fixed at 1 for identification, so estimates are
    on the scale :math:`\\beta \\cdot` parameter. ``alpha`` is an L2 penalty on the
    network terms (``W`` and ``Lam``).
    """
    X, Y = _pairs(data)
    n, N = X.shape
    values = np.asarray(values, float)
    W = np.zeros((N, N)); L = np.zeros((N, N))
    h = np.zeros(N); d = np.zeros(N); r = np.zeros(N)
    conv = True
    for i in range(N):
        Phi, pn = _design(X, X[:, i], values, i, lam, rho)
        yidx = np.searchsorted(values, Y[:, i])
        pen = np.array([1.0 if k in ("w", "lam") else 0.0 for k, _ in pn])
        theta, res = _fit_softmax(Phi, yidx, pen, alpha)
        conv &= bool(res.success)
        for (kind, j), val in zip(pn, theta):
            if kind == "w": W[i, j] = val
            elif kind == "lam": L[i, j] = val
            elif kind == "h": h[i] = val
            elif kind == "delta": d[i] = val
            elif kind == "rho": r[i] = val
    return TransitionFit(list(names), W, L, h, d, r, conv, {"n_transitions": n, "alpha": alpha})


@dataclass
class EquilibriumFit:
    """Symmetrized pseudo-likelihood estimates of the equilibrium model; ``to_model()``
    returns a :class:`~bcpnm.equilibrium.BlumeCapelModel`."""

    names: List[str]
    K: np.ndarray
    Lam: np.ndarray
    h: np.ndarray
    delta: np.ndarray
    converged: bool

    def to_model(self, beta: float = 1.0):
        from .equilibrium import BlumeCapelModel

        return BlumeCapelModel(self.names, self.K, self.Lam, self.delta, self.h, beta=beta)


def fit_equilibrium(samples: np.ndarray, names: Sequence[str], lam: bool = True, alpha: float = 1e-3) -> EquilibriumFit:
    """Nodewise pseudo-likelihood for the equilibrium Blume–Capel model (beta = 1).

    Each node's full conditional (Equation 5 of the tutorial) is a multinomial logit in
    the other nodes; the two estimates of each symmetric coupling are averaged.
    """
    X = np.asarray(samples, float)
    n, N = X.shape
    values = np.array([-1.0, 0.0, 1.0])
    K = np.zeros((N, N)); L = np.zeros((N, N)); h = np.zeros(N); d = np.zeros(N)
    conv = True
    for i in range(N):
        Phi, pn = _design(X, None, values, i, lam, use_rho=False)
        yidx = np.searchsorted(values, X[:, i])
        pen = np.array([1.0 if k in ("w", "lam") else 0.0 for k, _ in pn])
        theta, res = _fit_softmax(Phi, yidx, pen, alpha)
        conv &= bool(res.success)
        for (kind, j), val in zip(pn, theta):
            if kind == "w": K[i, j] = val
            elif kind == "lam": L[i, j] = val
            elif kind == "h": h[i] = val
            elif kind == "delta": d[i] = val
    return EquilibriumFit(list(names), (K + K.T) / 2, (L + L.T) / 2, h, d, conv)


def log_score(graph: PNMGraph, data: ArrayList) -> float:
    """Mean log predictive probability of observed synchronous transitions (nats per node)."""
    X, Y = _pairs(data)
    total = 0.0
    for i, name in enumerate(graph.names):
        rule = graph.rules[name]
        pidx = [graph.index(p) for p in rule.inputs]
        values = graph.nodes[name].states.array
        P = rule.probs(X[:, pidx] if pidx else np.empty((X.shape[0], 0)), X[:, i], values)
        idx = graph.nodes[name].states.index(Y[:, i])
        total += np.log(np.clip(P[np.arange(X.shape[0]), idx], 1e-12, 1)).sum()
    return float(total / (X.shape[0] * graph.N))
