"""Equilibrium Blume–Capel probabilistic network model (BC PNM).

States :math:`x_i \\in \\{-1, 0, +1\\}`; symmetric polarity couplings :math:`K`,
symmetric zero–zero coordination :math:`\\Lambda`, zero-state stability
:math:`\\delta_i`, field :math:`h_i` and concentration :math:`\\beta`:

.. math::
    H(\\mathbf{x}) = -\\sum_{i<j} K_{ij} x_i x_j - \\sum_{i<j} \\Lambda_{ij}(1-x_i^2)(1-x_j^2)
                     + \\sum_i \\delta_i x_i^2 - \\sum_i h_i x_i,
    \\qquad P(\\mathbf{x}) = \\exp[-\\beta H(\\mathbf{x})] / Z.

The module provides exact enumeration (small N), Gibbs sampling, conditional
probabilities, homogeneous mean-field equations and continuation sweeps for hysteresis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import PNMGraph, StateSpace
from .rules import SoftmaxRule
from .utils import make_rng

TERNARY = np.array([-1.0, 0.0, 1.0])


def _as_vec(x, N):
    a = np.asarray(x, dtype=float)
    return np.full(N, float(a)) if a.ndim == 0 else a.astype(float)


def _as_mat(M, N):
    if M is None:
        return np.zeros((N, N))
    M = np.asarray(M, dtype=float)
    if M.shape != (N, N):
        raise ValueError(f"Matrix must be {N} x {N}.")
    if not np.allclose(M, M.T):
        raise ValueError("Equilibrium couplings must be symmetric; use the directed model for asymmetry.")
    M = M.copy()
    np.fill_diagonal(M, 0.0)
    return M


class BlumeCapelModel:
    """Equilibrium Blume–Capel PNM.

    Parameters
    ----------
    names : sequence of str
    K, Lam : (N, N) symmetric arrays, optional
        Polarity coupling and zero–zero coordination (zero diagonal).
    delta, h : float or (N,) arrays
    beta : float
    labels : sequence of three str
        Labels for the states -1, 0, +1.
    """

    def __init__(self, names: Sequence[str], K=None, Lam=None, delta=0.0, h=0.0, beta: float = 1.0,
                 labels: Sequence[str] = ("negative", "neutral", "positive")):
        self.names = list(names)
        N = len(self.names)
        self.K = _as_mat(K, N)
        self.Lam = _as_mat(Lam, N)
        self.delta = _as_vec(delta, N)
        self.h = _as_vec(h, N)
        self.beta = float(beta)
        self.labels = tuple(labels)

    @property
    def N(self) -> int:
        return len(self.names)

    @classmethod
    def from_edges(cls, names, edges: Sequence[Tuple[str, str, float, float]], **kwargs) -> "BlumeCapelModel":
        """Build from ``(i, j, K_ij, Lam_ij)`` tuples (undirected)."""
        N = len(names)
        K = np.zeros((N, N)); L = np.zeros((N, N))
        for a, b, k, l in edges:
            i, j = names.index(a), names.index(b)
            K[i, j] = K[j, i] = k
            L[i, j] = L[j, i] = l
        return cls(names, K, L, **kwargs)

    # energy and probabilities ------------------------------------------------
    def energy(self, X) -> np.ndarray:
        """Energy :math:`H(\\mathbf{x})` of configurations (R x N)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        A = 1.0 - X ** 2
        pol = 0.5 * np.einsum("ri,ij,rj->r", X, self.K, X)
        zz = 0.5 * np.einsum("ri,ij,rj->r", A, self.Lam, A)
        return -pol - zz + (X ** 2) @ self.delta - X @ self.h

    def configurations(self) -> np.ndarray:
        if self.N > 13:
            raise MemoryError("Exact enumeration is limited to N <= 13 (3^13 configurations).")
        idx = np.indices((3,) * self.N).reshape(self.N, -1).T
        return TERNARY[idx]

    def exact(self) -> Tuple[np.ndarray, np.ndarray]:
        """All configurations and their exact Boltzmann probabilities."""
        X = self.configurations()
        E = self.energy(X)
        w = np.exp(-self.beta * (E - E.min()))
        return X, w / w.sum()

    def conditional(self, X, i: int) -> np.ndarray:
        """:math:`P(X_i = s \\mid \\mathbf{x}_{-i})` for s in (-1, 0, 1); shape (R, 3)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        eta = self.h[i] + X @ self.K[i]
        zeta = (1.0 - X ** 2) @ self.Lam[i]
        u = self.beta * (TERNARY[None, :] * eta[:, None] - self.delta[i] * TERNARY[None, :] ** 2
                         + zeta[:, None] * (1 - TERNARY[None, :] ** 2))
        u -= u.max(axis=1, keepdims=True)
        p = np.exp(u)
        return p / p.sum(axis=1, keepdims=True)

    def log_odds_zero(self, X, i: int) -> np.ndarray:
        """Log-odds of state 0 versus +1 and versus -1 (columns), i.e.
        :math:`\\beta(\\delta_i + \\zeta_i \\mp \\eta_i)`."""
        p = self.conditional(X, i)
        return np.column_stack([np.log(p[:, 1] / p[:, 2]), np.log(p[:, 1] / p[:, 0])])

    def beg_form(self) -> Dict[str, np.ndarray]:
        """Equivalent Blume–Emery–Griffiths parameters.

        The zero–zero term equals a biquadratic coupling :math:`-\\Lambda x_i^2 x_j^2`,
        a shift of each zero-state stability by :math:`\\sum_j \\Lambda_{ij}`, and a constant.
        """
        return {"J": self.K.copy(), "biquadratic": self.Lam.copy(),
                "delta_effective": self.delta + self.Lam.sum(axis=1), "h": self.h.copy(),
                "constant": -0.5 * self.Lam.sum()}

    # summaries -----------------------------------------------------------------
    @staticmethod
    def occupancy(X, value: float = 0.0, weights=None) -> float:
        """Expected proportion of nodes in ``value`` (e.g. 0 = non-disclosure)."""
        X = np.atleast_2d(X)
        v = (X == value).mean(axis=1)
        return float(np.average(v, weights=weights))

    def co_occupancy(self, X, value: float = 0.0, weights=None, adjacency=None) -> float:
        """Expected proportion of ties whose two ends are both in ``value`` (connected co-silence)."""
        A = (self.K != 0) | (self.Lam != 0) if adjacency is None else np.asarray(adjacency, bool)
        iu = np.argwhere(np.triu(A, 1))
        X = np.atleast_2d(X)
        both = ((X[:, iu[:, 0]] == value) & (X[:, iu[:, 1]] == value)).mean(axis=1)
        return float(np.average(both, weights=weights))

    def marginals(self) -> np.ndarray:
        """Exact marginal probabilities P(x_i = s), shape (N, 3)."""
        X, p = self.exact()
        return np.stack([[p[X[:, i] == s].sum() for s in TERNARY] for i in range(self.N)])

    # sampling and conversion ---------------------------------------------------
    def to_graph(self, name: str = "blume_capel") -> PNMGraph:
        """Equivalent :class:`PNMGraph` with heat-bath :class:`SoftmaxRule` per node.

        Asynchronous simulation of this graph is Gibbs sampling from the model.
        """
        g = PNMGraph(name, meta={"model": "Blume-Capel equilibrium"})
        ss = StateSpace((-1, 0, 1), dict(zip((-1, 0, 1), self.labels)), name="ternary")
        for n in self.names:
            g.add_node(n, ss)
        for i, n in enumerate(self.names):
            nb = [j for j in range(self.N) if self.K[i, j] != 0 or self.Lam[i, j] != 0]
            g.set_rule(n, SoftmaxRule({self.names[j]: self.K[i, j] for j in nb},
                                      {self.names[j]: self.Lam[i, j] for j in nb},
                                      h=self.h[i], delta=self.delta[i], rho=0.0, beta=self.beta))
        return g

    def gibbs(self, sweeps: int = 1000, burn: int = 100, chains: int = 4, seed=None, x0=None) -> np.ndarray:
        """Heat-bath Gibbs samples after burn-in, shape ``(chains * (sweeps - burn), N)``.

        Each sweep updates every node once in random order.
        """
        rng = make_rng(seed)
        X = rng.choice(TERNARY, size=(chains, self.N)) if x0 is None else np.tile(np.asarray(x0, float), (chains, 1))
        out = []
        for s in range(sweeps):
            for i in rng.permutation(self.N):
                p = self.conditional(X, i)
                c = np.cumsum(p, axis=1)
                X[:, i] = TERNARY[(rng.random((chains, 1)) > c).sum(axis=1).clip(0, 2)]
            if s >= burn:
                out.append(X.copy())
        return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------------- mean-field theory
def mean_field(beta: float, J: float, Lam: float, delta: float, h: float = 0.0,
               m0: float = 0.0, q0: float = 0.67, tol: float = 1e-11, max_iter: int = 20000,
               damping: float = 0.5) -> Tuple[float, float]:
    """Homogeneous mean-field fixed point ``(m, q)`` for a complete network with total
    couplings ``J`` and ``Lam`` (each normalized by N - 1).

    :math:`m = 2e^{-\\beta d}\\sinh(\\beta\\eta)/[1 + 2e^{-\\beta d}\\cosh(\\beta\\eta)]`,
    :math:`q = 2e^{-\\beta d}\\cosh(\\beta\\eta)/[1 + 2e^{-\\beta d}\\cosh(\\beta\\eta)]`,
    with :math:`\\eta = h + Jm` and :math:`d = \\delta + \\Lambda(1-q)`.
    """
    m, q = m0, q0
    for _ in range(max_iter):
        eta = h + J * m
        d = delta + Lam * (1 - q)
        a = -beta * d
        lp, lm = a + beta * eta, a - beta * eta
        mx = max(0.0, lp, lm)
        w0, wp, wm = np.exp(-mx), np.exp(lp - mx), np.exp(lm - mx)
        Z = w0 + wp + wm
        mn, qn = (wp - wm) / Z, (wp + wm) / Z
        if abs(mn - m) < tol and abs(qn - q) < tol:
            return float(mn), float(qn)
        m = damping * m + (1 - damping) * mn
        q = damping * q + (1 - damping) * qn
    return float(m), float(q)


def mean_field_free_energy(m: float, q: float, beta: float, J: float, Lam: float, delta: float, h: float = 0.0) -> float:
    """Variational free energy per node at the product distribution implied by ``(m, q)``."""
    pp, pm = (q + m) / 2, (q - m) / 2
    p0 = 1 - q
    p = np.clip(np.array([pp, p0, pm]), 1e-300, 1)
    E = -0.5 * J * m ** 2 - 0.5 * Lam * (1 - q) ** 2 + delta * q - h * m
    return float(E + (p * np.log(p)).sum() / beta)


def mean_field_global(beta, J, Lam, delta, h=0.0) -> Tuple[float, float]:
    """Globally stable mean-field solution (lowest free energy over several starts)."""
    best = None
    for m0, q0 in ((0, .05), (0, .5), (0, .95), (.5, .6), (.95, .97), (-.95, .97)):
        m, q = mean_field(beta, J, Lam, delta, h, m0, q0)
        f = mean_field_free_energy(m, q, beta, J, Lam, delta, h)
        if best is None or f < best[0] - 1e-12:
            best = (f, m, q)
    return best[1], best[2]


@dataclass
class Sweep:
    """Result of a continuation sweep: parameter values and (m, q) going up and down."""

    values: np.ndarray
    up: np.ndarray
    down: np.ndarray
    parameter: str

    def jumps(self, column: int = 0, min_jump: float = 0.2) -> Tuple[Optional[float], Optional[float]]:
        """Parameter values at which the upward and downward branches jump."""
        du, dd = np.diff(self.up[:, column]), np.diff(self.down[:, column])
        up = self.values[np.argmax(np.abs(du) > min_jump) + 1] if (np.abs(du) > min_jump).any() else None
        dn = self.values[np.argmax(np.abs(dd) > min_jump)] if (np.abs(dd) > min_jump).any() else None
        return (float(up) if up is not None else None, float(dn) if dn is not None else None)


def continuation(parameter: str, values: Sequence[float], beta: float, J: float, Lam: float,
                 delta: float, h: float = 0.0) -> Sweep:
    """Follow the mean-field solution as ``parameter`` ("h" or "delta") increases, then decreases.

    Differences between the two branches indicate hysteresis (path dependence).
    """
    values = np.asarray(values, dtype=float)
    base = dict(beta=beta, J=J, Lam=Lam, delta=delta, h=h)

    def run(vals, m, q):
        out = []
        for v in vals:
            kw = dict(base); kw[parameter] = v
            m, q = mean_field(kw["beta"], kw["J"], kw["Lam"], kw["delta"], kw["h"], m, q)
            out.append((m, q))
        return np.array(out)

    if parameter == "h":
        up = run(values, -0.95, 0.97)
        down = run(values[::-1], 0.95, 0.97)[::-1]
    elif parameter == "delta":
        up = run(values, 0.0, 0.95)
        down = run(values[::-1], 0.0, 0.02)[::-1]
    else:
        raise ValueError("parameter must be 'h' or 'delta'")
    return Sweep(values, up, down, parameter)


def phase_diagram(betas: Sequence[float], deltas: Sequence[float], J: float = 1.0, Lam: float = 0.0,
                  h: float = 0.0) -> np.ndarray:
    """|m*| of the globally stable mean-field solution on a (delta x beta) grid."""
    M = np.zeros((len(deltas), len(betas)))
    for a, d in enumerate(deltas):
        for b, be in enumerate(betas):
            M[a, b] = abs(mean_field_global(be, J, Lam, d, h)[0])
    return M
