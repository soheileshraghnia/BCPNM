"""Attractors, basins and stationary distributions.

* Deterministic networks: exhaustive or sampled attractor search under synchronous
  updating (fixed points and cycles; Kauffman, 1969) and exhaustive search under
  general asynchronous updating (terminal strongly connected components of the state
  transition graph; Garg et al., 2008).
* Stochastic networks: exact stationary distribution for small state spaces and the
  probability mass carried by the basins of the zero-noise ("quasi-") attractors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, connected_components

from .core import PNMGraph
from .dynamics import step, successors
from .rules import deterministic_limit
from .utils import make_rng


@dataclass
class Attractor:
    """A fixed point, synchronous cycle, or asynchronous complex attractor."""

    states: np.ndarray          # (L, N) state values
    kind: str                   # "fixed point", "cycle" or "complex"
    basin: int                  # basin size (synchronous) or strong basin size (asynchronous)
    basin_fraction: float
    weak_basin: Optional[int] = None
    label: str = ""

    @property
    def size(self) -> int:
        return self.states.shape[0]

    def as_dicts(self, names) -> List[Dict[str, int]]:
        return [{n: int(v) for n, v in zip(names, row)} for row in self.states]


@dataclass
class AttractorResult:
    """Attractors of a deterministic network with basin information; iterate over it, or
    use ``summary()``, ``table()`` and ``label_by()``."""

    attractors: List[Attractor]
    names: List[str]
    scheme: str
    method: str
    n_states: int
    basin_map: Optional[np.ndarray] = None   # attractor index per state code (synchronous exhaustive)

    def __len__(self):
        return len(self.attractors)

    def __iter__(self):
        return iter(self.attractors)

    def table(self):
        """Attractors as a ``pandas.DataFrame`` (one row per attractor)."""
        import pandas as pd

        rows = []
        sg = self._signed()
        for k, a in enumerate(self.attractors):
            rows.append({"attractor": k, "label": a.label, "kind": a.kind, "length": a.size,
                         "basin": a.basin, "basin_fraction": round(a.basin_fraction, 4),
                         "weak_basin": a.weak_basin,
                         "state": " | ".join(_fmt_row(row, sg) for row in a.states[:3])
                         + (" ..." if a.size > 3 else "")})
        return pd.DataFrame(rows)

    def _signed(self) -> bool:
        return any((a.states < 0).any() for a in self.attractors)

    def label_by(self, fn: Callable[[np.ndarray], str]) -> "AttractorResult":
        """Attach labels computed from each attractor's states (e.g. 'all silent')."""
        for a in self.attractors:
            a.label = fn(a.states)
        return self

    def summary(self) -> str:
        lines = [f"{len(self.attractors)} attractor(s) [{self.scheme}, {self.method}, {self.n_states:,} states]"]
        sg = self._signed()
        for k, a in enumerate(self.attractors):
            st = _fmt_row(a.states[0], sg)
            lab = f" '{a.label}'" if a.label else ""
            lines.append(f"  A{k}{lab}: {a.kind} (length {a.size}), basin {a.basin_fraction:.1%}  [{st}{' ...' if a.size > 1 else ''}]")
        return "\n".join(lines)


def _fmt_row(row, signed: bool) -> str:
    if signed:
        return " ".join(f"{int(v):+d}" if v else "0" for v in row)
    return " ".join(str(int(v)) for v in row)


def deterministic_graph(graph: PNMGraph) -> PNMGraph:
    """Copy of ``graph`` with every rule replaced by its zero-noise limit."""
    g = graph.copy()
    for n, r in g.rules.items():
        g.rules[n] = deterministic_limit(r)
    g.meta["deterministic_limit_of"] = graph.name
    return g


def _check_deterministic(graph):
    if not graph.deterministic:
        raise ValueError("Attractor search needs deterministic rules. Use deterministic_graph(graph) "
                         "for the zero-noise limit, or stationary_distribution() for stochastic models.")


# ----------------------------------------------------------------------------- synchronous
def _sync_exhaustive(graph: PNMGraph) -> AttractorResult:
    X = graph.all_states()
    S = X.shape[0]
    nxt = graph.encode(successors(graph, X))
    codes = np.arange(S)
    M = sparse.csr_matrix((np.ones(S), (codes, nxt)), shape=(S, S))
    ncomp, comp = connected_components(M, directed=True, connection="weak")
    # peel transient states (in-degree zero) to leave the cycle states
    indeg = np.bincount(nxt, minlength=S)
    alive = np.ones(S, dtype=bool)
    frontier = np.flatnonzero(indeg == 0)
    while frontier.size:
        alive[frontier] = False
        tgt = nxt[frontier]
        np.subtract.at(indeg, tgt, 1)
        cand = np.unique(tgt)
        frontier = cand[(indeg[cand] == 0) & alive[cand]]
    sizes = np.bincount(comp, minlength=ncomp)
    atts, comp_to_att = [], {}
    for c in range(ncomp):
        cyc_nodes = np.flatnonzero(alive & (comp == c))
        start = int(cyc_nodes.min())
        order = [start]
        nx_ = int(nxt[start])
        while nx_ != start:
            order.append(nx_); nx_ = int(nxt[nx_])
        comp_to_att[c] = len(atts)
        atts.append(Attractor(graph.decode(order), "fixed point" if len(order) == 1 else "cycle",
                              int(sizes[c]), float(sizes[c] / S)))
    idx = np.argsort([-a.basin for a in atts], kind="stable")
    remap = {int(old): new for new, old in enumerate(idx)}
    atts = [atts[i] for i in idx]
    basin_map = np.array([remap[comp_to_att[c]] for c in comp])
    return AttractorResult(atts, graph.names, "synchronous", "exhaustive", S, basin_map)


def _sync_sampling(graph: PNMGraph, samples: int, max_steps: int, rng) -> AttractorResult:
    X = graph.random_states(samples, rng)
    hist = [graph.encode(X)]
    for _ in range(max_steps):
        X = successors(graph, X)
        hist.append(graph.encode(X))
    H = np.asarray(hist)  # (T+1, R)
    found: Dict[tuple, int] = {}
    counts: Dict[tuple, int] = {}
    unresolved = 0
    for r in range(H.shape[1]):
        seen = {}
        cyc = None
        for t, c in enumerate(H[:, r]):
            c = int(c)
            if c in seen:
                cyc = [int(x) for x in H[seen[c]:t, r]]
                break
            seen[c] = t
        if cyc is None:
            unresolved += 1
            continue
        k = cyc.index(min(cyc))
        key = tuple(cyc[k:] + cyc[:k])
        counts[key] = counts.get(key, 0) + 1
    atts = []
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        atts.append(Attractor(graph.decode(list(key)), "fixed point" if len(key) == 1 else "cycle",
                              n, n / samples))
    res = AttractorResult(atts, graph.names, "synchronous", f"sampling ({samples} starts)", samples)
    if unresolved:
        res.method += f"; {unresolved} runs unresolved after {max_steps} steps"
    return res


# ----------------------------------------------------------------------------- asynchronous
def async_transition_graph(graph: PNMGraph) -> sparse.csr_matrix:
    """Sparse adjacency of the asynchronous state transition graph (deterministic rules)."""
    _check_deterministic(graph)
    X = graph.all_states()
    S = X.shape[0]
    rows, cols = [], []
    codes = np.arange(S)
    for i, name in enumerate(graph.names):
        rule = graph.rules[name]
        pidx = [graph.index(p) for p in rule.inputs]
        parents = X[:, pidx] if pidx else np.empty((S, 0))
        new_i = rule.next_value(parents, X[:, i], graph.nodes[name].states.array)
        changed = new_i != X[:, i]
        if changed.any():
            Y = X[changed].copy()
            Y[:, i] = new_i[changed]
            rows.append(codes[changed]); cols.append(graph.encode(Y))
    if rows:
        r, c = np.concatenate(rows), np.concatenate(cols)
    else:
        r = c = np.array([], dtype=np.int64)
    return sparse.csr_matrix((np.ones(r.size), (r, c)), shape=(S, S))


def _async_exhaustive(graph: PNMGraph) -> AttractorResult:
    M = async_transition_graph(graph)
    S = M.shape[0]
    ncomp, lab = connected_components(M, directed=True, connection="strong")
    coo = M.tocoo()
    exits = lab[coo.row] != lab[coo.col]
    nonterminal = np.zeros(ncomp, dtype=bool)
    nonterminal[np.unique(lab[coo.row[exits]])] = True
    terminal = np.flatnonzero(~nonterminal)
    MT = M.T.tocsr()
    reach = []
    atts = []
    for c in terminal:
        members = np.flatnonzero(lab == c)
        basin = breadth_first_order(MT, int(members[0]), directed=True, return_predecessors=False)
        reach.append(basin)
        atts.append(Attractor(graph.decode(members), "fixed point" if members.size == 1 else "complex",
                              0, 0.0, weak_basin=int(basin.size)))
    count = np.zeros(S, dtype=np.int64)
    for b in reach:
        count[b] += 1
    for a, b in zip(atts, reach):
        strong = int((count[b] == 1).sum())
        a.basin, a.basin_fraction = strong, strong / S
    order = np.argsort([-a.basin for a in atts], kind="stable")
    return AttractorResult([atts[i] for i in order], graph.names, "asynchronous", "exhaustive", S)


def find_attractors(graph: PNMGraph, scheme: str = "synchronous", method: str = "auto",
                    samples: int = 2000, max_steps: int = 500, seed=None,
                    exhaustive_limit: int = 2_000_000) -> AttractorResult:
    """Find the attractors of a deterministic network.

    Parameters
    ----------
    scheme : {"synchronous", "asynchronous"}
    method : {"auto", "exhaustive", "sampling"}
        ``"auto"`` enumerates the full state space when it has at most
        ``exhaustive_limit`` configurations (200,000 for asynchronous search).
    samples, max_steps : int
        Random starts and maximal path length for synchronous sampling.
    seed : int, optional
    """
    _check_deterministic(graph)
    graph.validate()
    S = graph.state_space_size()
    if scheme == "synchronous":
        if method == "exhaustive" or (method == "auto" and S <= exhaustive_limit):
            return _sync_exhaustive(graph)
        return _sync_sampling(graph, samples, max_steps, make_rng(seed))
    if scheme == "asynchronous":
        if S > min(exhaustive_limit, 200_000):
            raise MemoryError("Exhaustive asynchronous search is limited to 200,000 states.")
        return _async_exhaustive(graph)
    raise ValueError("scheme must be 'synchronous' or 'asynchronous'.")


# ----------------------------------------------------------------------------- stochastic models
@dataclass
class StationaryResult:
    """Stationary distribution over all configurations of a stochastic network."""

    pi: np.ndarray
    graph: PNMGraph
    scheme: str
    iterations: int
    X: np.ndarray = field(repr=False, default=None)

    def __post_init__(self):
        if self.X is None:
            self.X = self.graph.all_states()

    def marginal(self, node: str) -> Dict[int, float]:
        i = self.graph.index(node)
        vals = self.graph.nodes[node].states.values
        return {v: float(self.pi[self.X[:, i] == v].sum()) for v in vals}

    def prob(self, predicate: Callable[[np.ndarray], np.ndarray]) -> float:
        """Probability of the event ``predicate(X)`` (a boolean vector over configurations)."""
        return float(self.pi[np.asarray(predicate(self.X), bool)].sum())

    def expectation(self, fn: Callable[[np.ndarray], np.ndarray]) -> float:
        return float(self.pi @ np.asarray(fn(self.X), float))

    def top(self, k: int = 5):
        idx = np.argsort(-self.pi)[:k]
        return [(self.graph.as_dict(self.X[j]), float(self.pi[j])) for j in idx]


def transition_matrix(graph: PNMGraph, scheme: str = "asynchronous") -> sparse.csr_matrix:
    """Exact Markov transition matrix over all configurations (small networks only)."""
    graph.validate()
    X = graph.all_states()
    S = X.shape[0]
    codes = np.arange(S)
    if scheme == "asynchronous":
        rows, cols, vals = [], [], []
        for i, name in enumerate(graph.names):
            rule = graph.rules[name]
            pidx = [graph.index(p) for p in rule.inputs]
            values = graph.nodes[name].states.array
            P = rule.probs(X[:, pidx] if pidx else np.empty((S, 0)), X[:, i], values)
            for k, v in enumerate(values):
                Y = X.copy(); Y[:, i] = v
                nz = P[:, k] > 0
                rows.append(codes[nz]); cols.append(graph.encode(Y[nz])); vals.append(P[nz, k] / graph.N)
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(S, S))
    if scheme == "synchronous":
        if S > 6561:
            raise MemoryError("Exact synchronous transition matrices are limited to 6,561 states.")
        T = np.ones((S, S))
        for i, name in enumerate(graph.names):
            rule = graph.rules[name]
            pidx = [graph.index(p) for p in rule.inputs]
            values = graph.nodes[name].states.array
            P = rule.probs(X[:, pidx] if pidx else np.empty((S, 0)), X[:, i], values)  # (S, k)
            col = graph.nodes[name].states.index(X[:, i])  # next-state index of node i in target y
            T *= P[:, col]
        return sparse.csr_matrix(T)
    raise ValueError("scheme must be 'asynchronous' or 'synchronous'.")


def stationary_distribution(graph: PNMGraph, scheme: str = "asynchronous", tol: float = 1e-12,
                            max_iter: int = 200000) -> StationaryResult:
    """Stationary distribution by power iteration on the lazy chain ``(I + P) / 2``."""
    P = transition_matrix(graph, scheme)
    S = P.shape[0]
    PT = P.T.tocsr()
    pi = np.full(S, 1.0 / S)
    for it in range(1, max_iter + 1):
        new = 0.5 * pi + 0.5 * (PT @ pi)
        new /= new.sum()
        if np.abs(new - pi).sum() < tol:
            pi = new
            break
        pi = new
    return StationaryResult(pi, graph, scheme, it)


def quasi_attractors(graph: PNMGraph, stationary: Optional[StationaryResult] = None,
                     scheme: str = "asynchronous") -> AttractorResult:
    """Attractors of the zero-noise limit, with basin mass under the stochastic model.

    ``basin_fraction`` is replaced by the stationary probability of the attractor's
    (strong) basin, i.e. the long-run share of time spent in that regime.
    """
    det = deterministic_graph(graph)
    res = find_attractors(det, scheme=scheme)
    st = stationary or stationary_distribution(graph, scheme=scheme)
    if scheme == "synchronous":
        for k, a in enumerate(res.attractors):
            a.basin_fraction = float(st.pi[res.basin_map == k].sum())
    else:
        M = async_transition_graph(det)
        MT = M.T.tocsr()
        S = M.shape[0]
        reach, count = [], np.zeros(S, dtype=np.int64)
        for a in res.attractors:
            c = int(det.encode(a.states[:1])[0])
            b = breadth_first_order(MT, c, directed=True, return_predecessors=False)
            reach.append(b); count[b] += 1
        for a, b in zip(res.attractors, reach):
            a.basin_fraction = float(st.pi[b[count[b] == 1]].sum())
    res.method = f"zero-noise attractors; basin mass under {graph.name}"
    return res
