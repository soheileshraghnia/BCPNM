"""Checks of model validity, sensitivity and robustness.

* :func:`expert_consistency` - does each specified influence have the sign experts expect?
* :func:`input_activities` / :func:`average_sensitivity` - how strongly each input moves
  each rule (Shmulevich & Kauffman, 2004), generalized to multi-valued and stochastic rules.
* :func:`canalizing_inputs` - input values that fix a rule's output regardless of the
  other inputs (Kauffman et al., 2003).
* :func:`derrida_curve` - one-step spreading of small differences (Derrida & Pomeau, 1986).
* :func:`attractor_robustness` - share of single-node perturbations of an attractor after
  which the network returns to the same attractor.
* :func:`threshold_ties` - input configurations at which a threshold rule sits exactly on
  a cut point, so that results depend on the tie-breaking convention.
* :func:`parameter_recovery` and :func:`oat_sensitivity` - simulation-based checks.
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .attractors import AttractorResult, find_attractors
from .core import PNMGraph
from .dynamics import step, successors
from .utils import make_rng


def _rule_probs(graph: PNMGraph, node: str, X: np.ndarray) -> np.ndarray:
    rule = graph.rules[node]
    i = graph.index(node)
    pidx = [graph.index(p) for p in rule.inputs]
    return rule.probs(X[:, pidx] if pidx else np.empty((X.shape[0], 0)), X[:, i], graph.nodes[node].states.array)


def expected_next(graph: PNMGraph, node: str, X: np.ndarray) -> np.ndarray:
    """Expected next value of ``node`` given states ``X`` (R x N)."""
    return _rule_probs(graph, node, X) @ graph.nodes[node].states.array


def expert_consistency(graph: PNMGraph, expectations: Sequence[Tuple[str, str, int]], samples: int = 2000,
                       seed=0):
    """Compare model-implied influence signs with expert expectations.

    For each ``(source, target, expected_sign)`` the source is raised by one state
    (where possible) in random reference configurations, and the average change in the
    target's expected next value is computed. Returns a DataFrame with the implied
    effect, its sign and agreement, and the proportion of agreements in ``df.attrs``.
    """
    import pandas as pd

    rng = make_rng(seed)
    X = graph.random_states(samples, rng).astype(float)
    rows = []
    for src, tgt, sign in expectations:
        j = graph.index(src)
        vals = graph.nodes[src].states.array
        pos = graph.nodes[src].states.index(X[:, j])
        ok = pos < len(vals) - 1
        Xu = X[ok].copy()
        Xu[:, j] = vals[pos[ok] + 1]
        eff = float((expected_next(graph, tgt, Xu) - expected_next(graph, tgt, X[ok])).mean())
        implied = int(np.sign(np.round(eff, 10)))
        rows.append({"source": src, "target": tgt, "expected": int(sign), "effect": round(eff, 4),
                     "implied": implied, "agree": implied == int(np.sign(sign))})
    df = pd.DataFrame(rows)
    df.attrs["agreement"] = float(df["agree"].mean()) if len(df) else float("nan")
    return df


def input_activities(graph: PNMGraph, samples: int = 4000, seed=0) -> Dict[str, Dict[str, float]]:
    """Activity of each input on each rule.

    For deterministic rules: probability that changing the input to another random
    value changes the output (the activity of Shmulevich & Kauffman, 2004). For
    stochastic rules: mean total-variation distance between next-state distributions.
    """
    rng = make_rng(seed)
    X = graph.random_states(samples, rng).astype(float)
    out = {}
    for node, rule in graph.rules.items():
        acts = {}
        base = _rule_probs(graph, node, X)
        for inp in rule.inputs:
            j = graph.index(inp)
            vals = graph.nodes[inp].states.array
            Xp = X.copy()
            cur = graph.nodes[inp].states.index(X[:, j])
            shift = rng.integers(1, len(vals), size=samples)
            Xp[:, j] = vals[(cur + shift) % len(vals)]
            new = _rule_probs(graph, node, Xp)
            acts[inp] = float(0.5 * np.abs(new - base).sum(axis=1).mean())
        out[node] = acts
    return out


def average_sensitivity(graph: PNMGraph, samples: int = 4000, seed=0) -> Dict[str, float]:
    """Sum of input activities per node, and their mean over nodes (key ``"mean"``)."""
    acts = input_activities(graph, samples, seed)
    s = {n: float(sum(a.values())) for n, a in acts.items()}
    s["mean"] = float(np.mean([v for k, v in s.items()]))
    return s


def canalizing_inputs(graph: PNMGraph, node: str) -> list:
    """Input values that alone determine a deterministic rule's output.

    Returns a list of ``(input, input_value, forced_output)``. The node's own current
    value is included as an input when the rule depends on it.
    """
    rule = graph.rules[node]
    if not rule.deterministic:
        raise ValueError("Canalization is defined for deterministic rules.")
    inputs = list(rule.inputs)
    spaces = [graph.nodes[n].states.array for n in inputs]
    own = graph.nodes[node].states.array
    grid = np.array(list(itertools.product(*spaces, own))) if inputs else own[:, None]
    parents = grid[:, :-1] if inputs else np.empty((grid.shape[0], 0))
    out = rule.next_value(parents, grid[:, -1], own)
    res = []
    for k, name in enumerate(inputs):
        for v in spaces[k]:
            sel = grid[:, k] == v
            vals = np.unique(out[sel])
            if vals.size == 1:
                res.append((name, int(v), int(vals[0])))
    return res


def derrida_curve(graph: PNMGraph, max_distance: Optional[int] = None, samples: int = 2000, seed=0,
                  scheme: str = "synchronous") -> Dict[int, float]:
    """Mean Hamming distance after one update between states that initially differ in
    ``d`` nodes. Values above ``d`` (slope > 1 at d = 1) indicate that small
    differences grow (sensitive regime); values below indicate that they shrink."""
    rng = make_rng(seed)
    N = graph.N
    max_distance = max_distance or N
    curve = {}
    for d in range(1, max_distance + 1):
        X = graph.random_states(samples, rng).astype(float)
        Y = X.copy()
        for r in range(samples):
            for j in rng.choice(N, size=d, replace=False):
                vals = graph.nodes[graph.names[j]].states.array
                choices = vals[vals != X[r, j]]
                Y[r, j] = rng.choice(choices)
        seed_r = int(rng.integers(1 << 31))
        X1 = step(graph, X, scheme, rng=np.random.default_rng(seed_r))
        Y1 = step(graph, Y, scheme, rng=np.random.default_rng(seed_r))
        curve[d] = float((X1 != Y1).sum(axis=1).mean())
    return curve


def attractor_robustness(graph: PNMGraph, result: Optional[AttractorResult] = None) -> Dict[int, float]:
    """For each synchronous attractor, the share of single-node changes to its states
    after which the network returns to the same attractor (requires exhaustive search)."""
    result = result or find_attractors(graph, scheme="synchronous", method="exhaustive")
    if result.basin_map is None:
        raise ValueError("attractor_robustness needs an exhaustive synchronous result.")
    out = {}
    for k, a in enumerate(result.attractors):
        hits = total = 0
        for row in a.states:
            for j, name in enumerate(graph.names):
                for v in graph.nodes[name].states.values:
                    if v == row[j]:
                        continue
                    y = row.copy(); y[j] = v
                    total += 1
                    hits += int(result.basin_map[int(graph.encode(y)[0])] == k)
        out[k] = hits / total if total else float("nan")
    return out


def threshold_ties(graph: PNMGraph):
    """Count input configurations at which a :class:`~bcpnm.rules.ThresholdRule` activation
    equals a cut point. Nonzero counts mean that results depend on the ``tie`` convention;
    shifting the bias by a small amount removes such knife-edge cases."""
    import pandas as pd

    from .rules import ThresholdRule

    rows = []
    for node, rule in graph.rules.items():
        if not isinstance(rule, ThresholdRule):
            continue
        spaces = [graph.nodes[n].states.array for n in rule.inputs]
        grid = np.array(list(itertools.product(*spaces))) if spaces else np.empty((1, 0))
        a = rule.activation(grid)
        ties = np.isin(a, np.round(rule.thresholds, 9))
        rows.append({"node": node, "configurations": int(grid.shape[0]), "ties": int(ties.sum())})
    return pd.DataFrame(rows, columns=["node", "configurations", "ties"])


def parameter_recovery(true: Mapping[str, np.ndarray], estimated: Mapping[str, np.ndarray]):
    """Correlation, RMSE and sign agreement between true and estimated parameter arrays."""
    import pandas as pd

    rows = []
    for k in true:
        t = np.asarray(true[k], float).ravel()
        e = np.asarray(estimated[k], float).ravel()
        nz = (t != 0) | (e != 0)
        r = float(np.corrcoef(t[nz], e[nz])[0, 1]) if nz.sum() > 2 and t[nz].std() > 0 else float("nan")
        big = np.abs(t) > 1e-9
        rows.append({"parameter": k, "n": int(t.size), "correlation": round(r, 3),
                     "rmse": round(float(np.sqrt(np.mean((t - e) ** 2))), 3),
                     "sign_agreement": round(float((np.sign(t[big]) == np.sign(e[big])).mean()), 3) if big.any() else float("nan")})
    return pd.DataFrame(rows)


def oat_sensitivity(outcome: Callable[[Dict[str, float]], float], params: Mapping[str, float],
                    rel_change: float = 0.1):
    """One-at-a-time elasticities: % change in ``outcome`` per % change in each parameter."""
    import pandas as pd

    base = outcome(dict(params))
    rows = []
    for k, v in params.items():
        p = dict(params)
        dv = rel_change * (abs(v) if v != 0 else 1.0)
        p[k] = v + dv
        up = outcome(p)
        p[k] = v - dv
        dn = outcome(p)
        deriv = (up - dn) / (2 * dv)
        el = deriv * (v / base) if base != 0 and v != 0 else float("nan")
        rows.append({"parameter": k, "value": v, "derivative": round(deriv, 4), "elasticity": round(el, 3)})
    df = pd.DataFrame(rows)
    df.attrs["baseline"] = base
    return df
