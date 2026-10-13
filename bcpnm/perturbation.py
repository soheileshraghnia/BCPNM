"""Perturbations and intervention analysis.

A perturbation returns a modified copy of a :class:`~bcpnm.core.PNMGraph`; the original
is never changed. Perturbations can be applied permanently, compared through their
effect on attractors and basins, or scheduled in time with
:class:`~bcpnm.dynamics.Intervention`.

Psychological readings
    * :class:`Clamp` - hold a construct or person at a state (e.g. a facilitator who
      always acknowledges; a symptom suppressed by treatment). With ``value=0`` in a
      Boolean model this is a "knockout"; with the highest value it is "overexpression".
    * :class:`RemoveEdge`, :class:`ScaleEdge`, :class:`FlipEdge` - remove, weaken or reverse
      an influence (e.g. a confidential channel that cuts a hierarchical pressure).
    * :class:`ShiftField` - change the contextual field or bias of a node (e.g. reduced
      anticipated stigma), and :class:`SetParameter` for any rule parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .attractors import AttractorResult, find_attractors
from .core import PNMGraph
from .rules import ClampRule, FixedInputRule, SoftmaxRule, ThresholdRule


class Perturbation:
    """Base class: a perturbation maps a graph to a modified copy (``apply`` or call)."""

    def apply(self, graph: PNMGraph) -> PNMGraph:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, graph: PNMGraph) -> PNMGraph:
        return self.apply(graph)


@dataclass
class Clamp(Perturbation):
    """Fix ``node`` at ``value``."""

    node: str
    value: float

    def apply(self, graph):
        if self.value not in graph.nodes[self.node].states:
            raise ValueError(f"{self.value} is not a state of {self.node}.")
        g = graph.copy()
        g.set_rule(self.node, ClampRule(self.value))
        return g


def knockout(node: str, value: float = 0) -> Clamp:
    """Hold a node at its 'off' (or other) state."""
    return Clamp(node, value)


def overexpress(graph: PNMGraph, node: str) -> Clamp:
    """Hold a node at its highest state."""
    return Clamp(node, max(graph.nodes[node].states.values))


def _weighted(rule):
    return isinstance(rule, (SoftmaxRule, ThresholdRule))


@dataclass
class RemoveEdge(Perturbation):
    """Remove the influence of ``source`` on ``target``.

    For weighted rules the weight (and zero–zero term) is set to zero; for other rules
    the input is replaced by the constant ``neutral`` value.
    """

    source: str
    target: str
    neutral: float = 0.0

    def apply(self, graph):
        g = graph.copy()
        r = g.rules[self.target]
        if self.source not in r.inputs:
            raise ValueError(f"{self.source} is not an input of {self.target}.")
        if _weighted(r):
            k = r.inputs.index(self.source)
            r.w[k] = 0.0
            if isinstance(r, SoftmaxRule):
                r.lam[k] = 0.0
        else:
            g.rules[self.target] = FixedInputRule(r, {self.source: self.neutral})
        return g


@dataclass
class ScaleEdge(Perturbation):
    """Multiply the influence of ``source -> target`` by ``factor`` (weighted rules only).

    ``include_w`` and ``include_lam`` choose whether the polarity weight, the zero–zero
    term, or both are scaled (e.g. ``include_w=False`` weakens only the coordination of
    silence along a tie).
    """

    source: str
    target: str
    factor: float
    include_lam: bool = True
    include_w: bool = True

    def apply(self, graph):
        g = graph.copy()
        r = g.rules[self.target]
        if not _weighted(r):
            raise TypeError("ScaleEdge needs a ThresholdRule or SoftmaxRule.")
        k = r.inputs.index(self.source)
        if self.include_w:
            r.w[k] *= self.factor
        if isinstance(r, SoftmaxRule) and self.include_lam:
            r.lam[k] *= self.factor
        return g


@dataclass
class FlipEdge(Perturbation):
    """Reverse the sign of ``source -> target`` (weighted rules only)."""

    source: str
    target: str

    def apply(self, graph):
        return ScaleEdge(self.source, self.target, -1.0, include_lam=False).apply(graph)


@dataclass
class ShiftField(Perturbation):
    """Add ``amount`` to the field ``h`` (SoftmaxRule) or ``bias`` (ThresholdRule) of ``node``."""

    node: str
    amount: float

    def apply(self, graph):
        g = graph.copy()
        r = g.rules[self.node]
        if isinstance(r, SoftmaxRule):
            r.h += self.amount
        elif isinstance(r, ThresholdRule):
            r.bias += self.amount
        else:
            raise TypeError("ShiftField needs a SoftmaxRule or ThresholdRule.")
        return g


@dataclass
class SetParameter(Perturbation):
    """Set a scalar rule attribute, e.g. ``SetParameter("Mother", "delta", 0.2)``."""

    node: str
    parameter: str
    value: float

    def apply(self, graph):
        g = graph.copy()
        r = g.rules[self.node]
        if not hasattr(r, self.parameter):
            raise AttributeError(f"Rule of {self.node} has no parameter {self.parameter!r}.")
        setattr(r, self.parameter, float(self.value))
        return g


@dataclass
class Combined(Perturbation):
    """Apply several perturbations in order."""

    parts: Sequence[Perturbation]

    def apply(self, graph):
        for p in self.parts:
            graph = p.apply(graph)
        return graph


# ----------------------------------------------------------------------------- analyses
def compare_attractors(base: AttractorResult, perturbed: AttractorResult,
                       key: Callable[[np.ndarray], str] = None):
    """Basin fractions before and after a perturbation, matched by attractor label
    (or by the attractor's states when no labels are set). Returns a DataFrame."""
    import pandas as pd

    def k(a):
        if key is not None:
            return key(a.states)
        return a.label or " | ".join(" ".join(str(int(v)) for v in row) for row in a.states)

    b = {k(a): a.basin_fraction for a in base}
    p = {k(a): a.basin_fraction for a in perturbed}
    keys = list(dict.fromkeys(list(b) + list(p)))
    df = pd.DataFrame({"attractor": keys, "before": [b.get(x, 0.0) for x in keys],
                       "after": [p.get(x, 0.0) for x in keys]})
    df["change"] = df["after"] - df["before"]
    return df


def leverage_scan(graph: PNMGraph, target: Callable[[np.ndarray], bool], value_fn: Callable[[str], float],
                  scheme: str = "synchronous", nodes: Optional[Sequence[str]] = None):
    """Clamp each node in turn and report the share of initial states whose attractor
    satisfies ``target`` (e.g. 'most members acknowledge').

    ``value_fn(node)`` gives the clamp value for each node. Returns a DataFrame sorted
    by the resulting share.
    """
    import pandas as pd

    def share(res):
        return float(sum(a.basin_fraction for a in res if all(target(s) for s in a.states)))

    base = share(find_attractors(graph, scheme=scheme))
    rows = []
    for n in nodes or graph.names:
        res = find_attractors(Clamp(n, value_fn(n)).apply(graph), scheme=scheme)
        rows.append({"node": n, "clamped_to": value_fn(n), "share": share(res), "change": share(res) - base})
    df = pd.DataFrame(rows).sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    df.attrs["baseline"] = base
    return df


def intervention_effect(graph: PNMGraph, perturbation: Perturbation, outcome: Callable[[np.ndarray], np.ndarray],
                        steps: int = 200, runs: int = 2000, scheme: str = "asynchronous", seed: int = 0,
                        burn: int = 50, x0=None) -> Dict[str, float]:
    """Mean of ``outcome(states)`` after burn-in without and with a sustained perturbation,
    using common random numbers (the same seed) for both conditions."""
    from .dynamics import simulate

    out = {}
    for label, g in (("baseline", graph), ("perturbed", perturbation.apply(graph))):
        tr = simulate(g, x0=x0, steps=steps, runs=runs, scheme=scheme, seed=seed)
        vals = np.asarray(outcome(tr.states[burn:].reshape(-1, graph.N)), float)
        out[label] = float(vals.mean())
    out["difference"] = out["perturbed"] - out["baseline"]
    return out
