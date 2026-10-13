"""Update-rule builders.

Deterministic (logical) rules
    :class:`BooleanRule`, :class:`CaseRule`, :class:`ThresholdRule`, :class:`TableRule`,
    :class:`ClampRule`, :class:`IdentityRule`.
Stochastic rules
    :class:`SoftmaxRule` (the Blume–Capel transition rule), :class:`NoisyRule`
    (random perturbation, as in probabilistic Boolean networks with perturbation) and
    :class:`PBNRule` (random choice among several functions, as in probabilistic
    Boolean networks).
Wrappers
    :class:`FixedInputRule` removes the influence of chosen inputs (edge knockout).

The Blume–Capel rule with ``beta=np.inf`` is deterministic: it becomes a ternary
threshold rule. This is how logical networks arise as the zero-noise limit of the
probabilistic model.
"""
from __future__ import annotations

import ast
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .core import Rule
from .utils import softmax

Number = Union[int, float]


# ----------------------------------------------------------------------------- logical expressions
_ALLOWED_CMP = {ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
                ast.Gt: np.greater, ast.GtE: np.greater_equal}
_ALLOWED_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply}


def _names_in(expr: str) -> List[str]:
    tree = ast.parse(expr, mode="eval")
    seen: List[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in seen and n.id not in ("True", "False"):
            seen.append(n.id)
    return seen


def _evaluate(expr_tree, env: Mapping[str, np.ndarray]):
    """Evaluate a restricted Python expression on NumPy arrays (no function calls)."""
    n = expr_tree
    if isinstance(n, ast.Expression):
        return _evaluate(n.body, env)
    if isinstance(n, ast.BoolOp):
        vals = [np.asarray(_evaluate(v, env)) != 0 for v in n.values]
        f = np.logical_and if isinstance(n.op, ast.And) else np.logical_or
        return f.reduce(vals)
    if isinstance(n, ast.UnaryOp):
        v = _evaluate(n.operand, env)
        if isinstance(n.op, ast.Not):
            return np.logical_not(np.asarray(v) != 0)
        if isinstance(n.op, ast.USub):
            return -np.asarray(v)
        raise ValueError("Unsupported unary operator.")
    if isinstance(n, ast.Compare):
        left = _evaluate(n.left, env)
        result = None
        for op, comp in zip(n.ops, n.comparators):
            right = _evaluate(comp, env)
            if type(op) not in _ALLOWED_CMP:
                raise ValueError("Unsupported comparison.")
            r = _ALLOWED_CMP[type(op)](left, right)
            result = r if result is None else np.logical_and(result, r)
            left = right
        return result
    if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN:
        return _ALLOWED_BIN[type(n.op)](_evaluate(n.left, env), _evaluate(n.right, env))
    if isinstance(n, ast.Name):
        if n.id == "True":
            return True
        if n.id == "False":
            return False
        return env[n.id]
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)):
        return n.value
    raise ValueError(f"Unsupported syntax in rule expression: {ast.dump(n)}")


class CaseRule(Rule):
    """Multi-valued logical rule: the first condition that holds sets the next state.

    Parameters
    ----------
    cases : list of (str, int)
        Pairs ``(condition, value)``; conditions use ``and``, ``or``, ``not``,
        comparisons (``>=``, ``==``, ...) and ``+ - *``. Node names are variables;
        a bare name is true when its value is nonzero.
    default : int or "keep"
        Next state when no condition holds; ``"keep"`` retains the current state.

    Example
    -------
    ``CaseRule([("Distress >= 2 and not Help", 2), ("Distress >= 1", 1)], default=0)``
    """

    kind = "logical"

    def __init__(self, cases: Sequence[Tuple[str, Number]], default: Union[Number, str] = "keep"):
        self.cases = [(str(c), v) for c, v in cases]
        names: List[str] = []
        for c, _ in self.cases:
            for nm in _names_in(c):
                if nm not in names:
                    names.append(nm)
        super().__init__(names)
        self._trees = [ast.parse(c, mode="eval") for c, _ in self.cases]
        self.default = default

    def next_value(self, parents, current, values):
        parents = np.atleast_2d(np.asarray(parents, dtype=float))
        R = parents.shape[0]
        env = {nm: parents[:, k] for k, nm in enumerate(self.inputs)}
        out = np.asarray(current, dtype=float).copy() if self.default == "keep" else np.full(R, float(self.default))
        decided = np.zeros(R, dtype=bool)
        for tree, (_, val) in zip(self._trees, self.cases):
            cond = np.broadcast_to(np.asarray(_evaluate(tree, env)) != 0, (R,))
            take = cond & ~decided
            out[take] = val
            decided |= take
        return out

    def params(self):
        return {"cases": [list(c) for c in self.cases], "default": self.default}


class BooleanRule(CaseRule):
    """Logical rule ``expr`` returning ``true_value`` when it holds and ``false_value`` otherwise.

    Example: ``BooleanRule("Stigma and not Support")``.
    """

    def __init__(self, expr: str, true_value: Number = 1, false_value: Number = 0):
        super().__init__([(expr, true_value)], default=false_value)
        self.expr = expr

    def params(self):
        return {"expr": self.expr, "true_value": self.cases[0][1], "false_value": self.default}


def AND(*names: str) -> BooleanRule:
    """Boolean conjunction of the named inputs."""
    return BooleanRule(" and ".join(names))


def OR(*names: str) -> BooleanRule:
    """Boolean disjunction of the named inputs."""
    return BooleanRule(" or ".join(names))


def NOT(name: str) -> BooleanRule:
    """Boolean negation of one input."""
    return BooleanRule(f"not {name}")


# ----------------------------------------------------------------------------- threshold and table rules
class ThresholdRule(Rule):
    """Weighted-sum threshold rule for Boolean, ternary or ordinal nodes.

    The activation is :math:`a = b + \\sum_j w_j x_j`. With cut points
    :math:`\\theta_1 < \\dots < \\theta_K` and outputs :math:`o_0, \\dots, o_K`, the next
    state is :math:`o_k` where :math:`k` is the number of cut points below :math:`a`.
    If :math:`a` equals a cut point, ``tie="keep"`` keeps the current state when it is
    one of the two adjacent outputs (the convention of Hopfield-type threshold networks).

    Parameters
    ----------
    weights : mapping {input: weight}
    thresholds : sequence of float, default ``(0,)``
    outputs : sequence of int, default ``(0, 1)``; must have ``len(thresholds) + 1`` entries
    bias : float
    tie : {"keep", "low", "high"}
    """

    kind = "threshold"

    def __init__(self, weights: Mapping[str, float], thresholds: Sequence[float] = (0.0,),
                 outputs: Sequence[Number] = (0, 1), bias: float = 0.0, tie: str = "keep"):
        super().__init__(list(weights))
        self.w = np.array([float(weights[k]) for k in weights])
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.outputs = np.asarray(outputs, dtype=float)
        if self.outputs.size != self.thresholds.size + 1:
            raise ValueError("outputs must have one more entry than thresholds.")
        self.bias = float(bias)
        self.tie = tie

    def activation(self, parents):
        """Weighted input sum; rounded to 9 decimals so that ties at a cut point do not
        depend on floating-point summation order."""
        parents = np.atleast_2d(np.asarray(parents, dtype=float))
        a = self.bias + parents @ self.w if self.w.size else np.full(parents.shape[0], self.bias)
        return np.round(a, 9)

    def next_value(self, parents, current, values):
        a = self.activation(parents)
        k_low = np.searchsorted(self.thresholds, a, side="left")   # cut points strictly below a
        k_high = np.searchsorted(self.thresholds, a, side="right")  # cut points at or below a
        out = self.outputs[k_high].copy()
        tie = k_low != k_high
        if tie.any():
            lo, hi = self.outputs[k_low[tie]], self.outputs[k_high[tie]]
            if self.tie == "keep":
                cur = np.asarray(current, dtype=float)[tie]
                out[tie] = np.where(cur == hi, hi, np.where(cur == lo, lo, lo))
            elif self.tie == "low":
                out[tie] = lo
            else:
                out[tie] = hi
        return out

    def params(self):
        return {"weights": self.w.tolist(), "bias": self.bias, "thresholds": self.thresholds.tolist(),
                "outputs": self.outputs.tolist(), "tie": self.tie}


def majority(inputs: Sequence[str], weights: Optional[Sequence[float]] = None, ternary: bool = True) -> ThresholdRule:
    """Weighted majority rule: sign of the weighted input sum (ties keep the current state)."""
    w = dict(zip(inputs, weights if weights is not None else [1.0] * len(inputs)))
    if ternary:
        return ThresholdRule(w, thresholds=(-0.5, 0.5), outputs=(-1, 0, 1))
    return ThresholdRule(w, thresholds=(0.0,), outputs=(0, 1))


class TableRule(Rule):
    """Explicit lookup table from input configurations to the next state.

    Parameters
    ----------
    inputs : sequence of str
    table : mapping {tuple of input values: next value}
    default : int or "keep"
        Used for configurations not listed in the table.
    """

    kind = "table"

    def __init__(self, inputs: Sequence[str], table: Mapping[Tuple, Number], default: Union[Number, str] = "keep"):
        super().__init__(inputs)
        self.table = {tuple(int(v) for v in k): v for k, v in table.items()}
        self.default = default

    def next_value(self, parents, current, values):
        parents = np.atleast_2d(np.asarray(parents))
        out = np.asarray(current, dtype=float).copy() if self.default == "keep" else np.full(parents.shape[0], float(self.default))
        keys = [tuple(int(v) for v in row) for row in parents]
        for r, k in enumerate(keys):
            if k in self.table:
                out[r] = self.table[k]
        return out

    def params(self):
        return {"n_entries": len(self.table), "default": self.default}


class ClampRule(Rule):
    """Fix a node at a constant state (knockout or forced expression)."""

    kind = "clamp"

    def __init__(self, value: Number):
        super().__init__(())
        self.value = float(value)

    def next_value(self, parents, current, values):
        return np.full(np.asarray(current).shape[0], self.value)

    def params(self):
        return {"value": self.value}


class IdentityRule(Rule):
    """Keep the current state (an exogenous node that changes only by intervention)."""

    kind = "identity"

    def __init__(self):
        super().__init__(())

    def next_value(self, parents, current, values):
        return np.asarray(current, dtype=float).copy()


# ----------------------------------------------------------------------------- Blume–Capel transition rule
class SoftmaxRule(Rule):
    """Blume–Capel (polytomous) transition rule.

    For candidate next state :math:`s`, the utility is

    .. math::
        u(s) = \\beta\\,[\\, s\\,\\eta - \\delta s^2 + \\zeta\\,\\mathbb{1}(s=0)
               + \\rho\\,\\mathbb{1}(s = x_{\\text{current}}) \\,],

    with polarity field :math:`\\eta = h + \\sum_j w_j x_j` and zero-neighbour field
    :math:`\\zeta = \\sum_j \\lambda_j\\,\\mathbb{1}(x_j = 0)`, and
    :math:`P(s) \\propto \\exp u(s)`. For states :math:`\\{-1,0,+1\\}`,
    :math:`\\mathbb{1}(x=0) = 1 - x^2`, which is the zero–zero coordination term of the
    Blume–Capel probabilistic network model. With symmetric ``weights`` and ``lam``
    and ``rho = 0``, asynchronous updating with this rule is heat-bath (Gibbs)
    sampling of the equilibrium model. With ``beta = np.inf`` the rule is
    deterministic (arg-max; ties keep the current state, then prefer 0).
    """

    kind = "softmax"

    def __init__(self, weights: Optional[Mapping[str, float]] = None, lam: Optional[Mapping[str, float]] = None,
                 h: float = 0.0, delta: float = 0.0, rho: float = 0.0, beta: float = 1.0):
        weights = dict(weights or {})
        lam = dict(lam or {})
        names = list(weights) + [k for k in lam if k not in weights]
        super().__init__(names)
        self.w = np.array([float(weights.get(k, 0.0)) for k in names])
        self.lam = np.array([float(lam.get(k, 0.0)) for k in names])
        self.h, self.delta, self.rho, self.beta = float(h), float(delta), float(rho), float(beta)

    @property
    def deterministic(self) -> bool:  # type: ignore[override]
        return np.isinf(self.beta)

    def utilities(self, parents, current, values, scale: Optional[float] = None) -> np.ndarray:
        parents = np.atleast_2d(np.asarray(parents, dtype=float))
        values = np.asarray(values, dtype=float)
        R = parents.shape[0]
        eta = self.h + (parents @ self.w if self.w.size else 0.0)
        zeta = ((parents == 0) @ self.lam) if self.lam.size else 0.0
        eta = np.broadcast_to(eta, (R,))
        zeta = np.broadcast_to(zeta, (R,))
        cur = np.asarray(current, dtype=float)
        u = (values[None, :] * eta[:, None] - self.delta * values[None, :] ** 2
             + zeta[:, None] * (values[None, :] == 0) + self.rho * (values[None, :] == cur[:, None]))
        b = self.beta if scale is None else scale
        return b * u

    def probs(self, parents, current, values):
        if self.deterministic:
            return super().probs(parents, current, values)
        return softmax(self.utilities(parents, current, values))

    def next_value(self, parents, current, values):
        u = self.utilities(parents, current, values, scale=1.0)
        values = np.asarray(values, dtype=float)
        cur = np.asarray(current, dtype=float)
        best = u.max(axis=1, keepdims=True)
        is_max = np.isclose(u, best, rtol=0, atol=1e-12)
        # tie-break: current state, then zero, then lowest value
        pref = is_max * (1 + 4 * (values[None, :] == cur[:, None]) + 2 * (values[None, :] == 0))
        return values[np.argmax(pref, axis=1)]

    def with_beta(self, beta: float) -> "SoftmaxRule":
        r = self.copy()
        r.beta = float(beta)
        return r

    def params(self):
        return {"weights": self.w.tolist(), "lam": self.lam.tolist(), "h": self.h, "delta": self.delta,
                "rho": self.rho, "beta": self.beta}


# ----------------------------------------------------------------------------- stochastic wrappers
class NoisyRule(Rule):
    """With probability ``p`` the node takes a uniformly random state; otherwise ``base`` applies.

    This is the random-perturbation mechanism of probabilistic Boolean networks
    (Shmulevich, Dougherty, & Zhang, 2002) and makes every state reachable.
    """

    kind = "noisy"
    deterministic = False

    def __init__(self, base: Rule, p: float):
        super().__init__(base.inputs)
        if not 0 <= p <= 1:
            raise ValueError("p must lie in [0, 1].")
        self.base, self.p = base, float(p)

    def probs(self, parents, current, values):
        S = len(values)
        return (1 - self.p) * self.base.probs(parents, current, values) + self.p / S

    def params(self):
        d = dict(self.base.params())
        d["noise"] = self.p
        return d


class PBNRule(Rule):
    """Probabilistic Boolean (or multi-valued) network rule.

    At each update one of the ``components`` is chosen with the given selection
    probabilities (Shmulevich, Dougherty, Kim, & Zhang, 2002). Components may have
    different inputs; the rule's inputs are their union.
    """

    kind = "pbn"
    deterministic = False

    def __init__(self, components: Sequence[Rule], probabilities: Sequence[float]):
        p = np.asarray(probabilities, dtype=float)
        if len(components) != p.size or not np.isclose(p.sum(), 1):
            raise ValueError("Give one selection probability per component, summing to 1.")
        names: List[str] = []
        for c in components:
            for nm in c.inputs:
                if nm not in names:
                    names.append(nm)
        super().__init__(names)
        self.components, self.c = list(components), p
        self._idx = [[names.index(nm) for nm in comp.inputs] for comp in self.components]

    def probs(self, parents, current, values):
        parents = np.atleast_2d(np.asarray(parents, dtype=float))
        out = 0.0
        for comp, ck, idx in zip(self.components, self.c, self._idx):
            out = out + ck * comp.probs(parents[:, idx], current, values)
        return out

    def params(self):
        return {"n_components": len(self.components), "selection": self.c.tolist()}


class FixedInputRule(Rule):
    """Wrap a rule so that selected inputs are replaced by constants (edge knockout)."""

    kind = "fixed-input"

    def __init__(self, base: Rule, fixed: Mapping[str, Number]):
        keep = [nm for nm in base.inputs if nm not in fixed]
        super().__init__(keep)
        self.base, self.fixed = base, {k: float(v) for k, v in fixed.items()}

    @property
    def deterministic(self) -> bool:  # type: ignore[override]
        return self.base.deterministic

    def _full(self, parents):
        parents = np.atleast_2d(np.asarray(parents, dtype=float))
        R = parents.shape[0]
        cols = []
        k = 0
        for nm in self.base.inputs:
            if nm in self.fixed:
                cols.append(np.full(R, self.fixed[nm]))
            else:
                cols.append(parents[:, k]); k += 1
        return np.column_stack(cols) if cols else np.empty((R, 0))

    def next_value(self, parents, current, values):
        return self.base.next_value(self._full(parents), current, values)

    def probs(self, parents, current, values):
        return self.base.probs(self._full(parents), current, values)

    def params(self):
        d = dict(self.base.params())
        d["fixed_inputs"] = dict(self.fixed)
        return d


def deterministic_limit(rule: Rule) -> Rule:
    """Zero-noise limit of a rule: Softmax with ``beta=inf``; noise and PBN mixing removed.

    For a :class:`PBNRule` the component with the largest selection probability is kept.
    """
    if isinstance(rule, SoftmaxRule):
        return rule.with_beta(np.inf)
    if isinstance(rule, NoisyRule):
        return deterministic_limit(rule.base)
    if isinstance(rule, PBNRule):
        return deterministic_limit(rule.components[int(np.argmax(rule.c))])
    if isinstance(rule, FixedInputRule):
        return FixedInputRule(deterministic_limit(rule.base), rule.fixed)
    return rule.copy()
