"""Core data structures: :class:`StateSpace`, :class:`Node`, :class:`Rule` and :class:`PNMGraph`.

A model is a set of nodes (psychological constructs or persons), each with a finite,
ordered set of integer-coded states, and one update rule per node. A rule maps the
current values of the node's inputs (and the node's own current value) to a
probability distribution over the node's next state. Deterministic rules (Boolean,
threshold, table) put all probability on one state.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import decode_indices, encode_indices, make_rng, sample_categorical


class StateSpace:
    """An ordered, finite set of integer-coded states with optional labels.

    Parameters
    ----------
    values : sequence of int
        The admissible states in increasing order, e.g. ``(-1, 0, 1)``.
    labels : mapping, optional
        Human-readable label per value, e.g. ``{-1: "conceal", 0: "silent", 1: "acknowledge"}``.
    name : str, optional
        Name used in configuration files and reports.
    """

    def __init__(self, values: Sequence[int], labels: Optional[Mapping[int, str]] = None, name: str = ""):
        vals = tuple(int(v) for v in values)
        if len(vals) < 2:
            raise ValueError("A state space needs at least two states.")
        if list(vals) != sorted(set(vals)):
            raise ValueError("State values must be unique and sorted in increasing order.")
        self.values: Tuple[int, ...] = vals
        self.labels: Dict[int, str] = {v: str(v) for v in vals}
        if labels:
            for k, v in labels.items():
                if int(k) not in vals:
                    raise ValueError(f"Label given for unknown state {k!r}.")
                self.labels[int(k)] = str(v)
        self.name = name

    # convenience constructors -------------------------------------------------
    @classmethod
    def boolean(cls, off: str = "off", on: str = "on") -> "StateSpace":
        """States ``(0, 1)``."""
        return cls((0, 1), {0: off, 1: on}, name="boolean")

    @classmethod
    def ternary(cls, labels: Sequence[str] = ("negative", "neutral", "positive")) -> "StateSpace":
        """Blume–Capel states ``(-1, 0, +1)``."""
        return cls((-1, 0, 1), dict(zip((-1, 0, 1), labels)), name="ternary")

    @classmethod
    def ordinal(cls, k: int, labels: Optional[Sequence[str]] = None) -> "StateSpace":
        """States ``(0, 1, ..., k-1)``."""
        lab = dict(zip(range(k), labels)) if labels else None
        return cls(range(k), lab, name=f"ordinal{k}")

    # basic API ---------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def index(self, value) -> np.ndarray:
        """Position(s) of state value(s) in the ordered state space."""
        arr = np.asarray(value)
        lut = {v: i for i, v in enumerate(self.values)}
        try:
            return np.vectorize(lambda x: lut[int(x)], otypes=[np.int64])(arr)
        except KeyError as e:  # pragma: no cover - defensive
            raise ValueError(f"Value {e} is not in state space {self.values}.") from None

    def __contains__(self, value) -> bool:
        return int(value) in self.values

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        lab = ", ".join(f"{v}={self.labels[v]}" for v in self.values)
        return f"StateSpace({lab})"

    def to_dict(self) -> dict:
        return {"values": list(self.values), "labels": {int(k): v for k, v in self.labels.items()}}


@dataclass
class Node:
    """A node in a :class:`PNMGraph`.

    ``construct`` and ``measurement`` document what the node represents and how
    its state would be measured; they are carried into generated reports.
    """

    name: str
    states: StateSpace
    description: str = ""
    construct: str = ""
    measurement: str = ""
    initial: Optional[int] = None

    def __post_init__(self):
        if self.initial is not None and self.initial not in self.states:
            raise ValueError(f"Initial state {self.initial} of node {self.name} is not admissible.")


class Rule:
    """Base class for update rules.

    Subclasses implement either :meth:`next_value` (deterministic rules) or
    :meth:`probs` (stochastic rules). All methods are vectorized over ``R``
    replicates: ``parents`` has shape ``(R, n_inputs)``, ``current`` has shape
    ``(R,)`` and ``values`` is the node's ordered state values ``(S,)``.
    """

    deterministic: bool = True
    kind: str = "rule"

    def __init__(self, inputs: Iterable[str] = ()):
        self.inputs: Tuple[str, ...] = tuple(inputs)

    # to be overridden ---------------------------------------------------------
    def next_value(self, parents: np.ndarray, current: np.ndarray, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def probs(self, parents: np.ndarray, current: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Probability of each next state (R x S)."""
        nxt = self.next_value(parents, current, values)
        return (np.asarray(nxt)[:, None] == np.asarray(values)[None, :]).astype(float)

    # shared ----------------------------------------------------------------
    def sample(self, parents, current, values, rng: np.random.Generator) -> np.ndarray:
        if self.deterministic:
            return np.asarray(self.next_value(parents, current, values))
        p = self.probs(parents, current, values)
        return np.asarray(values)[sample_categorical(p, rng)]

    def params(self) -> dict:
        """Parameters for reporting and configuration export."""
        return {}

    def copy(self) -> "Rule":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        inp = ", ".join(self.inputs)
        return f"{self.__class__.__name__}(inputs=[{inp}])"


class PNMGraph:
    """A polytomous network model: nodes, state spaces and one update rule per node.

    Parameters
    ----------
    name : str
        Model name used in reports.
    meta : dict, optional
        Free-form metadata (domain, version, sources of rules, seed).

    Examples
    --------
    >>> from bcpnm import PNMGraph, StateSpace
    >>> from bcpnm.rules import BooleanRule
    >>> g = PNMGraph("toggle")
    >>> g.add_node("A", StateSpace.boolean()); g.add_node("B", StateSpace.boolean())
    >>> g.set_rule("A", BooleanRule("not B")); g.set_rule("B", BooleanRule("not A"))
    >>> g.N
    2
    """

    def __init__(self, name: str = "model", meta: Optional[dict] = None):
        self.name = name
        self.meta: dict = dict(meta or {})
        self.nodes: Dict[str, Node] = {}
        self.rules: Dict[str, Rule] = {}

    # construction -------------------------------------------------------------
    def add_node(self, name: str, states: StateSpace, **kwargs) -> Node:
        if name in self.nodes:
            raise ValueError(f"Node {name!r} already exists.")
        node = Node(name, states, **kwargs)
        self.nodes[name] = node
        return node

    def set_rule(self, node: str, rule: Rule) -> None:
        if node not in self.nodes:
            raise KeyError(f"Unknown node {node!r}.")
        self.rules[node] = rule

    # structure ---------------------------------------------------------------
    @property
    def names(self) -> List[str]:
        return list(self.nodes)

    @property
    def N(self) -> int:
        return len(self.nodes)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def sizes(self) -> List[int]:
        return [n.states.size for n in self.nodes.values()]

    def state_space_size(self) -> int:
        return int(np.prod(self.sizes(), dtype=np.int64))

    def parents(self, node: str) -> Tuple[str, ...]:
        return self.rules[node].inputs

    def edges(self) -> List[Tuple[str, str, dict]]:
        """Directed edges ``(source, target, attributes)`` implied by the rules."""
        out = []
        for tgt, rule in self.rules.items():
            par = rule.params()
            w = par.get("weights")
            lam = par.get("lam")
            for k, src in enumerate(rule.inputs):
                attrs = {"rule": rule.kind}
                if w is not None:
                    attrs["weight"] = float(w[k])
                if lam is not None:
                    attrs["lam"] = float(lam[k])
                out.append((src, tgt, attrs))
        return out

    def to_networkx(self):
        """Return a ``networkx.DiGraph`` with node metadata and edge attributes."""
        import networkx as nx

        G = nx.DiGraph(name=self.name)
        for n in self.nodes.values():
            G.add_node(n.name, construct=n.construct, states=n.states.values)
        for s, t, a in self.edges():
            G.add_edge(s, t, **a)
        return G

    def validate(self) -> None:
        """Check that every node has a rule and every rule input is a node."""
        missing = [n for n in self.nodes if n not in self.rules]
        if missing:
            raise ValueError(f"Nodes without a rule: {missing}")
        for n, r in self.rules.items():
            bad = [i for i in r.inputs if i not in self.nodes]
            if bad:
                raise ValueError(f"Rule of {n!r} uses unknown inputs {bad}")

    @property
    def deterministic(self) -> bool:
        return all(r.deterministic for r in self.rules.values())

    # state handling ----------------------------------------------------------
    def value_arrays(self) -> List[np.ndarray]:
        return [n.states.array for n in self.nodes.values()]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode state vectors (R x N values) as integers."""
        X = np.atleast_2d(np.asarray(X))
        idx = np.column_stack([self.nodes[n].states.index(X[:, i]) for i, n in enumerate(self.names)])
        return encode_indices(idx, self.sizes())

    def decode(self, codes) -> np.ndarray:
        """Decode integers into state vectors (R x N values)."""
        idx = decode_indices(codes, self.sizes())
        vals = self.value_arrays()
        return np.column_stack([vals[i][idx[:, i]] for i in range(self.N)])

    def all_states(self) -> np.ndarray:
        """Every configuration of the network (use only for small state spaces)."""
        S = self.state_space_size()
        if S > 5_000_000:
            raise MemoryError(f"State space has {S} configurations; use sampling methods instead.")
        return self.decode(np.arange(S))

    def random_states(self, n: int, rng=None) -> np.ndarray:
        rng = make_rng(rng)
        cols = [rng.choice(v, size=n) for v in self.value_arrays()]
        return np.column_stack(cols)

    def initial_state(self, rng=None) -> np.ndarray:
        """Initial states from node metadata; nodes without ``initial`` are drawn at random."""
        rng = make_rng(rng)
        x = self.random_states(1, rng)[0]
        for i, n in enumerate(self.nodes.values()):
            if n.initial is not None:
                x[i] = n.initial
        return x

    def as_dict(self, x: Sequence) -> Dict[str, int]:
        return {n: int(v) for n, v in zip(self.names, x)}

    def from_dict(self, d: Mapping[str, int]) -> np.ndarray:
        return np.array([d[n] for n in self.names], dtype=float)

    # misc --------------------------------------------------------------------
    def copy(self) -> "PNMGraph":
        return copy.deepcopy(self)

    def summary(self) -> str:
        lines = [f"PNMGraph '{self.name}': {self.N} nodes, {len(self.edges())} directed edges, "
                 f"{self.state_space_size():,} configurations"]
        for n in self.nodes.values():
            r = self.rules.get(n.name)
            rs = r.__class__.__name__ if r else "no rule"
            inp = ", ".join(r.inputs) if r else ""
            lines.append(f"  {n.name:<14} states={n.states.values}  rule={rs}({inp})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PNMGraph(name={self.name!r}, N={self.N})"
