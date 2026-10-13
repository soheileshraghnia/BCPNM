"""Configuration files (YAML/JSON) and data import/export.

A configuration fully specifies a model so that it can be shared, versioned and
re-run. Minimal example (YAML)::

    meta: {name: toy, domain: family communication, seed: 1}
    state_spaces:
      talk: {values: [-1, 0, 1], labels: {-1: conceal, 0: silent, 1: acknowledge}}
    nodes:
      - {name: A, states: talk, construct: "parent's communication"}
      - {name: B, states: talk}
    model:
      type: softmax          # softmax | threshold | logical | table
      beta: 1.0
      defaults: {delta: 0.3, h: 0.0, rho: 0.0}
      edges:                 # weights w and zero–zero terms lam, source -> target
        - {source: A, target: B, w: 0.8, lam: 0.5}
      symmetric: true        # also add target -> source with the same values
    dynamics: {scheme: asynchronous, steps: 200, runs: 500}

See ``examples/*.yaml`` for complete Boolean, threshold and Blume–Capel models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .core import PNMGraph, StateSpace
from .rules import BooleanRule, CaseRule, ClampRule, IdentityRule, NoisyRule, SoftmaxRule, TableRule, ThresholdRule

PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    """Read a YAML or JSON configuration file into a dictionary."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def save_config(cfg: Dict[str, Any], path: PathLike) -> None:
    """Write a configuration dictionary to YAML (``.yaml``/``.yml``) or JSON."""
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _state_space(spec, name: str) -> StateSpace:
    if isinstance(spec, str):
        presets = {"boolean": StateSpace.boolean(), "ternary": StateSpace.ternary()}
        if spec in presets:
            return presets[spec]
        raise ValueError(f"Unknown state-space preset {spec!r}.")
    labels = {int(k): v for k, v in (spec.get("labels") or {}).items()}
    return StateSpace(spec["values"], labels, name=name)


def _per_node(value, node: str, default=0.0):
    if isinstance(value, dict):
        return float(value.get(node, default))
    return float(default if value is None else value)


def _rule_from_spec(spec: Dict[str, Any]):
    t = spec["type"]
    if t in ("boolean", "logical") and "expr" in spec:
        r = BooleanRule(spec["expr"], spec.get("true_value", 1), spec.get("false_value", 0))
    elif t in ("case", "logical"):
        r = CaseRule([tuple(c) for c in spec["cases"]], spec.get("default", "keep"))
    elif t == "threshold":
        r = ThresholdRule(spec["weights"], spec.get("thresholds", [0.0]), spec.get("outputs", [0, 1]),
                          spec.get("bias", 0.0), spec.get("tie", "keep"))
    elif t == "table":
        table = {tuple(int(v) for v in str(k).split(",")): v for k, v in spec["table"].items()}
        r = TableRule(spec["inputs"], table, spec.get("default", "keep"))
    elif t == "clamp":
        r = ClampRule(spec["value"])
    elif t == "identity":
        r = IdentityRule()
    elif t == "softmax":
        r = SoftmaxRule(spec.get("weights"), spec.get("lam"), spec.get("h", 0.0), spec.get("delta", 0.0),
                        spec.get("rho", 0.0), spec.get("beta", 1.0))
    else:
        raise ValueError(f"Unknown rule type {t!r}.")
    if spec.get("noise"):
        r = NoisyRule(r, float(spec["noise"]))
    return r


def build_model(cfg: Dict[str, Any]) -> PNMGraph:
    """Construct a :class:`PNMGraph` from a configuration dictionary.

    Two styles are supported and can be mixed:

    * ``model.type: softmax`` with an ``edges`` list builds Blume–Capel transition rules
      for every node (``symmetric: true`` makes the network undirected, i.e. an
      equilibrium Blume–Capel model under asynchronous updating);
    * a ``rules`` mapping gives an explicit rule per node (``boolean``, ``case``,
      ``threshold``, ``table``, ``clamp``, ``identity`` or ``softmax``), overriding the
      edge-based rule for that node.
    """
    meta = dict(cfg.get("meta", {}))
    g = PNMGraph(meta.get("name", "model"), meta=meta)
    spaces = {k: _state_space(v, k) for k, v in (cfg.get("state_spaces") or {}).items()}
    for nd in cfg["nodes"]:
        ss = nd.get("states", "ternary")
        ss = spaces[ss] if isinstance(ss, str) and ss in spaces else _state_space(ss, nd["name"])
        g.add_node(nd["name"], ss, description=nd.get("description", ""), construct=nd.get("construct", ""),
                   measurement=nd.get("measurement", ""), initial=nd.get("initial"))
    model = cfg.get("model") or {}
    if model.get("type") == "softmax":
        defaults = model.get("defaults", {})
        params = model.get("parameters", {})
        W = {n: {} for n in g.names}
        L = {n: {} for n in g.names}
        for e in model.get("edges", []):
            s, t = e["source"], e["target"]
            pairs = [(s, t)] + ([(t, s)] if model.get("symmetric", False) else [])
            for a, b in pairs:
                W[b][a] = float(e.get("w", 0.0))
                L[b][a] = float(e.get("lam", 0.0))
        for n in g.names:
            g.set_rule(n, SoftmaxRule(
                W[n], L[n],
                h=_per_node(params.get("h"), n, defaults.get("h", 0.0)),
                delta=_per_node(params.get("delta"), n, defaults.get("delta", 0.0)),
                rho=_per_node(params.get("rho"), n, defaults.get("rho", 0.0)),
                beta=float(model.get("beta", 1.0))))
            if model.get("noise"):
                g.set_rule(n, NoisyRule(g.rules[n], float(model["noise"])))
    for n, spec in (cfg.get("rules") or {}).items():
        g.set_rule(n, _rule_from_spec(spec))
    g.validate()
    return g


def load_model(path: PathLike) -> Tuple[PNMGraph, Dict[str, Any]]:
    """Load a configuration file and build its model. Returns ``(graph, config)``."""
    cfg = load_config(path)
    return build_model(cfg), cfg


def graph_to_config(graph: PNMGraph) -> Dict[str, Any]:
    """Export a graph to a configuration dictionary (rules written explicitly per node)."""
    spaces, nodes = {}, []
    for n in graph.nodes.values():
        key = n.states.name or f"{n.name}_states"
        spaces[key] = n.states.to_dict()
        nodes.append({"name": n.name, "states": key, "construct": n.construct,
                      "measurement": n.measurement, "description": n.description})
    rules = {}
    for n, r in graph.rules.items():
        p = r.params()
        if isinstance(r, SoftmaxRule):
            rules[n] = {"type": "softmax", "weights": dict(zip(r.inputs, r.w.tolist())),
                        "lam": dict(zip(r.inputs, r.lam.tolist())), "h": r.h, "delta": r.delta,
                        "rho": r.rho, "beta": r.beta if np.isfinite(r.beta) else ".inf"}
        elif isinstance(r, ThresholdRule):
            rules[n] = {"type": "threshold", "weights": dict(zip(r.inputs, r.w.tolist())), **{k: p[k] for k in ("bias", "thresholds", "outputs", "tie")}}
        elif isinstance(r, BooleanRule):
            rules[n] = {"type": "boolean", "expr": r.expr, "true_value": p["true_value"], "false_value": p["false_value"]}
        elif isinstance(r, CaseRule):
            rules[n] = {"type": "case", "cases": p["cases"], "default": p["default"]}
        elif isinstance(r, ClampRule):
            rules[n] = {"type": "clamp", "value": r.value}
        elif isinstance(r, IdentityRule):
            rules[n] = {"type": "identity"}
        else:
            rules[n] = {"type": r.kind, "note": "not exportable; rebuild in code"}
    return {"meta": dict(graph.meta, name=graph.name), "state_spaces": spaces, "nodes": nodes, "rules": rules}


def save_trajectory(traj, path: PathLike) -> None:
    """Write a :class:`~bcpnm.dynamics.Trajectory` in long CSV format."""
    traj.to_long().to_csv(path, index=False)


def read_panel(path: PathLike, names=None) -> np.ndarray:
    """Read a wide CSV (rows = measurement occasions, columns = nodes) into a T x N array."""
    import pandas as pd

    df = pd.read_csv(path)
    if names is not None:
        df = df[list(names)]
    return df.to_numpy(dtype=float)
