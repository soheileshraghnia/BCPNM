"""Reporting helpers: model cards, draft methods/results text and the PNM reporting checklist.

The checklist adapts reporting items from STROBE (von Elm et al., 2007), PRISMA 2020
(Page et al., 2021) and the ODD protocol for simulation models (Grimm et al., 2020)
to probabilistic network models. :func:`write_report` produces a Markdown file with
the model card, analysis settings, software environment and the completed checklist,
which can be attached as supplementary material.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from .core import PNMGraph
from .utils import environment_report

# (section, item, what to report, adapted from)
CHECKLIST: List[tuple] = [
    ("Purpose", "1. Research question", "State the question, the target system and whether the model is explanatory, predictive or exploratory.", "STROBE 2-3; ODD 1"),
    ("Purpose", "2. Level of analysis", "Say whether nodes are persons (interpersonal network) or constructs (intrapersonal network) and define the population or setting the model refers to.", "STROBE 5-6; ODD 2"),
    ("Specification", "3. Nodes", "List every node with its construct, operational definition and, where available, its measure.", "STROBE 7-8; ODD 2"),
    ("Specification", "4. State spaces", "Give each node's admissible states and labels, and justify the number of states (Boolean, ternary, ordinal).", "STROBE 11; ODD 2"),
    ("Specification", "5. Update rules", "Report every rule (logical expression, threshold, table or parameterized transition rule) and its source: theory, literature, expert elicitation or estimation.", "ODD 7"),
    ("Specification", "6. Parameters", "Tabulate all parameter values, their scale and justification; state which were fixed, elicited or estimated.", "ODD 6-7"),
    ("Specification", "7. Update scheme and time", "State the update scheme (synchronous, asynchronous, sequential) and what one update step represents in real time.", "ODD 3"),
    ("Specification", "8. Initial conditions", "Describe how initial states were set (observed, theoretically motivated, random).", "ODD 5"),
    ("Analysis", "9. Attractor analysis", "Report the search method (exhaustive or sampling), the size of the state space, the number of starts and the treatment of unresolved runs.", "STROBE 12"),
    ("Analysis", "10. Stochastic simulation", "Report runs, steps, burn-in, random seeds and Monte Carlo error of reported proportions.", "STROBE 10, 12"),
    ("Analysis", "11. Perturbations", "Define each intervention: type (clamp, edge removal, field shift), target, timing and duration.", "ODD 7"),
    ("Analysis", "12. Sensitivity analysis", "State which parameters and rules were varied, over which ranges, and the method (one-at-a-time, grid, Derrida curve).", "STROBE 12e; PRISMA 13f"),
    ("Analysis", "13. Inference from data", "If rules or parameters were estimated: data source, preprocessing and discretization, estimator, penalty, selection criterion and held-out evaluation.", "STROBE 8, 12; PRISMA 13"),
    ("Validity", "14. Content validity", "Describe expert or stakeholder review of nodes and rules and report agreement (e.g., sign agreement).", "-"),
    ("Validity", "15. Construct validity", "Show that node states correspond to the intended constructs (measurement evidence, cut-offs for discretization).", "STROBE 8"),
    ("Validity", "16. Convergent validity", "Compare model output with independent empirical patterns or alternative models (pattern-oriented modeling).", "ODD 1"),
    ("Validity", "17. Predictive validity", "Report out-of-sample predictive performance against simple benchmarks (e.g., log score versus an independence or persistence model).", "-"),
    ("Results", "18. Main results", "Report attractors with basin shares, stationary probabilities or intervention effects with their uncertainty.", "STROBE 16-17"),
    ("Results", "19. Limitations", "Discuss discretization, time-scale, homogeneity and closure assumptions and their likely effect.", "STROBE 19-21"),
    ("Reproducibility", "20. Software", "Name the software and versions (bcpnm, Python, NumPy, SciPy) and the operating system.", "PRISMA 27"),
    ("Reproducibility", "21. Code, configuration and data", "Share configuration files, analysis scripts and data (or synthetic equivalents) in a public repository with a persistent identifier.", "PRISMA 27"),
    ("Ethics", "22. Responsible use", "State the intended use and warn against unsupported uses (e.g., individual risk prediction from a theoretical model).", "STROBE 22"),
]


def checklist(as_frame: bool = True):
    """The PNM reporting checklist as a ``pandas.DataFrame`` (or list of dicts)."""
    rows = [{"section": s, "item": i, "report": r, "adapted_from": a} for s, i, r, a in CHECKLIST]
    if not as_frame:
        return rows
    import pandas as pd

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- model cards
def _fmt(v) -> str:
    if isinstance(v, float):
        if np.isinf(v):
            return "inf"
        return f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def node_table(graph: PNMGraph):
    """Nodes, state spaces and constructs as a ``pandas.DataFrame``."""
    import pandas as pd

    rows = []
    for n in graph.nodes.values():
        rows.append({"node": n.name, "states": ", ".join(f"{v} ({n.states.labels[v]})" for v in n.states.values),
                     "construct": n.construct, "measurement": n.measurement})
    return pd.DataFrame(rows)


def rule_table(graph: PNMGraph):
    """Rules, inputs and parameters as a ``pandas.DataFrame``."""
    import pandas as pd

    rows = []
    for n, r in graph.rules.items():
        p = r.params()
        if "weights" in p:
            p = dict(p)
            p["weights"] = {k: round(float(w), 3) for k, w in zip(r.inputs, p["weights"])}
            if "lam" in p:
                p["lam"] = {k: round(float(w), 3) for k, w in zip(r.inputs, p["lam"]) if w}
        rows.append({"node": n, "rule": r.__class__.__name__, "inputs": ", ".join(r.inputs),
                     "parameters": "; ".join(f"{k}={_fmt(v)}" for k, v in p.items())})
    return pd.DataFrame(rows)


def _md_table(df) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def model_card(graph: PNMGraph) -> str:
    """Markdown description of a model (nodes, state spaces, rules)."""
    lines = [f"## Model: {graph.name}", ""]
    if graph.meta:
        for k, v in graph.meta.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines += [f"{graph.N} nodes, {len(graph.edges())} directed influences, "
              f"{graph.state_space_size():,} configurations; "
              f"{'deterministic' if graph.deterministic else 'stochastic'} rules.", "",
              "### Nodes", "", _md_table(node_table(graph)), "", "### Rules", "", _md_table(rule_table(graph)), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- text drafts
def methods_text(graph: PNMGraph, scheme: str = "asynchronous", runs: Optional[int] = None,
                 steps: Optional[int] = None, seed: Optional[int] = None, extra: str = "") -> str:
    """Draft a methods paragraph describing the model and simulation settings.

    The text is a starting point: authors should edit it and add the substantive
    justification of nodes and rules.
    """
    from . import __version__

    kinds = sorted({r.__class__.__name__ for r in graph.rules.values()})
    sizes = sorted({n.states.size for n in graph.nodes.values()})
    st = " or ".join(str(s) for s in sizes)
    txt = (f"We specified a probabilistic network model ('{graph.name}') with {graph.N} nodes, each taking "
           f"{st} ordered states, and {len(graph.edges())} directed influences "
           f"({graph.state_space_size():,} possible configurations). Update rules were of type "
           f"{', '.join(kinds)}. ")
    sch = {"synchronous": "synchronously (all nodes updated at each step)",
           "asynchronous": "asynchronously (one randomly chosen node updated per step)",
           "sequential": "sequentially (all nodes once per step in random order)"}.get(scheme, scheme)
    txt += f"Nodes were updated {sch}. "
    if runs and steps:
        txt += f"We simulated {runs:,} independent runs of {steps:,} steps"
        txt += f" (random seed {seed}). " if seed is not None else ". "
    txt += (f"Analyses used bcpnm {__version__} (Python {environment_report(())['python']}). " + extra).strip()
    return txt


def results_text(attractors=None, stationary=None, value_labels: Optional[Mapping[int, str]] = None) -> str:
    """Draft a results sentence from an :class:`~bcpnm.attractors.AttractorResult` and/or a
    :class:`~bcpnm.attractors.StationaryResult`."""
    parts = []
    if attractors is not None:
        k = len(attractors)
        fp = sum(a.kind == "fixed point" for a in attractors)
        desc = []
        for a in attractors.attractors[:5]:
            lab = a.label or ("fixed point" if a.kind == "fixed point" else f"{a.kind} of length {a.size}")
            desc.append(f"{lab} ({a.basin_fraction:.1%})")
        parts.append(f"Under {attractors.scheme} updating the network had {k} attractor{'s' if k != 1 else ''} "
                     f"({fp} fixed point{'s' if fp != 1 else ''}); basin shares: " + "; ".join(desc) + ".")
    if stationary is not None:
        top = stationary.top(1)[0]
        cfg = ", ".join(f"{n}={v:+d}" if v else f"{n}=0" for n, v in top[0].items())
        parts.append(f"The most probable configuration in the stationary distribution was ({cfg}) "
                     f"with probability {top[1]:.3f}.")
    return " ".join(parts)


def write_report(path, graph: PNMGraph, config: Optional[dict] = None, sections: Optional[Mapping[str, str]] = None,
                 completed: Optional[Mapping[str, str]] = None) -> Path:
    """Write a Markdown report: model card, extra sections, environment and checklist.

    Parameters
    ----------
    path : output file (``.md``)
    graph : the model
    config : the configuration used (embedded as JSON for reproducibility)
    sections : mapping of heading -> Markdown text (e.g. results)
    completed : mapping of checklist item number (e.g. "5") -> where/how it is reported
    """
    path = Path(path)
    env = environment_report()
    lines = [f"# bcpnm report: {graph.name}", "",
             f"Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} with bcpnm {env['bcpnm']}.", "",
             model_card(graph)]
    for head, body in (sections or {}).items():
        lines += [f"## {head}", "", body, ""]
    lines += ["## Software environment", "", _md_table(_env_frame(env)), ""]
    if config is not None:
        lines += ["## Configuration", "", "```json", json.dumps(config, indent=2, default=str), "```", ""]
    ck = checklist()
    done = {str(k): v for k, v in (completed or {}).items()}
    ck["reported"] = [done.get(i.split(".")[0], "") for i in ck["item"]]
    lines += ["## PNM reporting checklist", "", _md_table(ck), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _env_frame(env: Mapping[str, str]):
    import pandas as pd

    return pd.DataFrame({"component": list(env), "version": list(env.values())})
