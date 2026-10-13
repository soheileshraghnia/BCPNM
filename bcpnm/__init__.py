"""bcpnm: Blume–Capel probabilistic network modeling for psychological systems.

One framework for discrete-state network models of psychological systems:

* Boolean and multi-valued logical networks (deterministic rules);
* probabilistic Boolean networks (random rule selection or perturbation);
* the Blume–Capel probabilistic network model (BC PNM) with states -1, 0, +1,
  in its equilibrium (symmetric) and directed (transition) forms.

Logical networks are the zero-noise limit (``beta -> inf``) of the probabilistic
rules, so the same attractor, basin, perturbation and sensitivity tools apply to all.

Quick start
-----------
>>> import bcpnm as bc
>>> g, cfg = bc.load_model("examples/01_conspiracy_of_silence.yaml")   # doctest: +SKIP
>>> traj = bc.simulate(g, steps=200, runs=500, scheme="asynchronous", seed=1)  # doctest: +SKIP
"""
__version__ = "0.1.0"

from .core import Node, PNMGraph, Rule, StateSpace
from .rules import (AND, NOT, OR, BooleanRule, CaseRule, ClampRule, FixedInputRule, IdentityRule, NoisyRule,
                    PBNRule, SoftmaxRule, TableRule, ThresholdRule, deterministic_limit, majority)
from .dynamics import Intervention, Trajectory, simulate, step, successors
from .attractors import (Attractor, AttractorResult, StationaryResult, deterministic_graph, find_attractors,
                         quasi_attractors, stationary_distribution, transition_matrix)
from .equilibrium import BlumeCapelModel, continuation, mean_field, mean_field_global, phase_diagram
from .perturbation import (Clamp, Combined, FlipEdge, RemoveEdge, ScaleEdge, SetParameter, ShiftField,
                           compare_attractors, intervention_effect, knockout, leverage_scan, overexpress)
from .inference import fit_equilibrium, fit_transition_model, log_score, reveal
from .io import build_model, graph_to_config, load_config, load_model, save_config
from .validation import (attractor_robustness, average_sensitivity, canalizing_inputs, derrida_curve,
                         expert_consistency, input_activities, oat_sensitivity, parameter_recovery, threshold_ties)
from .reporting import checklist, methods_text, model_card, results_text, write_report
from .utils import environment_report, make_rng

__all__ = [
    "__version__",
    # core
    "StateSpace", "Node", "Rule", "PNMGraph",
    # rules
    "BooleanRule", "CaseRule", "ThresholdRule", "TableRule", "ClampRule", "IdentityRule", "SoftmaxRule",
    "NoisyRule", "PBNRule", "FixedInputRule", "AND", "OR", "NOT", "majority", "deterministic_limit",
    # dynamics
    "simulate", "step", "successors", "Intervention", "Trajectory",
    # attractors
    "find_attractors", "deterministic_graph", "stationary_distribution", "transition_matrix",
    "quasi_attractors", "Attractor", "AttractorResult", "StationaryResult",
    # equilibrium
    "BlumeCapelModel", "mean_field", "mean_field_global", "continuation", "phase_diagram",
    # perturbation
    "Clamp", "knockout", "overexpress", "RemoveEdge", "ScaleEdge", "FlipEdge", "ShiftField", "SetParameter",
    "Combined", "compare_attractors", "leverage_scan", "intervention_effect",
    # inference
    "reveal", "fit_transition_model", "fit_equilibrium", "log_score",
    # io
    "load_config", "save_config", "build_model", "load_model", "graph_to_config",
    # validation
    "expert_consistency", "input_activities", "average_sensitivity", "canalizing_inputs", "derrida_curve",
    "attractor_robustness", "parameter_recovery", "oat_sensitivity", "threshold_ties",
    # reporting
    "checklist", "methods_text", "results_text", "model_card", "write_report",
    # utils
    "make_rng", "environment_report",
]
