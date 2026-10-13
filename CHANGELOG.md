# Changelog

All notable changes are recorded here. The project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-12

First public release.

- Core data structures: `StateSpace`, `Node`, `Rule`, `PNMGraph`.
- Rules: Boolean, multi-valued case, threshold, lookup table, clamp, identity; the Blume–Capel
  `SoftmaxRule`; probabilistic Boolean network (`PBNRule`) and random-perturbation (`NoisyRule`) wrappers.
- Synchronous, general asynchronous and sequential updating with time-windowed interventions.
- Attractors and basins (synchronous and asynchronous), exact stationary distributions, quasi-attractors.
- Equilibrium Blume–Capel model: exact enumeration, Gibbs sampling, mean-field theory, hysteresis sweeps.
- Perturbations (clamp, edge removal/scaling/reversal, field shift) and leverage scans.
- Inference: REVEAL-style lookup tables, penalized likelihood for directed and equilibrium Blume–Capel models.
- Validation: expert consistency, input activity and sensitivity, canalization, Derrida curves, robustness,
  threshold ties, parameter recovery, one-at-a-time sensitivity.
- Reporting: model cards, draft methods/results text, PNM reporting checklist.
- A configuration template, three worked examples, two tutorial notebooks and an executable tutorial paper.
