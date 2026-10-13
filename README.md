# bcpnm — Blume–Capel probabilistic network modeling for psychological systems

`bcpnm` is an open-source Python package for building, simulating and analyzing
**discrete-state network models of psychological systems**: symptom networks, families,
teams and organizations, or networks of beliefs and attitudes. Nodes take a small number of
ordered states, such as *conceal / silent / acknowledge*, *absent / present* or
*low / moderate / high*, and update according to logical, threshold or probabilistic rules.

The package accompanies the tutorial paper *Blume–Capel Probabilistic Network Modeling
(BC PNM) for Psychological Systems: A Tutorial and Open-Source Python Package for Dynamic
Conspiracy-of-Silence and Beyond* (`paper/tutorial_paper.md`).

## One framework, many models

A single transition rule (the Blume–Capel `SoftmaxRule`) controlled by a concentration
parameter β spans the model family, so the same attractor, perturbation and sensitivity
tools apply to all of them:

| | deterministic (β = ∞) | stochastic (finite β) |
|---|---|---|
| **directed influence** | Boolean and multi-valued logical networks | probabilistic Boolean networks; directed BC transition networks |
| **symmetric influence** | ternary threshold (Hopfield-type) networks | equilibrium Blume–Capel model (Gibbs sampling) |

The neutral state of the Blume–Capel model gives silence, abstention and ambivalence a formal
home; its zero–zero coordination term Λ captures the mutual reinforcement of silence.

## Installation

```bash
git clone https://github.com/YOUR-ORG/bcpnm.git
cd bcpnm
pip install -r requirements.txt   # optional: exact versions used in the paper
pip install -e .                  # or: pip install -e ".[dev]" for tests and notebooks
pytest
```

Requires Python ≥ 3.9 with NumPy, SciPy, pandas, NetworkX, Matplotlib and PyYAML.

## Quick start

```python
import numpy as np
import bcpnm as bc

# a model from a configuration file
fam, cfg = bc.load_model("examples/01_conspiracy_of_silence.yaml")
st = bc.stationary_distribution(fam)                     # exact long-run distribution
print(st.marginal("Adolescent"))                         # {-1: 0.258, 0: 0.576, 1: 0.166}
print(bc.quasi_attractors(fam, st).summary())            # regimes and time spent in each

# a Boolean model in code
g = bc.PNMGraph("toggle")
for n in ("A", "B"):
    g.add_node(n, bc.StateSpace.boolean())
g.set_rule("A", bc.NOT("B"))
g.set_rule("B", bc.NOT("A"))
print(bc.find_attractors(g, scheme="asynchronous").summary())

# an intervention
gate = bc.Clamp("Friend", 1).apply(fam)
print(bc.stationary_distribution(gate).marginal("Adolescent"))
```

## Features

* **Specification**: `StateSpace`, `PNMGraph`, YAML/JSON configuration files
  (`load_model`, `save_config`), rule types `BooleanRule`, `CaseRule`, `ThresholdRule`,
  `TableRule`, `ClampRule`, `SoftmaxRule`, `NoisyRule`, `PBNRule`, and custom `Rule` subclasses.
* **Dynamics**: synchronous, asynchronous and sequential updating; time-windowed
  interventions; vectorized replicate runs with explicit seeds.
* **Long-run behavior**: exhaustive or sampled attractor search (fixed points, cycles,
  asynchronous complex attractors; weak and strong basins), exact stationary distributions,
  quasi-attractors.
* **Equilibrium Blume–Capel model**: exact enumeration, Gibbs sampling, conditional
  log-odds, Blume–Emery–Griffiths form, mean-field equations, hysteresis sweeps, phase diagrams.
* **Perturbation**: clamps, edge removal/scaling/reversal, field shifts, parameter changes,
  leverage scans, intervention effects.
* **Inference**: REVEAL input sets and lookup tables; penalized likelihood for directed and
  equilibrium Blume–Capel models; held-out log scores; parameter recovery.
* **Validation**: expert sign agreement, input activities and sensitivity, canalizing
  inputs, Derrida curves, attractor robustness, threshold ties, one-at-a-time sensitivity.
* **Reporting**: model cards, draft methods and results text, the 22-item PNM reporting
  checklist, Markdown reports with the software environment.

## Repository layout

```
bcpnm/          the package (core, rules, dynamics, attractors, equilibrium, perturbation,
                inference, io, validation, visualization, reporting, utils)
examples/       configuration files (00 template, 01-03 case studies), tutorial scripts,
                notebooks/
docs/           index, tutorial, API reference, examples
tests/          pytest suite
paper/          tutorial paper (Markdown and Word), figures, results.json, reports, tools/
```

## Reproducing the paper

```bash
python examples/run_tutorial.py              
python paper/tools/run_listings.py --check   # re-executes every code listing in the paper
python paper/tools/build_docx.py             # renders paper/tutorial_paper.docx
```

## Documentation

See `docs/index.md`, the tutorial `docs/tutorial.md`, the API reference `docs/api.md` and
the notebooks in `examples/notebooks/`.

## Citing

Please cite the tutorial paper and the software version you used (see `CITATION.cff`).

## Contributing

Contributions are welcome, from new example domains to new algorithms. See
`CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

## License

MIT (see `LICENSE`).
