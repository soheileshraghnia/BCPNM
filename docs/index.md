# bcpnm documentation

`bcpnm` implements **Blume–Capel probabilistic network modeling (BC PNM)** for psychological
systems: discrete-state network models in which persons or constructs take a small number of
ordered states (for example conceal / silent / acknowledge, absent / present, or low /
moderate / high) and update according to logical, threshold or probabilistic rules.

One transition rule spans the whole model family:

| | deterministic (β = ∞) | stochastic (finite β) |
|---|---|---|
| **asymmetric influence** | Boolean and multi-valued logical networks | probabilistic Boolean networks, directed BC transition networks |
| **symmetric influence** | ternary threshold (Hopfield-type) networks | equilibrium Blume–Capel model (Gibbs sampling) |

## Contents

* [Tutorial](tutorial.md): the seven-step workflow on a small symptom network.
* [API reference](api.md): every public class and function, generated from the docstrings.
* [Examples](examples/README.md): the three case studies of the tutorial paper and the
  configuration template for new domains.
* Notebooks: `examples/notebooks/tutorial_basic.ipynb` (workflow) and
  `examples/notebooks/tutorial_extension.ipynb` (new domains, custom rules, Blume–Capel
  models, inference).
* The tutorial paper: `paper/tutorial_paper.md` (also as `.docx`).

## What can I do with it?

* **Specify** models in code or in YAML configuration files (`load_model`, `PNMGraph`).
* **Simulate** synchronous, asynchronous or sequential dynamics with time-windowed
  interventions (`simulate`, `Intervention`).
* **Analyze long-run behavior**: attractors and basins of deterministic networks
  (`find_attractors`), exact stationary distributions of stochastic networks
  (`stationary_distribution`), and regimes of stochastic networks (`quasi_attractors`).
* **Study the equilibrium Blume–Capel model**: exact probabilities, Gibbs sampling,
  mean-field theory and hysteresis (`BlumeCapelModel`, `continuation`).
* **Intervene**: clamp nodes, remove or rescale influences, shift fields, scan leverage
  (`Clamp`, `ScaleEdge`, `ShiftField`, `leverage_scan`).
* **Learn from data**: REVEAL input sets, directed and equilibrium Blume–Capel likelihoods,
  held-out log scores (`reveal`, `fit_transition_model`, `fit_equilibrium`, `log_score`).
* **Validate and report**: expert sign agreement, sensitivity, canalization, Derrida
  curves, robustness, threshold ties, model cards and the PNM reporting checklist.

## Installation

```bash
pip install -r requirements.txt   # exact versions used in the paper (optional)
pip install -e .
pytest
```
