# Contributing to bcpnm

Thank you for helping. Contributions from psychologists are as valuable as those from
programmers: a well-documented example model in a new domain is one of the most useful
things you can add.

## Reporting a problem or proposing a feature

Open an issue on GitHub.

* **Bug reports** should include a minimal example that reproduces the problem and the output
  of `import bcpnm as bc; print(bc.environment_report())`.
* **Feature proposals** should describe the psychological or methodological question the
  feature would help answer, and, where possible, a reference for the method.

## Development workflow

1. Fork the repository and create a branch from `main` (`git checkout -b my-feature`).
2. Install in development mode: `pip install -e ".[dev]"`.
3. Make focused changes. Follow the existing style: small functions, NumPy-style docstrings,
   vectorized code, explicit `seed` arguments for anything random, and perturbations that
   return copies rather than modifying their input.
4. Add tests under `tests/`. Prefer analytical checks (for example, that a Markov chain
   reproduces a known distribution, or that an attractor of a textbook network is found) over
   stored regression values.
5. Run the checks:
   ```bash
   pytest
   python paper/tools/run_listings.py --check     # if you touched code used in the paper
   python paper/tools/build_api_docs.py           # regenerate docs/api.md
   ```
6. Update `CHANGELOG.md` and, if relevant, `docs/`.
7. Open a pull request describing what changed and why. Continuous integration runs the test
   suite on several Python versions and the executable-paper check with pinned versions.

## Contributing an example domain

Add a configuration file `examples/NN_short_name.yaml` (start from
`examples/00_template_burnout.yaml`) with:

* a `meta.description` and `meta.sources` field that state where the rules come from;
* a `construct` (and, where possible, `measurement`) for every node;
* rules whose weights avoid exact threshold ties (`bc.threshold_ties(graph)` must report 0);

and a short script or notebook that reproduces the analysis, plus a paragraph in
`docs/examples/README.md`. The test `tests/test_core.py::test_example_configs_load` will
check that the file loads.

## Contributing a rule type or algorithm

Subclass `bcpnm.core.Rule` and implement `next_value()` (deterministic) or `probs()`
(stochastic), vectorized over the first axis of `parents`. If the rule should be usable in
configuration files, add it to `bcpnm/io.py::_rule_from_spec` and to `graph_to_config`.
Algorithms should state their complexity and the size of networks they handle in the docstring.

## Code of conduct

Everyone participating in this project is expected to follow `CODE_OF_CONDUCT.md`.
