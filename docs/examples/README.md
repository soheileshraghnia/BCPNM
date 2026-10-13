# Examples

| File | Domain | Model | Paper |
|---|---|---|---|
| `examples/00_template_burnout.yaml` | Burnout (template for new domains) | Threshold, case and Boolean rules, ordinal states | Appendix B.1 |
| `examples/01_conspiracy_of_silence.yaml` | Suicidology: family silence | Equilibrium Blume–Capel, ternary | Section 7 |
| `examples/02_organizational_silence.yaml` | Organizational silence and voice | Directed Blume–Capel, ternary | Section 8 |
| `examples/03_stigma_help_seeking.yaml` | Stigma and help-seeking | Threshold, case and Boolean rules, mixed states | Section 9 |
| `examples/tutorial_steps.py` | Depression–anxiety comorbidity | Boolean | Section 6 |
| `examples/run_tutorial.py` | All of the above | — | every number and figure |

Run `python examples/run_tutorial.py` from the repository root. It writes the figures to
`paper/figures/`, every reported number to `paper/results.json`, and a bcpnm report (model
card, software environment and the PNM reporting checklist) per case study to
`paper/reports/`. Short summaries:

* [Case Study 1: the conspiracy of silence](case1_conspiracy_of_silence.md)
* [Case Study 2: organizational silence](case2_organizational_silence.md)
* [Case Study 3: stigma and help-seeking](case3_stigma_help_seeking.md)
