# Tutorial: the seven-step workflow

This page condenses Section 6 of the tutorial paper. The complete script is
`examples/tutorial_steps.py`; the same analysis with figures is in
`examples/notebooks/tutorial_basic.ipynb`.

The running example is a six-node Boolean network of depression–anxiety comorbidity:
a stressor, worry, insomnia, fatigue, concentration problems and depressed mood.

## Step 1. Define the question and the nodes

*Can a brief stressor leave a lasting depressive–anxious state, and which symptom is the best
target for prevention?* Choose nodes that the question requires, and document what each means.

```python
import numpy as np
import bcpnm as bc

g = bc.PNMGraph("comorbidity", meta={"domain": "depression-anxiety comorbidity"})
B = bc.StateSpace.boolean("absent", "present")
for name in ["Stressor", "Worry", "Insomnia", "Fatigue", "Concentration", "Mood"]:
    g.add_node(name, B)
```

## Step 2. Choose the state spaces

Boolean here; use `StateSpace.ternary()` for −1/0/+1 (e.g. conceal/silent/acknowledge) and
`StateSpace.ordinal(3)` for low/moderate/high.

## Step 3. Specify the rules

```python
g.set_rule("Stressor", bc.ClampRule(0))
g.set_rule("Worry", bc.BooleanRule("Stressor or (Worry and Insomnia)"))
g.set_rule("Insomnia", bc.OR("Worry", "Mood"))
g.set_rule("Fatigue", bc.BooleanRule("Insomnia"))
g.set_rule("Concentration", bc.BooleanRule("Fatigue and (Worry or Mood)"))
g.set_rule("Mood", bc.AND("Fatigue", "Concentration"))
```

Other rule types: `CaseRule` (multi-valued logic), `ThresholdRule` (weighted sums with cut
points), `TableRule`, `SoftmaxRule` (the Blume–Capel transition rule), and the stochastic
wrappers `NoisyRule` and `PBNRule`.

## Step 4. Choose the update scheme and simulate

```python
pulse = bc.Intervention(bc.Clamp("Stressor", 1), start=2, end=5)
tr = bc.simulate(g, x0=np.zeros(6), steps=12, scheme="synchronous", interventions=[pulse])
print(tr.to_panel(0).T)   # the symptom configuration persists after the stressor ends
```

## Step 5. Find attractors and long-run behavior

```python
print(bc.find_attractors(g, scheme="synchronous").summary())
print(bc.find_attractors(g, scheme="asynchronous").summary())
gn = g.copy()
for n in g.names[1:]:
    gn.set_rule(n, bc.NoisyRule(g.rules[n], 0.02))
st = bc.stationary_distribution(gn, scheme="asynchronous")
print(bc.quasi_attractors(gn, st).summary())
```

Report results under both schemes when they differ: in this example the long-run share of
time in the healthy regime is 98.1% under synchronous and 56.5% under asynchronous updating.

## Step 6. Perturb

```python
for node in ("Worry", "Insomnia", "Mood"):
    print(node, bc.find_attractors(bc.Clamp(node, 0).apply(g)).summary())
```

Suppressing insomnia removes every pathological attractor.

## Step 7. Validate, check sensitivity, report

```python
print(bc.canalizing_inputs(g, "Insomnia"))
print(bc.derrida_curve(g, samples=4000, seed=1))
print(bc.expert_consistency(g, [("Worry", "Insomnia", +1)]))
bc.write_report("comorbidity_report.md", g)          # model card + environment + checklist
bc.save_config(bc.graph_to_config(g), "comorbidity.yaml")
```

See the paper (Section 10) for the validity framework and the 22-item reporting checklist
(`bc.checklist()`).
