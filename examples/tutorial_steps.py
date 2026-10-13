"""Section 6 of the tutorial paper in one script: the seven modeling steps on a
depression-anxiety comorbidity network. Run from the repository root:

    python examples/tutorial_steps.py
"""
import pathlib

import numpy as np

import bcpnm as bc

OUT = pathlib.Path("outputs")
OUT.mkdir(exist_ok=True)

# Steps 1-2: nodes and state spaces ---------------------------------------------------------
g = bc.PNMGraph("comorbidity", meta={"domain": "depression-anxiety comorbidity"})
B = bc.StateSpace.boolean("absent", "present")
constructs = {"Stressor": "acute life stressor (context)", "Worry": "excessive worry",
              "Insomnia": "sleep problems", "Fatigue": "fatigue / loss of energy",
              "Concentration": "concentration problems", "Mood": "depressed mood"}
for name, construct in constructs.items():
    g.add_node(name, B, construct=construct)

# Step 3: rules ----------------------------------------------------------------------------
g.set_rule("Stressor", bc.ClampRule(0))
g.set_rule("Worry", bc.BooleanRule("Stressor or (Worry and Insomnia)"))
g.set_rule("Insomnia", bc.OR("Worry", "Mood"))
g.set_rule("Fatigue", bc.BooleanRule("Insomnia"))
g.set_rule("Concentration", bc.BooleanRule("Fatigue and (Worry or Mood)"))
g.set_rule("Mood", bc.AND("Fatigue", "Concentration"))
g.validate()
print(g.summary())

# Step 4: update scheme and simulation -----------------------------------------------------
pulse = bc.Intervention(bc.Clamp("Stressor", 1), start=2, end=5, name="stressor")
tr = bc.simulate(g, x0=np.zeros(6), steps=12, scheme="synchronous", interventions=[pulse])
print(tr.to_panel(run=0).T)

# Step 5: attractors and stationary distributions -----------------------------------------
regimes = {(0, 0, 0, 0, 0, 0): "healthy", (0, 1, 1, 1, 1, 1): "comorbid",
           (0, 0, 1, 1, 1, 1): "depressive", (0, 1, 1, 1, 1, 0): "anxious"}
name_of = lambda S: regimes.get(tuple(int(v) for v in S[0]), "other")  # noqa: E731
sync = bc.find_attractors(g, scheme="synchronous").label_by(name_of)
asyn = bc.find_attractors(g, scheme="asynchronous").label_by(name_of)
print(sync.summary())
print(asyn.summary())
gn = g.copy()
for n in g.names[1:]:
    gn.set_rule(n, bc.NoisyRule(g.rules[n], 0.02))
for scheme in ("synchronous", "asynchronous"):
    st = bc.stationary_distribution(gn, scheme=scheme)
    qa = bc.quasi_attractors(gn, st, scheme=scheme).label_by(name_of)
    print(scheme, {a.label: round(a.basin_fraction, 3) for a in qa},
          "P(Mood present) =", round(st.marginal("Mood")[1], 3))

# Step 6: perturbations --------------------------------------------------------------------
for node in ("Worry", "Insomnia", "Mood"):
    after = bc.find_attractors(bc.Clamp(node, 0).apply(g)).label_by(name_of)
    print(f"suppress {node:<9}", {a.label: round(a.basin_fraction, 3) for a in after})

# Step 7: validation, sensitivity and reporting --------------------------------------------
print("canalizing inputs of Insomnia:", bc.canalizing_inputs(g, "Insomnia"))
print("mean average sensitivity:", round(bc.average_sensitivity(g, seed=1)["mean"], 3))
print("Derrida curve:", {d: round(v, 2) for d, v in bc.derrida_curve(g, samples=4000, seed=1).items()})
expected = [("Worry", "Insomnia", +1), ("Insomnia", "Fatigue", +1),
            ("Fatigue", "Mood", +1), ("Mood", "Insomnia", +1)]
print("expert sign agreement:", bc.expert_consistency(g, expected, seed=1).attrs["agreement"])
bc.write_report(OUT / "comorbidity_report.md", g, sections={"Attractors": bc.results_text(sync)})
bc.save_config(bc.graph_to_config(g), OUT / "comorbidity.yaml")
print("report and configuration written to", OUT)
