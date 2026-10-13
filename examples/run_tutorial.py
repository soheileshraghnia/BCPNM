"""Reproduce every number and figure reported in paper/tutorial_paper.md.

Usage
-----
    python examples/run_tutorial.py            # full run (about two minutes)
    python examples/run_tutorial.py --fast     # fewer Monte Carlo runs, for a quick check

Outputs
-------
    paper/figures/fig*.png and .pdf   figures
    paper/results.json                every number quoted in the paper, plus the software environment
    paper/reports/*.md                bcpnm reports (model card + PNM reporting checklist) per example

All randomness flows from the seeds below, so repeated runs give identical results on
the same software versions (see requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bcpnm as bc  # noqa: E402
from bcpnm import visualization as viz  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "paper" / "figures"
REP = ROOT / "paper" / "reports"
SEED = 20261012

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 110,
                     "savefig.dpi": 300, "axes.edgecolor": "#52514e", "axes.linewidth": 0.8})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def r3(x):
    return float(np.round(x, 3))


# ============================================================================ Figure 1: the model family
def figure_framework():
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7.1); ax.axis("off")
    cells = {
        (0, 1): ("Logical networks", "Boolean and multi-valued rules;\nfixed points and cycles\n(Kauffman, 1969; Thomas, 1973)"),
        (1, 1): ("Probabilistic logical networks", "PBNs, random perturbation;\nstationary distribution,\nquasi-attractors\n(Shmulevich et al., 2002)"),
        (0, 0): ("Threshold networks", "symmetric weights: energy\ndescent to fixed points\n(Hopfield, 1982)"),
        (1, 0): ("Equilibrium BC PNM", "Boltzmann distribution, Gibbs\nsampling, phase transitions,\nhysteresis (Blume, 1966;\nCapel, 1966)"),
    }
    x0, y0, w, h, gap = 1.9, 0.9, 4.15, 2.6, 0.9
    for (cx, cy), (title, body) in cells.items():
        x, y = x0 + cx * (w + gap), y0 + cy * (h + 0.3)
        face = "#e8f1fc" if cx == 1 else "#f6f6f3"
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                                       fc=face, ec="#52514e", lw=0.8))
        ax.text(x + 0.2, y + h - 0.3, title, fontweight="bold", fontsize=9.5, va="top")
        ax.text(x + 0.2, y + h - 0.8, body, fontsize=8.3, va="top", color="#2c2c2a", linespacing=1.4)
        if cx == 0:
            yy = y + h / 2
            ax.annotate("", xy=(x + w + 0.08, yy), xytext=(x + w + gap - 0.08, yy),
                        arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=1.4))
            ax.text(x + w + gap / 2, yy + 0.16, "β → ∞", ha="center", fontsize=8.5, color="#2a78d6")
    ax.text(x0 + w / 2, 6.85, "deterministic (β = ∞)", ha="center", fontsize=9.5, color="#52514e", fontweight="bold")
    ax.text(x0 + 1.5 * w + gap, 6.85, "stochastic (finite β)", ha="center", fontsize=9.5, color="#52514e", fontweight="bold")
    ax.text(0.95, y0 + 1.5 * h + 0.3, "asymmetric\n(directed)\ninfluence", ha="center", va="center", fontsize=9, color="#52514e", fontweight="bold")
    ax.text(0.95, y0 + h / 2, "symmetric\ninfluence", ha="center", va="center", fontsize=9, color="#52514e", fontweight="bold")
    ax.text(x0 + w + gap / 2, 0.35,
            "One bcpnm rule (SoftmaxRule) covers the right column: symmetric weights give the equilibrium model,\n"
            "asymmetric weights a directed transition network; its β → ∞ limit is a threshold network (left column).",
            ha="center", fontsize=7.8, color="#2a78d6", style="italic")
    save(fig, "fig1_model_family")


# ============================================================================ Section 6: step-by-step tutorial
def comorbidity_graph():
    g = bc.PNMGraph("comorbidity", meta={"domain": "depression-anxiety comorbidity", "seed": SEED})
    B = bc.StateSpace.boolean("absent", "present")
    constructs = {"Stressor": "acute life stressor (context)", "Worry": "excessive worry",
                  "Insomnia": "sleep problems", "Fatigue": "fatigue / loss of energy",
                  "Concentration": "concentration problems", "Mood": "depressed mood"}
    for name, c in constructs.items():
        g.add_node(name, B, construct=c)
    g.set_rule("Stressor", bc.ClampRule(0))
    g.set_rule("Worry", bc.BooleanRule("Stressor or (Worry and Insomnia)"))
    g.set_rule("Insomnia", bc.OR("Worry", "Mood"))
    g.set_rule("Fatigue", bc.BooleanRule("Insomnia"))
    g.set_rule("Concentration", bc.BooleanRule("Fatigue and (Worry or Mood)"))
    g.set_rule("Mood", bc.AND("Fatigue", "Concentration"))
    return g


def tutorial(res):
    g = comorbidity_graph()
    out = {}
    sync = bc.find_attractors(g, scheme="synchronous")
    asyn = bc.find_attractors(g, scheme="asynchronous")
    lab = {(0, 0, 0, 0, 0, 0): "healthy", (0, 1, 1, 1, 1, 1): "comorbid", (0, 0, 1, 1, 1, 1): "depressive"}
    for r in (sync, asyn):
        r.label_by(lambda S: lab.get(tuple(int(v) for v in S[0]), ""))
    out["sync"] = {a.label: r3(a.basin_fraction) for a in sync}
    out["async_strong"] = {a.label: r3(a.basin_fraction) for a in asyn}
    out["async_weak"] = {a.label: int(a.weak_basin) for a in asyn}
    tr = bc.simulate(g, x0=np.zeros(6), steps=12, interventions=[bc.Intervention(bc.Clamp("Stressor", 1), 2, 5, "stressor")])
    out["pulse_final"] = tr.states[-1, 0].tolist()
    # probabilistic version
    gn = g.copy()
    for n in g.names[1:]:
        gn.set_rule(n, bc.NoisyRule(g.rules[n], 0.02))
    for scheme in ("synchronous", "asynchronous"):
        st = bc.stationary_distribution(gn, scheme=scheme)
        qa = bc.quasi_attractors(gn, st, scheme=scheme)
        qa.label_by(lambda S: lab.get(tuple(int(v) for v in S[0]), ""))
        out[f"noisy_{scheme}"] = {a.label: r3(a.basin_fraction) for a in qa}
        out[f"noisy_{scheme}_P_mood"] = r3(st.marginal("Mood")[1])
    # perturbations
    out["clamp"] = {}
    for node in ("Worry", "Insomnia", "Mood"):
        cmp = bc.compare_attractors(sync, bc.find_attractors(bc.Clamp(node, 0).apply(g)).label_by(
            lambda S: lab.get(tuple(int(v) for v in S[0]), "other")))
        out["clamp"][node] = {row.attractor: r3(row.after) for row in cmp.itertuples()}
    out["canalizing_insomnia"] = bc.canalizing_inputs(g, "Insomnia")
    out["activities"] = {k: {kk: r3(vv) for kk, vv in v.items()} for k, v in bc.input_activities(g, seed=1).items()}
    out["sensitivity"] = {k: r3(v) for k, v in bc.average_sensitivity(g, seed=1).items()}
    out["derrida"] = {k: r3(v) for k, v in bc.derrida_curve(g, samples=4000, seed=1).items()}
    exp = [("Worry", "Insomnia", 1), ("Insomnia", "Fatigue", 1), ("Fatigue", "Mood", 1), ("Mood", "Insomnia", 1)]
    out["expert_agreement"] = bc.expert_consistency(g, exp, seed=1).attrs["agreement"]
    res["tutorial"] = out

    # Figure 2
    fig = plt.figure(figsize=(7.0, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.35, wspace=0.15)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[0, 1])]
    pos = {"Stressor": (-1.3, 1.0), "Worry": (-0.5, 0.45), "Insomnia": (0.5, 1.0), "Fatigue": (1.3, 0.3),
           "Concentration": (0.95, -0.6), "Mood": (-0.6, -0.6)}
    viz.plot_network(g, ax=axs[0], pos=pos, node_size=1250, font_size=6,
                     labels={n: ("Concen-\ntration" if n == "Concentration" else n) for n in g.names})
    axs[0].set_title("A  Symptom network")
    viz.plot_trajectory(tr, ax=axs[1], labels={0: "absent", 1: "present"})
    axs[1].text(4, 0, "on", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")
    axs[1].set_title("B  A three-step stressor pulse")
    viz.plot_state_transition_graph(g, ax=axs[2], result=sync, seed=4)
    axs[2].set_title("C  Synchronous state transitions")
    save(fig, "fig2_tutorial")
    return g, gn


# ============================================================================ Case Study 1: conspiracy of silence
def case1(res, fast):
    out = {}
    # 7.1 exact three-person network
    for lam in (0.0, 0.8):
        K = 0.6 * (np.ones((3, 3)) - np.eye(3))
        L = lam * (np.ones((3, 3)) - np.eye(3))
        m = bc.BlumeCapelModel(["A", "B", "C"], K, L, delta=0.3, beta=1.0)
        X, p = m.exact()
        out[f"exact_lam{lam}"] = {"P_all_silent": r3(p[(X == 0).all(1)].sum()), "S0": r3(m.occupancy(X, 0, p))}
    # 7.2 family network from the configuration file
    g, cfg = bc.load_model(HERE / "01_conspiracy_of_silence.yaml")
    g0 = g.copy()
    for r in g0.rules.values():
        r.lam[:] = 0.0
    lab = {(0,) * 5: "all silent", (-1,) * 5: "all conceal", (1,) * 5: "all acknowledge"}
    fam = {}
    for name, gg in (("with_lambda", g), ("without_lambda", g0)):
        st = bc.stationary_distribution(gg)
        qa = bc.quasi_attractors(gg, st).label_by(lambda S: lab.get(tuple(int(v) for v in S[0]), "mixed"))
        fam[name] = {"P_all_silent": r3(st.prob(lambda X: (X == 0).all(1))),
                     "S0": r3(st.expectation(lambda X: (X == 0).mean(1))),
                     "P_adolescent": {k: r3(v) for k, v in st.marginal("Adolescent").items()},
                     "quasi": {a.label: r3(a.basin_fraction) for a in qa}}
    out["family"] = fam
    # interventions (exact stationary distributions)
    edges = [(e["source"], e["target"]) for e in cfg["model"]["edges"]]
    halve_lam = bc.Combined([bc.ScaleEdge(a, b, 0.5, include_w=False) for s, t in edges for a, b in ((s, t), (t, s))])
    ivs = {"baseline": None,
           "friend as gatekeeper (clamp Friend = +1)": bc.Clamp("Friend", 1),
           "psychoeducation for mother (h_Mother + 1)": bc.ShiftField("Mother", 1.0),
           "family talk norms (all Lambda halved)": halve_lam}
    fam_iv = {}
    for k, p in ivs.items():
        gg = p.apply(g) if p else g
        st = bc.stationary_distribution(gg)
        fam_iv[k] = {"P_adolescent_ack": r3(st.marginal("Adolescent")[1]),
                     "P_adolescent_silent": r3(st.marginal("Adolescent")[0]),
                     "P_all_silent": r3(st.prob(lambda X: (X == 0).all(1))),
                     "S0": r3(st.expectation(lambda X: (X == 0).mean(1)))}
    out["family_interventions"] = fam_iv
    # 7.3 hysteresis (mean field)
    swh = bc.continuation("h", np.linspace(-0.7, 0.7, 281), beta=1.0, J=4.0, Lam=0.0, delta=1.6)
    swd = bc.continuation("delta", np.linspace(-4.0, 1.0, 251), beta=1.0, J=0.5, Lam=5.0, delta=0.0)
    out["hysteresis_h"] = [r3(v) for v in swh.jumps(0)]
    out["hysteresis_delta"] = [r3(v) for v in swd.jumps(1)]
    # 7.4 directed five-person network with opportunity windows
    dg = directed_graph()
    runs = 4000 if fast else 20000
    gains = {}
    for k, target in enumerate(("A", "B")):
        iv = bc.Intervention(bc.SetParameter(target, "h", 1.6), start=40, end=60, name=f"opportunity for {target}")
        tr = bc.simulate(dg, x0=np.zeros(5), steps=100, runs=runs, scheme="synchronous", seed=100 + k, interventions=[iv])
        P = tr.prob(1)
        gains[target] = {n: r3(P[40:63, j].max() - P[30:40, j].mean()) for j, n in enumerate(dg.names)}
        if target == "A":
            trA = tr
            out["directed_baseline"] = {"P_plus": r3(P[30:40].mean()), "P_zero": r3(tr.prob(0)[30:40].mean())}
    out["directed_gains"] = gains
    # 7.5 predictive validity: does the zero-zero term improve out-of-sample prediction?
    train = bc.simulate(dg, steps=40, runs=150, scheme="synchronous", seed=301)
    test = bc.simulate(dg, steps=40, runs=150, scheme="synchronous", seed=302)
    fit_full = bc.fit_transition_model(train.states, dg.names, lam=True)
    fit_nolam = bc.fit_transition_model(train.states, dg.names, lam=False)
    fit_null = bc.fit_transition_model(train.states, dg.names, lam=False, alpha=1e6)
    true_W = np.array([[dg.rules[t].w[list(dg.rules[t].inputs).index(s)] if s in dg.rules[t].inputs else 0.0
                        for s in dg.names] for t in dg.names])
    rec = bc.parameter_recovery({"W": true_W}, {"W": fit_full.W})
    out["predictive"] = {"log_score_true": r3(bc.log_score(dg, test.states)),
                         "log_score_full": r3(bc.log_score(fit_full.to_graph(), test.states)),
                         "log_score_no_lambda": r3(bc.log_score(fit_nolam.to_graph(), test.states)),
                         "log_score_no_network": r3(bc.log_score(fit_null.to_graph(), test.states)),
                         "W_correlation": float(rec.loc[0, "correlation"]),
                         "n_transitions": int(fit_full.meta["n_transitions"])}
    res["case1"] = out

    # Figure 3
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.4))
    pos = {"Adolescent": (0, 0), "Mother": (-1, 0.8), "Father": (-1.3, -0.5), "Sibling": (0.2, 1.1), "Friend": (1.2, 0.2)}
    viz.plot_network(g, state=[n.initial for n in g.nodes.values()], ax=axs[0, 0], pos=pos, node_size=1100, font_size=6,
                     labels={n: ("Adoles-\ncent" if n == "Adolescent" else n) for n in g.names})
    axs[0, 0].set_title("A  Family network (initial states)")
    from matplotlib.patches import Patch
    axs[0, 0].legend(handles=[Patch(color=viz.NEG, label="conceal (−1)"), Patch(color=viz.NEU, label="silent (0)"),
                              Patch(color=viz.POS, label="acknowledge (+1)")], loc="lower right", frameon=False, fontsize=7.5)
    ax = axs[0, 1]
    keys = ["all silent", "all conceal", "all acknowledge"]
    xx = np.arange(3)
    ax.bar(xx - 0.19, [fam["with_lambda"]["quasi"].get(k, 0) for k in keys], 0.36, color=viz.POS, label="with Λ")
    ax.bar(xx + 0.19, [fam["without_lambda"]["quasi"].get(k, 0) for k in keys], 0.36, color="#b7b5ad", label="Λ = 0")
    for x_, k in zip(xx, keys):
        if k not in fam["without_lambda"]["quasi"]:
            ax.text(x_ + 0.19, 0.01, "none", ha="center", va="bottom", fontsize=7, color="#6a6964")
    ax.set_xticks(xx, keys); ax.set_ylabel("long-run share of time in basin")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    ax.set_title("B  Regimes of the family network")
    viz.plot_hysteresis(swd, column=1, ax=axs[1, 0], xlabel="zero-state stability δ")
    axs[1, 0].set_title("C  Hysteresis of collective silence")
    viz.plot_probabilities(trA, 1, ax=axs[1, 1], shade=(40, 60))
    axs[1, 1].set_ylabel("P(acknowledge)")
    axs[1, 1].legend(loc="upper left", ncol=2, frameon=False, fontsize=7.5)
    axs[1, 1].set_title("D  A's opportunity window spreads downstream")
    fig.tight_layout()
    save(fig, "fig3_case1")
    bc.write_report(REP / "case1_family.md", g, cfg, sections={"Results": json.dumps(fam, indent=1)},
                    completed={"3": "Section 7.2", "5": "Section 7.2, Appendix B", "7": "Section 7.2"})
    return g


def directed_graph():
    """Directed five-person network of Section 7.4 (W[target <- source])."""
    g = bc.PNMGraph("directed_five", meta={"note": "asymmetric influence; opportunity windows"})
    T = bc.StateSpace.ternary(("conceal", "silent", "acknowledge"))
    for n in "ABCDE":
        g.add_node(n, T)
    w = {"A": {"B": 0.25}, "B": {"A": 1.15}, "C": {"B": 0.95}, "D": {"C": 0.30}, "E": {"B": 0.18}}
    for n in "ABCDE":
        g.set_rule(n, bc.SoftmaxRule(w[n], {k: 0.4 for k in w[n]}, h=-0.15, delta=0.3, rho=0.6, beta=1.5))
    return g


# ============================================================================ Case Study 2: organizational silence
def case2(res):
    out = {}
    g, cfg = bc.load_model(HERE / "02_organizational_silence.yaml")
    det = bc.deterministic_graph(g)

    def label(S):
        s = S[0]
        if (s[:3] == -1).all() and (s[3:] == 0).all():
            return "climate of silence"
        if (s == 1).all():
            return "full voice"
        if (s[:3] == 1).all():
            return "leaders voice, some employees silent"
        if (s[:3] == -1).all():
            return "leaders endorse, some employees endorse"
        return "other"

    sync = bc.find_attractors(det).label_by(label)
    out["n_attractors"] = len(sync)
    grouped = {}
    for a in sync:
        grouped[a.label] = grouped.get(a.label, 0) + a.basin_fraction
    grouped = {k: r3(v) for k, v in grouped.items()}
    out["sync_basins"] = grouped
    out["climate_basin"] = r3(sync.attractors[0].basin_fraction)
    out["full_voice_basin"] = r3(sum(a.basin_fraction for a in sync if a.label == "full voice"))
    lev = bc.leverage_scan(det, target=lambda s: (s[3:] == 1).sum() >= 2, value_fn=lambda n: 1)
    out["leverage_baseline"] = r3(lev.attrs["baseline"])
    out["leverage"] = {row.node: r3(row.share) for row in lev.itertuples()}
    emps = [n for n in g.names if n.startswith("Emp")]
    peer = [("EmpA1", "EmpA2"), ("EmpA2", "EmpA1"), ("EmpB1", "EmpB2"), ("EmpB2", "EmpB1")]
    ivs = {"baseline": None,
           "voice training (employees: h + 0.5)": bc.Combined([bc.ShiftField(e, 0.5) for e in emps]),
           "leader openness (manager: h + 0.6)": bc.ShiftField("Manager", 0.6),
           "psychological safety (employees: δ 0.4 → 0.1)": bc.Combined([bc.SetParameter(e, "delta", 0.1) for e in emps]),
           "no peer silence coordination (peer Λ = 0)": bc.Combined([bc.ScaleEdge(a, b, 0.0, include_w=False) for a, b in peer]),
           "openness + safety": bc.Combined([bc.ShiftField("Manager", 0.6)] + [bc.SetParameter(e, "delta", 0.1) for e in emps])}
    eff = {}
    for k, p in ivs.items():
        gg = p.apply(g) if p else g
        st = bc.stationary_distribution(gg)
        eff[k] = {"voice": r3(st.expectation(lambda X: (X == 1).mean(1))),
                  "silence": r3(st.expectation(lambda X: (X == 0).mean(1))),
                  "employee_voice": r3(st.expectation(lambda X: (X[:, 3:] == 1).mean(1))),
                  "employee_silence": r3(st.expectation(lambda X: (X[:, 3:] == 0).mean(1)))}
    out["interventions"] = eff
    exp = [("Manager", "LeadA", 1), ("LeadA", "EmpA1", 1), ("EmpA1", "EmpA2", 1), ("EmpA1", "LeadA", 1)]
    ec = bc.expert_consistency(g, exp, seed=2)
    out["expert_effects"] = {f"{r.source}->{r.target}": r.effect for r in ec.itertuples()}
    out["derrida"] = {k: r3(v) for k, v in bc.derrida_curve(g, samples=3000, seed=2, scheme="synchronous").items()}
    res["case2"] = out

    # Figure 4
    fig = plt.figure(figsize=(7.0, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.4, wspace=0.55)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    pos = {"Manager": (0, 2), "LeadA": (-1.05, 1), "LeadB": (1.05, 1), "EmpA1": (-1.6, 0), "EmpA2": (-0.5, 0),
           "EmpB1": (0.5, 0), "EmpB2": (1.6, 0)}
    viz.plot_network(g, state=sync.attractors[0].states[0], ax=axs[0], pos=pos, node_size=480, font_size=4.8)
    axs[0].set_title("A  Climate-of-silence attractor")
    order = sorted(grouped, key=lambda k: -grouped[k])
    axs[1].barh(range(len(order))[::-1], [grouped[k] for k in order], color=viz.POS, height=0.55)
    for y, k in zip(range(len(order))[::-1], order):
        axs[1].text(grouped[k] + 0.01, y, f"{grouped[k]:.1%}", va="center", fontsize=7.5)
    axs[1].set_yticks(range(len(order))[::-1], [k.replace(", ", ",\n") for k in order], fontsize=7.5)
    axs[1].set_xlim(0, 0.95); axs[1].set_xlabel("share of initial states (β = ∞)")
    axs[1].spines[["top", "right"]].set_visible(False)
    axs[1].set_title("B  Attractor basins")
    ks = list(eff)
    y = np.arange(len(ks))[::-1]
    axs[2].barh(y + 0.18, [eff[k]["employee_voice"] for k in ks], 0.34, color=viz.POS, label="employee voice")
    axs[2].barh(y - 0.18, [eff[k]["employee_silence"] for k in ks], 0.34, color=viz.NEU, label="employee silence")
    axs[2].set_yticks(y, [k.split(" (")[0] for k in ks], fontsize=7.5)
    axs[2].set_xlim(0, 0.75); axs[2].set_xlabel("long-run probability (β = 1.5)")
    axs[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    axs[2].spines[["top", "right"]].set_visible(False)
    axs[2].set_title("C  Interventions")
    save(fig, "fig4_case2")
    bc.write_report(REP / "case2_organization.md", g, cfg, sections={"Results": json.dumps(out, indent=1, default=str)})
    return g


# ============================================================================ Case Study 3: stigma and help-seeking
def case3(res, fast):
    out = {}
    g, cfg = bc.load_model(HERE / "03_stigma_help_seeking.yaml")

    def label(S):
        s = S[0]
        if S.shape[0] > 1:
            return "cycle (synchronous only)"
        if s[1] == 0 and s[2] == 2:
            return "recovered"
        if s[1] == 2 and s[2] == 0:
            return "stuck"
        if s[5] >= 1 and s[6] == 0:
            return "intention without action"
        return "other"

    land = {}
    for ps in (0, 1, 2):
        gg = bc.Clamp("PublicStigma", ps).apply(g)
        sync = bc.find_attractors(gg).label_by(label)
        asyn = bc.find_attractors(gg, scheme="asynchronous").label_by(label)
        d = {}
        for a in sync:
            d[a.label] = d.get(a.label, 0) + a.basin_fraction
        d = {k: r3(v) for k, v in d.items()}
        land[ps] = {"sync": d, "async_attractors": sorted({a.label for a in asyn})}
    out["landscape"] = land
    gg = bc.Clamp("PublicStigma", 1).apply(g)
    sync1 = bc.find_attractors(gg).label_by(label)
    rob = bc.attractor_robustness(gg, sync1)
    out["robustness"] = {sync1.attractors[k].label: r3(v) for k, v in rob.items()}
    out["canalizing_intention"] = bc.canalizing_inputs(g, "Intention")
    # dose-response of pulse interventions starting from the 'stuck' attractor
    gn = g.copy()
    for n in g.names[1:]:
        gn.set_rule(n, bc.NoisyRule(g.rules[n], 0.005))
    stuck = np.array([1, 2, 0, 2, 0, 0, 0])
    runs = 600 if fast else 3000
    doses = [0, 1, 2, 3, 4, 6, 8, 12]
    targets = {"self-stigma (clamp SelfStigma = 0)": ("SelfStigma", 0),
               "facilitated contact (clamp HelpSeeking = 1)": ("HelpSeeking", 1),
               "support boost (clamp Support = 2)": ("Support", 2),
               "anti-stigma campaign (clamp PublicStigma = 0)": ("PublicStigma", 0)}
    dose = {}
    for k, (node, val) in targets.items():
        dose[k] = []
        for dd in doses:
            ivs = [bc.Intervention(bc.Clamp(node, val), start=5, end=5 + dd)] if dd else []
            tr = bc.simulate(gn, x0=stuck, steps=50, runs=runs, scheme="synchronous", seed=SEED, interventions=ivs)
            S = tr.states[-1]
            dose[k].append(r3(((S[:, 1] == 0) & (S[:, 2] == 2)).mean()))
    out["doses"] = doses
    out["dose_response"] = dose
    # inference from simulated panel data
    gd = g.copy(); gd.set_rule("PublicStigma", bc.IdentityRule())
    gdn = gd.copy()
    for n in g.names[1:]:
        gdn.set_rule(n, bc.NoisyRule(gd.rules[n], 0.03))
    inf = {}
    for persons in (300, 1000):
        rng = np.random.default_rng(SEED + persons)
        x0 = gd.random_states(persons, rng)
        tr = bc.simulate(gdn, x0=x0, steps=11, runs=persons, scheme="synchronous", seed=SEED + persons)
        rv = bc.reveal(tr.states, g.names, max_parents=3, criterion="bic")
        true = {n: set(g.rules[n].inputs) for n in g.names}
        true["PublicStigma"] = {"PublicStigma"}
        tp = sum(len(set(rv.parents[n]) & true[n]) for n in g.names)
        fp = sum(len(set(rv.parents[n]) - true[n]) for n in g.names)
        fn = sum(len(true[n] - set(rv.parents[n])) for n in g.names)
        inf[persons] = {"parents": {n: list(rv.parents[n]) for n in g.names}, "precision": r3(tp / (tp + fp)),
                        "recall": r3(tp / (tp + fn)), "accuracy": {n: r3(v) for n, v in rv.accuracy.items()}}
    out["inference"] = inf
    res["case3"] = out

    # Figure 5
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.6), gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.55})
    cats = ["recovered", "intention without action", "stuck", "cycle (synchronous only)"]
    cols = [viz.POS, "#86b6ef", viz.NEG, "#b7b5ad"]
    left = np.zeros(3)
    for c, col in zip(cats, cols):
        v = np.array([land[ps]["sync"].get(c, 0) for ps in (0, 1, 2)])
        axs[0].barh([2, 1, 0], v, left=left, color=col, height=0.55, label=c)
        left += v
    axs[0].set_yticks([2, 1, 0], ["low", "moderate", "high"]); axs[0].set_ylabel("public stigma")
    axs[0].set_xlabel("share of initial states"); axs[0].set_xlim(0, 1)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
    axs[0].spines[["top", "right"]].set_visible(False)
    axs[0].set_title("A  Attractor landscape by context")
    styles = ["-", "--", "-.", ":"]
    for (k, v), ls, col in zip(dose.items(), styles, [viz.POS, viz.INK, "#6a6964", viz.NEG]):
        axs[1].plot(doses, v, ls=ls, marker="o", ms=3.5, color=col, lw=1.6, label=k)
    axs[1].set_xlabel("intervention length (update steps)"); axs[1].set_ylabel("P(recovered at t = 50)")
    axs[1].set_ylim(0, 1.02); axs[1].grid(axis="y", color=viz.GRID, lw=0.6)
    axs[1].spines[["top", "right"]].set_visible(False)
    axs[1].legend(frameon=False, fontsize=7.5, loc="center right")
    axs[1].set_title("B  Dose–response from the 'stuck' attractor")
    save(fig, "fig5_case3")
    bc.write_report(REP / "case3_stigma.md", g, cfg, sections={"Results": json.dumps(out, indent=1, default=str)})
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true", help="fewer Monte Carlo runs")
    args = ap.parse_args(argv)
    REP.mkdir(parents=True, exist_ok=True)
    res = {"environment": bc.environment_report(), "seed": SEED, "fast": args.fast}
    figure_framework()
    tutorial(res)
    case1(res, args.fast)
    case2(res)
    case3(res, args.fast)
    (ROOT / "paper" / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
