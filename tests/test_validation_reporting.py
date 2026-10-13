import numpy as np

import bcpnm as bc


def test_canalization_of_and():
    g = bc.PNMGraph()
    for n in "abc":
        g.add_node(n, bc.StateSpace.boolean())
    g.set_rule("a", bc.AND("b", "c")); g.set_rule("b", bc.IdentityRule()); g.set_rule("c", bc.IdentityRule())
    can = bc.canalizing_inputs(g, "a")
    assert ("b", 0, 0) in can and ("c", 0, 0) in can and ("b", 1, 1) not in can
    act = bc.input_activities(g, samples=4000, seed=0)["a"]
    assert np.isclose(act["b"], 0.5, atol=0.05)   # activity of each AND input is 1/2


def test_expert_consistency_and_derrida(small_bc):
    g = small_bc.to_graph()
    exp = [("A", "B", int(np.sign(small_bc.K[1, 0])))]
    df = bc.expert_consistency(g, exp, samples=3000)
    assert df.attrs["agreement"] == 1.0
    curve = bc.derrida_curve(bc.deterministic_graph(g), samples=300, seed=1)
    assert set(curve) == {1, 2, 3, 4}


def test_perturbations_do_not_mutate(small_bc):
    g = small_bc.to_graph()
    w0 = g.rules["B"].w.copy()
    g2 = bc.ScaleEdge("A", "B", 0.0).apply(g)
    assert np.array_equal(g.rules["B"].w, w0)
    assert g2.rules["B"].w[list(g2.rules["B"].inputs).index("A")] == 0
    g3 = bc.ShiftField("A", 1.0).apply(g)
    assert np.isclose(g3.rules["A"].h, g.rules["A"].h + 1)


def test_leverage_scan(repressilator):
    df = bc.leverage_scan(repressilator, target=lambda s: s[0] == 1, value_fn=lambda n: 1)
    assert set(df["node"]) == {"A", "B", "C"}
    assert df.loc[df["node"] == "A", "share"].item() == 1.0


def test_reporting(tmp_path, toggle):
    ck = bc.checklist()
    assert len(ck) >= 20 and {"section", "item", "report"} <= set(ck.columns)
    txt = bc.methods_text(toggle, scheme="synchronous", runs=10, steps=5, seed=1)
    assert "2 nodes" in txt
    res = bc.find_attractors(toggle)
    assert "attractor" in bc.results_text(res)
    p = bc.write_report(tmp_path / "r.md", toggle, sections={"Results": bc.results_text(res)})
    assert "PNM reporting checklist" in p.read_text()


def test_threshold_ties():
    g = bc.PNMGraph()
    for n in "ab":
        g.add_node(n, bc.StateSpace.boolean())
    g.set_rule("a", bc.ThresholdRule({"b": 1.0}, thresholds=(1.0,)))
    g.set_rule("b", bc.ThresholdRule({"a": 1.0}, thresholds=(0.5,)))
    df = bc.threshold_ties(g)
    assert df.set_index("node")["ties"].to_dict() == {"a": 1, "b": 0}
