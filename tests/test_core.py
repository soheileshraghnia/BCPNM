import numpy as np
import pytest

import bcpnm as bc
from bcpnm.rules import ThresholdRule


def test_state_space_validation():
    with pytest.raises(ValueError):
        bc.StateSpace([1, 0])
    with pytest.raises(ValueError):
        bc.StateSpace([0])
    s = bc.StateSpace.ternary(("conceal", "silent", "acknowledge"))
    assert s.size == 3 and s.labels[0] == "silent"
    assert list(s.index([-1, 1])) == [0, 2]
    assert 0 in s and 2 not in s


def test_encode_decode_roundtrip():
    g = bc.PNMGraph()
    g.add_node("a", bc.StateSpace.ternary())
    g.add_node("b", bc.StateSpace.boolean())
    g.add_node("c", bc.StateSpace.ordinal(4))
    for n in g.names:
        g.set_rule(n, bc.IdentityRule())
    X = g.all_states()
    assert X.shape == (24, 3)
    assert np.array_equal(g.decode(g.encode(X)), X)
    assert np.array_equal(g.encode(X), np.arange(24))


def test_validate_reports_problems():
    g = bc.PNMGraph()
    g.add_node("a", bc.StateSpace.boolean())
    with pytest.raises(ValueError):
        g.validate()
    g.set_rule("a", bc.BooleanRule("b"))
    with pytest.raises(ValueError):
        g.validate()


def test_boolean_and_case_rules():
    r = bc.BooleanRule("A and not B")
    P = np.array([[1, 0], [1, 1], [0, 0]])
    assert list(r.next_value(P, np.zeros(3), np.array([0, 1]))) == [1, 0, 0]
    c = bc.CaseRule([("D >= 2 and not S", 2), ("D >= 1", 1)], default=0)
    P = np.array([[2, 0], [2, 1], [1, 1], [0, 0]])
    assert c.inputs == ("D", "S")
    assert list(c.next_value(P, np.zeros(4), np.arange(3))) == [2, 1, 1, 0]


def test_expression_rejects_function_calls():
    r = bc.BooleanRule("__import__('os')")
    with pytest.raises(Exception):
        r.next_value(np.zeros((1, 0)), np.zeros(1), np.array([0, 1]))


def test_threshold_rule_and_ties():
    r = ThresholdRule({"a": 1, "b": -1}, thresholds=(0,), outputs=(0, 1))
    P = np.array([[1, 0], [0, 1], [1, 1], [1, 1]])
    out = r.next_value(P, np.array([0, 0, 0, 1]), np.array([0, 1]))
    assert list(out) == [1, 0, 0, 1]          # ties keep the current state
    m = bc.majority(["a", "b", "c"])
    P = np.array([[1, 1, -1], [-1, -1, 0], [1, -1, 0]])
    assert list(m.next_value(P, np.array([0, 0, 1]), np.array([-1, 0, 1]))) == [1, -1, 0]


def test_softmax_rule_probabilities_and_limit():
    r = bc.SoftmaxRule({"a": 0.8}, {"a": 0.5}, h=0.1, delta=0.3, rho=0.2, beta=1.5)
    P = np.array([[-1.0], [0.0], [1.0]])
    pr = r.probs(P, np.array([0, 0, 1]), np.array([-1, 0, 1]))
    assert np.allclose(pr.sum(1), 1)
    # hand computation for the second row: eta = 0.1, zeta = 0.5, current = 0
    u = 1.5 * np.array([-0.1 - 0.3, 0.5 + 0.2, 0.1 - 0.3])
    assert np.allclose(pr[1], np.exp(u) / np.exp(u).sum())
    det = bc.deterministic_limit(r)
    assert det.deterministic and not r.deterministic
    assert list(det.next_value(P, np.array([0, 0, 1]), np.array([-1, 0, 1]))) == list(np.argmax(pr, 1) - 1)


def test_noisy_and_pbn_rules():
    base = bc.BooleanRule("a")
    n = bc.NoisyRule(base, 0.2)
    pr = n.probs(np.array([[1], [0]]), np.zeros(2), np.array([0, 1]))
    assert np.allclose(pr, [[0.1, 0.9], [0.9, 0.1]])
    p = bc.PBNRule([bc.BooleanRule("a"), bc.BooleanRule("not b")], [0.7, 0.3])
    assert p.inputs == ("a", "b")
    pr = p.probs(np.array([[1, 1]]), np.zeros(1), np.array([0, 1]))
    assert np.allclose(pr, [[0.3, 0.7]])


def test_config_roundtrip(tmp_path):
    cfg = {
        "meta": {"name": "toy"},
        "state_spaces": {"talk": {"values": [-1, 0, 1], "labels": {-1: "conceal", 0: "silent", 1: "acknowledge"}}},
        "nodes": [{"name": "A", "states": "talk"}, {"name": "B", "states": "talk"}],
        "model": {"type": "softmax", "beta": 1.0, "defaults": {"delta": 0.3},
                  "edges": [{"source": "A", "target": "B", "w": 0.8, "lam": 0.5}], "symmetric": True},
    }
    bc.save_config(cfg, tmp_path / "toy.yaml")
    g, cfg2 = bc.load_model(tmp_path / "toy.yaml")
    assert g.rules["A"].inputs == ("B",) and np.isclose(g.rules["A"].w[0], 0.8)
    assert np.isclose(g.rules["B"].lam[0], 0.5)
    exported = bc.graph_to_config(g)
    g2 = bc.build_model(exported)
    assert np.isclose(g2.rules["B"].delta, 0.3)


def test_example_configs_load():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "examples"
    for path in sorted(root.glob("*.yaml")):
        g, cfg = bc.load_model(path)
        g.validate()
        assert g.N >= 5
        ties = bc.threshold_ties(g)
        assert ties.empty or ties["ties"].sum() == 0, path.name
