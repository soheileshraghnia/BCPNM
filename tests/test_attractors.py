import numpy as np
import pytest

import bcpnm as bc


def test_toggle_synchronous(toggle):
    res = bc.find_attractors(toggle)
    kinds = sorted(a.kind for a in res)
    assert kinds == ["cycle", "fixed point", "fixed point"]
    assert np.isclose(sum(a.basin_fraction for a in res), 1)


def test_toggle_asynchronous(toggle):
    res = bc.find_attractors(toggle, scheme="asynchronous")
    fps = sorted(tuple(a.states[0]) for a in res)
    assert fps == [(0, 1), (1, 0)]
    assert all(a.weak_basin == 3 for a in res)
    assert all(a.basin == 1 for a in res)


def test_repressilator_cycles(repressilator):
    res = bc.find_attractors(repressilator)
    assert sorted(a.size for a in res) == [2, 6]
    ares = bc.find_attractors(repressilator, scheme="asynchronous")
    assert len(ares) == 1 and ares.attractors[0].size == 6 and ares.attractors[0].kind == "complex"


def test_sampling_agrees_with_exhaustive(repressilator):
    ex = bc.find_attractors(repressilator, method="exhaustive")
    sm = bc.find_attractors(repressilator, method="sampling", samples=400, seed=1)
    assert sorted(a.size for a in ex) == sorted(a.size for a in sm)


def test_attractor_search_needs_deterministic_rules(small_bc):
    with pytest.raises(ValueError):
        bc.find_attractors(small_bc.to_graph())
    det = bc.deterministic_graph(small_bc.to_graph())
    assert len(bc.find_attractors(det)) >= 1


def test_stationary_equals_boltzmann(small_bc):
    X, p = small_bc.exact()
    g = small_bc.to_graph()
    st = bc.stationary_distribution(g)
    assert np.allclose(st.pi[g.encode(X)], p, atol=1e-9)


def test_gibbs_matches_exact(small_bc):
    X, p = small_bc.exact()
    S = small_bc.gibbs(sweeps=2500, burn=200, chains=8, seed=1)
    assert abs(small_bc.occupancy(S, 0) - small_bc.occupancy(X, 0, p)) < 0.02


def test_quasi_attractors_mass(small_bc):
    g = small_bc.to_graph()
    qa = bc.quasi_attractors(g)
    total = sum(a.basin_fraction for a in qa)
    assert 0 < total <= 1 + 1e-9


def test_noisy_boolean_network_stationary(toggle):
    g = toggle.copy()
    for n in g.names:
        g.set_rule(n, bc.NoisyRule(g.rules[n], 0.05))
    # synchronous: the 2-cycle 00 <-> 11 is as attractive as the fixed points (exactly 1/2)
    st = bc.stationary_distribution(g, scheme="synchronous")
    assert np.isclose(st.pi.sum(), 1)
    assert np.isclose(st.prob(lambda X: X[:, 0] != X[:, 1]), 0.5)
    # asynchronous: the two fixed points carry most of the mass
    st = bc.stationary_distribution(g, scheme="asynchronous")
    assert st.prob(lambda X: X[:, 0] != X[:, 1]) > 0.9
