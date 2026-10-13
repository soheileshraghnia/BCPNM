import numpy as np

import bcpnm as bc


def test_reveal_recovers_boolean_network(repressilator):
    tr = bc.simulate(repressilator, steps=6, runs=60, scheme="synchronous", seed=3)
    res = bc.reveal(tr.states, repressilator.names, max_parents=2, criterion="reveal")
    assert res.parents == {"A": ("C",), "B": ("A",), "C": ("B",)}
    g = res.to_graph(repressilator)
    assert len(bc.find_attractors(g)) == 2


def test_transition_model_parameter_recovery():
    names = list("ABCD")
    W = np.array([[0, .9, 0, 0], [.6, 0, 0, -.7], [0, 1.0, 0, 0], [0, 0, .8, 0]])
    L = 0.5 * (W != 0)
    g = bc.PNMGraph("true")
    for n in names:
        g.add_node(n, bc.StateSpace.ternary())
    for i, n in enumerate(names):
        g.set_rule(n, bc.SoftmaxRule({names[j]: W[i, j] for j in range(4) if W[i, j]},
                                     {names[j]: L[i, j] for j in range(4) if W[i, j]},
                                     h=0.1, delta=0.4, rho=0.5, beta=1.0))
    tr = bc.simulate(g, steps=60, runs=150, scheme="synchronous", seed=11)
    fit = bc.fit_transition_model(tr.states, names)
    rec = bc.parameter_recovery({"W": W}, {"W": fit.W})
    assert rec.loc[0, "correlation"] > 0.95
    assert rec.loc[0, "sign_agreement"] == 1.0
    assert np.allclose(fit.delta, 0.4, atol=0.15) and np.allclose(fit.rho, 0.5, atol=0.15)
    # predictive check: fitted model beats a model without network terms on new data
    test = bc.simulate(g, steps=60, runs=50, scheme="synchronous", seed=12)
    null = bc.fit_transition_model(tr.states, names, lam=False, alpha=1e6)
    assert bc.log_score(fit.to_graph(), test.states) > bc.log_score(null.to_graph(), test.states)


def test_equilibrium_pseudolikelihood(small_bc):
    m = bc.BlumeCapelModel(small_bc.names, small_bc.K, small_bc.Lam, small_bc.delta, small_bc.h, beta=1.0)
    S = m.gibbs(sweeps=6000, burn=500, chains=6, seed=4)
    fit = bc.fit_equilibrium(S, m.names)
    iu = np.triu_indices(4, 1)
    assert np.corrcoef(fit.K[iu], m.K[iu])[0, 1] > 0.95
    assert np.max(np.abs(fit.K[iu] - m.K[iu])) < 0.15
