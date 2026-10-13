import numpy as np
import pytest

import bcpnm as bc


@pytest.fixture
def toggle():
    g = bc.PNMGraph("toggle")
    for n in "AB":
        g.add_node(n, bc.StateSpace.boolean())
    g.set_rule("A", bc.BooleanRule("not B"))
    g.set_rule("B", bc.BooleanRule("not A"))
    return g


@pytest.fixture
def repressilator():
    g = bc.PNMGraph("repressilator")
    for n in "ABC":
        g.add_node(n, bc.StateSpace.boolean())
    g.set_rule("A", bc.NOT("C"))
    g.set_rule("B", bc.NOT("A"))
    g.set_rule("C", bc.NOT("B"))
    return g


@pytest.fixture
def small_bc():
    rng = np.random.default_rng(3)
    N = 4
    K = rng.normal(0, 0.6, (N, N)); K = (K + K.T) / 2
    L = np.abs(rng.normal(0, 0.5, (N, N))); L = (L + L.T) / 2
    return bc.BlumeCapelModel(list("ABCD"), K, L, delta=rng.normal(0.3, 0.2, N),
                              h=rng.normal(0, 0.3, N), beta=1.2)
