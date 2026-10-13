import numpy as np

import bcpnm as bc


def test_seed_reproducibility(small_bc):
    g = small_bc.to_graph()
    a = bc.simulate(g, steps=50, runs=20, scheme="asynchronous", seed=7)
    b = bc.simulate(g, steps=50, runs=20, scheme="asynchronous", seed=7)
    c = bc.simulate(g, steps=50, runs=20, scheme="asynchronous", seed=8)
    assert np.array_equal(a.states, b.states)
    assert not np.array_equal(a.states, c.states)


def test_synchronous_toggle_oscillates(toggle):
    tr = bc.simulate(toggle, x0=[0, 0], steps=4, scheme="synchronous")
    assert tr.states[:, 0, :].tolist() == [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0]]


def test_asynchronous_changes_at_most_one_node(small_bc):
    tr = bc.simulate(small_bc.to_graph(), steps=200, runs=50, scheme="asynchronous", seed=1)
    changes = (np.diff(tr.states.astype(int), axis=0) != 0).sum(axis=2)
    assert changes.max() <= 1


def test_sequential_sweep_updates_every_node(repressilator):
    tr = bc.simulate(repressilator, x0=[0, 0, 0], steps=1, runs=10, scheme="sequential", seed=0)
    assert tr.states.shape == (2, 10, 3)


def test_time_windowed_intervention(toggle):
    iv = bc.Intervention(bc.Clamp("A", 1), start=2, end=5)
    tr = bc.simulate(toggle, x0=[0, 1], steps=8, scheme="synchronous", interventions=[iv])
    A = tr.node("A")[:, 0].tolist()
    assert A[:3] == [0, 0, 0] and A[3:6] == [1, 1, 1]
    # after the clamp is released the network stays in the A-on fixed point
    assert tr.states[-1, 0].tolist() == [1, 0]


def test_trajectory_helpers(small_bc):
    tr = bc.simulate(small_bc.to_graph(), steps=10, runs=5, scheme="synchronous", seed=2)
    assert tr.T == 10 and tr.runs == 5
    assert tr.prob(0).shape == (11, 4)
    assert np.allclose(tr.prob(-1) + tr.prob(0) + tr.prob(1), 1)
    df = tr.to_long()
    assert len(df) == 11 * 5 * 4
