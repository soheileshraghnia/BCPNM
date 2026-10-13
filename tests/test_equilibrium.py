import numpy as np

import bcpnm as bc
from bcpnm.equilibrium import continuation, mean_field


def test_beg_identity(small_bc):
    X = small_bc.configurations()
    b = small_bc.beg_form()
    Q = X ** 2
    E = (-0.5 * np.einsum("ri,ij,rj->r", X, b["J"], X) - 0.5 * np.einsum("ri,ij,rj->r", Q, b["biquadratic"], Q)
         + Q @ b["delta_effective"] - X @ b["h"] + b["constant"])
    assert np.allclose(E, small_bc.energy(X))


def test_log_odds_formula(small_bc):
    X = small_bc.configurations()
    i = 2
    eta = small_bc.h[i] + X @ small_bc.K[i]
    zeta = (1 - X ** 2) @ small_bc.Lam[i]
    lo = small_bc.log_odds_zero(X, i)
    assert np.allclose(lo[:, 0], small_bc.beta * (small_bc.delta[i] + zeta - eta))
    assert np.allclose(lo[:, 1], small_bc.beta * (small_bc.delta[i] + zeta + eta))


def test_zero_coordination_raises_collective_silence():
    out = {}
    for lam in (0.0, 0.8):
        K = 0.6 * (np.ones((3, 3)) - np.eye(3))
        L = lam * (np.ones((3, 3)) - np.eye(3))
        m = bc.BlumeCapelModel(list("ABC"), K, L, delta=0.3, beta=1.0)
        X, p = m.exact()
        out[lam] = (p[(X == 0).all(1)].sum(), m.occupancy(X, 0, p))
    assert np.isclose(out[0.0][0], 0.0513, atol=5e-4) and np.isclose(out[0.8][0], 0.3151, atol=5e-4)
    assert np.isclose(out[0.0][1], 0.3365, atol=5e-4) and np.isclose(out[0.8][1], 0.5780, atol=5e-4)


def test_tricritical_point():
    beta, delta = 3.0, np.log(4) / 3
    q0 = 2 * np.exp(-beta * delta) / (1 + 2 * np.exp(-beta * delta))
    assert np.isclose(beta * q0, 1.0)
    # just inside the ordered region the mean-field magnetization is nonzero
    m, _ = mean_field(3.3, 1.0, 0.0, 0.3, m0=0.5, q0=0.8)
    assert abs(m) > 0.1
    m, _ = mean_field(1.5, 1.0, 0.0, 0.3, m0=0.5, q0=0.8)
    assert abs(m) < 1e-4


def test_polarity_hysteresis():
    sw = continuation("h", np.linspace(-0.7, 0.7, 281), beta=1, J=4, Lam=0, delta=1.6)
    up, down = sw.jumps(0)
    assert np.isclose(up, 0.335, atol=0.01) and np.isclose(down, -0.335, atol=0.01)


def test_silence_hysteresis():
    sw = continuation("delta", np.linspace(-4, 1, 251), beta=1, J=0.5, Lam=5, delta=0)
    up, down = sw.jumps(1)
    assert np.isclose(up, -1.64, atol=0.03) and np.isclose(down, -1.98, atol=0.03)
