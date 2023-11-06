import numpy as np
from fastqaoa.ctypes.qaoa import qaoa
from fastqaoa.ctypes.optimize import optimize_qaoa_adam, optimize_qaoa_lbfgs
from fastqaoa.ctypes import Diagonals
from fastqaoa.ctypes.qpe_qaoa import qpe_qaoa
from fastqaoa.indicator import get_indicator_interpolator, interpolate_diagonals
import warnings
import qubovert as qv

from fastqaoa.ctypes.lib import NP_REAL

import networkx as nx

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pennylane as qml
    import pennylane.numpy as qnp


def test_opt_qaoa_min_vertex_cover():
    # replica of the hamiltonian
    puso = qv.PUSO(
        {
            (0,): 0.5,
            (1,): 0.5,
            (2,): 1.25,
            (3,): -0.25,
            (0, 1): 0.75,
            (0, 2): 0.75,
            (1, 2): 0.75,
            (2, 3): 0.75,
        }
    )

    b = puso.to_pubo()

    b = {sum(1 << i for i in k) if len(k) > 0 else 0: v for k,v in b.items()}
    keys = list(b.keys())
    vals = list(b.values())

    dg = Diagonals.brute_force(4, keys, vals)

    betas = np.ones(5) * 0.1
    gammas = np.ones(5) * 0.1

    a = np.abs(qaoa(dg, betas, gammas).to_numpy()) ** 2
    a = a.dot(dg.to_numpy())

    betas, gammas = optimize_qaoa_adam(dg, dg, betas, gammas)

    b = np.abs(qaoa(dg, betas, gammas).to_numpy()) ** 2
    b = b.dot(dg.to_numpy())

    assert b < a

def test_lbfgs_qaoa_min_vertex_cover():
    # replica of the hamiltonian
    puso = qv.PUSO(
        {
            (0,): 0.5,
            (1,): 0.5,
            (2,): 1.25,
            (3,): -0.25,
            (0, 1): 0.75,
            (0, 2): 0.75,
            (1, 2): 0.75,
            (2, 3): 0.75,
        }
    )

    b = puso.to_pubo()

    b = {sum(1 << i for i in k) if len(k) > 0 else 0: v for k,v in b.items()}
    keys = list(b.keys())
    vals = list(b.values())

    dg = Diagonals.brute_force(4, keys, vals)

    betas = np.ones(5) * 0.1
    gammas = np.ones(5) * 0.1

    a = np.abs(qaoa(dg, betas, gammas).to_numpy()) ** 2
    a = a.dot(dg.to_numpy())

    _, betas, gammas = optimize_qaoa_lbfgs(dg, dg, betas, gammas, maxiter=10)

    b = np.abs(qaoa(dg, betas, gammas).to_numpy()) ** 2
    b = b.dot(dg.to_numpy())

    assert b < a



def test_opt_qaoa_min_vertex_cover2():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    graph = nx.Graph(edges)
    cost_h, mixer_h = qml.qaoa.min_vertex_cover(graph, constrained=False)

    betas = np.ones(5) * 0.1
    gammas = np.ones(5) * 0.1

    tbetas = qnp.array(betas, requires_grad=True)
    tgammas = qnp.array(gammas, requires_grad=True)

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(betas, gammas):
        for w in range(4):
            qml.Hadamard(wires=w)
        for beta, gamma in zip(betas, gammas):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(beta, mixer_h)
        return qml.expval(cost_h)

    opt = qml.AdamOptimizer()

    for _ in range(10):
        (tbetas, tgammas), _ = opt.step_and_cost(circuit, tbetas, tgammas)
    cost = circuit(tbetas, tgammas)

    # replica of the hamiltonian:
    puso = qv.PUSO(
        {
            (0,): 0.5,
            (1,): 0.5,
            (2,): 1.25,
            (3,): -0.25,
            (0, 1): 0.75,
            (0, 2): 0.75,
            (1, 2): 0.75,
            (2, 3): 0.75,
        }
    )

    b = puso.to_pubo()

    b = {sum(1 << i for i in k) if len(k) > 0 else 0: v for k,v in b.items()}
    keys = list(b.keys())
    vals = list(b.values())

    dg = Diagonals.brute_force(4, keys, vals)

    betas, gammas = optimize_qaoa_adam(dg, dg, betas, gammas, maxiter=10)

    b = np.abs(qaoa(dg, betas, gammas).to_numpy()) ** 2
    cost2 = b.dot(dg.to_numpy())

    atol = 1e-8
    if NP_REAL == np.float32:
        atol = 1e-6

    assert np.isclose(cost, cost2, atol=atol)
    assert np.allclose(tbetas, betas, atol=atol)
    assert np.allclose(tgammas, gammas, atol=atol)


def test_opt_qpe_qaoa():
    N = 4
    M = 4

    gd = 2 * np.random.rand(1 << N)
    gs = Diagonals.from_numpy(gd)

    fd = 2 * np.random.rand(1 << N)
    fs = Diagonals.from_numpy(fd)

    cd = 2 * np.random.rand(1 << N)
    cs = Diagonals.from_numpy(cd)

    interpolator = get_indicator_interpolator(M, 6)
    constr = interpolate_diagonals(interpolator, gs)

    betas = np.random.rand(10)
    gammas = np.random.rand(10)

    sv, _ = qpe_qaoa(fs, constr, betas, gammas)
    obj = cd.dot(np.abs(sv.to_numpy()) ** 2)
    betas, gammas = optimize_qaoa_adam(fs, cs, betas, gammas, constr=constr, maxiter=100)
    sv, _ = qpe_qaoa(fs, constr, betas, gammas)
    assert cd.dot(np.abs(sv.to_numpy()) ** 2) < obj

def test_opt_qpe_qaoa_lbfgs():
    N = 4
    M = 4

    gd = 2 * np.random.rand(1 << N)
    gs = Diagonals.from_numpy(gd)

    fd = 2 * np.random.rand(1 << N)
    fs = Diagonals.from_numpy(fd)

    cd = 2 * np.random.rand(1 << N)
    cs = Diagonals.from_numpy(cd)

    interpolator = get_indicator_interpolator(M, 6)
    constr = interpolate_diagonals(interpolator, gs)

    betas = np.random.rand(10)
    gammas = np.random.rand(10)

    sv, _ = qpe_qaoa(fs, constr, betas, gammas)
    obj = cd.dot(np.abs(sv.to_numpy()) ** 2)
    _, betas, gammas = optimize_qaoa_lbfgs(fs, cs, betas, gammas, constr=constr)

    sv, _ = qpe_qaoa(fs, constr, betas, gammas)
    assert cd.dot(np.abs(sv.to_numpy()) ** 2) < obj
