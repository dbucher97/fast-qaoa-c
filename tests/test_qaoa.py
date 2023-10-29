import warnings

import jax.numpy as jnp
import jax
import networkx as nx
import numpy as np
import qubovert as qv

from fastqaoa.ctypes import Diagonals, Statevector
from fastqaoa.ctypes.qaoa import apply_diagonals, apply_rx, grad_qaoa, qaoa
from fastqaoa.utils.jax_config import set_accuracy

set_accuracy(64)



with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pennylane as qml


def test_apply_diagonals():
    N = 5
    sv = Statevector.make_plus_state(N)
    npsv = sv.to_numpy()

    npdiags = np.linspace(0, 1, 1 << N)
    diags = Diagonals.from_numpy(npdiags)

    gamma = 0.37

    apply_diagonals(sv, diags, gamma)
    res = sv.to_numpy()

    assert np.allclose(res, npsv * np.exp(-1j * gamma * npdiags))


def test_apply_rx():
    N = 5

    npsv = np.random.randn(1 << N) + 1j * np.random.randn(1 << N)
    npsv /= np.sqrt((np.abs(npsv) ** 2).sum())
    sv = Statevector.from_numpy(npsv)

    beta = 0.37234

    apply_rx(sv, beta)
    res = sv.to_numpy()

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="jax")
    def circuit():
        qml.QubitStateVector(npsv, range(N))
        for i in range(N):
            qml.RX(2 * beta, wires=i)
        return qml.state()

    plres = circuit()

    assert np.allclose(res, plres)


def test_qaoa():
    N = 5

    npdiags = np.random.randn(1 << N)
    diags = Diagonals.from_numpy(npdiags)

    betas = np.random.rand(5)
    gammas = np.random.rand(5)

    sv = qaoa(diags, betas, gammas)
    res = sv.to_numpy()

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="jax")
    def circuit():
        for i in range(N):
            qml.Hadamard(i)
        for beta, gamma in zip(betas, gammas):
            qml.DiagonalQubitUnitary(np.exp(-1j * npdiags * gamma), wires=range(N))
            for i in range(N):
                qml.RX(2 * beta, wires=i)
        return qml.state()

    plres = circuit()

    assert np.isclose(1, np.abs(res.dot(plres.conj())))


def test_grad_qaoa():
    N = 5

    npdiags = np.random.randn(1 << N)
    diags = Diagonals.from_numpy(npdiags)

    npcost = np.random.randn(1 << N)
    cost = Diagonals.from_numpy(npcost)

    betas = np.random.rand(5)
    gammas = np.random.rand(5)

    tbetas = jnp.array(betas)
    tgammas = jnp.array(gammas)

    v, grad_betas, grad_gammas = grad_qaoa(diags, cost, betas, gammas)

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="jax")
    def circuit(betas, gammas):
        for i in range(N):
            qml.Hadamard(i)
        for beta, gamma in zip(betas, gammas):
            qml.DiagonalQubitUnitary(jnp.exp(-1j * npdiags * gamma), wires=range(N))
            for i in range(N):
                qml.RX(2 * beta, wires=i)
        return qml.expval(qml.Hermitian(np.diag(npcost), wires=range(N)))

    plres = circuit(tbetas, tgammas)

    assert np.isclose(plres, v)

    plgbetas, plggammas = jax.grad(circuit, (0, 1))(tbetas, tgammas)

    assert np.allclose(grad_betas, plgbetas)
    assert np.allclose(grad_gammas, plggammas)


def test_apply_diagonals_min_vertex_cover():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    graph = nx.Graph(edges)
    cost_h, _ = qml.qaoa.min_vertex_cover(graph, constrained=False)

    gamma = 0.9042

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev, interface="jax")
    def circuit():
        for w in range(4):
            qml.Hadamard(wires=w)
        qml.qaoa.cost_layer(gamma, cost_h)
        for w in range(2):
            qml.SWAP((w, 3 - w))
        return qml.state()

    state = circuit()

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

    b = {sum(1 << i for i in k) if len(k) > 0 else 0: v for k, v in b.items()}
    keys = list(b.keys())
    vals = list(b.values())

    sv = Statevector.make_plus_state(4)
    dg = Diagonals.brute_force(4, keys, vals)

    apply_diagonals(sv, dg, gamma)

    npsv = sv.to_numpy()

    assert np.isclose(np.abs(npsv.conj().dot(state)), 1)


def test_qaoa_min_vertex_cover():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    graph = nx.Graph(edges)
    cost_h, mixer_h = qml.qaoa.min_vertex_cover(graph, constrained=False)

    betas = np.random.rand(5)
    gammas = np.random.rand(5)

    # tbetas = qnp.array(betas, requires_grad=True)
    # tgammas = qnp.array(gammas, requires_grad=True)

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev, interface="jax")
    def circuit(betas, gammas):
        for w in range(4):
            qml.Hadamard(wires=w)
        for beta, gamma in zip(betas, gammas):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(beta, mixer_h)
        for w in range(2):
            qml.SWAP((w, 3 - w))
        return qml.state()

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

    b = {sum(1 << i for i in k) if len(k) > 0 else 0: v for k, v in b.items()}
    keys = list(b.keys())
    vals = list(b.values())

    dg = Diagonals.brute_force(4, keys, vals)

    sv = qaoa(dg, betas, gammas)
    npsv = sv.to_numpy()

    state = circuit(betas, gammas)

    assert np.isclose(np.abs(npsv.conj().dot(state)), 1)


def test_grad_qaoa_min_vertex_cover():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    graph = nx.Graph(edges)
    cost_h, mixer_h = qml.qaoa.min_vertex_cover(graph, constrained=False)

    betas = np.random.rand(5)
    gammas = np.random.rand(5)

    tbetas = jnp.array(betas)
    tgammas = jnp.array(gammas)

    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev, interface="jax")
    def circuit(betas, gammas):
        for w in range(4):
            qml.Hadamard(wires=w)
        for beta, gamma in zip(betas, gammas):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(beta, mixer_h)
        return qml.expval(cost_h)

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

    b = {sum(1 << i for i in k) if len(k) > 0 else 0: v for k, v in b.items()}
    keys = list(b.keys())
    vals = list(b.values())

    dg = Diagonals.brute_force(4, keys, vals)

    val, grad_betas, grad_gammas = grad_qaoa(dg, dg, betas, gammas)

    plval = circuit(betas, gammas)
    pl_grad_betas, pl_grad_gammas = jax.grad(circuit, (0, 1))(tbetas, tgammas)

    assert np.isclose(plval, val)
    assert np.allclose(pl_grad_betas, grad_betas)
    assert np.allclose(pl_grad_gammas, grad_gammas)
