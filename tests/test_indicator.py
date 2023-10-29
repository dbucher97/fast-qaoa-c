import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from fastqaoa.indicator import get_indicator_base, get_indicator_interpolator
from fastqaoa.utils.jax_config import set_accuracy
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pennylane as qml


set_accuracy(64)

def test_indicator_base():
    M = 7
    subdiv = 4

    x = 2 * np.linspace(0, 1, 2 ** (M + subdiv), endpoint=False)
    ft = get_indicator_base(M, subdiv)

    wires = range(M + 1)
    dev = qml.device("default.qubit", wires=wires)
    @qml.qnode(dev, interface="jax")
    def gen_state(g):
        qml.PauliX(M)
        for w in range(M):
            qml.Hadamard(wires=w)
        for i in range(M):
            qml.PhaseShift(2 ** i * g * np.pi, i)
        qftidx = list(reversed(range(M)))
        qml.adjoint(qml.QFT(wires=qftidx))

        qml.PauliX(M - 1)
        qml.ControlledPhaseShift(np.pi, wires=[M-1, M])
        qml.PauliX(M - 1)

        qml.QFT(wires=qftidx)
        for i in range(M):
            qml.PhaseShift(-2 ** i * g * np.pi, i)
        for w in range(M):
            qml.Hadamard(wires=w)
        return qml.state()

    @jax.jit
    @jax.vmap
    def func(g):
        st = gen_state(g)
        return st[1].real

    res = func(x)

    assert jnp.allclose(res, -ft)

def test_indicator_interpolation():
    M = 4

    wires = range(M + 1)
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface="jax")
    def gen_state(g):
        qml.PauliX(M)
        for w in range(M):
            qml.Hadamard(wires=w)
        for i in range(M):
            qml.PhaseShift(2 ** i * g * np.pi, i)
        qftidx = list(reversed(range(M)))
        qml.adjoint(qml.QFT(wires=qftidx))

        qml.PauliX(M - 1)
        qml.ControlledPhaseShift(np.pi, wires=[M-1, M])
        qml.PauliX(M - 1)

        qml.QFT(wires=qftidx)
        for i in range(M):
            qml.PhaseShift(-2 ** i * g * np.pi, i)
        for w in range(M):
            qml.Hadamard(wires=w)
        return qml.state()


    @jax.jit
    @jax.vmap
    def func(g):
        st = gen_state(g)
        return st[1].real

    gs = np.linspace(0, 2, 100)

    rest = func(gs)
    plt.plot(gs, rest)


    # plt.plot(get_indicator_base(M, 1))
    indicator1 = get_indicator_interpolator(M, 1)
    indicator2 = get_indicator_interpolator(M, 2)
    indicator3 = get_indicator_interpolator(M, 3)
    indicator4 = get_indicator_interpolator(M, 4)

    res1 = indicator1(gs)
    res2 = indicator2(gs)
    res3 = indicator3(gs)
    res4 = indicator4(gs)

    d1 = np.mean(np.abs(res1 + rest) ** 2)
    d2 = np.mean(np.abs(res2 + rest) ** 2)
    d3 = np.mean(np.abs(res3 + rest) ** 2)
    d4 = np.mean(np.abs(res4 + rest) ** 2)

    assert d1 < 1e-2
    assert d2 < d1
    assert d3 < d2
    assert d4 < d3
    assert d4 < 1e-11

def test_indicator_application():
    M = 4

    wires = range(M + 1)
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface="jax")
    def gen_state(g, f):
        qml.PauliX(M)
        for w in range(M):
            qml.Hadamard(wires=w)
        for i in range(M):
            qml.PhaseShift(2 ** i * g * np.pi, i)
        qftidx = list(reversed(range(M)))
        qml.adjoint(qml.QFT(wires=qftidx))

        qml.PauliX(M - 1)
        qml.ControlledPhaseShift(np.pi * f, wires=[M-1, M])
        qml.PauliX(M - 1)

        qml.QFT(wires=qftidx)
        for i in range(M):
            qml.PhaseShift(-2 ** i * g * np.pi, i)
        for w in range(M):
            qml.Hadamard(wires=w)
        return qml.state()

    @jax.jit
    @jax.vmap
    def func(g, f):
        st = gen_state(g, f)
        return st[1]

    gs = np.random.rand(1000) * 2
    fs = np.random.rand(1000) * 2

    indicator = get_indicator_interpolator(M, 6)

    a = 0.5 * np.exp(1j * np.pi * fs)
    resm = (a + 0.5) + (a - 0.5) * indicator(gs)

    rest = func(gs, fs)

    assert np.allclose(resm, rest)
