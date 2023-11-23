import warnings
import jax
import jax.numpy as jnp
import numpy as np

from fastqaoa.ctypes import Diagonals
from fastqaoa.ctypes.qpe_qaoa import apply_qpe_diagonals, apply_qpe_diagonals_normalized, grad_qpe_qaoa, qpe_qaoa
from fastqaoa.ctypes.statevector import Statevector
from fastqaoa.indicator import get_indicator_interpolator, interpolate_diagonals
from fastqaoa.ctypes.lib import NP_REAL
from fastqaoa.utils.jax_config import set_accuracy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pennylane as qml
    from pennylane import DiagonalQubitUnitary

set_accuracy(64)

def allclose(a, b):
    atol = 1e-8
    if NP_REAL == np.float64:
        atol = 1e-4
    return np.allclose(a, b, atol=atol)

def isclose(a, b):
    atol = 1e-8
    if NP_REAL == np.float64:
        atol = 1e-4
    return np.isclose(a, b, atol=atol)

def construct_pennylane_circuit(dev, N, M, fs: Diagonals, gs: Diagonals):
    gd = gs.to_numpy()
    fd = fs.to_numpy()

    wires = range(M, N + M)
    est_wires = range(M)

    gdops = []
    for i in range(M):
        gdops.append(
            DiagonalQubitUnitary(np.exp(1j * np.pi * gd * 2**i), wires=wires)
        )

    def apply_gamma(gamma):
        for w in est_wires:
            qml.Hadamard(wires=w)

        for i in est_wires:
            qml.ControlledQubitUnitary(gdops[i], control_wires=[i])

        qftidx = list(reversed(range(M)))
        qml.adjoint(qml.QFT(wires=qftidx))
        #
        qml.PauliX(M - 1)
        qml.ControlledQubitUnitary(
            DiagonalQubitUnitary(jnp.exp(1j * fd * gamma), wires=wires),
            control_wires=[M - 1],
        )
        qml.PauliX(M - 1)

        qml.QFT(wires=qftidx)

        for i in range(M):
            qml.adjoint(
                qml.ControlledQubitUnitary(gdops[i], control_wires=[i])
            )

        for w in range(M):
            qml.Hadamard(wires=w)

    @qml.qnode(dev, interface="jax")
    def gen_state(gamma):
        for w in wires:
            qml.Hadamard(wires=w)
        apply_gamma(gamma)

        return qml.state()

    return apply_gamma, gen_state

def test_apply_qpe_diagonals():
    N = 4
    M = 4

    dev = qml.device("default.qubit", wires=range(M + N))

    gd = 2 * np.random.rand(1 << N)
    gs = Diagonals.from_numpy(gd)

    fd = 2 * np.random.rand(1 << N)
    fs = Diagonals.from_numpy(fd)


    _, gen_state = construct_pennylane_circuit(dev, N, M, fs, gs)

    def extract_state(gamma):
        res = gen_state(gamma)
        return res[:1<<N]

    gamma = 1.3252

    s = extract_state(gamma)
    p = np.sum(np.abs(s) ** 2)

    ## My stuff
    sv = Statevector.make_plus_state(N)

    interp = get_indicator_interpolator(M, 6)
    constr = interpolate_diagonals(interp, gs)

    apply_qpe_diagonals(sv, fs, constr, gamma)

    sv = sv.to_numpy()
    p2 = np.sum(np.abs(sv) ** 2)

    assert isclose(p, p2)
    assert isclose(np.abs(sv.conj().dot(s)), p)

    sv = Statevector.make_plus_state(N)
    p3 = apply_qpe_diagonals_normalized(sv, fs, constr, gamma)
    sv = sv.to_numpy()

    assert isclose(p, p3)
    s /= np.sqrt(p)
    assert isclose(np.abs(sv.conj().dot(s)), 1)


def test_qpe_qaoa():
    N = 4
    M = 4

    dev = qml.device("default.qubit", wires=range(M + N))

    gd = 2 * np.random.rand(1 << N)
    gs = Diagonals.from_numpy(gd)

    fd = 2 * np.random.rand(1 << N)
    fs = Diagonals.from_numpy(fd)

    apply_gamma, _ = construct_pennylane_circuit(dev, N, M, fs, gs)

    wires = range(M, N + M)

    @qml.qnode(dev, interface="jax")
    def layer(state, gamma, beta):
        qml.QubitStateVector(state, wires=range(N + M))
        apply_gamma(gamma)
        for w in wires:
            qml.RX(2 * beta, w)
        return qml.state()

    betas = np.random.rand(4)
    gammas = np.random.rand(4)

    state = np.zeros(1 << (M + N), dtype=np.complex128)
    state[:1 << N] = 1 / np.sqrt(1 << N)
    ps =[]
    for beta, gamma in zip(betas, gammas):
        state = layer(state, gamma, beta)
        state = state.at[1<<N:].set(0)
        p = jnp.sum(jnp.abs(state) ** 2)
        ps.append(float(p))
        state = state / jnp.sqrt(p)

    interpolator = get_indicator_interpolator(M, 6)
    constr = interpolate_diagonals(interpolator, gs)
    sv, p = qpe_qaoa(fs, constr, betas, gammas)

    sv = sv.to_numpy()

    assert isclose(p, np.prod(ps))

    assert isclose(np.abs(sv.conj().dot(state[:1<<N])), 1)

def test_grad_qpe_qaoa():
    N = 4
    M = 4

    dev = qml.device("default.qubit", wires=range(M + N))

    gd = 2 * np.random.rand(1 << N)
    gs = Diagonals.from_numpy(gd)

    fd = 2 * np.random.rand(1 << N)
    fs = Diagonals.from_numpy(fd)

    cd = 2 * np.random.rand(1 << N)
    cs = Diagonals.from_numpy(cd)

    apply_gamma, _ = construct_pennylane_circuit(dev, N, M, fs, gs)

    wires = range(M, N + M)

    @qml.qnode(dev, interface="jax")
    def layer(state, gamma, beta):
        qml.QubitStateVector(state, wires=range(N + M))
        apply_gamma(gamma)
        for w in wires:
            qml.RX(2 * beta, w)
        return qml.state()

    betas = np.random.rand(5)
    gammas = np.random.rand(5)

    def func(betas, gammas):
        state = np.zeros(1 << (M + N), dtype=np.complex128)
        state[:1 << N] = 1 / np.sqrt(1 << N)

        ptot = 1.
        for beta, gamma in zip(betas, gammas):
            state = layer(state, gamma, beta)
            state = state.at[1<<N:].set(0)
            p = jnp.sum(jnp.abs(state) ** 2)
            ptot *= p
            state = state / jnp.sqrt(p)

        return ptot * jnp.dot(cd, jnp.abs(state[:1 << N]) ** 2) / ptot

    pl_beta_gradients, pl_gamma_gradients = jax.grad(func, argnums=(0, 1))(betas, gammas)

    interpolator = get_indicator_interpolator(M, 6)
    constr = interpolate_diagonals(interpolator, gs)
    beta_gradients, gamma_gradients = grad_qpe_qaoa(fs, cs, constr, betas, gammas)

    assert allclose(pl_beta_gradients, beta_gradients)
    assert allclose(pl_gamma_gradients, gamma_gradients)
