import circuit_lengths as cl
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.synthesis.qft import synth_qft_full

from problems import IntegerKnapsack, Knapsack


def optimal_edge_coloring(n: int):
    if n < 2:
        return {}
    odd = n % 2 == 1
    N = n + 1 if odd else n
    colors = {i: [] for i in range(N)}
    vertices = list(range(N))
    rounds = N - 1
    for r in range(rounds):
        for i in range(N // 2):
            u = vertices[i]
            v = vertices[N - 1 - i]
            if odd and (u == N - 1 or v == N - 1):
                continue
            if u < n and v < n:
                colors[r].append((min(u, v), max(u, v)))
        vertices = [vertices[0]] + [vertices[-1]] + vertices[1:-1]
    return colors


def optimal_bipartite_edge_coloring(N: int, M: int):
    colors = {i: [] for i in range(max(N, M))}
    for i in range(N):
        for j in range(M):
            color = (i + j) % max(N, M)
            colors[color].append((i, j))
    return colors


def penalty_layer(problem: IntegerKnapsack, M: int | None = None):
    if M is None:
        M = cl.get_M_penalty(problem)
    qubo = problem.slack_problem(M)

    N = problem.n_qubits + M

    qc = QuantumCircuit(N)
    maxval = max(map(abs, qubo.values()))
    qubo *= 10 * (N + M) / maxval

    gamma = Parameter("g")
    beta = Parameter("b")

    for i in range(0, N):
        qc.p(gamma * qubo[i,], i)

    coloring = optimal_edge_coloring(N)
    for c in coloring.values():
        for i, j in c:
            qc.cp(gamma * qubo[i, j], i, j)

    qc.rx(-2 * beta, qc.qubits)
    return qc


def qpe_phase_part(
    qc: QuantumCircuit,
    x: QuantumRegister,
    qpe: QuantumRegister,
    coloring: dict,
    values: list[float],
    offset: float,
    inverse: bool = False,
    with_hp: bool = True,
):
    iter = coloring.values()
    rev = 1
    if inverse:
        iter = reversed(coloring.values())
        rev = -1
    if with_hp:
        for qi in range(len(qpe)):
            angle = rev * 2**qi * offset
            qc.p(angle, qpe[qi])
    for color in iter:
        for xi, qi in color:
            angle = rev * 2**qi * values[xi]
            qc.cp(angle, x[xi], qpe[qi])


def qpe_cost_part(
    qc: QuantumCircuit,
    x: QuantumRegister,
    qpe: QuantumRegister,
    anc: QuantumRegister,
    values: list[float],
    gamma: Parameter,
):
    avail = [qpe[-1]]
    anciter = iter(anc)
    ancs = [qpe[-1], *anc]
    log = []
    while True:
        avail_c = [c for c in avail]
        new = None
        log_inner = []
        for s in avail:
            new = next(anciter, None)
            if new is None:
                break
            qc.cx(s, new)
            log_inner.append((s, new))
            avail_c.append(new)
        log.append(log_inner)
        if new is None:
            break
        avail = avail_c

    for i, xi in enumerate(x):
        angle = -gamma * values[i]
        qc.cp(angle, ancs[i % len(ancs)], xi)

    for layer in reversed(log):
        for s, t in layer:
            qc.cx(s, t)


def qpe_layer(problem: Knapsack, M: int | None = None, with_hp: bool = True):
    if M is None:
        M = cl.get_M_qpe(problem)

    anc, _ = cl.how_many_ancilla(problem.n_qubits)

    x = QuantumRegister(problem.n_qubits, name="x")
    qpe = QuantumRegister(M, name="qpe")
    anc = QuantumRegister(anc, name="anc")
    qc = QuantumCircuit(x, qpe, anc)

    gamma = Parameter("g")
    beta = Parameter("b")

    coloring = optimal_bipartite_edge_coloring(problem.n_qubits, M)

    maxval = max(sum(problem.weights) - problem.max_capacity, problem.max_capacity)
    scale: float = np.pi * (1 - 2**-M) / maxval

    offset = -problem.max_capacity * scale
    weights = [-w * scale for w in problem.weights]

    if with_hp:
        qc.h(qpe)
    qpe_phase_part(qc, x, qpe, coloring, weights, offset, with_hp=with_hp)

    qc.compose(synth_qft_full(M, do_swaps=False, inverse=True), qpe, inplace=True)

    qpe_cost_part(qc, x, qpe, anc, problem.costs, gamma)

    qc.compose(synth_qft_full(M, do_swaps=False, inverse=False), qpe, inplace=True)

    qpe_phase_part(qc, x, qpe, coloring, weights, offset, inverse=True, with_hp=with_hp)

    if with_hp:
        qc.h(qpe)

    qc.rx(-2 * beta, x)
    return qc
