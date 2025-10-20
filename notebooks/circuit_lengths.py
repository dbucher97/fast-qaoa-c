import numpy as np
import pandas as pd

from problems import IntegerKnapsack, Knapsack
from problems.experiment_structure import QAOAKind


def ceillog2(x):
    return int(np.ceil(np.log2(x)))


def how_many_ancilla(N) -> tuple[int, int]:
    current = N

    best = 0
    for a in range(1, N):
        new = np.ceil(N / (a + 1)) + 2 * ceillog2(a + 1)
        if new < current:
            current = new
            best = a

    return best, ceillog2(best + 1)


def get_M_qpe(problem: Knapsack):
    if not isinstance(problem, IntegerKnapsack):
        raise ValueError("Cannot automatically infer M from non integer Knapsack")
    min_val = -sum(problem.weights) + problem.max_capacity
    max_val = problem.max_capacity

    return max(ceillog2(abs(min_val)), ceillog2(abs(max_val))) + 1


def get_M_penalty(problem: Knapsack):
    if not isinstance(problem, IntegerKnapsack):
        raise ValueError("Cannot automatically infer M from non integer Knapsack")
    return ceillog2(problem.max_capacity)


def layer_depth_qpe(problem: Knapsack, M: int | None = None):
    if M is None:
        M = get_M_qpe(problem)

    N = problem.n_qubits

    phase = max(M, N)
    # phase2 = min(N * (1 + 2 * ceillog2(M)), M * (1 + 2 * ceillog2(N)))

    qft = 2 * M - 1
    anc, layer = how_many_ancilla(N)
    cop = 2 * layer + int(np.ceil(N / (anc + 1)))

    print(f"{qft=}")
    print(f"{phase=}")
    print(f"{cop=}")

    layers = 2 * phase + 2 * qft + cop  # + measure

    return layers


def circuit_depth_qpe(problem: Knapsack, p: int = 1, M: int | None = None):
    return 1 + p * (layer_depth_qpe(problem, M) + 1)


def two_qubit_ops_qpe(problem: Knapsack, p: int = 1, M: int | None = None):
    if M is None:
        M = get_M_qpe(problem)
    N = problem.n_qubits

    phase = M * N
    qft = (M * (M - 1)) // 2
    cop = N

    return p * (2 * phase + 2 * qft + cop)


def layer_qubit_ops_qpe(problem: Knapsack, M: int | None = None):
    if M is None:
        M = get_M_qpe(problem)
    N = problem.n_qubits

    phase2 = M * N  # contorlled phase applications
    phase1 = 2 * M  # Hadamard + constant offset

    qft2 = (M * (M - 1)) // 2
    qft1 = M

    anc, _ = how_many_ancilla(N)
    cop2 = N + 2 * anc
    cop1 = 0

    return {
        "twoq": (cop2 + 2 * qft2 + 2 * phase2),
        "singleq": (cop1 + 2 * qft1 + 2 * phase1 + 1),  # +1 for mixer
    }


def circuit_depth_penalty_legacy(problem: Knapsack, p: int = 1, M: int | None = None):
    if M is None:
        M = get_M_penalty(problem)
    N = problem.n_qubits

    inter = max(M, N)
    in_main = int(N * (N - 1) / 2 / (np.floor(N / 2)))
    in_ancilla = int(M * (M - 1) / 2 / (np.floor(M / 2)))

    if N % 2 == 0:
        in_main += 1
    if M % 2 == 0:
        in_ancilla += 1

    layers = inter + max(in_main, in_ancilla) + 1
    return 1 + layers * p


def layer_depth_penalty(problem: Knapsack, M: int | None = None):
    if M is None:
        M = get_M_penalty(problem)
    N = problem.n_qubits

    # https://en.wikipedia.org/wiki/Edge_coloring
    layers = M + N  # - 1
    if layers % 2 == 0:
        layers -= 1
    # if layers % 2 == 1:
    #    layers += 1

    return 2 + layers


def circuit_depth_penalty(problem: Knapsack, p: int = 1, M: int | None = None):
    layer = layer_depth_penalty(problem, M)
    return p * layer + 1


def two_qubit_ops_penalty(problem: Knapsack, p: int = 1, M: int | None = None):
    if M is None:
        M = get_M_penalty(problem)
    N = problem.n_qubits

    inter = M * N
    in_main = (N * (N - 1)) // 2
    in_ancilla = (M * (M - 1)) // 2

    num = inter + in_main + in_ancilla
    return num * p


def layer_qubit_ops_penalty(problem: Knapsack, M: int | None = None):
    if M is None:
        M = get_M_penalty(problem)
    N = problem.n_qubits

    total = M + N
    return {"twoq": total * (total - 1) / 2, "singleq": 2 * total}


def add_lengths_to_df(df: pd.DataFrame, problem: Knapsack):
    instances = problem.get_instances()

    def eval_clops(x):
        prb = instances[int(x.n_qubits)][int(x.problem_id)]

        if (
            x.qaoa == QAOAKind.QuadPenaltyCost.value
            or x.qaoa == QAOAKind.QuadPenaltyFullProblem.value
        ):
            try:
                return circuit_depth_penalty(prb, int(x.depth))
            except:
                return float("nan")
        elif x.qaoa == QAOAKind.QPE.value:
            return circuit_depth_qpe(prb, int(x.depth), M=x.ancilla)
        else:
            try:
                return circuit_depth_qpe(prb, int(x.depth))
            except ValueError:
                return float("nan")

    def eval_two_qubit_ops(x):
        prb = instances[int(x.n_qubits)][int(x.problem_id)]

        if (
            x.qaoa == QAOAKind.QuadPenaltyCost.value
            or x.qaoa == QAOAKind.QuadPenaltyFullProblem.value
        ):
            try:
                return two_qubit_ops_penalty(prb, int(x.depth))
            except:
                return float("nan")
        elif x.qaoa == QAOAKind.QPE.value:
            return two_qubit_ops_qpe(prb, int(x.depth), M=x.ancilla)
        else:
            try:
                return two_qubit_ops_qpe(prb, int(x.depth))
            except ValueError:
                return float("nan")

    def eval_weight_ratio(x):
        prb = instances[int(x.n_qubits)][int(x.problem_id)]

        return prb.max_capacity / sum(prb.weights)

    def eval_feas_perc(x):
        prb = instances[int(x.n_qubits)][int(x.problem_id)]

        _, constr = prb.diagonalized()
        y = (constr >= 0).to_numpy().sum()

        return y / (1 << prb.n_qubits)

    df["clops"] = df.apply(eval_clops, axis=1)
    df["two_qubit_ops"] = df.apply(eval_two_qubit_ops, axis=1)
    df["weight_ratio"] = df.apply(eval_weight_ratio, axis=1)
    # df["feas_perc"] = df.apply(eval_feas_perc, axis=1)
