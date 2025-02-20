import numpy as np
from problems import Knapsack, IntegerKnapsack, IntegerKnapsackPisinger
import pandas as pd

from problems.experiment_structure import QAOAKind

def ceillog2(x):
    return int(np.ceil(np.log2(x)))

def get_M_qpe(problem: Knapsack):
    if not isinstance(problem, IntegerKnapsackPisinger) and not isinstance(
        problem, IntegerKnapsack
    ):
        raise ValueError("Cannot automatically infer M from non integer Knapsack")
    return ceillog2(np.sum(problem.weights))


def get_M_penalty(problem: Knapsack):
    if not isinstance(problem, IntegerKnapsackPisinger) and not isinstance(
        problem, IntegerKnapsack
    ):
        raise ValueError("Cannot automatically infer M from non integer Knapsack")
    return ceillog2(problem.max_capacity)


def circuit_depth_qpe(problem: Knapsack, p: int = 1, M: int = None):
    if M is None:
        M = get_M_qpe(problem)

    N = problem.n_qubits

    phase = max(M, N)
    # phase2 = min(N * (1 + 2 * ceillog2(M)), M * (1 + 2 * ceillog2(N)))
    # print(phase, phase2)
    qft = 2 * M - 1
    cop = 2 * ceillog2(N) + 1


    layers = 2 * phase + 2 * qft + cop # + measure

    return p * layers


def two_qubit_ops_qpe(problem: Knapsack, p: int = 1, M: int = None):
    if M is None:
        M = get_M_qpe(problem)
    N = problem.n_qubits

    phase = M * N
    qft = (M * (M - 1)) // 2
    cop = N

    return p * (2 * phase + 2 * qft + cop)


def circuit_depth_penalty(problem: Knapsack, p: int = 1, M: int = None):
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


def two_qubit_ops_penalty(problem: Knapsack, p: int = 1, M: int = None):
    if M is None:
        M = get_M_penalty(problem)
    N = problem.n_qubits

    inter = M * N
    in_main = (N * (N - 1)) // 2
    in_ancilla = (M * (M - 1)) // 2

    num = inter + in_main + in_ancilla
    return num * p


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
    df["feas_perc"] = df.apply(eval_feas_perc, axis=1)
