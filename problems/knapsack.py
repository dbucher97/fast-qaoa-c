from typing import List, Tuple
from dataclasses import dataclass

from fastqaoa.ctypes.diagonals import Diagonals
from problems.problem import ProblemBase

import qubovert as qv

import numpy as np


@dataclass
class Knapsack(ProblemBase):
    max_capacity: float
    weights: List[float]
    costs: List[float]

    @classmethod
    def random_instance(cls, n_qubits, crange=0.2, id=None):
        w = np.random.rand(n_qubits)
        c = np.random.rand(n_qubits)

        max_capacity = w.sum() * (crange + (1 - 2 * crange) * np.random.rand())

        if id is None:
            id = cls.next_id(n_qubits)

        return cls(
            id=id, n_qubits=n_qubits, max_capacity=max_capacity, weights=w, costs=c
        )

    def diagonalized(self) -> Tuple[Diagonals, Diagonals]:
        if hasattr(self, "_cost"):
            return self._cost, self._constr
        idxs = [1 << i for i in range(self.n_qubits)]
        costs = [-c for c in self.costs]
        cost = Diagonals.brute_force(self.n_qubits, idxs, costs)
        weights = [-c for c in self.weights]
        constr = Diagonals.brute_force(self.n_qubits, idxs, weights)
        constr += self.max_capacity

        self._cost = cost
        self._constr = constr

        return self._cost, self._constr

    def masked_cost(self, mask_val: float = 0.):
        if hasattr(self, "_masked") and mask_val == 0:
            return self._masked
        cost, constr = self.diagonalized()
        masked = cost.mask(constr, 0, Diagonals.GTE, mask_val)
        if mask_val == 0:
            self._masked = masked
        return masked

    def kickback_cost(self):
        masked = self.masked_cost()
        cost, _ = self.diagonalized()
        kickback = -cost.to_numpy() + 2 * masked.to_numpy()
        return Diagonals.from_numpy(kickback)

    def quad_penalty_cost(self, penalty: float = None):
        cost, constr = self.diagonalized()
        if penalty is None and hasattr(self, "_penalty"):
            penalty = self._penalty
        diag, new_penalty = cost.quad_penalty(constr, 0, Diagonals.GTE, penalty=penalty)
        if penalty is None:
            self._penalty = new_penalty
        return diag

    def slack_problem(self, n_ancilla: int, penalty: float = None):
        if penalty is None:
            if not hasattr(self, "_penalty"):
                self.quad_penalty_cost()

        penalty = self._penalty

        xs = [qv.boolean_var(i) for i in range(self.n_qubits)]
        ys = [qv.boolean_var(self.n_qubits + i) for i in range(n_ancilla)]
        step = self.max_capacity / (2**n_ancilla - 1)
        yfacs = [step * 2**i for i in range(n_ancilla)]

        cost = sum(-c * x for c, x in zip(self.costs, xs))
        weights = sum(w * x for w, x in zip(self.weights, xs))
        slack = sum(f * y for f, y in zip(yfacs, ys))

        return cost + penalty * (weights - slack) ** 2


    def quad_penalty_full_problem(self, n_ancilla: int, penalty: float = None):
        obj = self.slack_problem(n_ancilla, penalty)

        return Diagonals.brute_force_qv(obj)

    def decache(self):
        if hasattr(self, "_masked"):
            del self._masked
        if hasattr(self, "_cost"):
            del self._cost
        if hasattr(self, "_constr"):
            del self._constr
        if hasattr(self, "_penalty"):
            del self._penalty

if __name__ == "__main__":
    for s in range(6, 30):
        while Knapsack.next_id(s) < 256:
            Knapsack.random_instance(s).add()
    Knapsack.store()
