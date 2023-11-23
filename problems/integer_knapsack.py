from problems.knapsack import Knapsack
import numpy as np
import qubovert as qv
from fastqaoa.ctypes import Diagonals


class IntegerKnapsack(Knapsack):
    @classmethod
    def random_instance(cls, n_qubits, max_capacity=10, crange=0.2, id=None):
        instance = super().random_instance(n_qubits, crange=crange, id=id)
        scale = max_capacity / instance.max_capacity
        instance.weights = [round(w * scale) for w in instance.weights]
        instance.costs = [round(c * scale) for c in instance.costs]
        instance.max_capacity = max_capacity

        return instance

    def quad_penalty_full_problem(self, penalty: float):
        if penalty is None:
            if not hasattr(self, "_penalty"):
                self.quad_penalty_cost()

        penalty = self._penalty


        n_ancilla = int(np.floor(np.log2(self.max_capacity) + 1))
        xs = [qv.boolean_var(i) for i in range(self.n_qubits)]
        ys = [qv.boolean_var(self.n_qubits + i) for i in range(n_ancilla)]
        yfacs = [2**i for i in range(n_ancilla - 1)] + [
            self.max_capacity - 2 ** (n_ancilla - 1) + 1
        ]

        cost = sum(-c * x for c, x in zip(self.costs, xs))
        weights = sum(w * x for w, x in zip(self.weights, xs))
        slack = sum(f * y for f, y in zip(yfacs, ys))

        obj = cost + penalty * (weights - slack) ** 2

        return Diagonals.brute_force_qv(obj)


if __name__ == "__main__":
    for s in range(6, 30):
        while IntegerKnapsack.next_id(s) < 256:
            IntegerKnapsack.random_instance(s, max_capacity=s * 10).add()
    IntegerKnapsack.store()
