from problems.integer_knapsack import IntegerKnapsack
from enum import Enum

from dataclasses import dataclass

@dataclass
class IntegerKnapsackPisinger(IntegerKnapsack):
    gen_type: int

def read_instances(path, gen_type: int, start_id: int):
    instances = []
    with open(path) as f:
        instances = f.read().split("-----\n\n")[:-1]

    id = start_id
    for instance in instances:
        data = instance.split("\n")[:-1]
        capacity = int(data[2].split(" ")[1])
        data = list(map(lambda x: x.split(","), data[5:]))
        costs = [int(d[1]) for d in data]
        weights = [int(d[2]) for d in data]
        IntegerKnapsackPisinger(n_qubits=20, max_capacity=capacity, weights=weights,
                                costs=costs, gen_type=gen_type, id=id).add()
        id += 1
    return id



if __name__ == "__main__":
    import sys
    from pathlib import Path
    gen_types = [11, 12, 13, 14, 15, 16]

    assert len(sys.argv) == 2, "Specify pisinger data folder"
    path = Path(sys.argv[1]).absolute()
    id = 0
    for gen_type in gen_types:
        id = read_instances(path / f"knapPI_{gen_type}_20_1000.csv", gen_type, id)
    print(IntegerKnapsackPisinger.get_instances())
    IntegerKnapsackPisinger.store()

