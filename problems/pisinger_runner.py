import sys
from time import perf_counter

from typing import List
from pydantic.dataclasses import dataclass
from pydantic import Field

from problems.experiment_structure import ExperimentCollection
from problems.experiment_runner import parse_settings, run_experiment_for_instance

import numpy as np
import pandas as pd
import toml
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

from problems.integer_knapsack_pisinger import IntegerKnapsackPisinger


@dataclass
class PExperimentCollection(ExperimentCollection):
    gen_types: List[int] = Field(default_factory=lambda: [11])

    def __repr__(self):
        res = super().__repr__()
        data = res.split("\n")
        data.insert(4, f"    Gen Types  = {self.gen_types}")
        return "\n".join(data)



def run_experiment(exp: PExperimentCollection, num_workers: int = 4):
    assert exp.problem.value == "IntegerKnapsackPisinger", "Only IntegerKnapsackPisinger problems"
    assert len(exp.sizes) == 1 and exp.sizes[0] == 20, "Only size == 20 supported"

    tpe = ThreadPoolExecutor(max_workers=num_workers)

    instances = IntegerKnapsackPisinger.get_instances()[20]

    runs = []
    # futures = []
    results = []

    for gen_type in exp.gen_types:
        gt_instances = [i for i in instances if i.gen_type == gen_type]
        if exp.instances is not None:
            if len(gt_instances) < exp.instances:
                raise ValueError(
                    f"Not enough instances: {exp.instances} required, but only {len(gt_instances)} found."
                )
            else:
                runs += gt_instances[: exp.instances]
        else:
            runs += gt_instances


    def _run(instance):
        ret = run_experiment_for_instance(instance, exp)
        for r in ret:
            r["gen_type"] = instance.gen_type
        return ret

    last_update = perf_counter()
    for res in tqdm(tpe.map(_run, runs),total=len(runs)):
        results += res

        if perf_counter() - last_update > 30:
            df = pd.DataFrame.from_records(results)
            last_update = perf_counter()
            exp.add_results(df)
            results = []

    if len(results) > 0:
        df = pd.DataFrame.from_records(results)
        exp.add_results(df)


if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise RuntimeError("Expected experiments .toml file")
    config = toml.load(sys.argv[1])

    n_workers = config.pop("n_workers", 4)

    for k, v in config.items():
        print("=" * 80)
        print(f"Starting ", end="")
        exp = PExperimentCollection(name=k, **v)
        parse_settings(exp)
        exp.load_results()
        print(exp)
        run_experiment(exp, n_workers)
