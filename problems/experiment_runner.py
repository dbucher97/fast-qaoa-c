from concurrent.futures import Future, ThreadPoolExecutor
import sys
from time import perf_counter
import time

import numpy as np
import pandas as pd
import toml
from tqdm import tqdm

from fastqaoa import params
from fastqaoa.optimize import grid_search
from fastqaoa.ctypes import Diagonals
from fastqaoa.ctypes.metrics import Metrics
from fastqaoa.ctypes.optimize import optimize_qaoa_adam, optimize_qaoa_lbfgs
from fastqaoa.ctypes.qaoa import qaoa
from problems.experiment_structure import *
from problems.problem import ProblemBase


def get_initial(exp: ExperimentCollection, depth: int, dg: Diagonals, cost: Diagonals):
    if exp.initial == InitialConditions.Constant:
        betas, gammas = params.init_const(depth)
    elif exp.initial == InitialConditions.Linear:
        betas, gammas = params.init_linear(depth)
    elif exp.initial == InitialConditions.Random:
        betas, gammas = params.init_random(depth)
    if exp.initial == InitialConditions.GridSearch:
        assert depth == 1, "Intial Grid Search only possible for depth == 1."
        betas, gammas = grid_search(
            dg,
            cost,
            gamma_extent=(0, exp.gamma_scale * np.pi),
            beta_extent=(0, exp.beta_scale * np.pi),
        )
        betas = [betas]
        gammas = [gammas]
    else:
        betas *= exp.beta_scale
        gammas *= exp.gamma_scale
    return betas, gammas


def run_qaoa(
    dg: Diagonals,
    cost: Diagonals,
    metrics_cost: Diagonals,
    constraint_data: Diagonals,
    exp: ExperimentCollection,
    query: Dict[str, Any],
    constr=None,
):
    results = []

    lquery = query.copy()

    isfirst = True
    betas = None
    for p in exp.depths:
        lquery.update({"depth": p})
        entry = exp.get_stored(lquery)
        if entry is not None:
            continue

        if exp.interpolate:
            if isfirst:
                idx = exp.depths.index(p)
                if idx > 0:
                    last_entry = lquery.update({"depth": exp.depths[idx - 1]})
                    if last_entry is not None:
                        betas = last_entry["betas"]
                        gammas = last_entry["gammas"]
                        betas, gammas = params.interpolate(p, betas, gammas)
                    else:
                        betas, gammas = get_initial(exp, p, dg, cost)
                else:
                    betas, gammas = get_initial(exp, p, dg, cost)
            else:
                betas, gammas = params.interpolate(p, betas, gammas)
        else:
            betas, gammas = get_initial(exp, p, dg, cost)

        a = perf_counter()
        result = optimize_qaoa_lbfgs(dg, cost, betas, gammas, constr=constr)
        b = perf_counter()
        betas = result.betas
        gammas = result.gammas

        sv = qaoa(dg, betas, gammas)

        isfirst = False
        results.append(
            {
                "depth": p,
                **Metrics.compute(sv, metrics_cost, constraint_data).dump(),
                "iterations": result.it,
                "status": result.status.name,
                "betas": betas,
                "gammas": gammas,
                "runtime": b - a,
            }
        )
    return results


# def run_qpe(instance: ProblemBase, run: Experiment, exp: ExperimentCollection):
#     raise NotImplementedError()


def get_query_dict(instance: ProblemBase, run: Experiment, exp: ExperimentCollection):
    return {
        "n_qubits": instance.n_qubits,
        "problem_id": instance.id,
        "cost": run.cost.value,
        "qaoa": run.kind.value,
        "interpolate": exp.interpolate,
        "initial": exp.initial.value,
        "beta_scale": exp.beta_scale,
        "gamma_scale": exp.gamma_scale,
        **asdict(run.settings_obj),
    }


def run_default(instance: ProblemBase, run: Experiment, exp: ExperimentCollection):
    results = []
    d = asdict(run.settings_obj)
    if isinstance(run.settings_obj, QuadPenaltyCostSettings):
        if run.settings_obj.penalty == -2:
            d["penalty"] = max(instance.weights) + 1
    dg = getattr(instance, run.kind.value)(**d)
    if run.cost == CostKind.Default:
        cost = dg
    else:
        cost = getattr(instance, run.cost.value)()

    scale = max(abs(dg.min_val), abs(dg.max_val))
    dg = dg * dg.n_qubits / scale

    query = get_query_dict(instance, run, exp)
    _, weights = instance.diagonalized()
    for r in range(exp.repeat):
        e = run_qaoa(dg, cost, instance.masked_cost(), weights, exp, query)
        for ei in e:
            ei.update({"rep": r})
        results += e

    mpenalty = None
    if hasattr(run.settings_obj, "penalty") and run.settings_obj.penalty is None:
        mpenalty = instance._penalty

    for r in results:
        r.update(query)
        if mpenalty is not None:
            r["set_penalty"] = mpenalty

    return results


def run_experiment_for_instance(instance: ProblemBase, exp: ExperimentCollection):
    results = []
    for run in aslist(exp.qaoa):
        if run.until_size is not None and instance.n_qubits > run.until_size:
            continue
        if run.kind == QAOAKind.QPE:
            # results += run_qpe(instance, run, exp)
            pass
        else:
            results += run_default(instance, run, exp)
    instance.decache()

    return results


def run_experiment(exp: ExperimentCollection, num_workers: int = 4):
    Problem = None
    for p in ALL_PROBLEMS:
        if p.__name__ == exp.problem.value:
            Problem = p
            break

    tpe = ThreadPoolExecutor(max_workers=num_workers)

    instances = Problem.get_instances()
    futures = []
    results = []

    for size in aslist(exp.sizes):
        print(f"\nRunning experiments for size {size}...")
        if exp.instances is not None:
            if size in instances:
                num_instances = len(instances[size])
            else:
                num_instances = 0
            if num_instances < exp.instances:
                raise ValueError(
                    f"Not enough instances: {exp.instances} required, but only {instances} found."
                )
            iter_instances = instances[size][: exp.instances]
        else:
            if size not in instances:
                continue
            iter_instances = instances[size]

        for instance in tqdm(iter_instances):
            futures.append(
                tpe.submit(lambda i: run_experiment_for_instance(i, exp), instance)
            )

            if len(futures) >= tpe._max_workers:
                res = [f for f in futures if f.done()]
                while len(res) == 0:
                    time.sleep(0.1)
                    res = [f for f in futures if f.done()]
                futures = [f for f in futures if f not in res]
                for f in res:
                    results += f.result()

            if len(results) >= 32:
                df = pd.DataFrame.from_records(results)
                exp.add_results(df)
                results = []

        # results = tpe.map(lambda i: run_experiment_for_instance(i, exp), iter_instances)

        # for i, res in enumerate(tqdm(results, total=len(iter_instances), ncols=80)):
        #     data += res
        #     if (i + 1) % 32 == 0:
        #         df = pd.DataFrame.from_records(data)
        #         exp.add_results(df)
        #         data = []

    for f in futures:
        results += f.result()

    if len(results) > 0:
        df = pd.DataFrame.from_records(results)
        exp.add_results(df)


def parse_settings(exp: ExperimentCollection):
    for run in exp.qaoa:
        SettingsObj = globals().get(run.kind.name + "Settings")
        if run.settings is not None:
            run.settings_obj = SettingsObj(**run.settings)
        else:
            run.settings_obj = SettingsObj()


if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise RuntimeError("Expected experiments .toml file")
    config = toml.load(sys.argv[1])

    n_workers = config.pop("n_workers", 4)

    for k, v in config.items():
        print("=" * 80)
        print(f"Starting ", end="")
        exp = ExperimentCollection(name=k, **v)
        parse_settings(exp)
        exp.load_results()
        print(exp)
        run_experiment(exp, n_workers)
