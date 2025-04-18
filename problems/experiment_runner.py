from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
import sys
from time import perf_counter
import itertools

import importlib

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
from fastqaoa.ctypes.qpe_qaoa import qpe_qaoa
from fastqaoa.indicator import get_indicator_interpolator, interpolate_diagonals

from problems import ALL_PROBLEMS
from problems import experiment_structure
from problems.experiment_structure import (
    CostKind,
    Experiment,
    ExperimentCollection,
    InitialConditions,
    QAOAKind,
    QuadPenaltyCostSettings,
    aslist,
)
from problems.problem import ProblemBase

__interpolators = {}


def get_interpolator(M: int, shift: float = 0.0):
    global __interpolators
    if (M, shift) not in __interpolators:
        __interpolators[(M, shift)] = get_indicator_interpolator(M, 4, shift=shift)
    return __interpolators[(M, shift)]


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
    stored: pd.DataFrame,
    constr=None,
):
    results = []

    isfirst = True
    betas = None
    for p in exp.depths:
        if stored is not None:
            entry = stored.query("depth == @p")
            if len(entry) != 0:
                continue

        if exp.interpolate:
            if isfirst:
                idx = exp.depths.index(p)
                if idx > 0:
                    dl = exp.depths[idx - 1]
                    last_entry = stored.query("depth == @dl").iloc[0]
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

        psucc = None
        if constr is None:
            sv = qaoa(dg, betas, gammas)
        else:
            sv, psucc = qpe_qaoa(dg, constr, betas, gammas)

        isfirst = False
        data = {
            "depth": p,
            **Metrics.compute(sv, metrics_cost, constraint_data).dump(),
            "iterations": result.it,
            "status": result.status.name,
            "betas": betas,
            "gammas": gammas,
            "runtime": b - a,
        }
        if constr is not None:
            data["p_succ"] = psucc
        results.append(data)
    return results


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


def check_all_done(df: pd.DataFrame, exp: ExperimentCollection):
    x = set(itertools.product(exp.depths, range(exp.repeat)))
    y = set(zip(map(int, df.depth), map(int, df.rep)))
    return x == y


def run_qpe(instance: ProblemBase, run: Experiment, exp: ExperimentCollection):
    results = []
    query = get_query_dict(instance, run, exp)
    query["ancilla"] = run.settings_obj.ancilla
    query["shift"] = run.settings_obj.shift
    stored = exp.get_stored(query)
    if stored is not None and check_all_done(stored, exp):
        return results
    dg, weights = instance.diagonalized()
    scale = max(abs(dg.min_val), abs(dg.max_val))
    dg = dg * dg.n_qubits / scale

    constr = weights.scale_between_sym()
    interp = get_interpolator(run.settings_obj.ancilla, run.settings_obj.shift)

    constr = interpolate_diagonals(interp, constr)

    cost = instance.masked_cost()

    for r in range(exp.repeat):
        e = run_qaoa(
            dg,
            cost,
            cost,
            weights,
            exp,
            None if stored is None else stored.query("rep == @r"),
            constr=constr,
        )
        for ei in e:
            ei.update({"rep": r})
        results += e

    for r in results:
        r.update(query)

    return results


def run_default(instance: ProblemBase, run: Experiment, exp: ExperimentCollection):
    results = []
    query = get_query_dict(instance, run, exp)
    stored = exp.get_stored(query)
    if stored is not None and check_all_done(stored, exp):
        return results

    d = asdict(run.settings_obj)
    if isinstance(run.settings_obj, QuadPenaltyCostSettings):
        if run.settings_obj.penalty == -2:
            d["penalty"] = max(instance.weights) + 1
    dg = getattr(instance, run.kind.value)(**d)
    if run.cost == CostKind.Default:
        cost = dg
    else:
        cost = getattr(instance, run.cost.value)()

    # c, _ = instance.diagonalized()
    scale = max(abs(dg.min_val), abs(dg.max_val))
    # scale = max(abs(c.min_val), abs(c.max_val))

    dg = dg * dg.n_qubits / scale

    _, weights = instance.diagonalized()
    for r in range(exp.repeat):
        e = run_qaoa(
            dg,
            cost,
            instance.masked_cost(),
            weights,
            exp,
            None if stored is None else stored.query("rep == @r"),
        )
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
            results += run_qpe(instance, run, exp)
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

    runs = []
    prog_bars = {}
    # futures = []
    results = []

    for pos, size in enumerate(aslist(exp.sizes)):
        # print(f"\nRunning experiments for size {size}...")
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

        runs += iter_instances
        prog_bars[size] = tqdm(
            total=len(iter_instances), position=pos, desc=f"Size {size:02}", ncols=80
        )

    def _run(instance):
        ret = run_experiment_for_instance(instance, exp)
        return instance.n_qubits, ret

    for size, res in tpe.map(_run, runs):
        results += res

        prog_bars[size].update()
        if prog_bars[size].n == prog_bars[size].total:
            prog_bars[size].refresh()

        if len(results) >= 120:
            df = pd.DataFrame.from_records(results)
            exp.add_results(df)
            results = []

        # old 2 -------------------------------------------
        # for instance in tqdm(iter_instances):
        #     futures.append(
        #         tpe.submit(lambda i: run_experiment_for_instance(i, exp), instance)
        #     )
        #
        #     if len(futures) >= tpe._max_workers:
        #         res = [f for f in futures if f.done()]
        #         while len(res) == 0:
        #             time.sleep(0.1)
        #             res = [f for f in futures if f.done()]
        #         futures = [f for f in futures if f not in res]
        #         for f in res:
        #             results += f.result()
        #
        #     if len(results) >= 32:
        #         df = pd.DataFrame.from_records(results)
        #         exp.add_results(df)
        #         results = []
    # for f in futures:
    #     results += f.result()

    # old 1 -------------------------------------------------
    # results = tpe.map(lambda i: run_experiment_for_instance(i, exp), iter_instances)

    # for i, res in enumerate(tqdm(results, total=len(iter_instances), ncols=80)):
    #     data += res
    #     if (i + 1) % 32 == 0:
    #         df = pd.DataFrame.from_records(data)
    #         exp.add_results(df)
    #         data = []

    if len(results) > 0:
        df = pd.DataFrame.from_records(results)
        exp.add_results(df)


def parse_settings(exp: ExperimentCollection):
    for run in exp.qaoa:
        settings_obj = run.kind.name + "Settings"
        SettingsObj = getattr(experiment_structure, settings_obj)
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
