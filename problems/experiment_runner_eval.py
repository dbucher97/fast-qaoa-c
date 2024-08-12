from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pydantic.dataclasses import dataclass
import sys
import os
import pandas as pd
import numpy as np

import toml
from tqdm import tqdm

from fastqaoa.ctypes.diagonals import Diagonals
from fastqaoa.ctypes.metrics import Metrics
from fastqaoa.ctypes.qaoa import qaoa
from fastqaoa.ctypes.qpe_qaoa import qpe_qaoa
from fastqaoa.indicator import interpolate_diagonals
from problems import ALL_PROBLEMS
from problems.experiment_runner import check_all_done, get_interpolator, parse_settings
from problems.experiment_structure import (
    CostKind,
    Experiment,
    ProblemsKind,
    QAOAKind,
    QuadPenaltyCostSettings,
    aslist,
)
from problems.problem import ProblemBase


@dataclass
class ExperimentCollectionEval:
    name: str
    problem: ProblemsKind
    input_data: str
    input_kind: str

    qaoa: Experiment | list[Experiment]

    _result = None
    _input = None

    path: str = "results"

    class Config:
        exclude = ["_result", "_input"]

    @property
    def df(self):
        return self._result

    @property
    def input_df(self) -> pd.DataFrame:
        return self._input

    def load_input(self):
        if self._input is None:
            df = pd.read_feather(f"{self.path}/{self.input_data}.feather")
            df = df.query("qaoa == @self.input_kind")
            self._input = df
        return self._input

    def get_stored(self, query):
        if self._result is None:
            return None
        else:
            res = self._result
            for k, v in query.items():
                if v is None:
                    res = res[res[k].isna()]
                else:
                    res = res[res[k] == v]
                if len(res) == 0:
                    return None
            return res

    def load_results(self):
        self.load_input()
        path = os.path.join(self.path, self.name + ".feather")
        if os.path.exists(path):
            self._result = pd.read_feather(path)
            return self._result
        else:
            return None

    def add_results(self, df: pd.DataFrame):
        path = os.path.join(self.path, self.name + ".feather")
        os.makedirs(self.path, exist_ok=True)
        self._result = pd.concat([self._result, df], ignore_index=True)
        self._result.to_feather(path)

    def __repr__(self):
        start = 4 * " "
        res = ""
        res += f"Experiment Collection Eval'{self.name}'\n"
        res += f"{start}Problem    = {self.problem.name}\n"

        return "\n".join([res] + list(map(repr, self.qaoa)))


def get_query_dict(instance: ProblemBase, run: Experiment):
    return {
        "n_qubits": instance.n_qubits,
        "problem_id": instance.id,
        "cost": run.cost.value,
        "qaoa": run.kind.value,
        **asdict(run.settings_obj),
    }


def run_qaoa(
    dg: Diagonals,
    _: Diagonals,
    metrics_cost: Diagonals,
    constraint_data: Diagonals,
    betas: np.ndarray,
    gammas: np.ndarray,
    constr=None,
):
    psucc = None
    if constr is None:
        sv = qaoa(dg, betas, gammas)
    else:
        sv, psucc = qpe_qaoa(dg, constr, betas, gammas)
    data = {
        "depth": len(betas),
        **Metrics.compute(sv, metrics_cost, constraint_data).dump(),
        "betas": betas,
        "gammas": gammas,
    }
    if constr is not None:
        data["p_succ"] = psucc
    return data


def run_qpe(
    instance: ProblemBase,
    run: Experiment,
    exp: ExperimentCollectionEval,
    betas: np.ndarray,
    gammas: np.ndarray,
):
    results = []
    query = get_query_dict(instance, run)
    query["ancilla"] = run.settings_obj.ancilla
    query["shift"] = run.settings_obj.shift
    query["depth"] = len(betas)
    stored = exp.get_stored(query)
    if stored is not None:
        return results
    dg, weights = instance.diagonalized()

    cost = instance.masked_cost()
    scale = max(abs(cost.min_val), abs(cost.max_val))
    dg = dg * dg.n_qubits / scale

    constr = weights.scale_between_sym()
    interp = get_interpolator(run.settings_obj.ancilla, run.settings_obj.shift)

    constr = interpolate_diagonals(interp, constr)

    e = run_qaoa(
        dg,
        cost,
        cost,
        weights,
        betas,
        gammas,
        constr=constr,
    )
    results.append(e)

    for r in results:
        r.update(query)

    return results


def run_default(
    instance: ProblemBase,
    run: Experiment,
    exp: ExperimentCollectionEval,
    betas: np.ndarray,
    gammas: np.ndarray,
):
    results = []
    query = get_query_dict(instance, run)
    query["depth"] = len(betas)
    stored = exp.get_stored(query)
    if stored is not None:
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

    dg = dg * dg.n_qubits / scale

    _, weights = instance.diagonalized()
    e = run_qaoa(
        dg,
        cost,
        instance.masked_cost(),
        weights,
        betas,
        gammas,
    )
    results.append(e)

    mpenalty = None
    if hasattr(run.settings_obj, "penalty") and run.settings_obj.penalty is None:
        mpenalty = instance._penalty

    for r in results:
        r.update(query)
        if mpenalty is not None:
            r["set_penalty"] = mpenalty

    return results


def run_experiments_for_input(
    instance: ProblemBase,
    betas: np.ndarray,
    gammas: np.ndarray,
    exp: ExperimentCollectionEval,
):
    results = []
    for run in aslist(exp.qaoa):
        if run.until_size is not None and instance.n_qubits > run.until_size:
            continue
        if run.kind == QAOAKind.QPE:
            results += run_qpe(instance, run, exp, betas, gammas)
        else:
            results += run_default(instance, run, exp, betas, gammas)

    return results


def run_experiment(exp: ExperimentCollectionEval, n_workers: int = 4):
    Problem = None
    for p in ALL_PROBLEMS:
        if p.__name__ == exp.problem.value:
            Problem = p
            break

    tpe = ThreadPoolExecutor(max_workers=n_workers)

    instances = Problem.get_instances()

    runs = []
    # futures = []
    results = []

    for _, row in exp.input_df.iterrows():
        size = row.n_qubits
        problem_id = row.problem_id

        runs.append((instances[size][problem_id], row.betas, row.gammas))

    prog_bar = tqdm(total=len(runs), ncols=80)

    def _run(x):
        instance, betas, gammas = x
        return run_experiments_for_input(instance, betas, gammas, exp)

    results = []
    for res in tpe.map(_run, runs):
        results += res

        prog_bar.update()

        if len(results) >= 50:
            df = pd.DataFrame.from_records(results)
            exp.add_results(df)
            results = []

    if len(results) > 0:
        df = pd.DataFrame.from_records(results)
        exp.add_results(df)


if __name__ == "__main__":
    config = toml.load(sys.argv[1])

    n_workers = config.pop("n_workers", 4)

    for k, v in config.items():
        print("=" * 80)
        print(f"Starting ", end="")
        exp = ExperimentCollectionEval(name=k, **v)
        parse_settings(exp)
        exp.load_results()
        print(exp)
        run_experiment(exp, n_workers)
