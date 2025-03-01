from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import pandas as pd
from fastqaoa import Metrics
from fastqaoa.ctypes.optimize import optimize_qaoa_lbfgs
from fastqaoa.ctypes.qpe_qaoa import qpe_qaoa_norm
from fastqaoa.indicator import interpolate_diagonals, get_indicator_interpolator
from problems.knapsack import Knapsack
import datetime
import time


SHIFT = 0.4
INTERPOLATORS = {
    M: get_indicator_interpolator(M, 4, shift=SHIFT) for M in [4, 6, 8, 12, 16]
}

INSTANCES = Knapsack.get_instances()


df = pd.read_feather("results/qpe_main2.feather").query("qaoa == 'masked_cost'")


exec = ThreadPoolExecutor(max_workers=24)


def finetune_instance(x):
    (size, idx), data = x
    print("started", size, idx, datetime.datetime.now())
    start = time.time()
    instance: Knapsack = INSTANCES[size][idx]
    f, g = instance.diagonalized()
    feas = g >= 0
    cost = instance.masked_cost()
    scale = max(abs(cost.min_val), abs(cost.max_val))

    f = f * f.n_qubits / scale

    ret = []
    for M, interp in INTERPOLATORS.items():
        g_loc = g.scale_between_sym(M=M)
        sgn = interpolate_diagonals(interp, g_loc)
        for _, row in data.sort_values(by="depth").iterrows():
            a = perf_counter()
            res = optimize_qaoa_lbfgs(
                f, cost, row.betas, row.gammas, constr=sgn, maxiter=5
            )
            delta = perf_counter() - a
            sv, p, q = qpe_qaoa_norm(f, sgn, res.betas, res.gammas)
            new_data = dict(row)


            new_data.update(Metrics.compute(sv, cost, feas).dump())
            new_data["status"] = res.status.name
            new_data["iterations"] = res.it
            new_data["runtime"] = delta
            new_data["betas"] = res.betas
            new_data["gammas"] = res.gammas
            new_data["ancilla"] = M
            new_data["shift"] = SHIFT
            new_data["p_succ"] = p
            new_data["q_succ"] = q
            new_data["qaoa"] = "qpe"

            ret.append(new_data)

    instance.decache()
    elapsed = datetime.timedelta(seconds=time.time() - start)
    print("finished", size, idx, elapsed)

    return ret

collected = []
for result in exec.map(finetune_instance, df.groupby(["n_qubits", "problem_id"])):
    collected += result

    if len(collected) >= 12 * 24:
        df = pd.concat([df, pd.DataFrame(collected)], ignore_index=True)
        collected = []
        df.to_feather("results/qpe_main2_ft.feather")

if len(collected) > 0:
    df = pd.concat([df, pd.DataFrame(collected)], ignore_index=True)
    df.to_feather("results/qpe_main2_ft.feather")
