from typing import Tuple, List
from fastqaoa.ctypes import Diagonals
from fastqaoa.ctypes.qaoa import multi_energy
from fastqaoa.ctypes.optimize import optimize_qaoa_adam, optimize_qaoa_lbfgs

import numpy as np

from fastqaoa.params import interpolate


def grid_search(
    dg: Diagonals,
    cost: Diagonals,
    dim_num: int = 20,
    beta_extent: Tuple[float] = (0, np.pi),
    gamma_extent: Tuple[float] = (0, np.pi),
):
    b = np.linspace(*beta_extent, dim_num, endpoint=False)
    g = np.linspace(*gamma_extent, dim_num, endpoint=False)
    B, G = np.meshgrid(b, g)
    B = np.expand_dims(B.ravel(), -1)
    G = np.expand_dims(G.ravel(), -1)
    res = multi_energy(dg, cost, B, G)
    idx = np.argmin(res)
    return B[idx, 0], G[idx, 0]


def optimize_interpolate(
    dg: Diagonals,
    cost: Diagonals,
    depths: List[int],
    constr: Diagonals = None,
    first_betas: List[float] = None,
    first_gammas: List[float] = None,
    init_val: float = 0.1,
    use_adam: bool = False,
    **kwargs
):
    assert not (
        (first_betas is None) ^ (first_gammas is None)
    ), "if first_betas(gammas) is specified, please also specify the other."

    if first_betas is None:
        first_betas = np.array([init_val] * depths[0])
        first_gammas = np.array([init_val] * depths[0])

    betas = first_betas
    gammas = first_gammas

    results = {}

    for depth in depths:
        betas, gammas = interpolate(depth, betas, gammas)
        if use_adam:
            res = optimize_qaoa_adam(dg, cost, betas, gammas, constr=constr, **kwargs)
        else:
            res = optimize_qaoa_lbfgs(dg, cost, betas, gammas, constr=constr, **kwargs)
        betas = res.betas
        gammas = res.gammas

        results[depth] = res

    return results
