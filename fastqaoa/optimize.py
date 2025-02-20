from dataclasses import dataclass
from typing import Tuple, List
from fastqaoa import params
from fastqaoa.ctypes import Diagonals
from fastqaoa.ctypes.qaoa import grad_qaoa, multi_energy
from fastqaoa.ctypes.qpe_qaoa import grad_qpe_qaoa
from fastqaoa.ctypes.optimize import optimize_qaoa_adam, optimize_qaoa_lbfgs

import numpy as np
from scipy.optimize import minimize

from fastqaoa.params import interpolate, init_linear


def grid_search(
    dg: Diagonals,
    cost: Diagonals,
    dim: int = 20,
    depth: int = 1,
    beta_extent: Tuple[float] = (0, np.pi),
    gamma_extent: Tuple[float] = (0, np.pi),
    return_matrix: bool = False
):
    print("grid_search started")
    if depth == 1:
        b = np.linspace(*beta_extent, dim, endpoint=False)
        g = np.linspace(*gamma_extent, dim, endpoint=False)
        B, G = np.meshgrid(b, g)
        B = np.expand_dims(B.ravel(), -1)
        G = np.expand_dims(G.ravel(), -1)
        res = multi_energy(dg, cost, B, G)
        idx = np.argmin(res)
        return (B[idx, 0], G[idx, 0]), res[idx]
    else:
        betas, gammas = init_linear(depth)
        b = np.linspace(*beta_extent, dim, endpoint=False)
        g = np.linspace(*gamma_extent, dim, endpoint=False)
        B, G = np.meshgrid(b, g)
        Bt = np.outer(B.ravel(), betas)
        Gt = np.outer(G.ravel(), gammas)
        res = multi_energy(dg, cost, Bt, Gt)
        if return_matrix:
            return res.reshape((dim, dim))
        idx = np.argmin(res)
        x, y = np.unravel_index(idx, shape=(dim, dim))
        return (b[x], g[y]), res[idx]

def optimize_interpolate(
    dg: Diagonals,
    cost: Diagonals,
    depths: List[int],
    constr: Diagonals = None,
    first_betas: List[float] = None,
    first_gammas: List[float] = None,
    init_val: float = 0.1,
    use_adam: bool = False,
    **kwargs,
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


@dataclass
class LinearResult:
    delta_beta: float
    delta_gamma: float
    depth: float
    betas: np.ndarray
    gammas: np.ndarray


def optimize_linear(
    dg: Diagonals,
    cost: Diagonals,
    depth: int,
    delta_beta: float = 0.5,
    delta_gamma: float = 0.5,
    constr: Diagonals = None,
    method: str = "BFGS",
    maxiter: int = 100,
):
    fbeta, fgamma = params.init_linear(depth)

    grad_data = {"beta": None, "gammas": None, "exp": None, "current_x": None}

    if constr is None:

        def evalf(x):
            if grad_data["current_x"] is not None and np.allclose(
                grad_data["current_x"], x
            ):
                return
            exp, grad_beta, grad_gamma = grad_qaoa(
                dg, cost, x[0] * fbeta, x[1] * fgamma
            )
            grad_data["beta"] = grad_beta
            grad_data["gamma"] = grad_gamma
            grad_data["exp"] = exp
            grad_data["current_x"] = x

    else:

        def evalf(x):
            if grad_data["current_x"] is not None and np.allclose(
                grad_data["current_x"], x
            ):
                return
            exp, grad_beta, grad_gamma = grad_qpe_qaoa(
                dg, cost, x[0] * fbeta, x[1] * fgamma
            )
            grad_data["beta"] = grad_beta
            grad_data["gamma"] = grad_gamma
            grad_data["exp"] = exp
            grad_data["current_x"] = x

    def fcost(x):
        evalf(x)
        return grad_data["exp"]

    def fjac(x):
        evalf(x)
        return np.array([grad_data["beta"].dot(fbeta), grad_data["gamma"].dot(fgamma)])

    res = minimize(
        fcost,
        x0=[delta_beta, delta_gamma],
        jac=fjac,
        method=method,
        options={"maxiter": maxiter},
    )

    return LinearResult(
        delta_beta=res.x[0],
        delta_gamma=res.x[1],
        depth=depth,
        betas=res.x[0] * fbeta,
        gammas=res.x[1] * fgamma,
    )
