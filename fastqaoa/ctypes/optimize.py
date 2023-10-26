from .diagonals import Diagonals
from .lib import _lib, C
from numpy.ctypeslib import ndpointer
import numpy as np

_lib.opt_optimize_qaoa.restype = None
_lib.opt_optimize_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(np.float64),
    ndpointer(np.float64),
    C.c_double,
    C.c_int,
    C.c_double,
]


def optimize_qaoa(
    diagonals: Diagonals,
    cost: Diagonals,
    betas: np.ndarray,
    gammas: np.ndarray,
    lr: float = 1e-2,
    maxiter: float = 1000,
    tol: float = 1e-4,
):
    betas = np.copy(betas)
    gammas = np.copy(betas)
    _lib.opt_optimize_qaoa(len(betas), diagonals, cost, betas, gammas, lr, maxiter, tol)
    return betas, gammas
