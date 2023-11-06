from typing import List, Tuple
from numpy.ctypeslib import ndpointer
import numpy as np

from .statevector import Statevector
from .diagonals import Diagonals
from .lib import C, _lib, NP_REAL


_lib.apply_diagonals.argtypes = [
    C.POINTER(Statevector),
    C.POINTER(Diagonals),
    C.m_real,
]
_lib.apply_diagonals.restype = None


def apply_diagonals(sv: Statevector, dg: Diagonals, gamma: float) -> Statevector:
    _lib.apply_diagonals(sv, dg, float(gamma))
    return sv


_lib.apply_rx.argtypes = [C.POINTER(Statevector), C.m_real]
_lib.apply_rx.restype = None


def apply_rx(sv: Statevector, beta: float) -> Statevector:
    _lib.apply_rx(sv, float(beta))
    return sv


_lib.qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
]
_lib.qaoa.restype = C.POINTER(Statevector)


def qaoa(dg: Diagonals, betas: List[float], gammas: List[float]) -> Statevector:
    gammas = np.array(gammas).astype(NP_REAL)
    betas = np.array(betas).astype(NP_REAL)
    assert len(gammas) == len(betas), "Betas and Gammas must have same size"
    sv = _lib.qaoa(len(betas), dg, betas, gammas).contents
    return sv


_lib.grad_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
]
_lib.grad_qaoa.restype = C.m_real


def grad_qaoa(
    dg: Diagonals, cost: Diagonals, betas: List[float], gammas: List[float]
) -> Tuple[float, np.ndarray, np.ndarray]:
    gammas = np.array(gammas).astype(NP_REAL)
    betas = np.array(betas).astype(NP_REAL)
    assert len(gammas) == len(betas), "Betas and Gammas must have same size"
    grad_betas = np.empty_like(betas)
    grad_gammas = np.empty_like(gammas)
    val = _lib.grad_qaoa(len(betas), dg, cost, betas, gammas, grad_betas, grad_gammas)
    return val, grad_betas, grad_gammas

