from typing import List, Tuple
from numpy.ctypeslib import ndpointer
import numpy as np

from fastqaoa.ctypes.diagonals import Diagonals
from fastqaoa.ctypes.statevector import Statevector
from .lib import _lib, C

_lib.apply_qpe_diagonals_normalized.argtypes = [
    C.POINTER(Statevector),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.c_double,
    C.POINTER(C.c_double),
]
_lib.apply_qpe_diagonals_normalized.restype = None


def apply_qpe_diagonals_normalized(
    sv: Statevector, dg: Diagonals, constr: Diagonals, gamma: float
) -> float:
    psucc = C.c_double(0)
    _lib.apply_qpe_diagonals_normalized(sv, dg, constr, gamma, psucc)
    return psucc.value


_lib.apply_qpe_diagonals.argtypes = [
    C.POINTER(Statevector),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.c_double,
]
_lib.apply_qpe_diagonals.restype = None


def apply_qpe_diagonals(
    sv: Statevector, dg: Diagonals, constr: Diagonals, gamma: float
) -> None:
    _lib.apply_qpe_diagonals(sv, dg, constr, gamma)


_lib.qpe_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(np.float64),
    ndpointer(np.float64),
    C.POINTER(C.c_double),
]

_lib.qpe_qaoa.restype = C.POINTER(Statevector)

def qpe_qaoa(dg: Diagonals, constr: Diagonals, betas: List[float], gammas: List[float]):
    gammas = np.array(gammas).astype(np.float64)
    betas = np.array(betas).astype(np.float64)
    psucc = C.c_double(0)
    assert len(gammas) == len(betas), "Betas and Gammas must have same size"
    sv = _lib.qpe_qaoa(len(betas), dg, constr, betas, gammas, psucc).contents
    return sv, psucc.value

_lib.grad_qpe_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(np.float64),
    ndpointer(np.float64),
    ndpointer(np.float64),
    ndpointer(np.float64),
    C.POINTER(C.c_double),
    C.POINTER(C.c_double),
]
_lib.grad_qpe_qaoa.restype = None

def grad_qpe_qaoa(
    dg: Diagonals, cost: Diagonals, constr: Diagonals, betas: List[float], gammas: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    gammas = np.array(gammas).astype(np.float64)
    betas = np.array(betas).astype(np.float64)
    assert len(gammas) == len(betas), "Betas and Gammas must have same size"
    grad_betas = np.empty_like(betas)
    grad_gammas = np.empty_like(gammas)
    psucc = C.c_double(0)
    expectation_value = C.c_double(0)
    _lib.grad_qpe_qaoa(len(betas), dg, cost, constr, betas, gammas, grad_betas,
                       grad_gammas, psucc, expectation_value)

    return grad_betas, grad_gammas

