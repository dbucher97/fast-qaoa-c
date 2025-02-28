from typing import List, Tuple
from numpy.ctypeslib import ndpointer
import numpy as np

from fastqaoa.ctypes.diagonals import Diagonals
from fastqaoa.ctypes.statevector import Statevector
from .lib import _lib, C, NP_REAL

_lib.apply_qpe_diagonals_normalized.argtypes = [
    C.POINTER(Statevector),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.m_real,
    C.POINTER(C.m_real),
]
_lib.apply_qpe_diagonals_normalized.restype = None


def apply_qpe_diagonals_normalized(
    sv: Statevector, dg: Diagonals, constr: Diagonals, gamma: float
) -> float:
    psucc = C.m_real(0)
    _lib.apply_qpe_diagonals_normalized(sv, dg, constr, gamma, psucc)
    return psucc.value


_lib.apply_qpe_diagonals.argtypes = [
    C.POINTER(Statevector),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.m_real,
]
_lib.apply_qpe_diagonals.restype = None


def apply_qpe_diagonals(
    sv: Statevector, dg: Diagonals, constr: Diagonals, gamma: float
) -> None:
    _lib.apply_qpe_diagonals(sv, dg, constr, gamma)
    return sv


_lib.qpe_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    C.POINTER(C.m_real),
]

_lib.qpe_qaoa.restype = C.POINTER(Statevector)


def qpe_qaoa(dg: Diagonals, constr: Diagonals, betas: List[float], gammas: List[float]):
    gammas = np.array(gammas).astype(NP_REAL)
    betas = np.array(betas).astype(NP_REAL)
    psucc = C.m_real(0)
    assert len(gammas) == len(betas), "Betas and Gammas must have same size"
    sv = _lib.qpe_qaoa(len(betas), dg, constr, betas, gammas, psucc).contents
    return sv, psucc.value


_lib.qpe_qaoa_norm.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    C.POINTER(C.m_real),
    ndpointer(NP_REAL),
]

_lib.qpe_qaoa_norm.restype = C.POINTER(Statevector)


def qpe_qaoa_norm(
    dg: Diagonals, constr: Diagonals, betas: List[float], gammas: List[float]
):
    gammas = np.array(gammas).astype(NP_REAL)
    betas = np.array(betas).astype(NP_REAL)
    psucc = C.m_real(0)
    qvals = np.zeros(len(betas), dtype=NP_REAL)
    assert len(gammas) == len(betas), "Betas and Gammas must have same size"
    sv = _lib.qpe_qaoa_norm(
        len(betas), dg, constr, betas, gammas, psucc, qvals
    ).contents
    return sv, psucc.value, qvals


_lib.grad_qpe_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    C.POINTER(C.m_real),
    C.POINTER(C.m_real),
]
_lib.grad_qpe_qaoa.restype = None


def grad_qpe_qaoa(
    dg: Diagonals,
    cost: Diagonals,
    constr: Diagonals,
    betas: List[float],
    gammas: List[float],
) -> Tuple[float, np.ndarray, np.ndarray]:
    gammas = np.array(gammas).astype(NP_REAL)
    betas = np.array(betas).astype(NP_REAL)
    assert len(gammas) == len(betas), "Betas and Gammas must have same size"
    grad_betas = np.empty_like(betas)
    grad_gammas = np.empty_like(gammas)
    psucc = C.m_real(0)
    expectation_value = C.m_real(0)
    _lib.grad_qpe_qaoa(
        len(betas),
        dg,
        cost,
        constr,
        betas,
        gammas,
        grad_betas,
        grad_gammas,
        psucc,
        expectation_value,
    )

    return expectation_value, grad_betas, grad_gammas
