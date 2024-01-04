from typing import NamedTuple
from .diagonals import Diagonals
from .lib import _lib, C, NP_REAL
from numpy.ctypeslib import ndpointer
import numpy as np
from enum import Enum
from collections import namedtuple

AdamStatus = Enum("AdamStatus", [("Converged", 0), ("MaxIter", 1)])

_lib.opt_adam_qaoa.restype = C.c_int
_lib.opt_adam_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    C.m_real,
    C.c_int,
    C.m_real,
    C.POINTER(C.c_int),
]

_lib.opt_adam_qpe_qaoa.restype = C.c_int
_lib.opt_adam_qpe_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    C.m_real,
    C.c_int,
    C.m_real,
    C.POINTER(C.c_int),
]


def optimize_qaoa_adam(
    diagonals: Diagonals,
    cost: Diagonals,
    betas: np.ndarray,
    gammas: np.ndarray,
    lr: float = 1e-2,
    maxiter: float = 1000,
    tol: float = 1e-5,
    constr: Diagonals = None,
) -> NamedTuple:
    betas = np.copy(betas).astype(NP_REAL)
    gammas = np.copy(gammas).astype(NP_REAL)
    it = C.c_int(0)
    if constr is None:
        ret = _lib.opt_adam_qaoa(
            len(betas), diagonals, cost, betas, gammas, lr, maxiter, tol, it
        )
    else:
        ret = _lib.opt_adam_qpe_qaoa(
            len(betas), diagonals, cost, constr, betas, gammas, lr, maxiter, tol, it
        )
    Result = namedtuple("AdamResult", "status it betas gammas")
    return Result(status=AdamStatus(ret), it=it.value, betas=betas, gammas=gammas)


LBFGSStatus = Enum(
    "LBFGSStatus",
    [("Success", 0), ("Convergence", 0), ("Stop", 1), ("AlreadyMinimized", 2)]
    + [
        ("Err" + k, i - 1024)
        for i, k in enumerate(
            [
                "UnknownError",
                "LogicError",
                "OutofMemory",
                "Canceled",
                "InvalidN",
                "InvalidNSse",
                "InvalidXSse",
                "InvalidEpsilon",
                "InvalidTestperiod",
                "InvalidDelta",
                "InvalidLinesearch",
                "InvalidMinstep",
                "InvalidMaxstep",
                "InvalidFtol",
                "InvalidWolfe",
                "InvalidGtol",
                "InvalidXtol",
                "InvalidMaxlinesearch",
                "InvalidOrthantwise",
                "InvalidOrthantwiseStart",
                "InvalidOrthantwiseEnd",
                "OutOfInterval",
                "IncorrectTminmax",
                "RoundingError",
                "MinimumStep",
                "MaximumStep",
                "MaximumLinesearch",
                "MaximumIteration",
                "WidthTooSmall",
                "InvalidParameters",
                "IncreaseGradient",
            ]
        )
    ],
)


_lib.opt_lbfgs_qaoa.restype = C.c_int
_lib.opt_lbfgs_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    C.POINTER(C.c_int),
    C.POINTER(C.c_int),
    ndpointer(NP_REAL),
    C.POINTER(C.c_int),
    C.POINTER(C.c_double),
    C.POINTER(C.c_int),
    C.POINTER(C.c_int),
]

_lib.opt_lbfgs_qpe_qaoa.restype = C.c_int
_lib.opt_lbfgs_qpe_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(NP_REAL),
    ndpointer(NP_REAL),
    C.POINTER(C.c_int),
    C.POINTER(C.c_int),
    ndpointer(NP_REAL),
    C.POINTER(C.c_int),
    C.POINTER(C.c_double),
    C.POINTER(C.c_int),
    C.POINTER(C.c_int),
]


def optimize_qaoa_lbfgs(
    diagonals: Diagonals,
    cost: Diagonals,
    betas: np.ndarray,
    gammas: np.ndarray,
    constr: Diagonals = None,
    maxiter: int = 100,
    tol: float = 1e-2,
    linesearch: int = None,
    m: int = 100,
) -> NamedTuple:
    betas = np.copy(betas).astype(NP_REAL)
    gammas = np.copy(gammas).astype(NP_REAL)
    it = C.c_int(0)
    calls = C.c_int(0)
    log = np.zeros(maxiter)
    tol = C.c_double(tol) if tol is not None else None
    linesearch = C.c_int(linesearch) if linesearch is not None else None
    maxiter = C.c_int(maxiter) if maxiter is not None else None
    m = C.c_int(m) if m else None
    if constr is None:
        res = _lib.opt_lbfgs_qaoa(
            len(betas),
            diagonals,
            cost,
            betas,
            gammas,
            it,
            calls,
            log,
            maxiter,
            tol,
            linesearch,
            m,
        )
    else:
        res = _lib.opt_lbfgs_qpe_qaoa(
            len(betas),
            diagonals,
            cost,
            constr,
            betas,
            gammas,
            it,
            calls,
            log,
            maxiter,
            tol,
            linesearch,
            m,
        )

    Result = namedtuple("LBFGSResult", "status it betas gammas calls log")
    log = log[: it.value]
    return Result(
        status=LBFGSStatus(res),
        betas=betas,
        gammas=gammas,
        it=it.value,
        calls=calls.value,
        log=log,
    )
