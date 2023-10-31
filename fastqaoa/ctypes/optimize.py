from .diagonals import Diagonals
from .lib import _lib, C
from numpy.ctypeslib import ndpointer
import numpy as np
from enum import Enum


_lib.opt_adam_qaoa.restype = None
_lib.opt_adam_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(np.float64),
    ndpointer(np.float64),
    C.c_double,
    C.c_int,
    C.c_double,
]

_lib.opt_adam_qpe_qaoa.restype = None
_lib.opt_adam_qpe_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(np.float64),
    ndpointer(np.float64),
    C.c_double,
    C.c_int,
    C.c_double,
]


def optimize_qaoa_adam(
    diagonals: Diagonals,
    cost: Diagonals,
    betas: np.ndarray,
    gammas: np.ndarray,
    lr: float = 1e-2,
    maxiter: float = 1000,
    tol: float = 1e-4,
    constr: Diagonals = None,
):
    betas = np.copy(betas)
    gammas = np.copy(betas)
    if constr is None:
        _lib.opt_adam_qaoa(len(betas), diagonals, cost, betas, gammas, lr, maxiter, tol)
    else:
        _lib.opt_adam_qpe_qaoa(
            len(betas), diagonals, cost, constr, betas, gammas, lr, maxiter, tol
        )
    return betas, gammas


LBFGSResult = Enum(
    "LBFGSResult",
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
    ndpointer(np.float64),
    ndpointer(np.float64),
    C.c_int,
]

_lib.opt_lbfgs_qpe_qaoa.restype = C.c_int
_lib.opt_lbfgs_qpe_qaoa.argtypes = [
    C.c_int,
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    ndpointer(np.float64),
    ndpointer(np.float64),
    C.c_int,
]


def optimize_qaoa_lbfgs(
    diagonals: Diagonals,
    cost: Diagonals,
    betas: np.ndarray,
    gammas: np.ndarray,
    maxiter: int = 1000,
    constr: Diagonals = None,
) -> LBFGSResult:
    betas = np.copy(betas)
    gammas = np.copy(betas)
    if constr is None:
        res = _lib.opt_lbfgs_qaoa(len(betas), diagonals, cost, betas, gammas, maxiter)
    else:
        res = _lib.opt_lbfgs_qpe_qaoa(
            len(betas), diagonals, cost, constr, betas, gammas, maxiter
        )
    return LBFGSResult(res), betas, gammas
