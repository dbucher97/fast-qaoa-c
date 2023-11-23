from typing import Dict

from fastqaoa.ctypes.diagonals import Diagonals
from fastqaoa.ctypes.statevector import Statevector
from .lib import _lib, C


class Metrics(C.Structure):
    _fields_ = [
        ("energy", C.m_real),
        ("approx_ratio", C.m_real),
        ("feas_ratio", C.m_real),
        ("feas_approx_ratio", C.m_real),
        ("p_opt", C.m_real),
        ("p_999", C.m_real),
        ("p_99", C.m_real),
        ("p_9", C.m_real),
        ("rnd_approx_ratio", C.m_real),
        ("min_val", C.m_real),
        ("rnd_val", C.m_real),
        ("max_val", C.m_real),
    ]

    def __del__(self):
        ...

    def compute(
        sv: Statevector, cost: Diagonals, feas: Diagonals  # pyright: ignore
    ) -> "Metrics":
        ...

    def dump(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k, _ in self._fields_}


_lib.mtr_free.argtypes = [C.POINTER(Metrics)]
_lib.mtr_free.restype = None


def __free(self):
    _lib.mtr_free(self)


Metrics.__del__ = __free


_lib.mtr_compute.argtypes = [
    C.POINTER(Statevector),
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
]
_lib.mtr_compute.restype = C.POINTER(Metrics)


def __compute(sv: Statevector, cost: Diagonals, feas: Diagonals) -> Metrics:
    return _lib.mtr_compute(sv, cost, feas).contents


Metrics.compute = __compute
