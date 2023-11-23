from typing import List
from numpy.ctypeslib import ndpointer
import numpy as np
import qubovert as qv
from functools import partial

from .lib import _lib, C, NP_REAL


class Diagonals(C.Structure):
    _fields_ = [
        ("n_qubits", C.c_uint8),
        ("data", C.POINTER(C.m_real)),
        ("min_val", C.m_real),
        ("max_val", C.m_real),
    ]

    LTE = 0
    GTE = 1
    LT = 2
    GT = 3
    EQ = 4
    NEQ = 5

    def make_plus_state(n_qubits: int) -> "Diagonals":
        ...

    def print(self):
        ...

    def __del__(self):
        ...

    def to_numpy(self) -> np.ndarray:
        ...

    def from_numpy(arr: np.ndarray) -> "Diagonals":
        ...

    def brute_force(
        n_qubits: int, keys: List[int], vals: List[float]  # pyright: ignore
    ) -> "Diagonals":
        ...

    def brute_force_qv(model: qv.PUBO) -> "Diagonals":
        b = {sum(1 << i for i in k) if len(k) > 0 else 0: v for k, v in model.items()}
        keys = list(b.keys())
        vals = list(b.values())

        return Diagonals.brute_force(model.num_binary_variables, keys, vals)

    def mask(
        self,
        lhs: "Diagonals",  # pyright: ignore
        rhs: float,  # pyright: ignore
        cmp_type: int,  # pyright: ignore
        val: float = 0.0,  # pyright: ignore
    ) -> "Diagonals":
        ...

    def quad_penalty(
        self,
        lhs: "Diagonals",  # pyright: ignore
        rhs: float,  # pyright: ignore
        cmp_type: int,  # pyright: ignore
        penalty: float = None,  # pyright: ignore
    ) -> "Diagonals":
        ...

    def __le__(self, other: float):  # pyright: ignore
        ...

    def __ge__(self, other: float):  # pyright: ignore
        ...

    def __lt__(self, other: float): # pyright: ignore
        ...

    def __gt__(self, other: float): # pyright: ignore
        ...

    def __eq__(self, other: float):  # pyright: ignore
        ...

    def __ne__(self, other: float):  # pyright: ignore
        ...

    def cmp(self, other: float, typ: int):  # pyright: ignore
        ...

    def clone(self) -> "Diagonals":
        ...

    def __iadd__(self, other: float) -> "Diagonals": # pyright: ignore
        ...

    def __imul__(self, other: float) -> "Diagonals": # pyright: ignore
        ...

    def __mul__(self, other: float):
        x = self.clone()
        x *= other
        return x

    def __add__(self, other: float):
        x = self.clone()
        x += other
        return x

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def __radd__(self, other: float):
        return self.__add__(other)

    def __truediv__(self, other: float):
        return self.__mul__(1 / other)

_lib.dg_print.argtypes = [C.POINTER(Diagonals)]
_lib.dg_print.restype = None


def __dg_print(self):
    return _lib.dg_print(self)


Diagonals.print = __dg_print

_lib.dg_free.argtypes = [C.POINTER(Diagonals)]
_lib.dg_free.restype = None


def __dg_del(self):
    return _lib.dg_free(self)


Diagonals.__del__ = __dg_del

_lib.dg_copy.argtypes = [C.POINTER(Diagonals), ndpointer(NP_REAL)]
_lib.dg_copy.restype = None


def __to_numpy(self):
    res = np.empty(1 << self.n_qubits, dtype=NP_REAL)
    _lib.dg_copy(self, res)
    return res


Diagonals.to_numpy = __to_numpy

_lib.dg_copy_from.argtypes = [ndpointer(NP_REAL), C.c_uint8]
_lib.dg_copy_from.restype = C.POINTER(Diagonals)


def __from_numpy(arr):
    n_qubits = int(np.log2(arr.shape[0]))
    assert 1 << n_qubits == arr.shape[0], "Not a valid diagonals dimension"
    arr = arr.astype(NP_REAL)
    return _lib.dg_copy_from(arr, n_qubits).contents


Diagonals.from_numpy = __from_numpy

_lib.dg_brute_force.argtypes = [
    C.c_uint8,
    C.c_int,
    ndpointer(np.uint64),
    ndpointer(NP_REAL),
]
_lib.dg_brute_force.restype = C.POINTER(Diagonals)


def __brute_force(n_qubits, keys, vals):
    keys = np.array(keys).astype(np.uint64)
    vals = np.array(vals).astype(NP_REAL)
    assert vals.shape[0] == keys.shape[0], "Keys and vals need to match in dimension"
    return _lib.dg_brute_force(n_qubits, keys.shape[0], keys, vals).contents


Diagonals.brute_force = __brute_force

_lib.dg_mask.argtypes = [
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.m_real,
    C.c_int,
    C.m_real,
]
_lib.dg_mask.restype = C.POINTER(Diagonals)


def __mask(self, lhs, rhs, typ, val=0.0):
    return _lib.dg_mask(self, lhs, rhs, typ, val).contents


Diagonals.mask = __mask


_lib.dg_quad_penalty.argtypes = [
    C.POINTER(Diagonals),
    C.POINTER(Diagonals),
    C.m_real,
    C.c_int,
    C.POINTER(C.m_real),
]
_lib.dg_quad_penalty.restype = C.POINTER(Diagonals)


def __quad_peanlty(self, lhs, rhs, typ, penalty=None):
    if penalty is None:
        penalty = C.m_real(-1)
    else:
        penalty = C.m_real(penalty)
    res = _lib.dg_quad_penalty(self, lhs, rhs, typ, penalty).contents
    return res, penalty.value


Diagonals.quad_penalty = __quad_peanlty

_lib.dg_cmp.argtypes = [C.POINTER(Diagonals), C.m_real, C.c_int]
_lib.dg_cmp.restype = C.POINTER(Diagonals)


def __cmp(lhs, rhs, typ=0):
    return _lib.dg_cmp(lhs, rhs, typ).contents

Diagonals.cmp = __cmp
Diagonals.__le__ = lambda self, other: __cmp(self, other, Diagonals.LTE)
Diagonals.__ge__ = lambda self, other: __cmp(self, other, Diagonals.GTE)
Diagonals.__lt__ = lambda self, other: __cmp(self, other, Diagonals.LT)
Diagonals.__gt__ = lambda self, other: __cmp(self, other, Diagonals.GT)
Diagonals.__eq__ = lambda self, other: __cmp(self, other, Diagonals.EQ)
Diagonals.__ne__ = lambda self, other: __cmp(self, other, Diagonals.NEQ)


_lib.dg_scale.argtypes = [C.POINTER(Diagonals), C.m_real]
_lib.dg_scale.restype = None

_lib.dg_shift.argtypes = [C.POINTER(Diagonals), C.m_real]
_lib.dg_shift.restype = None

_lib.dg_clone.argtypes = [C.POINTER(Diagonals)]
_lib.dg_clone.restype = C.POINTER(Diagonals)

def __clone(self):
    return _lib.dg_clone(self).contents
Diagonals.clone = __clone

def __mul(self, other):
    _lib.dg_scale(self, other)
    return self
Diagonals.__imul__ = __mul
Diagonals.__mul = __mul

def __add(self, other):
    _lib.dg_shift(self, other)
    return self
Diagonals.__iadd__ = __add


