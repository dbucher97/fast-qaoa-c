import ctypes as C
from .cmplx import c_double_complex
from numpy.ctypeslib import ndpointer
import numpy as np

from .lib import _lib

class Statevector(C.Structure):
    _fields_ = [
        ("n_qubits", C.c_uint8),
        ("data", C.POINTER(c_double_complex)),
    ]

    def make_plus_state(n_qubits: int) -> "Statevector":
        ...

    def print(self):
        ...

    def __del__(self):
        ...

    def to_numpy(self) -> np.ndarray:
        ...

    def from_numpy(arr: np.ndarray) -> "Statevector":
        ...

_lib.sv_make_plus_state.argtypes = [C.c_uint8]
_lib.sv_make_plus_state.restype = C.POINTER(Statevector)

def __make_plus_state(n):
    return _lib.sv_make_plus_state(n).contents
Statevector.make_plus_state = __make_plus_state

_lib.sv_print.argtypes = [C.POINTER(Statevector)]
_lib.sv_print.restype = None
def __sv_print(self):
    return _lib.sv_print(self)
Statevector.print = __sv_print

_lib.sv_free.argtypes = [C.POINTER(Statevector)]
_lib.sv_free.restype = None

def __sv_del(self):
    return _lib.sv_free(self)
Statevector.__del__ = __sv_del

_lib.sv_copy.argtypes = [C.POINTER(Statevector), ndpointer(np.complex128)]
_lib.sv_copy.restype = None

def __to_numpy(self):
    res = np.empty(1 << self.n_qubits, dtype=np.complex128)
    _lib.sv_copy(self, res)
    return res
Statevector.to_numpy = __to_numpy

_lib.sv_copy_from.argtypes = [ndpointer(np.complex128), C.c_uint8]
_lib.sv_copy_from.restype = C.POINTER(Statevector)

def __from_numpy(arr):
    n_qubits = int(np.log2(arr.shape[0]))
    assert 1 << n_qubits == arr.shape[0], "Not a valid statevector dimension"
    return _lib.sv_copy_from(arr, n_qubits).contents

Statevector.from_numpy = __from_numpy
