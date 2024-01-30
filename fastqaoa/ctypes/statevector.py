from .cmplx import c_complex
from numpy.ctypeslib import ndpointer
import numpy as np

from .lib import _lib, C, NP_COMPLEX

class Statevector(C.Structure):
    _fields_ = [
        ("n_qubits", C.c_uint8),
        ("data", C.POINTER(c_complex)),
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

    def sample(self, num: int) -> np.ndarray:
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

_lib.sv_copy.argtypes = [C.POINTER(Statevector), ndpointer(NP_COMPLEX)]
_lib.sv_copy.restype = None

def __to_numpy(self):
    res = np.empty(1 << self.n_qubits, dtype=NP_COMPLEX)
    _lib.sv_copy(self, res)
    return res
Statevector.to_numpy = __to_numpy

_lib.sv_copy_from.argtypes = [ndpointer(NP_COMPLEX), C.c_uint8]
_lib.sv_copy_from.restype = C.POINTER(Statevector)

def __from_numpy(arr):
    n_qubits = int(np.log2(arr.shape[0]))
    assert 1 << n_qubits == arr.shape[0], "Not a valid statevector dimension"
    arr = arr.astype(NP_COMPLEX)
    return _lib.sv_copy_from(arr, n_qubits).contents

Statevector.from_numpy = __from_numpy


_lib.sv_sample.argtypes = [C.POINTER(Statevector), C.c_int, ndpointer(np.uint32)]
_lib.sv_sample.restype = None

def __sample(self, num: int):
    ret = np.empty(num, dtype=np.uint32)
    _lib.sv_sample(self, num, ret)
    return ret

Statevector.sample = __sample




