import os
import ctypes as C
import numpy as np
from sys import platform

__libname = "libqaoa"

if os.environ.get("QAOA_FLOAT", "64") == "32":
    __libname += "32"
    C.m_real = C.c_float
    NP_REAL = np.float32
    NP_COMPLEX = np.complex64
else:
    C.m_real = C.c_double
    NP_REAL = np.float64
    NP_COMPLEX = np.complex128

if platform == "linux":
    __ext = "so"
elif platform == "darwin":
    __ext = "dylib"
else:
    raise RuntimeError(f"Platform {platform} not supported.")

_lib = C.CDLL(f"build/{__libname}.{__ext}")

__all__ = ["_lib", "C", "NP_REAL", "NP_COMPLEX"]
