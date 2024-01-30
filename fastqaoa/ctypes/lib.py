import os
import ctypes as C
import numpy as np
from sys import platform
from pathlib import Path

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

path = Path(__file__).absolute().parent
_lib = C.CDLL(path / f"{__libname}.{__ext}")
del path

__all__ = ["_lib", "C", "NP_REAL", "NP_COMPLEX"]
