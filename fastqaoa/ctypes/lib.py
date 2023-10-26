import ctypes as C

_lib = C.CDLL("build/libqaoa.dylib")

__all__ = ["_lib", "C"]
