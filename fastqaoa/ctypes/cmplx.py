import ctypes as C

class c_double_complex(C.Structure): 
    _fields_ = [("real", C.c_double),("imag", C.c_double)]

    @property
    def value(self):
        return self.real+1j*self.imag
