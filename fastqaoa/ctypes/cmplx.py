from .lib import C

class c_complex(C.Structure):
    _fields_ = [("real", C.m_real),("imag", C.m_real)]

    @property
    def value(self):
        return self.real+1j*self.imag
