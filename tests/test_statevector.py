import pytest
from fastqaoa.ctypes import Statevector
import numpy as np

def test_numpy_conversion():
    s = np.random.randn(1 << 10) + 1j * np.random.randn(1 << 10)
    sv = Statevector.from_numpy(s)
    s2 = sv.to_numpy()
    assert np.allclose(s, s2)

def test_numpy_conversion_failed():
    s = np.random.randn(1000) + 1j * np.random.randn(1000)
    with pytest.raises(AssertionError):
        Statevector.from_numpy(s)

def test_plus_state():
    d = Statevector.make_plus_state(10).to_numpy()
    assert np.isclose((np.abs(d) ** 2).sum(), 1)
    assert np.allclose(d, d[0])
