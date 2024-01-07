import numpy as np
from scipy.interpolate import CubicSpline

from fastqaoa.ctypes.diagonals import Diagonals


def get_indicator_base(M: int, subdiv: int):
    subdivs = 2 * np.pi * np.arange(0, 1 << subdiv) / (2 ** (M + subdiv))
    k = 2 * np.arange(2 ** (M - 1)) + 1
    f = 8 * (1 - k / (2**M)) / (1 - np.exp(-2j * np.pi * k / (2**M)))

    h = np.zeros((2**subdiv, 2**M), dtype=np.complex128)
    h[:, k] = f * np.exp(1j * np.outer(subdivs, k))

    res = np.fft.ifft(h)

    # the minus sign cannot be explained in my opinion
    return res.T.ravel().real


def get_indicator_interpolator(
    M: int, subdiv: int, low: float = 0, high: float = 2, shift: float = 0, base=None
):
    x = np.linspace(low, high, (1 << (M + subdiv)) + 1, endpoint=True)
    if base is None:
        base = get_indicator_base(M, subdiv)
    base = np.append(base, [base[0]])
    shift = shift / (1 << M)
    return CubicSpline(x + shift, base, bc_type="periodic")


def interpolate_diagonals(interpolator, diag: Diagonals) -> Diagonals:
    return Diagonals.from_numpy(interpolator(diag.to_numpy()))
