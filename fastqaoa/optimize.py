from typing import Tuple
from fastqaoa.ctypes import Diagonals
from fastqaoa.ctypes.qaoa import multi_energy

import numpy as np


def grid_search(
    dg: Diagonals,
    cost: Diagonals,
    dim_num: int = 20,
    beta_extent: Tuple[float] = (0, np.pi),
    gamma_extent: Tuple[float] = (0, np.pi),
):
    b = np.linspace(*beta_extent, dim_num, endpoint=False)
    g = np.linspace(*gamma_extent, dim_num, endpoint=False)
    B, G = np.meshgrid(b, g)
    B = np.expand_dims(B.ravel(), -1)
    G = np.expand_dims(G.ravel(), -1)
    res = multi_energy(dg, cost, B, G)
    idx = np.argmin(res)
    return B[idx, 0], G[idx, 0]
