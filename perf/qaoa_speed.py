import numpy as np
import time
from fastqaoa.ctypes.optimize import optimize_qaoa_lbfgs
from fastqaoa.ctypes.qaoa import qaoa
from fastqaoa.ctypes import Diagonals
import matplotlib.pyplot as plt

import numpy as np

depth = 1

betas = np.ones(depth) * 0.1
gammas = np.ones(depth) * 0.1

times = []
times_fft = []

Ns = range(6, 26 ,2)
for i in Ns:
    x = np.random.randn(1 << i)
    dg = Diagonals.from_numpy(x)

    a = time.perf_counter()
    optimize_qaoa_lbfgs(dg, dg, betas, gammas)
    delta = time.perf_counter() - a
    delta *= 1e3

    times.append(delta)

    print(f"{i}: {delta} ms")


plt.plot(Ns, times)
plt.yscale("log")
plt.show()
