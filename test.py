import qubovert as qv
from fastqaoa.ctypes import Diagonals
from fastqaoa.ctypes.qaoa import qaoa
from fastqaoa.ctypes.optimize import optimize_qaoa_lbfgs
import numpy as np

# replica of the hamiltonian
puso = qv.PUSO(
    {
        (0,): 0.5,
        (1,): 0.5,
        (2,): 1.25,
        (3,): -0.25,
        (0, 1): 0.75,
        (0, 2): 0.75,
        (1, 2): 0.75,
        (2, 3): 0.75,
    }
)

b = puso.to_pubo()

b = {sum(1 << i for i in k) if len(k) > 0 else 0: v for k,v in b.items()}
keys = list(b.keys())
vals = list(b.values())

dg = Diagonals.brute_force(4, keys, vals)

betas = np.ones(5) * 0.1
gammas = np.ones(5) * 0.1

a = np.abs(qaoa(dg, betas, gammas).to_numpy()) ** 2
a = a.dot(dg.to_numpy())

_, betas, gammas = optimize_qaoa_lbfgs(dg, dg, betas, gammas, maxiter=10)

b = np.abs(qaoa(dg, betas, gammas).to_numpy()) ** 2
b = b.dot(dg.to_numpy())

assert b < a
