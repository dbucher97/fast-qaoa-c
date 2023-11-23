import numpy as np


def init_random(depth, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(depth), np.random.randn(depth)


def init_const(depth):
    return np.ones(depth) / depth, np.ones(depth) / depth


def init_linear(depth):
    r = (np.arange(depth) + 0.5) / depth
    return 1 - r, r


def interpolate(depth, betas, gammas):
    a, b = np.linspace(0, 1, len(betas)), np.linspace(0, 1, depth)
    return (
        np.interp(b, a, betas) * len(betas) / depth,
        np.interp(b, a, gammas) * len(gammas) / depth,
    )
