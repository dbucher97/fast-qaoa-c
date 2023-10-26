import pytest
from fastqaoa.ctypes import Diagonals
import numpy as np

def test_numpy_conversion():
    s = np.random.randn(1 << 10)
    dg = Diagonals.from_numpy(s)
    s2 = dg.to_numpy()
    assert np.allclose(s, s2)

def test_numpy_conversion_failed():
    s = np.random.randn(1000) + 1j * np.random.randn(1000)
    with pytest.raises(AssertionError):
        Diagonals.from_numpy(s)

def test_brute_force():
    keys = [0b01, 0b10, 0b11]
    vals = [-1, -1, 1.5]
    diag = Diagonals.brute_force(2, keys, vals)
    assert np.all(diag.to_numpy() == np.array([0, -1, -1, -0.5]))

def test_cmp():
    x = np.arange(16)
    dg = Diagonals.from_numpy(x)

    assert np.allclose((dg <= 5).to_numpy(), x <= 5)
    assert np.allclose((dg >= 7).to_numpy(), x >= 7)
    assert np.allclose((dg == 9).to_numpy(), x == 9)
    assert np.allclose((dg != 12).to_numpy(), x != 12)


def test_mask():
    x = np.arange(16)
    y = np.random.randn(16)

    dgx = Diagonals.from_numpy(x)
    dgy = Diagonals.from_numpy(y)

    ry = np.copy(y)
    ry[x > 5] = 0
    assert np.allclose(dgy.mask(dgx, 5, Diagonals.LTE).to_numpy(), ry)
    ry = np.copy(y)
    ry[x < 7] = 1.5
    assert np.allclose(dgy.mask(dgx, 7, Diagonals.GTE, 1.5).to_numpy(), ry)
    ry = np.copy(y)
    ry[x == 12] = -3
    assert np.allclose(dgy.mask(dgx, 12, Diagonals.NEQ, -3).to_numpy(), ry)

def test_quad_penalty():
    x = np.arange(16)
    y = np.random.randn(16)

    dgx = Diagonals.from_numpy(x)
    dgy = Diagonals.from_numpy(y)

    ry = np.copy(y)
    ry[x > 5] += ((x - 5) ** 2)[x > 5]
    assert np.allclose(dgy.quad_penalty(dgx, 5, Diagonals.LTE, 1)[0].to_numpy(), ry)
    ry = np.copy(y)
    ry[x < 7] += (1.5 * (x - 7) ** 2)[x < 7]
    assert np.allclose(dgy.quad_penalty(dgx, 7, Diagonals.GTE, 1.5)[0].to_numpy(), ry)
    ry = np.copy(y)
    ry[x != 12] += (5 * (x - 12) ** 2)[x != 12]
    assert np.allclose(dgy.quad_penalty(dgx, 12, Diagonals.EQ, 5)[0].to_numpy(), ry)

def test_auto_quad_penalty():
    x = np.arange(16)
    y = 0.1 * np.abs(np.arange(16) - 8.1)

    dgx = Diagonals.from_numpy(x)
    dgy = Diagonals.from_numpy(y)

    dgp, _ = dgy.quad_penalty(dgx, 5, Diagonals.LTE)

    z = dgp.to_numpy()
    assert np.min(z) == dgp.min_val
    idxs = np.argsort(z)

    assert x[idxs[0]] <= 5
    assert np.isclose(z[idxs[1]], z[idxs[2]])
    assert np.any(x[idxs[1:3]] <= 5)
    assert np.any(x[idxs[1:3]] > 5)

    dgp, _ = dgy.quad_penalty(dgx, 12, Diagonals.GTE)

    z = dgp.to_numpy()
    assert np.min(z) == dgp.min_val
    idxs = np.argsort(z)

    assert x[idxs[0]] >= 12
    assert np.isclose(z[idxs[1]], z[idxs[2]])
    assert np.any(x[idxs[1:3]] <= 12)
    assert np.any(x[idxs[1:3]] > 12)

def test_auto_quad_penalty2():
    x = np.arange(16)

    for _ in range(100):
        y = np.random.randn(16)

        dgx = Diagonals.from_numpy(x)
        dgy = Diagonals.from_numpy(y)

        ub = np.random.randint(1, 16)
        dgp, _ = dgy.quad_penalty(dgx, ub, Diagonals.LTE)

        y2 = np.copy(y)
        y2[x > 5] = 1000
        idxs = np.argsort(y2)
        e2 = y[idxs[1]]

        z = dgp.to_numpy()
        assert np.min(z) == dgp.min_val

        idxs = np.argsort(z)

        assert x[idxs[0]] <= ub
        assert e2 <= x[idxs[1]] + 1e-7

def test_scale_and_shift():
    y = np.random.randn(16)
    x = np.random.randn()
    z = np.random.randn()
    dg = Diagonals.from_numpy(y)

    dg *= x
    dg += z

    assert np.allclose(dg.to_numpy(), x * y + z)

    assert np.isclose(dg.min_val, np.min(x * y + z))
    assert np.isclose(dg.max_val, np.max(x * y + z))
