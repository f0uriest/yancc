"""Tests for finite differences."""

import numpy as np

from yancc.finite_diff import fdbwd, fdctr, fdfwd


def test_fdctr_periodic():
    """Validate centered finite differences w/ periodic bc."""
    N = 21
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    f = np.sin(x)
    df = np.cos(x)
    ddf = -np.sin(x)
    h = 2 * np.pi / N
    for order in [2, 4, 6]:
        dfi = fdctr(f, order, 1, h, bc="periodic")
        np.testing.assert_allclose(
            df, dfi, atol=h ** (order + 1), rtol=h ** (order + 1)
        )
    for order in [2, 4, 6]:
        ddfi = fdctr(f, order, 2, h, bc="periodic")
        np.testing.assert_allclose(
            ddf, ddfi, atol=h ** (order + 1), rtol=h ** (order + 1)
        )


def test_fdfwd_periodic():
    """Validate forward finite differences w/ periodic bc."""
    N = 21
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    f = np.sin(x)
    df = np.cos(x)
    ddf = -np.sin(x)
    h = 2 * np.pi / N
    for order in [1, 2, 3, 4, 5, 6]:
        dfi = fdfwd(f, order, 1, h, bc="periodic")
        np.testing.assert_allclose(df, dfi, atol=h ** (order), rtol=h ** (order))
    for order in [1, 2, 3, 4, 5, 6]:
        ddfi = fdfwd(f, order, 2, h, bc="periodic")
        np.testing.assert_allclose(ddf, ddfi, atol=h ** (order), rtol=h ** (order))


def test_fdbwd_periodic():
    """Validate backward finite differences w/ periodic bc."""
    N = 21
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    f = np.sin(x)
    df = np.cos(x)
    ddf = -np.sin(x)
    h = 2 * np.pi / N
    for order in [1, 2, 3, 4, 5, 6]:
        dfi = fdbwd(f, order, 1, h, bc="periodic")
        np.testing.assert_allclose(df, dfi, atol=h ** (order), rtol=h ** (order))
    for order in [1, 2, 3, 4, 5, 6]:
        ddfi = fdbwd(f, order, 2, h, bc="periodic")
        np.testing.assert_allclose(ddf, ddfi, atol=h ** (order), rtol=h ** (order))


def test_fdctr_symmetric():
    """Validate centered finite differences w/ symmetric bc."""
    N = 21
    x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
    f = np.cos(x)
    df = -np.sin(x)
    ddf = -np.cos(x)
    h = np.pi / N
    for order in [2, 4, 6]:
        dfi = fdctr(f, order, 1, h, bc="symmetric")
        np.testing.assert_allclose(
            df, dfi, atol=h ** (order + 1), rtol=h ** (order + 1)
        )
    for order in [2, 4, 6]:
        ddfi = fdctr(f, order, 2, h, bc="symmetric")
        np.testing.assert_allclose(
            ddf, ddfi, atol=h ** (order + 1), rtol=h ** (order + 1)
        )


def test_fdfwd_symmetric():
    """Validate forward finite differences w/ symmetric bc."""
    N = 21
    x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
    f = np.cos(x)
    df = -np.sin(x)
    ddf = -np.cos(x)
    h = np.pi / N
    for order in [1, 2, 3, 4, 5, 6]:
        dfi = fdfwd(f, order, 1, h, bc="symmetric")
        np.testing.assert_allclose(df, dfi, atol=h ** (order), rtol=h ** (order))
    for order in [1, 2, 3, 4, 5, 6]:
        ddfi = fdfwd(f, order, 2, h, bc="symmetric")
        np.testing.assert_allclose(ddf, ddfi, atol=h ** (order), rtol=h ** (order))


def test_fdbwd_symmetric():
    """Validate backward finite differences w/ symmetric bc."""
    N = 21
    x = np.linspace(0, np.pi, N, endpoint=False) + np.pi / 2 / N
    f = np.cos(x)
    df = -np.sin(x)
    ddf = -np.cos(x)
    h = np.pi / N
    for order in [1, 2, 3, 4, 5, 6]:
        dfi = fdbwd(f, order, 1, h, bc="symmetric")
        np.testing.assert_allclose(df, dfi, atol=h ** (order), rtol=h ** (order))
    for order in [1, 2, 3, 4, 5, 6]:
        ddfi = fdbwd(f, order, 2, h, bc="symmetric")
        np.testing.assert_allclose(ddf, ddfi, atol=h ** (order), rtol=h ** (order))
