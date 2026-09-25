"""Tests for scalar root finding."""

import pathlib

import interpax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yancc.root import deflated_root_scalar

_DATA = pathlib.Path(__file__).parent / "data"
# solved flux scans to interpolate, as (file, column of the field, columns of the
# particle flux of each species, charge of each species in units of e)
_SCANS = {
    "w7x": ("w7x_kjm_ambipolar_scan.txt", 0, (2, 3), (1.0, -1.0)),
    "ncsx": (
        "20260528-01_sfincs_yancc_benchmark_NCSX_2species_Er_scan.txt",
        0,
        (3, 4),
        (-1.0, 1.0),
    ),
}


def _ambipolar_residual(case, noise):
    """Normalized ambipolar current of an interpolated flux scan, and its bounds."""
    name, xcol, fluxcols, charges = _SCANS[case]
    data = np.loadtxt(_DATA / name)
    x = jnp.array(data[:, xcol])
    currents = jnp.array(data[:, fluxcols] * np.array(charges))
    frequencies = jnp.array([3e4, 7e4]) / float(x[-1] - x[0])

    def fun(y, nevals):
        qG = interpax.interp1d(y, x, currents, method="cubic")
        qG = qG * (1 + noise * jnp.sin(frequencies * y))
        return qG.sum() / jnp.abs(qG).sum(), nevals + 1

    return fun, (float(x[0]), float(x[-1]))


def _sweep_roots(fun, lo, hi, num=20001):
    """Roots of fun in (lo, hi), and its slope there, to check the search against."""
    f_of = jax.jit(jax.vmap(lambda y: fun(y, 0)[0]))
    x = np.linspace(lo, hi, num)
    f = np.asarray(f_of(jnp.array(x)))
    i = np.nonzero(f[:-1] * f[1:] < 0)[0]
    # the sweep only brackets each root, so bisect to make the reference exact enough
    # to compare the located roots against rather than only their residuals
    a, b = x[i], x[i + 1]
    fa = f[i]
    for _ in range(60):
        m = (a + b) / 2
        fm = np.asarray(f_of(jnp.array(m)))
        left = fa * fm < 0
        a, b = np.where(left, a, m), np.where(left, m, b)
        fa = np.where(left, fa, fm)
    roots = (a + b) / 2
    i = np.argsort(roots)
    slope = np.asarray(jax.vmap(jax.grad(lambda y: fun(y, 0)[0]))(jnp.array(roots)))
    return roots[i], slope[i]


def test_deflated_root_scalar_multiple_roots():
    """Deflation finds all roots of a stateful cubic, then reports failure.

    The state records the last evaluated point and the number of evaluations, to check
    that the stored state for each root is consistent with the returned root and that
    the step count covers all evaluations, or, when the state is carried from one
    search to the next, that the evaluations keep adding up across searches.
    """
    roots = np.array([-3.0, 0.5, 2.0])

    def fun(x, state):
        _, nevals = state
        return (x - roots[0]) * (x - roots[1]) * (x - roots[2]), (x, nevals + 1)

    with pytest.raises(ValueError):
        deflated_root_scalar(fun, 0.0, 1, args=(0.0, 0), method="bisect")

    for x0, method, carry_state in [(0.0, "newton", False), (3.9, "secant", True)]:
        xs, (nsearch, success, fs, steps, (last_x, nevals), searches) = (
            deflated_root_scalar(
                fun,
                jnp.array(x0),
                4,
                args=(jnp.array(0.0), jnp.array(0)),
                bounds=(-4.0, 4.0),
                ftol=jnp.array(1e-10),
                xatol=jnp.array(0.0),
                xrtol=jnp.array(1e-10),
                maxiter=50,
                method=method,
                carry_state=carry_state,
                full_output=True,
            )
        )
        np.testing.assert_array_equal(success, [True, True, True, False])
        np.testing.assert_allclose(np.sort(xs[:3]), roots, rtol=1e-8)
        assert np.isinf(xs[3])
        np.testing.assert_allclose(fs[:3], 0.0, atol=1e-8)
        np.testing.assert_array_equal(last_x[:3], xs[:3])
        if carry_state:
            assert np.all(np.diff(nevals[:3]) >= steps[1:3])
            assert nevals[0] >= steps[0]
        else:
            np.testing.assert_array_equal(nevals[:3], steps[:3])
        # the bounds are sampled and then searched from, so finding three roots takes
        # more than three searches, but the search still stops on its own
        nsearch = int(nsearch)
        assert nsearch < len(searches[0])
        np.testing.assert_array_equal(
            searches[0], [True] * nsearch + [False] * (len(searches[0]) - nsearch)
        )


@pytest.mark.parametrize(
    "case, noise, nroots",
    [("w7x", 0.0, 3), ("w7x", 1e-5, 3), ("ncsx", 1e-5, 1)],
)
def test_deflated_root_scalar_ambipolar_currents(case, noise, nroots):
    """Every ambipolar root of an interpolated current scan is found, and no others.

    Interpolating a solved scan gives the shape of a real ambipolar residual at no
    solver cost. The W7-X case has three roots, the upper two separated by a shallow
    extremum; the NCSX case has one, at the edge of a region where the residual
    saturates because both fluxes there have the same sign. The fluxes are perturbed
    by a rapidly varying term to mimic evaluating them with a finite solver tolerance.
    More roots are asked for than exist so that the search has to stop on its own.
    """
    ftol = 1e-4
    fun, (lo, hi) = _ambipolar_residual(case, noise)
    expected, slope = _sweep_roots(fun, lo, hi)
    assert len(expected) == nroots
    # driving the residual to ftol locates a root only to ftol over the slope there,
    # which differs by orders of magnitude between the two cases
    atol = 2 * ftol / np.abs(slope).min()

    xs, (_, success, fs, _, nevals, _) = deflated_root_scalar(
        fun,
        jnp.array(hi),
        3,
        args=jnp.array(0),
        bounds=(lo, hi),
        ftol=jnp.array(ftol),
        xatol=jnp.array(0.0),
        xrtol=jnp.array(1e-8),
        full_output=True,
    )
    assert success.sum() == nroots
    np.testing.assert_allclose(np.sort(xs[success]), expected, rtol=0, atol=atol)
    assert np.all(np.abs(fs[success]) < ftol)
    assert np.all(nevals[success] > 0)


@pytest.mark.parametrize(
    "drive, nroots, x0", [(1.3, 3, 0.0), (1.5, 1, 0.0), (0.7, 3, 5.0)]
)
def test_deflated_root_scalar_merging_roots(drive, nroots, x0):
    """Roots are found on both sides of the point where a pair of them merges.

    Fluxes of the form of transport quenched by the electric field give an ambipolar
    residual whose outer pair of roots approaches and annihilates as the drive is
    raised, so this covers a nearly tangential root, where the residual pins the root
    location only to the square root of its own tolerance, and the case just past the
    merge, where the search has to conclude that the pair is gone. At low drive the
    pair is wide apart but, started from a bound, no sign change points at either of
    them, so they are only reachable by sampling the interior.
    """
    ftol = 1e-4
    bounds = (-2.0, 5.0)

    def fun(x, nevals):
        # transport coefficient of each species, peaked at zero field and quenched by
        # it on a scale set by the species, on top of a field independent floor
        D = (
            0.01
            + jnp.array([0.3, 0.03]) / (1 + (x / jnp.array([0.3, 6.0])) ** 2) ** 1.5
        )
        qG = D * jnp.array([-(drive + x), 2.0 - x])
        return qG.sum() / jnp.abs(qG).sum(), nevals + 1

    expected, _ = _sweep_roots(fun, *bounds)
    assert len(expected) == nroots

    xs, (_, success, fs, *_) = deflated_root_scalar(
        fun,
        jnp.array(x0),
        3,
        args=jnp.array(0),
        bounds=bounds,
        ftol=jnp.array(ftol),
        xatol=jnp.array(0.0),
        xrtol=jnp.array(1e-8),
        full_output=True,
    )
    assert success.sum() == nroots
    # near the merge the residual is quadratic in the distance to either root, so the
    # roots are located to much less accuracy than the residual is driven to
    np.testing.assert_allclose(np.sort(xs[success]), expected, atol=1e-3)
    assert np.all(np.abs(fs[success]) < ftol)


def test_deflated_root_scalar_seeded():
    """Guesses are searched from one at a time, and missing ones are still found.

    Ambipolar roots move smoothly from one flux surface to the next, so the roots of a
    solved surface are good guesses for the next one. Guesses displaced from the roots
    must still reach all of them, and giving fewer guesses than there are roots must
    fall back to searching for the rest.
    """
    ftol = 1e-4
    fun, (lo, hi) = _ambipolar_residual("w7x", 0.0)
    expected, slope = _sweep_roots(fun, lo, hi)
    atol = 2 * ftol / np.abs(slope).min()
    # displaced as if carried over from a neighboring surface, and a single guess
    displaced = expected + 0.03 * (hi - lo)
    for guess in [displaced, displaced[:1]]:
        xs, (_, success, fs, *_) = deflated_root_scalar(
            fun,
            jnp.asarray(guess),
            3,
            args=jnp.array(0),
            bounds=(lo, hi),
            ftol=jnp.array(ftol),
            xatol=jnp.array(0.0),
            xrtol=jnp.array(1e-8),
            full_output=True,
        )
        assert success.sum() == len(expected)
        np.testing.assert_allclose(np.sort(xs[success]), expected, rtol=0, atol=atol)
        assert np.all(np.abs(fs[success]) < ftol)


def test_deflated_root_scalar_saturated_residual():
    """A residual that is constant away from its root doesn't look like a root there.

    Where the function is flat its derivative is zero, which leaves the Newton step
    without a direction, and a step of zero would satisfy the step size test. This is
    not a corner case for ambipolar transport: a residual normalized by a sum of
    magnitudes is exactly plus or minus one wherever its terms share a sign.
    """
    ftol = 1e-4

    def fun(x, nevals):
        return jnp.tanh(40 * (x - 0.3)), nevals + 1

    xs, (_, success, fs, *_) = deflated_root_scalar(
        fun,
        jnp.array(0.0),
        2,
        args=jnp.array(0),
        bounds=(-1.0, 1.0),
        ftol=jnp.array(ftol),
        xatol=jnp.array(0.0),
        xrtol=jnp.array(1e-8),
        full_output=True,
    )
    assert success.sum() == 1
    np.testing.assert_allclose(xs[success], [0.3], atol=1e-6)
    assert np.all(np.abs(fs[success]) < ftol)


def test_deflated_root_scalar_wide_bounds():
    """Roots that are close together compared to the bounds are found from the middle.

    With bounds far wider than the range where the residual changes sign, the pair of
    roots on one side of the first has no sign change pointing at it, and only samples
    that land between them reveal it. The residual is a solved flux scan continued out
    to the bounds, where it is large and no longer monotonic.
    """
    ftol = 1e-4
    name, xcol, fluxcols, charges = _SCANS["w7x"]
    data = np.loadtxt(_DATA / name)
    x = data[:, xcol]
    current = (data[:, fluxcols] * np.array(charges)).sum(axis=1)
    f = current / np.abs(data[:, fluxcols]).sum(axis=1)[np.argmin(np.abs(x))]
    far = np.array(
        [
            (-0.1, -1.2),
            (-0.085, -3.3),
            (-0.05, -2.3),
            (-0.017, -1.9),
            (0.016, 2.1),
            (0.075, 3.3),
            (0.1, 1.2),
        ]
    )
    xs = np.concatenate([x, far[:, 0]])
    fs = np.concatenate([f, far[:, 1]])
    order = np.argsort(xs)
    xs, fs = jnp.array(xs[order]), jnp.array(fs[order])

    def fun(y, nevals):
        return interpax.interp1d(y, xs, fs, method="monotonic"), nevals + 1

    lo, hi = -0.1, 0.1
    expected, slope = _sweep_roots(fun, lo, hi, num=40001)
    assert len(expected) == 3
    xr, (_, success, fs_root, *_) = deflated_root_scalar(
        fun,
        jnp.array(0.0),
        3,
        args=jnp.array(0),
        bounds=(lo, hi),
        ftol=jnp.array(ftol),
        xatol=jnp.array(0.0),
        xrtol=jnp.array(1e-8),
        full_output=True,
    )
    assert success.sum() == len(expected)
    atol = 2 * ftol / np.abs(slope).min()
    np.testing.assert_allclose(np.sort(xr[success]), expected, rtol=0, atol=atol)
    assert np.all(np.abs(fs_root[success]) < ftol)
