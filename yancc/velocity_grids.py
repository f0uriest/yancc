"""Velocity grids for yancc."""

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import orthax
import quadax
from jax import config
from jaxtyping import ArrayLike, Float

from .field import Field
from .species import (
    LocalMaxwellian,
    _normalize_collisionality,
    _nustar,
    _poloidal_mach,
    _poloidal_mach_hat,
)

# need this here as well so that const default_xrec uses 64 bit
config.update("jax_enable_x64", True)


# save some time by precomputing this, usually only need nx <= 5-8
def _default_weight(x: jax.Array):
    return jnp.exp(-(x**2))


_default_xrec = orthax.recurrence.TabulatedRecurrenceRelation(
    weight=_default_weight,
    domain=(0, jnp.inf),
    ak=jnp.array(
        [
            5.6418958354775617e-01,
            9.8842539284680075e-01,
            1.2859676193639398e00,
            1.5247208440801154e00,
            1.7301922743094393e00,
            1.9134998431431025e00,
            2.0806203364008327e00,
            2.2352283805046405e00,
            2.3797824435046384e00,
            2.5160256434438635e00,
            2.6452479250569576e00,
            2.7684359535042509e00,
            2.8863645940326998e00,
            2.9996556533536003e00,
            3.1088171759249232e00,
            3.2142706360711242e00,
            3.3163702970830897e00,
            3.4154173324133374e00,
            3.5116703446156343e00,
            3.6053533459055607e00,
            3.6966619115045933e00,
        ]
    ),
    bk=jnp.array(
        [
            8.8622692545275805e-01,
            1.8169011381620928e-01,
            3.4132512895943917e-01,
            5.0496215298800162e-01,
            6.7026419463961828e-01,
            8.3617049928031195e-01,
            1.0023478510110115e00,
            1.1686711647442711e00,
            1.3350829222423350e00,
            1.5015525993447623e00,
            1.6680623621881157e00,
            1.8346010527937686e00,
            2.0011613185512132e00,
            2.1677381117632635e00,
            2.3343278495405015e00,
            2.5009279171337027e00,
            2.6675363609572034e00,
            2.8341516916678309e00,
            3.0007727537827202e00,
            3.1673986369644243e00,
            3.3340286142031150e00,
        ]
    ),
    gk=jnp.array(
        [
            9.4139626377671481e-01,
            4.0127131837760449e-01,
            2.3443489208677928e-01,
            1.6659104971719907e-01,
            1.3638753329773556e-01,
            1.2471597763037882e-01,
            1.2486229906258491e-01,
            1.3498250749713830e-01,
            1.5596660252146680e-01,
            1.9111812978996492e-01,
            2.4683573200285880e-01,
            3.3433269390222509e-01,
            4.7295508316917689e-01,
            6.9634314409069387e-01,
            1.0639083802042895e00,
            1.6824990086304130e00,
            2.7479573693766617e00,
            4.6261694527525501e00,
            8.0137924519792083e00,
            1.4262296252814270e01,
            2.6041986814437465e01,
        ]
    ),
    mk=jnp.array(
        [
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
            1.0000000000000000e00,
        ]
    ),
)


class _AbstractSpeedGrid(eqx.Module):
    """Abstract base class for speed grids."""

    nx: int = eqx.field(static=True)
    x: jax.Array
    wx: jax.Array
    xvander: jax.Array
    xvander_inv: jax.Array
    Dx: jax.Array
    Dx_pseudospectral: jax.Array
    D2x_pseudospectral: jax.Array
    gauge_idx: jax.Array


class _MonoenergeticSpeedGrid(_AbstractSpeedGrid):
    """Speed grid for monoenergetic problem, ie single speed.

    Parameters
    ----------
    x : float
        Normalized speed being considered.
    """

    def __init__(self, x: ArrayLike):
        x = jnp.asarray(x)
        assert x.size == 1
        self.nx = 1
        self.x = x
        self.wx = jnp.ones(1)
        self.xvander = jnp.ones((1, 1))
        self.xvander_inv = jnp.ones((1, 1))
        self.Dx = jnp.zeros((1, 1))
        self.Dx_pseudospectral = jnp.zeros((1, 1))
        self.D2x_pseudospectral = jnp.zeros((1, 1))
        self.gauge_idx = jnp.array([0])


class MaxwellSpeedGrid(_AbstractSpeedGrid):
    r"""Grid for speed variable :math:`x = v/v_{th}`.

    Uses Maxwell Polynomials, which are orthogonal on :math:`[0, \infty)` with the
    weight function :math:`\exp(-x^2)`

    Parameters
    ----------
    nx : int
        Number of grid points.

    """

    nx: int = eqx.field(static=True)
    xrec: orthax.recurrence.AbstractRecurrenceRelation
    x: jax.Array
    wx: jax.Array
    xvander: jax.Array
    xvander_inv: jax.Array
    Dx: jax.Array
    Dx_pseudospectral: jax.Array
    D2x: jax.Array
    D2x_pseudospectral: jax.Array
    gauge_idx: jax.Array

    def __init__(self, nx: int):
        assert nx >= 2, "MaxwellSpeedGrid requires nx >= 2"
        self.nx = nx
        if nx < 20:
            self.xrec = _default_xrec
        else:
            self.xrec = orthax.recurrence.generate_recurrence(
                weight=_default_weight,
                domain=(0, jnp.inf),
                n=nx + 1,
            )
        # f at collocation points includes weight function
        # ie we store f(x) = weight(x)*poly(x) not f(x) = poly(x)
        x, wx = orthax.orthgauss(nx, self.xrec)
        self.x = x
        self.wx = wx / self.xrec.weight(x)

        # modal basis is weight(x)*poly(x). orthvander gives polynomial part,
        # need to multiply by weight to get correct basis fn.
        self.xvander = orthax.orthvander(
            self.x, self.nx - 1, self.xrec
        ) * self.xrec.weight(self.x[:, None])
        self.xvander_inv = jnp.linalg.pinv(self.xvander)

        # d/dx p(x) w(x) = p'(x) w(x) + p(x) w'(x) = D p(x) w(x) + p(x) w'(x)
        # w'(x) = k x^(k-1) exp(-x^2) - 2x^(k+1) exp(-x^2) = (k/x - 2x) w(x)
        # d/dx p(x) w(x) = D p(x) w(x) + (k/x - 2x) w(x) p(x) = (D + k/x - 2x) p(x) w(x)
        def _dxfun(c):
            dc = orthax.orthder(c, self.xrec)
            dc = jnp.append(dc, jnp.array([0.0]))
            cc = 2 * orthax.orthmulx(c, self.xrec, mode="same")
            # leave off k/x term for now since its not polynomial unless k==0
            return dc - cc

        # d2/dx2 p(x) w(x) = d/dx (p'(x) w(x) + p(x) w'(x)) = p'' w + 2 p' w' + p w''
        # w'(x) = k x^(k-1) exp(-x^2) - 2x^(k+1) exp(-x^2) = (k/x - 2x) w(x)
        # w'' = (-k/x^2 - 2) w + (k/x - 2x) w'
        #     = (-k/x^2 - 2) w + (k/x - 2x) (k/x - 2x) w(x)
        #     = (-k/x^2 - 2 + k^2/x^2 - 2k - 2k + 4x^2) w
        #     = ((k^2-k)/x^2 + 4x^2 - 4k - 2) w
        # p'' = D^2 p
        # d2/dx2 p w = [D^2 + 2 D (k/x - 2x) + ((k^2-k)/x^2 + 4x^2 - 4k - 2)] p w
        # when k=0, = [D^2 - 4 D x + 4x^2 - 2)] p w

        def _d2xfun(p):
            D = orthax.orthder(p, self.xrec)
            Dx = orthax.orthmulx(D, self.xrec, mode="full")
            D2 = orthax.orthder(p, self.xrec, m=2)
            D2 = jnp.append(D2, jnp.array([0.0, 0.0]))
            x2 = orthax.orthmulx(
                orthax.orthmulx(p, self.xrec, mode="same"), self.xrec, mode="same"
            )
            # leave off k/x term for now since its not polynomial unless k==0
            return D2 - 4 * Dx + 4 * x2 - 2 * p

        self.Dx = jax.jacfwd(_dxfun)(self.x)
        self.Dx_pseudospectral = self.xvander @ self.Dx @ self.xvander_inv

        self.D2x = jax.jacfwd(_d2xfun)(self.x)
        self.D2x_pseudospectral = self.xvander @ self.D2x @ self.xvander_inv

        # the equations at these points are replaced to fix the gauge freedom in
        # density and energy
        self.gauge_idx = jnp.sort(jnp.argsort(jnp.abs(x - 1))[:2])

    def resample(self, nx: int) -> "MaxwellSpeedGrid":
        """Resample grid to a lower or higher resolution."""
        return self.__class__(nx)


class AbstractPitchAngleGrid(ABC, eqx.Module):
    """Base class for pitch angle coordinate grids."""

    nalpha: int = eqx.field(static=True)
    alpha: jax.Array
    xi: jax.Array
    wxi: jax.Array

    @abstractmethod
    def resample(self, nalpha: int) -> "AbstractPitchAngleGrid":
        """Resample grid to a lower or higher resolution."""


class _LegendrePitchAngleGrid(AbstractPitchAngleGrid):
    r"""Grid for pitch angle variable :math:`\xi = v_{||} / v`.

    Uses Legendre Polynomials, which are orthogonal on (-1, 1) with the weight
    function 1.

    Parameters
    ----------
    nalpha : int
        Number of grid points.

    """

    xirec: orthax.recurrence.AbstractRecurrenceRelation

    def __init__(self, nalpha: int):
        self.nalpha = nalpha
        self.xirec = orthax.recurrence.Legendre()
        self.xi, self.wxi = orthax.orthgauss(nalpha, self.xirec)
        self.alpha = -jnp.acos(self.xi)

    def resample(self, nalpha: int) -> "_LegendrePitchAngleGrid":
        """Resample grid to a lower or higher resolution."""
        return self.__class__(nalpha)


class _DensityMap(eqx.Module):
    """Maps uniformly spaced alpha in [0, pi] to nodes equidistributing a density.

    density(xi, *params) is a static (non-traced) callable giving the relative number
    of nodes per unit alpha at pitch xi = -cos(alpha). params holds its dynamic
    (traceable) arguments.
    """

    density: Callable = eqx.field(static=True)
    params: tuple = ()

    @eqx.filter_jit
    def __call__(self, alpha):
        # Invert the normalized cumulative density, tabulated on a fine alpha grid and
        # linearly interpolated. The table is fine enough that interpolation error is
        # negligible next to the discretization error of any practical nalpha.
        table = jnp.linspace(0, jnp.pi, 16001)
        rho = jnp.broadcast_to(self.density(-jnp.cos(table), *self.params), table.shape)
        rho = eqx.error_if(rho, jnp.any(rho <= 0), "density must be positive")
        cdf = quadax.cumulative_trapezoid(rho, x=table, initial=0.0)
        return jnp.interp(alpha, cdf * jnp.pi / cdf[-1], table)


class NonUniformPitchAngleGrid(AbstractPitchAngleGrid):
    r"""User controlled non-uniform grid in pitch angle coordinate α = arccos(-v||/v).

    Nodes are placed so that their density in :math:`\alpha` follows a user supplied
    function of :math:`\xi = v_{||}/v`. A constant density gives nodes uniformly
    spaced in :math:`\alpha`, the same as :class:`UniformPitchAngleGrid`.

    Parameters
    ----------
    nalpha : int
        Number of points in pitch angle coordinate.
    density : callable
        Smooth, positive function ``density(xi, *params)`` giving the relative number
        of nodes per unit :math:`\alpha` at pitch :math:`\xi \in [-1, 1]`. Does not
        need to be normalized.
    params : tuple of jax.Array, optional
        Additional arguments to ``density``.
    """

    density_map: _DensityMap

    def __init__(self, nalpha: int, density: Callable, params: tuple = ()):
        self.nalpha = nalpha
        alpha = jnp.linspace(0, jnp.pi, nalpha, endpoint=False) + jnp.pi / (2 * nalpha)
        self.density_map = _DensityMap(density, tuple(params))

        self.alpha = self.density_map(alpha)
        self.xi = -jnp.cos(self.alpha)
        self.wxi = _composite_newton_cotes_weights(self.xi, 4, (-1, 1))

    def resample(self, nalpha: int) -> "NonUniformPitchAngleGrid":
        """Resample grid to a lower or higher resolution."""
        return NonUniformPitchAngleGrid(
            nalpha, self.density_map.density, self.density_map.params
        )


def _uniform_density(xi):
    return jnp.ones_like(xi)


class UniformPitchAngleGrid(NonUniformPitchAngleGrid):
    r"""Grid for pitch angle variable :math:`\alpha = \arccos(-v_{||}/v)`.

    Uniform grid on :math:`[0, \pi]`, not including endpoints.

    Parameters
    ----------
    nalpha : int
        Number of grid points.

    """

    def __init__(self, nalpha: int):
        self.nalpha = nalpha
        self.density_map = _DensityMap(_uniform_density)
        self.alpha = jnp.linspace(0, jnp.pi, nalpha, endpoint=False) + jnp.pi / (
            2 * nalpha
        )
        self.xi = -jnp.cos(self.alpha)
        # uniform in a means chebyshev nodes in xi, so we can do better than
        # newton-cotes: fejer type 1 quadrature
        self.wxi = _fejer_type_1_weights(nalpha)

    def resample(self, nalpha: int) -> "UniformPitchAngleGrid":
        """Resample grid to a lower or higher resolution."""
        return self.__class__(nalpha)


# Cluster sharpness: a = coef * nustar^(-1/3) is the coefficient of a Gaussian cluster
# exp(-a xi^2) that narrows with the collisional boundary layer. Both the exponent and
# the coefficients are empirical fits to the pitch resolution error of the transport
# coefficients on resolved flux surfaces. The optimum scatters by tens of percent
# between fields, so the coefficients are kept to one significant figure.
_MDKE_CLUSTER_COEF = 0.2
# The DKE optimum is sharper than the MDKE one for the fluxes and flows.
_DKE_CLUSTER_COEF = 0.3
# Poloidal Mach number at which a cluster has relaxed by a factor e toward uniform.
_CLUSTER_MACH = 0.3
# Normalized speeds x = v/v_th whose MDKE clusters set the DKE grid: the slower one
# keeps resolution toward xi = +-1 for parallel flows, the faster one resolves the
# narrow layer that dominates the fluxes.
_DKE_CLUSTER_SPEEDS = (1.0, 2.5)


def _cluster_density(xi, a, center, weight, floor):
    # rho(xi) = floor + sum_k weight_k exp(-a_k (xi - center_k)^2)
    return floor + jnp.sum(
        weight[:, None] * jnp.exp(-a[:, None] * (xi - center[:, None]) ** 2),
        axis=0,
    )


def _cluster_sharpness(coef, nustar, mach):
    # The boundary layer narrows with collisionality, while a strong radial electric
    # field mostly widens the region of pitch where the solution has fine structure
    # (it moves the region very little), so the cluster relaxes toward uniform as the
    # poloidal Mach number grows.
    # The exponent is fitted, not derived: the best cluster width in xi scales like
    # nustar^(1/6), much more weakly than the nu^(1/2) width of the layer itself.
    return coef * nustar ** (-1 / 3) * jnp.exp(-((mach / _CLUSTER_MACH) ** 2))


class MDKEGaussianPitchAngleGrid(NonUniformPitchAngleGrid):
    r"""MDKE pitch angle grid clustered in the collisional boundary layer.

    Designed for the monoenergetic DKE (:func:`~yancc.solve.solve_mdke`), where the
    collisionality and electric field are single numbers that set the layer.

    Nodes are distributed in :math:`\alpha = \arccos(-\xi)` with density

    .. math::

        \rho(\xi) = \rho_0 + \exp\left(-a (\xi - \xi_s)^2\right),
        \qquad a = 0.2 \, \nu_*^{-1/3} \exp\left(-(M_p / 0.3)^2\right),

    where :math:`\nu_* = \hat{\nu} R / |\iota|`. The cluster narrows as the
    collisionality decreases, following the width of the boundary layer near the
    trapped/passing transition. The floor
    :math:`\rho_0` keeps a minimum fraction of nodes away from the cluster, which
    bounds the ratio of largest to smallest node spacing at very low collisionality.

    A strong radial electric field spreads the fine structure of the solution over a
    wider range of pitch, so the cluster relaxes toward a uniform grid as the poloidal
    Mach number :math:`M_p = \omega_E / \omega_{transit}` grows, where
    :math:`\omega_E = E_r / (\langle B \rangle a)` and
    :math:`\omega_{transit} = v \langle \mathbf{b} \cdot \nabla \theta \rangle`.

    Parameters
    ----------
    nalpha : int
        Number of points in pitch angle coordinate.
    field : Field
        Magnetic field information.
    erhohat : float
        Monoenergetic electric field, as used in :func:`~yancc.solve.solve_mdke`.
    nuhat : float
        Normalized collisionality, as used in :func:`~yancc.solve.solve_mdke`.
    floor : float
        Minimum relative node density :math:`\rho_0` away from the cluster. Must be
        positive.
    xi_s : float
        Value of :math:`\xi = v_{||}/v` at the center of the cluster.
    """

    nuhat: jax.Array
    xi_s: jax.Array
    floor: jax.Array

    def __init__(
        self,
        nalpha: int,
        field: Field,
        erhohat: Float[ArrayLike, ""],
        nuhat: Float[ArrayLike, ""],
        floor: Float[ArrayLike, ""] = 0.2,
        xi_s: Float[ArrayLike, ""] = 0.0,
    ):
        nuhat = jnp.asarray(nuhat)
        nuhat = eqx.error_if(nuhat, nuhat <= 0, "nuhat must be positive")
        floor = jnp.asarray(floor)
        floor = eqx.error_if(floor, floor <= 0, "floor must be positive")
        xi_s = jnp.asarray(xi_s)
        xi_s = eqx.error_if(xi_s, jnp.abs(xi_s) >= 1, "xi_s must be in (-1, 1)")
        # error_if loses the static type, so reassert it (all are arrays post-asarray)
        self.nalpha = nalpha
        self.nuhat = cast(jax.Array, nuhat)
        self.xi_s = cast(jax.Array, xi_s)
        self.floor = cast(jax.Array, floor)
        mach = _poloidal_mach_hat(field, erhohat)
        # nuhat = nu / v, so nustar follows with v = 1
        nustar = _normalize_collisionality(self.nuhat, 1.0, field)
        a = _cluster_sharpness(_MDKE_CLUSTER_COEF, nustar, mach)
        alpha = jnp.linspace(0, jnp.pi, nalpha, endpoint=False) + jnp.pi / (2 * nalpha)
        # parameters are dynamic args of the map rather than baked into a static
        # callable, so the grid can be traced and differentiated through jit.
        self.density_map = _DensityMap(
            _cluster_density, (a[None], self.xi_s[None], jnp.ones(1), self.floor)
        )

        self.alpha = self.density_map(alpha)
        self.xi = -jnp.cos(self.alpha)
        self.wxi = _composite_newton_cotes_weights(self.xi, 4, (-1, 1))


class DKEGaussianPitchAngleGrid(NonUniformPitchAngleGrid):
    r"""DKE pitch angle grid clustered in the collisional layers of all species.

    Designed for :func:`~yancc.solve.solve_dke`, where every species and speed sees a
    different collisionality and electric field but all share one pitch angle grid.

    Nodes are distributed in :math:`\alpha = \arccos(-\xi)` with density

    .. math::

        \rho(\xi) = \rho_0 + \sum_s \frac{n_s}{\sum_{s'} n_{s'}}
        \frac{1}{2} \sum_{x_k \in \{1, 2.5\}} \exp\left(-a_{s,k} \xi^2\right),

    where :math:`a_{s,k} = 0.3 \, \nu_{*,s,k}^{-1/3}
    \exp\left(-(M_{p,s,k} / 0.3)^2\right)`, with the normalized collisionality
    :math:`\nu_* = \nu R / (v |\iota|)` and poloidal Mach number of species :math:`s`
    at normalized speed :math:`x_k = v / v_{th,s}` (the cluster of
    :class:`MDKEGaussianPitchAngleGrid`, somewhat sharper). The slower
    speed keeps resolution toward :math:`|\xi| = 1`, needed for parallel flows; the
    faster one resolves the narrow layer that dominates the radial fluxes. Weighting
    species by density keeps trace impurities, whose own layers are much wider, from
    spreading the grid.

    Parameters
    ----------
    nalpha : int
        Number of points in pitch angle coordinate.
    field : Field
        Magnetic field information.
    species : list[LocalMaxwellian]
        Species information, as passed to :func:`~yancc.solve.solve_dke`.
    Erho : float
        Radial electric field, Erho = -∂Φ /∂ρ, in Volts.
    background : list[LocalMaxwellian], optional
        Additional background species included in the collision operator, as passed
        to :func:`~yancc.solve.solve_dke`.
    floor : float
        Minimum relative node density :math:`\rho_0` away from the clusters. Must be
        positive.
    """

    floor: jax.Array

    def __init__(
        self,
        nalpha: int,
        field: Field,
        species: list[LocalMaxwellian],
        Erho: Float[ArrayLike, ""],
        background: list[LocalMaxwellian] | None = None,
        floor: Float[ArrayLike, ""] = 0.2,
    ):
        background = [] if background is None else background
        floor = jnp.asarray(floor)
        floor = eqx.error_if(floor, floor <= 0, "floor must be positive")
        self.nalpha = nalpha
        self.floor = cast(jax.Array, floor)

        density = jnp.array([s.density for s in species])
        a, weight = [], []
        for i, spec in enumerate(species):
            others = species[:i] + species[i + 1 :] + background
            for x in _DKE_CLUSTER_SPEEDS:
                nustar = _nustar(spec, field, x, *others)
                mach = _poloidal_mach(spec, field, Erho, x)
                a.append(_cluster_sharpness(_DKE_CLUSTER_COEF, nustar, mach))
                weight.append(density[i] / density.sum() / len(_DKE_CLUSTER_SPEEDS))
        a = jnp.array(a)
        alpha = jnp.linspace(0, jnp.pi, nalpha, endpoint=False) + jnp.pi / (2 * nalpha)
        self.density_map = _DensityMap(
            _cluster_density, (a, jnp.zeros_like(a), jnp.array(weight), self.floor)
        )

        self.alpha = self.density_map(alpha)
        self.xi = -jnp.cos(self.alpha)
        self.wxi = _composite_newton_cotes_weights(self.xi, 4, (-1, 1))


@functools.partial(jax.jit, static_argnames="order")
def _composite_newton_cotes_weights(
    x: jax.Array, order: int, global_limits: tuple | None = None
):
    """Computes composite quadrature weights.

    Parameters
    ----------
    x : jax.Array
        Sorted sample points. May be non-uniform.
    order : int
        Formal order of accuracy desired.
    global_limits : tuple of floats
        Limits for integration. If not given assumed to be x[0] and x[-1]

    Returns
    -------
    w : jax.Array
        Quadrature weights. All weights are non-negative, and symmetric for nodes
        symmetric about the center of the integration limits.
    """
    N = x.shape[0]
    if global_limits is not None:
        a, b = global_limits
    else:
        a, b = x[0], x[-1]

    # Each interval between neighboring nodes (and the gaps from the limits to the
    # end nodes) is integrated with the interpolant through the `order + 1` nodes
    # nearest to it. Overlapping stencils keep every interval inside its own stencil, so
    # interior intervals never extrapolate; disjoint panels on strongly non-uniform
    # nodes extrapolate across the gaps between panels and produce large weights of
    # both signs. Only the end gaps extrapolate.
    left = jnp.concatenate([jnp.atleast_1d(a), x])
    right = jnp.concatenate([x, jnp.atleast_1d(b)])
    length = right - left
    center = (left + right) / 2
    k = jnp.arange(N + 1)
    max_npts = min(order + 1, N)

    # The weights are the integrals of the Lagrange basis polynomials through the used
    # points; unused points are left out of every product and get zero weight.
    #
    # Interval k lies between nodes k-1 and k, so an odd number of points can't be
    # centered on it; averaging the two stencils shifted one node either way keeps
    # the rule exact to the same degree and makes the weights of a symmetric grid
    # symmetric.
    i = jnp.arange(max_npts)
    npts = i[:, None, None] + 1
    shifts = jnp.stack([npts // 2, (npts + 1) // 2], axis=1)[..., 0]
    start = jnp.clip(k - shifts, 0, N - npts)
    used = i < npts[..., None]
    idx = jnp.minimum(start[..., None] + i, N - 1)
    # local scaled coordinate keeps the polynomial coefficients well scaled
    h = jnp.maximum((x[start + npts - 1] - x[start]) / 2, jnp.abs(length))
    t = (x[idx] - center[:, None]) / h[..., None]
    tlo, thi = (left - center) / h, (right - center) / h
    # root[..., m, j]: t[j] is a root of the m-th basis polynomial
    root = (i[:, None] != i[None, :]) & used[..., None, :]
    # ascending monomial coefficients of prod_j (s - t[j]) over the roots of each
    # basis polynomial. Its degree is at most max_npts - 1, so multiplying by s never
    # drops a nonzero coefficient.
    coef = jnp.broadcast_to((i == 0).astype(x.dtype), (*t.shape, max_npts))
    for j in range(max_npts):
        times_s = jnp.pad(coef[..., :-1], [(0, 0)] * (coef.ndim - 1) + [(1, 0)])
        coef = jnp.where(
            root[..., j, None], times_s - t[..., j, None, None] * coef, coef
        )
    denom = jnp.prod(jnp.where(root, t[..., :, None] - t[..., None, :], 1), axis=-1)
    moments = (
        h[..., None] * (thi[..., None] ** (i + 1) - tlo[..., None] ** (i + 1)) / (i + 1)
    )
    w = jnp.where(used, jnp.einsum("...mp,...p->...m", coef, moments) / denom, 0)
    # (npts, shift, interval, point) -> (npts * interval, shift * point), so a rule is
    # looked up with a single flat index (npts - 1) * (N + 1) + interval
    rules_idx = idx.transpose(0, 2, 1, 3).reshape(max_npts * (N + 1), -1)
    rules_w = w.transpose(0, 2, 1, 3).reshape(max_npts * (N + 1), -1) / 2

    def select(npts, rules_w):
        rule = (npts - 1) * (N + 1) + k
        return (
            rules_idx.at[rule].get(mode="promise_in_bounds"),
            rules_w.at[rule].get(mode="promise_in_bounds"),
        )

    def assemble(idx, w):
        return (
            jnp.zeros(N, dtype=x.dtype)
            .at[idx.ravel()]
            .add(w.ravel(), mode="promise_in_bounds")
        )

    # Local rules of degree >= 2 have negative weights even on uniform grids, and on
    # stretched grids neighboring intervals no longer cancel them, so some summed
    # weights can come out negative. Starting from full degree everywhere, while any
    # node has a negative weight, lower the degree of each interval that contributes a
    # negative local weight to it. A negative summed weight always has a negative
    # contribution, so the loop stops exactly when all weights are non-negative.
    # Non-negative weights summing to b - a also have the smallest possible sum of
    # absolute weights, so this is the only stability safeguard needed, including for
    # the extrapolating end gaps. This terminates: two-point rules on interior
    # intervals and one-point rules on the end gaps have only positive weights. The
    # degree choice is discrete, so it is made on constant weights and the final
    # weights stay differentiable in x.
    frozen = jax.lax.stop_gradient(rules_w)

    def lower_degree(carry):
        npts, _ = carry
        idx, w = select(npts, frozen)
        weights = assemble(idx, w)
        bad = jnp.any(
            (weights.at[idx].get(mode="promise_in_bounds") < 0) & (w < 0), axis=1
        )
        return jnp.where(bad, npts - 1, npts), jnp.any(bad)

    npts, _ = jax.lax.while_loop(
        lambda carry: carry[1],
        lower_degree,
        (jnp.full(N + 1, max_npts), jnp.array(True)),
    )
    weights = assemble(*select(npts, rules_w))

    return weights


def _fejer_type_1_weights(n):
    """Fejer (chebyshev) type 1 quadrature."""
    length = n // 2
    r = n - length
    kappa = jnp.arange(r)
    beta = jnp.hstack(
        [
            2 * jnp.exp(1j * jnp.pi * kappa / n) / (1 - 4 * kappa**2),
            jnp.zeros(length + 1),
        ]
    )
    beta = beta[:-1] + jnp.conjugate(beta[:0:-1])
    w = jnp.fft.ifft(beta)
    return w.real
