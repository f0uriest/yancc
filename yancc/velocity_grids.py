"""Velocity grids for yancc."""

import equinox as eqx
import jax
import jax.numpy as jnp
import orthax
from jax import config

# need this here as well so that const default_xrec uses 64 bit
config.update("jax_enable_x64", True)


# save some time by precomputing this, usually only need nx <= 5-8
def _default_weight(x: jax.Array):
    return jnp.exp(-(x**2))


default_xrec = orthax.recurrence.TabulatedRecurrenceRelation(
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


class AbstractSpeedGrid(eqx.Module):
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


class MonoenergeticSpeedGrid(AbstractSpeedGrid):
    """Speed grid for monoenergetic problem, ie single speed.

    Parameters
    ----------
    x : float
        Normalized speed being considered.
    """

    def __init__(self, x: jax.Array):
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


class MaxwellSpeedGrid(AbstractSpeedGrid):
    """Grid for speed variable x=v/vth.

    Uses Maxwell Polynomials, which are orthogonal on (0, xmax) with the weight
    function x^k exp(-x^2)

    Parameters
    ----------
    nx : int
        Number of grid points.
    k : int, optional
        Power of x in weight function
    xmax : float, optional
        Upper bound for orthogonality inner product.

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

    def __init__(self, nx, **kwargs):
        self.nx = nx
        if nx < 20:
            self.xrec = default_xrec
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

        gauge_idx = kwargs.get("gauge_idx", None)
        if gauge_idx is None:
            gauge_idx = jnp.atleast_1d(jnp.argsort(jnp.abs(x - 1))[:2])
            if self.nx == 1:
                gauge_idx = gauge_idx[0]
        self.gauge_idx = jnp.sort(gauge_idx)

    def _dfdx(self, f):
        # this only knows about a single species,
        # f assumed to be shape(xi, x, theta, zeta)
        return jnp.einsum("ax,ixtz->iatz", self.Dx_pseudospectral, f)

    def _interp(self, x, f, xq, weight=True):
        # f assumed to be shape(xi, x, theta, zeta)
        M = orthax.orthvander(x, len(x) - 1, self.xrec)
        if weight:
            M *= jnp.sqrt(self.xrec.weight(x[:, None]))
        f = jnp.moveaxis(f, 1, 0)
        shp = f.shape
        c = jnp.linalg.lstsq(M, f.reshape((self.nx, -1)))[0].reshape(shp)
        fq = orthax.orthval(xq, c, self.xrec)
        if weight:
            fq *= jnp.sqrt(self.xrec.weight(xq))
        # fq now of shape (xi, theta, zeta, x)
        return jnp.moveaxis(fq, -1, 1)

    def _integral(self, f):
        # f assumed to be shape(xi, x, theta, zeta)
        return (f * self.wx[None, :, None, None]).sum(axis=1)


class LegendrePitchAngleGrid(eqx.Module):
    """Grid for pitch angle variable xi=v||/v.

    Uses Legendre Polynomials, which are orthogonal on (-1, 1) with the weight
    function 1.

    Parameters
    ----------
    nxi : int
        Number of grid points.

    """

    nxi: int = eqx.field(static=True)
    xirec: orthax.recurrence.AbstractRecurrenceRelation
    xi: jax.Array
    wxi: jax.Array
    xivander: jax.Array
    xivander_inv: jax.Array
    Dxi: jax.Array
    Dxi_pseudospectral: jax.Array
    L: jax.Array

    def __init__(self, nxi):
        self.nxi = nxi
        self.xirec = orthax.recurrence.Legendre()
        self.xi, self.wxi = orthax.orthgauss(nxi, self.xirec)
        self.xivander = orthax.orthvander(self.xi, self.nxi - 1, self.xirec)
        self.xivander_inv = jnp.linalg.pinv(self.xivander)

        def _dxifun(c):
            c = jnp.append(c, jnp.array([0.0]))
            dc = orthax.orthder(c, self.xirec)
            return dc

        self.Dxi = jax.jacfwd(_dxifun)(self.xi)
        self.Dxi_pseudospectral = self.xivander @ self.Dxi @ self.xivander_inv
        k = jnp.arange(self.nxi)
        kk = -jnp.diag(k * (k + 1))
        # pitch angle scattering operator ~ -k(k+1)
        self.L = self.xivander @ kk @ self.xivander_inv

    def resample(self, nxi):
        """Resample grid to a lower or higher resolution."""
        return self.__class__(nxi)


class UniformPitchAngleGrid(eqx.Module):
    """Grid for pitch angle variable gamma= arccos(v||/v).

    Uniform grid not including endpoints.

    Parameters
    ----------
    nxi : int
        Number of grid points.

    """

    nxi: int = eqx.field(static=True)
    gamma: jax.Array
    xi: jax.Array
    wxi: jax.Array

    def __init__(self, nxi):
        nxi = eqx.error_if(nxi, nxi % 2 == 0, "nxi must be odd")
        self.nxi = nxi
        gamma = jnp.linspace(0, jnp.pi, nxi, endpoint=False)
        gamma += jnp.pi / (2 * nxi)
        self.gamma = gamma
        self.xi = -jnp.cos(gamma)

        # fejer type 1 quadrature
        length = nxi // 2
        r = nxi - length

        kappa = jnp.arange(r)
        beta = jnp.hstack(
            [
                2 * jnp.exp(1j * jnp.pi * kappa / nxi) / (1 - 4 * kappa**2),
                jnp.zeros(length + 1),
            ]
        )
        beta = beta[:-1] + jnp.conjugate(beta[:0:-1])

        wxi = jnp.fft.ifft(beta)
        self.wxi = wxi.real

    def resample(self, nxi):
        """Resample grid to a lower or higher resolution."""
        return self.__class__(nxi)
