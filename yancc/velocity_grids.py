"""Velocity grids for yancc."""

import equinox as eqx
import jax
import jax.numpy as jnp
import orthax


class SpeedGrid(eqx.Module):
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
    k: int
    xmax: float
    xrec: orthax.recurrence.AbstractRecurrenceRelation
    x: jax.Array
    wx: jax.Array
    xvander: jax.Array
    xvander_inv: jax.Array
    Dx: jax.Array
    Dx_pseudospectral: jax.Array

    def __init__(self, nx, k=0, xmax=jnp.inf):
        self.nx = nx
        self.k = k
        self.xmax = xmax
        self.xrec = orthax.recurrence.generate_recurrence(
            weight=lambda x: x**self.k * jnp.exp(-(x**2)),
            domain=(0, self.xmax),
            n=nx + 1,
        )
        self.x, self.wx = orthax.orthgauss(nx, self.xrec)
        self.xvander = orthax.orthvander(
            self.x, self.nx - 1, self.xrec
        ) * self.xrec.weight(self.x[:, None])
        self.xvander_inv = jnp.linalg.pinv(self.xvander)

        def _dxfun(c):
            dc = orthax.orthder(c, self.xrec)
            dc = jnp.append(dc, jnp.array([0.0]))
            cc = 2 * orthax.orthmulx(c, self.xrec, mode="same")
            return dc - cc

        self.Dx = jax.jacfwd(_dxfun)(self.x)
        self.Dx_pseudospectral = self.xvander @ self.Dx @ self.xvander_inv

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


class PitchAngleGrid(eqx.Module):
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
        kk = jnp.diag(k * (k + 1))
        self.L = self.xivander @ kk @ self.xivander_inv
