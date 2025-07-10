"""Collision operators and methods for computing Rosenbluth potentials."""

import functools
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import orthax
import quadax
from jaxtyping import Array, ArrayLike, Bool, Float

from .field import Field
from .finite_diff import fd2, fd_coeffs, fdfwd
from .species import LocalMaxwellian, gamma_ab, nuD_ab, nupar_ab
from .utils import (
    _parse_axorder_shape_3d,
    _parse_axorder_shape_4d,
    lGammainc,
    lGammaincc,
)
from .velocity_grids import LegendrePitchAngleGrid, SpeedGrid, UniformPitchAngleGrid


class MDKEPitchAngleScattering(lx.AbstractLinearOperator):
    """Diffusion operator in xi direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    nu : float
        Normalized collisionality, nu/v
    p1 : str
        Stencil to use for first derivatives. Generally of the form "1a", "2b" etc.
        Number denotes formal order of accuracy, letter denotes degree of upwinding.
        "a" is fully upwinded, "b" and "c" if they exist are upwind biased but
        not fully.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"atz", "zat", "tza"}
        Ordering for variables in f, eg how the 3d array is flattened
    gauge : bool
        Whether to impose gauge constraint by fixing f at a single point on the surface.
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    nu: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        nu: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = True,
    ):
        assert pitchgrid.nxi > fd_coeffs[1][p1].size // 2
        assert pitchgrid.nxi > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.nu = jnp.array(nu)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))
        h = jnp.pi / self.pitchgrid.nxi

        f1 = fdfwd(f, str(self.p2) + "z", h=h, bc="symmetric", axis=0)
        f1 *= -(self.nu / 2 * cosa / sina)[:, None, None]
        f2 = fd2(f, self.p2, h=h, bc="symmetric", axis=0)
        f2 *= -self.nu / 2
        df = f1 + f2

        idx = self.pitchgrid.nxi // 2
        scale = self.nu / h**2
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale * f[idx, 0, 0]), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi

        h = jnp.pi / self.pitchgrid.nxi

        f1 = jnp.diag(
            jax.jacfwd(fdfwd)(
                f[:, 0, 0], str(self.p2) + "z", h=h, bc="symmetric", axis=0
            )
        )[:, None, None]
        f1 *= -(self.nu / 2 * cosa / sina)[:, None, None]
        f2 = jnp.diag(
            jax.jacfwd(fd2)(f[:, 0, 0], self.p2, h=h, bc="symmetric", axis=0)
        )[:, None, None]
        f2 *= -self.nu / 2
        df = f1 + f2
        df = jnp.tile(df, (1, self.field.ntheta, self.field.nzeta))

        idx = self.pitchgrid.nxi // 2
        scale = self.nu / h**2
        df = jnp.where(self.gauge, df.at[idx, 0, 0].set(scale), df)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))

        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nxi, self.axorder
        )
        f = jnp.ones((self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta))
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi

        h = jnp.pi / self.pitchgrid.nxi

        f1 = jax.jacfwd(fdfwd)(
            f[:, 0, 0], str(self.p2) + "z", h=h, bc="symmetric", axis=0
        )[:, None, None, :]
        f1 *= -(self.nu / 2 * cosa / sina)[:, None, None, None]
        f2 = jax.jacfwd(fd2)(f[:, 0, 0], self.p2, h=h, bc="symmetric", axis=0)[
            :, None, None, :
        ]
        f2 *= -self.nu / 2
        df = f1 + f2
        df = jnp.tile(df, (1, self.field.ntheta, self.field.nzeta, 1))

        idx = self.pitchgrid.nxi // 2
        scale = self.nu / h**2
        df = jnp.where(
            self.gauge, df.at[idx, 0, 0, :].set(0).at[idx, 0, 0, idx].set(scale), df
        )
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, self.pitchgrid.nxi, self.pitchgrid.nxi))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (self.field.ntheta * self.field.nzeta * self.pitchgrid.nxi,),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class RosenbluthPotentials(eqx.Module):
    """Thing to calculate Rosenbluth Potentials.

    Parameters
    ----------
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    nL : int
        Number of Legendre modes to use for potentials.
    quad : bool
        Whether to compute potentials using quadrature (slow) or incomplete gamma
        functions (fast)
    """

    speedgrid: SpeedGrid
    pitchgrid: UniformPitchAngleGrid
    legendregrid: LegendrePitchAngleGrid
    quad: bool = eqx.field(static=True)
    ddGxlk: jax.Array
    Hxlk: jax.Array
    dHxlk: jax.Array
    Txi: jax.Array
    Txi_inv: jax.Array

    def __init__(self, speedgrid, pitchgrid, species, nL=4, quad=False):
        if not quad:
            assert speedgrid.k == 0
            assert speedgrid.xmax == jnp.inf

        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.legendregrid = LegendrePitchAngleGrid(nL)
        self.quad = quad

        self.Txi = orthax.orthvander(pitchgrid.xi, nL - 1, self.legendregrid.xirec)
        self.Txi_inv = jnp.linalg.pinv(self.Txi)

        ns = len(species)
        x = self.speedgrid.x[:, None, None]
        l = jnp.arange(nL)[None, :, None]
        k = jnp.arange(self.speedgrid.nx)[None, None, :]
        self.ddGxlk = jnp.zeros(
            (ns, ns, self.speedgrid.nx, self.pitchgrid.nxi, self.speedgrid.nx)
        )
        self.dHxlk = jnp.zeros(
            (ns, ns, self.speedgrid.nx, self.pitchgrid.nxi, self.speedgrid.nx)
        )
        self.Hxlk = jnp.zeros(
            (ns, ns, self.speedgrid.nx, self.pitchgrid.nxi, self.speedgrid.nx)
        )
        # arr[a,b] is potential operator from species b to species a
        for a, spa in enumerate(species):
            for b, spb in enumerate(species):
                va, vb = spa.v_thermal, spb.v_thermal
                v = x * va  # speed on a grid
                xb = v / vb  # on b grid
                ddG = self._ddGlk(xb, l, k)
                dH = self._dHlk(xb, l, k)
                H = self._Hlk(xb, l, k)
                self.ddGxlk = self.ddGxlk.at[a, b, :, :nL, :].set(ddG)
                self.dHxlk = self.dHxlk.at[a, b, :, :nL, :].set(dH)
                self.Hxlk = self.Hxlk.at[a, b, :, :nL, :].set(H)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _Hlk(self, xb, l, k):
        term1 = 1 / xb ** (l + 1) * self._I_2(xb, l, k)
        term2 = xb**l * self._I_1(xb, l, k)
        return (4 * jnp.pi) / (2 * l + 1) * (term1 + term2)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dHlk(self, xb, l, k):
        term11 = -(l + 1) / xb ** (l + 2) * self._I_2(xb, l, k)
        term12 = 1 / xb ** (l + 1) * self._dI_2(xb, l, k)
        term21 = l * xb ** (l - 1) * self._I_1(xb, l, k)
        term22 = xb**l * self._dI_1(xb, l, k)
        return (4 * jnp.pi) / (2 * l + 1) * (term11 + term12 + term21 + term22)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _Glk(self, xb, l, k):
        term1 = xb**l * self._I_3(xb, l, k)
        term2 = -(2 * l - 1) / (2 * l + 3) * xb ** (l + 2) * self._I_1(xb, l, k)
        term3 = -(2 * l - 1) / (2 * l + 3) / xb ** (l + 1) * self._I_4(xb, l, k)
        term4 = 1 / xb ** (l - 1) * self._I_2(xb, l, k)
        return -(4 * jnp.pi) / (4 * l**2 - 1) * (term1 + term2 + term3 + term4)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dGlk(self, xb, l, k):
        term11 = l * xb ** (l - 1) * self._I_3(xb, l, k)
        term12 = xb**l * self._dI_3(xb, l, k)
        term21 = (
            -(2 * l - 1) / (2 * l + 3) * (l + 2) * xb ** (l + 1) * self._I_1(xb, l, k)
        )
        term22 = -(2 * l - 1) / (2 * l + 3) * xb ** (l + 2) * self._dI_1(xb, l, k)
        term31 = (
            (2 * l - 1) / (2 * l + 3) * (l + 1) / xb ** (l + 2) * self._I_4(xb, l, k)
        )
        term32 = -(2 * l - 1) / (2 * l + 3) / xb ** (l + 1) * self._dI_4(xb, l, k)
        term41 = -(l - 1) / xb ** (l) * self._I_2(xb, l, k)
        term42 = 1 / xb ** (l - 1) * self._dI_2(xb, l, k)
        return (
            -(4 * jnp.pi)
            / (4 * l**2 - 1)
            * (term11 + term12 + term21 + term22 + term31 + term32 + term41 + term42)
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddGlk(self, xb, l, k):
        term111 = l * (l - 1) * xb ** (l - 2) * self._I_3(xb, l, k)
        term112 = l * xb ** (l - 1) * self._dI_3(xb, l, k)
        term121 = l * xb ** (l - 1) * self._dI_3(xb, l, k)
        term122 = xb**l * self._ddI_3(xb, l, k)
        term211 = (
            -(2 * l - 1)
            / (2 * l + 3)
            * (l + 2)
            * (l + 1)
            * xb ** (l)
            * self._I_1(xb, l, k)
        )
        term212 = (
            -(2 * l - 1) / (2 * l + 3) * (l + 2) * xb ** (l + 1) * self._dI_1(xb, l, k)
        )
        term221 = (
            -(2 * l - 1) / (2 * l + 3) * (l + 2) * xb ** (l + 1) * self._dI_1(xb, l, k)
        )
        term222 = -(2 * l - 1) / (2 * l + 3) * xb ** (l + 2) * self._ddI_1(xb, l, k)
        term311 = (
            -(2 * l - 1)
            / (2 * l + 3)
            * (l + 1)
            * (l + 2)
            / xb ** (l + 3)
            * self._I_4(xb, l, k)
        )
        term312 = (
            (2 * l - 1) / (2 * l + 3) * (l + 1) / xb ** (l + 2) * self._dI_4(xb, l, k)
        )
        term321 = (
            +(2 * l - 1) / (2 * l + 3) * (l + 1) / xb ** (l + 2) * self._dI_4(xb, l, k)
        )
        term322 = -(2 * l - 1) / (2 * l + 3) / xb ** (l + 1) * self._ddI_4(xb, l, k)
        term411 = l * (l - 1) / xb ** (l + 1) * self._I_2(xb, l, k)
        term412 = -(l - 1) / xb ** (l) * self._dI_2(xb, l, k)
        term421 = -(l - 1) / xb ** (l) * self._dI_2(xb, l, k)
        term422 = 1 / xb ** (l - 1) * self._ddI_2(xb, l, k)
        return (
            -(4 * jnp.pi)
            / (4 * l**2 - 1)
            * (
                term111
                + term121
                + term211
                + term221
                + term311
                + term321
                + term411
                + term421
                + term112
                + term122
                + term212
                + term222
                + term312
                + term322
                + term412
                + term422
            )
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand1(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        return (
            z ** (-l + 1)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand2(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        return (
            z ** (l + 2)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand3(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        return (
            z ** (-l + 3)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand4(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        return (
            z ** (l + 4)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_1(self, x, l, k):
        interval = jnp.array([x, jnp.inf])
        if self.quad:
            f, info = quadax.quadcc(
                self._integrand1,
                interval,
                (l, k),
                order=256,
                max_ninter=20,
                epsabs=1e-12,
                epsrel=1e-8,
            )
            return f
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.speedgrid.xrec)
        n = jnp.arange(self.speedgrid.nx)
        sgn, lg = lGammaincc(-l / 2 + n / 2 + 1, x**2)
        li, sgn = jax.scipy.special.logsumexp(lg, b=sgn * p, return_sign=True)
        return sgn * jnp.exp(li) / 2

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_2(self, x, l, k):
        if self.quad:
            interval = jnp.array([0.0, x])
            f, info = quadax.quadcc(
                self._integrand2,
                interval,
                (l, k),
                order=256,
                max_ninter=20,
                epsabs=1e-12,
                epsrel=1e-8,
            )
            return f
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.speedgrid.xrec)
        n = jnp.arange(self.speedgrid.nx)
        sgn, lg = lGammainc(l / 2 + n / 2 + 3 / 2, x**2)
        li, sgn = jax.scipy.special.logsumexp(lg, b=sgn * p, return_sign=True)
        return sgn * jnp.exp(li) / 2

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_3(self, x, l, k):
        if self.quad:
            interval = jnp.array([x, jnp.inf])
            f, info = quadax.quadcc(
                self._integrand3,
                interval,
                (l, k),
                order=256,
                max_ninter=20,
                epsabs=1e-12,
                epsrel=1e-8,
            )
            return f
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.speedgrid.xrec)
        n = jnp.arange(self.speedgrid.nx)
        sgn, lg = lGammaincc(-l / 2 + n / 2 + 2, x**2)
        li, sgn = jax.scipy.special.logsumexp(lg, b=sgn * p, return_sign=True)
        return sgn * jnp.exp(li) / 2

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_4(self, x, l, k):
        if self.quad:
            interval = jnp.array([0.0, x])
            f, info = quadax.quadcc(
                self._integrand4,
                interval,
                (l, k),
                order=256,
                max_ninter=20,
                epsabs=1e-12,
                epsrel=1e-8,
            )
            return f
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.speedgrid.xrec)
        n = jnp.arange(self.speedgrid.nx)
        sgn, lg = lGammainc(l / 2 + n / 2 + 5 / 2, x**2)
        li, sgn = jax.scipy.special.logsumexp(lg, b=sgn * p, return_sign=True)
        return sgn * jnp.exp(li) / 2

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_1(self, x, l, k):
        return -self._integrand1(x, l, k)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_2(self, x, l, k):
        return self._integrand2(x, l, k)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_3(self, x, l, k):
        return -self._integrand3(x, l, k)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_4(self, x, l, k):
        return self._integrand4(x, l, k)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_1(self, x, l, k):
        return jax.grad(self._dI_1)(x, l, k)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_2(self, x, l, k):
        return jax.grad(self._dI_2)(x, l, k)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_3(self, x, l, k):
        return jax.grad(self._dI_3)(x, l, k)

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_4(self, x, l, k):
        return jax.grad(self._dI_4)(x, l, k)


class PitchAngleScattering(lx.AbstractLinearOperator):
    """Diffusion operator in pitch angle direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    nus: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        p2: int = 4,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.p2 = p2
        self.axorder = axorder
        nus = []
        x = speedgrid.x
        for spa in species:
            nu = 0.0
            for spb in species:
                nu += nuD_ab(spa, spb, x * spa.v_thermal)
            nus.append(nu)
        self.nus = jnp.asarray(nus)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))

        h = jnp.pi / self.pitchgrid.nxi
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi

        f1 = fdfwd(f, str(self.p2) + "z", h=h, bc="symmetric", axis=2)
        f1 *= (cosa / sina)[:, None, None]
        f2 = fd2(f, self.p2, h=h, bc="symmetric", axis=2)
        df = f1 + f2
        df *= -self.nus[:, :, None, None, None] / 2

        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )

        h = jnp.pi / self.pitchgrid.nxi
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi
        f = jnp.ones(self.pitchgrid.nxi)
        f1 = jnp.diag(
            jax.jacfwd(fdfwd)(f, str(self.p2) + "z", h=h, bc="symmetric", axis=0)
        )[None, None, :, None, None]
        f1 *= (cosa / sina)[None, None, :, None, None]
        f2 = jnp.diag(jax.jacfwd(fd2)(f, self.p2, h=h, bc="symmetric", axis=0))[
            None, None, :, None, None
        ]
        df = f1 + f2
        df = jnp.tile(df, (1, 1, 1, self.field.ntheta, self.field.nzeta))

        df *= -self.nus[:, :, None, None, None] / 2

        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "s":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, len(self.species))))
        if self.axorder[-1] == "x":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.speedgrid.nx)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )

        h = jnp.pi / self.pitchgrid.nxi
        sina = jnp.sqrt(1 - self.pitchgrid.xi**2)
        cosa = -self.pitchgrid.xi
        f = jnp.ones(self.pitchgrid.nxi)
        f1 = jax.jacfwd(fdfwd)(f, str(self.p2) + "z", h=h, bc="symmetric", axis=0)[
            None, None, :, None, None, :
        ]
        f1 *= (cosa / sina)[None, None, :, None, None, None]
        f2 = jax.jacfwd(fd2)(f, self.p2, h=h, bc="symmetric", axis=0)[
            None, None, :, None, None, :
        ]
        df = f1 + f2
        df = jnp.tile(df, (1, 1, 1, self.field.ntheta, self.field.nzeta, 1))

        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, self.pitchgrid.nxi, self.pitchgrid.nxi))
        return df

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class EnergyScattering(lx.AbstractLinearOperator):
    """Diffusion operator in speed direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    axorder: str = eqx.field(static=True)
    coeff0: jax.Array
    coeff1: jax.Array
    coeff2: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.axorder = axorder
        coeff0 = []
        coeff1 = []
        coeff2 = []
        x = speedgrid.x

        for spa in species:
            vta = spa.v_thermal
            v = x * vta
            term0 = 0.0
            term1 = 0.0
            term2 = 0.0
            for spb in species:
                nupar = nupar_ab(spa, spb, v)
                nuD = nuD_ab(spa, spb, v)
                gamma = gamma_ab(spa, spb)
                ma, mb = spa.species.mass, spb.species.mass
                vtb = spb.v_thermal
                term0 += 4 * jnp.pi * gamma * ma / mb * spb(v)
                term1 += nuD * x - nupar * (x * vta / vtb) ** 2 * (1 - ma / mb) * x
                term2 += nupar * x**2 / 2
            coeff0.append(term0)
            coeff1.append(term1)
            coeff2.append(term2)
        self.coeff0 = jnp.asarray(coeff0)
        self.coeff1 = jnp.asarray(coeff1)
        self.coeff2 = jnp.asarray(coeff2)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        Dx = self.speedgrid.Dx_pseudospectral
        df = jnp.einsum("yx,sxatz->syatz", Dx, f)
        ddf = jnp.einsum("yx,sxatz->syatz", Dx, df)

        out = (
            self.coeff2[:, :, None, None, None] * ddf
            + self.coeff1[:, :, None, None, None] * df
            + self.coeff0[:, :, None, None, None] * f
        )
        out = jnp.moveaxis(out, (0, 1, 2, 3, 4), caxorder)
        return -out.reshape(shp)

    @eqx.filter_jit
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )

        f = jnp.ones(self.speedgrid.nx)[None, :, None, None, None]
        df = jnp.diag(self.speedgrid.Dx_pseudospectral)[None, :, None, None, None]
        ddf = jnp.diag(
            self.speedgrid.Dx_pseudospectral @ self.speedgrid.Dx_pseudospectral
        )[None, :, None, None, None]
        out = (
            self.coeff2[:, :, None, None, None] * ddf
            + self.coeff1[:, :, None, None, None] * df
            + self.coeff0[:, :, None, None, None] * f
        )
        out = jnp.tile(
            out, (1, 1, self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta)
        )
        out = jnp.moveaxis(out, (0, 1, 2, 3, 4), caxorder)
        return -out.flatten()

    @eqx.filter_jit
    def block_diagonal(self):
        """Block diagonal of operator as (N,M,M) array."""
        if self.axorder[-1] == "s":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, len(self.species))))
        if self.axorder[-1] == "a":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.pitchgrid.nxi)))
        if self.axorder[-1] == "t":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.ntheta)))
        if self.axorder[-1] == "z":
            return jax.vmap(jnp.diag)(self.diagonal().reshape((-1, self.field.nzeta)))

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )

        f = jnp.eye(self.speedgrid.nx)[None, :, None, None, None, :]
        df = self.speedgrid.Dx_pseudospectral[None, :, None, None, None, :]
        ddf = (self.speedgrid.Dx_pseudospectral @ self.speedgrid.Dx_pseudospectral)[
            None, :, None, None, None, :
        ]
        out = (
            self.coeff2[:, :, None, None, None, None] * ddf
            + self.coeff1[:, :, None, None, None, None] * df
            + self.coeff0[:, :, None, None, None, None] * f
        )

        out = jnp.tile(
            out, (1, 1, self.pitchgrid.nxi, self.field.ntheta, self.field.nzeta, 1)
        )
        out = jnp.moveaxis(out, (0, 1, 2, 3, 4), caxorder)
        out = out.reshape((-1, self.speedgrid.nx, self.speedgrid.nx))
        return -out

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.eye(self.in_size())
        return jax.vmap(self.mv)(x).T

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class FieldPartCD(lx.AbstractLinearOperator):
    """Diagonal part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    axorder: str = eqx.field(static=True)
    C: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.potentials = potentials
        self.axorder = axorder

        x = speedgrid.x

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |

        C = []
        Ca = []
        for a, spa in enumerate(species):
            va = spa.v_thermal
            ma = spa.species.mass
            v = x * va
            Fa = spa(v)
            for b, spb in enumerate(species):
                gamma = gamma_ab(spa, spb)
                vb = spb.v_thermal
                mb = spb.species.mass
                # need to evaluate fb on the speed grid for fa
                # if va >> vb, then fa is "wider" in speed, and we're evaluating in
                # the tail of fb, ie xq >> 1, so xq = va/vb x
                xq = va / vb * x
                # matrix to evaluate fb at xq
                Dab = orthax.orthvander(
                    xq, speedgrid.nx - 1, speedgrid.xrec
                ) * speedgrid.xrec.weight(xq[:, None])
                prefactor = jnp.diag(gamma * Fa * 4 * jnp.pi * ma / mb)
                CDab = prefactor @ Dab @ speedgrid.xvander_inv
                Ca.append(CDab)
            C.append(Ca)
            Ca = []
        self.C = jnp.asarray(C)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        df = jnp.einsum("psyx,sxatz->pyatz", self.C, f)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return -df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        raise NotImplementedError

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        raise NotImplementedError

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class FieldPartCG(lx.AbstractLinearOperator):
    """Rosenbluth G part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    axorder: str = eqx.field(static=True)
    prefactor: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.potentials = potentials
        self.axorder = axorder

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |

        x = speedgrid.x
        prefactor = []
        for a, spa in enumerate(species):
            va = spa.v_thermal
            v = x * va
            Fa = spa(v)
            pb = []
            for b, spb in enumerate(species):
                gamma = gamma_ab(spa, spb)
                vb = spb.v_thermal
                # ddG == dG/dx^2, so d/dv^2 also picks up an extra 1/vb^2
                # G from potentials is really G/vb^4 so we multiply by vb^4 and cancel
                pb.append(gamma * Fa * 2 * v**2 * vb**2 / va**4)
            prefactor.append(pb)

        self.prefactor = jnp.array(prefactor)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        # G is in modal basis in legendre/xi
        # these go from nodal alpha to modal l, and back
        f = jnp.einsum("la,sxatz->sxltz", self.potentials.Txi_inv, f)
        # convert to modal basis in x
        f = jnp.einsum("kx,sxltz->skltz", self.speedgrid.xvander_inv, f)
        # apply potential, G is effectively block diagonal in l
        Gabxlk = (
            self.prefactor[:, :, :, None, None]
            * self.potentials.ddGxlk[:, :, :, : self.potentials.legendregrid.nxi]
        )
        df = jnp.einsum("psxlk,skltz->pxltz", Gabxlk, f)
        # transform back to real space in pitch angle
        df = jnp.einsum("al,pxltz->pxatz", self.potentials.Txi, df)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return -df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        raise NotImplementedError

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        raise NotImplementedError

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class FieldPartCH(lx.AbstractLinearOperator):
    """Rosenbluth H part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    axorder: str = eqx.field(static=True)
    prefactor_H: jax.Array
    prefactor_dH: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.potentials = potentials
        self.axorder = axorder

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |

        x = speedgrid.x
        prefactor_H = []
        prefactor_dH = []
        for a, spa in enumerate(species):
            va = spa.v_thermal
            ma = spa.species.mass
            v = x * va
            Fa = spa(v)
            temp_prefactor_H = []
            temp_prefactor_dH = []
            for b, spb in enumerate(species):
                gamma = gamma_ab(spa, spb)
                vb = spb.v_thermal
                mb = spb.species.mass
                # dH == dH/dx, so d/dv also picks up an extra 1/vb
                # H from potentials is really H/vb^2 so we multiply by vb^2 and cancel
                temp_prefactor_H.append(-2 * vb**2 / va**2 * gamma * Fa)
                temp_prefactor_dH.append(
                    -2 * v * vb / va**2 * (1 - ma / mb) * gamma * Fa
                )
            prefactor_H.append(temp_prefactor_H)
            prefactor_dH.append(temp_prefactor_dH)

        self.prefactor_H = jnp.array(prefactor_H)
        self.prefactor_dH = jnp.array(prefactor_dH)

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nxi,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        # H is in modal basis in legendre/xi
        # these go from nodal alpha to modal l, and back
        f = jnp.einsum("la,sxatz->sxltz", self.potentials.Txi_inv, f)
        # convert to modal basis in x
        f = jnp.einsum("kx,sxltz->skltz", self.speedgrid.xvander_inv, f)
        # apply potential, H is effectively block diagonal in l
        Habxlk = (
            self.prefactor_H[:, :, :, None, None]
            * self.potentials.Hxlk[:, :, :, : self.potentials.legendregrid.nxi]
        )
        dHabxlk = (
            self.prefactor_dH[:, :, :, None, None]
            * self.potentials.dHxlk[:, :, :, : self.potentials.legendregrid.nxi]
        )
        df = jnp.einsum("psxlk,skltz->pxltz", Habxlk + dHabxlk, f)
        # transform back to real space in pitch angle
        df = jnp.einsum("al,pxltz->pxatz", self.potentials.Txi, df)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return -df.reshape(shp)

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        raise NotImplementedError

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        raise NotImplementedError

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class FieldParticleScattering(lx.AbstractLinearOperator):
    """Field-particle part of Fokker-Planck Landau collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened

    """

    field: Field
    speedgrid: SpeedGrid
    pitchgrid: UniformPitchAngleGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    axorder: str = eqx.field(static=True)
    CD: FieldPartCD
    CG: FieldPartCG
    CH: FieldPartCH

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.potentials = potentials
        self.axorder = axorder

        self.CG = FieldPartCG(
            field,
            pitchgrid,
            speedgrid,
            species,
            potentials,
        )
        self.CH = FieldPartCH(
            field,
            pitchgrid,
            speedgrid,
            species,
            potentials,
        )
        self.CD = FieldPartCD(
            field,
            pitchgrid,
            speedgrid,
            species,
            potentials,
        )

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        out1 = self.CD.mv(vector)
        out2 = self.CG.mv(vector)
        out3 = self.CH.mv(vector)
        return out1 + out2 + out3

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        raise NotImplementedError

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        raise NotImplementedError

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


class FokkerPlanckLandau(lx.AbstractLinearOperator):
    """Fokker-Planck Landau collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: SpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    CL: PitchAngleScattering
    CE: EnergyScattering
    CF: FieldParticleScattering

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: SpeedGrid,
        species: list[LocalMaxwellian],
        potentials: Optional[RosenbluthPotentials] = None,
        p2: int = 4,
        axorder: str = "sxatz",
    ):
        assert axorder in {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        if potentials is None:
            potentials = RosenbluthPotentials(speedgrid, pitchgrid, species)
        self.potentials = potentials
        self.p2 = p2
        self.axorder = axorder

        self.CL = PitchAngleScattering(
            field, pitchgrid, speedgrid, species, p2, axorder
        )
        self.CE = EnergyScattering(field, pitchgrid, speedgrid, species, axorder)
        self.CF = FieldParticleScattering(
            field, pitchgrid, speedgrid, species, potentials, axorder
        )

    @eqx.filter_jit
    def mv(self, vector):
        """Matrix vector product."""
        out1 = self.CL.mv(vector)
        out2 = self.CE.mv(vector)
        out3 = self.CF.mv(vector)
        return out1 + out2 + out3

    @eqx.filter_jit
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        d1 = self.CL.diagonal()
        d2 = self.CE.diagonal()
        # TODO: add field part terms
        return d1 + d2

    @eqx.filter_jit
    def block_diagonal(self) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        d1 = self.CL.block_diagonal()
        d2 = self.CE.block_diagonal()
        # TODO: add field part terms
        return d1 + d2

    def as_matrix(self):
        """Materialize the operator as a dense matrix."""
        x = jnp.zeros(self.in_size())
        return jax.jacfwd(self.mv)(x)

    def in_structure(self):
        """Pytree structure of expected input."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def out_structure(self):
        """Pytree structure of expected output."""
        return jax.ShapeDtypeStruct(
            (
                self.field.ntheta
                * self.field.nzeta
                * self.pitchgrid.nxi
                * self.speedgrid.nx
                * len(self.species),
            ),
            dtype=self.field.Bmag.dtype,
        )

    def transpose(self):
        """Transpose of the operator."""
        x = jnp.zeros(self.in_size())

        def fun(y):
            return jax.linear_transpose(self.mv, x)(y)[0]

        return lx.FunctionLinearOperator(fun, x)


@lx.is_symmetric.register(MDKEPitchAngleScattering)
@lx.is_diagonal.register(MDKEPitchAngleScattering)
@lx.is_tridiagonal.register(MDKEPitchAngleScattering)
@lx.is_symmetric.register(FieldPartCH)
@lx.is_diagonal.register(FieldPartCH)
@lx.is_tridiagonal.register(FieldPartCH)
@lx.is_symmetric.register(FieldPartCD)
@lx.is_diagonal.register(FieldPartCD)
@lx.is_tridiagonal.register(FieldPartCD)
@lx.is_symmetric.register(FieldPartCG)
@lx.is_diagonal.register(FieldPartCG)
@lx.is_tridiagonal.register(FieldPartCG)
@lx.is_symmetric.register(FieldParticleScattering)
@lx.is_diagonal.register(FieldParticleScattering)
@lx.is_tridiagonal.register(FieldParticleScattering)
@lx.is_symmetric.register(EnergyScattering)
@lx.is_diagonal.register(EnergyScattering)
@lx.is_tridiagonal.register(EnergyScattering)
@lx.is_symmetric.register(PitchAngleScattering)
@lx.is_diagonal.register(PitchAngleScattering)
@lx.is_tridiagonal.register(PitchAngleScattering)
@lx.is_symmetric.register(FokkerPlanckLandau)
@lx.is_diagonal.register(FokkerPlanckLandau)
@lx.is_tridiagonal.register(FokkerPlanckLandau)
def _(operator):
    return False
