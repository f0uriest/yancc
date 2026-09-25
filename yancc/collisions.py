"""Collision operators and methods for computing Rosenbluth potentials."""

import functools
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orthax
import quadax
from jaxtyping import Array, ArrayLike, Bool, Float

from .field import Field
from .finite_diff import fd2, fd_coeffs, fdfwd
from .linalg import (
    AbstractDKEOperator,
    banded_to_dense,
    dense_to_banded,
)
from .species import LocalMaxwellian, _species_pairs, gamma_ab, nuD_ab, nupar_ab
from .utils import (
    _parse_axorder_shape_3d,
    _parse_axorder_shape_4d,
)
from .velocity_grids import (
    AbstractSpeedGrid,
    LegendrePitchAngleGrid,
    MaxwellSpeedGrid,
    UniformPitchAngleGrid,
)


class MDKEPitchAngleScattering(AbstractDKEOperator):
    """Diffusion operator in xi direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    nuhat : float
        Monoenergetic collisionality, nu/v in units of 1/m
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
    nuhat: Float[Array, ""]
    p1: str = eqx.field(static=True)
    p2: int = eqx.field(static=True)
    gauge: Bool[Array, ""]
    axorder: str = eqx.field(static=True)
    _D: Float[Array, "na na"]
    _scale: Float[Array, ""]

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        nuhat: Float[ArrayLike, ""],
        p1: str = "4d",
        p2: int = 4,
        axorder: str = "atz",
        gauge: Bool[ArrayLike, ""] = False,
    ):
        assert pitchgrid.nalpha > fd_coeffs[1][p1].size // 2
        assert pitchgrid.nalpha > fd_coeffs[2][p2].size // 2
        self.field = field
        self.pitchgrid = pitchgrid
        self.nuhat = jnp.array(nuhat)
        self.p1 = p1
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        h = jnp.pi / pitchgrid.nalpha
        f1 = jnp.ones(pitchgrid.nalpha)
        D1 = jax.jacfwd(fdfwd)(f1, str(p2) + "z", h=h, bc="symmetric")
        D2 = jax.jacfwd(fd2)(f1, p2, h=h, bc="symmetric")
        sina = jnp.sqrt(1 - pitchgrid.xi**2)
        cosa = -pitchgrid.xi
        w1 = -(self.nuhat / 2 * cosa / sina)
        w2 = -self.nuhat / 2
        # w1, w2 only depend on pitch (na), not state size, so fold into a
        # single (na, na) operator.
        self._D = w1[:, None] * D1 + w2 * D2
        self._scale = self.nuhat / h**2

    @eqx.filter_jit
    @jax.named_scope("MDKEPitchAngleScattering.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nalpha, self.axorder
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2))  # (na, nt, nz)
        f1 = jnp.moveaxis(f, 0, -1)  # (nt, nz, na) - convolved axis last
        df = jnp.moveaxis(f1 @ self._D.T, -1, 0)

        idx = self.pitchgrid.nalpha // 2
        gval = jnp.where(self.gauge, self._scale * f[idx, 0, 0], df[idx, 0, 0])
        df = df.at[idx, 0, 0].set(gval, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("MDKEPitchAngleScattering.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nalpha, self.axorder
        )
        df = jnp.diag(self._D)[:, None, None]
        df = jnp.broadcast_to(df, df.shape[:1] + (self.field.ntheta, self.field.nzeta))

        idx = self.pitchgrid.nalpha // 2
        gval = jnp.where(self.gauge, self._scale, df[idx, 0, 0])
        df = df.at[idx, 0, 0].set(gval, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("MDKEPitchAngleScattering.abs_row_sum")
    def abs_row_sum(self) -> Float[Array, " nf"]:
        """L1 norm of each row, sum_j |A_ij|, as a 1d array."""
        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nalpha, self.axorder
        )
        df = jnp.abs(self._D).sum(axis=1)[:, None, None]
        df = jnp.broadcast_to(df, df.shape[:1] + (self.field.ntheta, self.field.nzeta))

        idx = self.pitchgrid.nalpha // 2
        gval = jnp.where(self.gauge, jnp.abs(self._scale), df[idx, 0, 0])
        df = df.at[idx, 0, 0].set(gval, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("MDKEPitchAngleScattering.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        assert fmt in ["dense", "banded"]

        if self.axorder[-1] != "a":  # its just diagonal
            if bw is None:
                bw = 0
            sizes = {
                "a": self.pitchgrid.nalpha,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            df = self.diagonal().reshape((-1, sizes[self.axorder[-1]]))
            if fmt == "dense":
                return jax.vmap(jnp.diag)(df)
            return jnp.pad(df[:, None, :], [(0, 0), (bw, bw), (0, 0)])

        if bw is None:
            bw = min(fd_coeffs[2][self.p2].size // 2, self.pitchgrid.nalpha // 2)

        _, caxorder = _parse_axorder_shape_3d(
            self.field.ntheta, self.field.nzeta, self.pitchgrid.nalpha, self.axorder
        )
        df = dense_to_banded(bw, bw, self._D)
        df = jnp.broadcast_to(df, (self.field.ntheta, self.field.nzeta) + df.shape)

        # gauge row is replaced by a single diagonal entry
        bandwidth = 2 * bw + 1
        bands = jnp.arange(bandwidth)
        idx = self.pitchgrid.nalpha // 2
        cols = (idx + bw - bands) % self.pitchgrid.nalpha
        basis = jnp.zeros(bandwidth, dtype=df.dtype).at[bw].set(1.0)
        gval = jnp.where(self.gauge, self._scale * basis, df[0, 0, bands, cols])
        df = df.at[0, 0, bands, cols].set(gval, unique_indices=True)
        # band axis takes the place of the convolved axis, which stays last
        df = jnp.moveaxis(df, 2, 0)
        df = jnp.moveaxis(df, (0, 1, 2), caxorder)
        df = df.reshape((-1, 2 * bw + 1, self.pitchgrid.nalpha))
        if fmt == "dense":
            df = banded_to_dense(bw, bw, df)
        return df


# Fixed Gauss-Legendre rule for the speed integrals of the Rosenbluth potentials.
# The integrands z^q L_k(z) exp(-z^2) are evaluated pointwise, with L_k from its
# three-term recurrence, so the quadrature error is limited primarily by the
# conditioning of the integral itself. A fixed rule (as opposed to an adaptive one) has
# no data dependent control flow, so it vectorizes cheaply and compiles to a small
# graph.
_GL_ORDER = 32
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_ORDER)
_GL_NODES = (_GL_NODES + 1) / 2  # map to [0, 1]
_GL_WEIGHTS = _GL_WEIGHTS / 2
# exp(-z^2) times the highest polynomial powers used (~20) is negligible beyond this
_GL_ZMAX = 10.0
_GL_LOWER_PANELS = 2
_GL_UPPER_GEOMETRIC_PANELS = 4
_GL_UPPER_UNIFORM_PANELS = 2


def _gauss_legendre_panels(edges):
    """Nodes and weights for panels with breakpoints edges[..., i], edges[..., i+1]."""
    lo = edges[..., :-1, None]
    hi = edges[..., 1:, None]
    z = lo + (hi - lo) * _GL_NODES
    w = (hi - lo) * _GL_WEIGHTS
    return z.reshape(*edges.shape[:-1], -1), w.reshape(*edges.shape[:-1], -1)


def _lower_speed_quadrature(x):
    """Nodes and weights for integrals over [0, x] with a Maxwellian weight."""
    # Beyond _GL_ZMAX the integrand is negligible, so larger x just truncates there.
    top = jnp.minimum(x, _GL_ZMAX)
    edges = top[..., None] * jnp.linspace(0.0, 1.0, _GL_LOWER_PANELS + 1)
    return _gauss_legendre_panels(edges)


def _upper_speed_quadrature(x):
    """Nodes and weights for integrals over [x, inf) with a Maxwellian weight."""
    # The integrands contain negative powers of z, which are sharply peaked at z = x
    # when x is small. Geometrically graded panels from x up to max(x, 1) resolve
    # that with a fixed number of nodes regardless of how small x is. Beyond that,
    # uniform panels cover the Maxwellian tail, whose decay length shrinks like 1/x
    # for large x.
    mid = jnp.maximum(x, 1.0)
    ratio = (mid / x) ** (1.0 / _GL_UPPER_GEOMETRIC_PANELS)
    geometric = x[..., None] * ratio[..., None] ** jnp.arange(
        _GL_UPPER_GEOMETRIC_PANELS + 1
    )
    far = mid + jnp.minimum(8.0, 22.0 / x + 1.0)
    uniform = mid[..., None] + (far - mid)[..., None] * jnp.linspace(
        0.0, 1.0, _GL_UPPER_UNIFORM_PANELS + 1
    )
    z1, w1 = _gauss_legendre_panels(geometric)
    z2, w2 = _gauss_legendre_panels(uniform)
    return jnp.concatenate([z1, z2], axis=-1), jnp.concatenate([w1, w2], axis=-1)


class RosenbluthPotentials(eqx.Module):
    """Thing to calculate Rosenbluth Potentials.

    Parameters
    ----------
    speedgrid : MaxwellSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    nL : int
        Number of Legendre modes to use for potentials.
    quad : bool
        Whether to compute potentials using adaptive quadrature (slow but robust) or a
        fixed Gauss-Legendre quadrature (fast, but may be inaccurate for nx > 20).
    """

    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    legendregrid: LegendrePitchAngleGrid
    quad: bool = eqx.field(static=True)
    ddGxlk: jax.Array
    Hxlk: jax.Array
    dHxlk: jax.Array

    def __init__(self, speedgrid, species, nL=8, quad=False):
        self.speedgrid = speedgrid
        self.species = species
        self.legendregrid = LegendrePitchAngleGrid(nL)
        self.quad = quad

        ns = len(species)
        x = self.speedgrid.x[:, None, None]
        l = jnp.arange(nL)[None, :, None]
        k = jnp.arange(self.speedgrid.nx)[None, None, :]
        self.ddGxlk = jnp.zeros((ns, ns, self.speedgrid.nx, nL, self.speedgrid.nx))
        self.dHxlk = jnp.zeros((ns, ns, self.speedgrid.nx, nL, self.speedgrid.nx))
        self.Hxlk = jnp.zeros((ns, ns, self.speedgrid.nx, nL, self.speedgrid.nx))

        # arr[a,b] is potential operator from species b evaluated at x grid of species a
        def potentials_ab(spa, spb):
            va, vb = spa.v_thermal, spb.v_thermal
            # suppose vb > va, then fb is wider in v space
            # so to get {H,G}b on fa grid, we evaluate at x << 1 ie x*va/vb
            xa = x * va / vb
            ddG = self._ddGlk(xa, l, k)
            dH = self._dHlk(xa, l, k)
            H = self._Hlk(xa, l, k)
            # ddG is in normalized units, needs to be scaled by vb^4
            # but ddG is dG/dx^2, want dG/dv^2 so gives extra factor of 1/vb^2
            # dH is in normalized units, needs to be scaled by vb^2
            # but dH is dH/dx, want dH/dv so gives extra factor of 1/vb
            # H is in normalized units, needs to be scaled by vb^2
            return ddG * vb**2, dH * vb, H * vb**2

        self.ddGxlk, self.dHxlk, self.Hxlk = _species_pairs(
            potentials_ab, species, species
        )

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
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1, unique_indices=True)
        return (
            z ** (-l + 1)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand2(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1, unique_indices=True)
        return (
            z ** (l + 2)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand3(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1, unique_indices=True)
        return (
            z ** (-l + 3)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @eqx.filter_jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand4(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1, unique_indices=True)
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
                epsrel=1e-12,
            )
            return f
        z, w = _upper_speed_quadrature(x)
        return jnp.sum(w * self._integrand1(z, l, k))

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
                epsrel=1e-12,
            )
            return f
        z, w = _lower_speed_quadrature(x)
        return jnp.sum(w * self._integrand2(z, l, k))

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
                epsrel=1e-12,
            )
            return f
        z, w = _upper_speed_quadrature(x)
        return jnp.sum(w * self._integrand3(z, l, k))

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
                epsrel=1e-12,
            )
            return f
        z, w = _lower_speed_quadrature(x)
        return jnp.sum(w * self._integrand4(z, l, k))

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


class PitchAngleScattering(AbstractDKEOperator):
    """Diffusion operator in pitch angle direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : AbstractSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    background : list[LocalMaxwellian]
        Background species to include in the collision operator without solving for df.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: AbstractSpeedGrid
    species: list[LocalMaxwellian]
    background: list[LocalMaxwellian]
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]
    nus: jax.Array
    _D: Float[Array, "na na"]
    _scale: Float[Array, "ns nidx"]

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: AbstractSpeedGrid,
        species: list[LocalMaxwellian],
        background: list[LocalMaxwellian] | None = None,
        p2: int = 4,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
        coulomb_log=None,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if background is None:
            background = []
        self.background = background
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        x = speedgrid.x

        def nu_ab(spa, spb):
            return nuD_ab(spa, spb, x * spa.v_thermal, lnlambda=coulomb_log)

        self.nus = _species_pairs(nu_ab, species, species + background).sum(axis=1)
        h = jnp.pi / pitchgrid.nalpha
        f1 = jnp.ones(pitchgrid.nalpha)
        D1 = jax.jacfwd(fdfwd)(f1, str(p2) + "z", h=h, bc="symmetric")
        D2 = jax.jacfwd(fd2)(f1, p2, h=h, bc="symmetric")
        sina = jnp.sqrt(1 - pitchgrid.xi**2)
        cosa = -pitchgrid.xi
        # cos/sin only depends on pitchgrid; fold into a single (na, na) op.
        # The species/x-dependent prefactor (-nus/2) is applied in mv.
        self._D = (cosa / sina)[:, None] * D1 + D2
        idxx = self.speedgrid.gauge_idx
        self._scale = self.nus[:, idxx] / h**2

    @eqx.filter_jit
    @jax.named_scope("PitchAngleScattering.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))  # (ns, nx, na, nt, nz)
        f1 = jnp.moveaxis(f, 2, -1)  # (ns, nx, nt, nz, na) - convolved axis last
        df = jnp.moveaxis(f1 @ self._D.T, -1, 2)
        df *= -self.nus[:, :, None, None, None] / 2

        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        gval = jnp.where(
            self.gauge,
            self._scale * f[:, idxx, idxa, 0, 0],
            df[:, idxx, idxa, 0, 0],
        )
        df = df.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("PitchAngleScattering.diagonal")
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        _, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )

        df = jnp.diag(self._D)[None, None, :, None, None]
        df = jnp.broadcast_to(df, df.shape[:3] + (self.field.ntheta, self.field.nzeta))
        df = -self.nus[:, :, None, None, None] / 2 * df

        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        gval = jnp.where(self.gauge, self._scale, df[:, idxx, idxa, 0, 0])
        df = df.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("PitchAngleScattering.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        assert fmt in ["dense", "banded"]

        if self.axorder[-1] != "a":  # its just diagonal
            if bw is None:
                bw = 0
            df = self.diagonal()
            sizes = {
                "s": len(self.species),
                "x": self.speedgrid.nx,
                "a": self.pitchgrid.nalpha,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            df = df.reshape((-1, sizes[self.axorder[-1]]))
            if fmt == "dense":
                op = jax.vmap(jnp.diag)
            else:
                op = lambda x: jnp.pad(x[:, None, :], [(0, 0), (bw, bw), (0, 0)])
            return op(df)

        if bw is None:
            bw = fd_coeffs[2][self.p2].size // 2

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )

        df = dense_to_banded(bw, bw, self._D)[None, None, :, None, None, :]
        df = -self.nus[:, :, None, None, None, None] / 2 * df
        df = jnp.broadcast_to(
            df, df.shape[:3] + (self.field.ntheta, self.field.nzeta) + df.shape[5:]
        )

        idxa = jnp.atleast_1d(self.pitchgrid.nalpha // 2)
        idxx = self.speedgrid.gauge_idx
        scale = self._scale

        bandwidth = 2 * bw + 1
        # 1. Band indices cover the entire bandwidth
        bands = jnp.arange(bandwidth)
        # 2. Compute the wrapped column indices (shape: M, bandwidth)
        cols = (idxa[:, None] + bw - bands[None, :]) % self.pitchgrid.nalpha
        # 3. Create the batched replacement block (shape: ns, M, bandwidth)
        vals = jnp.zeros((len(self.species), idxx.size, bandwidth))
        # Drop the (ns, M) scales precisely onto the main diagonal across the batch
        vals = vals.at[:, :, bw].set(scale)
        # 4. Reshape indices for orthogonal broadcasting across dimensions
        idxx_mesh = idxx[:, None]
        bands_mesh = bands[None, :]
        # 5. Apply the update
        gval = jnp.where(self.gauge, vals, df[:, idxx_mesh, bands_mesh, 0, 0, cols])
        df = df.at[:, idxx_mesh, bands_mesh, 0, 0, cols].set(gval, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, 2 * bw + 1, self.pitchgrid.nalpha))
        if fmt == "dense":
            df = banded_to_dense(bw, bw, df)
        return df


class EnergyScattering(AbstractDKEOperator):
    """Diffusion operator in speed direction.

    Parameters
    ----------
    field : Field
        Magnetic field data.
    pitchgrid : UniformPitchAngleGrid
        Pitch angle grid data.
    speedgrid : MaxwellSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    background : list[LocalMaxwellian]
        Background species to include in the collision operator without solving for df.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened
    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    background: list[LocalMaxwellian]
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]
    coeff0: jax.Array
    coeff1: jax.Array
    coeff2: jax.Array
    _M: jax.Array
    _scale: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        background: list[LocalMaxwellian] | None = None,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
        coulomb_log=None,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        if background is None:
            background = []
        self.background = background
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        x = speedgrid.x
        xrec = speedgrid.xrec

        def terms_ab(spa, spb):
            vta = spa.v_thermal
            v = x * vta
            nupar = nupar_ab(spa, spb, v, lnlambda=coulomb_log)
            nuD = nuD_ab(spa, spb, v, lnlambda=coulomb_log)
            gamma = gamma_ab(spa, spb, lnlambda=coulomb_log)
            ma, mb = spa.species.mass, spb.species.mass
            vtb = spb.v_thermal
            term0 = 4 * jnp.pi * gamma * ma / mb * spb(v)
            term1 = nuD * x - nupar * (x * vta / vtb) ** 2 * (1 - ma / mb) * x
            term2 = nupar * x**2 / 2
            # flux form: C f = x^-2 d/dx [a (f' + 2 tau x f)], a = nupar x^4 / 2,
            # tau = T_a / T_b, which expands to term2 f'' + term1 f' + term0 f.
            a = nupar * x**4 / 2
            tau = spa.temperature / spb.temperature
            diffusion = a
            relaxation = 2 * (tau - 1) * a * x
            return term0, term1, term2, diffusion, relaxation

        term0, term1, term2, diffusion, relaxation = _species_pairs(
            terms_ab, species, species + background
        )
        self.coeff0 = term0.sum(axis=1)
        self.coeff1 = term1.sum(axis=1)
        self.coeff2 = term2.sum(axis=1)
        diffusion = diffusion.sum(axis=1)
        relaxation = relaxation.sum(axis=1)

        # The speed operator acts only on the x axis and is independent of the
        # vector, so it is collapsed into a single (ns, ny, nx) matrix applied with
        # one einsum per mv.
        #
        # It is discretized in weak (Galerkin) form. The grid represents
        # f = exp(-x^2) p(x) with p expanded in the orthogonal polynomials P_n. Testing
        # the flux form against P_m and integrating by parts gives
        #   (P_m, x^2 P_n) d = -[(P_m', a P_n') + (P_m', 2(tau-1) a x P_n)] c
        # with (u, v) the exp(-x^2)-weighted inner product evaluated by the grid's
        # quadrature. The boundary term vanishes since a ~ x^4 at x=0. For tau=1 the
        # stiffness matrix is symmetric positive semidefinite and the mass matrix is
        # positive definite, so the discrete operator is dissipative for any set of
        # colliding species.
        #
        # Collocating term2 f'' + term1 f' + term0 f at the nodes instead has no such
        # guarantee: the lowest node has no neighbour below it, so its one-sided second
        # derivative is anti-dissipative and relies on a positive term1 there to cancel
        # it. A background whose thermal speed puts that node inside its collisional
        # transition makes term1 negative, giving a spurious growing mode that the
        # multigrid smoothers cannot handle.
        #
        # The price is that the weak form returns the projection of C_E f onto the
        # polynomial space rather than its nodal values. The collision coefficients are
        # not polynomial, so the cancellation of C_E against the collocated pitch-angle
        # and field-particle terms on the momentum and energy invariants holds only to
        # the speed resolution, converging spectrally with nx. The density invariant
        # has zero flux pointwise and is annihilated exactly.
        Vp = orthax.orthvander(x, speedgrid.nx - 1, xrec)
        Dmod = jax.jacfwd(lambda c: jnp.append(orthax.orthder(c, xrec), 0.0))(x)
        dVp = Vp @ Dmod
        gq = speedgrid.wx * xrec.weight(x)
        mass = jnp.einsum("im,i,in->mn", Vp, gq * x**2, Vp)
        stiffness = jnp.einsum("im,si,in->smn", dVp, gq * diffusion, dVp) + jnp.einsum(
            "im,si,in->smn", dVp, gq * relaxation, Vp
        )
        modal = -jnp.linalg.solve(mass[None], stiffness)
        self._M = speedgrid.xvander @ modal @ speedgrid.xvander_inv
        idxx = speedgrid.gauge_idx
        self._scale = (
            jnp.abs(self.coeff2[:, idxx] / jnp.mean(speedgrid.wx) ** 2)
            + jnp.abs(self.coeff1[:, idxx] / jnp.mean(speedgrid.wx))
            + jnp.abs(self.coeff0[:, idxx])
        )

    @eqx.filter_jit
    @jax.named_scope("EnergyScattering.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        out = jnp.einsum("syx,sxatz->syatz", self._M, f)
        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        gval = jnp.where(
            self.gauge,
            self._scale * f[:, idxx, idxa, 0, 0],
            out[:, idxx, idxa, 0, 0],
        )
        out = out.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)
        out = jnp.moveaxis(out, (0, 1, 2, 3, 4), caxorder)
        return -out.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("EnergyScattering.diagonal")
    def diagonal(self):
        """Diagonal of the operator as a 1d array."""
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )

        out = jnp.diagonal(self._M, axis1=1, axis2=2)[:, :, None, None, None]
        out = jnp.broadcast_to(
            out,
            out.shape[:2]
            + (self.pitchgrid.nalpha, self.field.ntheta, self.field.nzeta),
        )
        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        scale = (
            jnp.abs(self.coeff2[:, idxx] / jnp.mean(self.speedgrid.wx) ** 2)
            + jnp.abs(self.coeff1[:, idxx] / jnp.mean(self.speedgrid.wx))
            + jnp.abs(self.coeff0[:, idxx])
        )
        gval = jnp.where(self.gauge, scale, out[:, idxx, idxa, 0, 0])
        out = out.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)
        out = jnp.moveaxis(out, (0, 1, 2, 3, 4), caxorder)
        return -out.flatten()

    @eqx.filter_jit
    @jax.named_scope("EnergyScattering.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        assert fmt in ["dense", "banded"]

        if self.axorder[-1] != "x":  # its just diagonal
            if bw is None:
                bw = 0
            df = self.diagonal()
            sizes = {
                "s": len(self.species),
                "x": self.speedgrid.nx,
                "a": self.pitchgrid.nalpha,
                "t": self.field.ntheta,
                "z": self.field.nzeta,
            }
            df = df.reshape((-1, sizes[self.axorder[-1]]))
            if fmt == "dense":
                op = jax.vmap(jnp.diag)
            else:
                op = lambda x: jnp.pad(x[:, None, :], [(0, 0), (bw, bw), (0, 0)])
            return op(df)

        # nx is basically always small and these matrices are usually dense
        # so we always compute it using dense fmt and convert to banded at the
        # end if needed.
        if bw is None:
            bw = self.speedgrid.nx // 2

        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )

        out = self._M[:, :, None, None, None, :]
        out = jnp.broadcast_to(
            out,
            out.shape[:2]
            + (self.pitchgrid.nalpha, self.field.ntheta, self.field.nzeta)
            + out.shape[5:],
        )
        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        scale = (
            jnp.abs(self.coeff2[:, idxx] / jnp.mean(self.speedgrid.wx) ** 2)
            + jnp.abs(self.coeff1[:, idxx] / jnp.mean(self.speedgrid.wx))
            + jnp.abs(self.coeff0[:, idxx])
        )
        g0 = jnp.where(self.gauge, 0.0, out[:, idxx, idxa, 0, 0, :])
        out = out.at[:, idxx, idxa, 0, 0, :].set(g0, unique_indices=True)
        g1 = jnp.where(self.gauge, scale, out[:, idxx, idxa, 0, 0, idxx])
        out = out.at[:, idxx, idxa, 0, 0, idxx].set(g1, unique_indices=True)
        out = jnp.moveaxis(out, (0, 1, 2, 3, 4), caxorder)
        out = out.reshape((-1, self.speedgrid.nx, self.speedgrid.nx))
        if fmt == "banded":
            out = dense_to_banded(bw, bw, out)
        return -out


# Shared field-particle math.


def _field_part_gh_diagonal(op, Ghat, scale):
    """Diagonal (1d) of a block-diagonal-in-l field-particle piece (G/H)."""
    shape, caxorder = _parse_axorder_shape_4d(
        op.field.ntheta,
        op.field.nzeta,
        op.pitchgrid.nalpha,
        op.speedgrid.nx,
        len(op.species),
        op.axorder,
    )
    Gabxly = Ghat
    Gabxliy = Gabxly[:, :, :, :, None, :] * op.Txi_inv[None, None, None, :, :, None]
    Gabxiy = jnp.einsum("il,abxliy->abxiy", op.Txi, Gabxliy)
    G = jnp.einsum("aaxix->axi", Gabxiy)
    df = jnp.broadcast_to(
        G[:, :, :, None, None],
        G.shape + (op.field.ntheta, op.field.nzeta),
    )
    idxa = op.pitchgrid.nalpha // 2
    idxx = op.speedgrid.gauge_idx
    gval = jnp.where(op.gauge, scale, df[:, idxx, idxa, 0, 0])
    df = df.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)
    df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
    return -df.flatten()


def _field_part_gh_block_diagonal(op, Ghat, scale, fmt="dense", bw=None):
    """Block diagonal (N,M,M) of a block-diagonal-in-l field-particle piece."""
    assert fmt in ["dense", "banded"]

    if op.axorder[-1] in ["t", "z"]:  # it's diagonal
        if bw is None:
            bw = 0
        df = _field_part_gh_diagonal(op, Ghat, scale)
        sizes = {
            "s": len(op.species),
            "x": op.speedgrid.nx,
            "a": op.pitchgrid.nalpha,
            "t": op.field.ntheta,
            "z": op.field.nzeta,
        }
        df = df.reshape((-1, sizes[op.axorder[-1]]))
        if fmt == "dense":
            fn = jax.vmap(jnp.diag)
        else:
            fn = lambda x: jnp.pad(x[:, None, :], [(0, 0), (bw, bw), (0, 0)])
        return fn(df)

    shape, caxorder = _parse_axorder_shape_4d(
        op.field.ntheta,
        op.field.nzeta,
        op.pitchgrid.nalpha,
        op.speedgrid.nx,
        len(op.species),
        op.axorder,
    )
    idxs = jnp.arange(len(op.species))
    idxa = jnp.atleast_1d(op.pitchgrid.nalpha // 2)
    idxx = op.speedgrid.gauge_idx
    Gabxly = Ghat

    # nx, ns are basically always small and these matrices are usually dense
    # so we always compute it using dense fmt and convert to banded at the
    # end if needed.
    if op.axorder[-1] == "s":
        if bw is None:
            bw = len(op.species) // 2
        Gabxliy = Gabxly[:, :, :, :, None, :] * op.Txi_inv[None, None, None, :, :, None]
        Gabxiy = jnp.einsum("il,abxliy->abxiy", op.Txi, Gabxliy)
        G = jnp.einsum("abxix->axib", Gabxiy)
        df = jnp.broadcast_to(
            G[:, :, :, None, None, :],
            G.shape[:3] + (op.field.ntheta, op.field.nzeta) + G.shape[3:],
        )
        idxs_mesh = idxs[:, None]
        idxx_mesh = idxx[None, :]
        # Step A: zero the gauge rows (all 'idxx' speed locations).
        g0 = jnp.where(op.gauge, 0.0, df[:, idxx, idxa, 0, 0, :])
        df = df.at[:, idxx, idxa, 0, 0, :].set(g0, unique_indices=True)
        # Step B: set the species diagonal at those locations (row=idxs,
        # x=idxx, col=idxs), broadcasting (ns, 1) against (1, N).
        g1 = jnp.where(
            op.gauge,
            scale,
            df[idxs_mesh, idxx_mesh, idxa, 0, 0, idxs_mesh],
        )
        df = df.at[idxs_mesh, idxx_mesh, idxa, 0, 0, idxs_mesh].set(
            g1, unique_indices=True
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, len(op.species), len(op.species)))
        if fmt == "banded":
            df = dense_to_banded(bw, bw, df)
        return -df
    elif op.axorder[-1] == "x":
        if bw is None:
            bw = op.speedgrid.nx // 2
        Gabxliy = Gabxly[:, :, :, :, None, :] * op.Txi_inv[None, None, None, :, :, None]
        Gabxiy = jnp.einsum("il,abxliy->abxiy", op.Txi, Gabxliy)
        G = jnp.einsum("aaxiy->axiy", Gabxiy)
        df = jnp.broadcast_to(
            G[:, :, :, None, None, :],
            G.shape[:3] + (op.field.ntheta, op.field.nzeta) + G.shape[3:],
        )
        g0 = jnp.where(op.gauge, 0.0, df[:, idxx, idxa, 0, 0, :])
        df = df.at[:, idxx, idxa, 0, 0, :].set(g0, unique_indices=True)
        g1 = jnp.where(op.gauge, scale, df[:, idxx, idxa, 0, 0, idxx])
        df = df.at[:, idxx, idxa, 0, 0, idxx].set(g1, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, op.speedgrid.nx, op.speedgrid.nx))
        if fmt == "banded":
            df = dense_to_banded(bw, bw, df)
        return -df
    # need trick here to avoid big matrices if banded format is desired.
    # note that it isn't actually banded in a, but we can still truncate it
    # if desired.
    elif op.axorder[-1] == "a":
        if bw is None:
            bw = op.pitchgrid.nalpha // 2  # full matrix
        Gsxl = jnp.einsum("aaxlx->axl", Gabxly)
        Gsxlj = jax.vmap(jax.vmap(jnp.diag))(Gsxl)
        Gsxij = jnp.einsum("il,sxlj->sxij", op.Txi, Gsxlj)
        df = jnp.einsum("ja,sxij->sxia", op.Txi_inv, Gsxij)

        if fmt == "banded":
            df = dense_to_banded(bw, bw, df)
            df = jnp.broadcast_to(
                df[:, :, :, None, None, :],
                df.shape[:3] + (op.field.ntheta, op.field.nzeta) + df.shape[3:],
            )
            bandwidth = 2 * bw + 1
            # 1. Band indices cover the entire bandwidth
            bands = jnp.arange(bandwidth)
            # 2. Compute the wrapped column indices for row 'idxa' across the batch
            cols = (idxa[:, None] + bw - bands[None, :]) % op.pitchgrid.nalpha
            # 3. Create the replacement values: zeros with 'scale' on the diagonal
            vals = jnp.zeros((len(op.species), idxx.size, bandwidth))
            vals = vals.at[:, :, bw].set(scale)
            # 4. Reshape indices for orthogonal broadcasting across dimensions
            idxx_mesh = idxx[:, None]
            bands_mesh = bands[None, :]
            # 5. Apply the update
            gval = jnp.where(op.gauge, vals, df[:, idxx_mesh, bands_mesh, 0, 0, cols])
            df = df.at[:, idxx_mesh, bands_mesh, 0, 0, cols].set(
                gval, unique_indices=True
            )
            df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
            df = df.reshape((-1, 2 * bw + 1, op.pitchgrid.nalpha))
            return -df
        else:
            idxa = idxa[0]
            df = jnp.broadcast_to(
                df[:, :, :, None, None, :],
                df.shape[:3] + (op.field.ntheta, op.field.nzeta) + df.shape[3:],
            )
            g0 = jnp.where(op.gauge, 0.0, df[:, idxx, idxa, 0, 0, :])
            df = df.at[:, idxx, idxa, 0, 0, :].set(g0, unique_indices=True)
            g1 = jnp.where(op.gauge, scale, df[:, idxx, idxa, 0, 0, idxa])
            df = df.at[:, idxx, idxa, 0, 0, idxa].set(g1, unique_indices=True)
            df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
            df = df.reshape((-1, op.pitchgrid.nalpha, op.pitchgrid.nalpha))
            return -df
    else:
        # unreachable, just kept to appease type checker
        raise ValueError()  # pragma: no cover


def _field_part_cd_diagonal(op, C, scale):
    """Diagonal (1d) of the diagonal-in-speed CD field-particle piece."""
    shape, caxorder = _parse_axorder_shape_4d(
        op.field.ntheta,
        op.field.nzeta,
        op.pitchgrid.nalpha,
        op.speedgrid.nx,
        len(op.species),
        op.axorder,
    )
    diag = jnp.einsum("iijj->ij", C)
    df = jnp.broadcast_to(
        diag[:, :, None, None, None],
        diag.shape + (op.pitchgrid.nalpha, op.field.ntheta, op.field.nzeta),
    )
    idxa = op.pitchgrid.nalpha // 2
    idxx = op.speedgrid.gauge_idx
    gval = jnp.where(op.gauge, scale, df[:, idxx, idxa, 0, 0])
    df = df.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)
    df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
    return -df.flatten()


def _field_part_cd_block_diagonal(op, C, scale, fmt="dense", bw=None):
    """Block diagonal (N,M,M) of the diagonal-in-speed CD field-particle piece."""
    assert fmt in ["dense", "banded"]

    if op.axorder[-1] != "x" and op.axorder[-1] != "s":  # its just diagonal
        if bw is None:
            bw = 0
        df = _field_part_cd_diagonal(op, C, scale)
        sizes = {
            "s": len(op.species),
            "x": op.speedgrid.nx,
            "a": op.pitchgrid.nalpha,
            "t": op.field.ntheta,
            "z": op.field.nzeta,
        }
        df = df.reshape((-1, sizes[op.axorder[-1]]))
        if fmt == "dense":
            fn = jax.vmap(jnp.diag)
        else:
            fn = lambda x: jnp.pad(x[:, None, :], [(0, 0), (bw, bw), (0, 0)])
        return fn(df)

    idxa = op.pitchgrid.nalpha // 2
    idxx = op.speedgrid.gauge_idx
    idxs = jnp.arange(len(op.species))
    idxs_mesh = idxs[:, None]
    idxx_mesh = idxx[None, :]

    # nx is basically always small and these matrices are usually dense
    # so we always compute it using dense fmt and convert to banded at the
    # end if needed.
    if op.axorder[-1] == "s":
        if bw is None:
            bw = len(op.species) // 2

        shape, caxorder = _parse_axorder_shape_4d(
            op.field.ntheta,
            op.field.nzeta,
            op.pitchgrid.nalpha,
            op.speedgrid.nx,
            len(op.species),
            op.axorder,
        )
        diag = jnp.einsum("ikjj->ijk", C)
        df = jnp.broadcast_to(
            diag[:, :, None, None, None, :],
            diag.shape[:2]
            + (op.pitchgrid.nalpha, op.field.ntheta, op.field.nzeta)
            + diag.shape[2:],
        )
        # Step A: zero the gauge rows (all 'idxx' speed locations).
        g0 = jnp.where(op.gauge, 0.0, df[:, idxx, idxa, 0, 0, :])
        df = df.at[:, idxx, idxa, 0, 0, :].set(g0, unique_indices=True)
        # Step B: set the species diagonal at those locations (row=idxs,
        # x=idxx, col=idxs), broadcasting (ns, 1) against (1, N).
        g1 = jnp.where(
            op.gauge,
            scale,
            df[idxs_mesh, idxx_mesh, idxa, 0, 0, idxs_mesh],
        )
        df = df.at[idxs_mesh, idxx_mesh, idxa, 0, 0, idxs_mesh].set(
            g1, unique_indices=True
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, len(op.species), len(op.species)))
        if fmt == "banded":
            df = dense_to_banded(bw, bw, df)
        return -df
    elif op.axorder[-1] == "x":
        if bw is None:
            bw = op.speedgrid.nx // 2
        shape, caxorder = _parse_axorder_shape_4d(
            op.field.ntheta,
            op.field.nzeta,
            op.pitchgrid.nalpha,
            op.speedgrid.nx,
            len(op.species),
            op.axorder,
        )
        diag = jnp.einsum("iijk->ijk", C)
        df = jnp.broadcast_to(
            diag[:, :, None, None, None, :],
            diag.shape[:2]
            + (op.pitchgrid.nalpha, op.field.ntheta, op.field.nzeta)
            + diag.shape[2:],
        )
        # Step A: zero the gauge rows (all 'idxx' speed locations).
        g0 = jnp.where(op.gauge, 0.0, df[:, idxx, idxa, 0, 0, :])
        df = df.at[:, idxx, idxa, 0, 0, :].set(g0, unique_indices=True)
        # Step B: set the speed diagonal at those locations (row=idxs,
        # x=idxx, col=idxx), broadcasting (ns, 1) against (1, N).
        g1 = jnp.where(
            op.gauge,
            scale,
            df[idxs_mesh, idxx_mesh, idxa, 0, 0, idxx_mesh],
        )
        df = df.at[idxs_mesh, idxx_mesh, idxa, 0, 0, idxx_mesh].set(
            g1, unique_indices=True
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        df = df.reshape((-1, op.speedgrid.nx, op.speedgrid.nx))
        if fmt == "banded":
            df = dense_to_banded(bw, bw, df)
        return -df
    else:
        # unreachable, just kept to appease type checker
        raise ValueError()  # pragma: no cover


class FieldPartCD(AbstractDKEOperator):
    """Diagonal part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : MaxwellSpeedGrid
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
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]
    C: jax.Array
    _scale: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
        coulomb_log=None,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.potentials = potentials
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

        x = speedgrid.x

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |

        def C_ab(spa, spb):
            va = spa.v_thermal
            ma = spa.species.mass
            v = x * va
            Fa = spa(v)
            gamma = gamma_ab(spa, spb, lnlambda=coulomb_log)
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
            return prefactor @ Dab @ speedgrid.xvander_inv

        self.C = _species_pairs(C_ab, species, species)

        # gauge scale depends only on operator data, not the input vector
        idxs = jnp.arange(len(species))
        self._scale = jnp.mean(jnp.abs(self.C[idxs, idxs]), axis=(2,))[
            :, speedgrid.gauge_idx
        ]

    @eqx.filter_jit
    @jax.named_scope("FieldPartCD.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f = vector
        shp = f.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f = f.reshape(shape)
        f = jnp.moveaxis(f, caxorder, (0, 1, 2, 3, 4))
        df = jnp.einsum("psyx,sxatz->pyatz", self.C, f)
        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        gval = jnp.where(
            self.gauge,
            self._scale * f[:, idxx, idxa, 0, 0],
            df[:, idxx, idxa, 0, 0],
        )
        df = df.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return -df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("FieldPartCD.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        return _field_part_cd_diagonal(self, self.C, self._scale)

    @eqx.filter_jit
    @jax.named_scope("FieldPartCD.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        return _field_part_cd_block_diagonal(self, self.C, self._scale, fmt, bw)


class FieldPartCG(AbstractDKEOperator):
    """Rosenbluth G part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : MaxwellSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]
    prefactor: jax.Array
    Txi: jax.Array
    Txi_inv: jax.Array
    _Ghat: jax.Array
    _scale: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
        coulomb_log=None,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.potentials = potentials
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |

        x = speedgrid.x

        def prefactor_ab(spa, spb):
            va = spa.v_thermal
            v = x * va
            Fa = spa(v)
            gamma = gamma_ab(spa, spb, lnlambda=coulomb_log)
            return gamma * Fa * 2 * v**2 / va**4

        self.prefactor = _species_pairs(prefactor_ab, species, species)
        self.Txi = orthax.orthvander(
            pitchgrid.xi,
            potentials.legendregrid.nalpha - 1,
            potentials.legendregrid.xirec,
        )
        self.Txi_inv = jnp.linalg.pinv(self.Txi)

        # Gabxlk and the gauge scale depend only on operator data, not the input
        # vector, so precompute them here rather than on every matvec. We also
        # pre-fold the nodal->modal speed transform (xvander_inv) into the
        # potential tensor, eliminating one einsum per matvec.
        Gabxlk = self.prefactor[:, :, :, None, None] * potentials.ddGxlk
        self._Ghat = jnp.einsum("psxlk,km->psxlm", Gabxlk, speedgrid.xvander_inv)
        idxs = jnp.arange(len(species))
        self._scale = jnp.mean(jnp.abs(Gabxlk[idxs, idxs]), axis=(2, 3))[
            :, speedgrid.gauge_idx
        ]

    @eqx.filter_jit
    @jax.named_scope("FieldPartCG.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f0 = vector
        shp = f0.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f0 = f0.reshape(shape)
        f0 = jnp.moveaxis(f0, caxorder, (0, 1, 2, 3, 4))
        # G is in modal basis in legendre/xi
        # this goes from nodal alpha to modal l
        f = jnp.einsum("la,sxatz->sxltz", self.Txi_inv, f0)
        # apply potential, G is effectively block diagonal in l. The nodal->modal
        # speed transform (xvander_inv) is pre-folded into Ghat.
        df = jnp.einsum("psxlk,skltz->pxltz", self._Ghat, f)
        # transform back to real space in pitch angle
        df = jnp.einsum("al,pxltz->pxatz", self.Txi, df)

        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        gval = jnp.where(
            self.gauge,
            self._scale * f0[:, idxx, idxa, 0, 0],
            df[:, idxx, idxa, 0, 0],
        )
        df = df.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)

        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return -df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("FieldPartCG.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        return _field_part_gh_diagonal(self, self._Ghat, self._scale)

    @eqx.filter_jit
    @jax.named_scope("FieldPartCG.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        return _field_part_gh_block_diagonal(self, self._Ghat, self._scale, fmt, bw)


class FieldPartCH(AbstractDKEOperator):
    """Rosenbluth H part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : MaxwellSpeedGrid
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
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]
    prefactor_H: jax.Array
    prefactor_dH: jax.Array
    Txi: jax.Array
    Txi_inv: jax.Array
    _Hhat: jax.Array
    _scale: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
        coulomb_log=None,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.pitchgrid = pitchgrid
        self.speedgrid = speedgrid
        self.species = species
        self.potentials = potentials
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |

        x = speedgrid.x

        def prefactors_ab(spa, spb):
            va = spa.v_thermal
            ma = spa.species.mass
            v = x * va
            Fa = spa(v)
            gamma = gamma_ab(spa, spb, lnlambda=coulomb_log)
            mb = spb.species.mass
            prefactor_H = -2 / va**2 * gamma * Fa
            prefactor_dH = -2 * v / va**2 * (1 - ma / mb) * gamma * Fa
            return prefactor_H, prefactor_dH

        self.prefactor_H, self.prefactor_dH = _species_pairs(
            prefactors_ab, species, species
        )
        self.Txi = orthax.orthvander(
            pitchgrid.xi,
            potentials.legendregrid.nalpha - 1,
            potentials.legendregrid.xirec,
        )
        self.Txi_inv = jnp.linalg.pinv(self.Txi)

        # The H potential tensor and gauge scale depend only on operator data, not
        # the input vector, so precompute them here rather than on every matvec. We
        # also pre-fold the nodal->modal speed transform (xvander_inv) into the
        # tensor, eliminating one einsum per matvec.
        Habxlk = self.prefactor_H[:, :, :, None, None] * potentials.Hxlk
        dHabxlk = self.prefactor_dH[:, :, :, None, None] * potentials.dHxlk
        Hsum = Habxlk + dHabxlk
        self._Hhat = jnp.einsum("psxlk,km->psxlm", Hsum, speedgrid.xvander_inv)
        idxs = jnp.arange(len(species))
        self._scale = jnp.mean(jnp.abs(Hsum[idxs, idxs]), axis=(2, 3))[
            :, speedgrid.gauge_idx
        ]

    @eqx.filter_jit
    @jax.named_scope("FieldPartCH.mv")
    def mv(self, vector):
        """Matrix vector product."""
        f0 = vector
        shp = f0.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f0 = f0.reshape(shape)
        f0 = jnp.moveaxis(f0, caxorder, (0, 1, 2, 3, 4))
        # H is in modal basis in legendre/xi
        # this goes from nodal alpha to modal l
        f = jnp.einsum("la,sxatz->sxltz", self.Txi_inv, f0)
        # apply potential, H is effectively block diagonal in l. The nodal->modal
        # speed transform (xvander_inv) is pre-folded into Hhat.
        df = jnp.einsum("psxlk,skltz->pxltz", self._Hhat, f)
        # transform back to real space in pitch angle
        df = jnp.einsum("al,pxltz->pxatz", self.Txi, df)

        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        gval = jnp.where(
            self.gauge,
            self._scale * f0[:, idxx, idxa, 0, 0],
            df[:, idxx, idxa, 0, 0],
        )
        df = df.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)

        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return -df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("FieldPartCH.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        return _field_part_gh_diagonal(self, self._Hhat, self._scale)

    @eqx.filter_jit
    @jax.named_scope("FieldPartCH.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None):
        """Block diagonal of operator as (N,M,M) array."""
        return _field_part_gh_block_diagonal(self, self._Hhat, self._scale, fmt, bw)


class FieldParticleScattering(AbstractDKEOperator):
    """Field-particle part of Fokker-Planck Landau collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : LegendrePitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : MaxwellSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened

    """

    field: Field
    speedgrid: MaxwellSpeedGrid
    pitchgrid: UniformPitchAngleGrid
    species: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]
    Txi: jax.Array
    Txi_inv: jax.Array
    C: jax.Array
    _GHhat: jax.Array
    _scale_GH: jax.Array
    _scale_D: jax.Array

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
        coulomb_log=None,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.potentials = potentials
        self.axorder = axorder
        self.gauge = jnp.array(gauge)

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |
        #
        # Three contributions share the species-pair (a, b) loop:
        #  * Rosenbluth G (block-diagonal in modal pitch l)
        #  * Rosenbluth H + dH (block-diagonal in modal pitch l)
        #  * the diagonal-in-pitch speed-space operator (CD)
        x = speedgrid.x
        idxs = jnp.arange(len(species))

        def terms_ab(spa, spb):
            va = spa.v_thermal
            ma = spa.species.mass
            v = x * va
            Fa = spa(v)
            gamma = gamma_ab(spa, spb, lnlambda=coulomb_log)
            vb = spb.v_thermal
            mb = spb.species.mass
            pg = gamma * Fa * 2 * v**2 / va**4
            pH = -2 / va**2 * gamma * Fa
            pdH = -2 * v / va**2 * (1 - ma / mb) * gamma * Fa
            # CD evaluates fb at xq = va/vb x: if va >> vb, fa is "wider" in
            # speed and we sample fb in its tail (xq >> 1).
            xq = va / vb * x
            Dab = orthax.orthvander(
                xq, speedgrid.nx - 1, speedgrid.xrec
            ) * speedgrid.xrec.weight(xq[:, None])
            cd_prefactor = jnp.diag(gamma * Fa * 4 * jnp.pi * ma / mb)
            return pg, pH, pdH, cd_prefactor @ Dab @ speedgrid.xvander_inv

        prefactor_G, prefactor_H, prefactor_dH, self.C = _species_pairs(
            terms_ab, species, species
        )

        self.Txi = orthax.orthvander(
            pitchgrid.xi,
            potentials.legendregrid.nalpha - 1,
            potentials.legendregrid.xirec,
        )
        self.Txi_inv = jnp.linalg.pinv(self.Txi)

        # G and H share an identical nodal<->modal-pitch pipeline differing only
        # in the potential tensor, so they fuse exactly into a single modal
        # tensor. The nodal->modal speed transform (xvander_inv) is pre-folded
        # in, eliminating one einsum per matvec.
        Gabxlk = prefactor_G[:, :, :, None, None] * potentials.ddGxlk
        Habxlk = (
            prefactor_H[:, :, :, None, None] * potentials.Hxlk
            + prefactor_dH[:, :, :, None, None] * potentials.dHxlk
        )
        self._GHhat = jnp.einsum(
            "psxlk,km->psxlm", Gabxlk + Habxlk, speedgrid.xvander_inv
        )

        # gauge scales depend only on operator data, not the input vector.
        self._scale_GH = (
            jnp.mean(jnp.abs(Gabxlk[idxs, idxs]), axis=(2, 3))
            + jnp.mean(jnp.abs(Habxlk[idxs, idxs]), axis=(2, 3))
        )[:, speedgrid.gauge_idx]
        self._scale_D = jnp.mean(jnp.abs(self.C[idxs, idxs]), axis=(2,))[
            :, speedgrid.gauge_idx
        ]

    @eqx.filter_jit
    @jax.named_scope("FieldParticleScattering.mv")
    def mv(self, vector):
        """Matrix vector product."""
        # The Rosenbluth G and H pieces share an identical nodal<->modal-pitch
        # pipeline differing only in the potential tensor, so they are fused into a
        # single tensor ``_GHhat`` (one nodal->modal pitch transform, one tensor
        # apply, one modal->nodal). The diagonal-in-pitch CD piece acts directly in
        # speed space and is added in as one extra einsum.
        f0 = vector
        shp = f0.shape
        shape, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            len(self.species),
            self.axorder,
        )
        f0 = f0.reshape(shape)
        f0 = jnp.moveaxis(f0, caxorder, (0, 1, 2, 3, 4))

        f = jnp.einsum("la,sxatz->sxltz", self.Txi_inv, f0)
        df = jnp.einsum("psxlk,skltz->pxltz", self._GHhat, f)
        df = jnp.einsum("al,pxltz->pxatz", self.Txi, df)
        # CD added directly in speed space
        df = df + jnp.einsum("psyx,sxatz->pyatz", self.C, f0)

        idxa = self.pitchgrid.nalpha // 2
        idxx = self.speedgrid.gauge_idx
        # gauge scale of the sum is the sum of the G+H and D scales
        scale = self._scale_GH + self._scale_D
        gval = jnp.where(
            self.gauge,
            scale * f0[:, idxx, idxa, 0, 0],
            df[:, idxx, idxa, 0, 0],
        )
        df = df.at[:, idxx, idxa, 0, 0].set(gval, unique_indices=True)
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return -df.reshape(shp)

    @eqx.filter_jit
    @jax.named_scope("FieldParticleScattering.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        return _field_part_gh_diagonal(
            self, self._GHhat, self._scale_GH
        ) + _field_part_cd_diagonal(self, self.C, self._scale_D)

    @eqx.filter_jit
    @jax.named_scope("FieldParticleScattering.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        return _field_part_gh_block_diagonal(
            self, self._GHhat, self._scale_GH, fmt, bw
        ) + _field_part_cd_block_diagonal(self, self.C, self._scale_D, fmt, bw)


class FokkerPlanckLandau(AbstractDKEOperator):
    """Fokker-Planck Landau collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    pitchgrid : UniformPitchAngleGrid
        Grid of coordinates in pitch angle.
    speedgrid : MaxwellSpeedGrid
        Grid of coordinates in speed.
    species : list[LocalMaxwellian]
        Species being considered
    background : list[LocalMaxwellian]
        Background species to include in the collision operator without solving for df.
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    p2 : int
        Order of approximation for second derivatives.
    axorder : {"sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"}
        Ordering for variables in f, eg how the 5d array is flattened

    """

    field: Field
    pitchgrid: UniformPitchAngleGrid
    speedgrid: MaxwellSpeedGrid
    species: list[LocalMaxwellian]
    background: list[LocalMaxwellian]
    potentials: RosenbluthPotentials
    p2: int = eqx.field(static=True)
    axorder: str = eqx.field(static=True)
    gauge: Bool[Array, ""]
    operator_weights: jax.Array
    CL: PitchAngleScattering
    CE: EnergyScattering
    CF: FieldParticleScattering

    def __init__(
        self,
        field: Field,
        pitchgrid: UniformPitchAngleGrid,
        speedgrid: MaxwellSpeedGrid,
        species: list[LocalMaxwellian],
        background: list[LocalMaxwellian] | None = None,
        potentials: RosenbluthPotentials | None = None,
        p2: int = 4,
        axorder: str = "sxatz",
        gauge: Bool[ArrayLike, ""] = False,
        operator_weights: jax.Array | None = None,
        coulomb_log=None,
    ):
        assert axorder in ["".join(p) for p in itertools.permutations("sxatz")]
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        if background is None:
            background = []
        self.background = background
        if potentials is None:
            potentials = RosenbluthPotentials(speedgrid, species)
        self.potentials = potentials
        self.p2 = p2
        self.axorder = axorder
        self.gauge = jnp.array(gauge)
        if operator_weights is None:
            operator_weights = jnp.ones(3)
        self.operator_weights = jnp.asarray(operator_weights)

        self.CL = PitchAngleScattering(
            field,
            pitchgrid,
            speedgrid,
            species,
            background,
            p2,
            axorder,
            gauge,
            coulomb_log=coulomb_log,
        )
        self.CE = EnergyScattering(
            field,
            pitchgrid,
            speedgrid,
            species,
            background,
            axorder,
            gauge,
            coulomb_log=coulomb_log,
        )
        self.CF = FieldParticleScattering(
            field,
            pitchgrid,
            speedgrid,
            species,
            potentials,
            axorder,
            gauge,
            coulomb_log=coulomb_log,
        )

    @eqx.filter_jit
    @jax.named_scope("FokkerPlanckLandau.mv")
    def mv(self, vector):
        """Matrix vector product."""
        out1 = self.CL.mv(vector)
        out2 = self.CE.mv(vector)
        out3 = self.CF.mv(vector)
        return (
            self.operator_weights[0] * out1
            + self.operator_weights[1] * out2
            + self.operator_weights[2] * out3
        )

    @eqx.filter_jit
    @jax.named_scope("FokkerPlanckLandau.diagonal")
    def diagonal(self) -> Float[Array, " nf"]:
        """Diagonal of the operator as a 1d array."""
        sizes = {
            "s": len(self.species),
            "x": self.speedgrid.nx,
            "a": self.pitchgrid.nalpha,
            "t": self.field.ntheta,
            "z": self.field.nzeta,
        }
        n1 = np.prod(list(sizes.values()))
        x = jnp.zeros(n1)
        intermediates = [
            lambda x: x + self.operator_weights[0] * self.CL.diagonal(),
            lambda x: x + self.operator_weights[1] * self.CE.diagonal(),
            lambda x: x + self.operator_weights[2] * self.CF.diagonal(),
        ]
        return eqx.internal.scan_trick(lambda x: x, intermediates, x)

    @eqx.filter_jit
    @jax.named_scope("FokkerPlanckLandau.abs_row_sum")
    def abs_row_sum(self) -> Float[Array, " nf"]:
        """L1 norm of each row, sum_j |A_ij|, as a 1d array."""
        # The collision operator is local in (theta, zeta) and its velocity-space
        # block is identical at every spatial point, so the row L1 norm depends
        # only on the (species, speed, pitch) coordinates.  We build that single
        # ``(ns*nx*na, ns*nx*na)`` block once, take its abs row sum, and broadcast
        # the result across the spatial grid (axorder-agnostic).  CL/CE/CF are
        # summed *before* taking absolute values so entries they share (e.g. CF
        # overlaps CL in pitch and CE in speed) are not double counted.

        # The gauge row (a single modified row per species, at one spatial point)
        # is not special-cased here; this uses the generic interior block, which
        # is exact when ``gauge`` is False.
        ns = len(self.species)
        nx = self.speedgrid.nx
        na = self.pitchgrid.nalpha
        w = self.operator_weights
        eye_s = jnp.eye(ns)
        eye_x = jnp.eye(nx)
        eye_a = jnp.eye(na)

        # CL: -nus/2 * D, diagonal in (species, speed), couples pitch
        cl = jnp.einsum(
            "su,xv,sx,ab->sxauvb", eye_s, eye_x, -self.CL.nus / 2, self.CL._D
        )
        # CE: -M, diagonal in (species, pitch), couples speed
        ce = jnp.einsum("su,ab,syv->syauvb", eye_s, eye_a, -self.CE._M)
        # CF: -(GH + CD); dense in (speed, pitch), couples species
        gh = jnp.einsum(
            "Al,psxlk,lB->psxAkB", self.CF.Txi, self.CF._GHhat, self.CF.Txi_inv
        )
        cd = jnp.einsum("psyx,ab->psyaxb", self.CF.C, eye_a)
        # both land as (out_s, in_s, out_x, out_a, in_x, in_a); reorder to
        # (out_s, out_x, out_a, in_s, in_x, in_a) to match CL/CE.
        cf = jnp.transpose(-(gh + cd), (0, 2, 3, 1, 4, 5))

        block = w[0] * cl + w[1] * ce + w[2] * cf
        rsum = jnp.abs(block).sum(axis=(3, 4, 5))  # (ns, nx, na)

        # broadcast the (theta, zeta)-independent velocity row sums onto the
        # full grid and lay them out in the requested axorder.
        _, caxorder = _parse_axorder_shape_4d(
            self.field.ntheta,
            self.field.nzeta,
            self.pitchgrid.nalpha,
            self.speedgrid.nx,
            ns,
            self.axorder,
        )
        df = jnp.broadcast_to(
            rsum[:, :, :, None, None],
            (ns, nx, na, self.field.ntheta, self.field.nzeta),
        )
        df = jnp.moveaxis(df, (0, 1, 2, 3, 4), caxorder)
        return df.flatten()

    @eqx.filter_jit
    @jax.named_scope("FokkerPlanckLandau.block_diagonal")
    def block_diagonal(self, fmt="dense", bw=None) -> Float[Array, "n1 n2 n2"]:
        """Block diagonal of operator as (N,M,M) array."""
        sizes = {
            "s": len(self.species),
            "x": self.speedgrid.nx,
            "a": self.pitchgrid.nalpha,
            "t": self.field.ntheta,
            "z": self.field.nzeta,
        }
        n2 = sizes[self.axorder[-1]]
        n1 = np.prod(list(sizes.values())) // n2
        if fmt == "dense":
            x = jnp.zeros((n1, n2, n2))
        else:
            assert isinstance(bw, int)
            x = jnp.zeros((n1, 2 * bw + 1, n2))
        intermediates = [
            lambda x: x + self.operator_weights[0] * self.CL.block_diagonal(fmt, bw),
            lambda x: x + self.operator_weights[1] * self.CE.block_diagonal(fmt, bw),
            lambda x: x + self.operator_weights[2] * self.CF.block_diagonal(fmt, bw),
        ]
        return eqx.internal.scan_trick(lambda x: x, intermediates, x)
