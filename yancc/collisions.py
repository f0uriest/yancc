"""Collision operators and methods for computing Rosenbluth potentials."""

import functools

import cola
import equinox as eqx
import jax
import jax.numpy as jnp
import monkes
import numpy as np
import orthax
import quadax
from monkes import Field, LocalMaxwellian

from .linalg import BlockOperator
from .utils import lGammainc, lGammaincc
from .velocity_grids import PitchAngleGrid, SpeedGrid


class RosenbluthPotentials(eqx.Module):
    """Thing to calculate Rosenbluth Potentials.

    Parameters
    ----------
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    nL : int
        Number of Legendre modes to use for potentials.
    quad : bool
        Whether to compute potentials using quadrature or incomplete gamma functions
    """

    speedgrid: SpeedGrid
    pitchgrid: PitchAngleGrid
    quad: bool = eqx.field(static=True)
    ddGxlk: jax.Array
    Hxlk: jax.Array
    dHxlk: jax.Array

    def __init__(self, speedgrid, pitchgrid, species, nL=4, quad=True):
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.quad = quad

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

    def rosenbluth_ddG(self, f, a, b):
        """Second derivative of G potential for species indices a, b"""
        # this only knows about a single species,
        # f assumed to be shape(xi, x, theta, zeta)
        # transform from nodal->modal for xi (x already modal)
        f = jnp.einsum("li,iktz->lktz", self.pitchgrid.xivander_inv, f)
        # project onto greens fn of poisson eqn
        ddGl = jnp.einsum("lktz,xlk->lxtz", f, self.ddGxlk[a, b])
        # convert back to real space
        ddG = jnp.einsum("il,lxtz->ixtz", self.pitchgrid.xivander, ddGl)
        return ddG

    def rosenbluth_H(self, f, a, b):
        """H potential for species indices a, b"""
        # this only knows about a single species,
        # f assumed to be shape(xi, x, theta, zeta)
        # transform from nodal->modal for xi (x already modal)
        f = jnp.einsum("li,iktz->lktz", self.pitchgrid.xivander_inv, f)
        # project onto greens fn of poisson eqn
        Hl = jnp.einsum("lktz,xlk->lxtz", f, self.Hxlk[a, b])
        # convert back to real space
        H = jnp.einsum("il,lxtz->ixtz", self.pitchgrid.xivander, Hl)
        return H

    def rosenbluth_dH(self, f, a, b):
        """Derivative of H potential for species indices a, b"""
        # this only knows about a single species,
        # f assumed to be shape(xi, x, theta, zeta)
        # transform from nodal->modal for xi (x already modal)
        f = jnp.einsum("li,iktz->lktz", self.pitchgrid.xivander_inv, f)
        # project onto greens fn of poisson eqn
        dHl = jnp.einsum("lktz,xlk->lxtz", f, self.dHxlk[a, b])
        # convert back to real space
        dH = jnp.einsum("il,lxtz->ixtz", self.pitchgrid.xivander, dHl)
        return dH

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand1(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        return (
            z ** (-l + 1)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand2(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        return (
            z ** (l + 2)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand3(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        return (
            z ** (-l + 3)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _integrand4(self, z, l, k):
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        return (
            z ** (l + 4)
            * orthax.orthval(z, c, self.speedgrid.xrec)
            * self.speedgrid.xrec.weight(z)
        )

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_1(self, x, l, k):
        if self.quad:
            f, info = quadax.quadcc(
                self._integrand1, (x, np.inf), (l, k), order=256, max_ninter=10
            )
            f = eqx.error_if(f, info.status, "I_1 did not converge")
            return f
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.speedgrid.xrec)
        n = jnp.arange(self.speedgrid.nx)
        sgn, lg = lGammaincc(-l / 2 + n / 2 + 1, x**2)
        li, sgn = jax.scipy.special.logsumexp(lg, b=sgn * p, return_sign=True)
        return sgn * jnp.exp(li) / 2

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_2(self, x, l, k):
        if self.quad:
            f, info = quadax.quadcc(
                self._integrand2, (0, x), (l, k), order=256, max_ninter=10
            )
            f = eqx.error_if(f, info.status, "I_2 did not converge")
            return f
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.speedgrid.xrec)
        n = jnp.arange(self.speedgrid.nx)
        sgn, lg = lGammainc(l / 2 + n / 2 + 3 / 2, x**2)
        li, sgn = jax.scipy.special.logsumexp(lg, b=sgn * p, return_sign=True)
        return sgn * jnp.exp(li) / 2

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_3(self, x, l, k):
        if self.quad:
            f, info = quadax.quadcc(
                self._integrand3, (x, np.inf), (l, k), order=256, max_ninter=10
            )
            f = eqx.error_if(f, info.status, "I_3 did not converge")
            return f
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.speedgrid.xrec)
        n = jnp.arange(self.speedgrid.nx)
        sgn, lg = lGammaincc(-l / 2 + n / 2 + 2, x**2)
        li, sgn = jax.scipy.special.logsumexp(lg, b=sgn * p, return_sign=True)
        return sgn * jnp.exp(li) / 2

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _I_4(self, x, l, k):
        if self.quad:
            f, info = quadax.quadcc(
                self._integrand4, (0, x), (l, k), order=256, max_ninter=10
            )
            f = eqx.error_if(f, info.status, "I_4 did not converge")
            return f
        c = jnp.zeros(self.speedgrid.nx).at[k].set(1)
        p = orthax.orth2poly(c, self.speedgrid.xrec)
        n = jnp.arange(self.speedgrid.nx)
        sgn, lg = lGammainc(l / 2 + n / 2 + 5 / 2, x**2)
        li, sgn = jax.scipy.special.logsumexp(lg, b=sgn * p, return_sign=True)
        return sgn * jnp.exp(li) / 2

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_1(self, x, l, k):
        return -self._integrand1(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_2(self, x, l, k):
        return self._integrand2(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_3(self, x, l, k):
        return -self._integrand3(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dI_4(self, x, l, k):
        return self._integrand4(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_1(self, x, l, k):
        return jax.grad(self._dI_1)(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_2(self, x, l, k):
        return jax.grad(self._dI_2)(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_3(self, x, l, k):
        return jax.grad(self._dI_3)(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _ddI_4(self, x, l, k):
        return jax.grad(self._dI_4)(x, l, k)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _Hlk(self, xb, l, k):
        term1 = 1 / xb ** (l + 1) * self._I_2(xb, l, k)
        term2 = xb**l * self._I_1(xb, l, k)
        return (4 * jnp.pi) / (2 * l + 1) * (term1 + term2)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _dHlk(self, xb, l, k):
        term11 = -(l + 1) / xb ** (l + 2) * self._I_2(xb, l, k)
        term12 = 1 / xb ** (l + 1) * self._dI_2(xb, l, k)
        term21 = l * xb ** (l - 1) * self._I_1(xb, l, k)
        term22 = xb**l * self._dI_1(xb, l, k)
        return (4 * jnp.pi) / (2 * l + 1) * (term11 + term12 + term21 + term22)

    @jax.jit
    @functools.partial(jnp.vectorize, excluded=[0])
    def _Glk(self, xb, l, k):
        term1 = xb**l * self._I_3(xb, l, k)
        term2 = -(2 * l - 1) / (2 * l + 3) * xb ** (l + 2) * self._I_1(xb, l, k)
        term3 = -(2 * l - 1) / (2 * l + 3) / xb ** (l + 1) * self._I_4(xb, l, k)
        term4 = 1 / xb ** (l - 1) * self._I_2(xb, l, k)
        return -(4 * jnp.pi) / (4 * l**2 - 1) * (term1 + term2 + term3 + term4)

    @jax.jit
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

    @jax.jit
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


class FokkerPlanckLandau(cola.ops.Sum):
    """Fokker-Planck Landau collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.potentials = potentials

        CL = PitchAngleScattering(field, speedgrid, pitchgrid, species)
        CE = EnergyScattering(field, speedgrid, pitchgrid, species)
        CF = FieldParticleScattering(field, speedgrid, pitchgrid, species, potentials)
        super().__init__(CL, CE, CF)


class FieldParticleScattering(cola.ops.Sum):
    """Field-particle part of Fokker-Planck Landau collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        approx_rdot=False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.potentials = potentials

        CG = _CG(field, speedgrid, pitchgrid, species, potentials, approx_rdot)
        CH = _CH(field, speedgrid, pitchgrid, species, potentials, approx_rdot)
        CD = _CD(field, speedgrid, pitchgrid, species, potentials, approx_rdot)
        super().__init__(CG, CH, CD)


class PitchAngleScattering(cola.ops.Kronecker):
    """Pitch angle scattering collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        approx_rdot: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species

        Is = cola.ops.Identity((len(species), len(species)), pitchgrid.xi.dtype)
        Ix = cola.ops.Dense(speedgrid.xvander)
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
        x = speedgrid.x
        L = cola.ops.Dense(pitchgrid.L)

        nus = []
        for spa in species:
            nu = 0.0
            for spb in species:
                nu += monkes._species.nuD_ab(spa, spb, x * spa.v_thermal)
            nus.append(nu / 2)
        nus = cola.ops.Diagonal(jnp.array(nus).flatten()) @ cola.ops.Kronecker(Is, Ix)
        if approx_rdot:
            Ms = (nus, L, It, Iz)
        else:
            Ms = (nus, L, cola.ops.Kronecker(It, Iz))
        super().__init__(*Ms)


class EnergyScattering(cola.ops.Kronecker):
    """Energy scattering collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        approx_rdot: bool = False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species

        Ix = cola.ops.Dense(speedgrid.xvander)
        Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
        Dx = cola.ops.Dense(speedgrid.xvander @ speedgrid.Dx)
        D2x = cola.ops.Dense(speedgrid.xvander @ speedgrid.Dx @ speedgrid.Dx)
        x = speedgrid.x

        out = []
        for spa in species:
            vta = spa.v_thermal
            v = x * vta
            term1 = 0.0
            term2 = 0.0
            term3 = 0.0
            for spb in species:
                nupar = monkes._species.nupar_ab(spa, spb, v)
                nuD = monkes._species.nuD_ab(spa, spb, v)
                gamma = monkes._species.gamma_ab(spa, spb, v)
                ma, mb = spa.species.mass, spb.species.mass
                vtb = spb.v_thermal
                term1 += nupar * x**2 / 2
                term2 += nuD * x - nupar * (x * vta / vtb) ** 2 * (1 - ma / mb) * x
                term3 += 4 * jnp.pi * gamma * ma / mb * spb(v)
            out.append(
                cola.ops.Diagonal(term1) @ D2x
                + cola.ops.Diagonal(term2) @ Dx
                + cola.ops.Diagonal(term3) @ Ix
            )
        out = cola.ops.BlockDiag(*out)
        if approx_rdot:
            Ms = (out, Ixi, It, Iz)
        else:
            Ms = (out, Ixi, cola.ops.Kronecker(It, Iz))
        super().__init__(*Ms)


class _CD(cola.ops.Kronecker):
    """Diagonal part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        approx_rdot=False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.potentials = potentials

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |
        x = speedgrid.x
        Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)

        C = []
        Ca = []
        for a, spa in enumerate(species):
            va = spa.v_thermal
            ma = spa.species.mass
            v = x * va
            Fa = spa(v)
            for b, spb in enumerate(species):
                gamma = monkes._species.gamma_ab(spa, spb, v)
                vb = spb.v_thermal
                mb = spb.species.mass
                # need to evaluate fb on the speed grid for fa
                # if va >> vb, then fa is "wider" in speed, and we're evaluating in
                # the tail of fb, ie xq >> 1, so xq = va/vb x
                xq = va / vb * x
                # matrix to evaluate fb at xq
                Dab = cola.ops.Dense(
                    orthax.orthvander(xq, speedgrid.nx - 1, speedgrid.xrec)
                    * speedgrid.xrec.weight(xq[:, None])
                )
                prefactor = cola.ops.Diagonal(gamma * Fa * 4 * jnp.pi * ma / mb)
                CDab = prefactor @ Dab
                Ca.append(CDab)
            C.append(Ca)
            Ca = []
        C = -BlockOperator(C)
        if approx_rdot:
            Ms = (C, Ixi, It, Iz)
        else:
            Ms = (C, Ixi, cola.ops.Kronecker(It, Iz))
        super().__init__(*Ms)


class _CG(cola.ops.Kronecker):
    """Rosenbluth G part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        approx_rdot=False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.potentials = potentials

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |

        x = speedgrid.x
        Ix = cola.ops.Identity((x.size, x.size), x.dtype)
        Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
        # G is in modal basis in xi
        # these go from nodal xi to modal l, and back
        # G is effectively block diagonal in l
        Txi_inv = cola.ops.Kronecker(Ix, cola.ops.Dense(pitchgrid.xivander_inv))
        Txi = cola.ops.Kronecker(Ix, cola.ops.Dense(pitchgrid.xivander))

        C = []
        Ca = []
        for a, spa in enumerate(species):
            va = spa.v_thermal
            v = x * va
            Fa = spa(v)
            for b, spb in enumerate(species):
                gamma = monkes._species.gamma_ab(spa, spb, v)
                vb = spb.v_thermal
                ddGxlk = potentials.ddGxlk[a, b]
                # its diagonal in legendre index (axis position 1)
                ddGxkli = jax.vmap(jax.vmap(jnp.diag, in_axes=0), in_axes=2)(ddGxlk)
                ddGxilk = jnp.swapaxes(ddGxkli, 3, 1)
                ddGxikl = jnp.swapaxes(ddGxilk, 2, 3)
                ddG = ddGxikl.reshape(
                    (speedgrid.nx * pitchgrid.nxi, speedgrid.nx * pitchgrid.nxi)
                ).T
                ddG = cola.ops.Dense(ddG)
                ddGab = (
                    Txi @ ddG @ Txi_inv
                )  # project onto legendre, apply potentials, and transform back
                prefactor = cola.ops.Diagonal(gamma * Fa * 2 * v**2 * vb**2 / va**4)
                CGab = cola.ops.Kronecker(prefactor, Ixi) @ ddGab
                Ca.append(CGab)
            C.append(Ca)
            Ca = []
        C = -BlockOperator(C)
        super().__init__(C, It, Iz)


class _CH(cola.ops.Kronecker):
    """Rosenbluth H part of the field particle collision operator.

    Parameters
    ----------
    field : Field
        Magnetic field information
    speedgrid : SpeedGrid
        Grid of coordinates in speed.
    pitchgrid : PitchAngleGrid
        Grid of coordinates in pitch angle.
    species : list[LocalMaxwellian]
        Species being considered
    potentials : RosenbluthPotentials
        Thing for calculating Rosenbluth potentials.
    approx_rdot : bool
        Whether to approximate the surface terms by decoupling theta and zeta. Should
        be False for the main operator, but setting to True for the preconditioner can
        improve performance.

    """

    def __init__(
        self,
        field: Field,
        speedgrid: SpeedGrid,
        pitchgrid: PitchAngleGrid,
        species: list[LocalMaxwellian],
        potentials: RosenbluthPotentials,
        approx_rdot=False,
    ):
        self.field = field
        self.speedgrid = speedgrid
        self.pitchgrid = pitchgrid
        self.species = species
        self.potentials = potentials

        # field particle collision operator has block structure
        # | C_aa  C_ab | | f_a | = | R_a |
        # | C_ba  C_bb | | f_b |   | R_b |

        x = speedgrid.x
        Ix = cola.ops.Identity((x.size, x.size), x.dtype)
        Ixi = cola.ops.Identity((pitchgrid.nxi, pitchgrid.nxi), pitchgrid.xi.dtype)
        It = cola.ops.Identity((field.ntheta, field.ntheta), field.theta.dtype)
        Iz = cola.ops.Identity((field.nzeta, field.nzeta), field.zeta.dtype)
        # H is in modal basis in xi
        # these go from nodal xi to modal l, and back
        # H is effectively block diagonal in l
        Txi_inv = cola.ops.Kronecker(Ix, cola.ops.Dense(pitchgrid.xivander_inv))
        Txi = cola.ops.Kronecker(Ix, cola.ops.Dense(pitchgrid.xivander))

        C = []
        Ca = []
        for a, spa in enumerate(species):
            va = spa.v_thermal
            ma = spa.species.mass
            v = x * va
            Fa = spa(v)
            for b, spb in enumerate(species):
                gamma = monkes._species.gamma_ab(spa, spb, v)
                vb = spb.v_thermal
                mb = spb.species.mass
                Hxlk = potentials.Hxlk[a, b]
                # its diagonal in legendre index (axis position 1)
                Hxkli = jax.vmap(jax.vmap(jnp.diag, in_axes=0), in_axes=2)(Hxlk)
                Hxilk = jnp.swapaxes(Hxkli, 3, 1)
                Hxikl = jnp.swapaxes(Hxilk, 2, 3)
                H = Hxikl.reshape(
                    (speedgrid.nx * pitchgrid.nxi, speedgrid.nx * pitchgrid.nxi)
                ).T
                H = cola.ops.Dense(H)
                Hab = (
                    Txi @ H @ Txi_inv
                )  # project onto legendre, apply potentials, and transform back
                dHxlk = potentials.dHxlk[a, b]
                # its diagonal in legendre index (axis position 1)
                dHxkli = jax.vmap(jax.vmap(jnp.diag, in_axes=0), in_axes=2)(dHxlk)
                dHxilk = jnp.swapaxes(dHxkli, 3, 1)
                dHxikl = jnp.swapaxes(dHxilk, 2, 3)
                dH = dHxikl.reshape(
                    (speedgrid.nx * pitchgrid.nxi, speedgrid.nx * pitchgrid.nxi)
                ).T
                dH = cola.ops.Dense(dH)
                dHab = (
                    Txi @ dH @ Txi_inv
                )  # project onto legendre, apply potentials, and transform back
                H_prefactor = cola.ops.Diagonal(-2 * vb**2 / va**2 * gamma * Fa)
                dH_prefactor = cola.ops.Diagonal(
                    -2 * v / va**2 * vb**2 * (1 - ma / mb) * gamma * Fa
                )
                CHab = (
                    cola.ops.Kronecker(H_prefactor, Ixi) @ Hab
                    + cola.ops.Kronecker(dH_prefactor, Ixi) @ dHab
                )
                Ca.append(CHab)
            C.append(Ca)
            Ca = []
        C = -BlockOperator(C)
        super().__init__(C, It, Iz)
