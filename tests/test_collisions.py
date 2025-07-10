"""Tests for collision operator."""

import jax.numpy as jnp
import numpy as np
import orthax
import pytest
import sympy

from yancc.collisions import (
    EnergyScattering,
    FieldPartCD,
    FieldPartCG,
    FieldPartCH,
    FokkerPlanckLandau,
    RosenbluthPotentials,
)
from yancc.field import Field
from yancc.species import GlobalMaxwellian, Hydrogen, gamma_ab
from yancc.velocity_grids import SpeedGrid, UniformPitchAngleGrid

from .conftest import (
    _compute_CD_sympy,
    _compute_CE_sympy,
    _compute_CG_sympy,
    _compute_CH_sympy,
)


def test_CE_single_species_vs_sympy():
    x = sympy.symbols("x", real=True)
    v = sympy.symbols("v", real=True, positive=True)
    na, nb, ma = sympy.symbols("n_a n_b m_a", real=True)
    va = sympy.symbols("v_a", real=True)
    Gamma_aa = sympy.symbols("Gamma_aa", real=True)
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    Fa = na / (pi32 * va**3) * sympy.exp(-(x**2))
    fa = (1 - x + x**2) * sympy.exp(-(x**2))
    CEaa = _compute_CE_sympy(Fa, fa, x, v, va, va, ma, ma, nb, Gamma_aa)

    ni = 5e19
    ti = 1000

    ions1 = GlobalMaxwellian(
        Hydrogen, lambda x: ti * (1 - x**2), lambda x: ni * (1 - x**4)
    ).localize(0.5)

    species = [
        ions1,
    ]

    # just need a dummy field for this
    nt = 1
    nz = 1
    field = Field(
        rho=0.5,
        B_sup_t=np.ones((nt, nz)),
        B_sup_z=np.ones((nt, nz)),
        B_sub_t=np.ones((nt, nz)),
        B_sub_z=np.ones((nt, nz)),
        Bmag=np.ones((nt, nz)),
        sqrtg=np.ones((nt, nz)),
        psi_r=1.0,
        iota=1.0,
        R_major=10.0,
        a_minor=1.0,
    )
    speedgrid = SpeedGrid(10)
    pitchgrid = UniformPitchAngleGrid(11)
    CE = EnergyScattering(field, pitchgrid, speedgrid, species)
    gamma_aa_jax = gamma_ab(species[0], species[0])

    CEsympy = np.array(
        [
            CEaa.evalf(
                subs={
                    v: xi_ * species[0].v_thermal,
                    va: species[0].v_thermal,
                    na: species[0].density,
                    nb: species[0].density,
                    Gamma_aa: gamma_aa_jax,
                }
            )
            for xi_ in speedgrid.x
        ],
        dtype=np.float64,
    )

    ffa = np.array([fa.evalf(subs={x: xi_}) for xi_ in speedgrid.x], dtype=np.float64)

    f = np.ones((1, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
    f[0] *= ffa[:, None, None, None]

    CEjax = -CE.mv(f)[
        0, :, 0, 0, 0
    ]  # collision operator has a minus sign in overall DKE

    np.testing.assert_allclose(CEjax, CEsympy)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_CF_single_species_vs_sympy(l):
    x = sympy.symbols("x", real=True)
    va, ma, v, na, Gamma_aa = sympy.symbols("v_a m_a v n_a Gamma_aa", real=True)
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    Fa = na / (pi32 * va**3) * sympy.exp(-(x**2))

    fa = (1 + x) * sympy.exp(-(x**2))
    CDaa = _compute_CD_sympy(Fa, fa, x, v, va, va, ma, ma, Gamma_aa)
    CHaa = _compute_CH_sympy(Fa, fa, x, l, v, va, va, ma, ma, Gamma_aa)
    CGaa = _compute_CG_sympy(Fa, fa, x, l, v, va, va, ma, ma, Gamma_aa)

    ni = 5e19
    ti = 1000

    ions1 = GlobalMaxwellian(
        Hydrogen, lambda x: ti * (1 - x**2), lambda x: ni * (1 - x**4)
    ).localize(0.5)

    species = [
        ions1,
    ]

    # just need a dummy field for this
    nt = 1
    nz = 1
    field = Field(
        rho=0.5,
        B_sup_t=np.ones((nt, nz)),
        B_sup_z=np.ones((nt, nz)),
        B_sub_t=np.ones((nt, nz)),
        B_sub_z=np.ones((nt, nz)),
        Bmag=np.ones((nt, nz)),
        sqrtg=np.ones((nt, nz)),
        psi_r=1.0,
        iota=1.0,
        R_major=10.0,
        a_minor=1.0,
    )
    speedgrid = SpeedGrid(10)
    pitchgrid = UniformPitchAngleGrid(11)
    potentials = RosenbluthPotentials(speedgrid, pitchgrid, species, nL=6)
    gamma_aa_jax = gamma_ab(species[0], species[0])
    CD = FieldPartCD(field, pitchgrid, speedgrid, species, potentials)
    CH = FieldPartCH(field, pitchgrid, speedgrid, species, potentials)
    CG = FieldPartCG(field, pitchgrid, speedgrid, species, potentials)

    CHsympy = (
        np.array(
            [
                CHaa.evalf(
                    subs={
                        v: xi_ * species[0].v_thermal,
                        va: species[0].v_thermal,
                        na: species[0].density,
                        Gamma_aa: gamma_aa_jax,
                    }
                )
                for xi_ in speedgrid.x
            ]
        )
        .astype(np.complex128)
        .real
    )
    CGsympy = (
        np.array(
            [
                CGaa.evalf(
                    subs={
                        v: xi_ * species[0].v_thermal,
                        va: species[0].v_thermal,
                        na: species[0].density,
                        Gamma_aa: gamma_aa_jax,
                    }
                )
                for xi_ in speedgrid.x
            ]
        )
        .astype(np.complex128)
        .real
    )
    CDsympy = (
        np.array(
            [
                CDaa.evalf(
                    subs={
                        v: xi_ * species[0].v_thermal,
                        va: species[0].v_thermal,
                        na: species[0].density,
                        Gamma_aa: gamma_aa_jax,
                    }
                )
                for xi_ in speedgrid.x
            ]
        )
        .astype(np.complex128)
        .real
    )
    ffa = np.array([fa.evalf(subs={x: xi_}) for xi_ in speedgrid.x], dtype=np.float64)

    f = np.ones((1, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
    f[0] *= (
        ffa[:, None, None, None]
        * orthax.orthval(
            pitchgrid.xi,
            jnp.zeros(potentials.legendregrid.nxi).at[l].set(1.0),
            potentials.legendregrid.xirec,
        )[None, :, None, None]
    )

    # collision operator has a minus sign in overall DKE
    ch = -CH.mv(f)
    cg = -CG.mv(f)
    cd = -CD.mv(f)

    ch = jnp.einsum("la,sxatz->sxltz", potentials.Txi_inv, ch)[0]
    cg = jnp.einsum("la,sxatz->sxltz", potentials.Txi_inv, cg)[0]
    cd = jnp.einsum("la,sxatz->sxltz", potentials.Txi_inv, cd)[0]

    # potentials are diagonal in legendre index, so outputs for idx != l should be 0
    np.testing.assert_allclose(ch[:, :l, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(ch[:, l + 1 :, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(cg[:, :l, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(cg[:, l + 1 :, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(cd[:, :l, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(cd[:, l + 1 :, :, :], 0, atol=1e-10)

    np.testing.assert_allclose(ch[:, l, 0, 0], CHsympy, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(cg[:, l, 0, 0], CGsympy, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(cd[:, l, 0, 0], CDsympy, rtol=2e-6, atol=1e-8)


def test_verify_collision_null_single_species():
    """Check the null space of single species collision operator."""
    speedgrid = SpeedGrid(5)
    pitchgrid = UniformPitchAngleGrid(129)

    # just need a dummy field for this
    nt = 1
    nz = 1
    field = Field(
        rho=0.5,
        B_sup_t=np.ones((nt, nz)),
        B_sup_z=np.ones((nt, nz)),
        B_sub_t=np.ones((nt, nz)),
        B_sub_z=np.ones((nt, nz)),
        Bmag=np.ones((nt, nz)),
        sqrtg=np.ones((nt, nz)),
        psi_r=1.0,
        iota=1.0,
        R_major=10.0,
        a_minor=1.0,
    )
    ni = 5e19
    ti = 1000
    ions1 = GlobalMaxwellian(
        Hydrogen, lambda x: ti * (1 - x**2), lambda x: ni * (1 - x**4)
    ).localize(0.5)

    R = RosenbluthPotentials(speedgrid, pitchgrid, [ions1], quad=False)
    C = FokkerPlanckLandau(field, pitchgrid, speedgrid, [ions1], R)
    shape = (1, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta)
    x = speedgrid.x
    xi = pitchgrid.xi

    # C acting on maxwellian = 0
    ff = np.exp(-(x**2))[None, :, None, None, None]
    f = np.ones(shape) * ff
    cf = C.mv(f.flatten()).reshape(f.shape)
    np.testing.assert_allclose(cf, 0, atol=1e-5)

    # C acting on v*maxwellian = 0
    ff = (x * np.exp(-(x**2)))[None, :, None, None, None] * xi[
        None, None, :, None, None
    ]
    f = np.ones(shape) * ff
    cf = C.mv(f.flatten()).reshape(f.shape)
    # need looser tolerance here bc finite differences in pitch angle are less
    # accurate than spectral derivatives in speed
    np.testing.assert_allclose(cf, 0, atol=1e-3)

    # C acting on v^2*maxwellian = 0
    ff = x**2 * np.exp(-(x**2))
    f = (
        np.ones((1, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
        * ff[None, :, None, None, None]
    )
    cf = C.mv(f.flatten()).reshape(f.shape)
    np.testing.assert_allclose(cf, 0, atol=1e-7)

    es = np.linalg.eigvals(C.as_matrix())
    # should have purely real eigvals
    np.testing.assert_allclose(es.imag, 0, atol=1e-7)
    # should all be positive, within fudge factor for zeros
    np.testing.assert_array_less(-1e-14 * es.real.max(), es.real)
    # should have a null space of dimension 3*nt*nz
    # maxwellian, v*maxwellian, v^2*maxwellian
    assert sum(np.abs(es) < 1e-14 * np.max(np.abs(es))) == 3 * nt * nz
