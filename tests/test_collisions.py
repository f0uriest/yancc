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
from yancc.species import JOULE_PER_EV, GlobalMaxwellian, Hydrogen, gamma_ab
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

from .conftest import (
    _compute_CDab_sympy,
    _compute_CEab_sympy,
    _compute_CGab_sympy,
    _compute_CHab_sympy,
    _eval_f,
)


def test_CE_single_species_vs_sympy(dummy_field, xigrid, xgrid, species1):
    field = dummy_field
    speedgrid = xgrid
    pitchgrid = xigrid
    species = species1
    na, nb, ma, v, va = sympy.symbols("n_a n_b m_a v v_a", real=True, positive=True)
    Gamma_aa = sympy.symbols("Gamma_aa", real=True)
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    x = v / va
    Fa = na / (pi32 * va**3) * sympy.exp(-(x**2))
    fa = (1 - x + x**2) * sympy.exp(-(x**2))

    CEaa = _compute_CEab_sympy(Fa, fa, v, va, va, ma, ma, na, Gamma_aa)

    CE = EnergyScattering(field, pitchgrid, speedgrid, species)
    gamma_aa_jax = gamma_ab(species[0], species[0])

    subs = {
        va: species[0].v_thermal,
        na: species[0].density,
        nb: species[0].density,
        Gamma_aa: gamma_aa_jax,
    }

    CEsympy = _eval_f(CEaa, v, speedgrid.x * species[0].v_thermal, subs)
    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)

    f = np.ones((1, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
    f[0] *= ffa[:, None, None, None]

    CEjax = -CE.mv(f)[
        0, :, 0, 0, 0
    ]  # collision operator has a minus sign in overall DKE

    np.testing.assert_allclose(CEjax, CEsympy)


def test_CE_2_species_vs_sympy(dummy_field, xigrid, xgrid, species2):
    field = dummy_field
    speedgrid = xgrid
    pitchgrid = xigrid
    species = species2

    v = sympy.symbols("v", real=True, positive=True)
    na, nb, ma, mb, Ta, Tb = sympy.symbols(
        "n_a n_b m_a m_b T_a T_b", real=True, positive=True
    )
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    vta = sympy.sqrt(2 * Ta / ma)
    vtb = sympy.sqrt(2 * Tb / mb)
    Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb = sympy.symbols(
        "Gamma_aa Gamma_ab Gamma_ba Gamma_bb", real=True
    )

    xa = v / vta
    xb = v / vtb

    Fa = na / (pi32 * vta**3) * sympy.exp(-(xa**2))
    Fb = nb / (pi32 * vtb**3) * sympy.exp(-(xb**2))

    fa = (1 - v + 3 * v**2) * sympy.exp(-(xa**2))
    fb = (4 + v - 2 * v**2) * sympy.exp(-(xb**2))

    CEaa = _compute_CEab_sympy(Fa, fa, v, vta, vta, ma, ma, na, Gamma_aa)
    CEab = _compute_CEab_sympy(Fb, fa, v, vta, vtb, ma, mb, nb, Gamma_ab)
    CEba = _compute_CEab_sympy(Fa, fb, v, vtb, vta, mb, ma, na, Gamma_ba)
    CEbb = _compute_CEab_sympy(Fb, fb, v, vtb, vtb, mb, mb, nb, Gamma_bb)

    CE = EnergyScattering(field, pitchgrid, speedgrid, species)
    gamma_aa_jax = gamma_ab(species[0], species[0])
    gamma_ab_jax = gamma_ab(species[0], species[1])
    gamma_ba_jax = gamma_ab(species[1], species[0])
    gamma_bb_jax = gamma_ab(species[1], species[1])

    subs = {
        na: species[0].density,
        nb: species[1].density,
        ma: species[0].species.mass,
        mb: species[1].species.mass,
        Ta: species[0].temperature * JOULE_PER_EV,
        Tb: species[1].temperature * JOULE_PER_EV,
        Gamma_aa: gamma_aa_jax,
        Gamma_ab: gamma_ab_jax,
        Gamma_ba: gamma_ba_jax,
        Gamma_bb: gamma_bb_jax,
    }

    CEaa_sympy = _eval_f(CEaa, v, speedgrid.x * species[0].v_thermal, subs)
    CEab_sympy = _eval_f(CEab, v, speedgrid.x * species[0].v_thermal, subs)
    CEba_sympy = _eval_f(CEba, v, speedgrid.x * species[1].v_thermal, subs)
    CEbb_sympy = _eval_f(CEbb, v, speedgrid.x * species[1].v_thermal, subs)

    CEa_sympy = CEaa_sympy + CEab_sympy
    CEb_sympy = CEba_sympy + CEbb_sympy

    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)
    ffb = _eval_f(fb, v, speedgrid.x * species[1].v_thermal, subs)
    f = np.ones((2, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
    f[0] *= ffa[:, None, None, None]
    f[1] *= ffb[:, None, None, None]
    CE_jax = -CE.mv(f)
    CEa_jax = CE_jax[0, :, 0, 0, 0]
    CEb_jax = CE_jax[1, :, 0, 0, 0]

    np.testing.assert_allclose(CEa_jax, CEa_sympy, rtol=1e-10)
    np.testing.assert_allclose(CEb_jax, CEb_sympy, rtol=1e-10)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_CD_single_species_vs_sympy(l, dummy_field, xigrid, potentials1):
    potentials = potentials1
    field = dummy_field
    speedgrid = potentials.speedgrid
    pitchgrid = xigrid
    species = potentials.species

    va, ma, v, na, Gamma_aa = sympy.symbols("v_a m_a v n_a Gamma_aa", real=True)
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    x = v / va
    Fa = na / (pi32 * va**3) * sympy.exp(-(x**2))
    fa = (1 + x) * sympy.exp(-(x**2))
    CDaa = _compute_CDab_sympy(Fa, fa, ma, ma, Gamma_aa)

    gamma_aa_jax = gamma_ab(species[0], species[0])
    CD = FieldPartCD(field, pitchgrid, speedgrid, species, potentials)
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nxi - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)

    subs = {
        va: species[0].v_thermal,
        na: species[0].density,
        Gamma_aa: gamma_aa_jax,
    }

    CDsympy = _eval_f(CDaa, v, speedgrid.x * species[0].v_thermal, subs)
    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)

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
    cd = -CD.mv(f)
    cd = jnp.einsum("la,sxatz->sxltz", Txi_inv, cd)[0]

    # potentials are diagonal in legendre index, so outputs for idx != l should be 0
    np.testing.assert_allclose(cd[:, :l, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(cd[:, l + 1 :, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(cd[:, l, 0, 0], CDsympy, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_CD_2_species_vs_sympy(l, dummy_field, xigrid, potential_gamma):
    potentials = potential_gamma
    field = dummy_field
    speedgrid = potentials.speedgrid
    pitchgrid = xigrid
    species = potentials.species

    v = sympy.symbols("v", real=True, positive=True)
    na, nb, ma, mb, Ta, Tb = sympy.symbols(
        "n_a n_b m_a m_b T_a T_b", real=True, positive=True
    )
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    vta = sympy.sqrt(2 * Ta / ma)
    vtb = sympy.sqrt(2 * Tb / mb)
    Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb = sympy.symbols(
        "Gamma_aa Gamma_ab Gamma_ba Gamma_bb", real=True
    )
    xa = v / vta
    xb = v / vtb
    Fa = na / (pi32 * vta**3) * sympy.exp(-(xa**2))
    Fb = nb / (pi32 * vtb**3) * sympy.exp(-(xb**2))
    fa = (1 - xa + 3 * xa**2) * sympy.exp(-(xa**2))
    fb = (4 + xb - 2 * xb**2) * sympy.exp(-(xb**2))

    CD = FieldPartCD(field, pitchgrid, speedgrid, species, potentials)
    gamma_aa_jax = gamma_ab(species[0], species[0])
    gamma_ab_jax = gamma_ab(species[0], species[1])
    gamma_ba_jax = gamma_ab(species[1], species[0])
    gamma_bb_jax = gamma_ab(species[1], species[1])
    subs = {
        na: float(species[0].density),
        nb: float(species[1].density),
        ma: float(species[0].species.mass),
        mb: float(species[1].species.mass),
        Ta: float(species[0].temperature * JOULE_PER_EV),
        Tb: float(species[1].temperature * JOULE_PER_EV),
        Gamma_aa: float(gamma_aa_jax),
        Gamma_ab: float(gamma_ab_jax),
        Gamma_ba: float(gamma_ba_jax),
        Gamma_bb: float(gamma_bb_jax),
    }

    CDaa = _compute_CDab_sympy(Fa, fa, ma, ma, Gamma_aa)
    CDab = _compute_CDab_sympy(Fa, fb, ma, mb, Gamma_ab)
    CDba = _compute_CDab_sympy(Fb, fa, mb, ma, Gamma_ba)
    CDbb = _compute_CDab_sympy(Fb, fb, mb, mb, Gamma_bb)
    CDaa_sympy = _eval_f(CDaa, v, speedgrid.x * species[0].v_thermal, subs)
    CDab_sympy = _eval_f(CDab, v, speedgrid.x * species[0].v_thermal, subs)
    CDba_sympy = _eval_f(CDba, v, speedgrid.x * species[1].v_thermal, subs)
    CDbb_sympy = _eval_f(CDbb, v, speedgrid.x * species[1].v_thermal, subs)

    CDa_sympy = CDaa_sympy + CDab_sympy
    CDb_sympy = CDba_sympy + CDbb_sympy

    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)
    ffb = _eval_f(fb, v, speedgrid.x * species[1].v_thermal, subs)
    f = np.ones((2, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
    T = orthax.orthval(
        pitchgrid.xi,
        jnp.zeros(potentials.legendregrid.nxi).at[l].set(1.0),
        potentials.legendregrid.xirec,
    )[None, :, None, None]
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nxi - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)
    f[0] *= ffa[:, None, None, None] * T
    f[1] *= ffb[:, None, None, None] * T
    CD_jax = -CD.mv(f.flatten()).reshape(f.shape)
    CD_jax = jnp.einsum("la,sxatz->sxltz", Txi_inv, CD_jax)
    CDa_jax = CD_jax[0, :, :, 0, 0]
    CDb_jax = CD_jax[1, :, :, 0, 0]
    np.testing.assert_allclose(CDa_jax[:, l], CDa_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(CDb_jax[:, l], CDb_sympy, rtol=1e-10, atol=0)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_CH_single_species_vs_sympy(l, dummy_field, xigrid, potentials1):
    potentials = potentials1
    field = dummy_field
    speedgrid = potentials.speedgrid
    pitchgrid = xigrid
    species = potentials.species

    va, ma, v, na, Gamma_aa = sympy.symbols("v_a m_a v n_a Gamma_aa", real=True)
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    x = v / va
    Fa = na / (pi32 * va**3) * sympy.exp(-(x**2))
    fa = (1 + x) * sympy.exp(-(x**2))
    CHaa = _compute_CHab_sympy(Fa, fa, l, v, va, va, ma, ma, Gamma_aa)

    gamma_aa_jax = gamma_ab(species[0], species[0])
    CH = FieldPartCH(field, pitchgrid, speedgrid, species, potentials)
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nxi - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)

    subs = {
        va: species[0].v_thermal,
        na: species[0].density,
        Gamma_aa: gamma_aa_jax,
    }

    CHsympy = _eval_f(CHaa, v, speedgrid.x * species[0].v_thermal, subs)
    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)

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
    ch = jnp.einsum("la,sxatz->sxltz", Txi_inv, ch)[0]

    # potentials are diagonal in legendre index, so outputs for idx != l should be 0
    np.testing.assert_allclose(ch[:, :l, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(ch[:, l + 1 :, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(ch[:, l, 0, 0], CHsympy, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_CH_2_species_vs_sympy(l, dummy_field, xigrid, potential_gamma):
    potentials = potential_gamma
    field = dummy_field
    speedgrid = potentials.speedgrid
    pitchgrid = xigrid
    species = potentials.species

    v = sympy.symbols("v", real=True, positive=True)
    na, nb, ma, mb, Ta, Tb = sympy.symbols(
        "n_a n_b m_a m_b T_a T_b", real=True, positive=True
    )
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    vta = sympy.sqrt(2 * Ta / ma)
    vtb = sympy.sqrt(2 * Tb / mb)
    Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb = sympy.symbols(
        "Gamma_aa Gamma_ab Gamma_ba Gamma_bb", real=True
    )
    xa = v / vta
    xb = v / vtb
    Fa = na / (pi32 * vta**3) * sympy.exp(-(xa**2))
    Fb = nb / (pi32 * vtb**3) * sympy.exp(-(xb**2))
    fa = (1 - xa + 3 * xa**2) * sympy.exp(-(xa**2))
    fb = (4 + xb - 2 * xb**2) * sympy.exp(-(xb**2))

    CH = FieldPartCH(field, pitchgrid, speedgrid, species, potentials)
    gamma_aa_jax = gamma_ab(species[0], species[0])
    gamma_ab_jax = gamma_ab(species[0], species[1])
    gamma_ba_jax = gamma_ab(species[1], species[0])
    gamma_bb_jax = gamma_ab(species[1], species[1])
    subs = {
        na: float(species[0].density),
        nb: float(species[1].density),
        ma: float(species[0].species.mass),
        mb: float(species[1].species.mass),
        Ta: float(species[0].temperature * JOULE_PER_EV),
        Tb: float(species[1].temperature * JOULE_PER_EV),
        Gamma_aa: float(gamma_aa_jax),
        Gamma_ab: float(gamma_ab_jax),
        Gamma_ba: float(gamma_ba_jax),
        Gamma_bb: float(gamma_bb_jax),
    }

    CHaa = _compute_CHab_sympy(Fa, fa, l, v, vta, vta, ma, ma, Gamma_aa)
    CHab = _compute_CHab_sympy(Fa, fb, l, v, vta, vtb, ma, mb, Gamma_ab)
    CHba = _compute_CHab_sympy(Fb, fa, l, v, vtb, vta, mb, ma, Gamma_ba)
    CHbb = _compute_CHab_sympy(Fb, fb, l, v, vtb, vtb, mb, mb, Gamma_bb)
    CHaa_sympy = _eval_f(CHaa, v, speedgrid.x * species[0].v_thermal, subs)
    CHab_sympy = _eval_f(CHab, v, speedgrid.x * species[0].v_thermal, subs)
    CHba_sympy = _eval_f(CHba, v, speedgrid.x * species[1].v_thermal, subs)
    CHbb_sympy = _eval_f(CHbb, v, speedgrid.x * species[1].v_thermal, subs)
    CHa_sympy = CHaa_sympy + CHab_sympy
    CHb_sympy = CHba_sympy + CHbb_sympy

    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)
    ffb = _eval_f(fb, v, speedgrid.x * species[1].v_thermal, subs)
    f = np.ones((2, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
    T = orthax.orthval(
        pitchgrid.xi,
        jnp.zeros(potentials.legendregrid.nxi).at[l].set(1.0),
        potentials.legendregrid.xirec,
    )[None, :, None, None]
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nxi - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)
    f[0] *= ffa[:, None, None, None] * T
    f[1] *= ffb[:, None, None, None] * T
    CH_jax = -CH.mv(f.flatten()).reshape(f.shape)
    CH_jax = jnp.einsum("la,sxatz->sxltz", Txi_inv, CH_jax)
    CHa_jax = CH_jax[0, :, :, 0, 0]
    CHb_jax = CH_jax[1, :, :, 0, 0]
    np.testing.assert_allclose(CHa_jax[:, l], CHa_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(CHb_jax[:, l], CHb_sympy, rtol=1e-8, atol=0)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_CG_single_species_vs_sympy(l, dummy_field, xigrid, potentials1):
    potentials = potentials1
    field = dummy_field
    speedgrid = potentials.speedgrid
    pitchgrid = xigrid
    species = potentials.species

    va, ma, v, na, Gamma_aa = sympy.symbols("v_a m_a v n_a Gamma_aa", real=True)
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    x = v / va
    Fa = na / (pi32 * va**3) * sympy.exp(-(x**2))
    fa = (1 + x) * sympy.exp(-(x**2))
    CGaa = _compute_CGab_sympy(Fa, fa, l, v, va, va, Gamma_aa)

    gamma_aa_jax = gamma_ab(species[0], species[0])
    CG = FieldPartCG(field, pitchgrid, speedgrid, species, potentials)
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nxi - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)

    subs = {
        va: species[0].v_thermal,
        na: species[0].density,
        Gamma_aa: gamma_aa_jax,
    }

    CGsympy = _eval_f(CGaa, v, speedgrid.x * species[0].v_thermal, subs)
    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)
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
    cg = -CG.mv(f)
    cg = jnp.einsum("la,sxatz->sxltz", Txi_inv, cg)[0]

    # potentials are diagonal in legendre index, so outputs for idx != l should be 0
    np.testing.assert_allclose(cg[:, :l, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(cg[:, l + 1 :, :, :], 0, atol=1e-10)
    np.testing.assert_allclose(cg[:, l, 0, 0], CGsympy, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_CG_2_species_vs_sympy(l, dummy_field, xigrid, potential_gamma):
    potentials = potential_gamma
    field = dummy_field
    speedgrid = potentials.speedgrid
    pitchgrid = xigrid
    species = potentials.species

    v = sympy.symbols("v", real=True, positive=True)
    na, nb, ma, mb, Ta, Tb = sympy.symbols(
        "n_a n_b m_a m_b T_a T_b", real=True, positive=True
    )
    pi32 = sympy.pi ** sympy.Rational(3, 2)
    vta = sympy.sqrt(2 * Ta / ma)
    vtb = sympy.sqrt(2 * Tb / mb)
    Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb = sympy.symbols(
        "Gamma_aa Gamma_ab Gamma_ba Gamma_bb", real=True
    )
    xa = v / vta
    xb = v / vtb
    Fa = na / (pi32 * vta**3) * sympy.exp(-(xa**2))
    Fb = nb / (pi32 * vtb**3) * sympy.exp(-(xb**2))
    fa = (1 - xa + 3 * xa**2) * sympy.exp(-(xa**2))
    fb = (4 + xb - 2 * xb**2) * sympy.exp(-(xb**2))

    CG = FieldPartCG(field, pitchgrid, speedgrid, species, potentials)
    gamma_aa_jax = gamma_ab(species[0], species[0])
    gamma_ab_jax = gamma_ab(species[0], species[1])
    gamma_ba_jax = gamma_ab(species[1], species[0])
    gamma_bb_jax = gamma_ab(species[1], species[1])
    subs = {
        na: float(species[0].density),
        nb: float(species[1].density),
        ma: float(species[0].species.mass),
        mb: float(species[1].species.mass),
        Ta: float(species[0].temperature * JOULE_PER_EV),
        Tb: float(species[1].temperature * JOULE_PER_EV),
        Gamma_aa: float(gamma_aa_jax),
        Gamma_ab: float(gamma_ab_jax),
        Gamma_ba: float(gamma_ba_jax),
        Gamma_bb: float(gamma_bb_jax),
    }

    CGaa = _compute_CGab_sympy(Fa, fa, l, v, vta, vta, Gamma_aa)
    CGab = _compute_CGab_sympy(Fa, fb, l, v, vta, vtb, Gamma_ab)
    CGba = _compute_CGab_sympy(Fb, fa, l, v, vtb, vta, Gamma_ba)
    CGbb = _compute_CGab_sympy(Fb, fb, l, v, vtb, vtb, Gamma_bb)
    CGaa_sympy = _eval_f(CGaa, v, speedgrid.x * species[0].v_thermal, subs)
    CGab_sympy = _eval_f(CGab, v, speedgrid.x * species[0].v_thermal, subs)
    CGba_sympy = _eval_f(CGba, v, speedgrid.x * species[1].v_thermal, subs)
    CGbb_sympy = _eval_f(CGbb, v, speedgrid.x * species[1].v_thermal, subs)
    CGa_sympy = CGaa_sympy + CGab_sympy
    CGb_sympy = CGba_sympy + CGbb_sympy

    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)
    ffb = _eval_f(fb, v, speedgrid.x * species[1].v_thermal, subs)
    f = np.ones((2, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
    T = orthax.orthval(
        pitchgrid.xi,
        jnp.zeros(potentials.legendregrid.nxi).at[l].set(1.0),
        potentials.legendregrid.xirec,
    )[None, :, None, None]
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nxi - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)
    f[0] *= ffa[:, None, None, None] * T
    f[1] *= ffb[:, None, None, None] * T
    CG_jax = -CG.mv(f.flatten()).reshape(f.shape)
    CG_jax = jnp.einsum("la,sxatz->sxltz", Txi_inv, CG_jax)
    CGa_jax = CG_jax[0, :, :, 0, 0]
    CGb_jax = CG_jax[1, :, :, 0, 0]
    np.testing.assert_allclose(CGa_jax[:, l], CGa_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(CGb_jax[:, l], CGb_sympy, rtol=1e-8, atol=0)


def test_verify_collision_null_single_species(dummy_field):
    """Check the null space of single species collision operator."""
    speedgrid = MaxwellSpeedGrid(5)
    pitchgrid = UniformPitchAngleGrid(129)
    field = dummy_field
    nt, nz = field.ntheta, field.nzeta

    ni = 5e19
    ti = 1000
    ions1 = GlobalMaxwellian(
        Hydrogen, lambda x: ti * (1 - x**2), lambda x: ni * (1 - x**4)
    ).localize(0.5)

    R = RosenbluthPotentials(speedgrid, [ions1], quad=False)
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
