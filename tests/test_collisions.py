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
    FieldParticleScattering,
    FokkerPlanckLandau,
    MDKEPitchAngleScattering,
    PitchAngleScattering,
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
    _eval_f_sampled,
    _speed_subset,
)


def _CE_flux_form_nodal(CE, speedgrid, f):
    """Pointwise C_E f from the operator's flux-form coefficients, shape (ns, nx).

    The sign is that of the physical operator; ``mv`` returns its negative, as the
    collision operator enters the DKE with a minus sign.

    EnergyScattering is assembled in weak form, so its action is the projection of
    C_E f onto the speed basis rather than C_E f at the nodes. Its coefficients are the
    exact continuum ones, so they are checked against the analytic operator by applying
    them pointwise with the grid's spectral derivatives, which are exact for f in the
    speed basis.
    """
    f = np.atleast_2d(f)
    return (
        CE.coeff2 * (f @ speedgrid.D2x_pseudospectral.T)
        + CE.coeff1 * (f @ speedgrid.Dx_pseudospectral.T)
        + CE.coeff0 * f
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

    CEjax = _CE_flux_form_nodal(CE, speedgrid, ffa)[0]

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
    CE_jax = _CE_flux_form_nodal(CE, speedgrid, np.stack([ffa, ffb]))
    CEa_jax = CE_jax[0]
    CEb_jax = CE_jax[1]

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
        pitchgrid.xi, potentials.legendregrid.nalpha - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)

    subs = {
        va: species[0].v_thermal,
        na: species[0].density,
        Gamma_aa: gamma_aa_jax,
    }

    CDsympy = _eval_f(CDaa, v, speedgrid.x * species[0].v_thermal, subs)
    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)

    f = np.ones((1, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta))
    f[0] *= (
        ffa[:, None, None, None]
        * orthax.orthval(
            pitchgrid.xi,
            jnp.zeros(potentials.legendregrid.nalpha).at[l].set(1.0),
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
def test_CD_2_species_vs_sympy(l, dummy_field, xigrid, potential_gauss_legendre):
    potentials = potential_gauss_legendre
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
    f = np.ones((2, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta))
    T = orthax.orthval(
        pitchgrid.xi,
        jnp.zeros(potentials.legendregrid.nalpha).at[l].set(1.0),
        potentials.legendregrid.xirec,
    )[None, :, None, None]
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nalpha - 1, potentials.legendregrid.xirec
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
        pitchgrid.xi, potentials.legendregrid.nalpha - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)

    subs = {
        va: species[0].v_thermal,
        na: species[0].density,
        Gamma_aa: gamma_aa_jax,
    }

    CHsympy = _eval_f(CHaa, v, speedgrid.x * species[0].v_thermal, subs)
    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)

    f = np.ones((1, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta))
    f[0] *= (
        ffa[:, None, None, None]
        * orthax.orthval(
            pitchgrid.xi,
            jnp.zeros(potentials.legendregrid.nalpha).at[l].set(1.0),
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


# Subset of l values: single-species variant exercises l=[0,1,2,3];
# the 2-species version only needs to verify cross-species coupling.
@pytest.mark.parametrize("l", [0, 2])
def test_CH_2_species_vs_sympy(l, dummy_field, xigrid, potential_gauss_legendre):
    potentials = potential_gauss_legendre
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
    CHaa_sympy = _eval_f_sampled(CHaa, v, speedgrid.x * species[0].v_thermal, subs)
    CHab_sympy = _eval_f_sampled(CHab, v, speedgrid.x * species[0].v_thermal, subs)
    CHba_sympy = _eval_f_sampled(CHba, v, speedgrid.x * species[1].v_thermal, subs)
    CHbb_sympy = _eval_f_sampled(CHbb, v, speedgrid.x * species[1].v_thermal, subs)
    CHa_sympy = CHaa_sympy + CHab_sympy
    CHb_sympy = CHba_sympy + CHbb_sympy

    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)
    ffb = _eval_f(fb, v, speedgrid.x * species[1].v_thermal, subs)
    f = np.ones((2, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta))
    T = orthax.orthval(
        pitchgrid.xi,
        jnp.zeros(potentials.legendregrid.nalpha).at[l].set(1.0),
        potentials.legendregrid.xirec,
    )[None, :, None, None]
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nalpha - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)
    f[0] *= ffa[:, None, None, None] * T
    f[1] *= ffb[:, None, None, None] * T
    CH_jax = -CH.mv(f.flatten()).reshape(f.shape)
    CH_jax = jnp.einsum("la,sxatz->sxltz", Txi_inv, CH_jax)
    CHa_jax = CH_jax[0, :, :, 0, 0]
    CHb_jax = CH_jax[1, :, :, 0, 0]
    i = _speed_subset(speedgrid.nx)
    np.testing.assert_allclose(CHa_jax[i, l], CHa_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(CHb_jax[i, l], CHb_sympy, rtol=1e-8, atol=0)


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
        pitchgrid.xi, potentials.legendregrid.nalpha - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)

    subs = {
        va: species[0].v_thermal,
        na: species[0].density,
        Gamma_aa: gamma_aa_jax,
    }

    CGsympy = _eval_f(CGaa, v, speedgrid.x * species[0].v_thermal, subs)
    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)
    f = np.ones((1, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta))
    f[0] *= (
        ffa[:, None, None, None]
        * orthax.orthval(
            pitchgrid.xi,
            jnp.zeros(potentials.legendregrid.nalpha).at[l].set(1.0),
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


# Subset of l values: single-species variant exercises l=[0,1,2,3];
# the 2-species version only needs to verify cross-species coupling.
@pytest.mark.parametrize("l", [0, 2])
def test_CG_2_species_vs_sympy(l, dummy_field, xigrid, potential_gauss_legendre):
    potentials = potential_gauss_legendre
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
    CGaa_sympy = _eval_f_sampled(CGaa, v, speedgrid.x * species[0].v_thermal, subs)
    CGab_sympy = _eval_f_sampled(CGab, v, speedgrid.x * species[0].v_thermal, subs)
    CGba_sympy = _eval_f_sampled(CGba, v, speedgrid.x * species[1].v_thermal, subs)
    CGbb_sympy = _eval_f_sampled(CGbb, v, speedgrid.x * species[1].v_thermal, subs)
    CGa_sympy = CGaa_sympy + CGab_sympy
    CGb_sympy = CGba_sympy + CGbb_sympy

    ffa = _eval_f(fa, v, speedgrid.x * species[0].v_thermal, subs)
    ffb = _eval_f(fb, v, speedgrid.x * species[1].v_thermal, subs)
    f = np.ones((2, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta))
    T = orthax.orthval(
        pitchgrid.xi,
        jnp.zeros(potentials.legendregrid.nalpha).at[l].set(1.0),
        potentials.legendregrid.xirec,
    )[None, :, None, None]
    Txi = orthax.orthvander(
        pitchgrid.xi, potentials.legendregrid.nalpha - 1, potentials.legendregrid.xirec
    )
    Txi_inv = jnp.linalg.pinv(Txi)
    f[0] *= ffa[:, None, None, None] * T
    f[1] *= ffb[:, None, None, None] * T
    CG_jax = -CG.mv(f.flatten()).reshape(f.shape)
    CG_jax = jnp.einsum("la,sxatz->sxltz", Txi_inv, CG_jax)
    CGa_jax = CG_jax[0, :, :, 0, 0]
    CGb_jax = CG_jax[1, :, :, 0, 0]
    i = _speed_subset(speedgrid.nx)
    np.testing.assert_allclose(CGa_jax[i, l], CGa_sympy, rtol=1e-10, atol=0)
    np.testing.assert_allclose(CGb_jax[i, l], CGb_sympy, rtol=1e-8, atol=0)


def test_verify_collision_null_single_species(dummy_field):
    """Check the null space of single species collision operator."""
    # C_E is assembled in weak form, so its cancellation against C_L and C_F on the
    # momentum and energy invariants holds only to the speed resolution. The speed grid
    # is fine enough here for that residual to be small relative to the terms that
    # cancel and for the three null modes to be resolved in the spectrum.
    speedgrid = MaxwellSpeedGrid(15)
    pitchgrid = UniformPitchAngleGrid(129)
    field = dummy_field
    nt, nz = field.ntheta, field.nzeta

    ni = 5e19
    ti = 1000
    ions1 = GlobalMaxwellian(
        Hydrogen, lambda x: ti * (1 - x**2), lambda x: ni * (1 - x**4)
    ).localize(0.5)

    R = RosenbluthPotentials(speedgrid, [ions1], quad=False)
    C = FokkerPlanckLandau(field, pitchgrid, speedgrid, [ions1], potentials=R)
    shape = (1, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta)
    x = speedgrid.x

    def term_scale(f):
        """Largest magnitude among the collision terms that must cancel on f."""
        return max(float(np.abs(op.mv(f.flatten())).max()) for op in (C.CL, C.CE, C.CF))

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
    # relative to the cancelling terms: the weak-form speed operator and finite
    # differences in pitch angle both leave a resolution-dependent residual
    np.testing.assert_allclose(cf, 0, atol=1e-4 * term_scale(f))

    # C acting on v^2*maxwellian = 0
    ff = x**2 * np.exp(-(x**2))
    f = (
        np.ones((1, speedgrid.nx, pitchgrid.nalpha, field.ntheta, field.nzeta))
        * ff[None, :, None, None, None]
    )
    cf = C.mv(f.flatten()).reshape(f.shape)
    np.testing.assert_allclose(cf, 0, atol=1e-2 * term_scale(f))

    es = np.linalg.eigvals(C.as_matrix())
    # should have purely real eigvals
    np.testing.assert_allclose(np.imag(es), 0, atol=1e-7)
    # should all be positive, within fudge factor for zeros
    np.testing.assert_array_less(-1e-14 * np.real(es).max(), np.real(es))
    # should have a null space of dimension 3*nt*nz
    # maxwellian, v*maxwellian, v^2*maxwellian
    assert sum(np.abs(es) < 1e-14 * np.max(np.abs(es))) == 3 * nt * nz


def _check_operator_interface(op, rng):
    """Exercise the lineax interface: structures, as_matrix, transpose."""
    ins = op.in_structure()
    outs = op.out_structure()
    A = np.asarray(op.as_matrix())
    assert A.shape == (outs.shape[0], ins.shape[0])

    x = rng.standard_normal(ins.shape[0])
    np.testing.assert_allclose(op.mv(x), A @ x, rtol=1e-6, atol=1e-10)

    opT = op.transpose()
    np.testing.assert_allclose(np.asarray(opT.as_matrix()), A.T, rtol=1e-6, atol=1e-10)
    y = rng.standard_normal(outs.shape[0])
    np.testing.assert_allclose(opT.mv(y), A.T @ y, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize(
    "build",
    [
        lambda f, pg, sg, sp, pot: PitchAngleScattering(f, pg, sg, sp),
        lambda f, pg, sg, sp, pot: EnergyScattering(f, pg, sg, sp),
        lambda f, pg, sg, sp, pot: FieldPartCD(f, pg, sg, sp, pot),
        lambda f, pg, sg, sp, pot: FieldPartCG(f, pg, sg, sp, pot),
        lambda f, pg, sg, sp, pot: FieldPartCH(f, pg, sg, sp, pot),
        lambda f, pg, sg, sp, pot: FieldParticleScattering(f, pg, sg, sp, pot),
        # potentials=None exercises the default RosenbluthPotentials construction
        lambda f, pg, sg, sp, pot: FokkerPlanckLandau(f, pg, sg, sp),
    ],
    ids=["CL", "CE", "CD", "CG", "CH", "FPS", "FPL"],
)
def test_collision_operator_interface(dummy_field, species1, build):
    """out_structure/transpose/as_matrix are self-consistent."""
    pitchgrid = UniformPitchAngleGrid(7)
    speedgrid = MaxwellSpeedGrid(3)
    potentials = RosenbluthPotentials(speedgrid, species1)
    op = build(dummy_field, pitchgrid, speedgrid, species1, potentials)
    _check_operator_interface(op, np.random.default_rng(0))


def test_mdke_pitch_angle_scattering_interface(dummy_field):
    """out_structure/transpose for the monoenergetic scattering operator."""
    op = MDKEPitchAngleScattering(dummy_field, UniformPitchAngleGrid(7), 1.0)
    _check_operator_interface(op, np.random.default_rng(0))
