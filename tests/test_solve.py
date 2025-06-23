"""Test solving the DKE/MDKE."""

import cola
import desc
import jax.numpy as jnp
import monkes
import numpy as np
import pytest

import yancc
from yancc.collisions import MonoenergeticPitchAngleScattering
from yancc.trajectories import DKESTrajectories


def _solve_mdke(field, pitchgrid, species, E_psi, v):
    CL = MonoenergeticPitchAngleScattering(field, pitchgrid, species, v)
    TMDKE = DKESTrajectories(field, pitchgrid, species, v, E_psi)
    MDKE = TMDKE + CL
    con = yancc.misc.MDKEConstraint(field, pitchgrid, species, v)
    sor = yancc.misc.MDKESources(field, pitchgrid, species, v)
    zero = cola.ops.Dense(jnp.zeros((1, 1)))
    rhs = yancc.misc.mdke_rhs(field, pitchgrid, v)
    operator = yancc.linalg.BorderedOperator(MDKE, sor, con, zero)
    fs = np.linalg.solve(operator.to_dense(), rhs)
    ss = rhs
    Dijs = yancc.misc.compute_monoenergetic_coefficients(fs, ss, field, pitchgrid, v)
    fs = fs[:-1]
    ss = ss[:-1]
    fs = fs.reshape((pitchgrid.nxi, field.ntheta, field.nzeta, 3))
    fs = jnp.moveaxis(fs, -1, 0)
    ss = ss.reshape((pitchgrid.nxi, field.ntheta, field.nzeta, 3))
    ss = jnp.moveaxis(ss, -1, 0)
    return Dijs, fs, ss


@pytest.mark.xfail
def test_solve_mdke_against_monkes():
    """Test solutions of MDKE against MONKES."""
    pitchgrid = yancc.velocity_grids.LegendrePitchAngleGrid(11)

    nt = 7
    nz = 7
    eq = desc.examples.get("W7-X")
    field = monkes.Field.from_desc(eq, 0.5, nt, nz)

    ni = 5e20
    ti = 10000

    ions1 = monkes.GlobalMaxwellian(
        monkes.Hydrogen, lambda x: ti * (1 - x**2), lambda x: ni * (1 - x**4)
    ).localize(0.5)

    E_psi = 0.0
    v = ions1.v_thermal
    Dijs, fs, ss = _solve_mdke(field, pitchgrid, ions1, E_psi, v)

    nu = monkes._species.nuD_ab(ions1, ions1, v)
    nuhat = nu / v
    Erhat = E_psi / v * jnp.abs(field.psi_r)

    Dijm, fm, sm = monkes._core.monoenergetic_dke_solve_internal(
        field, pitchgrid.nxi, Erhat, nuhat
    )
    # convert monkes to nodal basis
    fm = np.einsum("il,altz->aitz", pitchgrid.xivander, fm)
    sm = np.einsum("il,altz->aitz", pitchgrid.xivander, sm)

    np.testing.assert_allclose(Dijs, Dijm, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fs[0], fm[0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fs[1], fm[1], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fs[2] / field.B0, fm[2], atol=1e-10, rtol=1e-10)

    E_psi = 1e-3 * v

    Dijs, fs, ss = _solve_mdke(field, pitchgrid, ions1, E_psi, v)

    Erhat = E_psi / v * jnp.abs(field.psi_r)

    Dijm, fm, sm = monkes._core.monoenergetic_dke_solve_internal(
        field, pitchgrid.nxi, Erhat, nuhat
    )
    # convert monkes to nodal basis
    fm = np.einsum("il,altz->aitz", pitchgrid.xivander, fm)
    sm = np.einsum("il,altz->aitz", pitchgrid.xivander, sm)

    # TODO: figure why errors are larger for nonzero Er
    np.testing.assert_allclose(Dijs, Dijm, atol=1e-4, rtol=1e-4)
