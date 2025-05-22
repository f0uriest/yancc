"""Tests for collision operator."""

import monkes
import numpy as np

from yancc.collisions import FokkerPlanckLandau, RosenbluthPotentials
from yancc.velocity_grids import LegendrePitchAngleGrid, SpeedGrid


def test_verify_collision_null_single_species():
    """Check the null space of single species collision operator."""
    speedgrid = SpeedGrid(11)
    pitchgrid = LegendrePitchAngleGrid(12)

    # just need a dummy field for this
    nt = 1
    nz = 3
    field = monkes.Field(
        rho=0.5,
        B_sup_t=np.ones((nt, nz)),
        B_sup_z=np.ones((nt, nz)),
        B_sub_t=np.ones((nt, nz)),
        B_sub_z=np.ones((nt, nz)),
        Bmag=np.ones((nt, nz)),
        sqrtg=np.ones((nt, nz)),
        psi_r=1.0,
        iota=1.0,
    )
    ni = 5e19
    ti = 1000
    ions1 = monkes.GlobalMaxwellian(
        monkes.Hydrogen, lambda x: ti * (1 - x**2), lambda x: ni * (1 - x**4)
    ).localize(0.5)

    R = RosenbluthPotentials(speedgrid, pitchgrid, [ions1], quad=False)
    C = FokkerPlanckLandau(field, speedgrid, pitchgrid, [ions1], R)
    shape = (1, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta)
    x = speedgrid.x
    xi = pitchgrid.xi

    # C acting on maxwellian = 0
    ff = (speedgrid.xvander_inv @ np.exp(-(x**2)))[None, :, None, None, None]
    f = np.ones(shape) * ff
    cf = (C @ f.flatten()).reshape(shape)
    np.testing.assert_allclose(cf, 0, atol=1e-7)

    # C acting on v*maxwellian = 0
    ff = (speedgrid.xvander_inv @ (x * np.exp(-(x**2))))[
        None, :, None, None, None
    ] * xi[None, None, :, None, None]
    f = np.ones(shape) * ff
    cf = (C @ f.flatten()).reshape(f.shape)
    np.testing.assert_allclose(cf, 0, atol=1e-7)

    # C acting on v^2*maxwellian = 0
    ff = speedgrid.xvander_inv @ (x**2 * np.exp(-(x**2)))
    f = (
        np.ones((1, speedgrid.nx, pitchgrid.nxi, field.ntheta, field.nzeta))
        * ff[None, :, None, None, None]
    )
    cf = (C @ f.flatten()).reshape(f.shape)
    np.testing.assert_allclose(cf, 0, atol=1e-7)

    xvander = np.kron(speedgrid.xvander_inv, np.eye(pitchgrid.nxi * nt * nz))
    C = xvander @ C
    es = np.linalg.eigvals(C)
    # should have purely real eigvals
    np.testing.assert_allclose(es.imag, 0, atol=1e-7)
    # should all be positive
    np.testing.assert_array_less(-1e-7, es.real)
    # should have a null space of dimension 3*nt*nz
    assert sum(abs(es) < 1e-8) == 3 * nt * nz
