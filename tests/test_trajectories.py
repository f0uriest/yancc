"""Tests for MDKE operators."""

import jax.numpy as jnp
import numpy as np
import pytest

import yancc.trajectories as trajectories
import yancc.trajectories_scipy as trajectories_scipy
from yancc.collisions import (
    EnergyScattering,
    FieldPartCD,
    FieldPartCG,
    FieldPartCH,
    FieldParticleScattering,
    FokkerPlanckLandau,
    MDKEPitchAngleScattering,
    PitchAngleScattering,
)
from yancc.finite_diff import build_advection_matrix, fd_kwargs
from yancc.linalg import banded_to_dense
from yancc.velocity_grids import QuadraticPitchAngleGrid


def extract_blocks(a, m):
    """Get diagonal blocks from a of size m."""
    return np.array(
        [a[i * m : (i + 1) * m, i * m : (i + 1) * m] for i in range(a.shape[0] // m)]
    )


@pytest.mark.parametrize("p1", ["1a", "4b"])
@pytest.mark.parametrize("p2", [2, 4])
@pytest.mark.parametrize("erhohat", [1e-3, 1e3])
@pytest.mark.parametrize("nuhat", [1e-3, 1e3])
@pytest.mark.parametrize("gauge", [True, False])
def test_scipy_operators(p1, p2, erhohat, nuhat, gauge, field, pitchgrid):
    """Test that scipy sparse matrices are the same as jax jacobians."""
    A1 = trajectories_scipy.mdke(field, pitchgrid, erhohat, nuhat, p1, p2, gauge=gauge)
    A2 = trajectories.MDKE(field, pitchgrid, erhohat, nuhat, p1, p2, "atz", gauge=gauge)
    np.testing.assert_allclose(A1.toarray(), A2.as_matrix())


@pytest.mark.parametrize("p1", ["1a", "4d"])
def test_pitch_operator_nonuniform(field, p1):
    """Pitch advection operator uses coordinate-based stencils on a non-uniform grid.

    The pitch grid coordinates ``a`` carry the non-uniform spacing, so the
    operator must build its forward/backward matrices from ``pitchgrid.a`` with
    symmetric BCs on [0, pi] (not the uniform ``h = pi/na`` assumption).
    """
    grid = QuadraticPitchAngleGrid(31, 0.6)
    op = trajectories.MDKEPitch(field, grid, erhohat=0.0, p1=p1)

    kwargs = fd_kwargs[p1]
    fd = build_advection_matrix(
        grid.a, direction="fwd", bc_type="symmetric", domain=(0, np.pi), **kwargs
    )
    bd = build_advection_matrix(
        grid.a, direction="bwd", bc_type="symmetric", domain=(0, np.pi), **kwargs
    )
    np.testing.assert_allclose(op._fd, fd)
    np.testing.assert_allclose(op._bd, bd)

    # operator applies cleanly on the non-uniform grid
    n = field.ntheta * field.nzeta * grid.na
    out = op.mv(jnp.ones(n))
    assert out.shape == (n,)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_speed(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKESpeed(
        field, pitchgrid, speedgrid, species2, Erho, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_theta(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2, p1
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKETheta(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal("dense"), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_zeta(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2, p1
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKEZeta(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_pitch(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2, p1
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKEPitch(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p2", [2, 4])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_pitch_angle_scattering(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2, p2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }

    f = PitchAngleScattering(
        field, pitchgrid, speedgrid, species2, p2=p2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_energy_scattering(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }

    f = EnergyScattering(
        field, pitchgrid, speedgrid, species2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_CD(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FieldPartCD(
        field, pitchgrid, speedgrid, species2, potentials2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_CG(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FieldPartCG(
        field, pitchgrid, speedgrid, species2, potentials2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_CH(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FieldPartCH(
        field, pitchgrid, speedgrid, species2, potentials2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    D = f.block_diagonal("banded")
    bw = D.shape[1] // 2
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_CF(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FieldParticleScattering(
        field, pitchgrid, speedgrid, species2, potentials2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    bw = sizes[axorder[-1]] // 2
    D = f.block_diagonal("banded", bw)
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_FokkerPlanck(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FokkerPlanckLandau(
        field,
        pitchgrid,
        speedgrid,
        species2,
        [],
        potentials2,
        axorder=axorder,
        gauge=gauge,
        operator_weights=jnp.linspace(1, 2, 3),
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    bw = sizes[axorder[-1]] // 2
    D = f.block_diagonal("banded", bw)
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_full(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKE(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        potentials=potentials2,
        p1="2d",
        p2=4,
        axorder=axorder,
        gauge=gauge,
        operator_weights=jnp.linspace(1, 5, 8),
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    bw = sizes[axorder[-1]] // 2
    D = f.block_diagonal("banded", bw)
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)

    # if we drop the field term it should have bandwidth 4 for our usual stencils
    f = trajectories.DKE(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        potentials=potentials2,
        p1="2d",
        p2=4,
        axorder=axorder,
        gauge=gauge,
        operator_weights=jnp.ones(8).at[-2:].set(0),
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)
    bw = min(4, sizes[axorder[-1]] // 2)
    D = f.block_diagonal("banded", bw)
    D = banded_to_dense(bw, bw, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)
    D = f.block_diagonal("banded")  # auto-compute bw from FD stencil
    bw_auto = D.shape[1] // 2
    D = banded_to_dense(bw_auto, bw_auto, D)
    np.testing.assert_allclose(B, D, err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_theta(gauge, axorder, field, pitchgrid, p1):
    sizes = {
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    erhohat = np.array(1e3)

    f = trajectories.MDKETheta(
        field,
        pitchgrid,
        erhohat,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_zeta(gauge, axorder, field, pitchgrid, p1):
    sizes = {
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    erhohat = np.array(1e3)

    f = trajectories.MDKEZeta(
        field,
        pitchgrid,
        erhohat,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_pitch(gauge, axorder, field, pitchgrid, p1):
    sizes = {
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    erhohat = np.array(1e3)

    f = trajectories.MDKEPitch(
        field,
        pitchgrid,
        erhohat,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p2", [2, 4])
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_pitch_angle_scattering(gauge, axorder, field, pitchgrid, p2):
    sizes = {
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    nuhat = np.array(1e-3)

    f = MDKEPitchAngleScattering(
        field, pitchgrid, nuhat, p2=p2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_full(gauge, axorder, field, pitchgrid):
    sizes = {
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    erhohat = np.array(1e3)
    nuhat = np.array(1e-3)

    f = trajectories.MDKE(
        field,
        pitchgrid,
        erhohat,
        nuhat,
        p1="2d",
        p2=4,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_speed(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKESpeed(
        field, pitchgrid, speedgrid, species2, Erho, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_theta(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2, p1
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKETheta(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_zeta(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2, p1
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKEZeta(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p1", ["2d", "4d"])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_pitch(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2, p1
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKEPitch(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        p1=p1,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("p2", [2, 4])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_pitch_angle_scattering(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2, p2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }

    f = PitchAngleScattering(
        field, pitchgrid, speedgrid, species2, p2=p2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_energy_scattering(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }

    f = EnergyScattering(
        field, pitchgrid, speedgrid, species2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_CD(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FieldPartCD(
        field, pitchgrid, speedgrid, species2, potentials2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_CG(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FieldPartCG(
        field, pitchgrid, speedgrid, species2, potentials2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_CH(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FieldPartCH(
        field, pitchgrid, speedgrid, species2, potentials2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_CF(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FieldParticleScattering(
        field, pitchgrid, speedgrid, species2, potentials2, axorder=axorder, gauge=gauge
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_FokkerPlanck(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    f = FokkerPlanckLandau(
        field,
        pitchgrid,
        speedgrid,
        species2,
        [],
        potentials2,
        axorder=axorder,
        gauge=gauge,
        operator_weights=jnp.linspace(1, 2, 3),
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_full(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    f = trajectories.DKE(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        potentials=potentials2,
        p1="2d",
        p2=4,
        axorder=axorder,
        gauge=gauge,
        operator_weights=jnp.linspace(1, 5, 8),
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("weight", [0, 1])
def test_background_species(field, pitchgrid, speedgrid, species2, gauge, weight):
    """Test that background species are included correctly."""
    # first we get the full 2 species DKE operator
    operator_weights = jnp.ones(8).at[-1].set(0).at[-2].set(weight)
    Erho = jnp.array(1e3)
    Aboth = trajectories.DKE(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        gauge=gauge,
        operator_weights=operator_weights,
    )
    # then single species w/ other as background
    Ai = trajectories.DKE(
        field,
        pitchgrid,
        speedgrid,
        [species2[0]],
        Erho,
        [species2[1]],
        gauge=gauge,
        operator_weights=operator_weights,
    )
    Ae = trajectories.DKE(
        field,
        pitchgrid,
        speedgrid,
        [species2[1]],
        Erho,
        [species2[0]],
        gauge=gauge,
        operator_weights=operator_weights,
    )
    Aboth = Aboth.as_matrix()
    Ai = Ai.as_matrix()
    Ae = Ae.as_matrix()
    # the diagonal blocks of the full operator should be the same as the single species
    # operators with the background included
    n = Aboth.shape[0] // 2
    Aboth_i = Aboth[:n, :n]
    Aboth_e = Aboth[n:, n:]
    np.testing.assert_allclose(Ai, Aboth_i)
    np.testing.assert_allclose(Ae, Aboth_e)


# ---------------------------------------------------------------------------
# operator protocol sweep: out_structure / in_structure / transpose
# ---------------------------------------------------------------------------


def _check_transpose_protocol(op):
    """Square in/out structures; transpose materializes/acts as the matrix transpose."""
    assert op.out_structure() == op.in_structure()
    opT = op.transpose()
    assert opT.in_structure() == op.out_structure()
    assert opT.out_structure() == op.in_structure()
    M = op.as_matrix()
    # TransposedLinearOperator.as_matrix is defined as operator.as_matrix().T
    np.testing.assert_allclose(opT.as_matrix(), M.T)
    # the transpose action (via jax.linear_transpose) matches M.T @ v
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal(M.shape[0]))
    ref = M.T @ v
    np.testing.assert_allclose(
        opT.mv(v), ref, rtol=1e-6, atol=1e-6 * np.max(np.abs(np.asarray(ref)))
    )


@pytest.mark.parametrize("cls", ["MDKETheta", "MDKEZeta", "MDKEPitch"])
def test_mdke_operator_transpose_protocol(cls, field, pitchgrid):
    op = getattr(trajectories, cls)(field, pitchgrid, np.array(1e3))
    _check_transpose_protocol(op)


@pytest.mark.parametrize("cls", ["DKETheta", "DKEZeta", "DKEPitch", "DKESpeed"])
def test_dke_operator_transpose_protocol(cls, field, pitchgrid, speedgrid, species2):
    op = getattr(trajectories, cls)(
        field, pitchgrid, speedgrid, species2, np.array(1e3)
    )
    _check_transpose_protocol(op)
