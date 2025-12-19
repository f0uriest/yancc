"""Tests for MDKE operators."""

import numpy as np
import pytest

import yancc.trajectories as trajectories
import yancc.trajectories_scipy as trajectories_scipy
from yancc.collisions import (
    EnergyScattering,
    FieldPartCD,
    FieldPartCG,
    FieldPartCH,
    MDKEPitchAngleScattering,
    PitchAngleScattering,
)


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


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_speed(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
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


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_theta(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    for p1 in ["2d", "4d"]:
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
        np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_zeta(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    for p1 in ["2d", "4d"]:
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


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_pitch(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    for p1 in ["2d", "4d"]:
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


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_pitch_angle_scattering(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }

    for p2 in [2, 4]:
        f = PitchAngleScattering(
            field, pitchgrid, speedgrid, species2, p2=p2, axorder=axorder, gauge=gauge
        )
        A = f.as_matrix()
        np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
        B = extract_blocks(A, sizes[axorder[-1]])
        np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_energy_scattering(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
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


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_CD(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
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


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_CG(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
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


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_CH(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
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


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["sxatz", "zsxat", "tzsxa", "atzsx", "xatzs"])
def test_diagonals_dke_full(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
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
        potentials2,
        "2d",
        4,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    np.testing.assert_allclose(np.diag(A), f.diagonal(), err_msg=axorder)
    B = extract_blocks(A, sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_theta(gauge, axorder, field, pitchgrid):
    sizes = {
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    erhohat = np.array(1e3)

    for p1 in ["2d", "4d"]:
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
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_zeta(gauge, axorder, field, pitchgrid):
    sizes = {
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    erhohat = np.array(1e3)

    for p1 in ["2d", "4d"]:
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
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_pitch(gauge, axorder, field, pitchgrid):
    sizes = {
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    erhohat = np.array(1e3)

    for p1 in ["2d", "4d"]:
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
@pytest.mark.parametrize("axorder", ["atz", "zat", "tza"])
def test_diagonals_mdke_pitch_angle_scattering(gauge, axorder, field, pitchgrid):
    sizes = {
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    nuhat = np.array(1e-3)

    for p2 in [2, 4]:
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
        "a": pitchgrid.nxi,
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
        "a": pitchgrid.nxi,
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
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_theta(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    for p1 in ["2d", "4d"]:
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
        B = extract_blocks(
            A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]]
        )
        np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_zeta(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    for p1 in ["2d", "4d"]:
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
        B = extract_blocks(
            A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]]
        )
        np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_pitch(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }
    Erho = np.array(1e3)

    for p1 in ["2d", "4d"]:
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
        B = extract_blocks(
            A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]]
        )
        np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_pitch_angle_scattering(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
        "t": field.ntheta,
        "z": field.nzeta,
    }

    for p2 in [2, 4]:
        f = PitchAngleScattering(
            field, pitchgrid, speedgrid, species2, p2=p2, axorder=axorder, gauge=gauge
        )
        A = f.as_matrix()
        B = extract_blocks(
            A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]]
        )
        np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)


@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("axorder", ["atzsx", "tzasx", "zatsx"])
def test_diagonals2_dke_energy_scattering(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
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
        "a": pitchgrid.nxi,
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
        "a": pitchgrid.nxi,
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
        "a": pitchgrid.nxi,
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
def test_diagonals2_dke_full(
    gauge, axorder, field, pitchgrid, speedgrid, species2, potentials2
):
    sizes = {
        "s": len(species2),
        "x": speedgrid.nx,
        "a": pitchgrid.nxi,
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
        potentials2,
        "2d",
        4,
        axorder=axorder,
        gauge=gauge,
    )
    A = f.as_matrix()
    B = extract_blocks(A, sizes[axorder[-3]] * sizes[axorder[-2]] * sizes[axorder[-1]])
    np.testing.assert_allclose(B, f.block_diagonal2(), err_msg=axorder)
