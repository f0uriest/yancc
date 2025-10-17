"""Tests for magnetic field structure."""

import numpy as np

from yancc.field import Field


def _compare_fields_scalars(field1, field2):
    np.testing.assert_allclose(field1.iota, field2.iota, rtol=1e-5)
    np.testing.assert_allclose(field1.Bmag_fsa, field2.Bmag_fsa, rtol=1e-5)
    np.testing.assert_allclose(field1.B2mag_fsa, field2.B2mag_fsa, rtol=1e-5)
    np.testing.assert_allclose(field1.a_minor, field2.a_minor, rtol=5e-2)
    np.testing.assert_allclose(field1.R_major, field2.R_major, rtol=5e-2)
    np.testing.assert_allclose(field1.psi_r, field2.psi_r, rtol=5e-2)
    np.testing.assert_allclose(field1.sqrtg.mean(), field2.sqrtg.mean(), rtol=1e-5)
    np.testing.assert_allclose(field1.B0, field2.B0, rtol=1e-2)


def _compare_fields_local(field1, field2):
    np.testing.assert_allclose(field1.Bmag, field2.Bmag, rtol=1e-3)
    np.testing.assert_allclose(field1.sqrtg, field2.sqrtg, rtol=1e-3)
    np.testing.assert_allclose(field1.B_sub_t, field2.B_sub_t, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(field1.B_sub_z, field2.B_sub_z, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(field1.B_sup_t, field2.B_sup_t, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(field1.B_sup_z, field2.B_sup_z, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(field1.dBdz, field2.dBdz, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(field1.dBdt, field2.dBdt, rtol=1e-2, atol=1e-2)


def test_field_types():
    """Test solving the MDKE with the same field in different formats."""
    import desc  # pyright: ignore[reportMissingImports]

    eq = desc.io.load("tests/data/NCSX_output.h5")[-1]

    nt = 17
    nz = 37
    rho = 0.5
    s = rho**2

    field1 = Field.from_desc(eq, rho, nt, nz)
    field2 = Field.from_vmec("tests/data/wout_NCSX.nc", s, nt, nz)
    field3 = Field.from_booz_xform("tests/data/boozmn_wout_NCSX.nc", s, nt, nz)
    field4 = Field.from_ipp_bc("tests/data/NCSX.bc", s, nt, nz)

    np.testing.assert_allclose(
        field1.B_sub_t * field1.B_sup_t + field1.B_sub_z * field1.B_sup_z,
        field1.Bmag**2,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        field2.B_sub_t * field2.B_sup_t + field2.B_sub_z * field2.B_sup_z,
        field2.Bmag**2,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        field3.B_sub_t * field3.B_sup_t + field3.B_sub_z * field3.B_sup_z,
        field3.Bmag**2,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        field4.B_sub_t * field4.B_sup_t + field4.B_sub_z * field4.B_sup_z,
        field4.Bmag**2,
        rtol=1e-5,
    )

    _compare_fields_scalars(field1, field2)
    _compare_fields_scalars(field1, field3)
    _compare_fields_scalars(field1, field4)
    # these are ostensibly in the same coordinates, so we can compare local qtys
    _compare_fields_local(field1, field2)
    _compare_fields_local(field3, field4)
