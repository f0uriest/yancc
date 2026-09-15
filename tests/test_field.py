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

    field1 = Field.from_desc(eq, rho, nt, nz)
    field2 = Field.from_vmec("tests/data/wout_NCSX.nc", rho, nt, nz)
    field3 = Field.from_booz_xform("tests/data/boozmn_wout_NCSX.nc", rho, nt, nz)
    field4 = Field.from_ipp_bc("tests/data/NCSX.bc", rho, nt, nz)

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


def test_derived_scalars_when_omitted():
    """Psi/iota/R_major/a_minor are recovered when omitted from the constructor."""
    ref = Field.from_vmec("tests/data/wout_NCSX.nc", 0.5, 17, 37)
    derived = Field(
        rho=ref.rho,
        B_sup_t=ref.B_sup_t,
        B_sup_z=ref.B_sup_z,
        B_sub_t=ref.B_sub_t,
        B_sub_z=ref.B_sub_z,
        Bmag=ref.Bmag,
        sqrtg=ref.sqrtg,
        NFP=ref.NFP,
    )
    # Psi and iota are flux integrals of the field -> recovered exactly
    np.testing.assert_allclose(derived.Psi, ref.Psi, rtol=1e-6)
    np.testing.assert_allclose(derived.iota, ref.iota, rtol=1e-6)
    # R_major/a_minor are circular-torus estimates -> good to a few percent.
    np.testing.assert_allclose(derived.R_major, ref.R_major, rtol=8e-2)
    np.testing.assert_allclose(derived.a_minor, ref.a_minor, rtol=5e-2)
    # supplying them explicitly overrides the estimate exactly.
    explicit = Field(
        rho=ref.rho,
        B_sup_t=ref.B_sup_t,
        B_sup_z=ref.B_sup_z,
        B_sub_t=ref.B_sub_t,
        B_sub_z=ref.B_sub_z,
        Bmag=ref.Bmag,
        sqrtg=ref.sqrtg,
        Psi=ref.Psi,
        iota=ref.iota,
        R_major=ref.R_major,
        a_minor=ref.a_minor,
        NFP=ref.NFP,
    )
    np.testing.assert_allclose(explicit.Psi, ref.Psi, rtol=1e-12)
    np.testing.assert_allclose(explicit.iota, ref.iota, rtol=1e-12)
    np.testing.assert_allclose(explicit.R_major, ref.R_major, rtol=1e-12)
    np.testing.assert_allclose(explicit.a_minor, ref.a_minor, rtol=1e-12)


def test_from_boozer_optional_geometry():
    """from_boozer estimates R_major/a_minor when omitted, leaving the field intact."""
    booz = Field.from_booz_xform("tests/data/boozmn_wout_NCSX.nc", 0.5, 17, 37)
    explicit = Field.from_boozer(
        rho=booz.rho,
        Bmag=booz.Bmag,
        I=booz.I,
        G=booz.G,
        iota=booz.iota,
        Psi=booz.Psi,
        R_major=booz.R_major,
        a_minor=booz.a_minor,
        NFP=booz.NFP,
    )
    derived = Field.from_boozer(
        rho=booz.rho,
        Bmag=booz.Bmag,
        I=booz.I,
        G=booz.G,
        iota=booz.iota,
        Psi=booz.Psi,
        NFP=booz.NFP,
    )
    # R_major/a_minor don't enter sqrtg (only Psi does), so the field is identical.
    np.testing.assert_allclose(derived.sqrtg, explicit.sqrtg, rtol=1e-12)
    np.testing.assert_allclose(derived.B_sup_t, explicit.B_sup_t, rtol=1e-12)
    np.testing.assert_allclose(derived.B_sup_z, explicit.B_sup_z, rtol=1e-12)
    # and the estimates land near the supplied geometry.
    np.testing.assert_allclose(derived.R_major, booz.R_major, rtol=8e-2)
    np.testing.assert_allclose(derived.a_minor, booz.a_minor, rtol=5e-2)
