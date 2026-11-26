"""Tests for magnetic field structure."""

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

import yancc.field
from yancc.field import Field


def _compare_fields_scalars(field1, field2, rtol=1e-5):
    np.testing.assert_allclose(field1.iota, field2.iota, rtol=rtol)
    np.testing.assert_allclose(field1.Bmag_fsa, field2.Bmag_fsa, rtol=rtol)
    np.testing.assert_allclose(field1.B2mag_fsa, field2.B2mag_fsa, rtol=rtol)
    np.testing.assert_allclose(field1.a_minor, field2.a_minor, rtol=5e-2)
    np.testing.assert_allclose(field1.R_major, field2.R_major, rtol=5e-2)
    np.testing.assert_allclose(field1.psi_r, field2.psi_r, rtol=5e-2)
    np.testing.assert_allclose(field1.sqrtg.mean(), field2.sqrtg.mean(), rtol=rtol)
    np.testing.assert_allclose(field1.B0, field2.B0, rtol=1e-2)
    np.testing.assert_allclose(
        field1.flux_surface_average(field1.g_sup_rr / field1.Bmag**2),
        field2.flux_surface_average(field2.g_sup_rr / field2.Bmag**2),
        rtol=1e-3,
    )


def _compare_fields_local(field1, field2):
    np.testing.assert_allclose(field1.Bmag, field2.Bmag, rtol=1e-3)
    np.testing.assert_allclose(field1.sqrtg, field2.sqrtg, rtol=1e-3)
    # g_sup_rr depends on angular derivatives of R, Z, so it is sensitive to
    # truncated Fourier modes in the input files
    np.testing.assert_allclose(field1.g_sup_rr, field2.g_sup_rr, rtol=3e-2)
    np.testing.assert_allclose(field1.B_sub_t, field2.B_sub_t, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(field1.B_sub_z, field2.B_sub_z, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(field1.B_sup_t, field2.B_sup_t, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(field1.B_sup_z, field2.B_sup_z, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(field1.dBdz, field2.dBdz, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(field1.dBdt, field2.dBdt, rtol=1e-2, atol=1e-2)


def test_field_types():
    """Test solving the MDKE with the same field in different formats."""
    # desc imports its submodules lazily, so import one that needs jax to check
    # that the installed desc is compatible
    pytest.importorskip("desc.equilibrium")
    pytest.importorskip("gvec")
    import desc.io  # pyright: ignore[reportMissingImports]

    eq = desc.io.load("tests/data/NCSX_output.h5")[-1]

    nt = 17
    nz = 37
    rho = 0.5

    # fields are built for several surfaces at once to check they work under jit
    def make_fields(r):
        return (
            Field.from_desc(eq, r, nt, nz),
            Field.from_vmec("tests/data/wout_NCSX.nc", r, nt, nz),
            Field.from_booz_xform("tests/data/boozmn_wout_NCSX.nc", r, nt, nz),
            Field.from_ipp_bc("tests/data/NCSX.bc", r, nt, nz),
            Field.from_gvec("tests/data/gvec_NCSX", r, nt, nz),
        )

    fields = jax.jit(jax.vmap(make_fields))(jnp.array([rho, 0.7]))
    field1, field2, field3, field4, field5 = jax.tree.map(lambda x: x[0], fields)

    for field in (field1, field2, field3, field4, field5):
        np.testing.assert_allclose(
            field.B_sub_t * field.B_sup_t + field.B_sub_z * field.B_sup_z,
            field.Bmag**2,
            rtol=1e-5,
        )

    _compare_fields_scalars(field1, field2)
    _compare_fields_scalars(field1, field3)
    _compare_fields_scalars(field1, field4)
    # the GVEC equilibrium is solved independently from the same boundary and
    # profiles, in a generalized frame whose toroidal angle is not the cylindrical
    # angle, so only surface quantities can be compared, to the accuracy to which
    # the two equilibrium solutions agree
    _compare_fields_scalars(field1, field5, rtol=1e-3)
    np.testing.assert_allclose(field5.Psi, field2.Psi, rtol=1e-8)
    # these are ostensibly in the same coordinates, so we can compare local qtys
    _compare_fields_local(field1, field2)
    _compare_fields_local(field3, field4)


def _netcdf_copy(src, dst, negate):
    shutil.copy(src, dst)
    with Dataset(dst, "r+") as f:
        for name in negate:
            f.variables[name][:] = -f.variables[name][:]
    return dst


def _bc_patched(negate):
    """Replacement for the IPP bc reader that negates some of the values read."""
    read = yancc.field._read_bc

    def _read(path):
        data = dict(read(path))
        for key in negate:
            data[key] = -np.asarray(data[key])
        return data

    return _read


def _reversed_field(field):
    """The same field with B -> -B, in the same coordinates."""
    return Field(
        rho=field.rho,
        B_sup_t=-field.B_sup_t,
        B_sup_z=-field.B_sup_z,
        B_sub_t=-field.B_sub_t,
        B_sub_z=-field.B_sub_z,
        Bmag=field.Bmag,
        sqrtg=field.sqrtg,
        Psi=-field.Psi,
        iota=field.iota,
        R_major=field.R_major,
        a_minor=field.a_minor,
        NFP=field.NFP,
        dBdt=field.dBdt,
        dBdz=field.dBdz,
        B0=field.B0,
        g_sup_rr=field.g_sup_rr,
    )


def test_field_orientation(tmp_path, monkeypatch):
    """Readers give right handed coordinates and keep the direction of the field.

    Each source file is modified in two ways: reversing the field (B -> -B, which
    negates the field components and the flux and leaves the geometry unchanged), and
    reversing the poloidal angle of the source, which describes the same field in
    angles of the opposite orientation. The first must give the reversed field, the
    second the same field as the original file.
    """
    rho, nt, nz = 0.5, 9, 11
    wout, booz, bc = (
        "tests/data/wout_NCSX.nc",
        "tests/data/boozmn_wout_NCSX.nc",
        "tests/data/NCSX.bc",
    )
    # reversing the poloidal angle of a series in cos/sin(m theta - n zeta) negates the
    # toroidal mode numbers and the sine coefficients, and the poloidal components,
    # Jacobian and rotational transform
    vmec_flip = ["xn", "xn_nyq", "zmns", "lmns", "bsubumnc", "bsupumnc", "gmnc"]
    vmec_flip += ["iotaf", "iotas", "signgs"]
    booz_flip = ["ixn_b", "zmns_b", "pmns_b", "buco_b", "iota_b", "gmn_b", "phi_b"]
    cases = {
        "vmec": (
            lambda path: Field.from_vmec(path, rho, nt, nz),
            wout,
            lambda: _netcdf_copy(
                wout,
                tmp_path / "wout_reversed.nc",
                ["bsupumnc", "bsupvmnc", "bsubumnc", "bsubvmnc", "phi"],
            ),
            lambda: _netcdf_copy(wout, tmp_path / "wout_flipped.nc", vmec_flip),
        ),
        "booz_xform": (
            lambda path: Field.from_booz_xform(path, rho, nt, nz),
            booz,
            lambda: _netcdf_copy(
                booz, tmp_path / "booz_reversed.nc", ["buco_b", "bvco_b", "phi_b"]
            ),
            lambda: _netcdf_copy(booz, tmp_path / "booz_flipped.nc", booz_flip),
        ),
        "ipp_bc": (
            lambda path: Field.from_ipp_bc(path, rho, nt, nz),
            bc,
            lambda: (
                monkeypatch.setattr(
                    yancc.field, "_read_bc", _bc_patched(["I", "G", "Psi"])
                )
                or bc
            ),
            lambda: (
                monkeypatch.setattr(
                    yancc.field,
                    "_read_bc",
                    _bc_patched(["n", "Zmn", "numn", "I", "iota", "Psi"]),
                )
                or bc
            ),
        ),
    }
    keys = ["Bmag", "sqrtg", "B_sub_t", "B_sub_z", "B_sup_t", "B_sup_z", "dBdt"]
    keys += ["dBdz", "g_sup_rr", "bdotgradB", "BxgradrhodotgradB"]
    for name, (read, path, reverse, flip) in cases.items():
        original = read(path)
        assert np.all(original.sqrtg > 0), name
        for label, make, expected in [
            ("reversed", reverse, _reversed_field(original)),
            ("flipped", flip, original),
        ]:
            field = read(make())
            monkeypatch.undo()
            for key in keys + ["Psi", "iota"]:
                a, b = (
                    np.asarray(getattr(field, key)),
                    np.asarray(getattr(expected, key)),
                )
                np.testing.assert_allclose(
                    a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max(), err_msg=name + label
                )


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
