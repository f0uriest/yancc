"""Regression tests comparing yancc to monkes/sfincs etc."""

import time

import jax
import numpy as np
import pytest

from yancc.field import Field
from yancc.misc import normalize_dkes
from yancc.solve import solve_mdke
from yancc.velocity_grids import UniformPitchAngleGrid


def _read_monkes_dat(path):
    data = np.loadtxt(path, skiprows=1)
    nuhat = data[:, 0]
    erhat = data[:, 1]
    nt = data[:, 2].astype(int)
    nz = data[:, 3].astype(int)
    nl = data[:, 4].astype(int)
    d11 = data[:, 5]
    d31 = data[:, 6]
    d13 = data[:, 7]
    d33 = data[:, 8]
    d33s = data[:, 9]
    walltime = data[:, 10]
    cputime = data[:, 11]

    data = {
        "nuhat": nuhat,
        "erhat": erhat,
        "nt": nt,
        "nz": nz,
        "nl": nl,
        "D11": d11,
        "D31": d31,
        "D13": d13,
        "D33": d33,
        "D33_spitzer": d33s,
        "wall time": walltime,
        "cpu time": cputime,
    }
    return data


# idx are chosen to have nu~1e-2-1e0, er~0,1e-3
# this is a reasonable range for testing at low resolution
@pytest.mark.parametrize("idx", [15, 20, 25, 51, 56, 61])
def test_solve_mdke_w7x_eim(idx):
    config = {
        "dat_path": "tests/data/monkes_Monoenergetic_Database_w7x_eim.dat",
        "booz_path": "tests/data/boozmn_wout_w7x_eim.nc",
        "s": 0.200,
        "K11": 3.6462,
        "K13": 1.9095,
        "K31": 1.9095,
        "K33": 1.0,
        "B0": 2.43114357927517,
    }
    data_fortran = _read_monkes_dat(config["dat_path"])

    nt = 17
    nz = 33
    nl = 65
    field = Field.from_booz_xform(config["booz_path"], config["s"], nt, nz, cutoff=1e-5)
    pitchgrid = UniformPitchAngleGrid(nl)

    erhat = data_fortran["erhat"][idx]
    nuhat = data_fortran["nuhat"][idx]
    print(f"Running i={idx}, nuhat={nuhat:.3e}, erhat={erhat:.3e}")
    t1 = time.perf_counter()
    x, rhs, Dij, info = jax.block_until_ready(
        solve_mdke(
            field,
            pitchgrid,
            erhat * field.a_minor,
            nuhat,
            verbose=2,
            multigrid_options={"coarse_N": 1000},
        )
    )
    t2 = time.perf_counter()
    print(f"Took {t2-t1:.3e} s")
    Dij = normalize_dkes(Dij, field)

    D11_yancc = Dij[0, 0]
    D31_yancc = Dij[2, 0]
    D13_yancc = Dij[0, 2]
    D33_yancc = Dij[2, 2]
    np.testing.assert_allclose(
        D11_yancc,
        data_fortran["D11"][idx],
        rtol=5e-2,
        err_msg=f"i={idx}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
    )
    np.testing.assert_allclose(
        D31_yancc,
        data_fortran["D31"][idx],
        rtol=5e-2,
        atol=1e-4,  # these can be near zero
        err_msg=f"i={idx}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
    )
    np.testing.assert_allclose(
        D13_yancc,
        data_fortran["D13"][idx],
        rtol=5e-2,
        atol=1e-4,  # these can be near zero
        err_msg=f"i={idx}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
    )
    np.testing.assert_allclose(
        D33_yancc,
        data_fortran["D33"][idx],
        rtol=5e-2,
        err_msg=f"i={idx}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
    )

    # also check onsager symmetry
    np.testing.assert_allclose(
        D31_yancc,
        -D13_yancc,
        rtol=1e-2,
        atol=1e-4,  # these can be near zero
        err_msg=f"i={idx}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
    )


@pytest.mark.parametrize("nuhat", [1e-2, 1e1])
@pytest.mark.parametrize("erhohat", [0.0, 1e-3])
def test_solve_field_types(nuhat, erhohat):
    """Test solving the MDKE with the same physical field in different coordinates."""
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
    pitchgrid = UniformPitchAngleGrid(73)

    f1, rhs1, D1, info1 = solve_mdke(field1, pitchgrid, erhohat, nuhat, verbose=True)
    f2, rhs2, D2, info2 = solve_mdke(field2, pitchgrid, erhohat, nuhat, verbose=True)
    f3, rhs3, D3, info3 = solve_mdke(field3, pitchgrid, erhohat, nuhat, verbose=True)
    f4, rhs4, D4, info4 = solve_mdke(field4, pitchgrid, erhohat, nuhat, verbose=True)

    np.testing.assert_allclose(D1[0, 0], D2[0, 0], rtol=1e-2, atol=0)
    np.testing.assert_allclose(D1[0, 0], D3[0, 0], rtol=2e-2, atol=0)
    np.testing.assert_allclose(D1[0, 0], D4[0, 0], rtol=2e-2, atol=0)

    np.testing.assert_allclose(D1[0, 2], D2[0, 2], rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(D1[0, 2], D3[0, 2], rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(D1[0, 2], D4[0, 2], rtol=1e-2, atol=1e-2)

    np.testing.assert_allclose(D1[2, 0], D2[2, 0], rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(D1[2, 0], D3[2, 0], rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(D1[2, 0], D4[2, 0], rtol=1e-2, atol=1e-2)

    np.testing.assert_allclose(D1[2, 2], D2[2, 2], rtol=1e-2, atol=0)
    np.testing.assert_allclose(D1[2, 2], D3[2, 2], rtol=1e-2, atol=0)
    np.testing.assert_allclose(D1[2, 2], D4[2, 2], rtol=1e-2, atol=0)
