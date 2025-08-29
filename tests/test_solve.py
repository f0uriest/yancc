"""Regression tests comparing yancc to monkes/sfincs etc."""

import time

import jax
import numpy as np

from yancc.field import Field
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


def test_solve_mdke_w7x_eim():
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

    # these are chosen to have nu~1e-2-1e0, er~0,1e-3
    # this is a reasonable range for testing at low resolution
    idxs = [15, 20, 25, 51, 56, 61]

    nt = 17
    nz = 33
    nl = 65
    field = Field.from_booz_xform(config["booz_path"], config["s"], nt, nz, cutoff=1e-5)
    pitchgrid = UniformPitchAngleGrid(nl)

    for i in idxs:
        erhat = data_fortran["erhat"][i]
        nuhat = data_fortran["nuhat"][i]
        print(f"Running i={i}, nuhat={nuhat:.3e}, erhat={erhat:.3e}")
        t1 = time.perf_counter()
        Dij, x, rhs, info = jax.block_until_ready(
            solve_mdke(
                field, pitchgrid, erhat / field.psi_r * 2 * np.pi, nuhat, coarse_N=1000
            )
        )
        t2 = time.perf_counter()
        print(f"Took {t2-t1:.3e} s")
        print(info)

        D11_yancc = Dij[0, 0] * config["K11"]
        D31_yancc = Dij[2, 0] * config["K31"]
        D13_yancc = Dij[0, 2] * config["K13"]
        D33_yancc = Dij[2, 2] * config["K33"]
        np.testing.assert_allclose(
            D11_yancc,
            data_fortran["D11"][i],
            rtol=5e-2,
            err_msg=f"i={i}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
        )
        np.testing.assert_allclose(
            D31_yancc,
            data_fortran["D31"][i],
            rtol=5e-2,
            atol=1e-4,  # these can be near zero
            err_msg=f"i={i}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
        )
        np.testing.assert_allclose(
            D13_yancc,
            data_fortran["D13"][i],
            rtol=5e-2,
            atol=1e-4,  # these can be near zero
            err_msg=f"i={i}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
        )
        np.testing.assert_allclose(
            D33_yancc,
            data_fortran["D33"][i],
            rtol=5e-2,
            err_msg=f"i={i}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
        )

        # also check onsager symmetry
        np.testing.assert_allclose(
            D31_yancc,
            -D13_yancc,
            rtol=1e-2,
            atol=1e-4,  # these can be near zero
            err_msg=f"i={i}, nuhat={nuhat:.3e}, erhat={erhat:.3e}",
        )
