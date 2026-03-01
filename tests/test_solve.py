"""Regression tests comparing yancc to monkes/sfincs etc."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import yancc
from yancc.field import Field
from yancc.misc import normalize_dkes, normalize_fluxes_sfincs
from yancc.solve import solve_dke, solve_mdke
from yancc.species import LocalMaxwellian
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid


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


@pytest.mark.parametrize("idx", [0, 10, 20])
def test_solve_dke_ncsx(idx):
    """Test solving DKE vs sfincs."""
    rho = 0.5
    # hydrogen ion charge and mass (no electrons)
    # n = 1.5e20 / m^3     # noqa E800
    # T = 0.8 keV
    # dn/dr = -0.4e20 / m^4
    # dT/dr = -2.0 keV/m
    # full Fokker-Planck collision operator with full trajectories.
    # The collisionality was set using ln(Lambda) = 17.
    # note that increasing this resolution slightly will get reduce errors w/ sfincs
    # to < 1% but we need this to run with limited time/memory on github CI
    nt = 15
    nz = 31
    na = 61
    nx = 6
    field = Field.from_vmec("tests/data/wout_NCSX.nc", rho**2, nt, nz)
    pitchgrid = UniformPitchAngleGrid(na)
    speedgrid = MaxwellSpeedGrid(nx)
    species = [
        LocalMaxwellian(
            yancc.species.Hydrogen,
            0.8e3,
            1.5e20,
            -2e3 * field.a_minor,
            -0.4e20 * field.a_minor,
        )
    ]

    path = "tests/data/20251212-01_sfincs_yancc_benchmark_NCSX_1species_Er_scan.txt"
    sfincs_data = np.loadtxt(path, skiprows=1)
    sfincs_data = {
        "Er": sfincs_data[:, 0],
        "FSABFlow": sfincs_data[:, 1],
        "particleFlux_vm_rHat": sfincs_data[:, 2] / field.a_minor,
        "heatFlux_vm_rHat": sfincs_data[:, 3] / field.a_minor,
        "energy_source": sfincs_data[:, 4],
    }

    C_scale = 17 / yancc.species.coulomb_logarithm(species[0], species[0])
    operator_weights = jnp.ones(8).at[-4:].set(C_scale).at[-1:].set(0)

    Er = sfincs_data["Er"][idx]
    print("Er:", Er)

    t0 = time.perf_counter()
    f, r, flux, info = solve_dke(
        field,
        pitchgrid,
        speedgrid,
        species,
        Erho=Er * field.a_minor * 1000,  # Er in kV/m
        operator_weights=operator_weights,
        verbose=2,
        rtol=1e-5,
    )
    t1 = time.perf_counter()
    print("TIME:", t1 - t0)
    normalized_fluxes = normalize_fluxes_sfincs(
        flux, field, pitchgrid, speedgrid, species
    )

    # tolerances could be tighter with higher resolution, but this isn't meant to be
    # a real benchmark, just a quick check for dumb mistakes. These qtys pass though
    # zero so we use an atol set to the average magnitude, so roughly rtol=5%
    np.testing.assert_allclose(
        normalized_fluxes["FSABFlow"],
        sfincs_data["FSABFlow"][idx],
        atol=5e-2 * float(np.mean(np.abs(sfincs_data["FSABFlow"]))),
    )
    np.testing.assert_allclose(
        normalized_fluxes["particleFlux_vm_rHat"],
        sfincs_data["particleFlux_vm_rHat"][idx],
        atol=5e-2 * float(np.mean(np.abs(sfincs_data["particleFlux_vm_rHat"]))),
    )
    np.testing.assert_allclose(
        normalized_fluxes["heatFlux_vm_rHat"],
        sfincs_data["heatFlux_vm_rHat"][idx],
        atol=5e-2 * float(np.mean(np.abs(sfincs_data["heatFlux_vm_rHat"]))),
    )


def _jvp_1_arg(fun, x0, argnum, rel_step, abs_step):
    xh = x0.copy().at[argnum].add(abs_step + rel_step * abs(x0[argnum]))
    xl = x0.copy().at[argnum].add(-abs_step - rel_step * abs(x0[argnum]))
    h = xh[argnum] - xl[argnum]
    fh = fun(xh)
    fl = fun(xl)
    return (fh - fl) / h


def test_solve_dke_derivatives(field, pitchgrid, speedgrid):
    # these are super low res, just to test jax logic, not physical correctness

    def foo(inputs):
        n, T, dn, dT, Er = inputs
        species = [yancc.species.LocalMaxwellian(yancc.species.Hydrogen, T, n, dT, dn)]
        f, r, fluxes, info = solve_dke(
            field, pitchgrid, speedgrid, species, Er, verbose=0, rtol=1e-12
        )
        return jnp.array(
            [fluxes["<particle_flux>"], fluxes["<heat_flux>"], fluxes["<BV||>"]]
        ).squeeze()

    n = 1e19
    T = 1e3
    dn = -1e19
    dT = -1e3
    Er = 1e3
    inputs = jnp.array([n, T, dn, dT, Er])
    Jfd = jnp.array([_jvp_1_arg(foo, inputs, i, 1e-3, 0.0) for i in range(len(inputs))])
    Jf = jax.jacfwd(foo)(inputs)
    Jr = jax.jacrev(foo)(inputs)
    np.testing.assert_allclose(Jr, Jf, rtol=1e-10)
    np.testing.assert_allclose(Jr, Jfd.T, rtol=1e-6)
    np.testing.assert_allclose(Jf, Jfd.T, rtol=1e-6)
