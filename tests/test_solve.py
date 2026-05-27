"""Regression tests comparing yancc to monkes/sfincs etc."""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import elementary_charge, proton_mass

import yancc
from yancc.field import Field
from yancc.preconditioner import DKEMPreconditioner
from yancc.solve import solve_dke, solve_mdke
from yancc.species import JOULE_PER_EV, LocalMaxwellian
from yancc.velocity_grids import (
    MaxwellSpeedGrid,
    QuadraticPitchAngleGrid,
    UniformPitchAngleGrid,
)


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
    if os.environ.get("CI"):
        jax.clear_caches()
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
    field = Field.from_booz_xform(
        config["booz_path"], np.sqrt(config["s"]), nt, nz, cutoff=1e-5
    )
    pitchgrid = UniformPitchAngleGrid(nl)

    erhat = data_fortran["erhat"][idx]
    nuhat = data_fortran["nuhat"][idx]
    print(f"Running i={idx}, nuhat={nuhat:.3e}, erhat={erhat:.3e}")
    t1 = time.perf_counter()
    sol, info = jax.block_until_ready(
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
    print(f"Took {t2 - t1:.3e} s")
    Dij = sol.get("Dij_DKES")

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
    if os.environ.get("CI"):
        jax.clear_caches()
    import desc  # pyright: ignore[reportMissingImports]

    eq = desc.io.load("tests/data/NCSX_output.h5")[-1]

    nt = 17
    nz = 37
    rho = 0.5

    field1 = Field.from_desc(eq, rho, nt, nz)
    field2 = Field.from_vmec("tests/data/wout_NCSX.nc", rho, nt, nz)
    field3 = Field.from_booz_xform("tests/data/boozmn_wout_NCSX.nc", rho, nt, nz)
    field4 = Field.from_ipp_bc("tests/data/NCSX.bc", rho, nt, nz)
    pitchgrid = UniformPitchAngleGrid(73)

    sol1, info1 = solve_mdke(field1, pitchgrid, erhohat, nuhat, verbose=True)
    sol2, info2 = solve_mdke(field2, pitchgrid, erhohat, nuhat, verbose=True)
    sol3, info3 = solve_mdke(field3, pitchgrid, erhohat, nuhat, verbose=True)
    sol4, info4 = solve_mdke(field4, pitchgrid, erhohat, nuhat, verbose=True)

    D1 = sol1.get("Dij")
    D2 = sol2.get("Dij")
    D3 = sol3.get("Dij")
    D4 = sol4.get("Dij")

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


@pytest.mark.parametrize("nuhat", [1e-1, 1e0])
def test_solve_mdke_nonuniform_pitch(field, nuhat):
    """solve_mdke accepts a non-uniform pitch grid and agrees with a uniform one.

    theta/zeta are discretized identically between the two solves, so any
    disagreement isolates the non-uniform pitch-angle handling. At a smooth,
    moderate collisionality both grids resolve the same physics, so the
    monoenergetic coefficients should match.
    """
    if os.environ.get("CI"):
        jax.clear_caches()
    na = 31
    erhohat = 0.0
    sol_u, info_u = solve_mdke(field, UniformPitchAngleGrid(na), erhohat, nuhat)
    sol_q, info_q = solve_mdke(field, QuadraticPitchAngleGrid(na, 0.6), erhohat, nuhat)
    assert info_u["success1"] and info_u["success2"]
    assert info_q["success1"] and info_q["success2"]
    Du = sol_u.get("Dij")
    Dq = sol_q.get("Dij")
    np.testing.assert_allclose(Dq[0, 0], Du[0, 0], rtol=1e-2)
    np.testing.assert_allclose(Dq[2, 2], Du[2, 2], rtol=1e-2)
    np.testing.assert_allclose(Dq[0, 2], Du[0, 2], rtol=1e-2)
    np.testing.assert_allclose(Dq[2, 0], Du[2, 0], rtol=1e-2)


@pytest.mark.parametrize("idx", [0, 10, 20])
def test_solve_dke_ncsx(idx):
    """Test solving DKE vs sfincs."""
    if os.environ.get("CI"):
        jax.clear_caches()
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
    field = Field.from_vmec("tests/data/wout_NCSX.nc", rho, nt, nz)
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
        "particleFlux_vm_rN": sfincs_data[:, 2] / field.a_minor,
        "heatFlux_vm_rN": sfincs_data[:, 3] / field.a_minor,
        "energy_source": sfincs_data[:, 4],
    }

    C_scale = 17 / yancc.species.coulomb_logarithm(species[0], species[0])
    operator_weights = jnp.ones(8).at[-4:].set(C_scale).at[-1:].set(0)

    Er = sfincs_data["Er"][idx]
    print("Er:", Er)

    t0 = time.perf_counter()
    sol, info = solve_dke(
        field,
        pitchgrid,
        speedgrid,
        species,
        Erho=Er * field.a_minor * 1000,  # Er in kV/m
        operator_weights=operator_weights,
        verbose=3,
        rtol=1e-5,
        multigrid_options={"max_grids": 3, "coarse_N": 2000},
    )
    t1 = time.perf_counter()
    print("TIME:", t1 - t0)

    # tolerances could be tighter with higher resolution, but this isn't meant to be
    # a real benchmark, just a quick check for dumb mistakes. These qtys pass though
    # zero so we use an atol set to the average magnitude, so roughly rtol=5%
    np.testing.assert_allclose(
        sol.get("FSABFlow_sfincs"),
        sfincs_data["FSABFlow"][idx],
        atol=5e-2 * float(np.mean(np.abs(sfincs_data["FSABFlow"]))),
    )
    np.testing.assert_allclose(
        sol.get("particleFlux_vm_rN_sfincs"),
        sfincs_data["particleFlux_vm_rN"][idx],
        atol=5e-2 * float(np.mean(np.abs(sfincs_data["particleFlux_vm_rN"]))),
    )
    np.testing.assert_allclose(
        sol.get("heatFlux_vm_rN_sfincs"),
        sfincs_data["heatFlux_vm_rN"][idx],
        atol=5e-2 * float(np.mean(np.abs(sfincs_data["heatFlux_vm_rN"]))),
    )
    # Check the remaining SFINCS-normalization outputs by inverting their
    # normalization and comparing to the raw physical quantity. These use the
    # default kwargs from yancc.solution: Tbar=1 keV, mbar=proton, nbar=1e20,
    # Bbar=1, Rbar=1.
    Tbar = 1e3 * JOULE_PER_EV
    mbar = proton_mass
    nbar = 1e20
    Bbar = 1.0
    Rbar = 1.0
    vbar = np.sqrt(2 * Tbar / mbar)
    density = np.array([sp.density for sp in species])

    # J|| = sum_s q_s n_s V||_s
    qs = np.array([sp.species.charge for sp in species])
    Vpar = np.asarray(sol.get("V||"))
    Jpar_expected = (qs[:, None, None] * density[:, None, None] * Vpar).sum(axis=0)
    np.testing.assert_allclose(sol.get("J||"), Jpar_expected, rtol=1e-10)

    # flow_sfincs = V|| * n_s / (nbar * vbar)
    np.testing.assert_allclose(
        sol.get("flow_sfincs") * (nbar * vbar) / density[:, None, None],
        Vpar,
        rtol=1e-10,
    )

    # FSABjHat_sfincs = <J||B> / (e * nbar * vbar * Bbar)
    np.testing.assert_allclose(
        sol.get("FSABjHat_sfincs") * (elementary_charge * nbar * vbar * Bbar),
        sol.get("<J||B>"),
        rtol=1e-10,
    )

    # j_rN_sfincs = J_rho * Rbar / (e * nbar * vbar)  # noqa: E800
    np.testing.assert_allclose(
        sol.get("j_rN_sfincs") * (elementary_charge * nbar * vbar) / Rbar,
        sol.get("J_rho"),
        rtol=1e-10,
    )

    # jHat_sfincs = J|| / (e * nbar * vbar)
    np.testing.assert_allclose(
        sol.get("jHat_sfincs") * (elementary_charge * nbar * vbar),
        sol.get("J||"),
        rtol=1e-10,
    )


def test_solve_dke_ncsx_with_dkem_preconditioner():
    """Solve the NCSX problem using DKEMPreconditioner (monoenergetic per-x
    preconditioner stack) passed in via the M= kwarg, and check it converges
    to the same answer as the default DKEPreconditioner.
    """
    if os.environ.get("CI"):
        jax.clear_caches()
    rho = 0.5
    nt = 15
    nz = 31
    na = 61
    nx = 6
    field = Field.from_vmec("tests/data/wout_NCSX.nc", rho, nt, nz)
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
    C_scale = 17 / yancc.species.coulomb_logarithm(species[0], species[0])
    operator_weights = jnp.ones(8).at[-4:].set(C_scale).at[-1:].set(0)

    path = "tests/data/20251212-01_sfincs_yancc_benchmark_NCSX_1species_Er_scan.txt"
    sfincs_data = np.loadtxt(path, skiprows=1)
    Er = sfincs_data[10, 0]  # one Er value, midrange
    Erho = Er * field.a_minor * 1000

    M = DKEMPreconditioner(
        field=field,
        pitchgrid=pitchgrid,
        speedgrid=speedgrid,
        species=species,
        Erho=Erho,
        max_grids=3,
        coarse_N=2000,
    )

    sol_dkem, info_dkem = solve_dke(
        field,
        pitchgrid,
        speedgrid,
        species,
        Erho=Erho,
        operator_weights=operator_weights,
        verbose=1,
        rtol=1e-5,
        M=M,
    )
    assert info_dkem["success"]

    # Cross-check against the default DKEPreconditioner result.
    sol_default, info_default = solve_dke(
        field,
        pitchgrid,
        speedgrid,
        species,
        Erho=Erho,
        operator_weights=operator_weights,
        verbose=1,
        rtol=1e-5,
        multigrid_options={"max_grids": 3, "coarse_N": 2000},
    )
    assert info_default["success"]

    for qty in ("<particle_flux>", "<heat_flux>", "<V||B>"):
        a = sol_dkem.get(qty)
        b = sol_default.get(qty)
        np.testing.assert_allclose(a, b, rtol=5e-3, atol=5e-3 * float(np.abs(b).max()))


def _jvp_1_arg(fun, x0, argnum, rel_step, abs_step):
    xh = x0.copy().at[argnum].add(abs_step + rel_step * abs(x0[argnum]))
    xl = x0.copy().at[argnum].add(-abs_step - rel_step * abs(x0[argnum]))
    h = xh[argnum] - xl[argnum]
    fh = fun(xh)
    fl = fun(xl)
    return (fh - fl) / h


def test_solve_dke_multispecies_warm_start(field, species2):
    """Two-species solve, then a warm-started re-solve reusing the subspace U and f1.

    Low resolution and maxiter=2 - this exercises the multi-species operator-weight
    path and the U/f1 recycling branches, not convergence/accuracy.
    """
    if os.environ.get("CI"):
        jax.clear_caches()
    pitchgrid = UniformPitchAngleGrid(7)
    speedgrid = MaxwellSpeedGrid(2)
    Erho = 100.0

    sol, info = solve_dke(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        maxiter=2,
        rtol=1e-12,
        verbose=2,
    )
    size = len(species2) * speedgrid.nx * pitchgrid.na * field.ntheta * field.nzeta
    f1 = np.asarray(sol.f1).reshape(-1)
    assert f1.size == size
    U = info["U"]
    assert U.shape[0] == size + 2 * len(species2)

    # warm-start: feed back the recycled Krylov subspace U and the previous iterate.
    sol2, info2 = solve_dke(
        field,
        pitchgrid,
        speedgrid,
        species2,
        Erho,
        maxiter=2,
        rtol=1e-12,
        verbose=2,
        U=U,
        f1=sol.f1_krylov,
    )
    np.testing.assert_allclose(sol.f1, sol2.f1, atol=1e-12, rtol=1e-8)
    assert info2["nmv"] < info["nmv"]


def test_solve_dke_derivatives(field, pitchgrid, speedgrid):
    # these are super low res, just to test jax logic, not physical correctness
    if os.environ.get("CI"):
        jax.clear_caches()

    def foo(inputs):
        n, T, dn, dT, Er = inputs
        species = [yancc.species.LocalMaxwellian(yancc.species.Hydrogen, T, n, dT, dn)]
        sol, info = solve_dke(
            field, pitchgrid, speedgrid, species, Er, verbose=2, rtol=1e-12
        )
        return jnp.array(
            [sol.get("<particle_flux>"), sol.get("<heat_flux>"), sol.get("<V||B>")]
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


def test_solve_mdke_tokamak_axisymmetric():
    """Axisymmetric (tokamak) MDKE: nzeta=1 reproduces a zeta-resolved solve.

    A tokamak field is independent of zeta, so the toroidal derivative drops
    (d/dzeta == 0) and a single toroidal point (nzeta=1) must reproduce the
    transport coefficients computed on a zeta-resolved grid.
    """
    if os.environ.get("CI"):
        jax.clear_caches()
    import desc.examples  # pyright: ignore[reportMissingImports]

    eq = desc.examples.get("DSHAPE")  # axisymmetric tokamak, NFP=1
    pitchgrid = UniformPitchAngleGrid(31)
    nuhat = 1e-1
    erhohat = 0.0

    field_axi = Field.from_desc(eq, 0.5, 11, 1)
    # nzeta=5 is the smallest zeta-resolved grid the default p1="4d" stencil allows
    field_res = Field.from_desc(eq, 0.5, 11, 5)
    assert field_axi.nzeta == 1
    # the field really is axisymmetric: no toroidal variation of |B|
    np.testing.assert_allclose(field_axi.dBdz, 0.0, atol=1e-12)

    sol_axi, info_axi = solve_mdke(field_axi, pitchgrid, erhohat, nuhat)
    sol_res, _ = solve_mdke(field_res, pitchgrid, erhohat, nuhat)
    assert bool(info_axi["success1"]) and bool(info_axi["success2"])

    Dij_axi = sol_axi.get("Dij_DKES")
    Dij_res = sol_res.get("Dij_DKES")
    # nzeta=1 must match the zeta-resolved solve to solver tolerance
    np.testing.assert_allclose(Dij_axi, Dij_res, rtol=1e-3, atol=1e-5)

    # Onsager symmetry D31 = -D13
    np.testing.assert_allclose(Dij_axi[2, 0], -Dij_axi[0, 2], rtol=1e-2, atol=1e-4)


def test_solve_dke_tokamak_axisymmetric():
    """Axisymmetric (tokamak) DKE: nzeta=1 reproduces a zeta-resolved solve.

    As for the MDKE, a tokamak field is zeta-independent so d/dzeta == 0 and a
    single toroidal point must reproduce the fluxes from a zeta-resolved grid.
    """
    if os.environ.get("CI"):
        jax.clear_caches()
    import desc.examples  # pyright: ignore[reportMissingImports]

    eq = desc.examples.get("DSHAPE")  # axisymmetric tokamak, NFP=1
    pitchgrid = UniformPitchAngleGrid(31)
    speedgrid = MaxwellSpeedGrid(5)

    def solve(nz):
        field = Field.from_desc(eq, 0.5, 15, nz)
        species = [
            LocalMaxwellian(
                yancc.species.Hydrogen,
                0.8e3,
                1.5e20,
                -2e3 * field.a_minor,
                -0.4e20 * field.a_minor,
            )
        ]
        C_scale = 17 / yancc.species.coulomb_logarithm(species[0], species[0])
        operator_weights = jnp.ones(8).at[-4:].set(C_scale).at[-1:].set(0)
        sol, info = solve_dke(
            field,
            pitchgrid,
            speedgrid,
            species,
            Erho=0.0,
            operator_weights=operator_weights,
            rtol=1e-6,
            multigrid_options={"max_grids": 3, "coarse_N": 2000},
        )
        fluxes = np.array(
            [
                sol.get("particleFlux_vm_rN_sfincs"),
                sol.get("heatFlux_vm_rN_sfincs"),
                sol.get("FSABFlow_sfincs"),
            ]
        ).squeeze()
        return fluxes, info

    # nzeta=5 is the smallest zeta-resolved grid the default p1="4d" stencil allows
    fluxes_axi, info_axi = solve(1)
    fluxes_res, _ = solve(5)
    assert bool(info_axi["success"])

    assert np.all(np.isfinite(fluxes_axi))
    # heat flux (component 1) is outward (down-gradient) for these profiles
    assert fluxes_axi[1] > 0
    # nzeta=1 must match the zeta-resolved solve to solver tolerance
    np.testing.assert_allclose(fluxes_axi, fluxes_res, rtol=5e-3, atol=1e-8)
