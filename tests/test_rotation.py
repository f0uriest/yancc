"""Tests for toroidal rotation of the plasma.

The reference data in ``tests/data/neo_sonic_rotation_reference.npz`` are results of
NEO (E. A. Belli and J. Candy) with its full Fokker-Planck collision operator and
sonic rotation. Each case is stored under a prefix: the field on the nodes of the
surface (with the major radius), the species (D, C, e), the Coulomb logarithm and the
rotation as yancc takes them, and the result of NEO in SI units per unit rho.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import elementary_charge, proton_mass

import yancc
from yancc._collisions import (
    EnergyScattering,
    FokkerPlanckLandau,
    PitchAngleScattering,
)
from yancc._linalg import BorderedOperator, banded_to_dense
from yancc._misc import (
    DKEConstraint,
    DKESources,
    _dke_thermodynamic_forces,
    angular_momentum_drift,
    dke_rhs,
    radial_drift,
)
from yancc._tokamak import TokamakPreconditioner
from yancc._trajectories import DKE
from yancc.rotation import RotatingEquilibrium, Rotation
from yancc.solution import DKESolution, _energy_flux
from yancc.solve import solve_dke, solve_dke_ambipolar
from yancc.species import Deuterium, Electron, LocalMaxwellian, Species
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

from .test_trajectories import extract_blocks, to_axorder

Carbon = Species(12, 6)
Tungsten = Species(184, 40)


def miller_field(
    ntheta,
    r=0.5,
    a=1.0,
    R0=3.0,
    q=2.0,
    Bunit=2.0,
    kappa=1.0,
    delta=0.0,
    s_zmag=0.0,
    ipccw=-1,
    btccw=-1,
):
    """Field of a Miller local equilibrium, with the major radius of its points.

    R = R0 + r cos(θ + asin(δ) sin θ), Z = Z0 + κ r sin θ, with rho = r/a,
    Bunit = (q/r) dψ/dr and s_zmag = dZ0/dr. zeta is the cylindrical angle,
    counter-clockwise seen from above, and ipccw, btccw = 1 are a current and a
    toroidal field in that direction.
    """
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    x = np.arcsin(delta)
    arg = theta + x * np.sin(theta)
    R = R0 + r * np.cos(arg)
    R_r = np.cos(arg)
    R_t = -r * np.sin(arg) * (1 + x * np.cos(theta))
    Z_r = s_zmag + kappa * np.sin(theta)
    Z_t = kappa * r * np.cos(theta)
    jac = R_r * Z_t - R_t * Z_r
    dl = np.hypot(R_t, Z_t)
    psi_r = r * abs(Bunit) / abs(q)
    F = r * abs(Bunit) / np.mean(jac / R)
    Bp = psi_r * dl / jac / R
    st, sz = -np.sign(ipccw), np.sign(btccw)
    return yancc.Field(
        rho=r / a,
        B_sup_t=(st * Bp / dl)[:, None],
        B_sup_z=(sz * F / R**2)[:, None],
        B_sub_t=(st * Bp * dl)[:, None],
        B_sub_z=(sz * F * np.ones(ntheta))[:, None],
        Bmag=np.hypot(Bp, F / R)[:, None],
        sqrtg=(-a * R * jac)[:, None],
        iota=st * sz / abs(q),
        R_major=R0,
        a_minor=a,
        R=R[:, None],
        dRdrho=(a * R_r)[:, None],
    )


def rotating_species(T=2e3, ne=5e19, nC=0.0167 * 5e19):
    """D + C + e with a/Ln = 1 and a/LT = 3."""
    return [
        LocalMaxwellian(Deuterium, T, ne - 6 * nC, -3 * T, -(ne - 6 * nC)),
        LocalMaxwellian(Carbon, T, nC, -3 * T, -nC),
        LocalMaxwellian(Electron, T, ne, -3 * T, -ne),
    ]


def mach_omega(mach, T=2e3):
    """Angular velocity at which deuterium has a given Mach number at R = 3.5 m."""
    return mach * np.sqrt(2 * T * elementary_charge / Deuterium.mass) / 3.5


@pytest.fixture(scope="module")
def reference():
    """NEO reference data."""
    return np.load("tests/data/neo_sonic_rotation_reference.npz")


def reference_cases(reference, kind):
    """Names of the reference cases of one kind."""
    start, end = kind + "_", "_omega"
    return sorted(
        key[len(start) : -len(end)]
        for key in reference.files
        if key.startswith(start) and key.endswith(end)
    )


def reference_case(reference, prefix):
    """Field, species, rotation and Coulomb logarithm of a reference case."""
    field = yancc.Field(
        **{
            key: reference[prefix + "field_" + key]
            for key in [
                "rho",
                "B_sup_t",
                "B_sup_z",
                "B_sub_t",
                "B_sub_z",
                "Bmag",
                "sqrtg",
                "Psi",
                "iota",
                "R_major",
                "a_minor",
                "NFP",
                "B0",
                "R",
                "dRdrho",
            ]
        }
    )
    species = [
        LocalMaxwellian(Species(m / proton_mass, q / elementary_charge), T, n, dT, dn)
        for m, q, T, n, dT, dn in zip(
            *(
                reference[prefix + key]
                for key in [
                    "mass",
                    "charge",
                    "temperature",
                    "density",
                    "dTdrho",
                    "dndrho",
                ]
            )
        )
    ]
    rotation = Rotation(reference[prefix + "omega"], reference[prefix + "domega_drho"])
    return field, species, rotation, reference[prefix + "lnlambda"]


def test_rotating_equilibrium_two_species():
    """Potential and densities of a rotating plasma of deuterium and electrons."""
    field = miller_field(17)
    assert field.R is not None
    omega, T = mach_omega(0.8), 2e3
    dR2 = (field.R**2 - field.R[0, 0] ** 2)[None]
    m_T = lambda sp: sp.mass / (T * elementary_charge)

    species = [
        LocalMaxwellian(Deuterium, T, 5e19, -3 * T, -5e19),
        LocalMaxwellian(Electron, T, 5e19, -3 * T, -5e19),
    ]
    eq = RotatingEquilibrium(field, species, Rotation(omega))
    # e Φ/T = (m_i - m_e) ω² (R² - R(0)²)/(4T), and equal densities
    phi = (m_T(Deuterium) - m_T(Electron)) * omega**2 * dR2[0] / 4
    np.testing.assert_allclose(eq.Phi / T, phi, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(eq.density[0], eq.density[1], rtol=1e-12)
    np.testing.assert_allclose(
        eq.density[0],
        np.exp((m_T(Deuterium) + m_T(Electron)) * omega**2 * dR2[0] / 4),
        rtol=1e-10,
    )
    assert eq.Phi[0, 0] == 0
    assert np.all(eq.lam[:, 0, 0] == 0)

    # trace tungsten in that plasma, as a kinetic and as a background species
    trace = LocalMaxwellian(Tungsten, T, 5e9, -3 * T, -5e9)
    lam = 40 * phi - m_T(Tungsten) * omega**2 * dR2[0] / 2
    eq3 = RotatingEquilibrium(field, [*species, trace], Rotation(omega))
    # tolerance is the tungsten density over the electron density
    np.testing.assert_allclose(eq3.lam[2], lam, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(eq3.density[2], np.exp(-lam), rtol=1e-6)
    # pushed to the outboard side
    assert eq3.density[2].min() < 1e-3
    eqb = RotatingEquilibrium(field, species, Rotation(omega), background=[trace])
    np.testing.assert_allclose(eqb.background_density[0], eq3.density[2], rtol=1e-12)
    assert eqb.density.shape == (2, 17, 1)


def test_rotating_equilibrium_impurity():
    """Quasineutrality and gradients of the potential energy with an impurity."""
    field = miller_field(17)
    assert field.R is not None
    omega, T = mach_omega(0.8), 2e3
    species = rotating_species(T)
    eq = RotatingEquilibrium(field, species, Rotation(omega))
    Z = np.array([sp.species.charge / elementary_charge for sp in species])
    n = np.array([sp.density for sp in species])
    m_T = np.array([sp.species.mass / (T * elementary_charge) for sp in species])
    dR2 = field.R**2 - field.R[0, 0] ** 2

    charge = np.einsum("s,stz->tz", Z * n, eq.density)
    np.testing.assert_allclose(charge, 0, atol=1e-10 * n[2])
    np.testing.assert_allclose(
        eq.lam[1], 6 * eq.Phi / T - m_T[1] * omega**2 * dR2 / 2, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(eq.dlamdt, field._dfdt(eq.lam), rtol=1e-8, atol=1e-10)

    # gradient of the potential energy at theta=0 against a difference of two surfaces
    shear, h = -0.7 * omega, 1e-5
    eq = RotatingEquilibrium(field, species, Rotation(omega, shear))

    def energy(d):
        R = miller_field(17, r=0.5 + d).R
        assert R is not None
        return -((omega + shear * d) ** 2) * R[0, 0] ** 2 / 2

    np.testing.assert_allclose(
        eq.dlnNdrho, m_T * (energy(h) - energy(-h)) / (2 * h), rtol=1e-8
    )


def test_rotating_equilibrium_derivatives():
    """Derivatives of the equilibrium through the solve for the potential."""
    field = miller_field(17)
    species = rotating_species()
    omega = mach_omega(0.8)

    def fun(x):
        sp = [
            LocalMaxwellian(s.species, x[1] * s.temperature, s.density, 0.0, 0.0)
            for s in species
        ]
        eq = RotatingEquilibrium(field, sp, Rotation(x[0] * omega, -omega))
        return jnp.concatenate(
            [eq.Phi[::4, 0], eq.density[:, 8, 0], eq.dlamdt[:, 3, 0], eq.dlnNdrho]
        )

    x0 = jnp.ones(2)
    h = 1e-6
    J_fd = jnp.stack(
        [
            (fun(x0.at[i].add(h)) - fun(x0.at[i].add(-h))) / (2 * h)
            for i in range(x0.size)
        ],
        axis=1,
    )
    J_fwd = jax.jacfwd(fun)(x0)
    J_rev = jax.jacrev(fun)(x0)
    scale = jnp.abs(J_fwd).max(axis=1, keepdims=True)
    np.testing.assert_allclose(J_fwd / scale, J_rev / scale, atol=1e-12)
    # tolerance is the truncation error of the finite difference
    np.testing.assert_allclose(J_fwd / scale, J_fd / scale, atol=1e-7)


def test_rotation_requirements(field):
    """Rotation needs an axisymmetric field with a major radius."""
    species = rotating_species()
    with pytest.raises(ValueError, match="axisymmetric"):
        RotatingEquilibrium(field, species, Rotation(1e4))
    tokamak = miller_field(9)
    no_R = yancc.Field(
        tokamak.rho,
        tokamak.B_sup_t,
        tokamak.B_sup_z,
        tokamak.B_sub_t,
        tokamak.B_sub_z,
        tokamak.Bmag,
        tokamak.sqrtg,
    )
    with pytest.raises(ValueError, match="dRdrho"):
        RotatingEquilibrium(no_R, species, Rotation(1e4))


@pytest.mark.parametrize(
    "case, neutral",
    [
        ("circular_rigid1", True),
        ("circular_rigid3", True),
        ("circular_shear", True),
        ("mastu_r025_rot", False),
        ("mastu_r045_rot", False),
        ("mastu_r045_double", False),
        ("mastu_r085_rot", False),
    ],
)
def test_rotating_equilibrium_neo(reference, case, neutral):
    """Potential and densities on the surface against NEO."""
    prefix = "equilibrium_" + case + "_"
    field, species, rotation, _ = reference_case(reference, prefix)
    eq = RotatingEquilibrium(field, species, rotation)
    # NEO's poloidal nodes are every other node of the field
    Phi_neo = reference[prefix + "Phi"]
    nodes = np.where(np.isfinite(Phi_neo))[0]
    Te, ne = species[-1].temperature, species[-1].density
    dPhi = np.abs(eq.Phi[nodes, 0] - Phi_neo[nodes]).max() / Te
    dn = np.abs(
        eq.density[:, nodes, 0].T / reference[prefix + "density_ratio"][nodes] - 1
    ).max()
    defect = abs(sum(sp.species.charge * sp.density for sp in species)) / (
        elementary_charge * ne
    )
    if neutral:
        # agreement to the digits NEO prints, up to a carbon Mach number of 2.4
        assert defect < 1e-12
        assert dPhi < 2e-6 * np.abs(Phi_neo[nodes]).max() / Te
        assert dn < 2e-6
    else:
        # These inputs are not neutral. NEO restores neutrality with a potential at
        # the reference point as well, here the charge density of the reference point
        # is kept on the surface. The two differ by the order of the charge defect.
        assert 1e-5 < defect < 1e-3
        assert dPhi < 5 * defect
        assert dn < 15 * defect


@pytest.fixture(scope="module")
def collisions():
    """Small grids and the equilibrium of a strongly rotating plasma."""
    field = miller_field(5)
    pitchgrid, speedgrid = UniformPitchAngleGrid(7), MaxwellSpeedGrid(3)
    species = rotating_species()
    rotating = RotatingEquilibrium(field, species, Rotation(mach_omega(1.0)))
    return field, pitchgrid, speedgrid, species, rotating


def _sizes(field, pitchgrid, speedgrid, species):
    return {
        "s": len(species),
        "x": speedgrid.nx,
        "a": pitchgrid.nalpha,
        "t": field.ntheta,
        "z": field.nzeta,
    }


def _check_diagonals(op, sizes, atol_diag=1e-12, atol=1e-12):
    """Diagonal, block diagonals and row sums of op against its matrix."""
    A = np.asarray(op.as_matrix())
    scale = np.abs(A).max()
    np.testing.assert_allclose(np.diag(A), op.diagonal(), atol=atol_diag * scale)
    for axorder in ["sxatz", "sxazt", "sxtza", "satzx", "xatzs"]:
        n = sizes[axorder[-1]]
        B = extract_blocks(to_axorder(A, sizes, axorder), n)
        np.testing.assert_allclose(
            B, op.block_diagonal(axorder=axorder), atol=atol * scale, err_msg=axorder
        )
        bw = n // 2
        D = banded_to_dense(bw, bw, op.block_diagonal("banded", bw, axorder))
        np.testing.assert_allclose(B, D, atol=atol * scale, err_msg=axorder)
    np.testing.assert_allclose(
        op.abs_row_sum(), np.abs(A).sum(axis=1), rtol=1e-9, atol=1e-12 * scale
    )


@pytest.mark.parametrize("energy_exchange", [True, False])
def test_rotating_collision_diagonals(collisions, energy_exchange):
    """Diagonals of the collision operators with densities that vary on the surface."""
    field, pitchgrid, speedgrid, species, rotating = collisions
    C = FokkerPlanckLandau(
        field,
        pitchgrid,
        speedgrid,
        species,
        operator_weights=jnp.array([1.0, 2.0, 3.0]),
        energy_exchange=energy_exchange,
        coulomb_log=16.0,
        rotating=rotating,
    )
    sizes = _sizes(field, pitchgrid, speedgrid, species)
    for op in [C.CL, C.CE, C.CF]:
        _check_diagonals(op, sizes)
    if energy_exchange:
        _check_diagonals(C, sizes)
    else:
        # The diagonal and blocks are those of the local terms, while the matrix also
        # holds the surface averaged exchange, which couples every point of the
        # surface. The row sums include it.
        _check_diagonals(C, sizes, atol_diag=1e-4, atol=1e-2)


def test_rotating_collisions_local_density(collisions):
    """At each point the operator is that of a plasma with the densities there."""
    field, pitchgrid, speedgrid, species, rotating = collisions
    args = (field, pitchgrid, speedgrid)
    kwargs = {"energy_exchange": True, "coulomb_log": 16.0}
    C = FokkerPlanckLandau(*args, species, rotating=rotating, **kwargs)
    n = np.prod(C.as_matrix().shape[:1]) // field.ntheta
    A = np.asarray(C.as_matrix()).reshape(n, field.ntheta, n, field.ntheta)
    for t in (1, 3):
        local = [
            LocalMaxwellian(
                sp.species, sp.temperature, sp.density * rotating.density[i, t, 0], 0, 0
            )
            for i, sp in enumerate(species)
        ]
        At = np.asarray(FokkerPlanckLandau(*args, local, **kwargs).as_matrix())
        At = At.reshape(n, field.ntheta, n, field.ntheta)
        # denser on the outboard side, and no coupling between the points
        assert rotating.density[1, t, 0] < 0.9
        np.testing.assert_allclose(
            A[:, t, :, t], At[:, t, :, t], rtol=1e-10, atol=1e-12 * np.abs(At).max()
        )
        assert np.all(A[:, t, :, t - 1] == 0)


def test_rotating_collisions_background(collisions):
    """Background species take part in the equilibrium and scatter by their density."""
    field, pitchgrid, speedgrid, species, rotating = collisions
    args = (field, pitchgrid, speedgrid)
    kinetic, background = [species[0], species[2]], [species[1]]
    eq = RotatingEquilibrium(field, kinetic, Rotation(rotating.omega), background)
    np.testing.assert_allclose(eq.density, rotating.density[::2], rtol=1e-12)
    np.testing.assert_allclose(eq.background_density, rotating.density[1:2], rtol=1e-12)
    n = speedgrid.nx * pitchgrid.nalpha * field.ntheta
    for op in [PitchAngleScattering, EnergyScattering]:
        # these do not couple the species
        A = np.asarray(op(*args, species, rotating=rotating).as_matrix())
        Ab = np.asarray(op(*args, kinetic, background, rotating=eq).as_matrix())
        A = A.reshape(3, n, 3, n)[::2, :, ::2].reshape(Ab.shape)
        np.testing.assert_allclose(Ab, A, rtol=1e-10, atol=1e-12 * np.abs(A).max())


def test_rotating_collisions_exchange(collisions):
    """Surface averaged density and energy of each species are conserved, and the
    Maxwellians of the rotating plasma are null.

    As test_collision_species_energy_projection, with the energy x² + λ and the
    densities that vary on the surface.
    """
    field, pitchgrid, speedgrid, species, rotating = collisions
    args = (field, pitchgrid, speedgrid, species)
    ns, nx, na = len(species), speedgrid.nx, pitchgrid.nalpha
    nt, nz = field.ntheta, field.nzeta
    x = np.asarray(speedgrid.x)
    F = np.array([np.asarray(sp(speedgrid.x * sp.v_thermal)) for sp in species])
    wv = (x**2 * np.asarray(speedgrid.wx))[:, None] * np.asarray(pitchgrid.wxi)[None]
    wr = np.asarray(field.sqrtg * field.wtheta[:, None] * field.wzeta[None, :])
    wr = wr / wr.sum()
    density, lam = np.asarray(rotating.density), np.asarray(rotating.lam)

    # perturbations of the density and the temperature of each species, and the
    # surface averaged density and energy moments of each species
    U, Mo = [], []
    for s in range(ns):
        for p in (np.ones((nx, nt, nz)), x[:, None, None] ** 2 + lam[s]):
            u = np.zeros((ns, nx, na, nt, nz))
            u[s] = (p * F[s][:, None, None] * density[s])[:, None]
            U.append(u.flatten())
            m = np.zeros((ns, nx, na, nt, nz))
            m[s] = wv[:, :, None, None] * p[:, None] * wr
            Mo.append(m.flatten())
    U, Mo = np.array(U).T, np.array(Mo)

    C = FokkerPlanckLandau(*args, coulomb_log=16.0, rotating=rotating)
    A = np.asarray(C.as_matrix())
    right = A @ U
    left = Mo @ (A - np.asarray(C.CL.as_matrix()))
    np.testing.assert_allclose(right, 0, atol=1e-12 * (np.abs(A) @ np.abs(U)).max())
    np.testing.assert_allclose(left, 0, atol=1e-12 * (np.abs(Mo) @ np.abs(A)).max())

    # The functions that carry the correction of the energy hold no particles at any
    # point, so that it moves no particles, and with them no angular momentum, along
    # the surface.
    assert C._exchange_factors is not None
    carriers = np.asarray(C._exchange_factors[0][ns : 2 * ns])
    wx = x**2 * np.asarray(speedgrid.wx)
    np.testing.assert_allclose(
        np.einsum("x,ksxtz->kstz", wx, carriers),
        0,
        atol=1e-12 * np.einsum("x,ksxtz->kstz", wx, np.abs(carriers)).max(),
    )

    # with the exchange a temperature perturbation moves energy between the species
    C = FokkerPlanckLandau(
        *args, energy_exchange=True, coulomb_log=16.0, rotating=rotating
    )
    assert C._exchange_factors is None
    Ax = np.asarray(C.as_matrix())
    energy = Mo[1::2] @ Ax @ U[:, 1::2]
    assert np.all(np.abs(energy) > 1e-6 * (np.abs(Mo[1::2]) @ np.abs(Ax @ U[:, 1::2])))


def test_rotation_zero(collisions):
    """Without rotation the operator is that of a plasma at rest."""
    field = miller_field(9)
    pitchgrid, speedgrid = UniformPitchAngleGrid(11), MaxwellSpeedGrid(4)
    species = rotating_species()
    for sp, background in [(species, []), (species[::2], species[1:2])]:
        args = (field, pitchgrid, speedgrid, sp, 0.0, background)
        A0 = DKE(*args, p1="4z", coulomb_log=16.0)
        Ar = DKE(*args, p1="4z", coulomb_log=16.0, rotation=Rotation(0.0, 0.0))
        assert A0.rotating is None
        f = jnp.cos(jnp.arange(1.0, A0.in_size() + 1))
        out0 = A0.mv(f)
        np.testing.assert_allclose(
            Ar.mv(f), out0, rtol=1e-12, atol=1e-12 * float(jnp.abs(out0).max())
        )


def test_rotating_dke_diagonals(collisions):
    """Diagonals of the DKE with the forces of the rotating frame."""
    field, pitchgrid, speedgrid, species, _ = collisions
    A = DKE(
        field,
        pitchgrid,
        speedgrid,
        species,
        1e3,
        p1="2d",
        operator_weights=jnp.linspace(1, 5, 8),
        coulomb_log=16.0,
        rotation=Rotation(mach_omega(1.0)),
    )
    sizes = _sizes(field, pitchgrid, speedgrid, species)
    # the forces change the advection in speed and pitch
    A0 = DKE(field, pitchgrid, speedgrid, species, 1e3, p1="2d")
    for op, op0 in [(A._opx, A0._opx), (A._opa, A0._opa)]:
        _check_diagonals(op, sizes)
        change = np.abs(op.as_matrix() - op0.as_matrix()).max()
        assert change > 0.1 * np.abs(op.as_matrix()).max()
    # the surface averaged exchange of the collision operator is not in the blocks
    Am = np.asarray(A.as_matrix())
    scale = np.abs(Am).max()
    np.testing.assert_allclose(np.diag(Am), A.diagonal(), atol=1e-4 * scale)
    for axorder in ["sxatz", "sxazt", "sxtza", "satzx", "xatzs"]:
        B = extract_blocks(to_axorder(Am, sizes, axorder), sizes[axorder[-1]])
        np.testing.assert_allclose(
            B, A.block_diagonal(axorder=axorder), atol=1e-2 * scale, err_msg=axorder
        )


def _collisionless(ntheta):
    """Collisionless DKE of a rotating plasma, with its Maxwellians and invariants.

    Returns the operator, the Maxwellians F of the local density, the energy over the
    temperature, I v||/B + ω R² and the square of the angular momentum over m², all
    (ns, nx, na, nt, nz), and the size v_th F/(q R) of the streaming term, (ns,).
    """
    field = miller_field(ntheta)
    pitchgrid, speedgrid = UniformPitchAngleGrid(33), MaxwellSpeedGrid(5)
    species = rotating_species()
    A = DKE(
        field,
        pitchgrid,
        speedgrid,
        species,
        0.0,
        p1="4z",
        operator_weights=jnp.array([1.0, 1, 1, 1, 0, 0, 0, 0]),
        rotation=Rotation(mach_omega(0.7)),
    )
    eq = A.rotating
    assert eq is not None and field.R is not None
    vth = jnp.array([sp.v_thermal for sp in species])[:, None, None, None, None]
    v = speedgrid.x[:, None, None, None] * vth
    xi = pitchgrid.xi[:, None, None]
    F = jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species])
    F = F[:, :, None, None, None] * eq.density[:, None, None] * jnp.ones_like(xi)
    energy = speedgrid.x[:, None, None, None] ** 2 + eq.lam[:, None, None]
    B, I, R = field.Bmag, field.B_sub_z, field.R
    g1 = I * v * xi / B + eq.omega * R**2
    g2 = g1**2 + v**2 * (1 - xi**2) * (R**2 * B**2 - I**2) / (2 * B**2)
    scale = jnp.array([sp.v_thermal * sp(0.0) / (2 * 3.0) for sp in species])
    return A, F, energy, g1, g2, scale[:, None, None, None, None]


def test_rotating_maxwellian_is_stationary():
    """Maxwellians in the energy of the rotating frame are collisionless solutions."""
    errors = []
    for ntheta in (17, 33):
        A, F, energy, _, _, scale = _collisionless(ntheta)
        LF = A.mv(F.flatten()).reshape(F.shape)
        LEF = A.mv((energy * F).flatten()).reshape(F.shape)
        errors.append([float(jnp.abs(g / scale).max()) for g in (LF, LEF)])
    errors = np.array(errors)
    assert np.all(errors[1] < 2e-3)
    # converging with the poloidal resolution
    assert np.all(errors[1] < errors[0] / 4)


def test_rotating_drifts():
    """Drifts against the collisionless operator.

    They are the rates of change of the canonical angular momentum and of its square
    along the collisionless trajectories (Sugama & Horton 1997, eq 11).
    """
    errors = []
    for ntheta in (17, 33):
        A, F, _, g1, g2, scale = _collisionless(ntheta)
        field = A.field
        args = (field, A.speedgrid, A.pitchgrid, A.species, A.rotating)
        drift, momentum = radial_drift(*args), angular_momentum_drift(*args)
        m = jnp.array([sp.species.mass for sp in A.species])[:, None, None, None, None]
        q = jnp.array([sp.species.charge for sp in A.species])[
            :, None, None, None, None
        ]
        dpsidrho = field.sqrtg[0, 0] * field.B_sup_t[0, 0]
        dg1 = A.mv((F * g1).flatten()).reshape(F.shape) * m / q / dpsidrho
        dg2 = A.mv((F * g2).flatten()).reshape(F.shape) * m**2 / q / (2 * dpsidrho)
        # in units of a typical value of each drift
        errors.append(
            [
                float(
                    jnp.abs(
                        (dg - F * d)
                        / jnp.abs(d).max(axis=(1, 2, 3, 4), keepdims=True)
                        / scale
                    ).max()
                )
                for dg, d in ((dg1, drift), (dg2, momentum))
            ]
        )
    errors = np.array(errors)
    assert np.all(errors[1] < 2e-3)
    assert np.all(errors[1] < errors[0] / 4)


def test_rotating_drive():
    """Drive of the DKE in a rotating plasma."""
    field = miller_field(13)
    pitchgrid, speedgrid = UniformPitchAngleGrid(17), MaxwellSpeedGrid(4)
    species = rotating_species()
    args = (field, pitchgrid, speedgrid)
    omega = mach_omega(0.6)
    eq = RotatingEquilibrium(field, species, Rotation(omega, -2 * omega))
    ns = len(species)

    # the combined drive is the sum of the unit drives times the forces
    unit = dke_rhs(*args, species, 10.0, single_rhs=False, rotating=eq)
    size = ns * speedgrid.nx * pitchgrid.nalpha * field.ntheta
    assert unit.shape == (3 * ns + 1, size)
    combined = dke_rhs(*args, species, 10.0, 0.02, rotating=eq)
    # ordered by species and then by force, followed by the rotation shear
    forces = _dke_thermodynamic_forces(species, field, 10.0, 0.02, eq)
    coeffs = jnp.append(forces.T.flatten(), eq.domega_drho)
    np.testing.assert_allclose(
        coeffs @ unit, combined, rtol=1e-12, atol=1e-12 * np.abs(combined).max()
    )
    assert np.abs(unit[-1]).max() > 0

    # without rotation it is the drive of a plasma at rest
    rest = RotatingEquilibrium(field, species, Rotation(0.0, 0.0))
    np.testing.assert_allclose(
        dke_rhs(*args, species, 10.0, 0.02, rotating=rest),
        dke_rhs(*args, species, 10.0, 0.02),
        rtol=1e-13,
    )

    # rigid rotation with densities and temperatures that are constant along the
    # field lines and have no gradient across them: no drive
    eq = RotatingEquilibrium(field, species, Rotation(omega))
    flat = [
        LocalMaxwellian(sp.species, sp.temperature, sp.density, 0.0, -sp.density * dlnN)
        for sp, dlnN in zip(species, eq.dlnNdrho)
    ]
    rhs = dke_rhs(*args, flat, 0.0, rotating=eq)
    rhs0 = dke_rhs(*args, species, 0.0, rotating=eq)
    assert float(jnp.abs(rhs).max()) <= 1e-12 * float(jnp.abs(rhs0).max())
    # the density at the outboard midplane rises with the major radius there
    assert np.all(eq.dlnNdrho < 0)


def test_angular_momentum_requires_major_radius(field, pitchgrid, speedgrid, species2):
    """The angular momentum needs the major radius of an axisymmetric field."""
    with pytest.raises(ValueError, match="axisymmetric"):
        angular_momentum_drift(field, speedgrid, pitchgrid, species2)


_FLUXES = ("<particle_flux>", "<heat_flux>", "<angular_momentum_flux>")


def _fluxes(sol, kind=""):
    """Fluxes of particles, heat and angular momentum, (3, ns)."""
    return jnp.array([sol.get(key.replace("<", "<" + kind)) for key in _FLUXES])


def _dense_solutions(field, pitchgrid, speedgrid, species, rotation, single_rhs=True):
    """Solutions of the DKE by a dense solve.

    One for the drive by the gradients of the species and the rotation shear, or one
    for each unit drive.
    """
    A = DKE(
        field,
        pitchgrid,
        speedgrid,
        species,
        0.0,
        p1="4z",
        coulomb_log=16.0,
        rotation=rotation,
    )
    B = DKESources(field, pitchgrid, speedgrid, species)
    C = DKEConstraint(field, pitchgrid, speedgrid, species, True)
    args = (field, pitchgrid, speedgrid, species)
    rhs = dke_rhs(*args, 0.0, single_rhs=single_rhs, rotating=A.rotating)
    rhs = jnp.atleast_2d(rhs)
    full = jnp.concatenate([rhs, jnp.zeros((rhs.shape[0], B.in_size()))], axis=1)
    f = jnp.linalg.solve(BorderedOperator(A, B, C).as_matrix(), full.T).T
    F0 = jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species])
    zero = jnp.array(0.0)
    sols = [
        DKESolution(F0, fi, ri, *args, zero, zero, [], A.rotating)
        for fi, ri in zip(f, rhs)
    ]
    return sols


def _dense_solution(field, pitchgrid, speedgrid, species, rotation):
    """Solution of the DKE by a dense solve."""
    return _dense_solutions(field, pitchgrid, speedgrid, species, rotation)[0]


def _maxwellian_solution(field, species, rotation):
    """Solution without perturbation, for the outputs that do not depend on it."""
    pitchgrid, speedgrid = UniformPitchAngleGrid(9), MaxwellSpeedGrid(3)
    rotating = None
    if rotation is not None:
        rotating = RotatingEquilibrium(field, species, rotation)
    F0 = jnp.array([sp(speedgrid.x * sp.v_thermal) for sp in species])
    f1 = jnp.zeros((len(species), speedgrid.nx, pitchgrid.nalpha, field.ntheta, 1))
    zero = jnp.array(0.0)
    return DKESolution(
        F0, f1, f1, field, pitchgrid, speedgrid, species, zero, zero, [], rotating
    )


def _transport_matrix(omega, nx):
    """Fluxes (Γ_s, Q_s/T_s, Σ_s Π_s/T_s) per unit of the forces (A1_s, A2_s, dω/dρ)."""
    field = miller_field(13)
    pitchgrid, speedgrid = UniformPitchAngleGrid(17), MaxwellSpeedGrid(nx)
    species = rotating_species()
    sols = _dense_solutions(
        field, pitchgrid, speedgrid, species, Rotation(omega), single_rhs=False
    )
    T = jnp.array([sp.temperature for sp in species]) * elementary_charge
    L = []
    for sol in sols:
        G, Q, P = _fluxes(sol)
        L.append(jnp.concatenate([G, Q / T, jnp.sum(P / T)[None]]))
    # the drives are ordered by species and then by force (A1, A2, A3)
    ns = len(species)
    forces = [3 * s + k for k in range(2) for s in range(ns)] + [3 * ns]
    return np.array(L).T[:, forces]


def test_rotating_onsager_symmetry():
    """Onsager symmetry and ambipolarity of the transport matrix.

    Sugama & Horton 1997, eqs 24-25: on an up-down symmetric surface the transport
    matrix is symmetric. The matrix of the gradients and the viscosity are even in ω,
    the coupling of the gradients to the rotation shear is odd in ω.
    """
    omega = mach_omega(0.5)
    Z = np.array([1.0, 6.0, -1.0])
    # The discrete matrix approaches this with the number of speeds: the particle flux
    # of deuterium is the small remainder left by momentum conservation.
    asymmetry, ambipolarity = [], []
    for nx in (4, 6):
        L = _transport_matrix(omega, nx)
        scale = np.sqrt(np.abs(np.diag(L)))
        asymmetry.append(np.abs((L - L.T) / (scale[:, None] * scale)).max())
        ambipolarity.append(np.abs(Z @ L[:3]).max() / np.abs(Z[:, None] * L[:3]).max())
    assert asymmetry[1] < 0.03 and asymmetry[1] < asymmetry[0] / 3
    # for every force
    assert ambipolarity[1] < 0.01 and ambipolarity[1] < ambipolarity[0] / 3
    # fluxes down the gradients: positive entropy production
    assert np.all(np.diag(L) < 0)
    # the coupling is not small
    assert np.abs(L[:6, 6]).max() > 0.05 * np.sqrt(np.abs(L[0, 0] * L[6, 6]))
    # to the accuracy of the dense solve
    Lm = _transport_matrix(-omega, 6)
    np.testing.assert_allclose(Lm[:6, :6], L[:6, :6], rtol=1e-5)
    np.testing.assert_allclose(Lm[:6, 6], -L[:6, 6], rtol=1e-5)
    np.testing.assert_allclose(Lm[6, :6], -L[6, :6], rtol=1e-5)
    np.testing.assert_allclose(Lm[6, 6], L[6, 6], rtol=1e-5)


def test_rotating_field_directions():
    """Fluxes with the field, the current and the rotation reversed.

    The fluxes of particles and energy do not depend on the directions of the field
    and of the rotation, the flux of angular momentum follows the rotation and its
    shear.
    """
    omega, shear = mach_omega(0.5), -2 * mach_omega(0.5)
    pitchgrid, speedgrid = UniformPitchAngleGrid(17), MaxwellSpeedGrid(4)

    def fluxes(ipccw, btccw, sign):
        field = miller_field(13, ipccw=ipccw, btccw=btccw)
        rotation = Rotation(sign * omega, sign * shear)
        return _fluxes(
            _dense_solution(field, pitchgrid, speedgrid, rotating_species(), rotation)
        )

    ref = fluxes(-1, -1, 1)
    for ipccw, btccw, sign in [(1, -1, 1), (-1, 1, 1), (1, 1, 1), (-1, -1, -1)]:
        G, Q, P = fluxes(ipccw, btccw, sign)
        np.testing.assert_allclose(G, ref[0], rtol=1e-6)
        np.testing.assert_allclose(Q, ref[1], rtol=1e-6)
        np.testing.assert_allclose(P, sign * ref[2], rtol=1e-6)


def test_rotating_outputs():
    """Outputs of the solution with and without rotation."""
    field = miller_field(9)
    pitchgrid, speedgrid = UniformPitchAngleGrid(11), MaxwellSpeedGrid(3)
    species = rotating_species()
    args = (field, pitchgrid, speedgrid, species)
    keys = ["<particle_flux>", "<heat_flux>", "<energy_flux>", "<V||B>", "<J||B>"]
    keys += ["J||", "n", "Phi_1", "n0"]

    # no rotation is a plasma at rest
    sol0 = _dense_solution(*args, None)
    solr = _dense_solution(*args, Rotation(0.0, 0.0))
    for key in keys:
        ref = sol0.get(key)
        np.testing.assert_allclose(
            solr.get(key),
            ref,
            rtol=1e-8,
            atol=1e-8 * float(jnp.abs(ref).max()),
            err_msg=key,
        )
    np.testing.assert_allclose(sol0.get("<energy_flux>"), sol0.get("<heat_flux>"))
    np.testing.assert_allclose(sol0.get("Phi_0"), 0)
    np.testing.assert_allclose(_fluxes(sol0, "nondissipative_"), 0)

    omega = mach_omega(0.8)
    sol = _dense_solution(*args, Rotation(omega, -omega))
    eq = sol.rotating
    assert eq is not None
    n = jnp.array([sp.density for sp in species])[:, None, None]
    np.testing.assert_allclose(sol.get("Phi_0"), eq.Phi)
    np.testing.assert_allclose(sol.get("n0"), n * eq.density)
    # the current is that of the local densities
    q = jnp.array([sp.species.charge for sp in species])[:, None, None]
    np.testing.assert_allclose(
        sol.get("J||"), (q * n * eq.density * sol.get("V||")).sum(axis=0), rtol=1e-12
    )
    # energy flux of the lab frame
    G, Q, P = _fluxes(sol)
    np.testing.assert_allclose(
        sol.get("<energy_flux>", frame="lab") - sol.get("<energy_flux>"),
        omega * P,
        rtol=1e-10,
    )
    assert np.all(np.abs(sol.get("<energy_flux>") - Q) > 1e-3 * np.abs(Q))
    with pytest.raises(ValueError, match="frame"):
        sol.get("<energy_flux>", frame="plasma")
    for key in ["V^theta", "<classical_particle_flux>"]:
        with pytest.raises(NotImplementedError, match="rotation"):
            sol.get(key)


def _tilted_field(ntheta, s_zmag=0.3, btccw=-1):
    """Surface without up-down symmetry: its center rises with the minor radius."""
    return miller_field(ntheta, kappa=1.5, delta=0.2, s_zmag=s_zmag, btccw=btccw)


def test_nondissipative_fluxes():
    """Fluxes that do not depend on the solution of the DKE."""
    omega = mach_omega(0.5)
    species = rotating_species()

    def fluxes(field, rotation, species=species):
        return _fluxes(
            _maxwellian_solution(field, species, rotation), "nondissipative_"
        )

    H = fluxes(_tilted_field(21), Rotation(omega, -2 * omega))
    scale = jnp.abs(H).max(axis=1, keepdims=True)
    assert np.all(scale > 0)
    # they vanish with up-down symmetry, and without rotation
    symmetric = fluxes(miller_field(13), Rotation(omega, -2 * omega))
    np.testing.assert_allclose(symmetric / scale, 0, atol=1e-10)
    assert np.all(fluxes(_tilted_field(13), None) == 0)
    # moments of the Maxwellian: spectral convergence in theta
    coarse = fluxes(_tilted_field(15), Rotation(omega, -2 * omega))
    np.testing.assert_allclose(coarse, H, rtol=1e-4)

    # no rotation shear: no fluxes of particles and energy
    rigid = fluxes(_tilted_field(21), Rotation(omega))
    assert np.all(rigid[:2] == 0)
    assert np.abs(rigid[2]).max() > 0.1 * scale[2]
    # no gradients and no rotation: no flux of angular momentum
    flat = [
        LocalMaxwellian(sp.species, sp.temperature, sp.density, 0.0, 0.0)
        for sp in species
    ]
    still = fluxes(_tilted_field(21), Rotation(0.0, 0.0), flat)
    np.testing.assert_allclose(still / scale, 0, atol=1e-10)

    # The surface turned upside down is the mirror image: the toroidal field
    # reverses, the current and the rotation do not, and no flux changes.
    mirror = fluxes(_tilted_field(21, -0.3, 1), Rotation(omega, -2 * omega))
    np.testing.assert_allclose(mirror, H, rtol=1e-9)


@pytest.mark.parametrize(
    "case",
    [
        "updown_circular_norot",
        "updown_circular_reversed",
        "updown_circular_rigid",
        "updown_circular_shear",
        "updown_circular_shearonly",
        "updown_mastu_r045_btflip",
        "updown_mastu_r045_rot",
    ],
)
def test_nondissipative_fluxes_neo(reference, case):
    """Fluxes that do not depend on the solution of the DKE against NEO.

    These are moments of the Maxwellian on surfaces without up-down symmetry, so no
    resolution is involved. NEO prints five digits. The differences are 2e-4 for the
    circular surfaces and 8e-4 for MAST-U, whose input is not neutral.
    """
    assert case in reference_cases(reference, "nondissipative")
    assert len(reference_cases(reference, "nondissipative")) == 7
    prefix = "nondissipative_" + case + "_"
    field, species, rotation, _ = reference_case(reference, prefix)
    sol = _maxwellian_solution(field, species, rotation)
    G, Q, P = _fluxes(sol, "nondissipative_")
    fluxes = np.array([G, _energy_flux(sol, G, Q, P, "lab"), P]).T
    ref = reference[prefix + "fluxes"]
    assert np.abs(ref[:, 2]).max() > 0
    # NaN: not printed by NEO (fluxes of particles and energy without rotation)
    printed = np.isfinite(ref)
    nonzero = printed & (ref != 0)
    np.testing.assert_allclose(fluxes[nonzero], ref[nonzero], rtol=2e-3)
    # no particle flux without rotation shear
    T = np.array([sp.temperature for sp in species]) * elementary_charge
    zero = printed & (ref == 0)
    if zero.any():
        assert np.all(
            np.abs(fluxes * T[:, None])[zero] < 1e-9 * np.abs(ref[:, 1]).max()
        )


def _solve(field, nalpha, nx, species, rotation, coulomb_log=16.0, rtol=1e-10):
    sol, info = solve_dke(
        field,
        UniformPitchAngleGrid(nalpha),
        MaxwellSpeedGrid(nx),
        species,
        0.0,
        rtol=rtol,
        coulomb_log=coulomb_log,
        preconditioner="tokamak",
        rotation=rotation,
    )
    assert info["success"]
    assert info["nmv"] < 60
    return sol


@pytest.mark.parametrize("mach, shear", [(0.5, -2.0), (1.5, 0.0)])
def test_solve_dke_rotation(mach, shear):
    """Solution of the DKE of a rotating plasma matches a dense solve."""
    if os.environ.get("CI"):
        jax.clear_caches()
    field = miller_field(9)
    species = rotating_species()
    rotation = Rotation(mach_omega(mach), shear * mach_omega(mach))
    sol = _solve(field, 11, 4, species, rotation)
    ref = _dense_solution(
        field, UniformPitchAngleGrid(11), MaxwellSpeedGrid(4), species, rotation
    )
    for key in (*_FLUXES, "<J||B>"):
        np.testing.assert_allclose(sol.get(key), ref.get(key), rtol=1e-6, err_msg=key)


def test_solve_dke_rotation_zero():
    """Without rotation the solution is that of a plasma at rest."""
    field = miller_field(9)
    species = rotating_species()
    sol = _solve(field, 11, 4, species, Rotation(0.0, 0.0))
    sol0, _ = solve_dke(
        field,
        UniformPitchAngleGrid(11),
        MaxwellSpeedGrid(4),
        species,
        0.0,
        p1="4z",
        rtol=1e-10,
        coulomb_log=16.0,
        preconditioner="tokamak",
    )
    for key in ("<particle_flux>", "<heat_flux>", "<J||B>"):
        np.testing.assert_allclose(sol.get(key), sol0.get(key), rtol=1e-7, err_msg=key)


def test_tokamak_preconditioner_rotation():
    """Coarse level of the preconditioner is the Galerkin inverse with rotation."""
    field = miller_field(9)
    pitchgrid, speedgrid = UniformPitchAngleGrid(11), MaxwellSpeedGrid(4)
    species = rotating_species()
    args = (field, pitchgrid, speedgrid, species)
    A = DKE(*args, 0.0, p1="4z", coulomb_log=16.0, rotation=Rotation(mach_omega(1.0)))
    B = DKESources(*args)
    C = DKEConstraint(*args, True)
    op = BorderedOperator(A, B, C)
    ns, moments = len(species), 4
    fs = TokamakPreconditioner(op, moments).coarse.fluid

    # the exchange of energy between the species is removed by a term of low rank,
    # which the coarse solve has to project as well
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    c0 = jax.random.normal(keys[0], (ns, speedgrid.nx * moments, field.ntheta, 1))
    h0 = jax.random.normal(keys[1], (2 * ns,))
    v = op.mv((fs.prolong(c0), h0))
    c1, h1 = fs.solve(fs.restrict(v[0]), v[1])
    v1 = op.mv((fs.prolong(c1), h1))
    r0, r1 = fs.restrict(v[0]), fs.restrict(v1[0])
    np.testing.assert_allclose(r1, r0, rtol=0, atol=1e-10 * float(jnp.max(jnp.abs(r0))))
    np.testing.assert_allclose(
        v1[1], v[1], rtol=0, atol=1e-10 * float(jnp.max(jnp.abs(v[1])))
    )


def test_solve_dke_rotation_errors(pitchgrid, speedgrid):
    """Rotation needs the preconditioner for axisymmetric fields."""
    args = (miller_field(9), pitchgrid, speedgrid, rotating_species())
    with pytest.raises(ValueError, match="rotation requires"):
        solve_dke(*args, 0.0, rotation=Rotation(1e4))
    with pytest.raises(ValueError, match="rotation"):
        solve_dke_ambipolar(*args, 1, rotation=Rotation(1e4))


def test_solve_dke_rotation_derivatives():
    """Derivatives with respect to the rotation, its shear and a temperature."""
    # these are super low res, just to test jax logic, not physical correctness
    if os.environ.get("CI"):
        jax.clear_caches()
    field = miller_field(9)
    species = rotating_species()
    p0 = jnp.array([mach_omega(0.6), -mach_omega(0.9), 2e3])

    def fluxes(p):
        nC = species[1].density
        carbon = LocalMaxwellian(Carbon, p[2], nC, -3 * p[2], -nC)
        sol, _ = solve_dke(
            field,
            UniformPitchAngleGrid(11),
            MaxwellSpeedGrid(3),
            [species[0], carbon, species[2]],
            0.0,
            rtol=1e-12,
            coulomb_log=16.0,
            preconditioner="tokamak",
            rotation=Rotation(p[0], p[1]),
        )
        return jnp.concatenate([_fluxes(sol).flatten(), sol.get("<J||B>")[None]])

    out = jnp.abs(fluxes(p0))
    scale = jnp.append(jnp.repeat(out[:9].reshape(3, 3).max(axis=1), 3), out[9])

    def foo(e):
        # relative changes of the parameters and of the fluxes of each kind
        return fluxes(p0 * (1 + e)) / scale

    e0, h = jnp.zeros(3), 1e-5
    Jfd = jnp.array(
        [(foo(e0.at[i].add(h)) - foo(e0.at[i].add(-h))) / (2 * h) for i in range(3)]
    ).T
    Jf = jax.jacfwd(foo)(e0)
    Jr = jax.jacrev(foo)(e0)
    # every parameter matters
    assert np.abs(Jf).max(axis=0).min() > 1e-3
    np.testing.assert_allclose(Jr, Jf, atol=1e-8)
    np.testing.assert_allclose(Jf, Jfd, atol=1e-5)


def test_nondissipative_fluxes_ambipolar():
    """Particle fluxes on a surface without up-down symmetry are ambipolar only with
    the fluxes that do not depend on the solution (Sugama & Horton 1997, eq 28).
    """
    if os.environ.get("CI"):
        jax.clear_caches()
    omega = mach_omega(0.5)
    species = rotating_species()
    rotation = Rotation(omega, -2 * omega)
    charge = np.array([1, 6, -1])

    def ambipolarity(G):
        return np.abs(np.sum(charge * G)) / np.sum(np.abs(charge * G))

    sol = _solve(_tilted_field(21), 33, 6, species, rotation)
    G = sol.get("<particle_flux>")
    H = sol.get("<nondissipative_particle_flux>")
    # The fluxes of the DKE alone are far from ambipolar here. The rest is the
    # discretization error.
    assert ambipolarity(G) > 0.5
    assert ambipolarity(G + H) < 0.02
    low = _solve(_tilted_field(13), 17, 4, species, rotation)
    assert ambipolarity(
        low.get("<particle_flux>") + low.get("<nondissipative_particle_flux>")
    ) > 3 * ambipolarity(G + H)

    # the surface turned upside down is the mirror image
    mirror = _solve(_tilted_field(21, -0.3, 1), 33, 6, species, rotation)
    for key in ("<particle_flux>", "<angular_momentum_flux>"):
        np.testing.assert_allclose(mirror.get(key), sol.get(key), rtol=1e-6)


_NAN = float("nan")
# Largest relative differences to NEO of the fluxes [species (D, C, e), (particles,
# energy, angular momentum)], about twice the differences found with these grids (6
# speeds, 65 pitch angles and the 41 poloidal nodes of the reference field). NaN is
# not asserted: the electron particle flux, the small remainder of the ion fluxes, and
# fluxes near a zero crossing.
_NEO_CASES = {
    # MAST-U H-mode (carbon Mach number above 1) at mid radius
    "mastu_r045_rot": (
        (6, 65),
        [[0.07, 0.03, 0.03], [0.08, 0.05, 0.06], [_NAN, 0.06, _NAN]],
        [[0.07, 0.03, 0.03], [_NAN, _NAN, _NAN], [_NAN, 0.06, _NAN]],
    ),
    # towards the edge, where the shape matters
    "mastu_r065_rot": (
        (6, 65),
        [[0.02, 0.01, 0.05], [0.01, _NAN, 0.01], [0.015, 0.01, _NAN]],
        None,
    ),
    "mastu_r085_rot": (
        (6, 65),
        [[0.02, 0.01, 0.02], [0.03, 0.02, 0.03], [_NAN, 0.01, _NAN]],
        None,
    ),
    # signs: rotation reversed, field and current reversed, shear without rotation
    "mastu_r045_reversed": (
        (6, 65),
        [[0.08, 0.05, 0.04], [0.09, 0.06, 0.07], [_NAN, _NAN, _NAN]],
        None,
    ),
    "mastu_r045_bothflip": (
        (6, 65),
        [[0.08, 0.05, 0.04], [0.09, 0.06, 0.07], [_NAN, _NAN, _NAN]],
        None,
    ),
    "mastu_r045_shearonly": (
        (6, 65),
        [[_NAN, _NAN, 0.04], [_NAN, _NAN, 0.05], [_NAN, _NAN, _NAN]],
        None,
    ),
    # strong rotation (carbon Mach number 2.4) without and with shear at high
    # collisionality (33 poloidal nodes), where the particle fluxes need 10 speeds
    "collisional_rigid3": (
        (10, 49),
        [[0.03, 0.01, 0.02], [0.05, 0.08, 0.05], [0.03, 0.01, _NAN]],
        None,
    ),
    "collisional_shear": (
        (10, 49),
        [[0.03, 0.01, 0.02], [0.05, 0.08, 0.05], [0.03, 0.01, _NAN]],
        None,
    ),
    # surface without up-down symmetry: the fluxes of the DKE, which are far from
    # ambipolar, and the total
    "updown_mastu_r045_rot": (
        (6, 65),
        [[0.04, 0.03, 0.01], [_NAN, _NAN, _NAN], [_NAN, _NAN, _NAN]],
        [[0.09, 0.04, 0.05], [0.10, 0.05, 0.08], [_NAN, _NAN, _NAN]],
    ),
}


@pytest.mark.parametrize("case", _NEO_CASES.keys())
def test_solve_dke_rotation_neo(reference, case):
    """Fluxes in a rotating plasma against NEO.

    Compared are the fluxes of the DKE with the energy flux of the lab frame, and the
    total with the fluxes that do not depend on the solution and the energy flux of
    the rotating frame.
    """
    if os.environ.get("CI"):
        jax.clear_caches()
    assert sorted(_NEO_CASES) == reference_cases(reference, "fluxes")
    (nx, nalpha), limits, total_limits = _NEO_CASES[case]
    prefix = "fluxes_" + case + "_"
    field, species, rotation, coulomb_log = reference_case(reference, prefix)
    sol = _solve(field, nalpha, nx, species, rotation, coulomb_log, rtol=1e-8)

    def deviation(G, Q, P, frame, key):
        fluxes = np.array([G, _energy_flux(sol, G, Q, P, frame), P]).T
        return fluxes / reference[prefix + key] - 1

    G, Q, P = _fluxes(sol)
    dev = deviation(G, Q, P, "lab", "fluxes")
    limits = np.array(limits)
    check = np.isfinite(limits)
    assert np.all(np.abs(dev)[check] < limits[check]), dev
    if total_limits is not None:
        H = _fluxes(sol, "nondissipative_")
        dev = deviation(G + H[0], Q + H[1], P + H[2], "rotating", "total_fluxes")
        limits = np.array(total_limits)
        check = np.isfinite(limits)
        assert np.all(np.abs(dev)[check] < limits[check]), dev
