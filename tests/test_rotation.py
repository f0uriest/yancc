"""Tests for toroidal rotation of the plasma.

The reference data in ``tests/data/neo_sonic_rotation_reference.npz`` are results of
NEO (E. A. Belli and J. Candy) with its full Fokker-Planck collision operator and
sonic rotation. Each case is stored under a prefix: the field on the nodes of the
surface (with the major radius), the species (D, C, e), the Coulomb logarithm and the
rotation as yancc takes them, and the result of NEO in SI units per unit rho.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import elementary_charge, proton_mass

import yancc
from yancc.rotation import RotatingEquilibrium, Rotation
from yancc.species import Deuterium, Electron, LocalMaxwellian, Species

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
