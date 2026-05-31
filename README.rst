yancc
=====

Yet Another NeoClassical Code.

``yancc`` solves the drift kinetic equation to compute neoclassical flows and transport
fluxes in toroidal geometry (both tokamaks and stellarators).

Installation
------------

``yancc`` is a pure-Python package built on `JAX <https://github.com/google/jax>`_
and requires Python 3.10 or newer.

From PyPI::

    pip install yancc

From source (for development)::

    git clone https://github.com/f0uriest/yancc.git
    cd yancc
    pip install -e ".[dev]"

JAX provides separate wheels for CPU and GPU backends; see the
`JAX install guide <https://docs.jax.dev/en/latest/installation.html>`_ to pick
the appropriate one for your hardware.

Example: Solving the Drift Kinetic Equation
-------------------------------------------

A minimal end-to-end DKE solve for a single hydrogen species:

.. code-block:: python

    from yancc.field import Field
    from yancc.solve import solve_dke
    from yancc.species import Hydrogen, LocalMaxwellian
    from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

    # Field and grids.
    rho = 0.5
    nt, nz, na, nx = 15, 31, 61, 6
    field = Field.from_vmec("wout_NCSX.nc", rho, nt, nz)
    pitchgrid = UniformPitchAngleGrid(na)
    speedgrid = MaxwellSpeedGrid(nx)

    # Single hydrogen species. Density and temperature gradients are with
    # respect to rho = sqrt(normalized toroidal flux), so multiply physical
    # gradients by the minor radius.
    species = [
        LocalMaxwellian(
            Hydrogen,
            temperature=0.8e3,                        # eV
            density=1.5e20,                           # 1/m^3
            dTdrho=-2.0e3 * field.a_minor,
            dndrho=-0.4e20 * field.a_minor,
        )
    ]

    # Radial electric field, in Volts. Erho = -dPhi/drho.
    Er_kV_per_m = 0.5
    Erho = Er_kV_per_m * field.a_minor * 1000

    sol, info = solve_dke(
        field,
        pitchgrid,
        speedgrid,
        species,
        Erho=Erho,
        verbose=2,
        rtol=1e-5,
    )

    print("<Gamma>  =", sol.get("<particle_flux>"))   # particles/(m^2 s)
    print("<Q>      =", sol.get("<heat_flux>"))       # J/(m^2 s)
    print("<V||B>   =", sol.get("<V||B>"))            # T*m/s
    print("<J||B>   =", sol.get("<J||B>"))            # T*A/m^2

See the `documentation <https://yancc.readthedocs.io/>`_ for the monoenergetic
solver, multi-species runs, the full list of output variables, and the API
reference.
