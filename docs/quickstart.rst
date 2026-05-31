==========
Quickstart
==========

This page walks through two complete examples: solving the full drift kinetic
equation (DKE) for one or more species, and solving the simplified
monoenergetic drift kinetic equation (MDKE) to get a 3×3 transport matrix.

.. _radial-coordinate:

Radial coordinate convention
----------------------------

yancc uses a single radial coordinate everywhere:

.. math::

    \rho = \sqrt{s} = \sqrt{\psi_t / \psi_{t,\mathrm{LCFS}}},

i.e. the square root of the normalized toroidal flux. This is the only radial
variable that appears in the public API.

This differs from several other codes:

  * Some codes (e.g. VMEC, BOOZ_XFORM) parameterize surfaces by
    :math:`s = \rho^2`, the normalized toroidal flux itself.
  * Others use the dimensional minor radius :math:`r = \rho\, a`, where
    :math:`a` is the minor radius of the LCFS.
  * Many codes mix conventions, e.g. labelling surfaces by :math:`s` but
    expressing gradients in :math:`r`.

In yancc, **every** radial input and output is in :math:`\rho`. In particular:

  * :class:`~yancc.field.Field` constructors (``from_desc``, ``from_vmec``,
    ``from_booz_xform``, ``from_ipp_bc``, ``from_boozer``) all take ``rho``,
    not ``s`` or ``r``. If your input is :math:`s`, pass ``rho = sqrt(s)``.
  * The radial electric field passed to :func:`~yancc.solve.solve_dke` is
    :math:`E_\rho = -\partial \Phi / \partial \rho`, in Volts. If you have
    :math:`E_r = -\partial \Phi / \partial r` (in V/m), multiply by
    ``field.a_minor`` to convert.
  * The density and temperature gradients on
    :class:`~yancc.species.LocalMaxwellian` are
    :math:`\partial n / \partial \rho` and :math:`\partial T / \partial \rho`.
    If you have :math:`dn/dr` and :math:`dT/dr`, multiply by ``field.a_minor``.
  * The monoenergetic drive ``erhohat`` passed to
    :func:`~yancc.solve.solve_mdke` is :math:`E_\rho / v` in V·s/m, again
    using :math:`\rho` (not :math:`r` or :math:`s`).

Make sure all inputs are converted to the :math:`\rho` convention before
calling into yancc; otherwise gradients will silently disagree with what the
solver expects, by factors of :math:`a` or :math:`2\rho`.

Solving the Full DKE
--------------------

The full DKE requires a magnetic field, pitch-angle and speed grids, one or
more species, and a radial electric field. Each species is a
:class:`~yancc.species.LocalMaxwellian` built from a
:class:`~yancc.species.Species` (for example one of the predefined isotope
constants like :data:`~yancc.species.Hydrogen`).

.. code-block:: python

    import yancc
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

    # Common moments. See variables.rst for the full list.
    print("<Gamma>  =", sol.get("<particle_flux>"))   # particles/(m^2 s)
    print("<Q>      =", sol.get("<heat_flux>"))       # J/(m^2 s)
    print("<V||B>   =", sol.get("<V||B>"))            # T*m/s
    print("<J||B>   =", sol.get("<J||B>"))            # T*A/m^2

The ``sol`` object is a :class:`~yancc.solution.DKESolution`; pass any name
listed in :doc:`variables` to ``sol.get(...)``.

Multiple species and background species
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``species`` can contain several entries to solve coupled species
simultaneously. The optional ``background`` argument adds species to the
collision operator without solving for their perturbed distribution.

Rather than building each :class:`~yancc.species.LocalMaxwellian` by hand,
it is often more convenient to define radial profiles once as a
:class:`~yancc.species.GlobalMaxwellian` and let
:meth:`~yancc.species.GlobalMaxwellian.localize` evaluate the temperature,
density, and their :math:`\rho`-gradients at the surface of interest:

.. code-block:: python

    import jax.numpy as jnp
    from yancc.species import Electron, Hydrogen, GlobalMaxwellian

    # Profiles in rho = sqrt(normalized toroidal flux).
    def T_profile(rho):
        return 0.8e3 * (1.0 - rho**2)        # eV

    def n_profile(rho):
        return 1.5e20 * (1.0 - rho**2)       # 1/m^3

    globals_ = [
        GlobalMaxwellian(Hydrogen, T_profile, n_profile),
        GlobalMaxwellian(Electron, T_profile, n_profile),
    ]

    # localize(rho) returns a LocalMaxwellian with dTdrho, dndrho filled in
    # automatically via JAX autodiff of the profile callables.
    species = [g.localize(rho) for g in globals_]

    sol, info = solve_dke(
        field, pitchgrid, speedgrid, species,
        Erho=Erho,
        EparB=0.0,
    )

Loading a field from other equilibria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~yancc.field.Field` can be constructed from several equilibrium
formats. The same physical surface can be obtained from any of them:

.. code-block:: python

    import desc

    rho = 0.5
    nt, nz = 17, 37

    eq = desc.io.load("NCSX_output.h5")[-1]
    field_desc = Field.from_desc(eq, rho, nt, nz)
    field_vmec = Field.from_vmec("wout_NCSX.nc", rho, nt, nz)
    field_booz = Field.from_booz_xform("boozmn_wout_NCSX.nc", rho, nt, nz)
    field_bc   = Field.from_ipp_bc("NCSX.bc", rho, nt, nz)


See :doc:`loading_fields` or :class:`yancc.field.Field` for more information.

JAX transformations
~~~~~~~~~~~~~~~~~~~

:func:`~yancc.solve.solve_dke` and :func:`~yancc.solve.solve_mdke` are
implemented in JAX, so the standard transformations (``jax.jit``,
``jax.vmap``, ``jax.jacfwd``, ``jax.jacrev``, ...) can be applied to them in
the usual way.

Solving the Monoenergetic DKE
-----------------------------

The MDKE depends only on a magnetic field, a pitch-angle grid, and two
normalized scalars: a monoenergetic electric field ``erhohat`` and a
monoenergetic collisionality ``nuhat``. The solution object exposes the monoenergetic
transport matrix via ``sol.get("Dij")``.

.. code-block:: python

    import yancc
    from yancc.field import Field
    from yancc.solve import solve_mdke
    from yancc.velocity_grids import UniformPitchAngleGrid

    # Magnetic field on a flux surface (here from a BOOZ_XFORM file).
    nt, nz = 17, 33
    rho = 0.200 ** 0.5      # rho = sqrt(s); pick the surface to load
    field = Field.from_booz_xform(
        "boozmn_wout_w7x_eim.nc",
        rho,
        nt,
        nz,
        cutoff=1e-5,
    )

    # Uniform finite-difference grid in pitch angle.
    pitchgrid = UniformPitchAngleGrid(65)

    # Monoenergetic drives.
    erhohat = 1.0e-4      # E_rho / v   [V*s/m]
    nuhat = 1.0e-2        # nu / v      [1/m]

    sol, info = solve_mdke(
        field,
        pitchgrid,
        erhohat,
        nuhat,
        verbose=2,
    )

    # 3x3 transport matrix.
    Dij = sol.get("Dij")
    print("D11 =", Dij[0, 0])
    print("D31 =", Dij[2, 0])
    print("D33 =", Dij[2, 2])

The ``sol`` object is a :class:`~yancc.solution.MDKESolution`; see
:doc:`variables` for the full list of names accepted by ``sol.get(...)``.
