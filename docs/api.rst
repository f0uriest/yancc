=================
API Documentation
=================

.. currentmodule:: yancc

Solving the Drift Kinetic Equation
----------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    solve.solve_dke     -- Solve the standard drift kinetic equation.
    solve.solve_mdke    -- Solve the monoenergetic drift kinetic equation.

Fields and Velocity Grids
-------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    field.Field                             -- Magnetic field on a flux surface.
    velocity_grids.MaxwellSpeedGrid         -- Collocation grid for speed coordinate based on Maxwell polynomials.
    velocity_grids.UniformPitchAngleGrid    -- Finite difference grid with uniform spacing for pitch angle coordinate.

Species
-------

.. autosummary::
    :toctree: _api/
    :recursive:

    species.Species         -- Atomic species of arbitrary charge and mass.
    species.LocalMaxwellian -- Local Maxwellian distribution function on a single surface.
    species.GlobalMaxwellian -- Global Maxwellian distribution function over radius.

Predefined :class:`~yancc.species.Species` instances for common isotopes:

.. autodata:: yancc.species.Electron
   :no-value:
.. autodata:: yancc.species.Hydrogen
   :no-value:
.. autodata:: yancc.species.Deuterium
   :no-value:
.. autodata:: yancc.species.Tritium
   :no-value:
.. autodata:: yancc.species.Helium4
   :no-value:
.. autodata:: yancc.species.Helium
   :no-value:
.. autodata:: yancc.species.Lithium6
   :no-value:
.. autodata:: yancc.species.Lithium7
   :no-value:
.. autodata:: yancc.species.Lithium
   :no-value:
.. autodata:: yancc.species.Beryllium9
   :no-value:
.. autodata:: yancc.species.Beryllium
   :no-value:
.. autodata:: yancc.species.Boron10
   :no-value:
.. autodata:: yancc.species.Boron11
   :no-value:
.. autodata:: yancc.species.Boron
   :no-value:
.. autodata:: yancc.species.Nitrogen14
   :no-value:
.. autodata:: yancc.species.Nitrogen
   :no-value:
.. autodata:: yancc.species.Oxygen16
   :no-value:
.. autodata:: yancc.species.Oxygen
   :no-value:

Solution Objects
----------------

.. autosummary::
    :toctree: _api/
    :recursive:

    solution.DKESolution    -- Solution returned by ``solve_dke``.
    solution.MDKESolution   -- Solution returned by ``solve_mdke``.
