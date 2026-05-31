=================
API Documentation
=================

Solving the Drift Kinetic Equation
----------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    yancc.solve.solve_dke     -- Solve the standard drift kinetic equation.
    yancc.solve.solve_mdke    -- Solve the monoenergetic drift kinetic equation.

Fields and Velocity Grids
-------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    yancc.field.Field                              -- Magnetic field on a flux surface.
    yancc.velocity_grids.MaxwellSpeedGrid          -- Collocation grid for speed coordinate based on Maxwell polynomials.
    yancc.velocity_grids.AbstractPitchAngleGrid    -- Base class for pitch angle coordinate grids.
    yancc.velocity_grids.UniformPitchAngleGrid     -- Finite difference grid with uniform spacing for pitch angle coordinate.
    yancc.velocity_grids.NonUniformPitchAngleGrid  -- Finite difference grid with node spacing set by a custom mapping function.
    yancc.velocity_grids.QuadraticPitchAngleGrid   -- Finite difference grid that packs nodes near v|| = 0 to resolve low-collisionality features.

Species
-------

.. autosummary::
    :toctree: _api/
    :recursive:

    yancc.species.Species         -- Atomic species of arbitrary charge and mass.
    yancc.species.LocalMaxwellian -- Local Maxwellian distribution function on a single surface.
    yancc.species.GlobalMaxwellian -- Global Maxwellian distribution function over radius.

Predefined :class:`yancc.species.Species` instances for common isotopes:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Mass (mp)
     - Charge (e)
   * - :data:`yancc.species.Electron`
     - 1/1836.15
     - -1
   * - :data:`yancc.species.Hydrogen`
     - 1
     - 1
   * - :data:`yancc.species.Deuterium`
     - 2
     - 1
   * - :data:`yancc.species.Tritium`
     - 3
     - 1
   * - :data:`yancc.species.Helium4` (alias :data:`yancc.species.Helium`)
     - 4
     - 2
   * - :data:`yancc.species.Lithium6`
     - 6
     - 3
   * - :data:`yancc.species.Lithium7` (alias :data:`yancc.species.Lithium`)
     - 7
     - 3
   * - :data:`yancc.species.Beryllium9` (alias :data:`yancc.species.Beryllium`)
     - 9
     - 4
   * - :data:`yancc.species.Boron10`
     - 10
     - 5
   * - :data:`yancc.species.Boron11` (alias :data:`yancc.species.Boron`)
     - 11
     - 5
   * - :data:`yancc.species.Nitrogen14` (alias :data:`yancc.species.Nitrogen`)
     - 14
     - 7
   * - :data:`yancc.species.Oxygen16` (alias :data:`yancc.species.Oxygen`)
     - 16
     - 8

Solution Objects
----------------

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: solution_class

    yancc.solution.DKESolution    -- Solution returned by ``solve_dke``.
    yancc.solution.MDKESolution   -- Solution returned by ``solve_mdke``.
