=================
API Documentation
=================

Solving the Drift Kinetic Equation
----------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    yancc.solve_dke           -- Solve the standard drift kinetic equation.
    yancc.solve_mdke          -- Solve the monoenergetic drift kinetic equation.
    yancc.solve_dke_ambipolar -- Find ambipolar radial electric fields and the corresponding DKE solutions.

Fields and Velocity Grids
-------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    yancc.Field                 -- Magnetic field on a flux surface.
    yancc.MaxwellSpeedGrid      -- Collocation grid for speed coordinate based on Maxwell polynomials.
    yancc.UniformPitchAngleGrid -- Finite difference grid with uniform spacing for pitch angle coordinate.

Species
-------

.. autosummary::
    :toctree: _api/
    :recursive:

    yancc.Species          -- Atomic species of arbitrary charge and mass.
    yancc.LocalMaxwellian  -- Local Maxwellian distribution function on a single surface.
    yancc.GlobalMaxwellian -- Global Maxwellian distribution function over radius.

Predefined :class:`yancc.Species` instances for common isotopes:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Mass (mp)
     - Charge (e)
   * - :data:`yancc.Electron`
     - 1/1836.15
     - -1
   * - :data:`yancc.Hydrogen`
     - 1
     - 1
   * - :data:`yancc.Deuterium`
     - 2
     - 1
   * - :data:`yancc.Tritium`
     - 3
     - 1
   * - :data:`yancc.Helium4` (alias :data:`yancc.Helium`)
     - 4
     - 2
   * - :data:`yancc.Lithium6`
     - 6
     - 3
   * - :data:`yancc.Lithium7` (alias :data:`yancc.Lithium`)
     - 7
     - 3
   * - :data:`yancc.Beryllium9` (alias :data:`yancc.Beryllium`)
     - 9
     - 4
   * - :data:`yancc.Boron10`
     - 10
     - 5
   * - :data:`yancc.Boron11` (alias :data:`yancc.Boron`)
     - 11
     - 5
   * - :data:`yancc.Nitrogen14` (alias :data:`yancc.Nitrogen`)
     - 14
     - 7
   * - :data:`yancc.Oxygen16` (alias :data:`yancc.Oxygen`)
     - 16
     - 8

Solution Objects
----------------

.. autosummary::
    :toctree: _api/
    :recursive:
    :template: solution_class

    yancc.DKESolution  -- Solution returned by ``solve_dke``.
    yancc.MDKESolution -- Solution returned by ``solve_mdke``.
