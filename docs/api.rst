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

    field.Field     -- Class representing magnetic field on a flux surface.
    velocity_grids.MaxwellSpeedGrid -- Collocation grid for speed coordinate based on Maxwell polynomials
    velocity_grids.UniformPitchAngleGrid -- Finite difference grid with uniform spacing for pitch angle coordinate.
