==========================
Performance and Resolution
==========================

This page is a practical guide to picking grid sizes, tuning the solver, and
making batched/JAX-transformed runs efficient. It is not a benchmark report —
the right resolutions depend on your device and the quantity you care about,
so the recommendations below are starting points, not final answers.

Picking grid resolutions
========================

A yancc solve has four resolution parameters:

- ``nx`` — collocation nodes in the speed coordinate :math:`x = v / v_{th}`.
  Set on :class:`~yancc.velocity_grids.MaxwellSpeedGrid`. Used by the full
  DKE only; the MDKE is monoenergetic and has no speed grid.
- ``nalpha`` — points along the pitch-angle :math:`\alpha \in [0, \pi]`. Set
  on :class:`~yancc.velocity_grids.UniformPitchAngleGrid`.
- ``ntheta`` — points along the poloidal angle :math:`\theta \in [0, 2\pi]`
  on the flux surface. Set when constructing :class:`~yancc.field.Field`.
- ``nzeta`` — points along the toroidal angle :math:`\zeta \in [0, 2\pi/NFP]`
  on the flux surface. Also set on :class:`~yancc.field.Field`.

What each grid actually resolves
--------------------------------

``ntheta, nzeta``
  The variation of :math:`B(\theta, \zeta)` and of the perturbed distribution
  on the flux surface. For ``ntheta = 15-45`` is the usual range, for ``nzeta = 33-129``,
  with larger values needed for low collisionality.


``nalpha``
  Pitch-angle resolution. This is almost always the most demanding axis at
  low collisionality, because trapped/passing boundary layers narrow as
  :math:`\nu^* \to 0`. The number of points usually scales like :math:`1/\sqrt{\nu^*}`,
  with the usual range of ``nalpha = 65-257``. At high collisionality this can be usually
  be reduced slightly.


``nx``
  Speed resolution. Because the speed grid is collocated on Maxwell-weighted
  polynomials (not finite differences), thermal moments converge with very
  few points: ``nx = 5–8`` is the usual range and ``nx = 6`` is a sensible
  starting point. Larger values may be needed for multiple species at high
  collisionality.

Doing a convergence study
-------------------------

The right way to pick resolutions is to fix the inputs (field, species,
:math:`E_\rho`) and run a sweep, ideally on the moment you actually care
about (e.g. ``<particle_flux>`` or ``Dij[1,1]``):

.. code-block:: python

    import jax
    from yancc.solve import solve_mdke

    def D11(na):
        pitchgrid = UniformPitchAngleGrid(na)
        sol, _ = solve_mdke(field, pitchgrid, erhohat, nuhat)
        return sol.get("Dij")[0, 0]

    for na in [33, 49, 65, 97, 129]:
        print(na, D11(na))

Double the resolution until the moment of interest changes by less than
your acceptable tolerance, then back off one step. Sweep one axis at a time:
``nalpha`` first (usually the binding one), then ``ntheta``/``nzeta``, then
``nx``.

How cost scales
===============

The DKE is solved on a grid of total size

.. math::

    N \;=\; n_s \cdot n_x \cdot n_\alpha \cdot n_\theta \cdot n_\zeta,

(where :math:`n_s` is the number of kinetic species). Memory and per-iteration
cost are roughly linear in :math:`N`. The MDKE has the same structure with
:math:`n_x = n_s = 1`, so it is much cheaper.

Several less obvious scalings are worth knowing:

- **Species count.** :math:`n_s` enters both as a grid axis and through the
  field-particle collision operator, which couples species pairwise. Going
  from 1 species to 2 typically increases solve time by more than 2×;
  going from 2 to 3 by less.
- **Background species** (the ``background`` argument to
  :func:`~yancc.solve.solve_dke`) only appear in the collision operator and
  are much cheaper than promoting them to kinetic species. Use ``background``
  for impurities you don't need flux information about.
- **Low collisionality.** Iteration counts grow as :math:`\nu^* \to 0`. If
  you are scanning down in collisionality, expect the lowest collisionality in the
  scan to dominate the total time — and to be the run most sensitive to
  ``nalpha``.
- **Tolerance.** The default ``rtol=1e-5`` is conservative for transport
  moments. Loosening to ``1e-4`` or ``1e-3`` often reduces the iteration count without
  visibly affecting fluxes; tighten only if you need it.

JAX, JIT, and vmap
==================

:func:`~yancc.solve.solve_dke` and :func:`~yancc.solve.solve_mdke` are JAX
functions, which has a few practical consequences.

Compilation cost
----------------

The first call with a given set of *static* shapes (grid sizes, number of
species) triggers a JIT compile that can take tens of seconds for the full
DKE. Subsequent calls with the same shapes reuse the compiled code and are
fast. To amortize the compile cost, run multiple cases at the same
resolution in the same Python process, or consider enabling
`JAX's persistent compile cache <https://docs.jax.dev/en/latest/persistent_compilation_cache.html>`_

Changing any grid size, the species list length, or the ``background`` list
length triggers a recompile.

Batching with vmap
------------------

For scans over inputs (eg ``Erho``, ``EparB``, ``erhohat``, ``nuhat``),
``jax.vmap`` is dramatically more efficient than a Python loop because it
fuses the solves into a single batched linear solve:

.. code-block:: python

    import jax, jax.numpy as jnp
    from yancc.solve import solve_mdke

    def one(erhohat, nuhat):
        sol, _ = solve_mdke(field, pitchgrid, erhohat, nuhat)
        return sol.get("Dij")

    erhohats = jnp.linspace(-1e-3, 1e-3, 21)
    nuhats   = jnp.full_like(erhohats, 1e-2)
    Dij_scan = jax.vmap(one)(erhohats, nuhats)   # shape (21, 3, 3)

This requires that the grid sizes and species list be the same across the
batch, since they are static.

Differentiation
---------------

The solvers are differentiable; ``jax.jacfwd`` / ``jax.jacrev`` work for
gradients of fluxes with respect to input parameters (e.g. field strength, temperature,
density, electric field). Differentiating with respect to a
:class:`~yancc.species.LocalMaxwellian` or :class:`~yancc.field.Field` also works, but
will return a pytree of gradients, so it may be helpful to define a helper function to
differentiate first:

.. code-block:: python

    import jax
    from yancc.solve import solve_mdke

    def solve(B):
        field = yancc.Field.from_boozer(B, **other_field_inputs)
        sol, _ = solve_mdke(field, pitchgrid, erhohat, nuhat)
        return sol.get("Dij")

    dDij_dB = jax.jacrev(solve)(B)   # shape (3, 3, nt, nz)


CPU vs GPU
==========

yancc inherits its hardware support from JAX. In broad terms:

- **CPU** is fine for the MDKE and for small-to-medium full-DKE runs (single
  species, modest grids).
- **GPU** wins for the full DKE at production resolution, large species
  counts, or large vmapped scans. Memory is usually the binding constraint:
  the dense distribution function alone is
  :math:`8 n_s n_x n_\alpha n_\theta n_\zeta` bytes, and the multigrid
  preconditioner adds several copies.

If you hit out-of-memory errors on GPU, the cheapest fix is usually to
reduce the size of the krylov space ``m`` and increase the iteration count ``maxiter``.

Other tips
==========

- **Verbose mode.** ``verbose=2`` prints Krylov residuals at every outer
  iteration; this is the first thing to look at when a solve is slow or not
  converging. ``verbose=3`` adds multigrid-level residuals if you want to
  debug preconditioner behavior.
- **Reuse the field.** :class:`~yancc.field.Field` construction (especially
  via ``from_vmec`` / ``from_booz_xform``) reads the file from disk and does
  Fourier work; build it once per surface and pass it to as many solves as
  you need.
- **Check that your gradients are in** :math:`\rho`. A wrong factor of
  ``a_minor`` in ``dndrho`` / ``dTdrho`` will not produce an error — the
  solve will succeed and the fluxes will silently be wrong by the same
  factor. See :ref:`radial-coordinate` in the quickstart.
