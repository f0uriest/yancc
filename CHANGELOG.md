Changelog
=========


- Refactoring to reduce compile time, should be ~50% less
- Fix small indexing bug with monoenergetic distribution function. Moments such as
  ``D_ij`` are unaffected, but directly indexing the distribution function was
  mis-ordered.
- ``vmap`` over ``solve_dke`` now no longer prints on every iteration with
  ``verbose=True``. Now the krylov solvers only print when requested, and when
  ``vmap``ed they print the largest residual in the batch.
- ``RosenbluthPotentials`` with ``quad=False`` now use a fixed Gauss-Legendre quadrature
  rather than using incomplete gamma functions. This is both faster and more accurate
  over the ranges commonly in use (``nx <= 20``, ``nL<=16``). ``quad=True`` still uses
  adaptive quadrature which is the most robust but slower.
- Fix ordering of particle and heat sources in ``DKESolution`` for multispecies cases.
  Sources are now correctly paired per species; previously the particle and heat
  sources could be assigned to the wrong species when ``ns > 1``.
- ``solve_dke`` now measures the residual of multispecies problems in an entropy-weighted
  norm, so each species' residual is measured relative to its own Maxwellian rather than
  being dominated by the species with the largest distribution function. This changes
  the meaning of ``rtol`` and the reported residual in ``info["res"]``.
- The energy scattering collision operator now uses a Galerkin discretization in speed,
  which guarantees the discrete operator is definite. The previous collocation
  discretization could introduce spurious unstable modes that caused slow or failed
  convergence in some multispecies cases.
- Fix multigrid coarsening when one or more axes are clipped at their minimum
  resolution. The remaining axes now absorb the extra coarsening, so each coarse level
  is close to its target size rather than being under-coarsened.
- ``Field`` creation methods can now be ``vmap``ed over radial position ``rho``.

v0.0.1
------
- Initial release
