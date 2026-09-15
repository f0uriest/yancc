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


v0.0.1
------
- Initial release
