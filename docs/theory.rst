======
Theory
======

This page describes what yancc actually computes: the equations being solved,
the coordinates they are written in, and how the inputs of the public API map
onto terms in those equations. It is meant as a reference, not a derivation -
for the latter, see e.g. Helander & Sigmar [1]_ or Beidler et al. [2]_.

yancc solves two related problems on a single flux surface of a given
magnetic equilibrium:

- The **drift kinetic equation** (DKE), linearized about a local Maxwellian,
  for one or more species coupled through their collision operators. This
  gives the perturbed distribution function :math:`f_{1,s}` from which radial
  fluxes, parallel flows, and the bootstrap current follow. The formulation
  is the same as in SFINCS [3]_.
- The **monoenergetic drift kinetic equation** (MDKE), a single-species
  reduction at fixed particle speed, which yields the 3×3 monoenergetic
  transport matrix :math:`D_{ij}`. This is the same problem solved by DKES [4]_ and,
  more recently, MONKES [5]_;

Both problems are linear and time-independent: each call to
:func:`~yancc.solve_dke` or :func:`~yancc.solve_mdke` is a single
sparse linear solve.

Coordinates
===========

Configuration space
-------------------

A flux surface is parameterized by arbitrary poloidal and toroidal angles
:math:`(\theta, \zeta)` on the surface labeled by the radial coordinate

.. math::

    \rho = \sqrt{\psi_t / \psi_{t,\mathrm{LCFS}}},

i.e. the square root of normalized toroidal flux. **Every** radial input and
output in yancc is in :math:`\rho` (see :ref:`/quickstart.ipynb#radial-coordinate-convention` in the
quickstart for conversions from :math:`s` or :math:`r`).

The on-surface grid sizes are ``ntheta`` and ``nzeta``, set when the
:class:`~yancc.Field` is constructed.

Velocity space
--------------

Velocity space is parameterized by the speed coordinate

.. math::

    x_s = v / v_{th,s}, \qquad v_{th,s} = \sqrt{2 T_s / m_s},

and the pitch-angle :math:`\alpha`:

.. math::

    \alpha = -\arccos{v_\parallel / v}

(Other codes use the cosine of the pitch angle :math:`\xi = v_\parallel / v`, we choose
the angle itself because it simplifies the boundary conditions.)

With these conventions the local Maxwellian for species :math:`s` is

.. math::

    F_{M,s}(\rho, x_s) = n_s \left( \frac{m_s}{2 \pi T_s} \right)^{3/2}
                        e^{-x_s^2}.

The pitch-angle grid (:class:`~yancc.UniformPitchAngleGrid`)
has ``nalpha`` uniformly spaced points on :math:`\alpha \in [0, \pi]`. The speed
grid (:class:`~yancc.MaxwellSpeedGrid`) has ``nx`` collocation
nodes chosen as the roots of polynomials orthogonal under the Maxwellian
weight :math:`e^{-x^2}` on :math:`[0, \infty)`; this is what makes a small
``nx`` (typically 5–8) sufficient to resolve thermal moments.

The full DKE
============

Following SFINCS [3]_, yancc writes the perturbed distribution as

.. math::

    f_{1,s} = f_s - F_{M,s},

and solves the linearized drift kinetic equation for :math:`f_{1,s}` on the
flux surface:

.. math::
   :label: dke

    \big( v_\parallel \mathbf{b} + \mathbf{v}_E \big) \cdot \nabla f_{1,s}
    \;-\; C_s[f_{1,s}]
    \;=\; S_s,

where :math:`\mathbf{b} = \mathbf{B}/B`, :math:`\mathbf{v}_E` is the
flux-surface :math:`\mathbf{E} \times \mathbf{B}` drift driven by
:math:`E_\rho`, and :math:`C_s` is the linearized Fokker–Planck collision
operator including both test-particle and field-particle pieces, summed over
all kinetic species and any additional :class:`~yancc.LocalMaxwellian`
``background`` species.

In terms of our chosen coordinates and field components this takes the form

.. math::
    :label: dke_full

    \dot{x}_s \frac{\partial f_{1,s}}{\partial x_s}
    + \dot{\alpha} \frac{\partial f_{1,s}}{\partial \alpha}
    + \dot{\theta} \frac{\partial f_{1,s}}{\partial \theta}
    + \dot{\zeta} \frac{\partial f_{1,s}}{\partial \zeta}
    + C_L[f_{1,s}, f_{1,s'}] + C_E[f_{1,s}, f_{1,s'}] + C_F[f_{1,s}, f_{1,s'}]
    = -\mathbf{v}_\mathrm{drift} \cdot \nabla \rho \frac{\partial F_{M,s}}{\partial \rho}

Where the particle trajectories are given by:

.. math::
    :label: sfincs_trajectories

    \begin{aligned}
    \dot{\theta} &= -\frac{v_{th,s} x_s \cos{(\alpha)} B^\theta}{B} + \frac{B_\zeta}{B^2\sqrt{g}}E_\rho \\
    \dot{\zeta} &= -\frac{v_{th,s} x_s \cos{(\alpha)} B^\zeta}{B} - \frac{B_\theta}{B^2\sqrt{g}}E_\rho \\
    \dot{\alpha} &= - \frac{\sin{(\alpha)}}{2 B^2} v_{th,s} x_s \Bigg( B^\theta \frac{\partial B}{\partial \theta}
    + B^\zeta \frac{\partial B}{\partial \zeta} \Bigg)
    + \cos{(\alpha)} \sin{(\alpha)} \frac{1}{2B^3 \sqrt{g}} E_\rho \Bigg( B_\zeta \frac{\partial B}{\partial \theta}
    - B_\theta \frac{\partial B}{\partial \zeta} \Bigg) \\
    \dot{x}_s &= (1 + \cos^2{\alpha}) \frac{x_s}{2B^3 \sqrt{g}} E_\rho \Bigg( B_\zeta \frac{\partial B}{\partial \theta} - B_\theta \frac{\partial B}{\partial \zeta} \Bigg)
    \end{aligned}

And the collision operator terms are:

.. math::
    :label: fp_collision_operator

    \begin{aligned}
    C_{L,ss'} &= \frac{\nu_{D,ss'}}{2 \sin \alpha} \frac{\partial}{\partial \alpha} \Bigg[\sin{\alpha} \frac{\partial f_{1,s}}{\partial \alpha}\Bigg] \\
    C_{E,ss'} &= \nu_{||,ss'}\Big[ \frac{v^2}{2} \frac{\partial^2 f_{1,s}}{\partial v^2} - \frac{v^2}{v_{th,s'}^2} \Big(1 - \frac{m_s}{m_{s'}} \Big) v \frac{\partial f_{1,s}}{\partial v}\Big] + \nu_{D,ss'} v \frac{\partial f_{1,s}}{\partial v} + 4\pi \Gamma_{ss'}\frac{m_s}{m_s'} F_{M,s'} f_{1,s} \\
    C_{F,ss'} &= \Gamma_{ss'} F_{M,s} \Big[\frac{2v^2}{v_{th,s}^4} \frac{\partial^2 G_{s'}}{\partial v^2} - \frac{2v}{v_{th,s}^2} \Big(1 - \frac{m_s}{m_{s'}} \Big) \frac{\partial H_{s'}}{\partial v} - \frac{2}{v_{th,s}^2} H_{s'} + 4\pi \frac{m_s}{m_{s'}} f_{1,s'} \Big] \\
    \nabla^2_v H_{s'} &= -4 \pi f_{s'} \\
    \nabla^2_v G_{s'} &= 2 H_{s'} \\
    \end{aligned}

The collision frequencies are given by:

.. math::
    :label: collisionalities

    \begin{aligned}
    \nu_{D,ss'} &= \frac{\Gamma_{ss'} n_{s'}}{v^3} [\mathrm{erf}(v/v_{th,s'}) - \Psi(v/v_{th,s'})] \\
    \nu_{||,ss'} &= 2 \frac{\Gamma_{ss'} n_{s'}}{v^3} \Psi(v/v_{th,s'}) \\
    \Gamma_{ss'} &= \frac{4\pi q_s^2 q_{s'}^2 \ln \Lambda_{ss'}}{(4\pi \epsilon_0)^2 m_s^2} \\
    \Psi(x) &= \frac{1}{2x^2}\Big[\mathrm{erf}(x) - \frac{2x}{\sqrt{\pi}} \exp(-x^2) \Big]
    \end{aligned}

where :math:`\ln \Lambda_{ss'}` is the Coulomb logarithm.

Drive term
----------

The right-hand side :math:`S_s` is a sum of three contributions weighted by
the species **thermodynamic forces**:

.. math::
   :label: forces

    \begin{aligned}
    A_{1,s} &= \frac{1}{n_s}\frac{\partial n_s}{\partial \rho}
              - \frac{q_s E_\rho}{T_s}
              - \frac{3}{2} \frac{1}{T_s} \frac{\partial T_s}{\partial \rho}, \\
    A_{2,s} &= \frac{1}{T_s}\frac{\partial T_s}{\partial \rho}, \\
    A_{3,s} &= \frac{q_s}{T_s} \frac{\langle E_\parallel B \rangle}{\langle B^2 \rangle}.
    \end{aligned}

Each is multiplied by a known geometric/Maxwellian factor:

.. math::

    S_s = \big( A_{1,s} + x_s^2 A_{2,s} \big)
          \big( -\mathbf{v}_{m,s} \cdot \nabla \rho \big) F_{M,s}
        \;+\; A_{3,s}\, B v_\parallel F_{M,s},

where :math:`\mathbf{v}_{m,s}` is the magnetic (curvature + grad-B) drift.
The :math:`A_1` term encodes the density gradient and the radial electric
field, :math:`A_2` the temperature gradient, and :math:`A_3` the inductive
parallel electric field (usually zero in stellarators).

Mapping onto the API
--------------------

The public inputs to :func:`~yancc.solve_dke` correspond to
equation :eq:`dke` and :eq:`forces` as follows:

================================================ ==============================================================
API input                                        Term in the equations
================================================ ==============================================================
``species[i].temperature``, ``.density``         :math:`T_s,\, n_s` in :math:`F_{M,s}` and the forces
``species[i].dTdrho``, ``.dndrho``               :math:`\partial T_s / \partial \rho,\,
                                                 \partial n_s / \partial \rho` in :math:`A_{1,s}, A_{2,s}`
``Erho``                                         :math:`E_\rho = -\partial \Phi / \partial \rho` in :math:`A_{1,s}`
``EparB``                                        :math:`\langle E_\parallel B \rangle` in :math:`A_{3,s}`
``background``                                   Extra Maxwellian species in :math:`C_s` only
``field``                                        :math:`B,\, \mathbf{b},\, \mathbf{v}_{m,s},\, \mathbf{v}_E`
``rotation``                                     :math:`\omega,\, \partial \omega / \partial \rho` of a toroidally
                                                 rotating plasma (axisymmetric fields)
================================================ ==============================================================

The returned :class:`~yancc.DKESolution` stores
:math:`f_{1,s}` on the full ``(ns, nx, na, nt, nz)`` grid; flux-surface
moments are computed on demand via ``sol.get(name)``.

Toroidal rotation
-----------------

In an axisymmetric field the plasma may rotate toroidally with an angular velocity
:math:`\omega(\rho)` that is common to all species and of the order of the ion thermal
speed over the major radius (sonic rotation) [6]_ [7]_. With

.. code-block:: python

    sol, info = solve_dke(
        field, pitchgrid, speedgrid, species, Erho=Erho,
        preconditioner="tokamak",
        rotation=yancc.Rotation(omega, domega_drho),
    )

the DKE is solved in the frame rotating with the plasma: :math:`v` and
:math:`v_\parallel` are the velocities relative to the rotation, :math:`\omega` is in
rad/s along the toroidal angle :math:`\zeta` of the field, and the field needs the
major radius ``R`` of its nodes and its radial derivative ``dRdrho`` (see
:doc:`loading_fields`).

**Lowest order.** Each species is a Maxwellian in the rotating frame whose density
varies on the surface with the potential energy in the centrifugal force and in the
electrostatic potential :math:`\Phi_0(\theta)` [7]_ [8]_:

.. math::
   :label: rotating_density

    n_s(\theta) = n_s(0)\, e^{-\lambda_s(\theta)}, \qquad
    \lambda_s = \frac{1}{T_s} \Big[ q_s \Phi_0(\theta)
    - \frac{m_s \omega^2}{2} \big( R^2(\theta) - R^2(0) \big) \Big].

:math:`\Phi_0`, which is zero at :math:`\theta = 0`, is such that the charge density of
all species, including the ``background`` species, is the same at every point of the
surface as at :math:`\theta = 0`. The densities and density gradients of the species
are those at :math:`\theta = 0` [9]_, and :math:`F_{M,s}` below is the Maxwellian of
the local density :math:`n_s(\theta)`.

**Trajectories.** The energy :math:`x_s^2 + \lambda_s` and the magnetic moment are
constant along the collisionless trajectories, so the parallel force
:math:`m_s \dot{v}_\parallel = -T_s \mathbf{b} \cdot \nabla \lambda_s` adds to
:eq:`sfincs_trajectories`

.. math::

    \begin{aligned}
    \dot{x}_s &\to \dot{x}_s - \frac{v_\parallel}{2 x_s} \frac{B^\theta}{B} \frac{\partial \lambda_s}{\partial \theta}, \\
    \dot{\alpha} &\to \dot{\alpha} - \frac{v_{th,s} \sin{(\alpha)}}{2 x_s} \frac{B^\theta}{B} \frac{\partial \lambda_s}{\partial \theta}.
    \end{aligned}

**Drifts.** The radial drift of the guiding centers is the magnetic drift, the drift
in the gradient of the potential energy (:math:`\mathbf{E} \times \mathbf{B}` and
centrifugal) and the Coriolis drift:

.. math::
   :label: rotating_drift

    \dot{\rho}_s = \mathbf{v}_{m,s} \cdot \nabla \rho
    - \frac{B_\zeta T_s}{q_s \sqrt{g} B^2} \frac{\partial \lambda_s}{\partial \theta}
    + \frac{2 m_s \omega v_\parallel}{q_s \sqrt{g} B} R \frac{\partial R}{\partial \theta}.

It is :math:`m_s / (q_s \psi')` times the rate of change of
:math:`B_\zeta v_\parallel / B + \omega R^2` along the collisionless trajectories, with
:math:`\psi' = \sqrt{g} B^\theta`, which is the conservation of the canonical angular
momentum [7]_. In the same way

.. math::
   :label: rotating_momentum_drift

    \dot{\pi}_s = \frac{m_s^2}{2 q_s \psi'} \frac{d}{dt} \Big[
    \Big( \frac{B_\zeta v_\parallel}{B} + \omega R^2 \Big)^2
    + \frac{v_\perp^2}{2 B^2} \big( R^2 B^2 - B_\zeta^2 \big) \Big]

is the toroidal angular momentum of a guiding center times its radial drift, with the
part of the gyromotion [7]_.

**Drive.** The right-hand side becomes

.. math::
   :label: rotating_drive

    S_s = -\big[ A_{1,s} + (x_s^2 + \lambda_s) A_{2,s} \big] \dot{\rho}_s F_{M,s}
        \;-\; \frac{1}{T_s} \frac{\partial \omega}{\partial \rho} \dot{\pi}_s F_{M,s}
        \;+\; A_{3,s}\, B v_\parallel F_{M,s},

with the forces of :eq:`forces`, where :math:`E_\rho` is the radial electric field in
the rotating frame at :math:`\theta = 0` and :math:`A_{1,s}` has the additional term

.. math::

    - \frac{m_s}{T_s} \Big( \omega \frac{\partial \omega}{\partial \rho} R^2
    + \omega^2 R \frac{\partial R}{\partial \rho} \Big)_{\theta = 0},

the radial derivative of :math:`\lambda_s` at :math:`\theta = 0`. A plasma that
rotates rigidly, with a density :math:`n_s e^{\lambda_s}` and a temperature that are
constant on and across the surfaces, has no drive.

**Collisions.** The collision operator is linearized about the local Maxwellians:
the collision frequencies :eq:`collisionalities` and :math:`F_{M,s}` in
:eq:`fp_collision_operator` take the densities :math:`n_s(\theta)`. The exchange of
energy between the species is removed as without rotation, for the surface averaged
density and energy :math:`x_s^2 + \lambda_s` of each species.

**Fluxes.** The fluxes of particles, heat and toroidal angular momentum,

.. math::

    \Gamma_s = \Big\langle \int d^3v\, f_s \dot{\rho}_s \Big\rangle, \qquad
    Q_s = \Big\langle \int d^3v\, f_s \big( \tfrac{1}{2} m_s v^2 + T_s \lambda_s \big) \dot{\rho}_s \Big\rangle, \qquad
    \Pi_s = \Big\langle \int d^3v\, f_s \dot{\pi}_s \Big\rangle,

are conjugate to the forces :math:`A_{1,s}`, :math:`A_{2,s}` and
:math:`\partial \omega / \partial \rho` of :eq:`rotating_drive`: on a surface with
up-down symmetry the transport matrix they define is symmetric [7]_. They are the
outputs ``"<particle_flux>"``, ``"<heat_flux>"`` and ``"<angular_momentum_flux>"``.
``"<energy_flux>"`` measures the potential energy from the surface average of
:math:`\Phi_0` and from the axis of rotation instead of :math:`\theta = 0`, and with
``frame="lab"`` it is the flux of the kinetic energy of the lab frame, which adds
:math:`\omega \Pi_s`.

On a surface without up-down symmetry these are not the whole fluxes. The part of the
distribution function driven by the rotation shear that is known in closed form adds
fluxes that do not depend on the solution of the DKE, the outputs
``"<nondissipative_particle_flux>"``, ``"<nondissipative_heat_flux>"`` and
``"<nondissipative_angular_momentum_flux>"``. Only the sum of both particle fluxes is
ambipolar [7]_.

**Numerics.** Rotation is supported by the preconditioner for axisymmetric fields only
(``preconditioner="tokamak"``, see :doc:`tuning`). The default stencil ``p1`` is then
the centered ``"4z"``: the upwind stencils converge slowly with resolution in a
collisional rotating plasma. The outputs for the perpendicular flows and the classical
fluxes are not available with rotation.

**Benchmark.** The tests compare with NEO [8]_ [9]_, with its full Fokker-Planck
collision operator, for deuterium, carbon and electrons at carbon Mach numbers up to
2.4: the potential and the densities on the surface, the fluxes that do not depend on
the solution, and the fluxes of particles, energy and angular momentum. With 12
speeds, 193 pitch angles and 65 poloidal points the fluxes of the DKE differ from those
of NEO by at most:

====================================================  ===============  ===============  ================
Case                                                  :math:`\Gamma`   :math:`Q`        :math:`\Pi`
                                                      of D, C          of D, e          of D, C
====================================================  ===============  ===============  ================
MAST-U H-mode, :math:`\rho` = 0.65 and 0.85           1.0%             0.2%             1.6%
The same at :math:`\rho` = 0.45, also with reversed   4.2%             2.6%             2.9%
rotation, reversed field and with shear only
Circular surface at high collisionality, carbon Mach  2.4%             0.4%             3.0%
number 2.4, with and without shear
Surface without up-down symmetry, with the fluxes     4.0%             1.6%             3.0%
that do not depend on the solution
====================================================  ===============  ===============  ================

Not listed are the electron particle flux, the small remainder of the ion fluxes (within
3%, except 10% at :math:`\rho` = 0.45 and 66% at :math:`\rho` = 0.85, where it nearly
vanishes), the carbon energy flux (within 5%, except at :math:`\rho` = 0.65, where it
is close to a change of sign) and the electron angular momentum flux, which is smaller
than that of the ions by the mass ratio. The fluxes that do not depend on the solution
agree within 0.2% by themselves, and the potential and the densities of a neutral
plasma to the digits NEO prints.

The Monoenergetic DKE
=====================

The MDKE is the same equation reduced to a single species at fixed particle
speed :math:`v`, with the collision operator replaced by pure pitch-angle
scattering at frequency :math:`\nu(v)`. This is the formulation introduced
by DKES [4]_ and used by MONKES [5]_:

.. math::
   :label: mdke

    \big( v\, \xi\, \mathbf{b} + \mathbf{v}_E \big) \cdot \nabla f_1
    \;-\; \frac{\nu}{2 \sin \alpha}
          \frac{\partial}{\partial \alpha}
          \left[ \sin \alpha \frac{\partial f_1}{\partial \alpha} \right]
    \;=\; S.

After dividing through by :math:`v`, only two scalar parameters remain:

.. math::

    \widehat{E_\rho} \;=\; E_\rho / v
    \quad [\mathrm{V\, s\, m^{-1}}],
    \qquad
    \widehat{\nu} \;=\; \nu / v
    \quad [\mathrm{m^{-1}}],

passed as ``erhohat`` and ``nuhat`` to
:func:`~yancc.solve_mdke`. yancc solves :eq:`mdke` two times - once
per unique drive (the first and second are the same when speed is ignored) - and
assembles the 3×3 monoenergetic transport matrix

.. math::

    D_{ij} = \langle s_i^{(j)} \cdot f^{(j)} \rangle,

returned by ``sol.get("Dij")``. A DKES-normalized form using the conventions
of [4]_ is also available (``"Dij_DKES"``); see :doc:`variables`.

Outputs
=======

For the full DKE, ``sol.get(name)`` computes radial particle and heat fluxes,
parallel flows, the bootstrap current, and SFINCS-normalized variants of
each, and the flux of toroidal angular momentum in axisymmetric fields. For the MDKE only the transport matrix is computed. The complete list
of names and the array shape returned by each is in :doc:`variables`.

References
==========

.. [1] P. Helander and D. J. Sigmar, *Collisional Transport in Magnetized
       Plasmas*, Cambridge University Press (2002).
.. [2] Beidler, C. D., et al. "Benchmarking of the mono-energetic transport
       coefficients—results from the International Collaboration on Neoclassical
       Transport in Stellarators (ICNTS)." Nuclear Fusion 51.7 (2011): 076001.
.. [3] Landreman, Matt, et al. "Comparison of particle trajectories and collision
       operators for collisional transport in nonaxisymmetric plasmas." Physics of
       Plasmas 21.4 (2014).
       SFINCS code: https://github.com/landreman/sfincs.
.. [4] Hirshman, S. P., et al. Plasma transport coefficients for nonsymmetric toroidal
       confinement systems. No. ORNL/TM--9925. Oak Ridge National Lab., TN (USA), 1986.
.. [5] Escoto, F. J., et al. "MONKES: a fast neoclassical code for the evaluation of
       monoenergetic transport coefficients in stellarator plasmas." Nuclear Fusion
       64.7 (2024): 076030.
       MONKES code: https://github.com/JavierEscoto/MONKES.
.. [6] Hinton, F. L., and S. K. Wong. "Neoclassical ion transport in rotating
       axisymmetric plasmas." Physics of Fluids 28.10 (1985): 3082.
.. [7] Sugama, H., and W. Horton. "Neoclassical electron and ion transport in
       toroidally rotating plasmas." Physics of Plasmas 4.6 (1997): 2215.
.. [8] Belli, E. A., J. Candy, and C. Angioni. "Pfirsch-Schlüter neoclassical heavy
       impurity transport in a rotating plasma." Plasma Physics and Controlled
       Fusion 56.12 (2014): 124002.
.. [9] Belli, E. A., and J. Candy. "Impact of centrifugal drifts on ion turbulent
       transport." Physics of Plasmas 25.3 (2018): 032301.
