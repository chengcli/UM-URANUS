Radiation and user output
=========================

The driver writes daily hydro slices to ``uranus.out1.*.nc`` (one-quarter),
``uranus.out3.*.nc`` (one-half), and ``uranus.out4.*.nc`` (three-quarters). Each contains
density, pressure, all three local velocity components, and CH4/H2S vapor,
cloud, and precipitation mass fractions. The dimensions are
``(time, x1, x3, x2)``, with shape ``(1, 1, 192, 288)`` on the production grid.

The three ``x1_slice`` radii in ``uranus.yaml`` are at 1/4, 1/2, and 3/4
of the radial domain measured from its lower boundary: 25619425 m,
25679850 m, and 25740275 m. These select active cell indices 16, 32,
and 48 (zero-based) on the 64-layer production grid. Snapy selects
the cell containing each radius; these are fixed-radius slices, not
pressure surfaces. Change those radii to choose different layers.

User variables and paths remain in ``uranus.out2.*.nc``. Its cell-centered
fields retain all 64 layers, while net flux retains all 65 radial faces.
All files use Snapy's stitched six-face horizontal layout.

``out2`` contains:

* ``temp``: temperature in K.
* ``qtol``: total non-dry mass fraction, kg/kg.
* ``rt_heating``: volumetric radiative heating, W/m^3.
* ``net_flux``: total upward-minus-downward radiative flux, W/m^2, on
  the ``CCF`` grid (horizontal cell centers and radial faces).

Following the Jupiter case, the ``path`` output writes vapor and condensate
paths separately for every species:

* Vapor: ``path_CH4``, ``path_H2S``.
* Cloud condensate: ``path_CH4_s_``, ``path_H2S_s_``.
* Precipitating condensate: ``path_CH4_s_p_``, ``path_H2S_s_p_``.

These fields have units ``kg/m^2`` and dimensions ``(time, x3, x2)``.
Snapy integrates species mass over active radial cells and divides by the
bottom face area, including the spherical geometry:

.. math::

   P_s = \frac{\sum_i \rho_i q_{s,i} V_i}{A_\mathrm{bottom}}.

Net flux includes both shortwave and longwave bands. Its dimensions are
``(time, x1f, x3, x2)``, or ``(1, 65, 192, 288)`` for the production grid.
Each radial interface is stored once, including the bottom and top
boundaries. The heating can be reconstructed from adjacent faces as

.. math::

   q_i=-\frac{A_{i+1/2}F_{i+1/2}-A_{i-1/2}F_{i-1/2}}{V_i}.

The step is clipped to hourly radiation deadlines and daily output times.
Radiation is refreshed after moist chemistry at those deadlines, before
writing the output. Heating remains fixed between solves. Thermal flux can
be nonzero on the night side; only incident shortwave radiation vanishes.
Snapy assigns the ``CCF`` grid and units ``W/m^2`` to names ending in
``_flux``; this suffix is reserved for energy fluxes. This driver requires
these suffix rules in the installed Snapy library.

Numbered restart bundles are written every ten days, and a final restart is
written when the driver finishes. A short ``--max-cycles`` test always writes
a final restart; NetCDF files follow their configured output cadence.
