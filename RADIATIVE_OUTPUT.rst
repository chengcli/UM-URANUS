Radiative flux output
=====================

The Earth-year configuration writes daily radiative flux snapshots to
``resolved_config.out2.*.nc`` using Snapy's user-output callback.
Fluxes are recomputed at the snapshot time, rather than using the last hourly
radiative heating update. This does not alter the heating update schedule.

The eight fields are ``vis_flux_{up,dn}_{lower,upper}`` and
``ir_flux_{up,dn}_{lower,upper}``, all in W/m^2. Upward and downward
fluxes are positive in their respective directions. Net outward flux is
upward minus downward. Visible flux is exactly zero on the night side.

Each field uses the cell grid, but stores the flux at that cell's lower or
upper radial face, not its center. Keeping both faces preserves the bottom
and top boundary fluxes and permits an exact flux-divergence calculation.
This avoids treating Snapy's built-in ``rflx_up`` and ``rflx_dn`` profile
metadata as three-dimensional fields. Custom-field units are documented
here because Snapy's metadata table does not expose Python registration.
