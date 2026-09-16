# UM-URANUS

Moist cubed-sphere Uranus GCM with CH4 and H2S condensation and pyharp
two-stream radiation. The production grid has 64 vertical layers and 96 x 96
cells per panel. Six panels are distributed as three blocks on each of two GPUs.

## Radiation and orbit

`run_uranus.py` builds seven serialized TorchScript modules beside `run_uranus.py`: visible
and IR gas opacity, visible and IR opacity for each CH4 and H2S cloud, and an
orbital-forcing module. Gas mass opacity follows
`kappa = kappa_ref (p / p_ref)^exponent`; nonprecipitating cloud extinction is
proportional to condensate mass density. Cloud single-scattering albedo and
asymmetry are configured independently for each species and band.

The orbital module solves Kepler's equation and derives normal stellar
irradiance from `Lstar / (4 pi r^2)`. It returns both the zenith-angle cosine
and beam flux. Nightside columns receive exactly zero stellar beam. Time zero
is perihelion with the prime meridian under the star. Opacity coefficients are
tunable grey starting values, informed by the pressure-dependent treatment in
`Zhang_2023_ApJ_957_22.pdf`, rather than calibrated Uranus retrievals.

Uniform internal heating uses snapy's built-in `forcing.bot-heat` module.

## Run

```bash
./run_uranus.sh output
```

Equivalent command:

```bash
DEVICE=cuda torchrun --standalone --nproc-per-node=2 run_uranus.py \
  --config uranus.yaml --output-dir output
```

Existing `.pt` files are reused by default. Use `--force-build` after changing
opacity or orbital parameters; missing files are built automatically. Molecular
weights come from kintera. Resume with `--restart FILE`.

## Tests

```bash
python -m pytest -q test_uranus.py
```
