# UM-URANUS

> Current driver: see [RUN_GLOBAL.rst](RUN_GLOBAL.rst) for the one-Earth-year
> six-face run and [RADIATIVE_OUTPUT.rst](RADIATIVE_OUTPUT.rst) for current
> output variables. The implementation and reproduction details below describe
> the earlier October 21 experiment and are retained as historical records.

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
Orbital insolation is evaluated through `torch.compile` in Python. Its Inductor
cache is shared by both GPU processes; TorchScript fusion remains enabled for
the opacity modules.

## Reference environment (2026-10-21)

The radiative-output experiment is in
`/home/chengcli/data/2026.Uranus/earth-year-flux-20261021` on the original host.
Its `source/` directory is the runnable source snapshot. It starts from
UM-URANUS commit `c3b4ac97bcede910a713e0d5d407dea2e712c1a5`, **plus** the
radiative-output changes to `run_uranus.py` and `rt_forcing.py` and the
`uranus-earth-year.yaml` overlay. Checking out that commit alone does not
reproduce `out2`.

| Component | Tested version / configuration |
| --- | --- |
| OS | RHEL 9.8, Linux x86_64, kernel `5.14.0-687.56.1.el9_8.x86_64` |
| GPUs | 2 NVIDIA GeForce RTX 5090, 32607 MiB each |
| NVIDIA driver | `615.71.10` |
| Python | `3.11.13` |
| PyTorch / CUDA runtime | `2.10.0` / `12.8` |
| Triton | `3.6.0` |
| UCX (system `ucx_info`) | `1.19.1` |
| snapy | `2.10.8` |
| paddle (atmospheric-model utilities, not PaddlePaddle) | `1.4.1` |
| pyharp | `2.6.5` |
| pydisort | `1.8.5` |
| kintera (installed metadata) | `2.4.11.dev6+gedfd3f855.d20260917` |
| NumPy / SciPy | `2.4.6` / `1.17.1` |
| PyYAML | `6.0.3` |
| xarray / netCDF4 | `2026.4.0` / `1.7.4` |
| matplotlib / pytest | `3.10.9` / `9.0.3` |

Use a Python 3.11 virtual environment and CUDA-enabled builds of the compiled
packages compatible with PyTorch 2.10.0 and the GPU architecture. Obtain the
reference wheels from the experiment owner for an exact binary repeat; package
version strings alone do not specify compiler options or linked libraries.
Install those wheels into the environment, not into the running experiment's
environment. The run's `requirements.freeze.txt` records additional Python
dependencies, but its Kintera `file:///data00/scix/repos/kintera` entry is a
host-local source path, **not** a portable installation requirement.

### Is this Kintera 2.5.0?

The installed distribution is not labelled `2.5.0`. However, upstream
[`v2.5.0`](https://github.com/chengcli/kintera/releases/tag/v2.5.0) is the squash
merge `9fa733312c09894f017ceeace128af3dd369c24f` (PR #110, "Add scaled KKT and
use memory pool"). Its **entire source tree is identical** to the local
`cli/scale_kkt` checkout `3e2c559364459ca8498708297d01eb368f8c253f`:

```text
Git tree, both revisions: 67548e68de9f3ea2345fe82fc60327a49b2a3f97
```

Thus `v2.5.0` is the matching published source revision for that checkout,
including the partition solver and KKT hardening. The `edfd3f855` commit named
in the installed version string is older and does differ from `v2.5.0`.
The installed extension exposes the newer `uv_solver` option and this run
uses `uv-solver: partition`; do not infer its complete source provenance from
the version label alone. Source-tree equivalence does not prove that a newly
built 2.5.0 wheel is binary-identical to the installed wheel, nor has that wheel
been substituted into this experiment.

For exact artifact comparison, the installed Kintera SHA-256 checksums are:

```text
d03e90541592f124ee247645fadcd28520faa456fdb915f8a739b2bf566c9fbf  kintera.cpython-311-x86_64-linux-gnu.so
ac55002cb6e210a08e5d7ca489d7a1b8dac3c71f04e84ab4324a1c9dcb33d33d  lib/libkintera_cuda_release.so
bb0a1e439bc6fc1684d8a927e2f4f0a948a017f141b0bf7c5859fb988a75d1ad  lib/libkintera_release.so
```

## Repeat the Earth-year run

Use the source snapshot above, or this working tree including the flux-output
changes and overlay. Do not launch inside another active run's source directory:
the seven `.pt` modules are stored beside `run_uranus.py`, not in `--output-dir`.

From the repository root, prepare a new, isolated run directory:

```bash
export RUN="$HOME/data/2026.Uranus/earth-year-repeat"
mkdir -p "$RUN"
mkdir "$RUN/source"
cp run_uranus.py rt_forcing.py opacity.py orbital.py \
  uranus.yaml uranus-earth-year.yaml "$RUN/source/"
cd "$RUN/source"

export DEVICE=cuda
export BACKEND=ucx
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$RUN/inductor-cache"

python - <<'PY'
from pathlib import Path
import yaml
from run_uranus import load_config, ensure_torchscripts

path = Path('resolved_config.yaml').resolve()
config = load_config('uranus-earth-year.yaml')
path.write_text(yaml.safe_dump(config, sort_keys=False))
ensure_torchscripts(config, path, force_build=True, thermo_config_path=path)
PY
python -m pip freeze > "$RUN/requirements.freeze.txt"
```

Resolve the YAML and build the scripts **once, serially**, before launching
workers. This avoids concurrent writes by both ranks. Existing `.pt` files are
otherwise reused without configuration/hash checks; rebuild serially whenever
opacity or orbital parameters change. The Inductor cache is explicitly set
above; the current runner does not set a project-local cache by default.

Validate with a two-step test at the full production resolution:

```bash
torchrun --standalone --nproc-per-node=2 run_uranus.py \
  --config resolved_config.yaml --output-dir "$RUN/smoke" --max-cycles 2 \
  > "$RUN/smoke.log" 2>&1
```

Confirm the command succeeds and `smoke/resolved_config.out2.00000.nc`
contains eight finite flux fields. A successful `--max-cycles` test may print
`Terminating abnormally` because it stops before `tlim`; check the exit status
and final restart rather than that message alone.

Start a fresh production run (not a continuation of the smoke test):

```bash
torchrun --standalone --nproc-per-node=2 run_uranus.py \
  --config resolved_config.yaml --output-dir "$RUN" > "$RUN/run.log" 2>&1
```

Run this command in a persistent terminal such as tmux. Use `BACKEND=ucx`:
this installed Snapy build does not accept `nccl` for its self-initialized
process group. Do not add a backend setting to the YAML. Both GPUs should be
visible to the process; optionally select them with `CUDA_VISIBLE_DEVICES=0,1`.
The short `run_uranus.sh` wrapper uses `uranus.yaml`, **not** the Earth-year
overlay, so it is not the command for this experiment.

### Configuration and outputs

- Six cubed-sphere panels, each 64 vertical layers by 96 x 96 cells;
  three panels per GPU. Nominal pressure range: about 20 to 0.01 bar.
- One Earth year is `31557600` simulated seconds (365.25 days).
- `uv-solver: partition`; internal heating is `0.042 W/m^2`.
- Radiation updates every `3600` simulated seconds.
- `out1`: primitive variables every `86400` seconds.
- `out2`: visible and IR upward/downward radiative fluxes every `86400`
  seconds, recomputed at output time. Lower/upper face values are stored
  on the cell grid; see [RADIATIVE_OUTPUT.rst](RADIATIVE_OUTPUT.rst) for
  names, units, staggering, and night-side handling.
- Numbered restarts every `864000` seconds (10 days), plus a final restart.

To stop gracefully, send SIGINT to both worker processes and wait for
`resolved_config.final.restart` to finish writing. To resume, use the same
source, scripts, and resolved configuration and append
`--restart /absolute/path/to/resolved_config.final.restart` to the command,
preferably with a new output directory. `tlim` is the absolute simulation end
time, not an additional year after restart.

Initial vertical-velocity perturbations use unseeded `torch.randn_like`.
Fresh runs repeat the setup, not bitwise-identical initial states. Preserve
the time-zero restart (`resolved_config.00000.restart`) to repeat the same
initial tensors. Different GPU kernels/builds can still change floating-point
trajectories. Archive the source snapshot, resolved YAML, scripts, package
wheels, logs, and restart together.

The reference flux run's `manifest.json` inherited its `command`,
`launch_time`, and `validation` entries from the preceding non-flux run;
those entries are not authoritative for the flux launch. Use its `launch.sh`,
`source/resolved_config.yaml`, `run.log`, and `smoke.log` instead.

## Tests

```bash
python -m pytest -q test_radiative_output.py
python -m pytest -q test_orbit_compile.py
```

The orbital test requires two available GPUs. The legacy `test_uranus.py`
currently imports removed symbols from `run_uranus.py`; it needs updating
before it can serve as a reproduction check.

## Plotting

```bash
python plot_uranus_overview.py uranus-latlon.00029.nc -o uranus-latlon.00029.png
```

The cloud panels combine the nonprecipitating and precipitating condensate
fields for each species. Cloud path integrates density times total condensate
mass fraction over layer thickness. The first two panels use zonal-mean
pressure on a logarithmic vertical axis.
