#!/usr/bin/env python3
"""Run a tidally locked Sub-Neptune GCM using snapy/paddle/kintera.

This script follows the existing run_* patterns in UM-EARTH while using
paddle.setup_profile for hydrostatic/isothermal initialization.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIV1, kICY, kIDN, kIPR
from kintera import Kinetics, KineticsOptions, ThermoX
from paddle import evolve_kinetics, setup_profile

SECONDS_PER_DAY = 86400.0


@dataclass
class ForcingState:
    cos_zenith_dayside: torch.Tensor
    absorbed_surface_flux: float
    mean_cooling_flux: float
    bottom_depth: int
    gaussian_cool_weights: torch.Tensor  # shape [nz_active], 1/m, normalised so sum(w_i*dz_i)=1


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(block: MeshBlock) -> torch.device:
    if torch.cuda.is_available() and block.options.layout().backend() == "nccl":
        return torch.device(block.device())
    return torch.device("cpu")


def create_models(config_file: str, output_dir: str | None = None):
    op = MeshBlockOptions.from_yaml(config_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        op.output_dir(output_dir)

    block = MeshBlock(op)
    device = select_device(block)
    block.to(device)

    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    op_kin = KineticsOptions.from_yaml(config_file)
    kinet = Kinetics(op_kin)
    kinet.to(device)

    eos = block.module("hydro.eos")
    return block, eos, thermo_y, thermo_x, kinet, device


def initialize_isothermal(block: MeshBlock, config: dict) -> tuple[dict[str, torch.Tensor], float]:
    grav = -float(config["forcing"]["const-gravity"]["grav1"])
    problem = config["problem"]

    param = {
        "Ts": float(problem["Ts"]),
        "Ps": float(problem["Ps"]),
        "Tmin": float(problem.get("Tmin", problem["Ts"])),
        "grav": grav,
    }

    thermo_y = block.module("hydro.eos.thermo")
    for name in thermo_y.options.species():
        param[f"x{name}"] = float(problem.get(f"x{name}", 0.0))

    hydro_w = setup_profile(block, param, method="pseudo-adiabat")

    # add random noise to IV1
    hydro_w[kIV1] += 1e-6 * torch.randn_like(hydro_w[kIV1])

    return block.initialize({"hydro_w": hydro_w})


def _resolve_local_face_name(block: MeshBlock) -> str:
    layout = snapy.distributed.get_layout()
    rank = int(snapy.distributed.get_rank())
    loc = layout.loc_of(rank)
    face_id = int(loc[2])
    return snapy.coord.get_cs_face_name(face_id)


def build_tidal_forcing_state(block: MeshBlock, config: dict, device: torch.device) -> ForcingState:
    coord = block.module("coord")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")

    beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
    face_name = _resolve_local_face_name(block)
    lon, lat = snapy.coord.cs_ab_to_lonlat(face_name, alpha, beta)

    problem = config["problem"]
    lon0 = math.radians(float(problem.get("substellar_lon_deg", 0.0)))
    lat0 = math.radians(float(problem.get("substellar_lat_deg", 0.0)))

    cos_zenith = (
        torch.sin(lat) * math.sin(lat0)
        + torch.cos(lat) * math.cos(lat0) * torch.cos(lon - lon0)
    )
    cos_zenith_dayside = torch.clamp(cos_zenith, min=0.0).to(device)

    stellar_flux = float(problem["stellar_flux_nadir"])
    frac_to_surface = float(problem["stellar_surface_fraction"])
    absorbed_surface_flux = stellar_flux * frac_to_surface

    # Mean of max(cos(zenith), 0) over the sphere is 1/4, so this cooling flux
    # exactly balances globally integrated dayside heating for a spherical planet.
    mean_cooling_flux = absorbed_surface_flux * 0.25

    # Build Gaussian cooling weights over active vertical levels.
    # Normalised so that sum(w_i * dz_i) = 1, which guarantees the total
    # column-integrated cooling equals mean_cooling_flux per unit area.
    il, iu = coord.il(), coord.iu()
    x1v = coord.buffer("x1v")
    dzf = coord.buffer("dx1f")
    z_active = x1v[il : iu + 1]
    dz_active = dzf[il : iu + 1]

    z0 = float(problem.get("cooling_center_height", float(x1v[iu].item())))
    sigma = float(
        # Default: domain height / 8 gives a spread covering ~1/8 of the column
        problem.get("cooling_sigma", float((x1v[iu] - x1v[il]).item()) / 8.0)
    )

    gaussian_weights = torch.exp(-0.5 * ((z_active - z0) / sigma) ** 2)
    norm = (gaussian_weights * dz_active).sum()
    gaussian_cool_weights = (gaussian_weights / norm).to(device)

    return ForcingState(
        cos_zenith_dayside=cos_zenith_dayside,
        absorbed_surface_flux=absorbed_surface_flux,
        mean_cooling_flux=mean_cooling_flux,
        bottom_depth=int(problem.get("forcing_depth_bottom", 1)),
        gaussian_cool_weights=gaussian_cool_weights,
    )


def apply_tidal_forcing(block: MeshBlock, block_vars: dict[str, torch.Tensor], forcing: ForcingState, dt: float) -> None:
    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    dzf = coord.buffer("dx1f")

    hydro_u = block_vars["hydro_u"]

    bot_depth = max(1, forcing.bottom_depth)

    bot_dz = dzf[il]

    heat_flux_local = forcing.absorbed_surface_flux * forcing.cos_zenith_dayside
    heat_src = (heat_flux_local / (bot_dz * bot_depth)) * dt

    # Gaussian cooling: weights are normalised so sum(w_i * dz_i) = 1,
    # guaranteeing total column-integrated cooling equals mean_cooling_flux per unit area.
    cool_src = forcing.mean_cooling_flux * forcing.gaussian_cool_weights * dt

    hydro_u[kIPR, ..., il : il + bot_depth] += heat_src.unsqueeze(-1)
    hydro_u[kIPR, ..., il : iu + 1] -= cool_src


def write_restart_manifest(
    checkpoint_dir: Path,
    checkpoint_day: int,
    current_time: float,
    config_file: str,
    output_dir: str,
    basename: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    restart_candidates = sorted(glob.glob(str(Path(output_dir) / f"{basename}.*.restart")))
    restart_file = restart_candidates[-1] if restart_candidates else None

    payload = {
        "checkpoint_day": checkpoint_day,
        "simulation_time_seconds": float(current_time),
        "simulation_time_days": float(current_time / SECONDS_PER_DAY),
        "config_file": str(Path(config_file).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "latest_restart_archive": restart_file,
        "resume_hint": {
            "command": (
                "python sub_neptune/run_sub_neptune.py "
                f"-c {config_file} --output-dir {output_dir} --restart-name "
                + (Path(restart_file).name if restart_file else "<restart-file-name>")
            )
        },
    }

    manifest_file = checkpoint_dir / f"checkpoint_day_{checkpoint_day:04d}.yaml"
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def run_simulation(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    kinet: Kinetics,
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    tlim: float,
    forcing: ForcingState,
    config_file: str,
    output_dir: str,
    basename: str,
) -> tuple[dict[str, torch.Tensor], float]:
    block.options.intg().tlim(tlim)

    next_checkpoint_day = int(current_time // (10.0 * SECONDS_PER_DAY)) * 10 + 10
    checkpoint_dir = Path(output_dir) / "restart_checkpoints"

    block.make_outputs(block_vars, current_time)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            apply_tidal_forcing(block, block_vars, forcing, dt)

        err = block.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        del_rho = evolve_kinetics(block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt)
        block_vars["hydro_u"][kICY:] += del_rho

        current_time += dt
        block.make_outputs(block_vars, current_time)

        while current_time >= next_checkpoint_day * SECONDS_PER_DAY:
            write_restart_manifest(
                checkpoint_dir=checkpoint_dir,
                checkpoint_day=next_checkpoint_day,
                current_time=current_time,
                config_file=config_file,
                output_dir=output_dir,
                basename=basename,
            )
            next_checkpoint_day += 10

    return block_vars, current_time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tidally locked Sub-Neptune simulation.")
    p.add_argument("-c", "--config", required=True, help="YAML configuration file")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument(
        "--restart-name",
        default="",
        help=(
            "Restart archive filename inside output dir (e.g. "
            "sub_neptune_tidallock.00005.restart or sub_neptune_tidallock.final.restart)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    block, eos, thermo_y, thermo_x, kinet, device = create_models(args.config, args.output_dir)

    if args.restart_name:
        block_vars, current_time = block.initialize_from_restart(args.restart_name)
    else:
        block_vars, current_time = initialize_isothermal(block, config)

    for key, data in block_vars.items():
        if isinstance(data, torch.Tensor):
            print(f"{key}: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}")

    forcing = build_tidal_forcing_state(block, config, device)
    print(
        "Forcing summary:",
        f"absorbed_surface_flux={forcing.absorbed_surface_flux:.3f} W/m^2,",
        f"gaussian_cooling_flux={forcing.mean_cooling_flux:.3f} W/m^2",
    )

    tlim = float(config["integration"]["tlim"])
    basename = Path(args.config).stem
    block_vars, current_time = run_simulation(
        block=block,
        eos=eos,
        thermo_y=thermo_y,
        thermo_x=thermo_x,
        kinet=kinet,
        block_vars=block_vars,
        current_time=current_time,
        tlim=tlim,
        forcing=forcing,
        config_file=args.config,
        output_dir=args.output_dir,
        basename=basename,
    )

    block.finalize(block_vars, current_time)


if __name__ == "__main__":
    main()
