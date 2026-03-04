#!/usr/bin/env python3
"""Run a dry Uranus GCM on a cubed-sphere grid.

Follows the style of run_sub_neptune.py. Uses:
- dry H2-He atmosphere with ideal-gas equation of state
- solar zenith angle calculated from Uranus orbit and obliquity
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import kintera
import torch
import yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

SECONDS_PER_DAY = 86400.0


@dataclass
class SolarForcingState:
    solar_flux: float          # Solar constant at Uranus [W/m^2]
    albedo: float              # Bond albedo
    obliquity_rad: float       # Planet obliquity [radians]
    true_anomaly0_rad: float   # True anomaly at t=0 [radians]
    orbital_period: float      # Orbital period [seconds]
    rotation_period: float     # Rotation period [seconds]
    lon: torch.Tensor          # Geographic longitude [radians], shape [nc3, nc2]
    lat: torch.Tensor          # Geographic latitude [radians], shape [nc3, nc2]
    gaussian_cool_weights: torch.Tensor  # shape [nz_active], normalised so sum(w_i*dz_i)=1
    bottom_depth: int


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(block: MeshBlock) -> torch.device:
    if torch.cuda.is_available() and block.options.layout().backend() == "nccl":
        return torch.device(block.device())
    return torch.device("cpu")


def create_model(config_file: str, output_dir: str | None = None):
    op = MeshBlockOptions.from_yaml(config_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        op.output_dir(output_dir)

    block = MeshBlock(op)
    device = select_device(block)
    block.to(device)

    eos = block.module("hydro.eos")
    return block, eos, device


def initialize_atmosphere(
    block: MeshBlock, eos, config: dict, device: torch.device
) -> tuple[dict, float]:
    """Initialize an isothermal atmosphere in hydrostatic balance."""
    coord = block.module("coord")
    grav = -block.options.hydro().grav().grav1()
    Rd = kintera.constants.Rgas / eos.options.weight()

    problem = config["problem"]
    Ts = float(problem["Ts"])
    Ps = float(problem["Ps"])

    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )
    z_surf = coord.buffer("x1f")[block.options.coord().nghost()]

    nc3, nc2, nc1 = x3v.shape
    w = torch.zeros((eos.nvar(), nc3, nc2, nc1), device=device)
    w[kIPR] = Ps * torch.exp(-grav * (x1v - z_surf) / (Rd * Ts))
    w[kIDN] = w[kIPR] / (Rd * Ts)
    w[kIV1] += 0.1 * torch.rand_like(w[kIV1])

    block_vars = {"hydro_w": w}
    return block.initialize(block_vars)


def _get_local_lonlat(block: MeshBlock) -> tuple[torch.Tensor, torch.Tensor]:
    """Return longitude and latitude arrays for the local MeshBlock face."""
    layout = snapy.distributed.get_layout()
    rank = int(snapy.distributed.get_rank())
    loc = layout.loc_of(rank)
    face_id = int(loc[2])
    face_name = snapy.coord.get_cs_face_name(face_id)

    coord = block.module("coord")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")
    beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
    lon, lat = snapy.coord.cs_ab_to_lonlat(face_name, alpha, beta)
    return lon, lat


def compute_cos_zenith(
    lat: torch.Tensor,
    lon: torch.Tensor,
    obliquity_rad: float,
    true_anomaly_rad: float,
    subsolar_lon_rad: float,
) -> torch.Tensor:
    """Compute cos(solar zenith angle) for a given Uranus orbital state.

    Args:
        lat: Geographic latitude [radians], shape [nc3, nc2].
        lon: Geographic longitude [radians], shape [nc3, nc2].
        obliquity_rad: Planet obliquity [radians].
        true_anomaly_rad: Orbital true anomaly from vernal equinox [radians].
        subsolar_lon_rad: Sub-solar longitude in the model frame [radians].

    Returns:
        cos(solar zenith angle), shape [nc3, nc2]; values not clipped to dayside.
    """
    # Sub-solar declination from the current orbital position
    declination = math.asin(math.sin(obliquity_rad) * math.sin(true_anomaly_rad))
    # Hour angle: difference between local longitude and sub-solar longitude
    cos_z = (
        torch.sin(lat) * math.sin(declination)
        + torch.cos(lat) * math.cos(declination) * torch.cos(lon - subsolar_lon_rad)
    )
    return cos_z


def build_solar_forcing_state(
    block: MeshBlock, config: dict, device: torch.device
) -> SolarForcingState:
    problem = config["problem"]

    lon, lat = _get_local_lonlat(block)
    lon = lon.to(device)
    lat = lat.to(device)

    # Build Gaussian cooling weights over active vertical levels.
    # Normalised so that sum(w_i * dz_i) = 1, guaranteeing total
    # column-integrated cooling equals mean_absorbed_flux per unit area.
    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    x1v = coord.buffer("x1v")
    dzf = coord.buffer("dx1f")
    z_active = x1v[il : iu + 1]
    dz_active = dzf[il : iu + 1]

    z0 = float(problem.get("cooling_center_height", float(x1v[iu].item())))
    sigma = float(
        problem.get("cooling_sigma", float((x1v[iu] - x1v[il]).item()) / 8.0)
    )
    gaussian_weights = torch.exp(-0.5 * ((z_active - z0) / sigma) ** 2)
    norm = (gaussian_weights * dz_active).sum()
    gaussian_cool_weights = (gaussian_weights / norm).to(device)

    return SolarForcingState(
        solar_flux=float(problem.get("solar_flux", 3.71)),
        albedo=float(problem.get("albedo", 0.3)),
        obliquity_rad=math.radians(float(problem.get("obliquity_deg", 97.77))),
        true_anomaly0_rad=math.radians(float(problem.get("true_anomaly_deg", 0.0))),
        orbital_period=float(problem.get("orbital_period", 2.651e9)),
        rotation_period=float(problem.get("rotation_period", 62064.0)),
        lon=lon,
        lat=lat,
        gaussian_cool_weights=gaussian_cool_weights,
        bottom_depth=int(problem.get("forcing_depth_bottom", 1)),
    )


def apply_solar_forcing(
    block: MeshBlock,
    block_vars: dict,
    forcing: SolarForcingState,
    current_time: float,
    dt: float,
) -> None:
    """Apply solar heating at the bottom and Gaussian cooling across all levels.

    The solar heating is computed from the instantaneous solar zenith angle,
    which accounts for both the diurnal cycle (planet rotation) and the
    seasonal cycle (orbital true anomaly evolving with obliquity). The global
    mean of max(cos(z), 0) over the sphere is always 1/4, so the cooling flux
    is set to solar_flux * (1 - albedo) * 0.25 to maintain energy balance.
    """
    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    dzf = coord.buffer("dx1f")
    bot_dz = dzf[il]

    # Advance orbital true anomaly and sub-solar longitude with time
    true_anomaly = forcing.true_anomaly0_rad + 2.0 * math.pi * current_time / forcing.orbital_period
    subsolar_lon = 2.0 * math.pi * current_time / forcing.rotation_period

    cos_z = compute_cos_zenith(
        forcing.lat, forcing.lon, forcing.obliquity_rad, true_anomaly, subsolar_lon
    )
    cos_z_day = torch.clamp(cos_z, min=0.0)

    absorbed_flux = forcing.solar_flux * (1.0 - forcing.albedo) * cos_z_day
    # Mean absorbed flux over the sphere for energy-balanced cooling
    mean_absorbed = forcing.solar_flux * (1.0 - forcing.albedo) * 0.25

    hydro_u = block_vars["hydro_u"]
    bot_depth = max(1, forcing.bottom_depth)

    # Heat the bottom layer(s) with absorbed solar flux
    heat_src = (absorbed_flux / (bot_dz * bot_depth)) * dt
    hydro_u[kIPR, ..., il : il + bot_depth] += heat_src.unsqueeze(-1)

    # Cool all active levels with Gaussian weights to balance heating
    cool_src = mean_absorbed * forcing.gaussian_cool_weights * dt
    hydro_u[kIPR, ..., il : iu + 1] -= cool_src


def run_simulation(
    block: MeshBlock,
    block_vars: dict,
    current_time: float,
    forcing: SolarForcingState,
) -> tuple[dict, float]:
    block.make_outputs(block_vars, current_time)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            apply_solar_forcing(block, block_vars, forcing, current_time, dt)

        err = block.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        current_time += dt
        block.make_outputs(block_vars, current_time)

    return block_vars, current_time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dry Uranus GCM simulation.")
    p.add_argument("-c", "--config", required=True, help="YAML configuration file")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument(
        "--restart-name",
        default="",
        help=(
            "Restart archive filename inside output dir (e.g. "
            "uranus_dry.00005.restart or uranus_dry.final.restart)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    block, eos, device = create_model(args.config, args.output_dir)

    if args.restart_name:
        block_vars, current_time = block.initialize_from_restart(args.restart_name)
    else:
        block_vars, current_time = initialize_atmosphere(block, eos, config, device)

    for key, data in block_vars.items():
        if isinstance(data, torch.Tensor):
            print(f"{key}: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}")

    forcing = build_solar_forcing_state(block, config, device)
    print(
        "Solar forcing:",
        f"solar_flux={forcing.solar_flux:.3f} W/m^2,",
        f"obliquity={math.degrees(forcing.obliquity_rad):.2f} deg,",
        f"true_anomaly0={math.degrees(forcing.true_anomaly0_rad):.2f} deg",
    )

    block_vars, current_time = run_simulation(block, block_vars, current_time, forcing)
    block.finalize(block_vars, current_time)


if __name__ == "__main__":
    main()
