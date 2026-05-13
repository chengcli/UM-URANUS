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
import kintera
from snapy import Mesh, MeshOptions, kIV1, kICY, kIDN, kIPR, kConserved
from kintera import Kinetics, KineticsOptions, ThermoX
from paddle import evolve_kinetics, setup_profile

SECONDS_PER_DAY = 86400.0


@dataclass
class ForcingState:
    cos_zenith_dayside: torch.Tensor
    absorbed_surface_flux: float
    mean_cooling_flux: float
    bottom_depth: int
    gaussian_cool_weights: torch.Tensor  # shape [nz_active], 1/m, half-Gaussian half-uniform, normalised so sum(w_i*dz_i)=1


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(block: snapy.MeshBlock) -> torch.device:
    if torch.cuda.is_available() and block.options.layout().backend() == "nccl":
        return torch.device(block.options.device_str())
    return torch.device("cpu")


def create_models(config_file: str, output_dir: str | None = None):
    op = MeshOptions.from_yaml(config_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        op.block().output_dir(output_dir)

    mesh = Mesh(op)
    block = mesh.blocks[0]
    device = select_device(block)
    mesh.to(device)

    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    op_kin = KineticsOptions.from_yaml(config_file)
    kinet = Kinetics(op_kin)
    kinet.to(device)

    eos = block.module("hydro.eos")
    return mesh, eos, thermo_y, thermo_x, kinet, device


def fill_initial_profile_ghosts(hydro_w: torch.Tensor) -> None:
    valid = torch.isfinite(hydro_w[kIDN]) & torch.isfinite(hydro_w[kIPR]) & (hydro_w[kIDN] > 0) & (hydro_w[kIPR] > 0)
    if not torch.any(valid):
        return

    valid_idx = torch.nonzero(valid, as_tuple=False)
    lo = valid_idx.min(dim=0).values
    hi = valid_idx.max(dim=0).values
    idx3 = torch.arange(hydro_w.shape[1], device=hydro_w.device).clamp(int(lo[0].item()), int(hi[0].item()))
    idx2 = torch.arange(hydro_w.shape[2], device=hydro_w.device).clamp(int(lo[1].item()), int(hi[1].item()))
    idx1 = torch.arange(hydro_w.shape[3], device=hydro_w.device).clamp(int(lo[2].item()), int(hi[2].item()))
    hydro_w.copy_(hydro_w[:, idx3, :, :][:, :, idx2, :][:, :, :, idx1])


def validate_initial_profile(hydro_w: torch.Tensor, param: dict[str, float]) -> None:
    density = hydro_w[kIDN]
    pressure = hydro_w[kIPR]
    if torch.isfinite(hydro_w).all() and torch.all(density > 0) and torch.all(pressure > 0):
        return

    species = {key: value for key, value in param.items() if key.startswith("x")}
    raise RuntimeError(
        "Invalid initial primitive state from setup_profile: "
        f"density_min={density.min().item():.6e}, density_max={density.max().item():.6e}, "
        f"pressure_min={pressure.min().item():.6e}, pressure_max={pressure.max().item():.6e}, "
        f"species={species}"
    )


def print_initial_profile_summary(hydro_w: torch.Tensor) -> None:
    density = hydro_w[kIDN]
    pressure = hydro_w[kIPR]
    print(
        "Initial primitive summary:",
        f"density=[{density.min().item():.6e}, {density.max().item():.6e}]",
        f"pressure=[{pressure.min().item():.6e}, {pressure.max().item():.6e}]",
        flush=True,
    )


def build_exact_isothermal_profile(block: snapy.MeshBlock, param: dict[str, float]) -> torch.Tensor:
    grav = float(param["grav"])
    temp_value = float(param["Ts"])
    pres_value = float(param["Ps"])

    coord = block.module("coord")
    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)

    x1v = coord.buffer("x1v")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")
    dx1f = coord.buffer("dx1f")

    thermo_x.to(dtype=x1v.dtype, device=x1v.device)

    ny = len(thermo_y.options.species()) - 1
    nvar = 5 + ny
    hydro_w = torch.zeros((nvar, x3v.shape[0], x2v.shape[0], x1v.shape[0]), dtype=x1v.dtype, device=x1v.device)

    temp = torch.full((x3v.shape[0], x2v.shape[0]), temp_value, dtype=x1v.dtype, device=x1v.device)
    pres = torch.full_like(temp, pres_value)
    xfrac = torch.zeros((*temp.shape, ny + 1), dtype=x1v.dtype, device=x1v.device)

    for name in thermo_y.options.species():
        index = thermo_y.options.species().index(name)
        xfrac[..., index] = float(param.get(f"x{name}", 0.0))
    xfrac[..., 0] = 1.0 - xfrac[..., 1:].sum(dim=-1)

    for cid in thermo_x.options.cloud_ids():
        xfrac[..., cid] = 0.0
    xfrac /= xfrac.sum(dim=-1, keepdim=True)
    mu = (thermo_x.mu * xfrac).sum(-1)

    il = coord.il()
    hydro_factor = torch.exp(-grav * mu.unsqueeze(-1) * dx1f / (kintera.constants.Rgas * temp.unsqueeze(-1)))

    for i in range(il - 1, -1, -1):
        pres /= hydro_factor[..., i + 1]
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
        hydro_w[kIPR, ..., i] = pres
        hydro_w[kIDN, ..., i] = thermo_x.compute("V->D", [conc])
        hydro_w[kICY:, ..., i] = thermo_x.compute("X->Y", [xfrac])

    pres = torch.full_like(temp, pres_value)
    for i in range(il, x1v.shape[0]):
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
        hydro_w[kIPR, ..., i] = pres
        hydro_w[kIDN, ..., i] = thermo_x.compute("V->D", [conc])
        hydro_w[kICY:, ..., i] = thermo_x.compute("X->Y", [xfrac])
        pres *= hydro_factor[..., i]

    return hydro_w


def initialize_isothermal(mesh: Mesh, config: dict) -> tuple[list[dict[str, torch.Tensor]], float]:
    grav = -float(config["forcing"]["const-gravity"]["grav1"])
    problem = config["problem"]

    param = {
        "Ts": float(problem["Ts"]),
        "Ps": float(problem["Ps"]),
        "Tmin": float(problem.get("Tmin", problem["Ts"])),
        "grav": grav,
    }

    thermo_y = mesh.blocks[0].module("hydro.eos.thermo")
    for name in thermo_y.options.species():
        param[f"x{name}"] = float(problem.get(f"x{name}", 0.0))

    mesh_vars: list[dict[str, torch.Tensor]] = []
    profile_method = str(problem.get("initial_profile_method", "pseudo-adiabat"))
    print(f"Initial profile method: {profile_method}", flush=True)
    for block in mesh.blocks:
        if profile_method == "isothermal":
            hydro_w = build_exact_isothermal_profile(block, param)
            print("Exact isothermal profile: preserving hydrostatic ghost cells", flush=True)
        else:
            hydro_w = setup_profile(block, param, method=profile_method)
            fill_initial_profile_ghosts(hydro_w)
        validate_initial_profile(hydro_w, param)
        print_initial_profile_summary(hydro_w)

        noise_amp = float(problem.get("initial_velocity_noise", 1.e-6))
        if noise_amp != 0.0:
            hydro_w[kIV1] += noise_amp * torch.randn_like(hydro_w[kIV1])

        mesh_vars.append({"hydro_w": hydro_w})

    return mesh.initialize(mesh_vars)


def _resolve_local_face_name(block: snapy.MeshBlock) -> str:
    layout = block.get_layout()
    rank = int(layout.options.rank())
    loc = layout.loc_of(rank)
    face_id = int(loc[2])
    return snapy.coord.get_cs_face_name(face_id)


def build_tidal_forcing_state(block: snapy.MeshBlock, config: dict, device: torch.device) -> ForcingState:
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

    # Build half-Gaussian, half-uniform cooling weights over active vertical levels.
    # Upper part (z >= z0): Gaussian profile
    # Lower part (z < z0): Constant value equal to Gaussian peak
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

    # Compute Gaussian profile for all levels
    gaussian_weights = torch.exp(-0.5 * ((z_active - z0) / sigma) ** 2)

    # Replace lower part (z < z0) with constant value equal to peak (which is 1.0)
    # The peak of the Gaussian is at z0, where exp(0) = 1.0
    mask_lower = z_active < z0
    gaussian_weights[mask_lower] = 1.0

    # Normalize so that sum(w_i * dz_i) = 1
    norm = (gaussian_weights * dz_active).sum()
    gaussian_cool_weights = (gaussian_weights / norm).to(device)

    return ForcingState(
        cos_zenith_dayside=cos_zenith_dayside,
        absorbed_surface_flux=absorbed_surface_flux,
        mean_cooling_flux=mean_cooling_flux,
        bottom_depth=int(problem.get("forcing_depth_bottom", 1)),
        gaussian_cool_weights=gaussian_cool_weights,
    )


def apply_tidal_forcing(
    block: snapy.MeshBlock, block_vars: dict[str, torch.Tensor], forcing: ForcingState, dt: float
) -> None:
    if forcing.absorbed_surface_flux == 0.0 and forcing.mean_cooling_flux == 0.0:
        return

    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    dzf = coord.buffer("dx1f")

    hydro_u = block_vars["hydro_u"]

    bot_depth = max(1, forcing.bottom_depth)

    bot_dz = dzf[il]

    heat_flux_local = forcing.absorbed_surface_flux * forcing.cos_zenith_dayside
    heat_src = (heat_flux_local / (bot_dz * bot_depth)) * dt

    # Half-Gaussian, half-uniform cooling: weights are normalised so sum(w_i * dz_i) = 1,
    # guaranteeing total column-integrated cooling equals mean_cooling_flux per unit area.
    cool_src = forcing.mean_cooling_flux * forcing.gaussian_cool_weights * dt

    hydro_u[kIPR, ..., il : il + bot_depth] += heat_src.unsqueeze(-1)
    hydro_u[kIPR, ..., il : iu + 1] -= cool_src

    # re-apply boundary conditions
    block.apply_hydro_bc(hydro_u, type=kConserved)


def apply_vertical_damping(
    block: snapy.MeshBlock, block_vars: dict[str, torch.Tensor], config: dict, dt: float
) -> None:
    problem = config.get("problem", {})
    top_layers = int(problem.get("vertical_damping_top_layers", 0))
    timescale = float(problem.get("vertical_damping_timescale", 0.0))
    if top_layers <= 0 or timescale <= 0.0:
        return

    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    start = max(il, iu - top_layers + 1)
    nlayer = iu - start + 1
    if nlayer <= 0:
        return

    ramp = torch.linspace(
        1.0 / nlayer,
        1.0,
        nlayer,
        dtype=block_vars["hydro_u"].dtype,
        device=block_vars["hydro_u"].device,
    )
    damping = torch.exp(-(dt / timescale) * ramp * ramp)

    hydro_u = block_vars["hydro_u"]
    hydro_u[kIV1, ..., start : iu + 1] *= damping

    hydro_w = block_vars.get("hydro_w")
    if hydro_w is not None:
        hydro_w[kIV1, ..., start : iu + 1] *= damping

    block.apply_hydro_bc(hydro_u, type=kConserved)


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


def _tensor_range(name: str, tensor: torch.Tensor) -> str:
    data = tensor.detach()
    finite = torch.isfinite(data)
    finite_count = int(finite.sum().item())
    total = data.numel()
    if finite_count == 0:
        return f"{name}=finite 0/{total}"
    values = data[finite]
    return (
        f"{name}=finite {finite_count}/{total} "
        f"min={values.min().item():.6e} max={values.max().item():.6e}"
    )


def _active_view(block: snapy.MeshBlock, tensor: torch.Tensor) -> torch.Tensor:
    coord = block.module("coord")
    return tensor[..., coord.kl() : coord.ku() + 1, coord.jl() : coord.ju() + 1, coord.il() : coord.iu() + 1]


def print_state_summary(label: str, mesh: Mesh, mesh_vars: list[dict[str, torch.Tensor]]) -> None:
    parts = [label]
    for block_index, (block, block_vars) in enumerate(zip(mesh.blocks, mesh_vars)):
        hydro_w = block_vars.get("hydro_w")
        hydro_u = block_vars.get("hydro_u")
        if hydro_w is not None:
            parts.extend(
                [
                    f"block={block_index}",
                    _tensor_range("rho_w", _active_view(block, hydro_w[kIDN])),
                    _tensor_range("p_w", _active_view(block, hydro_w[kIPR])),
                    _tensor_range("v1_w", _active_view(block, hydro_w[kIV1])),
                    _tensor_range("v2_w", _active_view(block, hydro_w[kIV1 + 1])),
                    _tensor_range("v3_w", _active_view(block, hydro_w[kIV1 + 2])),
                ]
            )
        if hydro_u is not None:
            parts.extend(
                [
                    _tensor_range("rho_u", _active_view(block, hydro_u[kIDN])),
                    _tensor_range("e_u", _active_view(block, hydro_u[kIPR])),
                ]
            )
    print(" | ".join(parts), flush=True)


def print_velocity_summary(label: str, mesh: Mesh, mesh_vars: list[dict[str, torch.Tensor]]) -> None:
    vmax = 0.0
    pmin = float("inf")
    pmax = float("-inf")
    for block, block_vars in zip(mesh.blocks, mesh_vars):
        hydro_w = block_vars["hydro_w"]
        v1 = _active_view(block, hydro_w[kIV1])
        pressure = _active_view(block, hydro_w[kIPR])
        vmax = max(vmax, float(torch.max(torch.abs(v1)).item()))
        pmin = min(pmin, float(torch.min(pressure).item()))
        pmax = max(pmax, float(torch.max(pressure).item()))
    print(f"{label} max_abs_v1={vmax:.6e} p=[{pmin:.6e}, {pmax:.6e}]", flush=True)


def run_simulation(
    mesh: Mesh,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    kinet: Kinetics,
    mesh_vars: list[dict[str, torch.Tensor]],
    current_time: float,
    forcing_states: list[ForcingState],
    config_file: str,
    output_dir: str,
    basename: str,
) -> tuple[list[dict[str, torch.Tensor]], float]:
    config = load_config(config_file)
    intg = mesh.module("block0.intg")
    tlim = float(config["integration"]["tlim"])
    nlim = int(config["integration"].get("nlim", -1))
    intg.options.tlim(tlim)
    skip_kinetics = bool(config.get("problem", {}).get("skip_kinetics", False))
    if skip_kinetics:
        print("Kinetics disabled by problem.skip_kinetics=true", flush=True)

    next_checkpoint_day = int(current_time // (10.0 * SECONDS_PER_DAY)) * 10 + 10
    checkpoint_dir = Path(output_dir) / "restart_checkpoints"
    stop_reason = "integration stop condition"

    print(f"[phase] before initial make_outputs time={current_time:.14e}", flush=True)
    mesh.make_outputs(mesh_vars, current_time)
    print(f"[phase] after initial make_outputs time={current_time:.14e}", flush=True)

    cycle = 0
    while not intg.stop(cycle, current_time):
        cycle += 1
        print(f"[phase] cycle {cycle} start time={current_time:.14e}", flush=True)
        mesh.set_cycle(cycle)

        print(f"[phase] cycle {cycle} before max_time_step", flush=True)
        dt = mesh.max_time_step(mesh_vars)
        print(f"[phase] cycle {cycle} after max_time_step dt={dt:.14e}", flush=True)
        mesh.print_cycle_info(mesh_vars, current_time, dt)

        for stage in range(len(intg.stages)):
            print(f"[phase] cycle {cycle} before forward stage={stage}", flush=True)
            mesh.forward(mesh_vars, dt, stage)
            print(f"[phase] cycle {cycle} after forward stage={stage}", flush=True)
            for block, block_vars, forcing in zip(mesh.blocks, mesh_vars, forcing_states):
                apply_tidal_forcing(block, block_vars, forcing, dt)
                apply_vertical_damping(block, block_vars, config, dt)
            print(f"[phase] cycle {cycle} after forcing stage={stage}", flush=True)
            if cycle <= 12:
                print_state_summary(f"[state] cycle={cycle} stage={stage}", mesh, mesh_vars)
            else:
                print_velocity_summary(f"[state] cycle={cycle} stage={stage}", mesh, mesh_vars)

        print(f"[phase] cycle {cycle} before check_redo", flush=True)
        err = mesh.check_redo(mesh_vars)
        print(f"[phase] cycle {cycle} after check_redo err={err}", flush=True)
        if err > 0:
            print(
                f"Redo requested by mesh.check_redo: cycle={cycle} "
                f"time={current_time:.14e} err={err}",
                flush=True,
            )
            continue
        if err < 0:
            stop_reason = f"mesh.check_redo returned {err}"
            print(
                f"Stopping because {stop_reason}: cycle={cycle} "
                f"time={current_time:.14e} dt={dt:.14e}",
                flush=True,
            )
            break

        if not skip_kinetics:
            for block_vars in mesh_vars:
                del_rho = evolve_kinetics(block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt)
                block_vars["hydro_u"][kICY:] += del_rho

        current_time += dt
        mesh.make_outputs(mesh_vars, current_time)

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

    if stop_reason == "integration stop condition":
        stop_reason = f"intg.stop reached; cycle={cycle}, nlim={nlim}, tlim={tlim:.14e}"
    print(
        f"Simulation loop finished: reason={stop_reason}, "
        f"cycle={cycle}, time={current_time:.14e} s "
        f"({current_time / SECONDS_PER_DAY:.6e} days)",
        flush=True,
    )

    return mesh_vars, current_time


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

    mesh, eos, thermo_y, thermo_x, kinet, device = create_models(args.config, args.output_dir)

    if args.restart_name:
        mesh_vars, current_time = mesh.initialize_from_restart(args.restart_name)
    else:
        mesh_vars, current_time = initialize_isothermal(mesh, config)

    forcing_states = [build_tidal_forcing_state(block, config, device) for block in mesh.blocks]
    print(
        "Forcing summary:",
        f"absorbed_surface_flux={forcing_states[0].absorbed_surface_flux:.3f} W/m^2,",
        f"gaussian_cooling_flux={forcing_states[0].mean_cooling_flux:.3f} W/m^2",
    )

    basename = Path(args.config).stem
    mesh_vars, current_time = run_simulation(
        mesh=mesh,
        eos=eos,
        thermo_y=thermo_y,
        thermo_x=thermo_x,
        kinet=kinet,
        mesh_vars=mesh_vars,
        current_time=current_time,
        forcing_states=forcing_states,
        config_file=args.config,
        output_dir=args.output_dir,
        basename=basename,
    )

    mesh.finalize(mesh_vars, current_time)


if __name__ == "__main__":
    main()
