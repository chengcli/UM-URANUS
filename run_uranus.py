#!/usr/bin/env python3
"""Moist Uranus GCM with scripted grey gas, cloud, and orbital forcing."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import yaml

import pyharp
import kintera
from kintera import Kinetics, KineticsOptions, ThermoOptions, ThermoX
from paddle import evolve_kinetics, setup_profile
import snapy
from snapy import Mesh, MeshOptions, kICY, kIPR, kIV1

from opacity import CloudOpacity, GasOpacity
from orbital import OrbitalForcing
from rt_forcing import build_rt_state, compute_heating, mask_nightside_visible_flux

AU = 1.495978707e11

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    include = config.pop("include", None)
    if include is None:
        return config
    base = load_config(path.parent / include)

    def merge(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                target[key] = merge(target[key], value)
            else:
                target[key] = value
        return target

    return merge(base, config)


def _save_script(module: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(module.eval()).save(str(path))


def ensure_torchscripts(
    config: dict[str, Any], config_path: Path, force_build: bool = False, thermo_config_path: Path | None = None
) -> Path:
    ThermoOptions.from_yaml(str(thermo_config_path or config_path))
    weights = dict(zip(kintera.species_names(), kintera.species_weights(), strict=True))
    species_indices = {item["name"]: index for index, item in enumerate(config["species"])}
    run_folder = Path(__file__).resolve().parent
    for opacity in config["opacities"].values():
        params = opacity["parameters"]
        path = (run_folder / opacity["data"][0]).resolve()
        if path.exists() and not force_build:
            continue
        species = opacity["species"]
        if params["kind"] == "gas":
            module: torch.nn.Module = GasOpacity(
                [species_indices[item] for item in species], [weights[item] for item in species],
                params["kappa_ref"], params["p_ref"], params["exponent"]
            )
        else:
            module = CloudOpacity(
                species_indices[species[0]], weights[species[0]], params["mass_extinction"],
                params["single_scattering_albedo"], params["asymmetry"]
            )
        _save_script(module, path)

    orbit_cfg = config["orbit"]
    orbit_path = (run_folder / orbit_cfg["data"]).resolve()
    if force_build or not orbit_path.exists():
        module = OrbitalForcing(
            orbit_cfg["stellar_luminosity"], orbit_cfg["semi_major_axis_au"] * AU,
            orbit_cfg["eccentricity"], math.radians(orbit_cfg["obliquity_deg"]), orbit_cfg["rotation_rate"],
            orbit_cfg["orbital_period"], math.radians(orbit_cfg.get("mean_anomaly_epoch_deg", 0.0)),
            math.radians(orbit_cfg.get("prime_meridian_epoch_deg", 0.0)),
        )
        _save_script(module, orbit_path)
    return orbit_path


def sync_primitives(variables: dict[str, torch.Tensor], eos: Any) -> None:
    variables["hydro_w"] = eos.compute("U->W", (variables["hydro_u"], variables["hydro_w"]))


def run(args: argparse.Namespace) -> None:
    torch._C._jit_set_texpr_fuser_enabled(False)
    source_config_path = Path(args.config).resolve()
    config = load_config(source_config_path)
    pyharp.add_resource_directory(str(Path(__file__).resolve().parent), prepend=True)
    config_path = source_config_path
    if "include" in yaml.safe_load(source_config_path.read_text()):
        resolved_dir = Path(args.output_dir).resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        config_path = resolved_dir / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    orbit_path = ensure_torchscripts(config, source_config_path, args.force_build, config_path)
    options = MeshOptions.from_yaml(str(config_path))
    options.block().output_dir(args.output_dir)
    mesh = Mesh(options)
    device = torch.device(options.device_str())
    mesh.to(device)
    mesh.set_user_stage_forcings([str(orbit_path)])
    thermos = []
    for block in mesh.blocks:
        thermo_y = block.module("hydro.eos.thermo")
        thermo_x = ThermoX(thermo_y.options)
        thermo_x.to(device)
        thermos.append((block.module("hydro.eos"), thermo_y, thermo_x))
    kinetics = Kinetics(KineticsOptions.from_yaml(str(config_path)))
    kinetics.to(device)
    if args.restart:
        block_vars, current_time = mesh.initialize_from_restart(args.restart)
    else:
        params = {"Ts": float(config["problem"]["Ts"]), "Ps": float(config["problem"]["Ps"]),
                  "Tmin": float(config["problem"]["Tmin"]), "grav": -float(config["forcing"]["const-gravity"]["grav1"])}
        for species in thermos[0][1].options.species():
            params[f"x{species}"] = float(config["problem"].get(f"x{species}", 0.0))
        initial = []
        for block in mesh.blocks:
            hydro_w = setup_profile(block, params, method="pseudo-adiabat")
            hydro_w[kIV1] += 1.0e-6 * torch.randn_like(hydro_w[kIV1])
            initial.append({"hydro_w": hydro_w})
        block_vars, current_time = mesh.initialize(initial)
    states = [build_rt_state(block, index, config, config_path, orbit_path, device) for index, block in enumerate(mesh.blocks)]
    for variables in block_vars:
        variables["rt_heating"] = torch.zeros_like(variables["hydro_u"][kIPR])
    integrator = mesh.module("block0.intg")
    cycle = 0
    if not args.restart:
        mesh.make_outputs(block_vars, current_time)
    while not integrator.stop(cycle, current_time):
        if args.max_cycles is not None and cycle >= args.max_cycles:
            break
        cycle += 1
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)
        for variables, (eos, _, _) in zip(block_vars, thermos):
            sync_primitives(variables, eos)
        for stage in range(len(integrator.stages)):
            for variables, thermo, state in zip(block_vars, thermos, states):
                if current_time + 1.0e-12 >= state.next_update:
                    state.last_heating = compute_heating(variables, *thermo, state, current_time)
                    variables["rt_heating"].zero_()
                    variables["rt_heating"][..., state.il:state.iu + 1] = state.last_heating
                    state.next_update = current_time + state.update_dt
            mesh.forward(block_vars, dt, stage)
        error = mesh.check_redo(block_vars)
        if error > 0:
            continue
        if error < 0:
            break
        for variables, (eos, thermo_y, thermo_x) in zip(block_vars, thermos):
            sync_primitives(variables, eos)
            variables["hydro_u"][kICY:] += evolve_kinetics(
                variables["hydro_w"], eos, thermo_x, thermo_y, kinetics, dt
            )
            sync_primitives(variables, eos)
        current_time += dt
        mesh.make_outputs(block_vars, current_time)
    mesh.finalize(block_vars, current_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="uranus.yaml")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("-r", "--restart", default="")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--max-cycles", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
