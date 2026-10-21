#!/usr/bin/env python3
"""Moist Uranus GCM with scripted grey gas, cloud, and orbital forcing."""

from __future__ import annotations

import argparse
import math
import os
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
from rt_state import RTState

AU = 1.495978707e11
RT_KEY = "rt_heating"

def _save_script(module: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(module.eval()).save(str(path))

@dataclass(frozen=True)
class ThermoConfigs:
    eos: EquationOfState
    thermo_x: ThermoX
    thermo_y: ThermoY

def call_user_output(bvars: dict[str, torch.Tensor]):
    hydro_w = bvars["hydro_w"]
    out = {}
    out["qtol"] = hydro_w[kICY:].sum(dim=0)
    out["heating_rate"] = bvars[RT_KEY]
    return out

def make_problem_params(config: dict, species: list[str]) -> dict[str, float]:
    param = {
        "Ts": float(config["problem"]["Ts"]),
        "Ps": float(config["problem"]["Ps"]),
        "grav": -float(config["forcing"]["const-gravity"]["grav1"]),
        "Tmin": float(config["problem"]["Tmin"]),
    }
    for name in species:
        param[f"x{name}"] = float(config["problem"].get(f"x{name}", 0.0))
    return param


def make_orbit(config: dict[str, Any]) -> OrbitalForcing:
    return OrbitalForcing(
        config["stellar_luminosity"], config["semi_major_axis_au"] * AU,
        config["eccentricity"], math.radians(config["obliquity_deg"]), config["rotation_rate"],
        config["orbital_period"], math.radians(config.get("mean_anomaly_epoch_deg", 0.0)),
        math.radians(config.get("prime_meridian_epoch_deg", 0.0)),
    )


def make_modules(config: dict[str, Any], config_path: Path)) -> Path:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.script(module.eval()).save(str(path))

    orbit_cfg = config["orbit"]
    orbit_path = (run_folder / orbit_cfg["data"]).resolve()
    if force_build or not orbit_path.exists():
        _save_script(make_orbit(orbit_cfg), orbit_path)
    return orbit_path

def run_with(args: argparse.Namespace) -> None:
    # read all configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # set up device and output dir
    options = MeshOptions.from_yaml(args.config)
    device = torch.device(options.device_str())
    options.block().output_dir(args.output_dir)

    # set up mesh
    mesh = Mesh(options)
    mesh.to(device)

    # configure thermodynamics
    thermo_configs: list[ThermoConfigs] = []
    for block in mesh.blocks:
        thermo_y = block.module("hydro.eos.thermo")
        thermo_x = ThermoX(thermo_y.options)
        thermo_x.to(device)
        thermo_configs.append(
            ThermoConfigs(
                eos=block.module("hydro.eos"),
                thermo_y=thermo_y,
                thermo_x=thermo_x
            )
        )

    # configure kinetics
    op_kinet = KineticsOptions.from_yaml(args.config)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # configure orbital insolation
    orbit_path = make_stage_forcing(config, source_config_path, args.force_build, config_path)
    mesh.set_user_stage_forcings([str(orbit_path)])
    orbit_insolation = torch.compile(make_orbit(config["orbit"]).to(device).eval().insolation, fullgraph=True)

    # configure radiative transfer
    rt_states: list[RTState] = build_rt_states(mesh.blocks)
    for block in zip(mesh.blocks, thermo_configs):
        rt_states.append([RTState()])

    if args.restart:
        block_vars, current_time = mesh.initialize_from_restart(args.restart)
    else:
        species = thermo_configs[0].thermo_y.options.species()
        param: dict[str, float] = make_problem_params(config, species)

        block_vars = []
        for block in mesh.blocks:
            hydro_w = setup_profile(block, params, method="pseudo-adiabat")
            hydro_w[kIV1] += 1.0e-6 * torch.randn_like(hydro_w[kIV1])
            block_vars.append({"hydro_w": hydro_w})
        block_vars, current_time = mesh.initialize(block_vars)


    states = [build_rt_state(block, index, config, config_path, orbit_insolation, device) for index, block in enumerate(mesh.blocks)]

    # integrator and start cycle number
    intg = mesh.module("block0.intg")
    cycle = mesh.blocks[0].cycle()

    if not args.restart:
        mesh.make_outputs(block_vars, current_time)

    while not integrator.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)

        # calculate time step and print cycle
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)

        # update radiative heating flux if needed
        for bvars, rt in zip(block_vars, rt_states):
            rt.update_heating(bvars, time)

        # update dynamics
        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)

        # update kinetics
        for bvars, thermo in zip(block_vars, thermo_configs):
            del_rho = evolve_kinetics(
                bvars["hydro_w"],
                thermo.eos,
                thermo.thermo_x,
                thermo.thermo_y,
                kinet,
                dt,
            )
            bvars["hydro_u"][kICY:] += del_rho

        error = mesh.check_redo(block_vars)
        if error > 0:
            continue    # redo current step
        if error < 0:
            break       # terminate

        current_time += dt
        mesh.make_outputs(block_vars, current_time)

    mesh.finalize(block_vars, current_time)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run hydrodynamic simulation.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, 
        default="uranus.yaml",
        help="Input YAML configuration file."
    )
    parser.add_argument(
        "--output-dir", type=str, required=False, 
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "-r", "--restart", type=str, required=False,
        help="Restart from restart dump.",
        default="",
    )
    parser.add_argument(
        "-f", "--force-build", action="store_true")
    run_with(parser.parse_args())


if __name__ == "__main__":
    main()
