"""Global moist Uranus simulation with orbital illumination and grey radiation."""
import argparse
import os
from pathlib import Path

import torch
import yaml
from kintera import Kinetics, KineticsOptions, ThermoOptions, ThermoX
from snapy import Mesh, MeshOptions, kICY, kIV1
from paddle import setup_profile, evolve_kinetics
from paddle.orbital import Uranus
from paddle.rt.opacity import GasOpacity, CloudOpacity
from paddle.rt.rt_state import RT_HEATING_KEY
from paddle.rt.rt_state_cubed_sphere import RTStateCubedSphere


PERTURBATION_SEED = 0
VERTICAL_VELOCITY_PERTURBATION = 1.0e-2


def prepare_config(infile: Path, output_dir: Path, device: torch.device) -> Path:
    """Write each rank's opacity scripts on its own device, without shared writes."""
    with infile.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rank = int(os.environ.get("RANK", "0"))
    artifact_dir = output_dir / "models" / f"rank{rank}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    thermo = ThermoX(ThermoOptions.from_yaml(str(infile)))
    species = thermo.options.species()
    for name, opacity in config["opacities"].items():
        indices = [species.index(sp) for sp in opacity["species"]]
        weights = [float(thermo.mu[i]) for i in indices]
        parameters = dict(opacity["parameters"])
        kind = parameters.pop("kind")
        if kind == "gas":
            model = GasOpacity(indices, weights, **parameters)
        elif kind == "cloud":
            model = CloudOpacity(indices, weights, **parameters)
        else:
            raise ValueError(f"Unknown opacity kind: {kind}")
        path = artifact_dir / f"{name}.pt"
        torch.jit.script(model.to(device=device, dtype=torch.float64).eval()).save(str(path))
        opacity["data"] = [str(path)]
    config_path = artifact_dir / infile.name
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return config_path


def make_user_output(state: RTStateCubedSphere):
    def call_user_output(bvars: dict[str, torch.Tensor]):
        hydro_w = bvars["hydro_w"]
        il, iu = state.coord.il(), state.coord.iu()
        shape = (*hydro_w.shape[1:3], iu - il + 2)
        net_flux = torch.zeros_like(state.coord.face_area1())
        net_flux[..., il:iu+2] = state.net_flux.reshape(shape)
        return {
            "qtol": hydro_w[kICY:].sum(dim=0),
            "rt_heating": bvars[RT_HEATING_KEY],
            "net_flux": net_flux,
        }
    return call_user_output


@torch.no_grad()
def run_with(args):
    # load yaml configure
    infile = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    with infile.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    # create output directory and opacity artifacts
    options = MeshOptions.from_yaml(str(infile))
    device = torch.device(options.device_str())
    config_path = prepare_config(infile, output_dir, device)
    options.block().output_dir(str(output_dir))
    # construct mesh
    mesh = Mesh(options)
    mesh.to(device=device, dtype=torch.float64)
    blocks = list(mesh.blocks)

    # register radiative heating at each rk stage
    orbit = Uranus(rt_heating_key=RT_HEATING_KEY)
    forcing_path = config_path.parent / "uranus_forcing.pt"
    torch.jit.script(orbit).save(str(forcing_path))
    mesh.set_user_stage_forcings([str(forcing_path)])

    # create radiation and thermodynamics modules for each block
    states = [RTStateCubedSphere([orbit], block, config_path) for block in blocks]

    if args.restart:
        block_vars, current_time = mesh.initialize_from_restart(str(Path(args.restart).resolve()))
    else:
        species = states[0].thermo_y.options.species()
        problem = config["problem"]
        param = {"Ts": float(problem["Ts"]), "Ps": float(problem["Ps"]),
                 "Tmin": float(problem["Tmin"]),
                 "grav": -float(config["forcing"]["const-gravity"]["grav1"])}
        for name in species:
            param[f"x{name}"] = float(problem.get(f"x{name}", 0.0))
        # set initial profile with perturbation
        block_vars = []
        for block in blocks:
            hydro_w = setup_profile(block, param, method="pseudo-adiabat")
            layout = block.get_layout()
            rank = layout.options.rank()
            _, _, face_id = layout.loc_of(rank)
            generator = torch.Generator(device=hydro_w.device).manual_seed(
                PERTURBATION_SEED + face_id + rank * 6
            )
            hydro_w[kIV1].uniform_(
                -VERTICAL_VELOCITY_PERTURBATION,
                VERTICAL_VELOCITY_PERTURBATION,
                generator=generator,
            )
            block_vars.append({"hydro_w": hydro_w})
        block_vars, current_time = mesh.initialize(block_vars)

    # initialize radiative heating and user output
    for block, state, bvars in zip(blocks, states, block_vars):
        state.update_heating(bvars, current_time)
        block.set_user_output_func(make_user_output(state))

    # create kinetics modules
    kinet = Kinetics(KineticsOptions.from_yaml(str(config_path)))
    kinet.to(device=device, dtype=torch.float64)

    # initialize cycle counts
    intg = mesh.module("block0.intg")
    cycle = blocks[0].cycle()
    end_time = float(config["integration"]["tlim"])
    if not args.restart:
        mesh.make_outputs(block_vars, current_time)

    mesh.set_cycle(cycle)

    while not intg.stop(cycle, current_time):
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)

        # run rk-stages
        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)

        # compute post-hydro primitives before applying moist chemistry.
        for bvars, state in zip(block_vars, states):
            bvars["hydro_w"] = state.eos.compute("U->W", (bvars["hydro_u"],))
            del_rho = evolve_kinetics(bvars["hydro_w"], state.eos,
                                     state.thermo_x, state.thermo_y, kinet, dt)
            bvars["hydro_u"][kICY:] += del_rho

        # check_redo will do a final hydro_u -> hydro_w
        err = mesh.check_redo(block_vars)
        if err > 0: # redo current step without advancing cycle
            continue
        if err < 0: # terminate
            break

        # all pass, advance a cycle
        cycle += 1
        mesh.set_cycle(cycle)
        current_time += dt
        for bvars, state in zip(block_vars, states):
            state.update_heating(bvars, current_time)
        mesh.make_outputs(block_vars, current_time)

    mesh.finalize(block_vars, current_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=str(Path(__file__).with_name("uranus.yaml")))
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("-r", "--restart", default="")
    run_with(parser.parse_args())
