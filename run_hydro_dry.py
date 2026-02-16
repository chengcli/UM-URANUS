import os
import yaml
import argparse
import torch
import kintera
from snapy import (
        MeshBlockOptions,
        MeshBlock,
        load_restart,
        kIDN, kIPR, kIV1
        )

def run_hydro_with(config_file:str, input_dir:str, output_dir:str,
                   verbose: bool):
    RESTART_FILE = f"{input_dir}/next.restart"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(config_file, verbose)
    op.output_dir(output_dir)
    block = MeshBlock(op)

    # use cuda if available
    if torch.cuda.is_available() and op.layout().backend() == "nccl":
        device = torch.device(block.device())
    else:
        device = torch.device("cpu")

    block.to(device)

    # get handles to modules
    coord = block.module("coord")
    eos = block.module("hydro.eos")
    grav = -block.options.hydro().grav().grav1()

    # thermodynamics
    Rd = kintera.constants.Rgas / eos.options.weight()
    cv = eos.species_cv_ref()
    cp = cv + Rd

    # setup a meshgrid for simulation
    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    # dimensions
    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]

    # get surface height
    z_surf = coord.buffer("x1f")[block.options.coord().nghost()]

    # set hydro dynamic variables
    block_vars = {}

    if os.path.exists(RESTART_FILE):
        block_vars = load_restart(RESTART_FILE)
        for name, data in block_vars.items():
            block_vars[name] = data.to(device)
    else:
        Ts = float(config["problem"]["Ts"])
        Ps = float(config["problem"]["Ps"])

        # set up an isothermal atmosphere
        w = torch.zeros((eos.nvar(), nc3, nc2, nc1), device=device)
        w[kIPR] = Ps * torch.exp(- grav * (x1v - z_surf) / (Rd * Ts))
        w[kIDN] = w[kIPR] / (Rd * Ts)

        # add random vertical velocity
        w[kIV1] += 0.1 * torch.rand_like(w[kIV1])

        block_vars["hydro_w"] = w
        block_vars, current_time = block.initialize(block_vars)

    # integration
    block.make_outputs(block_vars, current_time)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)

        err = block.check_redo(block_vars)
        if err > 0:
            continue  # redo current step
        if err < 0:
            break  # terminate

        current_time += dt
        block.make_outputs(block_vars, current_time)

    block.finalize(block_vars, current_time)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run hydrodynamic simulation.")
    parser.add_argument(
        "-c", "--config", type=str, 
        required=True, help="YAML configuration file."
    )
    parser.add_argument(
        "-i", "--input_dir", type=str,
        default='.', help="input directory."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str,
        default='.', help="output directory."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="print verbose debug information"
        )

    args = parser.parse_args()

    run_hydro_with(args.config, args.input_dir, args.output_dir,
                   args.verbose)

if __name__ == "__main__":
    main()
