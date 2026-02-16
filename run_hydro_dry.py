import torch
import yaml
import argparse
import os
from snapy import MeshBlockOptions, MeshBlock, kICY, kIV1

def call_user_output(bvars: dict[str, torch.Tensor]):
    hydro_w = bvars["hydro_w"]
    out = {}
    out["qtol"] = hydro_w[kICY:].sum(dim=0)
    return out


def run_hydro_with(config_file:str, input_dir:str, output_dir:str,
                   verbose: bool):
    RESTART_FILE = f"{input_dir}/next.restart"
    TOPO_FILE = f"{input_dir}/topo.pt"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(config_file, verbose)
    op.output_dir(output_dir)
    block = MeshBlock(op)

    # use cuda if available
    if torch.cuda.is_available() and op.layout().backend() == "nccl":
        device = torch.device(block.device())
        print("device = ", device)
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

    block_vars = {}

    if os.path.exists(RESTART_FILE):
        module = torch.jit.load(RESTART_FILE)
        for name, data in module.named_buffers():
            block_vars[name] = data.to(device)
    else:
        Ts = float(config["problem"]["Ts"])
        Ps = float(config["problem"]["Ps"])
        param["Tmin"] = float(config["problem"]["Tmin"])

        block_vars["hydro_w"] = setup_profile(block, param, method="pseudo-adiabat")
        temp = Ts - grav * x1v / cp

        # add random vertical velocity
        block_vars["hydro_w"][kIV1] += 0.1 * torch.rand_like(block_vars["hydro_w"][kIV1])
        block_vars, current_time = block.initialize(block_vars)

    block.set_user_output_func(call_user_output)

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
        default='./input', help="input directory."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str,
        default='./output', help="output directory."
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
