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
import bisect

import torch
import yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIV1, kICY, kIDN, kIPR
# from paddle import evolve_kinetics
import time

SECONDS_PER_DAY = 86400.0
NA=6.022e23
mu = 2.016*0.001/NA
R = 4124.2 #H2

@dataclass
@dataclass
class ForcingState:
    fluxmean: float
    fluxstd: float
    umumean: float
    umustd: float
    sponge_tau: float
    spongeheight: float
    tempmean: float
    tempstd: float
    heatthr: float
    heatsf: float
    model: torch.jit.ScriptModule
    syear: float
    top_depth: int
    bottom_depth: int

    NA: float
    mu: float
    R: float
    mask: torch.Tensor
    basepress: torch.Tensor
    lat: torch.Tensor

import torch

def regrid_tensor(x, y, xq, tempmean, tempstd):
    """
    Vectorized regrid + normalization.

    x  : (B, Nz) decreasing pressure grid
    y  : (B, Nz) values at x
    xq : (Nq,) target grid

    returns: (B, 256, 1)
    """

    B, Nz = x.shape
    Nq = xq.shape[0]

    device = x.device

    # flip because searchsorted expects increasing
    x_flip = torch.flip(x, dims=[1])
    y_flip = torch.flip(y, dims=[1])

    # broadcast xq across batch
    xq_expand = xq.unsqueeze(0).expand(B, Nq)

    # find interpolation indices
    idx = torch.searchsorted(x_flip, xq_expand)

    idx = torch.clamp(idx, 1, Nz - 1)

    x0 = torch.gather(x_flip, 1, idx - 1)
    x1 = torch.gather(x_flip, 1, idx)

    y0 = torch.gather(y_flip, 1, idx - 1)
    y1 = torch.gather(y_flip, 1, idx)

    # linear interpolation
    t = (xq_expand - x0) / (x1 - x0 + 1e-12)
    interp = y0 + t * (y1 - y0)

    # clamp edges (same logic as original)
    interp = torch.where(xq_expand >= x[:, :1], y[:, :1], interp)
    interp = torch.where(xq_expand <= x[:, -1:], y[:, -1:], interp)

    # pad to 256
    result = torch.full((B, 256), -9999.0, device=device)

    result[:, :Nq] = interp

    # normalize
    result = (result - tempmean) / tempstd

    return result.unsqueeze(-1)
def torch_denormalize_symlog(x: torch.Tensor, thr: float, sf: float) -> torch.Tensor:
    """
    Denormalize a tensor from a symmetric log scale.

    Args:
        x (torch.Tensor): input tensor (normalized)
        thr (float): linear threshold
        sf (float): scaling factor

    Returns:
        torch.Tensor: denormalized tensor
    """
    unscaled = x * sf
    abs_unscaled = torch.abs(unscaled)

    # masks
    linear_mask = abs_unscaled <= 1.0
    log_mask = ~linear_mask

    y = torch.zeros_like(x)

    # linear region
    y[linear_mask] = unscaled[linear_mask] * thr

    # logarithmic region
    if log_mask.any():
        log_values = torch.sign(unscaled[log_mask]) * thr * 10**(abs_unscaled[log_mask] - 1.0)
        y[log_mask] = log_values

    return y
def degrid(x, normy, xq, heatthr, heatsf):
    """
    Vectorized inverse interpolation (NN grid -> model grid),
    supports padding/masked region beyond actual data.

    Args:
        x      : (Ntotal,) full target grid (monotonic, padded)
        normy  : (B, Ndata, 1) normalized NN outputs (only first Ndata elements contain real data)
        xq     : (B, Nz) query points
        heatthr: scalar
        heatsf : scalar

    Returns:
        interp : (B, Nz) interpolated values
    """
    B, Ndata, _ = normy.shape  # 100
    Nz = xq.shape[1]
    y = torch_denormalize_symlog(normy, heatthr, heatsf).squeeze(-1)  # (B, 100)

    # Take only actual data from x
    x_data = x[:Ndata]        # (100,)
    x_flip = torch.flip(x_data, dims=[0])
    y_flip = torch.flip(y, dims=[1])

    idx = torch.searchsorted(x_flip, xq)  # (B, Nz)

    # Clamp idx to [1, Ndata-1] to avoid out-of-bounds gather
    idx = torch.clamp(idx, 1,99)

    # Expand for batch
    x_exp = x_flip.unsqueeze(0).repeat(B, 1)  # (B, Ndata)
    y_exp = y_flip                             # (B, Ndata)

    # Gather lower and upper values
    x0 = torch.gather(x_exp, 1, idx - 1)
    x1 = torch.gather(x_exp, 1, idx)
    y0 = torch.gather(y_exp, 1, idx - 1)
    y1 = torch.gather(y_exp, 1, idx)

    # Linear interpolation
    t = (xq - x0) / (x1 - x0 + 1e-12)
    interp = y0 + t * (y1 - y0)

    # Clamp values outside real data to first/last real points
    Ndata = x_data.shape[0]  # total data points
    real_data_len = 100       # number of real data points

    # Use only the real data for clamping
    interp = torch.where(xq >= x_data[0], y[:,0:1], interp)
    interp = torch.where(xq <= x_data[-1], y[:,-1:], interp)
    return interp

def normalize_standard(x, mean, std):
    return (x - mean) / std
def calcglobal(lat, time, fluxmean, fluxstd, umumean, umustd, syear):

    pi = torch.pi

    Ls = (time % (2 * syear)) * 2 * pi / syear
    obliquity = torch.tensor(97.77 * pi / 180, device=lat.device)

    sindelta = torch.sin(obliquity) * torch.sin(torch.tensor(Ls, device=lat.device))
    cosdelta = torch.sqrt(1 - sindelta**2)

    sinlat = torch.sin(lat)
    coslat = torch.cos(lat)

    arg = - sinlat * sindelta / (coslat * cosdelta + 1e-12)
    arg = torch.clamp(arg, -1.0, 1.0)

    hsunset = torch.acos(arg)

    avgcostheta = (
        sinlat * sindelta * hsunset +
        coslat * cosdelta * torch.sin(hsunset)
    ) / pi

    flux = normalize_standard(hsunset / pi, fluxmean, fluxstd)
    umu  = normalize_standard(avgcostheta, umumean, umustd)

    return torch.stack([flux, umu], dim=-1)
# def physicalmodel(pressvec,tempvec,basepress,lat,time,fluxmean,fluxstd,umumean,umustd,tempmean,tempstd,heatthr,heatsf,model,syear):
#     regridtemp = regrid(pressvec, tempvec, basepress,tempmean,tempstd)  # normalized and regridded
#     mask = torch.ones((1, 256), dtype=torch.bool, device=regridtemp.device)
#     mask[0, 0:100] = False  # equivalent to C++ index_put

#     # --- compute global features ---
#     global_features = calcglobal(lat, time,fluxmean,fluxstd,umumean,umustd,syear)  # normalized global features

#     # --- forward pass ---
#     output = model.forward(regridtemp, global_features, mask)
#     # output should be a tensor
#     output = output.to(regridtemp.dtype)  # ensure same dtype

#     # --- de-normalize / de-interpolate ---
#     heating = degrid(basepress, output, pressvec,heatthr,heatsf)  # heating[i] corresponds to i-pmb.is
#     return heating
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


    eos = block.module("hydro.eos")
    return block, eos, device
def setup_profile(
    block: snapy.MeshBlock, param: dict[str, float] = {}, method: str = "moist-adiabat"
) -> torch.Tensor:
    """
    Set up an adiabatic initial condition for the mesh block.

    This function initializes the primitive variables in the mesh block
    and returns the initialized tensor.

    Args:
        block (snapy.MeshBlock): The mesh block to set up.
        param (dict[str, float], optional): Parameters for the adiabat setup. Defaults to {}.
        method (str, optional): Method for the adiabat setup. Choose between
            (1) "dry-adiabat"
            (2) "moist-adiabat"
            (3) "isothermal"
            (4) "pseudo-adiabat"
            (5) "neutral"
            Defaults to "moist-adiabat".

        Required parameters in `param`:
            Ts (float): Surface temperature in Kelvin. Default is 300 K.
            Ps (float): Surface pressure in Pascals. Default is 1e5 Pa.
            x<species> (float): Mole fraction of a specific species (e.g., xH2O for
            water vapor). Default is 0.0.
            grav (float): Gravitational acceleration in m/s^2. Default is 9.8 m/s^2.

    Returns:
        torch.Tensor: The initialized primitive variables tensor.
    """

    # check method
    valid_methods = [
        "dry-adiabat",
        "moist-adiabat",
        "isothermal",
        "pseudo-adiabat",
        "neutral",
    ]

    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")

    Ts = param.get("Ts", 70.0)
    Ps = param.get("Ps", 5.0e5)
    grav = param.get("grav", 9.19)
    Tmin = param.get("Tmin", 0.0)

    # get handles to modules
    coord = block.module("coord")

    # get coordinates
    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    # handling mole fractions


    # get dimensions
    nc3, nc2, nc1 = x1v.shape
    # ny = len(thermo_y.options.species()) - 1
    nvar = 5 

    w = torch.zeros((nvar, nc3, nc2, nc1), dtype=x1v.dtype, device=x1v.device)

    temp = Ts * torch.ones((nc3, nc2), dtype=w.dtype, device=w.device)
    pres = Ps * torch.ones((nc3, nc2), dtype=w.dtype, device=w.device)
    # xfrac = torch.zeros((nc3, nc2, ny + 1), dtype=w.dtype, device=w.device)

 

    # start and end indices for the vertical direction
    # excluding ghost cells
    ifirst =0
    ilast = len(coord.buffer("dx1f"))
    print(ilast)

    # vertical grid distance of the first cell
    dz = coord.buffer("dx1f")[ifirst]

    # half a grid to cell center
    # thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz / 2.0)




    # isothermal extrapolation
    for i in range(ifirst, ilast):

        # mu = (thermo_x.mu * xfrac).sum(-1)
        dz = coord.buffer("dx1f")[i]
        pres *= torch.exp(-grav * mu * dz / (R * temp))
        # conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
        w[kIPR, ..., i] = pres
        w[kIDN, ..., i] = pres/(R*temp)
        # w[snapy.index.icy :, ..., i] = thermo_x.compute("X->Y", [xfrac])
    return w

def initialize_isothermal(block: MeshBlock, config: dict) -> tuple[dict[str, torch.Tensor], float]:
    grav = -float(config["forcing"]["const-gravity"]["grav1"])
    problem = config["problem"]

    param = {
        "Ts": float(problem["Ts"]),
        "Ps": float(problem["Ps"]),
        "Tmin": float(problem.get("Tmin", problem["Ts"])),
        "grav": grav,
    }



    hydro_w = setup_profile(block, param)

    # add random noise to IV1
    hydro_w[kIV1] += 1e-6 * torch.randn_like(hydro_w[kIV1])

    return block.initialize({"hydro_w": hydro_w})


def _resolve_local_face_name(block: MeshBlock) -> str:
    layout = snapy.distributed.get_layout()
    rank = int(snapy.distributed.get_rank())
    loc = layout.loc_of(rank)
    face_id = int(loc[2])
    return snapy.coord.get_cs_face_name(face_id)

#created once
def build_tidal_forcing_state(block: MeshBlock, config: dict, device: torch.device) -> ForcingState:
    coord = block.module("coord")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")

    beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
    face_name = _resolve_local_face_name(block)
    lon, lat = snapy.coord.cs_ab_to_lonlat(face_name, alpha, beta)

    problem = config["problem"]
    # lon0 = math.radians(float(problem.get("substellar_lon_deg", 0.0)))
    # lat0 = math.radians(float(problem.get("substellar_lat_deg", 0.0)))
    modelfile = problem["modelfile"]

    modulea = torch.jit.load(modelfile).to(device)

    # Set the model to evaluation mode (important for inference)
    modulea.eval()


    # cos_zenith = (
    #     torch.sin(lat) * math.sin(lat0)
    #     + torch.cos(lat) * math.cos(lat0) * torch.cos(lon - lon0)
    # )
    # cos_zenith_dayside = torch.clamp(cos_zenith, min=0.0).to(device)

    fluxmeana = float(problem["fluxmean"])
    fluxstda = float(problem["fluxstd"])
    umumeana = float(problem["umumean"])
    umustda = float(problem["umustd"])
    sponge_taua = float(problem["sponge_tau"])
    spongeheighta = float(problem["spongeheight"])
    tempmeana = float(problem["tempmean"])
    tempstda = float(problem["tempstd"])
    heatthra = float(problem["heatthr"])
    heatsfa = float(problem["heatsf"])
    syeara = float(problem["syear"])
    # frac_to_surface = float(problem["stellar_surface_fraction"])
    # absorbed_surface_flux = stellar_flux * frac_to_surface

    # Mean of max(cos(zenith), 0) over the sphere is 1/4, so this cooling flux
    # exactly balances globally integrated dayside heating for a spherical planet.
    # mean_cooling_flux = absorbed_surface_flux * 0.25
    NA =6.022e23
    nx2 = x2v.shape[0]
    nx3 = x3v.shape[0]
    nz = coord.buffer("x1v").shape[0]
    batch = nx2*nx3
    Nmax = 256
    mask = torch.ones((batch, Nmax), dtype=torch.bool, device=beta.device)
    Nvalid = 100
# 2. Mark valid elements as False
    mask[:, :Nvalid] = False
    return ForcingState(
        fluxmean=fluxmeana,
        fluxstd=fluxstda,
        umumean=umumeana,
        umustd=umustda,
        sponge_tau=sponge_taua, #taken from hot jupiter case, must always be greater than dt!
        spongeheight=spongeheighta,
        tempmean=tempmeana,
        tempstd=tempstda,
        heatthr=heatthra,
        heatsf=heatsfa,
        model = modulea,
        syear = syeara,
        top_depth=int(problem.get("forcing_depth_top", 1)),
        bottom_depth=int(problem.get("forcing_depth_bottom", 1)),
        NA=6.022e23,
        mu = 2.016*0.001/NA,
        R = 4124.2,
        mask = mask,
        basepress = torch.tensor([ 474464.0, 444378.0, 416200.0, 389808.0, 365090.0, 341939.0, 320256.0, 299948.0, 280928.0, 263114.0, 246430.0, 230804.0, 216168.0, 202461.0, 189622.0, 177598.0, 166337.0, 155789.0, 145910.0, 136658.0, 127992.0, 119876.0, 112275.0, 105155.0, 98487.2, 92242.1, 86392.9, 80914.6, 75783.7, 70978.2, 66477.4, 62262.0, 58313.9, 54616.2, 51152.9, 47909.2, 44871.3, 42025.9, 39361.0, 36865.1, 34527.4, 32338.0, 30287.4, 28366.9, 26568.1, 24883.4, 23305.5, 21827.7, 20443.6, 19147.2, 17933.1, 16795.9, 15730.9, 14733.4, 13799.1, 12924.1, 12104.5, 11337.0, 10618.1, 9944.8, 9314.2, 8723.6, 8170.4, 7652.3, 7167.1, 6712.6, 6286.9, 5888.3, 5514.9, 5165.2, 4837.7, 4530.9, 4243.6, 3974.5, 3722.5, 3486.4, 3265.3, 3058.3, 2864.4, 2682.7, 2512.6, 2353.3, 2204.1, 2064.3, 1933.4, 1810.8, 1696.0, 1588.4, 1487.7, 1393.4, 1305.0, 1222.3, 1144.8, 1072.2, 1004.2, 940.5, 880.9, 825.0, 772.7, 723.7 ],device=beta.device),
        lat=lat,

    )


#apply per timestep:
def apply_tidal_forcing(block: MeshBlock, block_vars: dict[str, torch.Tensor], forcing: ForcingState, dt: float,current_time:float,oldforcing) -> float:
    with torch.inference_mode():
        coord = block.module("coord")
        il, iu = coord.il(), coord.iu()
        dzf = coord.buffer("dx1f")
        model = forcing.model
        hydro_u = block_vars["hydro_u"]
        x2v = coord.buffer("x2v")
        x3v = coord.buffer("x3v")

        hydro_u[kIPR] += oldforcing*dt

def compute_radiative_heating(block: MeshBlock, block_vars: dict[str, torch.Tensor], forcing: ForcingState, dt: float,current_time:float) -> float:
    with torch.inference_mode():
        coord = block.module("coord")
        il, iu = coord.il(), coord.iu()
        dzf = coord.buffer("dx1f")
        model = forcing.model
        hydro_u = block_vars["hydro_u"]
        x2v = coord.buffer("x2v")
        x3v = coord.buffer("x3v")
        # --- get lat, lon (replace pexo3->GetLatLon) ---
        lat = forcing.lat
        basepress = forcing.basepress
        # --- create mask ---
        fluxmean =forcing.fluxmean
        fluxstd = forcing.fluxstd
        umumean = forcing.umumean
        umustd=forcing.umustd 
        tempmean=forcing.tempmean
        tempstd=forcing.tempstd
        heatthr=forcing.heatthr
        heatsf=forcing.heatsf
        model = forcing.model
        syear= forcing.syear
        #change pressure according to heating, hydro_u is 4D array
        nx2 = x2v.shape[0]
        nx3 = x3v.shape[0]
        nz = coord.buffer("x1v").shape[0]
        batch = nx2*nx3

                #pressure hydro_u[kIPR, i,j, :]
                #temp PV = nRT
                #temp P*mu/(rho*R)
        # print("NN Start")
        globals = calcglobal(lat , current_time , fluxmean , fluxstd , umumean , umustd , syear )
        globals = globals.reshape(batch,2)
        Nvalid = 100       # actual number of data points
        Nmax = 256         # maximum input length the NN expects

        # Create the mask like in C++: True = ignore/pad, False = valid
    # 1. Initialize mask: True = padding

        pressbatch = hydro_u[kIPR].view(batch,nz)
        rhobatch = hydro_u[kIDN].view(batch,nz)
        tempbatch = pressbatch/(rhobatch*R)
        regridtemp = regrid_tensor(pressbatch,tempbatch,basepress,tempmean,tempstd)
        regridtemp_padded = regridtemp.to(torch.float32)
        # pad = torch.full((batch, Nmax-Nvalid, regridtemp.shape[2]),
        #                 -9999.0, device=regridtemp.device, dtype=regridtemp.dtype)
        # regridtemp_padded = torch.cat([regridtemp, pad], dim=1)
        globals = globals.to(torch.float32)

        # t0 = time.time()
        output = model(regridtemp_padded, globals, forcing.mask)

        oldforcing = (R*hydro_u[kIDN])*(degrid(basepress, output, pressbatch,heatthr,heatsf).reshape(nx2,nx3,nz))

        #hydro_u[kIPR, ..., iu + 1 - top_depth : iu + 1] -= cool_src

        # icool = int(il * 0.4 + iu * 0.6)
        # hydro_u[kIPR, ..., icool] -= cool_src
    return oldforcing
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
    nextforcing = 0
    oldforcing=0
    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        recompute_forcing = current_time >= nextforcing
        if recompute_forcing:
            oldforcing = compute_radiative_heating(block,block_vars,forcing,dt,current_time)
            nextforcing=current_time+1000*dt
        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            apply_tidal_forcing(block, block_vars, forcing, dt,current_time,oldforcing)

        err = block.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        # del_rho = evolve_kinetics(block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt)
        # block_vars["hydro_u"][kICY:] += del_rho

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

    block, eos, device = create_models(args.config, args.output_dir)

    if args.restart_name:
        block_vars, current_time = block.initialize_from_restart(args.restart_name)
    else:
        block_vars, current_time = initialize_isothermal(block, config)

    for key, data in block_vars.items():
        if isinstance(data, torch.Tensor):
            print(f"{key}: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}")

    forcing = build_tidal_forcing_state(block, config, device) #run once


    tlim = float(config["integration"]["tlim"])
    basename = Path(args.config).stem
    block_vars, current_time = run_simulation(
        block=block,
        eos=eos,
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
