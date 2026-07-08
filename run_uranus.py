#!/usr/bin/env python3
"""Run the Uranus GCM with mesh-level snapy entry points and EOS-driven ICs."""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
# import numpy
import torch
import yaml
import snapy
from snapy import Mesh, MeshOptions, kConserved, kIDN, kIPR, kIV1

SECONDS_PER_DAY = 86400.0


@dataclass
class ForcingState:
    fluxmean: float
    fluxstd: float
    umumean: float
    umustd: float
    tempmean: float
    tempstd: float
    heatthr: float
    heatsf: float
    model: torch.jit.ScriptModule
    syear: float
    top_depth: int
    bottom_depth: int
    gas_constant: float
    mask: torch.Tensor
    basepress: torch.Tensor
    lon: torch.Tensor
    lat: torch.Tensor
    sinlat: torch.Tensor
    coslat: torch.Tensor
    stellar_flux_nadir: float
    substellar_lon: float
    substellar_lat: float
    rotation_rate: float
    rt_update_cadence: float
    tidal_heating_decay_tau: float


@dataclass
class BlockDiagnostics:
    solar_zenith_angle: torch.Tensor
    solar_forcing: torch.Tensor
    heating_tendency: torch.Tensor


def regrid_tensor(x: torch.Tensor, y: torch.Tensor, xq: torch.Tensor, tempmean: float, tempstd: float) -> torch.Tensor:
    """Interpolate model data onto the NN pressure grid and normalize it."""
    batch_size, nz = x.shape
    nq = xq.shape[0]

    x_flip = torch.flip(x, dims=[1])
    y_flip = torch.flip(y, dims=[1])
    xq_expand = xq.unsqueeze(0).expand(batch_size, nq).contiguous()

    idx = torch.searchsorted(x_flip, xq_expand)
    idx = torch.clamp(idx, 1, nz - 1)

    x0 = torch.gather(x_flip, 1, idx - 1)
    x1 = torch.gather(x_flip, 1, idx)
    y0 = torch.gather(y_flip, 1, idx - 1)
    y1 = torch.gather(y_flip, 1, idx)

    t = (xq_expand - x0) / (x1 - x0 + 1e-12)
    interp = y0 + t * (y1 - y0)

    interp = torch.where(xq_expand >= x[:, :1], y[:, :1], interp)
    interp = torch.where(xq_expand <= x[:, -1:], y[:, -1:], interp)

    result = torch.full((batch_size, 256), -9999.0, dtype=y.dtype, device=y.device)
    result[:, :nq] = interp
    result = (result - tempmean) / tempstd
    return result.unsqueeze(-1)


def torch_denormalize_symlog(x: torch.Tensor, thr: float, sf: float) -> torch.Tensor:
    """Undo the symmetric-log normalization used by the heating model."""
    unscaled = x * sf
    abs_unscaled = torch.abs(unscaled)

    linear = unscaled * thr
    log = torch.sign(unscaled) * thr * torch.pow(x.new_tensor(10.0), abs_unscaled - 1.0)
    return torch.where(abs_unscaled <= 1.0, linear, log)


def degrid(x: torch.Tensor, normy: torch.Tensor, xq: torch.Tensor, heatthr: float, heatsf: float) -> torch.Tensor:
    """Interpolate NN heating rates from the NN grid back to the model grid."""
    batch_size, ndata, _ = normy.shape
    y = torch_denormalize_symlog(normy, heatthr, heatsf).squeeze(-1)

    nreal = min(x.shape[0], ndata)
    x_data = x[:nreal]
    y = y[:, :nreal]
    x_flip = torch.flip(x_data, dims=[0])
    y_flip = torch.flip(y, dims=[1])

    idx = torch.searchsorted(x_flip, xq.contiguous())
    idx = torch.clamp(idx, 1, nreal - 1)

    x_exp = x_flip.unsqueeze(0).expand(batch_size, -1)
    x0 = torch.gather(x_exp, 1, idx - 1)
    x1 = torch.gather(x_exp, 1, idx)
    y0 = torch.gather(y_flip, 1, idx - 1)
    y1 = torch.gather(y_flip, 1, idx)

    t = (xq - x0) / (x1 - x0 + 1e-12)
    interp = y0 + t * (y1 - y0)

    interp = torch.where(xq >= x_data[0], y[:, :1], interp)
    interp = torch.where(xq <= x_data[-1], y[:, -1:], interp)
    return interp


def normalize_standard(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


def calcglobal(
    lon: torch.Tensor,
    lat: torch.Tensor,
    sinlat: torch.Tensor,
    coslat: torch.Tensor,
    current_time: float,
    fluxmean: float,
    fluxstd: float,
    umumean: float,
    umustd: float,
    syear: float,
    stellar_flux_nadir: float,
    substellar_lon: float,
    substellar_lat: float,
    rotation_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pi = torch.pi
    ls = (current_time % (2 * syear)) * 2 * pi / syear
    obliquity = lat.new_tensor(97.77 * pi / 180.0)
    seasonal_declination = torch.asin(torch.sin(obliquity) * torch.sin(lat.new_tensor(ls)))
    subsolar_lat = torch.clamp(
        lat.new_tensor(substellar_lat) + seasonal_declination,
        -0.5 * pi,
        0.5 * pi,
    )
    rotating_subsolar_lon = substellar_lon - rotation_rate * current_time
    subsolar_lon = torch.remainder(
        lon.new_tensor(rotating_subsolar_lon),
        2.0 * pi,
    )
    hour_angle = torch.atan2(torch.sin(lon - subsolar_lon), torch.cos(lon - subsolar_lon))

    sin_subsolar_lat = torch.sin(subsolar_lat)
    cos_subsolar_lat = torch.cos(subsolar_lat)
    cos_zenith = sinlat * sin_subsolar_lat + coslat * cos_subsolar_lat * torch.cos(hour_angle)
    cos_zenith = torch.clamp(cos_zenith, -1.0, 1.0)
    day_side_cos_zenith = torch.clamp(cos_zenith, min=0.0)

    zenith_angle = torch.acos(torch.clamp(day_side_cos_zenith, 0.0, 1.0))
    night_side_zenith = torch.full_like(zenith_angle, 0.5 * pi)
    zenith_angle = torch.where(day_side_cos_zenith > 0.0, zenith_angle, night_side_zenith)

    solar_forcing = stellar_flux_nadir * day_side_cos_zenith
    flux = normalize_standard(day_side_cos_zenith, fluxmean, fluxstd)
    umu = normalize_standard(zenith_angle / (0.5 * pi), umumean, umustd)
    return torch.stack([flux, umu], dim=-1), zenith_angle * (180.0 / pi), solar_forcing, day_side_cos_zenith


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_relative_path(config_file: str, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(config_file).resolve().parent / candidate


def select_device(block: snapy.MeshBlock) -> torch.device:
    backend = block.options.layout().backend()
    if backend == "gloo":
        return torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        return torch.device(block.options.device_str())
    return torch.device("cpu")


def eos_gas_constant(eos) -> float:
    gamma = float(eos.options.gammad())
    cv = float(eos.species_cv_ref())
    return cv * (gamma - 1.0)


def create_models(config_file: str, output_dir: str | None = None):
    config_stem = Path(config_file).stem
    op = MeshOptions.from_yaml(config_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        op.block().output_dir(output_dir)
        if hasattr(op.block(), "file_basename"):
            op.block().file_basename(config_stem)
        else:
            print(
                f"[WARN] snapy MeshBlockOptions.file_basename() is unavailable; "
                f"using snapy default output basename from {config_file}",
                flush=True,
            )

    mesh = Mesh(op)
    device = select_device(mesh.blocks[0])
    mesh.to(device)

    eos = mesh.blocks[0].module("hydro.eos")
    return mesh, eos, device


def load_radiative_model(config_file: str, config: dict, device: torch.device) -> torch.jit.ScriptModule:
    model_path = resolve_config_relative_path(config_file, str(config["problem"]["modelfile"]))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Radiative model file not found: {model_path} "
            f"(from problem.modelfile={config['problem']['modelfile']!r})"
        )

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def build_isothermal_profile(block: snapy.MeshBlock, eos, config: dict) -> torch.Tensor:
    grav = -float(config["forcing"]["const-gravity"]["grav1"])
    problem = config["problem"]
    ts = float(problem["Ts"])
    ps = float(problem["Ps"])
    gas_constant = eos_gas_constant(eos)

    coord = block.module("coord")
    x1v = coord.buffer("x1v")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")
    dzf = coord.buffer("dx1f")

    nvar = eos.nvar()
    hydro_w = torch.zeros((nvar, x3v.shape[0], x2v.shape[0], x1v.shape[0]), dtype=x1v.dtype, device=x1v.device)

    pres = torch.full((x3v.shape[0], x2v.shape[0]), ps, dtype=x1v.dtype, device=x1v.device)
    temp = torch.full_like(pres, ts)

    for i in range(x1v.shape[0]):
        pres = pres * torch.exp(-grav * dzf[i] / (gas_constant * temp))
        hydro_w[kIPR, ..., i] = pres
        hydro_w[kIDN, ..., i] = pres / (gas_constant * temp)

    return hydro_w


def initialize_isothermal(mesh: Mesh, eos, config: dict) -> tuple[list[dict[str, torch.Tensor]], float]:
    mesh_vars: list[dict[str, torch.Tensor]] = []
    for block in mesh.blocks:

        hydro_w = build_isothermal_profile(block, eos, config)
        hydro_w[kIV1] += 1e-6 * torch.randn_like(hydro_w[kIV1])
        mesh_vars.append({"hydro_w": hydro_w})

    return mesh.initialize(mesh_vars)


def _resolve_local_face_name(block: snapy.MeshBlock) -> str:
    layout = block.get_layout()
    rank = int(layout.options.rank())
    loc = layout.loc_of(rank)
    face_id = int(loc[2])
    return snapy.coord.get_cs_face_name(face_id)


def build_tidal_forcing_state(
    block: snapy.MeshBlock,
    config: dict,
    device: torch.device,
    eos,
    model: torch.jit.ScriptModule,
) -> ForcingState:
    coord = block.module("coord")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")

    beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
    face_name = _resolve_local_face_name(block)
    lon, lat = snapy.coord.cs_ab_to_lonlat(face_name, alpha, beta)

    problem = config["problem"]
    coriolis = config["forcing"]["coriolis"]
    batch = x2v.shape[0] * x3v.shape[0]
    mask = torch.ones((batch, 256), dtype=torch.bool, device=device)
    mask[:, :100] = False

    basepress = torch.tensor(
        [
            474464.0, 444378.0, 416200.0, 389808.0, 365090.0, 341939.0, 320256.0, 299948.0,
            280928.0, 263114.0, 246430.0, 230804.0, 216168.0, 202461.0, 189622.0, 177598.0,
            166337.0, 155789.0, 145910.0, 136658.0, 127992.0, 119876.0, 112275.0, 105155.0,
            98487.2, 92242.1, 86392.9, 80914.6, 75783.7, 70978.2, 66477.4, 62262.0, 58313.9,
            54616.2, 51152.9, 47909.2, 44871.3, 42025.9, 39361.0, 36865.1, 34527.4, 32338.0,
            30287.4, 28366.9, 26568.1, 24883.4, 23305.5, 21827.7, 20443.6, 19147.2, 17933.1,
            16795.9, 15730.9, 14733.4, 13799.1, 12924.1, 12104.5, 11337.0, 10618.1, 9944.8,
            9314.2, 8723.6, 8170.4, 7652.3, 7167.1, 6712.6, 6286.9, 5888.3, 5514.9, 5165.2,
            4837.7, 4530.9, 4243.6, 3974.5, 3722.5, 3486.4, 3265.3, 3058.3, 2864.4, 2682.7,
            2512.6, 2353.3, 2204.1, 2064.3, 1933.4, 1810.8, 1696.0, 1588.4, 1487.7, 1393.4,
            1305.0, 1222.3, 1144.8, 1072.2, 1004.2, 940.5, 880.9, 825.0, 772.7, 723.7,
        ],
        dtype=x2v.dtype,
        device=device,
    )

    return ForcingState(
        fluxmean=float(problem["fluxmean"]),
        fluxstd=float(problem["fluxstd"]),
        umumean=float(problem["umumean"]),
        umustd=float(problem["umustd"]),
        tempmean=float(problem["tempmean"]),
        tempstd=float(problem["tempstd"]),
        heatthr=float(problem["heatthr"]),
        heatsf=float(problem["heatsf"]),
        model=model,
        syear=float(problem["syear"]),
        top_depth=int(problem.get("forcing_depth_top", 1)),
        bottom_depth=int(problem.get("forcing_depth_bottom", 1)),
        gas_constant=eos_gas_constant(eos),
        mask=mask,
        basepress=basepress,
        lon=lon.to(device),
        lat=lat.to(device),
        sinlat=torch.sin(lat).to(device),
        coslat=torch.cos(lat).to(device),
        stellar_flux_nadir=float(problem["stellar_flux_nadir"]),
        substellar_lon=float(problem.get("substellar_lon_deg", 0.0) * math.pi / 180.0),
        substellar_lat=float(problem.get("substellar_lat_deg", 0.0) * math.pi / 180.0),
        rotation_rate=float(coriolis.get("omega1", 0.0)),
        rt_update_cadence=float(problem.get("rt_update_cadence", 1.0e4)),
        tidal_heating_decay_tau=float(problem.get("tidal_heating_decay_tau", 5.0e5)),
    )


def initialize_block_diagnostics(block: snapy.MeshBlock) -> BlockDiagnostics:
    coord = block.module("coord")
    x1v = coord.buffer("x1v")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")
    shape = (x3v.shape[0], x2v.shape[0], x1v.shape[0])
    zeros = torch.zeros(shape, dtype=x1v.dtype, device=x1v.device)
    return BlockDiagnostics(
        solar_zenith_angle=torch.full(shape, 90.0, dtype=x1v.dtype, device=x1v.device),
        solar_forcing=zeros.clone(),
        heating_tendency=zeros.clone(),
    )


def register_user_output(block: snapy.MeshBlock, diagnostics: BlockDiagnostics) -> None:
    def user_output(_vars: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "solar_zenith_angle": diagnostics.solar_zenith_angle.contiguous(),
            "solar_forcing": diagnostics.solar_forcing.contiguous(),
            "heating_tendency": diagnostics.heating_tendency.contiguous(),
        }

    set_output = getattr(block, "set_user_output_func")
    set_output(user_output)


def apply_tidal_forcing(
    block: snapy.MeshBlock,
    block_vars: dict[str, torch.Tensor],
    dt: float,
    heating_tendency: torch.Tensor,
    forcing: ForcingState,
    current_time: float,
) -> None:
    hydro_u = block_vars["hydro_u"]
    tau = forcing.tidal_heating_decay_tau
    if tau > 0.0 and current_time < 50.0 * tau:
        hydro_u[kIPR] += (1 + 1e6 * math.exp(-current_time / tau)) * heating_tendency * dt
    else:
        hydro_u[kIPR] += heating_tendency * dt

    block.apply_hydro_bc(hydro_u, type=kConserved)


def compute_radiative_heating_batched(
    mesh_vars: list[dict[str, torch.Tensor]],
    forcing_states: list[ForcingState],
    eos,
    current_time: float,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not mesh_vars:
        return []

    with torch.inference_mode():
        regridded_temps: list[torch.Tensor] = []
        global_feature_batches: list[torch.Tensor] = []
        mask_batches: list[torch.Tensor] = []
        hydro_ws: list[torch.Tensor] = []
        pressbatches: list[torch.Tensor] = []
        block_shapes: list[tuple[int, int, int]] = []
        solar_zenith_angles: list[torch.Tensor] = []
        solar_forcings: list[torch.Tensor] = []

        for block_vars, forcing in zip(mesh_vars, forcing_states):
            hydro_u = block_vars["hydro_u"]
            hydro_w = eos.compute("U->W", [hydro_u])
            temperature = eos.compute("W->T", [hydro_w])

            nx3, nx2, nz = hydro_w[kIPR].shape
            batch = nx2 * nx3

            pressbatch = hydro_w[kIPR].reshape(batch, nz)
            tempbatch = temperature.reshape(batch, nz)
            regridtemp = regrid_tensor(pressbatch, tempbatch, forcing.basepress, forcing.tempmean, forcing.tempstd)

            global_features, solar_zenith_angle, solar_forcing, _ = calcglobal(
                forcing.lon,
                forcing.lat,
                forcing.sinlat,
                forcing.coslat,
                current_time,
                forcing.fluxmean,
                forcing.fluxstd,
                forcing.umumean,
                forcing.umustd,
                forcing.syear,
                forcing.stellar_flux_nadir,
                forcing.substellar_lon,
                forcing.substellar_lat,
                forcing.rotation_rate,
            )

            regridded_temps.append(regridtemp)
            global_feature_batches.append(global_features.reshape(batch, 2))
            mask_batches.append(forcing.mask)
            hydro_ws.append(hydro_w)
            pressbatches.append(pressbatch)
            block_shapes.append((nx3, nx2, nz))
            solar_zenith_angles.append(solar_zenith_angle)
            solar_forcings.append(solar_forcing)

        model = forcing_states[0].model
        output = model(
            torch.cat(regridded_temps, dim=0).to(torch.float32),
            torch.cat(global_feature_batches, dim=0).to(torch.float32),
            torch.cat(mask_batches, dim=0),
        )

        results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        offset = 0
        for forcing, hydro_w, pressbatch, shape, solar_zenith_angle, solar_forcing in zip(
            forcing_states,
            hydro_ws,
            pressbatches,
            block_shapes,
            solar_zenith_angles,
            solar_forcings,
        ):
            nx3, nx2, nz = shape
            batch = nx2 * nx3
            block_output = output[offset : offset + batch]
            offset += batch

            heating = degrid(forcing.basepress, block_output, pressbatch, forcing.heatthr, forcing.heatsf).reshape(nx3, nx2, nz)
            heating_tendency = forcing.gas_constant * hydro_w[kIDN] * heating
            results.append(
                (
                    heating_tendency,
                    solar_zenith_angle.reshape(nx3, nx2, 1).expand(nx3, nx2, nz),
                    solar_forcing.reshape(nx3, nx2, 1).expand(nx3, nx2, nz),
                )
            )

        return results


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
                "python run_uranus.py "
                f"-c {config_file} --output-dir {output_dir} --restart-name "
                + (Path(restart_file).name if restart_file else "<restart-file-name>")
            )
        },
    }

    manifest_file = checkpoint_dir / f"checkpoint_day_{checkpoint_day:04d}.yaml"
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def run_simulation(
    mesh: Mesh,
    eos,
    mesh_vars: list[dict[str, torch.Tensor]],
    current_time: float,
    tlim: float,
    forcing_states: list[ForcingState],
    block_diagnostics: list[BlockDiagnostics],
    config_file: str,
    output_dir: str,
    basename: str,
) -> tuple[list[dict[str, torch.Tensor]], float]:
    intg = mesh.module("block0.intg")
    intg.options.tlim(tlim)

    next_checkpoint_day = int(current_time // (10.0 * SECONDS_PER_DAY)) * 10 + 10
    checkpoint_dir = Path(output_dir) / "restart_checkpoints"

    cycle = 0
    next_rt_update_time = current_time
    heating_tendencies: list[torch.Tensor | None] = [None] * len(mesh.blocks)

    if forcing_states:
        rt_results = compute_radiative_heating_batched(mesh_vars, forcing_states, eos, current_time)
        heating_tendencies = []
        for diagnostics, (heating_tendency, solar_zenith_angle, solar_forcing) in zip(block_diagnostics, rt_results):
            diagnostics.heating_tendency = heating_tendency.contiguous()
            diagnostics.solar_zenith_angle = solar_zenith_angle.contiguous()
            diagnostics.solar_forcing = solar_forcing.contiguous()
            heating_tendencies.append(heating_tendency)
        rt_update_cadence = forcing_states[0].rt_update_cadence
        next_rt_update_time = current_time if rt_update_cadence <= 0.0 else current_time + rt_update_cadence

    mesh.make_outputs(mesh_vars, current_time)

    while not intg.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        mesh.print_cycle_info(mesh_vars, current_time, dt)

        if forcing_states and current_time >= next_rt_update_time:
            rt_results = compute_radiative_heating_batched(mesh_vars, forcing_states, eos, current_time)
            heating_tendencies = []
            for diagnostics, (heating_tendency, solar_zenith_angle, solar_forcing) in zip(block_diagnostics, rt_results):
                diagnostics.heating_tendency = heating_tendency.contiguous()
                diagnostics.solar_zenith_angle = solar_zenith_angle.contiguous()
                diagnostics.solar_forcing = solar_forcing.contiguous()
                heating_tendencies.append(heating_tendency)
            rt_update_cadence = forcing_states[0].rt_update_cadence if forcing_states else 0.0
            if rt_update_cadence <= 0.0:
                next_rt_update_time = current_time
            else:
                next_rt_update_time = current_time + rt_update_cadence
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
            for block, block_vars, heating_tendency, forcing in zip(mesh.blocks, mesh_vars, heating_tendencies, forcing_states):
                if heating_tendency is not None:
                    apply_tidal_forcing(block, block_vars, dt, heating_tendency, forcing, current_time)

        err = mesh.check_redo(mesh_vars)
        if err > 0:
            continue
        if err < 0:
            break

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

    return mesh_vars, current_time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Uranus simulation.")
    p.add_argument("-c", "--config", required=True, help="YAML configuration file")
    p.add_argument("--output-dir", default="/home/chengcli/data", help="Output directory")
    p.add_argument(
        "--restart-name",
        default="",
        help=(
            "Restart archive filename inside output dir (e.g. "
            "uranus.00005.restart or uranus.final.restart)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    mesh, eos, device = create_models(args.config, args.output_dir)
    model = load_radiative_model(args.config, config, device)

    if args.restart_name:
        mesh_vars, current_time = mesh.initialize_from_restart(args.restart_name)
    else:
        mesh_vars, current_time = initialize_isothermal(mesh, eos, config)

    forcing_states = [build_tidal_forcing_state(block, config, device, eos, model) for block in mesh.blocks]
    block_diagnostics = [initialize_block_diagnostics(block) for block in mesh.blocks]
    for block, diagnostics in zip(mesh.blocks, block_diagnostics):
        register_user_output(block, diagnostics)

    tlim = float(config["integration"]["tlim"])
    basename = Path(args.config).stem
    mesh_vars, current_time = run_simulation(
        mesh=mesh,
        eos=eos,
        mesh_vars=mesh_vars,
        current_time=current_time,
        tlim=tlim,
        forcing_states=forcing_states,
        block_diagnostics=block_diagnostics,
        config_file=args.config,
        output_dir=args.output_dir,
        basename=basename,
    )

    mesh.finalize(mesh_vars, current_time)


if __name__ == "__main__":
    main()
