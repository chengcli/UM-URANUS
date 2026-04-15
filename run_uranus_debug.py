#!/usr/bin/env python3
"""Run the Uranus GCM with mesh-level snapy entry points and EOS-driven ICs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import glob
import math
import os
import resource
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
# import numpy
import torch
import yaml
import snapy
from snapy import Mesh, MeshOptions, kConserved, kIDN, kIPR, kIV1

SECONDS_PER_DAY = 86400.0


@dataclass
class ForcingState:
    top_depth: int
    bottom_depth: int
    gas_constant: float
    bottom_temp_rate: float
    top_temp_rate: float
    lon: torch.Tensor
    lat: torch.Tensor
    stellar_flux_nadir: float
    substellar_lon: float
    substellar_lat: float
    rotation_rate: float
    rt_update_cadence: float


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
    xq_expand = xq.unsqueeze(0).expand(batch_size, nq)

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

    linear_mask = abs_unscaled <= 1.0
    y = torch.zeros_like(x)
    y[linear_mask] = unscaled[linear_mask] * thr

    log_mask = ~linear_mask
    if log_mask.any():
        y[log_mask] = torch.sign(unscaled[log_mask]) * thr * 10 ** (abs_unscaled[log_mask] - 1.0)

    return y


def degrid(x: torch.Tensor, normy: torch.Tensor, xq: torch.Tensor, heatthr: float, heatsf: float) -> torch.Tensor:
    """Interpolate NN heating rates from the NN grid back to the model grid."""
    batch_size, ndata, _ = normy.shape
    y = torch_denormalize_symlog(normy, heatthr, heatsf).squeeze(-1)

    nreal = min(x.shape[0], ndata)
    x_data = x[:nreal]
    y = y[:, :nreal]
    x_flip = torch.flip(x_data, dims=[0])
    y_flip = torch.flip(y, dims=[1])

    idx = torch.searchsorted(x_flip, xq)
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
    obliquity = torch.tensor(97.77 * pi / 180.0, dtype=lat.dtype, device=lat.device)
    seasonal_declination = torch.asin(torch.sin(obliquity) * torch.sin(torch.tensor(ls, dtype=lat.dtype, device=lat.device)))
    subsolar_lat = torch.clamp(
        torch.tensor(substellar_lat, dtype=lat.dtype, device=lat.device) + seasonal_declination,
        -0.5 * pi,
        0.5 * pi,
    )
    rotating_subsolar_lon = substellar_lon - rotation_rate * current_time
    subsolar_lon = torch.remainder(
        torch.tensor(rotating_subsolar_lon, dtype=lon.dtype, device=lon.device),
        2.0 * pi,
    )
    hour_angle = torch.atan2(torch.sin(lon - subsolar_lon), torch.cos(lon - subsolar_lon))

    sin_subsolar_lat = torch.sin(subsolar_lat)
    cos_subsolar_lat = torch.cos(subsolar_lat)
    sinlat = torch.sin(lat)
    coslat = torch.cos(lat)
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


def apply_snapy_top_sponge(config: dict) -> dict:
    forcing = config.setdefault("forcing", {})
    problem = config.get("problem", {})

    tau = problem.get("sponge_tau")
    width = problem.get("spongeheight")
    if tau is None or width is None:
        return config

    forcing.setdefault(
        "top-sponge-lyr",
        {
            "tau": float(tau),
            "width": float(width),
        },
    )
    return config


def select_device(block: snapy.MeshBlock) -> torch.device:
    if torch.cuda.is_available() and block.options.layout().backend() == "nccl":
        return torch.device(block.options.device_str())
    return torch.device("cpu")


def eos_gas_constant(eos) -> float:
    gamma = float(eos.options.gammad())
    cv = float(eos.species_cv_ref())
    return cv * (gamma - 1.0)


def create_models(config_file: str, config: dict, output_dir: str | None = None):
    config_stem = Path(config_file).stem
    temp_config_path: str | None = None
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="uranus_snapy_", delete=False, encoding="utf-8"
    ) as f:
        yaml.safe_dump(config, f, sort_keys=False)
        temp_config_path = f.name

    try:
        op = MeshOptions.from_yaml(temp_config_path)
    finally:
        if temp_config_path and os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        op.block().output_dir(output_dir)
        if hasattr(op.block(), "file_basename"):
            op.block().file_basename(config_stem)
        else:
            print(
                f"[WARN] snapy MeshBlockOptions.file_basename() is unavailable; "
                f"using snapy default output basename instead of {config_stem}",
                flush=True,
            )

    mesh = Mesh(op)
    device = select_device(mesh.blocks[0])
    mesh.to(device)

    eos = mesh.blocks[0].module("hydro.eos")
    return mesh, eos, device


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


def snapshot_block_cycles(mesh: Mesh) -> list[tuple[int, int, int, str]]:
    states: list[tuple[int, int, int, str]] = []
    for block_index, block in enumerate(mesh.blocks):
        layout = block.get_layout()
        rank = int(layout.options.rank())
        cycle = int(block.cycle())
        face_name = _resolve_local_face_name(block)
        states.append((block_index, rank, cycle, face_name))
    return states


def summarize_block_cycles(states: list[tuple[int, int, int, str]]) -> str:
    if not states:
        return "[]"
    return "[" + ", ".join(
        f"block={block_index}:rank={rank}:cycle={cycle}:face={face_name}"
        for block_index, rank, cycle, face_name in states
    ) + "]"


def block_cycles_in_sync(states: list[tuple[int, int, int, str]]) -> bool:
    if not states:
        return True
    cycles = {cycle for _, _, cycle, _ in states}
    return len(cycles) == 1


def format_bytes(nbytes: int | float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(nbytes)
    unit = units[0]
    for next_unit in units[1:]:
        if abs(value) < 1024.0:
            break
        value /= 1024.0
        unit = next_unit
    return f"{value:.2f}{unit}"


def _finite_stat_line(name: str, tensor: torch.Tensor) -> str:
    data = tensor.detach()
    total = data.numel()
    finite_mask = torch.isfinite(data)
    finite_count = int(finite_mask.sum().item())
    nan_count = int(torch.isnan(data).sum().item())
    posinf_count = int(torch.isposinf(data).sum().item())
    neginf_count = int(torch.isneginf(data).sum().item())

    if finite_count > 0:
        finite_values = data[finite_mask]
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        mean_value = float(finite_values.mean().item())
        range_text = f"min={min_value:.6e} max={max_value:.6e} mean={mean_value:.6e}"
    else:
        range_text = "min=<nonfinite> max=<nonfinite> mean=<nonfinite>"

    return (
        f"{name}: finite={finite_count}/{total} nan={nan_count} "
        f"+inf={posinf_count} -inf={neginf_count} {range_text}"
    )


def log_block_physical_state(
    label: str,
    *,
    cycle: int,
    current_time: float,
    mesh: Mesh,
    mesh_vars: list[dict[str, torch.Tensor]],
    eos,
    forcing_states: list[ForcingState],
) -> None:
    print(
        f"[STATE-CHECK] {label}: cycle={cycle} time={current_time:.14e} blocks={len(mesh.blocks)}",
        flush=True,
    )
    for block_index, (block, block_vars) in enumerate(zip(mesh.blocks, mesh_vars)):
        face_name = _resolve_local_face_name(block)
        hydro_u = block_vars.get("hydro_u")
        if not isinstance(hydro_u, torch.Tensor):
            print(
                f"[STATE-CHECK][ERROR] block={block_index} face={face_name} missing hydro_u tensor",
                flush=True,
            )
            continue

        try:
            hydro_w = eos.compute("U->W", [hydro_u])
        except Exception as exc:
            print(
                f"[STATE-CHECK][ERROR] block={block_index} face={face_name} "
                f"failed U->W conversion: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        lines = [
            _finite_stat_line("conserved_density[kIDN]", hydro_u[kIDN]),
            _finite_stat_line("conserved_energy_slot[kIPR]", hydro_u[kIPR]),
            _finite_stat_line("primitive_density[kIDN]", hydro_w[kIDN]),
            _finite_stat_line("primitive_pressure[kIPR]", hydro_w[kIPR]),
        ]

        forcing = forcing_states[block_index] if block_index < len(forcing_states) else None
        if forcing is not None:
            gas_constant = float(forcing.gas_constant)
            density = hydro_w[kIDN]
            pressure = hydro_w[kIPR]
            safe_denominator = gas_constant * density
            valid_mask = torch.isfinite(pressure) & torch.isfinite(safe_denominator) & (safe_denominator != 0)
            temperature = torch.full_like(pressure, float("nan"))
            temperature[valid_mask] = pressure[valid_mask] / safe_denominator[valid_mask]
            lines.append(_finite_stat_line("derived_temperature", temperature))

        print(
            f"[STATE-CHECK] block={block_index} face={face_name} " + " | ".join(lines),
            flush=True,
        )

        negative_density_count = int((hydro_w[kIDN] < 0).sum().item())
        nonpositive_pressure_count = int((hydro_w[kIPR] <= 0).sum().item())
        if negative_density_count > 0:
            print(
                f"[STATE-CHECK][ERROR] block={block_index} face={face_name} "
                f"negative primitive density count={negative_density_count}",
                flush=True,
            )
        if nonpositive_pressure_count > 0:
            print(
                f"[STATE-CHECK][ERROR] block={block_index} face={face_name} "
                f"nonpositive primitive pressure count={nonpositive_pressure_count}",
                flush=True,
            )
        if not torch.isfinite(hydro_u).all():
            print(
                f"[STATE-CHECK][ERROR] block={block_index} face={face_name} "
                f"non-finite values detected in hydro_u",
                flush=True,
            )
        if not torch.isfinite(hydro_w).all():
            print(
                f"[STATE-CHECK][ERROR] block={block_index} face={face_name} "
                f"non-finite values detected in hydro_w",
                flush=True,
            )


def current_rank(mesh: Mesh | None = None) -> int:
    if mesh is not None and mesh.blocks:
        layout = mesh.blocks[0].get_layout()
        return int(layout.options.rank())
    return int(os.environ.get("RANK", "-1"))


def estimate_tensor_bytes(mesh_vars: list[dict[str, torch.Tensor]] | None) -> int:
    if mesh_vars is None:
        return 0

    total = 0
    for block_vars in mesh_vars:
        for data in block_vars.values():
            if isinstance(data, torch.Tensor):
                total += data.numel() * data.element_size()
    return total


def get_rss_bytes() -> int:
    # Linux ru_maxrss is reported in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def log_memory(
    label: str,
    *,
    cycle: int,
    current_time: float,
    device: torch.device | None,
    mesh: Mesh | None = None,
    mesh_vars: list[dict[str, torch.Tensor]] | None = None,
    include_summary: bool = False,
) -> None:
    rank = current_rank(mesh)
    tensor_bytes = estimate_tensor_bytes(mesh_vars)
    rss_bytes = get_rss_bytes()

    if device is None:
        print(
            f"[MEMORY] {label}: rank={rank} cycle={cycle} time={current_time:.14e} "
            f"device=unknown rss={format_bytes(rss_bytes)} tensor_bytes={format_bytes(tensor_bytes)}",
            flush=True,
        )
        return

    if device.type != "cuda" or not torch.cuda.is_available():
        print(
            f"[MEMORY] {label}: rank={rank} cycle={cycle} time={current_time:.14e} "
            f"device={device} rss={format_bytes(rss_bytes)} tensor_bytes={format_bytes(tensor_bytes)}",
            flush=True,
        )
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    max_allocated_bytes = torch.cuda.max_memory_allocated(device)
    max_reserved_bytes = torch.cuda.max_memory_reserved(device)

    print(
        f"[MEMORY] {label}: rank={rank} cycle={cycle} time={current_time:.14e} "
        f"device={device} alloc={format_bytes(allocated_bytes)} "
        f"reserved={format_bytes(reserved_bytes)} max_alloc={format_bytes(max_allocated_bytes)} "
        f"max_reserved={format_bytes(max_reserved_bytes)} free={format_bytes(free_bytes)} "
        f"total={format_bytes(total_bytes)} rss={format_bytes(rss_bytes)} "
        f"tensor_bytes={format_bytes(tensor_bytes)}",
        flush=True,
    )
    if include_summary:
        print(torch.cuda.memory_summary(device=device, abbreviated=False), flush=True)


def log_runtime_diagnostics(
    *,
    args: argparse.Namespace,
    config: dict,
    mesh: Mesh | None,
    device: torch.device | None,
) -> None:
    tracked_env = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_LAUNCH_BLOCKING",
        "TORCH_DISTRIBUTED_DEBUG",
        "TORCH_NCCL_ENABLE_MONITORING",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_BLOCKING_WAIT",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
    ]
    env_summary = ", ".join(f"{key}={os.environ.get(key, '<unset>')}" for key in tracked_env)
    distribute = config.get("distribute", {})
    integration = config.get("integration", {})

    print(
        f"[RUNTIME] args: config={args.config} output_dir={args.output_dir} "
        f"restart_name={args.restart_name} debug_memory={args.debug_memory} "
        f"debug_memory_summary={args.debug_memory_summary} hang_sync_cuda={args.hang_sync_cuda} "
        f"hang_diagnosis={args.hang_diagnosis} hang_timeout_sec={args.hang_timeout_sec} "
        f"hang_diagnosis_repeat={args.hang_diagnosis_repeat}",
        flush=True,
    )
    print(
        f"[RUNTIME] config: backend={distribute.get('backend', '<missing>')} "
        f"layout={distribute.get('layout', '<missing>')} "
        f"blocks_per_process={distribute.get('blocks_per_process', '<missing>')} "
        f"nb2={distribute.get('nb2', '<missing>')} nb3={distribute.get('nb3', '<missing>')} "
        f"integrator={integration.get('type', '<missing>')} cfl={integration.get('cfl', '<missing>')}",
        flush=True,
    )
    print(
        f"[RUNTIME] env: {env_summary}",
        flush=True,
    )
    print(
        f"[RUNTIME] torch: version={torch.__version__} cuda_compiled={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}",
        flush=True,
    )

    if device is not None:
        print(f"[RUNTIME] selected_device: {device}", flush=True)

    if torch.cuda.is_available():
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        print(
            f"[RUNTIME] cuda-current: index={current_idx} name={props.name} "
            f"capability={props.major}.{props.minor} total_memory={format_bytes(props.total_memory)}",
            flush=True,
        )
        for idx in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(idx)
            print(
                f"[RUNTIME] cuda-device[{idx}]: name={device_props.name} "
                f"capability={device_props.major}.{device_props.minor} "
                f"total_memory={format_bytes(device_props.total_memory)}",
                flush=True,
            )

    if mesh is not None and mesh.blocks:
        local_rank = current_rank(mesh)
        print(
            f"[RUNTIME] mesh: local_rank={local_rank} local_blocks={len(mesh.blocks)}",
            flush=True,
        )
        for block_index, block in enumerate(mesh.blocks):
            layout = block.get_layout()
            face_name = _resolve_local_face_name(block)
            neighbors = summarize_block_neighbors(block)
            print(
                f"[RUNTIME] mesh-block[{block_index}]: rank={int(layout.options.rank())} "
                f"cycle={int(block.cycle())} face={face_name} neighbors={neighbors}",
                flush=True,
            )


def log_block_sync(label: str, *, cycle: int, current_time: float, mesh: Mesh) -> None:
    states = snapshot_block_cycles(mesh)
    print(
        f"[BLOCK-SYNC] {label}: cycle={cycle} time={current_time:.14e} "
        f"in_sync={block_cycles_in_sync(states)} states={summarize_block_cycles(states)}",
        flush=True,
    )


def summarize_block_neighbors(block: snapy.MeshBlock) -> str:
    layout = block.get_layout()
    rank = int(layout.options.rank())
    loc = tuple(int(v) for v in layout.loc_of(rank))
    entries = []
    for label, offset in (
        ("self", (0, 0, 0)),
        ("x2-", (0, -1, 0)),
        ("x2+", (0, 1, 0)),
        ("x3-", (-1, 0, 0)),
        ("x3+", (1, 0, 0)),
    ):
        try:
            neighbor_rank = int(layout.neighbor_rank(loc, offset))
        except TypeError:
            neighbor_rank = int(layout.neighbor_rank(*loc, *offset))
        except Exception:
            entries.append(f"{label}=ERR")
            continue
        locality = "self" if neighbor_rank == rank else ("none" if neighbor_rank < 0 else "peer")
        entries.append(f"{label}={neighbor_rank}:{locality}")
    return "[" + ", ".join(entries) + "]"


def install_forward_debug(mesh: Mesh) -> None:
    original_forward = mesh.forward

    def forward_python(
        self: Mesh,
        vars: list[dict[str, torch.Tensor]],
        dt: float,
        stage: int,
        *,
        cycle: int | None = None,
        current_time: float | None = None,
    ) -> None:
        if len(vars) != len(self.blocks):
            raise ValueError(
                f"forward_python expects one Variables map per local MeshBlock; "
                f"got {len(vars)} for {len(self.blocks)} blocks"
            )

        rank = current_rank(self)

        def run_block(block_index: int) -> None:
            block = self.blocks[block_index]
            layout = block.get_layout()
            block_rank = int(layout.options.rank())
            face_name = _resolve_local_face_name(block)
            cycle_text = cycle if cycle is not None else int(block.cycle())
            time_text = (
                f"{current_time:.14e}" if current_time is not None else "<unknown>"
            )
            print(
                f"[FORWARD-PY] block-begin: rank={rank} block={block_index} "
                f"block_rank={block_rank} face={face_name} cycle={cycle_text} "
                f"time={time_text} stage={stage} dt={dt:.14e}",
                flush=True,
            )
            t0 = time.monotonic()
            try:
                block.forward(vars[block_index], dt, stage)
            except Exception as exc:
                print(
                    f"[FORWARD-PY] block-error: rank={rank} block={block_index} "
                    f"block_rank={block_rank} face={face_name} cycle={cycle_text} "
                    f"time={time_text} stage={stage} "
                    f"exc_type={type(exc).__name__} exc={exc}",
                    flush=True,
                )
                traceback.print_exc()
                raise
            print(
                f"[FORWARD-PY] block-done: rank={rank} block={block_index} "
                f"block_rank={block_rank} face={face_name} cycle={cycle_text} "
                f"time={time_text} stage={stage} "
                f"elapsed_sec={time.monotonic() - t0:.6f}",
                flush=True,
            )

        max_workers = max(1, len(self.blocks))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mesh-forward") as pool:
            futures = [pool.submit(run_block, block_index) for block_index in range(len(self.blocks))]
            for block_index, future in enumerate(futures):
                print(
                    f"[FORWARD-PY] wait-block: rank={rank} block={block_index} stage={stage}",
                    flush=True,
                )
                future.result()

    def forward_debug(
        self: Mesh,
        vars: list[dict[str, torch.Tensor]],
        dt: float,
        stage: int,
        *,
        cycle: int,
        current_time: float,
    ) -> None:
        if len(vars) != len(self.blocks):
            raise ValueError(
                f"forward_debug expects one Variables map per local MeshBlock; "
                f"got {len(vars)} for {len(self.blocks)} blocks"
            )

        rank = current_rank(self)
        print(
            f"[FORWARD-DEBUG] begin: rank={rank} cycle={cycle} time={current_time:.14e} "
            f"stage={stage} dt={dt:.14e} blocks={len(self.blocks)}",
            flush=True,
        )
        log_block_sync(
            f"forward-debug-stage-{stage}-entry",
            cycle=cycle,
            current_time=current_time,
            mesh=self,
        )
        t0 = time.monotonic()
        try:
            forward_python(
                self,
                vars,
                dt,
                stage,
                cycle=cycle,
                current_time=current_time,
            )
        except Exception as exc:
            print(
                f"[FORWARD-DEBUG] error: rank={rank} cycle={cycle} "
                f"time={current_time:.14e} stage={stage} "
                f"exc_type={type(exc).__name__} exc={exc}",
                flush=True,
            )
            traceback.print_exc()
            raise

        log_block_sync(
            f"forward-debug-stage-{stage}-exit",
            cycle=cycle,
            current_time=current_time,
            mesh=self,
        )
        print(
            f"[FORWARD-DEBUG] done: rank={rank} cycle={cycle} time={current_time:.14e} "
            f"stage={stage} elapsed_sec={time.monotonic() - t0:.6f}",
            flush=True,
        )

    mesh.native_forward = original_forward
    mesh.forward = MethodType(forward_python, mesh)
    mesh.forward_debug = MethodType(forward_debug, mesh)


def build_tidal_forcing_state(block: snapy.MeshBlock, config: dict, device: torch.device, eos) -> ForcingState:
    coord = block.module("coord")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")

    beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
    face_name = _resolve_local_face_name(block)
    lon, lat = snapy.coord.cs_ab_to_lonlat(face_name, alpha, beta)

    problem = config["problem"]
    coriolis = config["forcing"]["coriolis"]
    return ForcingState(
        top_depth=int(problem.get("forcing_depth_top", 1)),
        bottom_depth=int(problem.get("forcing_depth_bottom", 1)),
        gas_constant=eos_gas_constant(eos),
        bottom_temp_rate=float(problem.get("debug_bottom_temp_rate", 1.0e-6)),
        top_temp_rate=float(problem.get("debug_top_temp_rate", -1.0e-6)),
        lon=lon.to(device),
        lat=lat.to(device),
        stellar_flux_nadir=float(problem["stellar_flux_nadir"]),
        substellar_lon=float(problem.get("substellar_lon_deg", 0.0) * math.pi / 180.0),
        substellar_lat=float(problem.get("substellar_lat_deg", 0.0) * math.pi / 180.0),
        rotation_rate=float(coriolis.get("omega1", 0.0)),
        rt_update_cadence=float(problem.get("rt_update_cadence", 1.0e4)),
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
    current_time: float,
) -> None:
    hydro_u = block_vars["hydro_u"]
    tau = 5e5
    if current_time < 50*tau:
        hydro_u[kIPR] += (1+1e6*math.exp(-current_time/tau))*heating_tendency * dt
    else:
        hydro_u[kIPR] += heating_tendency * dt

    block.apply_hydro_bc(hydro_u, type=kConserved)


def compute_radiative_heating(
    block_vars: dict[str, torch.Tensor],
    forcing: ForcingState,
    eos,
    current_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        hydro_u = block_vars["hydro_u"]
        hydro_w = eos.compute("U->W", [hydro_u])

        nx3, nx2, nz = hydro_w[kIPR].shape
        heating = torch.zeros_like(hydro_w[kIPR])

        bottom_depth = max(0, min(forcing.bottom_depth, nz))
        top_depth = max(0, min(forcing.top_depth, nz))
        if bottom_depth > 0:
            heating[..., :bottom_depth] = forcing.bottom_temp_rate
        if top_depth > 0:
            heating[..., -top_depth:] = forcing.top_temp_rate

        _, solar_zenith_angle, solar_forcing, _ = calcglobal(
            forcing.lon,
            forcing.lat,
            current_time,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            forcing.stellar_flux_nadir,
            forcing.substellar_lon,
            forcing.substellar_lat,
            forcing.rotation_rate,
        )
        solar_zenith_angle = solar_zenith_angle.reshape(nx3, nx2, 1).expand(nx3, nx2, nz)
        solar_forcing = solar_forcing.reshape(nx3, nx2, 1).expand(nx3, nx2, nz)
        heating_tendency = forcing.gas_constant * hydro_w[kIDN] * heating
        return heating_tendency, solar_zenith_angle, solar_forcing


def compute_solar_diagnostics(
    block_vars: dict[str, torch.Tensor],
    forcing: ForcingState,
    current_time: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hydro_u = block_vars["hydro_u"]
    nx3, nx2, nz = hydro_u[kIPR].shape

    _, solar_zenith_angle, solar_forcing, _ = calcglobal(
        forcing.lon,
        forcing.lat,
        current_time,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        forcing.stellar_flux_nadir,
        forcing.substellar_lon,
        forcing.substellar_lat,
        forcing.rotation_rate,
    )

    solar_zenith_angle = solar_zenith_angle.reshape(nx3, nx2, 1).expand(nx3, nx2, nz)
    solar_forcing = solar_forcing.reshape(nx3, nx2, 1).expand(nx3, nx2, nz)
    return solar_zenith_angle, solar_forcing


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


def snapshot_outputs(
    output_dir: str,
    basename: str,
    *,
    include_block: bool,
) -> dict[str, tuple[int, int]]:
    pattern_nc = Path(output_dir) / f"{basename}.*.nc"
    pattern_restart = Path(output_dir) / f"{basename}.*.restart"
    snapshots: dict[str, tuple[int, int]] = {}

    for pattern in (pattern_nc, pattern_restart):
        for match in glob.glob(str(pattern)):
            path = Path(match)
            is_block = ".block" in path.name
            if is_block and not include_block:
                continue
            if (not is_block) and include_block:
                continue
            if not path.is_file():
                continue
            stat = path.stat()
            snapshots[path.name] = (stat.st_mtime_ns, stat.st_size)

    return snapshots


def report_output_changes(
    label: str,
    *,
    before_merged: dict[str, tuple[int, int]],
    after_merged: dict[str, tuple[int, int]],
    before_block: dict[str, tuple[int, int]],
    after_block: dict[str, tuple[int, int]],
    cycle: int,
    current_time: float,
) -> None:
    changed_merged = []
    for name, meta in sorted(after_merged.items()):
        if before_merged.get(name) != meta:
            changed_merged.append(name)

    changed_block = []
    for name, meta in sorted(after_block.items()):
        if before_block.get(name) != meta:
            changed_block.append(name)

    print(
        f"[OUTPUT-MERGE] {label}: cycle={cycle} time={current_time:.14e} "
        f"case_no_block_and_no_merge={not changed_block and not changed_merged} "
        f"case_block_no_merge={bool(changed_block) and not changed_merged} "
        f"case_block_and_merge={bool(changed_block) and bool(changed_merged)} "
        f"case_merge_without_block={not changed_block and bool(changed_merged)} "
        f"merged={changed_merged} block={changed_block}",
        flush=True,
    )


def log_phase(
    label: str,
    *,
    cycle: int,
    current_time: float,
    mesh: Mesh | None = None,
    mesh_vars: list[dict[str, torch.Tensor]] | None = None,
    device: torch.device | None = None,
    debug_memory: bool = False,
    debug_memory_summary: bool = False,
) -> None:
    print(
        f"[PHASE] {label}: cycle={cycle} time={current_time:.14e}",
        flush=True,
    )
    if mesh is not None:
        log_block_sync(f"{label}-block-state", cycle=cycle, current_time=current_time, mesh=mesh)
    if debug_memory:
        log_memory(
            f"{label}-memory",
            cycle=cycle,
            current_time=current_time,
            device=device,
            mesh=mesh,
            mesh_vars=mesh_vars,
            include_summary=debug_memory_summary,
        )


class HangDiagnosisWatchdog:
    def __init__(
        self,
        *,
        enabled: bool,
        timeout_sec: float,
        repeat: bool,
        cycle: int,
        current_time: float,
        rank: int,
        label: str,
        mesh: Mesh | None = None,
    ) -> None:
        self.enabled = enabled
        self.timeout_sec = timeout_sec
        self.repeat = repeat
        self.cycle = cycle
        self.current_time = current_time
        self.rank = rank
        self.label = label
        self.mesh = mesh
        self.armed = False
        self.started_at = 0.0

    def __enter__(self) -> "HangDiagnosisWatchdog":
        if not self.enabled:
            return self

        self.started_at = time.monotonic()
        print(
            f"[HANG-DIAG] arm: rank={self.rank} cycle={self.cycle} "
            f"time={self.current_time:.14e} label={self.label} "
            f"timeout_sec={self.timeout_sec} repeat={self.repeat}",
            flush=True,
        )
        if self.mesh is not None:
            log_block_sync(
                f"{self.label}-entry-block-state",
                cycle=self.cycle,
                current_time=self.current_time,
                mesh=self.mesh,
            )
        faulthandler.dump_traceback_later(self.timeout_sec, repeat=self.repeat)
        self.armed = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False

        if self.armed:
            faulthandler.cancel_dump_traceback_later()
        elapsed = time.monotonic() - self.started_at
        status = "error" if exc_type is not None else "done"
        print(
            f"[HANG-DIAG] {status}: rank={self.rank} cycle={self.cycle} "
            f"time={self.current_time:.14e} label={self.label} elapsed_sec={elapsed:.3f}",
            flush=True,
        )
        if self.mesh is not None:
            log_block_sync(
                f"{self.label}-exit-block-state",
                cycle=self.cycle,
                current_time=self.current_time,
                mesh=self.mesh,
            )
        return False


def maybe_cuda_sync(
    *,
    enabled: bool,
    device: torch.device | None,
    label: str,
    cycle: int,
    current_time: float,
    mesh: Mesh | None = None,
    mesh_vars: list[dict[str, torch.Tensor]] | None = None,
    debug_memory: bool = False,
    debug_memory_summary: bool = False,
) -> None:
    if not enabled or device is None or device.type != "cuda" or not torch.cuda.is_available():
        return

    log_phase(
        f"{label}-before-cuda-sync",
        cycle=cycle,
        current_time=current_time,
        mesh=mesh,
        mesh_vars=mesh_vars,
        device=device,
        debug_memory=debug_memory,
        debug_memory_summary=debug_memory_summary,
    )
    torch.cuda.synchronize(device)
    log_phase(
        f"{label}-after-cuda-sync",
        cycle=cycle,
        current_time=current_time,
        mesh=mesh,
        mesh_vars=mesh_vars,
        device=device,
        debug_memory=debug_memory,
        debug_memory_summary=debug_memory_summary,
    )


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
    device: torch.device,
    debug_memory: bool = False,
    debug_memory_summary: bool = False,
    hang_diagnosis: bool = False,
    hang_timeout_sec: float = 120.0,
    hang_diagnosis_repeat: bool = False,
    hang_sync_cuda: bool = False,
) -> tuple[list[dict[str, torch.Tensor]], float]:
    intg = mesh.module("block0.intg")
    intg.options.tlim(tlim)

    next_checkpoint_day = int(current_time // (10.0 * SECONDS_PER_DAY)) * 10 + 10
    checkpoint_dir = Path(output_dir) / "restart_checkpoints"

    cycle = 0
    before_merged_outputs = snapshot_outputs(output_dir, basename, include_block=False)
    before_block_outputs = snapshot_outputs(output_dir, basename, include_block=True)
    log_phase(
        "before-initial-make_outputs",
        cycle=cycle,
        current_time=current_time,
        mesh=mesh,
        mesh_vars=mesh_vars,
        device=device,
        debug_memory=debug_memory,
        debug_memory_summary=debug_memory_summary,
    )
    mesh.make_outputs(mesh_vars, current_time)
    log_phase(
        "after-initial-make_outputs",
        cycle=cycle,
        current_time=current_time,
        mesh=mesh,
        mesh_vars=mesh_vars,
        device=device,
        debug_memory=debug_memory,
        debug_memory_summary=debug_memory_summary,
    )
    after_merged_outputs = snapshot_outputs(output_dir, basename, include_block=False)
    after_block_outputs = snapshot_outputs(output_dir, basename, include_block=True)
    report_output_changes(
        "after-make_outputs",
        before_merged=before_merged_outputs,
        after_merged=after_merged_outputs,
        before_block=before_block_outputs,
        after_block=after_block_outputs,
        cycle=cycle,
        current_time=current_time,
    )

    next_rt_update_time = current_time
    heating_tendencies: list[torch.Tensor | None] = [None] * len(mesh.blocks)

    while not intg.stop(cycle, current_time):
        cycle += 1
        log_phase(
            "loop-start",
            cycle=cycle,
            current_time=current_time,
            mesh=mesh,
            mesh_vars=mesh_vars,
            device=device,
            debug_memory=debug_memory,
            debug_memory_summary=debug_memory_summary,
        )
        log_block_sync("before-set_cycle", cycle=cycle, current_time=current_time, mesh=mesh)
        mesh.set_cycle(cycle)
        log_block_sync("after-set_cycle", cycle=cycle, current_time=current_time, mesh=mesh)

        log_phase(
            "before-max_time_step",
            cycle=cycle,
            current_time=current_time,
            mesh=mesh,
            mesh_vars=mesh_vars,
            device=device,
            debug_memory=debug_memory,
        )
        dt = mesh.max_time_step(mesh_vars)
        log_phase(
            "after-max_time_step",
            cycle=cycle,
            current_time=current_time,
            mesh=mesh,
            mesh_vars=mesh_vars,
            device=device,
            debug_memory=debug_memory,
        )
        mesh.print_cycle_info(mesh_vars, current_time, dt)

        for block_vars, forcing, diagnostics in zip(mesh_vars, forcing_states, block_diagnostics):
            solar_zenith_angle, solar_forcing = compute_solar_diagnostics(block_vars, forcing, current_time)
            diagnostics.solar_zenith_angle = solar_zenith_angle.contiguous()
            diagnostics.solar_forcing = solar_forcing.contiguous()

        if current_time >= next_rt_update_time:
            log_phase(
                "before-radiative-heating",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                device=device,
                debug_memory=debug_memory,
            )
            heating_tendencies = []
            for block_vars, forcing, diagnostics in zip(mesh_vars, forcing_states, block_diagnostics):
                heating_tendency, solar_zenith_angle, solar_forcing = compute_radiative_heating(block_vars, forcing, eos, current_time)
                diagnostics.heating_tendency = heating_tendency.contiguous()
                diagnostics.solar_zenith_angle = solar_zenith_angle.contiguous()
                diagnostics.solar_forcing = solar_forcing.contiguous()
                heating_tendencies.append(heating_tendency)
            log_phase(
                "after-radiative-heating",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                device=device,
                debug_memory=debug_memory,
            )
            rt_update_cadence = forcing_states[0].rt_update_cadence if forcing_states else 0.0
            if rt_update_cadence <= 0.0:
                next_rt_update_time = current_time
            else:
                next_rt_update_time = current_time + rt_update_cadence
        # print(heating_tendencies[0][:,:,10])
        for stage in range(len(intg.stages)):
            log_phase(
                f"before-forward-stage-{stage}",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                device=device,
                debug_memory=debug_memory,
            )
            log_block_physical_state(
                f"before-forward-stage-{stage}",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                eos=eos,
                forcing_states=forcing_states,
            )
            maybe_cuda_sync(
                enabled=hang_sync_cuda,
                device=device,
                label=f"forward-stage-{stage}-pre",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                debug_memory=debug_memory,
                debug_memory_summary=debug_memory_summary,
            )
            with HangDiagnosisWatchdog(
                enabled=hang_diagnosis,
                timeout_sec=hang_timeout_sec,
                repeat=hang_diagnosis_repeat,
                cycle=cycle,
                current_time=current_time,
                rank=current_rank(mesh),
                label=f"mesh.forward(stage={stage})",
                mesh=mesh,
            ):
                mesh.forward_debug(
                    mesh_vars,
                    dt,
                    stage,
                    cycle=cycle,
                    current_time=current_time,
                )
            maybe_cuda_sync(
                enabled=hang_sync_cuda,
                device=device,
                label=f"forward-stage-{stage}-post",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                debug_memory=debug_memory,
                debug_memory_summary=debug_memory_summary,
            )
            log_phase(
                f"after-forward-stage-{stage}",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                device=device,
                debug_memory=debug_memory,
            )
            log_block_physical_state(
                f"after-forward-stage-{stage}",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                eos=eos,
                forcing_states=forcing_states,
            )
            for block, block_vars, heating_tendency in zip(mesh.blocks, mesh_vars, heating_tendencies):
                if heating_tendency is not None:
                    apply_tidal_forcing(block, block_vars, dt, heating_tendency,current_time)
            log_phase(
                f"after-forcing-stage-{stage}",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                device=device,
                debug_memory=debug_memory,
            )
            log_block_physical_state(
                f"after-forcing-stage-{stage}",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                eos=eos,
                forcing_states=forcing_states,
            )

        err = mesh.check_redo(mesh_vars)
        if debug_memory and err != 0:
            log_memory(
                "after-check_redo",
                cycle=cycle,
                current_time=current_time,
                device=device,
                mesh=mesh,
                mesh_vars=mesh_vars,
                include_summary=debug_memory_summary,
            )
        if err > 0:
            continue
        if err < 0:
            break

        current_time += dt
        before_merged_outputs = snapshot_outputs(output_dir, basename, include_block=False)
        before_block_outputs = snapshot_outputs(output_dir, basename, include_block=True)
        log_phase(
            "before-make_outputs",
            cycle=cycle,
            current_time=current_time,
            mesh=mesh,
            mesh_vars=mesh_vars,
            device=device,
            debug_memory=debug_memory,
        )
        mesh.make_outputs(mesh_vars, current_time)
        log_phase(
            "after-make_outputs",
            cycle=cycle,
            current_time=current_time,
            mesh=mesh,
            mesh_vars=mesh_vars,
            device=device,
            debug_memory=debug_memory,
        )
        after_merged_outputs = snapshot_outputs(output_dir, basename, include_block=False)
        after_block_outputs = snapshot_outputs(output_dir, basename, include_block=True)
        report_output_changes(
            "after-make_outputs",
            before_merged=before_merged_outputs,
            after_merged=after_merged_outputs,
            before_block=before_block_outputs,
            after_block=after_block_outputs,
            cycle=cycle,
            current_time=current_time,
        )

        while current_time >= next_checkpoint_day * SECONDS_PER_DAY:
            log_phase(
                "before-write_restart_manifest",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                device=device,
                debug_memory=debug_memory,
            )
            write_restart_manifest(
                checkpoint_dir=checkpoint_dir,
                checkpoint_day=next_checkpoint_day,
                current_time=current_time,
                config_file=config_file,
                output_dir=output_dir,
                basename=basename,
            )
            log_phase(
                "after-write_restart_manifest",
                cycle=cycle,
                current_time=current_time,
                mesh=mesh,
                mesh_vars=mesh_vars,
                device=device,
                debug_memory=debug_memory,
            )
            next_checkpoint_day += 10

    return mesh_vars, current_time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Uranus simulation.")
    p.add_argument("-c", "--config", required=True, help="YAML configuration file")
    p.add_argument("--output-dir", default="/home/chengcli/data", help="Output directory")
    p.add_argument(
        "--debug-memory",
        action="store_true",
        help="Print rank-aware CPU/GPU memory diagnostics at major simulation phases",
    )
    p.add_argument(
        "--debug-memory-summary",
        action="store_true",
        help="Also print full torch.cuda.memory_summary() output with each memory diagnostic",
    )
    p.add_argument(
        "--restart-name",
        default="",
        help=(
            "Restart archive filename inside output dir (e.g. "
            "uranus.00005.restart or uranus.final.restart)"
        ),
    )
    p.add_argument(
        "--hang-diagnosis",
        action="store_true",
        help="Arm a watchdog around mesh.forward() that dumps Python stack traces if a stage stalls.",
    )
    p.add_argument(
        "--hang-timeout-sec",
        type=float,
        default=120.0,
        help="Seconds to wait before the hang watchdog emits a traceback dump.",
    )
    p.add_argument(
        "--hang-diagnosis-repeat",
        action="store_true",
        help="Keep emitting traceback dumps every timeout interval until the stalled call returns.",
    )
    p.add_argument(
        "--hang-sync-cuda",
        action="store_true",
        help="Force cuda synchronize() immediately before and after mesh.forward() to isolate queued GPU work.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_snapy_top_sponge(load_config(args.config))

    mesh, eos, device = create_models(args.config, config, args.output_dir)
    install_forward_debug(mesh)
    log_runtime_diagnostics(
        args=args,
        config=config,
        mesh=mesh,
        device=device,
    )

    if args.restart_name:
        mesh_vars, current_time = mesh.initialize_from_restart(args.restart_name)
    else:
        mesh_vars, current_time = initialize_isothermal(mesh, eos, config)

    for i, block_vars in enumerate(mesh_vars):
        for key, data in block_vars.items():
            if isinstance(data, torch.Tensor):
                print(f"block[{i}] {key}: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}")

    forcing_states = [build_tidal_forcing_state(block, config, device, eos) for block in mesh.blocks]
    block_diagnostics = [initialize_block_diagnostics(block) for block in mesh.blocks]
    for block, diagnostics in zip(mesh.blocks, block_diagnostics):
        register_user_output(block, diagnostics)
    log_block_physical_state(
        "post-initialize",
        cycle=int(mesh.blocks[0].cycle()) if mesh.blocks else -1,
        current_time=current_time,
        mesh=mesh,
        mesh_vars=mesh_vars,
        eos=eos,
        forcing_states=forcing_states,
    )

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
        device=device,
        debug_memory=args.debug_memory,
        debug_memory_summary=args.debug_memory_summary,
        hang_diagnosis=args.hang_diagnosis,
        hang_timeout_sec=args.hang_timeout_sec,
        hang_diagnosis_repeat=args.hang_diagnosis_repeat,
        hang_sync_cuda=args.hang_sync_cuda,
    )

    final_cycle = int(mesh.blocks[0].cycle()) if mesh.blocks else -1
    before_merged_outputs = snapshot_outputs(args.output_dir, basename, include_block=False)
    before_block_outputs = snapshot_outputs(args.output_dir, basename, include_block=True)
    log_phase(
        "before-finalize",
        cycle=final_cycle,
        current_time=current_time,
        mesh=mesh,
        mesh_vars=mesh_vars,
        device=device,
        debug_memory=args.debug_memory,
        debug_memory_summary=args.debug_memory_summary,
    )
    mesh.finalize(mesh_vars, current_time)
    log_phase(
        "after-finalize",
        cycle=final_cycle,
        current_time=current_time,
        mesh=mesh,
        mesh_vars=mesh_vars,
        device=device,
        debug_memory=args.debug_memory,
        debug_memory_summary=args.debug_memory_summary,
    )
    after_merged_outputs = snapshot_outputs(args.output_dir, basename, include_block=False)
    after_block_outputs = snapshot_outputs(args.output_dir, basename, include_block=True)
    report_output_changes(
        "after-finalize",
        before_merged=before_merged_outputs,
        after_merged=after_merged_outputs,
        before_block=before_block_outputs,
        after_block=after_block_outputs,
        cycle=final_cycle,
        current_time=current_time,
    )


if __name__ == "__main__":
    main()
