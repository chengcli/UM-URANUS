#!/usr/bin/env python3
"""Run a tidally locked Sub-Neptune GCM with pyharp Toon radiative transfer."""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import torch
import yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIV1, kICY, kIDN, kIPR
from kintera import Kinetics, KineticsOptions, ThermoX
from paddle import evolve_kinetics, setup_profile

import pyharp
from pyharp import Radiation, RadiationOptions

SECONDS_PER_DAY = 86400.0
torch.set_default_dtype(torch.float64)

@dataclass
class RadiativeTransferConfig:
    update_dt: float
    sw_surface_albedo: float
    lw_surface_albedo: float
    stellar_flux_nadir: float


@dataclass
class RadiativeTransferState:
    cfg: RadiativeTransferConfig

    grey_sw: torch.nn.Module
    grey_lw: torch.nn.Module
    toon_sw: torch.nn.Module
    toon_lw: torch.nn.Module

    cos_zenith_dayside: torch.Tensor  # (ny, nx)
    dz: torch.Tensor  # (nlyr,)
    area1: torch.Tensor  # (ncol, nlyr + 1)
    vol: torch.Tensor  # (ncol, nlyr)
    il: int
    iu: int
    last_heating: torch.Tensor  # (ny, nx, nlyr), W/m^3 == Pa/s
    next_update_time: float


class GreyOpacity(torch.nn.Module):
    def __init__(
        self,
        species_weights: list[float],
        kappa_a: float,
        kappa_b: float,
        kappa_cut: float,
        w0: float = 0.0,
        g: float = 0.0,
        nwave: int = 1,
        nmom: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "species_weights",
            torch.tensor(species_weights),
            persistent=True,
        )
        self.kappa_a = float(kappa_a)
        self.kappa_b = float(kappa_b)
        self.kappa_cut = float(kappa_cut)
        self.nwave = int(nwave)
        self.w0 = float(w0)
        self.g = float(g)
        self.nprop = 2 + int(nmom)  # (extinction, w0, g)

    def forward(self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        '''
        Compute grey opacity properties for each layer and column.

        Args:
            conc: (ncol, nlyr, nspecies) [mol/m^3]
            pres: (ncol, nlyr) [pa]
            temp: (ncol, nlyr) [K]

        Returns:
            prop: (nwave, ncol, nlyr, nprop) where nprop includes extinction [1/m],
            single scattering albedo, and asymmetry factor.
        '''

        ncol = conc.shape[0]
        nlyr = conc.shape[1]

        # extinction = rho * kappa(pres)
        rho = (conc * self.species_weights.view(1, 1, -1)).sum(dim=-1)

        kappa = self.kappa_a * torch.pow(pres, self.kappa_b)
        kappa = torch.clamp(kappa, min=self.kappa_cut)
        extinction = rho * kappa  # [1/m]

        out = torch.zeros(
            (self.nwave, ncol, nlyr, self.nprop),
            dtype=conc.dtype,
            device=conc.device,
        )

        out[..., 0] = extinction.unsqueeze(0)
        out[..., 1] = self.w0
        out[..., 2] = self.g

        return out


def load_config(path: str) -> dict[str, Any]:
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

    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    op_kin = KineticsOptions.from_yaml(config_file)
    kinet = Kinetics(op_kin)
    kinet.to(device)

    eos = block.module("hydro.eos")
    return block, eos, thermo_y, thermo_x, kinet, device


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


def initialize_atm(block: MeshBlock, config: dict[str, Any]) -> tuple[dict[str, torch.Tensor], float]:
    grav = -float(config["forcing"]["const-gravity"]["grav1"])
    problem = config["problem"]

    param: dict[str, float] = {
        "Ts": float(problem["Ts"]),
        "Ps": float(problem["Ps"]),
        "Tmin": float(problem.get("Tmin", problem["Ts"])),
        "grav": grav,
    }

    thermo_y = block.module("hydro.eos.thermo")
    for name in thermo_y.options.species():
        param[f"x{name}"] = float(problem.get(f"x{name}", 0.0))

    hydro_w = setup_profile(block, param, method="pseudo-adiabat")
    fill_initial_profile_ghosts(hydro_w)
    validate_initial_profile(hydro_w, param)

    # add random noise to IV1
    hydro_w[kIV1] += 1e-6 * torch.randn_like(hydro_w[kIV1])

    return block.initialize({"hydro_w": hydro_w})


def _resolve_local_face_name(block: MeshBlock) -> str:
    layout = snapy.distributed.get_layout()
    rank = int(snapy.distributed.get_rank())
    loc = layout.loc_of(rank)
    face_id = int(loc[2])
    return snapy.coord.get_cs_face_name(face_id)


def _build_local_cos_zenith(block: MeshBlock, config: dict[str, Any]) -> torch.Tensor:
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
    return torch.clamp(cos_zenith, min=0.0)


def _extract_species_weights_from_config(config: dict[str, Any]) -> list[float]:
    # Atomic masses [kg/mol] for elements used in current RT species definitions.
    atomic_mass = {
        "H": 1.00784e-3,
        "He": 4.002602e-3,
        "C": 12.0107e-3,
        "N": 14.0067e-3,
        "O": 15.999e-3,
        "S": 32.065e-3,
    }
    out: list[float] = []
    for sp in config["species"]:
        mw = 0.0
        for el, stoich in sp["composition"].items():
            if el not in atomic_mass:
                raise KeyError(f"Unsupported element '{el}' in species '{sp['name']}' for JIT opacity.")
            mw += float(stoich) * atomic_mass[el]
        out.append(mw)
    return out

def _parse_band_range(config: dict[str, Any], key: str) -> Tuple[float, float]:
    for band in config["bands"]:
        if band["name"] == key:
            return band["range"]
    raise ValueError(f"Band '{key}' not found in config 'bands' section.")


def create_grey_opacities(config: dict[str, Any]) -> Tuple[torch.nn.Module,
                                                           torch.nn.Module]:
    opacities = config.get("opacities", {})
    species_weights = _extract_species_weights_from_config(config)

    opacity_models = []
    for name, spec in opacities.items():
        files = spec.get("data", [])
        if not files:
            raise ValueError(f"Opacity '{name}' is type=jit but has no data files.")
        out_file = Path(files[0])

        nmom = int(spec.get("nmom", 1))
        params = spec.get("parameters", {})
        model = GreyOpacity(
            species_weights=species_weights,
            kappa_a=float(params["kappa_a"]),
            kappa_b=float(params["kappa_b"]),
            kappa_cut=float(params["kappa_cut"]),
            w0=float(params.get("w0", 0.0)),
            g=float(params.get("g", 0.0)),
            nwave=1,
            nmom=nmom,
        )
        #scripted = torch.jit.script(model)
        #scripted.save(str(out_file))
        opacity_models.append(model)
    return opacity_models

def create_toon_solvers(config: dict[str, Any]) -> Tuple[torch.nn.Module,
                                                         torch.nn.Module]:
    # shortwave solver
    op_sw = pyharp.ToonMcKay89Options()
    toon_sw = pyharp.ToonMcKay89(op_sw)

    # longwave solver
    op_lw = pyharp.ToonMcKay89Options()
    wave_lo, wave_hi = _parse_band_range(config, "lw")
    op_lw.wave_lower([wave_lo])
    op_lw.wave_upper([wave_hi])
    toon_lw = pyharp.ToonMcKay89(op_lw)

    return toon_sw, toon_lw

def build_rt_state(
    block: MeshBlock,
    config: dict[str, Any],
    config_file: str,
) -> RadiativeTransferState:
    rt_cfg_raw = config.get("radiative-transfer", {})
    cfg = RadiativeTransferConfig(
        update_dt=float(rt_cfg_raw.get("update_dt", 3600.0)),
        sw_surface_albedo=float(rt_cfg_raw.get("sw_surface_albedo", 0.0)),
        lw_surface_albedo=float(rt_cfg_raw.get("lw_surface_albedo", 0.0)),
        stellar_flux_nadir=float(rt_cfg_raw.get("stellar_flux_nadir", 1800.)),
    )

    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    nlyr = iu - il + 1
    ny = int(coord.buffer("x3v").numel())
    nx = int(coord.buffer("x2v").numel())
    ncol = ny * nx

    dz = coord.buffer("dx1f")[il : iu + 1]

    area1 = coord.face_area1()[..., il : iu + 2].view(ncol, nlyr + 1)
    vol = coord.cell_volume()[..., il : iu + 1].view(ncol, nlyr)
    area1 = area1.view(ncol, nlyr + 1)
    vol = vol.view(ncol, nlyr)

    cosz = _build_local_cos_zenith(block, config)
    last_heating = torch.zeros((ny, nx, nlyr), device=cosz.device)

    grey_sw, grey_lw = create_grey_opacities(config)
    toon_sw, toon_lw = create_toon_solvers(config)

    grey_sw.to(cosz.device)
    grey_lw.to(cosz.device)

    toon_sw.to(cosz.device)
    toon_lw.to(cosz.device)

    return RadiativeTransferState(
        cfg=cfg,
        grey_sw=grey_sw,
        grey_lw=grey_lw,
        toon_sw=toon_sw,
        toon_lw=toon_lw,
        cos_zenith_dayside=cosz,
        dz=dz,
        area1=area1,
        vol=vol,
        il=il,
        iu=iu,
        last_heating=last_heating,
        next_update_time=0.0,
    )

def _compute_shortwave_rt(rt_state: RadiativeTransferState, 
                          conc_i: torch.Tensor,
                          pres_i: torch.Tensor,
                          temp_i: torch.Tensor) -> torch.Tensor:
    ncol = conc_i.shape[0]

    bc: dict[str, torch.Tensor] = {
        f"fbeam": (
            rt_state.cfg.stellar_flux_nadir
            * torch.ones((1, ncol), device=conc_i.device) # (nwave, ncol)
        ),
        f"umu0": rt_state.cos_zenith_dayside.view(ncol), # (ncol,)
        f"albedo": (
            rt_state.cfg.sw_surface_albedo
            * torch.ones((1, ncol), device=conc_i.device) # (nwave, ncol)
        ),
    }

    # (ncol, nlyr, nspecies) -> (nwave, ncol, nlyr, nprop)
    prop = rt_state.grey_sw(conc_i, pres_i, temp_i)

    # extinction [1/m] -> optical thickness [unitless]
    prop *= rt_state.dz.view(1, 1, -1, 1)

    result = rt_state.toon_sw(prop, **bc).sum(0)  # (ncol, nlyr+1, 2)

    # set net flux to zero in layers with zero cos(zenith) to avoid numerical issues
    # with Toon solver output
    zero_cosz_mask = (bc["umu0"] == 0.0).view(ncol, 1, 1).expand_as(result)
    result[zero_cosz_mask] = 0.0

    # net flux = upward - downward
    return result[...,0] - result[...,1]

def _compute_longwave_rt(rt_state: RadiativeTransferState,
                         conc_i: torch.Tensor,
                         pres_i: torch.Tensor,
                         temp_i: torch.Tensor) -> None:
    ncol, nlyr, _ = conc_i.shape

    bc: dict[str, torch.Tensor] = {
        f"albedo": (
            rt_state.cfg.lw_surface_albedo
            * torch.ones((1, ncol), device=conc_i.device) # (nwave, ncol)
        ),
    }

    # (ncol, nlyr, nspecies) -> (nwave, ncol, nlyr, nprop)
    prop = rt_state.grey_lw(conc_i, pres_i, temp_i)

    # extinction [1/m] -> optical thickness [unitless]
    prop *= rt_state.dz.view(1, 1, -1, 1)

    # temperature at layer interfaces for thermal emission
    temf = torch.zeros((ncol, nlyr + 1), device=conc_i.device)

    temf[:, 0] = 2 * temp_i[:, 0] - temp_i[:, 1]  # extrapolate bottom interface
    temf[:, 1:-1] = 0.5 * (temp_i[:, :-1] + temp_i[:, 1:]) # average for interior
    temf[:, -1] = 2 * temp_i[:, -1] - temp_i[:, -2]  # extrapolate top interface

    result = rt_state.toon_lw(prop, temf=temf, **bc).sum(0)  # (ncol, nlyr+1, 2)

    # net flux = upward - downward
    return result[...,0] - result[...,1]

def _compute_rt_heating(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    block_vars: dict[str, torch.Tensor],
    rt_state: RadiativeTransferState,
) -> torch.Tensor:
    hydro_w = block_vars["hydro_w"]

    temp = eos.compute("W->T", (hydro_w,))
    pres = hydro_w[kIPR]
    xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))

    il, iu = rt_state.il, rt_state.iu
    nlyr = iu - il + 1
    ny, nx = rt_state.cos_zenith_dayside.shape
    ncol = ny * nx

    temp_i = temp[..., il : iu + 1].view(ncol, nlyr)
    pres_i = pres[..., il : iu + 1].view(ncol, nlyr)
    conc_i = conc[..., il : iu + 1, :].view(ncol, nlyr, conc.shape[-1])
    
    net_flux_sw = _compute_shortwave_rt(rt_state, conc_i, pres_i, temp_i)
    net_flux_lw = _compute_longwave_rt(rt_state, conc_i, pres_i, temp_i)

    net_flux = net_flux_sw + net_flux_lw # (ncol, nlyr+1)

    # Volumetric heating [W/m^3] = -div(F) on a spherical shell using face areas.
    div_f = (
        rt_state.area1[:, 1:] * net_flux[:, 1:] - rt_state.area1[:, :-1] * net_flux[:, :-1]
    ) / rt_state.vol
    heating = -div_f

    # (ncol, nlyr) -> (ny, nx, nlyr)
    return heating.view(ny, nx, nlyr)


def update_rt_tendency_if_needed(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    rt_state: RadiativeTransferState,
) -> None:
    if current_time + 1.0e-12 < rt_state.next_update_time:
        return

    rt_state.last_heating = _compute_rt_heating(block, eos, thermo_y, thermo_x, block_vars, rt_state)
    rt_state.next_update_time = current_time + max(rt_state.cfg.update_dt, 0.0)


def apply_rt_forcing(block_vars: dict[str, torch.Tensor], rt_state: RadiativeTransferState, dt: float) -> None:
    block_vars["hydro_u"][kIPR, ..., rt_state.il : rt_state.iu + 1] += rt_state.last_heating * dt


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
                "python sub_neptune_rt/run_sub_neptune_rt.py "
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
    thermo_y,
    thermo_x: ThermoX,
    kinet: Kinetics,
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    tlim: float,
    rt_state: RadiativeTransferState,
    config_file: str,
    output_dir: str,
    basename: str,
) -> tuple[dict[str, torch.Tensor], float]:
    block.options.intg().tlim(tlim)

    next_checkpoint_day = int(current_time // (10.0 * SECONDS_PER_DAY)) * 10 + 10
    checkpoint_dir = Path(output_dir) / "restart_checkpoints"

    update_rt_tendency_if_needed(block, eos, thermo_y, thermo_x, block_vars, current_time, rt_state)
    block.make_outputs(block_vars, current_time)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            update_rt_tendency_if_needed(block, eos, thermo_y, thermo_x, block_vars, current_time, rt_state)
            apply_rt_forcing(block_vars, rt_state, dt)

        err = block.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        del_rho = evolve_kinetics(block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt)
        block_vars["hydro_u"][kICY:] += del_rho

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
    p = argparse.ArgumentParser(description="Run Sub-Neptune simulation with Toon radiative transfer.")
    p.add_argument("-c", "--config", required=True, help="YAML configuration file")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument(
        "--restart-name",
        default="",
        help=(
            "Restart archive filename inside output dir (e.g. "
            "sub_neptune_rt_doublegrey.00005.restart or sub_neptune_rt_doublegrey.final.restart)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    block, eos, thermo_y, thermo_x, kinet, device = create_models(args.config, args.output_dir)

    if args.restart_name:
        block_vars, current_time = block.initialize_from_restart(args.restart_name)
    else:
        block_vars, current_time = initialize_atm(block, config)

    rt_state = build_rt_state(
        block=block,
        config=config,
        config_file=args.config,
    )

    print(
        "RT forcing summary:",
        f"stellar_flux_nadir={rt_state.cfg.stellar_flux_nadir:.3f} W/m^2,",
        f"rt_update_dt={rt_state.cfg.update_dt:.1f} s,",
    )

    tlim = float(config["integration"]["tlim"])
    basename = Path(args.config).stem
    block_vars, current_time = run_simulation(
        block=block,
        eos=eos,
        thermo_y=thermo_y,
        thermo_x=thermo_x,
        kinet=kinet,
        block_vars=block_vars,
        current_time=current_time,
        tlim=tlim,
        rt_state=rt_state,
        config_file=args.config,
        output_dir=args.output_dir,
        basename=basename,
    )

    block.finalize(block_vars, current_time)


if __name__ == "__main__":
    main()
