from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import snapy
from pyharp import Radiation, RadiationOptions
from snapy import MeshBlock, kICY, kIPR


def _face_lon_lat(block: MeshBlock, local_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    coord = block.module("coord")
    beta, alpha = torch.meshgrid(coord.buffer("x3v"), coord.buffer("x2v"), indexing="ij")
    layout = block.get_layout()
    process_rank = int(snapy.distributed.get_rank())
    global_block = process_rank * int(layout.options.blocks_per_process()) + local_index
    face = int(layout.loc_of(global_block)[2])
    return snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face), alpha, beta)


def mask_nightside_visible_flux(visible_flux: torch.Tensor, mu0: torch.Tensor) -> torch.Tensor:
    return torch.where((mu0 > 0.0).view(-1, 1), visible_flux, torch.zeros_like(visible_flux))

@dataclass
class RTState:
    vis_radiation: Radiation
    ir_radiation: Radiation
    orbit: torch.jit.ScriptModule
    lon: torch.Tensor
    lat: torch.Tensor
    dz: torch.Tensor
    area: torch.Tensor
    volume: torch.Tensor
    il: int
    iu: int
    shortwave_weight: float
    last_heating: torch.Tensor
    next_update: float
    update_dt: float
    sw_albedo: float
    lw_albedo: float

def build_rt_state(block: MeshBlock, local_index: int, config: dict[str, Any], config_path: Path, orbit_path: Path, device: torch.device) -> RTState:
    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    nlyr = iu - il + 1
    lon, lat = _face_lon_lat(block, local_index)
    ncol = lon.numel()
    options = RadiationOptions.from_yaml(str(config_path))
    sw_weight = 1.0
    for band in options.bands():
        spec = next(item for item in config["bands"] if item["name"] == band.name())
        lower, upper = spec["range"]
        band.ncol(ncol)
        band.nlyr(nlyr)
        band.wavenumber([0.5 * (lower + upper)])
        band.weight([upper - lower])
        if band.name() == "vis":
            sw_weight = upper - lower
    bands = {band.name(): band for band in options.bands()}
    vis_options = RadiationOptions()
    vis_options.bands([bands["vis"]])
    ir_options = RadiationOptions()
    ir_options.bands([bands["ir"]])
    vis_radiation = Radiation(vis_options)
    ir_radiation = Radiation(ir_options)
    vis_radiation.to(device)
    ir_radiation.to(device)
    area = coord.face_area1()[..., il:iu + 2].reshape(ncol, nlyr + 1)
    volume = coord.cell_volume()[..., il:iu + 1].reshape(ncol, nlyr)
    rt = config["radiative-transfer"]
    return RTState(
        vis_radiation, ir_radiation, torch.jit.load(str(orbit_path), map_location=device), lon.to(device), lat.to(device),
        coord.buffer("dx1f")[il:iu + 1].to(device), area.to(device), volume.to(device), il, iu, sw_weight,
        torch.zeros((lon.shape[0], lon.shape[1], nlyr), dtype=lon.dtype, device=device), 0.0,
        float(rt["update_dt"]), float(rt["vis_surface_albedo"]), float(rt["ir_surface_albedo"]),
    )

def compute_heating(block_vars: dict[str, torch.Tensor], eos: Any, thermo_y: Any, thermo_x: Any, state: RTState, current_time: float) -> torch.Tensor:
    hydro_w = block_vars["hydro_w"]
    temp = eos.compute("W->T", (hydro_w,))
    pres = hydro_w[kIPR]
    xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))
    ncol, nlyr = state.lon.numel(), state.iu - state.il + 1
    temp_i = temp[..., state.il:state.iu + 1].reshape(ncol, nlyr).to(torch.float64)
    pres_i = pres[..., state.il:state.iu + 1].reshape(ncol, nlyr).to(torch.float64)
    conc_i = conc[..., state.il:state.iu + 1, :].reshape(ncol, nlyr, conc.shape[-1]).to(torch.float64)
    time = torch.as_tensor(current_time, dtype=state.lon.dtype, device=state.lon.device)
    mu0, beam, _, _, _ = state.orbit.insolation(state.lon, state.lat, time)
    beam = beam.reshape(ncol)
    mu0 = mu0.reshape(ncol)
    safe_mu0 = torch.where(mu0 > 0.0, mu0, torch.ones_like(mu0))
    vis_bc = {
        "vis/fbeam": (beam / state.shortwave_weight).view(1, ncol),
        "vis/umu0": safe_mu0,
        "vis/albedo": torch.full((1, ncol), state.sw_albedo, dtype=torch.float64, device=temp_i.device),
    }
    ir_bc = {
        "ir/albedo": torch.full((1, ncol), state.lw_albedo, dtype=torch.float64, device=temp_i.device),
    }
    atmosphere = {"pres": pres_i, "temp": temp_i}
    vis_flux, _, _ = state.vis_radiation.forward(
        conc_i, state.dz.to(torch.float64), vis_bc, atmosphere
    )

    vis_flux = mask_nightside_visible_flux(vis_flux, mu0)
    ir_flux, _, _ = state.ir_radiation.forward(
        conc_i, state.dz.to(torch.float64), ir_bc, atmosphere
    )
    net_flux = vis_flux + ir_flux
    divergence = (state.area[:, 1:] * net_flux[:, 1:] - state.area[:, :-1] * net_flux[:, :-1]) / state.volume
    return (-divergence).reshape(state.lon.shape[0], state.lon.shape[1], nlyr).to(hydro_w.dtype)
