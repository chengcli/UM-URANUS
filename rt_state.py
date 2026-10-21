from pathlib import Path
from typing import Any

import yaml
import torch
import snapy

from kintera import ThermoX
from pyharp import Radiation, RadiationOptions
from snapy import MeshBlock, kICY, kIPR

class RTState:
    block: MeshBlock
    thermo_x: ThermoX

    radiation: Radiation
    insolation: Any
    last_heating: torch.Tensor
    lon: torch.Tensor
    lat: torch.Tensor
    nlyr: int
    ncol: int
    sw_albedo: float
    lw_albedo: float
    update_dt: float
    next_update: float = 0.
    key = "rt_heating"

    def __init__(self, block: MeshBlock, thermo_x: ThermoX,
                 config_path: Path):
        coord = block.module("coord")
        x2v = coord.buffer("x2v")
        x3v = coord.buffer("x3v")

        il, iu = coord.il(), coord.iu()
        self.nlyr = iu - il + 1

        layout = self.block.get_layout()
        rank = int(layout.options.rank())
        loc = layout.loc_of(rank)
        face_id = int(loc[2])
        face_name = coord.get_cs_face_name(face_id)

        beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
        self.lon, self.lat = coord.cs_ab_to_lonlat(face_name, alpha, beta)
        self.ncol = lon.numel()

        op_rad = RadiationOptions.from_yaml(str(config_path))
        radiation= Radiation(op_rad)

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            rt_config = config["radiative-transfer"]

        sw_albedo = rt_config["sw_albedo"]
        lw_albedo = rt_config["lw_albedo"]

        self.sw_albedo = torch.full((1, ncol), sw_albedo, dtype=self.lon.dtype, device=self.lon.device),
        self.lw_albedo = torch.full((1, ncol), lw_albedo, dtype=self.lon.dtype, device=self.lon.device),

        self.update_dt = rt_config["update_dt"]

    def update_heating(self, bvars: dict[str, torch.Tensor], current_time: float) -> torch.Tensor:
        if current_time < self.next_update:
            return

        hydro_w = bvars["hydro_w"]
        eos = self.block.module("hydro.eos"),
        thermo_y = self.block.module("hydro.eos.thermo")

        pres = hydro_w[kIPR]
        temp = eos.compute("W->T", (hydro_w,))
        xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))
        conc = self.thermo_x.compute("TPX->V", (temp, pres, xfrac))

        coord = self.block.coord
        il, iu = coord.il(), coord.iu()
        area = coord.face_area1()[..., il:iu + 2].reshape(ncol, nlyr + 1)
        volume = coord.cell_volume()[..., il:iu + 1].reshape(ncol, nlyr)

        temp_i = temp[..., il:iu + 1].reshape(self.ncol, self.nlyr)
        pres_i = pres[..., il:iu + 1].reshape(self.ncol, self.nlyr)
        conc_i = conc[..., il:iu + 1, :].reshape(self.ncol, self.nlyr, conc.shape[-1])

        time = torch.as_tensor(current_time, dtype=self.lon.dtype, device=self.lon.device)
        mu0, beam, _, _, _ = self.insolation(self.lon, self.lat, time)
        beam = beam.reshape(ncol)
        mu0 = mu0.reshape(ncol)

        bc = {
            "sw/fbeam": (beam / state.shortwave_weight).view(1, ncol),
            "sw/umu0": mu0,
            "sw/albedo": self.sw_albedo
            "lw/albedo": self.lw_albedo
        }

        atm = {"pres": pres_i, "temp": temp_i}
        net_flux, _, _ = self.radiation.forward(conc_i, dz, bc, atm)
        divergence = (area[:, 1:] * net_flux[:, 1:] - area[:, :-1] * net_flux[:, :-1]) / volume
        self.next_update = current_time + self.update_dt
        if not key in bvars.keys():
            bvars[key] = torch.zeros_like(bvars["hydro_u"])

        bvars[key][..., il, iu + 1] = (-divergence).reshape(self.lon.shape[0], self.lon.shape[1], nlyr)
