#!/usr/bin/env python3
"""Moist Uranus GCM with scripted grey gas, cloud, and orbital forcing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

import pyharp
from pyharp import Radiation, RadiationOptions
from kintera import Kinetics, KineticsOptions, ThermoX
from paddle import evolve_kinetics, setup_profile
import snapy
from snapy import Mesh, MeshOptions, kICY, kIPR, kIV1


AU = 1.495978707e11


class GasOpacity(torch.nn.Module):
    """Pressure-power-law mass opacity converted to extinction per length."""

    def __init__(self, species_indices: list[int], molecular_weights: list[float], kappa_ref: float, p_ref: float, exponent: float):
        super().__init__()
        self.register_buffer("species_indices", torch.tensor(species_indices, dtype=torch.int64))
        self.register_buffer("molecular_weights", torch.tensor(molecular_weights, dtype=torch.float64))
        self.kappa_ref = float(kappa_ref)
        self.p_ref = float(p_ref)
        self.exponent = float(exponent)

    def forward(self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        del temp
        indices = self.species_indices.to(device=conc.device)
        gas_conc = torch.index_select(conc, -1, indices)
        weights = self.molecular_weights.to(dtype=conc.dtype, device=conc.device)
        density = (gas_conc * weights.view(1, 1, -1)).sum(-1)
        kappa = self.kappa_ref * torch.pow(torch.clamp_min(pres, 0.0) / self.p_ref, self.exponent)
        output = torch.zeros((1, conc.shape[0], conc.shape[1], 3), dtype=conc.dtype, device=conc.device)
        output[0, :, :, 0] = density * kappa
        return output


class CloudOpacity(torch.nn.Module):
    """Cloud extinction proportional to condensate mass density."""

    def __init__(self, species_index: int, molecular_weight: float, mass_extinction: float, single_scattering_albedo: float, asymmetry: float):
        super().__init__()
        self.species_index = int(species_index)
        self.molecular_weight = float(molecular_weight)
        self.mass_extinction = float(mass_extinction)
        self.single_scattering_albedo = float(single_scattering_albedo)
        self.asymmetry = float(asymmetry)

    def forward(self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        del pres, temp
        density = conc[:, :, self.species_index] * self.molecular_weight
        output = torch.zeros((1, conc.shape[0], conc.shape[1], 3), dtype=conc.dtype, device=conc.device)
        output[0, :, :, 0] = density * self.mass_extinction
        output[0, :, :, 1] = self.single_scattering_albedo
        output[0, :, :, 2] = self.asymmetry
        return output


class OrbitalForcing(torch.nn.Module):
    """Keplerian illumination plus snapy radiative stage forcing."""

    def __init__(
        self,
        stellar_luminosity: float,
        semi_major_axis: float,
        eccentricity: float,
        obliquity: float,
        rotation_rate: float,
        orbital_period: float,
        mean_anomaly_epoch: float = 0.0,
        prime_meridian_epoch: float = 0.0,
    ) -> None:
        super().__init__()
        self.stellar_luminosity = float(stellar_luminosity)
        self.semi_major_axis = float(semi_major_axis)
        self.eccentricity = float(eccentricity)
        self.obliquity = float(obliquity)
        self.rotation_rate = float(rotation_rate)
        self.orbital_period = float(orbital_period)
        self.mean_anomaly_epoch = float(mean_anomaly_epoch)
        self.prime_meridian_epoch = float(prime_meridian_epoch)
        self.energy_index = 4

    def forward(
        self,
        variables: dict[str, torch.Tensor],
        dt: float,
        stage: int,
    ) -> dict[str, torch.Tensor]:
        del stage
        hydro_du = torch.zeros_like(variables["hydro_u"])
        hydro_du[self.energy_index] = variables["rt_heating"] * dt
        return {"hydro_du": hydro_du}

    @torch.jit.export
    def insolation(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        two_pi = 2.0 * math.pi
        mean_anomaly = self.mean_anomaly_epoch + two_pi * time / self.orbital_period
        eccentric_anomaly = mean_anomaly
        for _ in range(8):
            eccentric_anomaly = eccentric_anomaly - (
                eccentric_anomaly - self.eccentricity * torch.sin(eccentric_anomaly) - mean_anomaly
            ) / (1.0 - self.eccentricity * torch.cos(eccentric_anomaly))
        distance = self.semi_major_axis * (1.0 - self.eccentricity * torch.cos(eccentric_anomaly))
        true_anomaly = 2.0 * torch.atan2(
            math.sqrt(1.0 + self.eccentricity) * torch.sin(0.5 * eccentric_anomaly),
            math.sqrt(1.0 - self.eccentricity) * torch.cos(0.5 * eccentric_anomaly),
        )
        obliquity = torch.as_tensor(self.obliquity, dtype=lat.dtype, device=lat.device)
        subsolar_lat = torch.asin(torch.sin(obliquity) * torch.sin(true_anomaly))
        stellar_right_ascension = torch.atan2(
            torch.cos(obliquity) * torch.sin(true_anomaly),
            torch.cos(true_anomaly),
        )
        subsolar_lon = stellar_right_ascension - self.rotation_rate * time - self.prime_meridian_epoch
        raw_mu0 = torch.sin(lat) * torch.sin(subsolar_lat) + torch.cos(lat) * torch.cos(subsolar_lat) * torch.cos(lon - subsolar_lon)
        dayside = raw_mu0 > 0.0
        mu0 = torch.where(dayside, raw_mu0, torch.zeros_like(raw_mu0))
        irradiance = self.stellar_luminosity / (4.0 * math.pi * distance * distance)
        beam = torch.where(dayside, torch.ones_like(raw_mu0) * irradiance, torch.zeros_like(raw_mu0))
        return mu0, beam, distance, subsolar_lon, subsolar_lat


@dataclass
class RTState:
    visible_radiation: Radiation
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


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    include = config.pop("include", None)
    if include is None:
        return config
    base = load_config(path.parent / include)

    def merge(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                target[key] = merge(target[key], value)
            else:
                target[key] = value
        return target

    return merge(base, config)


def molecular_weights(config: dict[str, Any]) -> dict[str, float]:
    atomic = {"H": 1.00784e-3, "He": 4.002602e-3, "C": 12.0107e-3, "S": 32.065e-3}
    return {
        item["name"]: sum(atomic[element] * float(number) for element, number in item["composition"].items())
        for item in config["species"]
    }


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _save_if_stale(module: torch.nn.Module, path: Path, signature: Any, rebuild: bool) -> None:
    marker = path.with_suffix(path.suffix + ".sha256")
    digest = _fingerprint(signature)
    if rebuild or not path.exists() or not marker.exists() or marker.read_text().strip() != digest:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.script(module.eval()).save(str(path))
        marker.write_text(digest + "\n")


def ensure_torchscripts(config: dict[str, Any], config_path: Path, rebuild: bool = False) -> Path:
    weights = molecular_weights(config)
    species_indices = {item["name"]: index for index, item in enumerate(config["species"])}
    for name, opacity in config["opacities"].items():
        params = opacity["parameters"]
        path = (config_path.parent / opacity["data"][0]).resolve()
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
        _save_if_stale(
            module,
            path,
            {"implementation": 2, "name": name, "species": species, "parameters": params},
            rebuild,
        )

    orbit_cfg = config["orbit"]
    orbit_path = (config_path.parent / orbit_cfg["data"]).resolve()
    module = OrbitalForcing(
        orbit_cfg["stellar_luminosity"], orbit_cfg["semi_major_axis_au"] * AU,
        orbit_cfg["eccentricity"], math.radians(orbit_cfg["obliquity_deg"]), orbit_cfg["rotation_rate"],
        orbit_cfg["orbital_period"], math.radians(orbit_cfg.get("mean_anomaly_epoch_deg", 0.0)),
        math.radians(orbit_cfg.get("prime_meridian_epoch_deg", 0.0)),
    )
    _save_if_stale(module, orbit_path, {"implementation": 2, "orbit": orbit_cfg}, rebuild)
    return orbit_path


def _face_lon_lat(block: Any, local_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    coord = block.module("coord")
    beta, alpha = torch.meshgrid(coord.buffer("x3v"), coord.buffer("x2v"), indexing="ij")
    layout = block.get_layout()
    process_rank = int(snapy.distributed.get_rank())
    global_block = process_rank * int(layout.options.blocks_per_process()) + local_index
    face = int(layout.loc_of(global_block)[2])
    return snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face), alpha, beta)


def build_rt_state(block: Any, local_index: int, config: dict[str, Any], config_path: Path, orbit_path: Path, device: torch.device) -> RTState:
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
        if band.name() == "visible":
            sw_weight = upper - lower
    bands = {band.name(): band for band in options.bands()}
    visible_options = RadiationOptions()
    visible_options.bands([bands["visible"]])
    ir_options = RadiationOptions()
    ir_options.bands([bands["ir"]])
    visible_radiation = Radiation(visible_options)
    ir_radiation = Radiation(ir_options)
    visible_radiation.to(device)
    ir_radiation.to(device)
    area = coord.face_area1()[..., il:iu + 2].reshape(ncol, nlyr + 1)
    volume = coord.cell_volume()[..., il:iu + 1].reshape(ncol, nlyr)
    rt = config["radiative-transfer"]
    return RTState(
        visible_radiation, ir_radiation, torch.jit.load(str(orbit_path), map_location=device), lon.to(device), lat.to(device),
        coord.buffer("dx1f")[il:iu + 1].to(device), area.to(device), volume.to(device), il, iu, sw_weight,
        torch.zeros((lon.shape[0], lon.shape[1], nlyr), dtype=lon.dtype, device=device), 0.0,
        float(rt["update_dt"]), float(rt["visible_surface_albedo"]), float(rt["ir_surface_albedo"]),
    )


def mask_nightside_visible_flux(visible_flux: torch.Tensor, mu0: torch.Tensor) -> torch.Tensor:
    """Force all visible interface fluxes to finite zero off the dayside."""
    return torch.where((mu0 > 0.0).view(-1, 1), visible_flux, torch.zeros_like(visible_flux))


def sync_primitives(variables: dict[str, torch.Tensor], eos: Any) -> None:
    variables["hydro_w"] = eos.compute("U->W", (variables["hydro_u"], variables["hydro_w"]))


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
    visible_bc = {
        "visible/fbeam": (beam / state.shortwave_weight).view(1, ncol),
        "visible/umu0": safe_mu0,
        "visible/albedo": torch.full((1, ncol), state.sw_albedo, dtype=torch.float64, device=temp_i.device),
    }
    ir_bc = {
        "ir/albedo": torch.full((1, ncol), state.lw_albedo, dtype=torch.float64, device=temp_i.device),
    }
    atmosphere = {"pres": pres_i, "temp": temp_i}
    visible_flux, _, _ = state.visible_radiation.forward(
        conc_i, state.dz.to(torch.float64), visible_bc, atmosphere
    )
    visible_flux = mask_nightside_visible_flux(visible_flux, mu0)
    ir_flux, _, _ = state.ir_radiation.forward(
        conc_i, state.dz.to(torch.float64), ir_bc, atmosphere
    )
    net_flux = visible_flux + ir_flux
    divergence = (state.area[:, 1:] * net_flux[:, 1:] - state.area[:, :-1] * net_flux[:, :-1]) / state.volume
    return (-divergence).reshape(state.lon.shape[0], state.lon.shape[1], nlyr).to(hydro_w.dtype)


def run(args: argparse.Namespace) -> None:
    source_config_path = Path(args.config).resolve()
    config = load_config(source_config_path)
    orbit_path = ensure_torchscripts(config, source_config_path, args.rebuild_jit)
    pyharp.add_resource_directory(str(source_config_path.parent), prepend=True)
    config_path = source_config_path
    if "include" in yaml.safe_load(source_config_path.read_text()):
        resolved_dir = Path(args.output_dir).resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        config_path = resolved_dir / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    options = MeshOptions.from_yaml(str(config_path))
    options.block().output_dir(args.output_dir)
    mesh = Mesh(options)
    device = torch.device(options.device_str())
    mesh.to(device)
    mesh.set_user_stage_forcings([str(orbit_path)])
    thermos = []
    for block in mesh.blocks:
        thermo_y = block.module("hydro.eos.thermo")
        thermo_x = ThermoX(thermo_y.options)
        thermo_x.to(device)
        thermos.append((block.module("hydro.eos"), thermo_y, thermo_x))
    kinetics = Kinetics(KineticsOptions.from_yaml(str(config_path)))
    kinetics.to(device)
    if args.restart:
        block_vars, current_time = mesh.initialize_from_restart(args.restart)
    else:
        params = {"Ts": float(config["problem"]["Ts"]), "Ps": float(config["problem"]["Ps"]),
                  "Tmin": float(config["problem"]["Tmin"]), "grav": -float(config["forcing"]["const-gravity"]["grav1"])}
        for species in thermos[0][1].options.species():
            params[f"x{species}"] = float(config["problem"].get(f"x{species}", 0.0))
        initial = []
        for block in mesh.blocks:
            hydro_w = setup_profile(block, params, method="pseudo-adiabat")
            hydro_w[kIV1] += 1.0e-6 * torch.randn_like(hydro_w[kIV1])
            initial.append({"hydro_w": hydro_w})
        block_vars, current_time = mesh.initialize(initial)
    states = [build_rt_state(block, index, config, config_path, orbit_path, device) for index, block in enumerate(mesh.blocks)]
    for variables in block_vars:
        variables["rt_heating"] = torch.zeros_like(variables["hydro_u"][kIPR])
    integrator = mesh.module("block0.intg")
    cycle = 0
    if not args.restart:
        mesh.make_outputs(block_vars, current_time)
    while not integrator.stop(cycle, current_time):
        if args.max_cycles is not None and cycle >= args.max_cycles:
            break
        cycle += 1
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)
        for variables, (eos, _, _) in zip(block_vars, thermos):
            sync_primitives(variables, eos)
        for stage in range(len(integrator.stages)):
            for variables, thermo, state in zip(block_vars, thermos, states):
                if current_time + 1.0e-12 >= state.next_update:
                    state.last_heating = compute_heating(variables, *thermo, state, current_time)
                    variables["rt_heating"].zero_()
                    variables["rt_heating"][..., state.il:state.iu + 1] = state.last_heating
                    state.next_update = current_time + state.update_dt
            mesh.forward(block_vars, dt, stage)
        error = mesh.check_redo(block_vars)
        if error > 0:
            continue
        if error < 0:
            break
        for variables, (eos, thermo_y, thermo_x) in zip(block_vars, thermos):
            sync_primitives(variables, eos)
            variables["hydro_u"][kICY:] += evolve_kinetics(
                variables["hydro_w"], eos, thermo_x, thermo_y, kinetics, dt
            )
            sync_primitives(variables, eos)
        current_time += dt
        mesh.make_outputs(block_vars, current_time)
    mesh.finalize(block_vars, current_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="uranus.yaml")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("-r", "--restart", default="")
    parser.add_argument("--rebuild-jit", action="store_true")
    parser.add_argument("--max-cycles", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
