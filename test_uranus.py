import math
from pathlib import Path

import torch
import yaml
import snapy
from paddle import setup_profile

from run_uranus import (
    AU,
    CloudOpacity,
    GasOpacity,
    OrbitalForcing,
    ensure_torchscripts,
    mask_nightside_visible_flux,
)


ROOT = Path(__file__).parent


def test_gas_opacity_pressure_power_law():
    model = GasOpacity([0], [2.0e-3], 0.5, 1.0e5, 1.0)
    conc = torch.tensor([[[1.0, 1000.0], [1.0, 2000.0]]], dtype=torch.float64)
    pressure = torch.tensor([[1.0e5, 2.0e5]], dtype=torch.float64)
    prop = model(conc, pressure, torch.ones_like(pressure))
    torch.testing.assert_close(prop[0, 0, :, 0], torch.tensor([1.0e-3, 2.0e-3]))
    assert torch.count_nonzero(prop[..., 1:]) == 0


def test_cloud_opacity_uses_only_condensate_density():
    model = CloudOpacity(1, 16.0e-3, 10.0, 0.9, 0.7)
    conc = torch.tensor([[[1000.0, 0.0], [2000.0, 2.0]]], dtype=torch.float64)
    prop = model(conc, torch.ones((1, 2)), torch.ones((1, 2)))
    torch.testing.assert_close(prop[0, 0, :, 0], torch.tensor([0.0, 0.32]))
    assert torch.all(prop[..., 1] == 0.9)
    assert torch.all(prop[..., 2] == 0.7)


def make_orbit(eccentricity=0.0):
    return OrbitalForcing(3.828e26, 19.191 * AU, eccentricity, math.radians(97.77), 1.0119e-4, 2.65121e9)


def test_orbit_sets_nightside_beam_exactly_zero():
    model = torch.jit.script(make_orbit())
    lon = torch.tensor([0.0, math.pi], dtype=torch.float64)
    lat = torch.zeros(2, dtype=torch.float64)
    mu0, beam, *_ = model.insolation(lon, lat, torch.tensor(0.0, dtype=torch.float64))
    assert mu0[0] == 1.0
    assert beam[0] > 0.0
    assert mu0[1] == 0.0
    assert beam[1] == 0.0


def test_orbit_inverse_square_flux_ratio():
    model = torch.jit.script(make_orbit(0.1))
    lon = torch.zeros(1, dtype=torch.float64)
    lat = torch.zeros(1, dtype=torch.float64)
    _, perihelion, *_ = model.insolation(lon, lat, torch.tensor(0.0, dtype=torch.float64))
    _, aphelion, *_ = model.insolation(lon, lat, torch.tensor(0.5 * 2.65121e9, dtype=torch.float64))
    expected = ((1.0 + 0.1) / (1.0 - 0.1)) ** 2
    assert math.isclose(float(perihelion / aphelion), expected, rel_tol=1.0e-10)


def test_nightside_visible_mask_removes_nan_without_masking_ir():
    visible = torch.tensor([[1.0, 2.0], [float("nan"), float("nan")]])
    infrared = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    masked = mask_nightside_visible_flux(visible, torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(masked, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))
    torch.testing.assert_close(masked + infrared, torch.tensor([[4.0, 6.0], [5.0, 6.0]]))


def test_orbital_stage_forcing_returns_energy_increment():
    model = torch.jit.script(make_orbit())
    hydro_u = torch.zeros((7, 2, 2, 3), dtype=torch.float64)
    heating = torch.full((2, 2, 3), 2.0, dtype=torch.float64)
    result = model({"hydro_u": hydro_u, "rt_heating": heating}, 3.0, 0)
    torch.testing.assert_close(result["hydro_du"][snapy.kIPR], heating * 3.0)
    assert torch.count_nonzero(result["hydro_du"][:snapy.kIPR]) == 0


def test_high_obliquity_subsolar_longitude_uses_right_ascension():
    obliquity = math.radians(97.77)
    period = 1000.0
    model = torch.jit.script(
        OrbitalForcing(1.0, 1.0, 0.0, obliquity, 0.0, period)
    )
    _, _, _, subsolar_lon, _ = model.insolation(
        torch.zeros(1), torch.zeros(1), torch.tensor(period / 4.0)
    )
    expected = math.atan2(math.cos(obliquity), 0.0)
    assert math.isclose(float(subsolar_lon), expected, abs_tol=1.0e-6)


def test_builds_seven_reloadable_torchscripts(tmp_path):
    config = yaml.safe_load((ROOT / "uranus.yaml").read_text())
    for opacity in config["opacities"].values():
        opacity["data"] = [str(tmp_path / Path(opacity["data"][0]).name)]
    config["orbit"]["data"] = str(tmp_path / "orbital_forcing.pt")
    orbit = ensure_torchscripts(config, ROOT / "uranus.yaml", rebuild=True)
    artifacts = sorted(tmp_path.glob("*.pt"))
    assert len(artifacts) == 7
    for artifact in artifacts:
        torch.jit.load(str(artifact))
    assert orbit == tmp_path / "orbital_forcing.pt"


def test_production_configuration():
    config = yaml.safe_load((ROOT / "uranus.yaml").read_text())
    assert config["geometry"]["cells"] == {"nx1": 64, "nx2": 96, "nx3": 96, "nghost": 3}
    assert config["distribute"]["blocks_per_process"] == 3
    assert config["forcing"]["bot-heat"]["flux"] == 0.042
    assert config["problem"]["Ps"] == 2.062e6
    assert len(config["opacities"]) == 6
    gas_species = {"dry", "CH4", "H2S"}
    assert set(config["opacities"]["gas-visible"]["species"]) == gas_species
    assert set(config["opacities"]["gas-ir"]["species"]) == gas_species
    ir_band = next(band for band in config["bands"] if band["name"] == "ir")
    assert ir_band["flags"] == "planck"
    assert math.isclose(
        float(config["geometry"]["bounds"]["x1max"])
        - float(config["geometry"]["bounds"]["x1min"]),
        241700.0,
    )


def test_initialized_physical_centers_cover_requested_pressure_range(tmp_path):
    config = yaml.safe_load((ROOT / "uranus.yaml").read_text())
    config["distribute"].update(backend="gloo", blocks_per_process=6)
    config["geometry"]["cells"].update(nx2=2, nx3=2)
    config["outputs"] = []
    config_path = tmp_path / "pressure_check.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    mesh = snapy.Mesh(snapy.MeshOptions.from_yaml(str(config_path)))
    block = mesh.blocks[0]
    params = {"Ts": 160.0, "Ps": 2.062e6, "Tmin": 55.0, "grav": 8.69}
    for species in block.module("hydro.eos.thermo").options.species():
        params[f"x{species}"] = float(config["problem"].get(f"x{species}", 0.0))
    hydro_w = setup_profile(block, params, method="pseudo-adiabat")
    coord = block.module("coord")
    pressure = hydro_w[snapy.kIPR, 3, 3, coord.il():coord.iu() + 1] / 1.0e5
    assert math.isclose(float(pressure[0]), 20.0, rel_tol=0.01)
    assert math.isclose(float(pressure[-1]), 0.01, rel_tol=0.15)
