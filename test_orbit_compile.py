import math
from pathlib import Path

import pytest
import torch
import yaml

from run_uranus import compile_orbit, make_orbit


ROOT = Path(__file__).parent


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_compiled_orbit_matches_eager_on_two_gpus():
    config = yaml.safe_load((ROOT / "uranus.yaml").read_text())
    for device_index in range(2):
        device = torch.device("cuda", device_index)
        compiled = compile_orbit(config, device)
        lat, lon = torch.meshgrid(
            torch.linspace(-math.pi / 2, math.pi / 2, 102, dtype=torch.float64, device=device),
            torch.linspace(-math.pi, math.pi, 102, dtype=torch.float64, device=device),
            indexing="ij",
        )
        lat = lat.contiguous()
        lon = lon.contiguous()
        orbit = make_orbit(config["orbit"])
        for seconds in (0.0, 3600.0, 0.25 * float(config["orbit"]["orbital_period"])):
            time = torch.tensor(seconds, dtype=torch.float64, device=device)
            actual = compiled(lon, lat, time)
            expected = orbit.insolation(lon, lat, time)
            for observed, reference in zip(actual, expected):
                torch.testing.assert_close(observed, reference, rtol=1.0e-11, atol=1.0e-11)
