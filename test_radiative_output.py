from types import SimpleNamespace

import torch

import rt_forcing


def test_flux_output_preserves_faces(monkeypatch):
    faces = torch.arange(24, dtype=torch.float64).reshape(2, 6, 2)
    state = SimpleNamespace(
        lon=torch.zeros(1, 2), il=1, iu=5,
        visible_flux=faces, infrared_flux=2 * faces,
    )
    monkeypatch.setattr(rt_forcing, "compute_heating", lambda *args: None)
    variables = {"hydro_w": torch.zeros(8, 1, 2, 7, dtype=torch.float64)}
    output = rt_forcing.radiative_flux_output(variables, None, None, None, state, 0.0)
    assert len(output) == 8
    for band, factor in (("vis", 1), ("ir", 2)):
        for direction, index in (("up", 0), ("dn", 1)):
            lower = output[f"{band}_flux_{direction}_lower"]
            upper = output[f"{band}_flux_{direction}_upper"]
            torch.testing.assert_close(lower[..., 1:6], factor * faces[..., :-1, index].reshape(1, 2, 5))
            torch.testing.assert_close(upper[..., 1:6], factor * faces[..., 1:, index].reshape(1, 2, 5))
            assert torch.count_nonzero(lower[..., [0, 6]]) == 0
