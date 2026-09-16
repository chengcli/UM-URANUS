import torch

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
