import math

import torch

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


class OrbitalInsolation(torch.nn.Module):
    def __init__(self, orbit: OrbitalForcing) -> None:
        super().__init__()
        self.orbit = orbit

    def forward(
        self, lon: torch.Tensor, lat: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.orbit.insolation(lon, lat, time)
