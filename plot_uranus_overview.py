#!/usr/bin/env python3
"""Plot zonal wind, condensate clouds, and total cloud path from a lat-lon file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import xarray as xr


def plot_overview(input_path: Path, output_path: Path) -> None:
    with xr.open_dataset(input_path) as dataset:
        fields = dataset.isel(time=-1, drop=True)
        latitude = dataset["lat"].values
        longitude = dataset["lon"].values
        layer_bounds = dataset["altitude_bounds"]
        layer_thickness = layer_bounds.isel(bnds=1) - layer_bounds.isel(bnds=0)

        wind = fields["vel_east"].mean("lon").values
        pressure = fields["press"].mean("lon").values / 1.0e5
        ch4 = (fields["CH4_s_"] + fields["CH4_s_p_"]).clip(min=0.0)
        h2s = (fields["H2S_s_"] + fields["H2S_s_p_"]).clip(min=0.0)
        ch4_zonal = 1000.0 * ch4.mean("lon").values
        h2s_zonal = 1000.0 * h2s.mean("lon").values
        cloud_path = (fields["rho"] * (ch4 + h2s) * layer_thickness).sum("altitude").values
        elapsed_days = float(dataset["time"].values[-1]) / 86400.0

    figure = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[20, 1], wspace=0.15)
    wind_axis = figure.add_subplot(grid[0, 0])
    cloud_axis = figure.add_subplot(grid[0, 1], sharey=wind_axis)
    path_axis = figure.add_subplot(grid[0, 2])
    wind_colorbar_axis = figure.add_subplot(grid[1, 0])
    cloud_colorbar_grid = grid[1, 1].subgridspec(1, 2, wspace=0.35)
    ch4_colorbar_axis = figure.add_subplot(cloud_colorbar_grid[0, 0])
    h2s_colorbar_axis = figure.add_subplot(cloud_colorbar_grid[0, 1])
    path_colorbar_axis = figure.add_subplot(grid[1, 2])

    wind_limit = 10.0 * np.ceil(np.nanmax(np.abs(wind)) / 10.0)
    wind_levels = np.linspace(-wind_limit, wind_limit, 25)
    latitude_grid = np.broadcast_to(latitude, pressure.shape)
    wind_contours = wind_axis.contourf(
        latitude_grid, pressure, wind, levels=wind_levels,
        cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-wind_limit, vmax=wind_limit),
        extend="both",
    )
    wind_axis.set(title="(a) Zonal-mean zonal wind", xlabel="Latitude [deg]", ylabel="Pressure (bar)")
    figure.colorbar(wind_contours, cax=wind_colorbar_axis, orientation="horizontal").set_label("Wind (m s$^{-1}$)")

    #ch4_levels = np.geomspace(0.01, 3.0, 9)
    ch4_levels = np.linspace(0.01, 3.0, 9)
    #h2s_levels = np.geomspace(0.001, 0.3, 9)
    h2s_levels = np.linspace(0.001, 0.3, 9)
    ch4_contours = cloud_axis.contourf(
        latitude_grid, pressure, np.ma.masked_less_equal(ch4_zonal, 0.0), levels=ch4_levels,
        #norm=LogNorm(vmin=ch4_levels[0], vmax=ch4_levels[-1]),
        cmap="Blues", alpha=0.72, extend="max",
    )
    h2s_contours = cloud_axis.contourf(
        latitude_grid, pressure, np.ma.masked_less_equal(h2s_zonal, 0.0), levels=h2s_levels,
        #norm=LogNorm(vmin=h2s_levels[0], vmax=h2s_levels[-1]),
        cmap="Oranges", alpha=0.72, extend="max",
    )
    cloud_axis.contour(latitude_grid, pressure, ch4_zonal, levels=[0.1, 1.0], colors="navy", linewidths=0.6)
    cloud_axis.contour(latitude_grid, pressure, h2s_zonal, levels=[0.01, 0.1], colors="darkorange", linewidths=0.6)
    cloud_axis.set(title="(b) Zonal-mean condensates", xlabel="Latitude [deg]")
    cloud_axis.tick_params(labelleft=False)
    ch4_colorbar = figure.colorbar(ch4_contours, cax=ch4_colorbar_axis, orientation="horizontal")
    ch4_colorbar.set_ticks([0.01, 3.0], labels=["0.01", "3"])
    ch4_colorbar.set_label("CH$_4$ (g kg$^{-1}$)")
    h2s_colorbar = figure.colorbar(h2s_contours, cax=h2s_colorbar_axis, orientation="horizontal")
    h2s_colorbar.set_ticks([0.001, 0.3], labels=["0.001", "0.3"])
    h2s_colorbar.set_label("H$_2$S (g kg$^{-1}$)")

    #path_levels = np.geomspace(0.2, 500.0, 16)
    path_levels = np.linspace(100., 500.0, 9)
    path_contours = path_axis.contourf(
        longitude, latitude, cloud_path, levels=path_levels,
        #norm=LogNorm(vmin=path_levels[0], vmax=path_levels[-1]),
        cmap="Greys_r", extend="both",
    )
    path_axis.set(title="(c) total cloud path", xlabel="Longitude [deg]", ylabel="Latitude [deg]")
    path_axis.set(xlim=(0, 360), ylim=(-90, 90), xticks=np.arange(0, 361, 60), yticks=np.arange(-90, 91, 30))
    path_colorbar = figure.colorbar(path_contours, cax=path_colorbar_axis, orientation="horizontal")
    #path_colorbar.set_ticks([100., 200., 300., 400.0, 500.0], labels=["100.", "200.", "400.", "500."])
    path_colorbar.set_ticks([100., 200., 300., 400.0, 500.0])
    path_colorbar.set_label("Cloud path (kg m$^{-2}$)")

    for axis in (wind_axis, cloud_axis):
        axis.set_yscale("log")
        axis.set(
            xlim=(-90, 90), ylim=(float(np.nanmax(pressure)) * 1.02, float(np.nanmin(pressure)) * 0.9),
            xticks=np.arange(-90, 91, 30),
        )
        axis.set_yticks([20, 10, 1, 0.1, 0.01, 0.001], labels=["20", "10", "1", "0.1", "0.01", "0.001"])
    #figure.suptitle(f"Uranus circulation and clouds — day {elapsed_days:.1f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Lat-lon remapped NetCDF output")
    parser.add_argument("-o", "--output", type=Path, help="Output image (default: input stem in current directory)")
    arguments = parser.parse_args()
    output = arguments.output or Path(f"{arguments.input.stem}.png")
    plot_overview(arguments.input, output)
    print(output)


if __name__ == "__main__":
    main()
