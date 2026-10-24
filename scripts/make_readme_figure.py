"""README figure: Sentinel-1 VV, water mask over VV, and a zoom on the default point (PDF then PNG).

Usage:
    python scripts/make_readme_figure.py IMG8_TIF WATER_TIF OUT_STEM "2026-08-28"
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

WATER = ListedColormap(["#1e6eff"])
HALO = [pe.withStroke(linewidth=3, foreground="black")]


def panel(ax, vv, mask, extent, letter, km_per_deg, bar_km):
    ax.imshow(np.ma.masked_equal(vv, 250), cmap="gray", vmin=100, vmax=230, extent=extent, interpolation="nearest")
    if mask is not None:
        ax.imshow(np.ma.masked_where(mask != 1, mask), cmap=WATER, alpha=0.75, extent=extent, interpolation="nearest")
    ax.set_axis_off()
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.text(0.97, 0.97, f"({letter})", transform=ax.transAxes, ha="right", va="top", color="white",
            fontsize=16, fontweight="bold", path_effects=HALO)
    # scale bar (km) and north arrow, drawn by hand
    w = extent[1] - extent[0]
    x0, y0 = extent[0] + 0.05 * w, extent[2] + 0.06 * (extent[3] - extent[2])
    ax.plot([x0, x0 + bar_km / km_per_deg], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    ax.text(x0 + bar_km / km_per_deg / 2, y0 + 0.02 * w, f"{bar_km:g} km", color="white", ha="center",
            fontsize=11, path_effects=HALO)
    ax.annotate("N", xy=(0.06, 0.95), xytext=(0.06, 0.84), xycoords="axes fraction", color="white",
                ha="center", fontsize=12, fontweight="bold", path_effects=HALO,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2))


def main(img8_tif, water_tif, out_stem, date, lat=-2.70, lon=-60.80, zoom_km=12):
    with rasterio.open(img8_tif) as s:
        vv = s.read(1)
        tr = s.transform
    with rasterio.open(water_tif) as s:
        mask = s.read(1)
    h, w = vv.shape
    extent = [tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f]
    km_per_deg = 111.32 * np.cos(np.radians(lat))
    step = max(1, w // 1500)
    r, c = rasterio.transform.rowcol(tr, lon, lat)
    half = int(zoom_km / 2 / (tr.a * km_per_deg))
    zs = (slice(r - half, r + half), slice(c - half, c + half))
    zext = [tr.c + tr.a * (c - half), tr.c + tr.a * (c + half), tr.f + tr.e * (r + half), tr.f + tr.e * (r - half)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="black")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.07, wspace=0.01)
    panel(axs[0], vv[::step, ::step], None, extent, "a", km_per_deg, 10)
    panel(axs[1], vv[::step, ::step], mask[::step, ::step], extent, "b", km_per_deg, 10)
    axs[1].plot([zext[0], zext[1], zext[1], zext[0], zext[0]], [zext[2], zext[2], zext[3], zext[3], zext[2]],
                color="gold", lw=1.5)
    panel(axs[2], vv[zs], mask[zs], zext, "c", km_per_deg, 2)
    fig.text(0.5, 0.025, f"Sentinel-1 VV and U-Net water mask (blue), Anavilhanas, Rio Negro, {date}",
             color="#F28C28", ha="center", va="center", fontsize=13)
    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), facecolor="black")
    fig.savefig(out.with_suffix(".png"), dpi=110, facecolor="black")
    print("written", out.with_suffix(".pdf"), out.with_suffix(".png"))


if __name__ == "__main__":
    main(*sys.argv[1:5])
