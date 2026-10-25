"""Post-processing of the water predictions, following Wagner et al. (2024).

1. Valid-data mask (R script 29.03 of the paper), always applied. SNAP no-data becomes 250
   in the 8-bit image, but 250 is also a valid value (backscatter of ~0 dB): the original
   code therefore never masks single pixels. It builds the valid-data polygon on a grid
   16 times coarser (gdalwarp -r average): a cell is no data only if all its pixels are
   250, i.e. outside the acquisition footprint. It removes the holes and erodes the scene
   borders by 1000 m on the left/right (range) edges and 500 m on the top/bottom
   (azimuth) edges, where GRD border noise and slice "staircases" remain.
   Here the same widths are applied with an anisotropic erosion of the coarse
   valid-data mask (the polygon vertex logic of the original R code cannot be
   ported exactly).

2. Mountain shade mask (paper section 2.3, R script SRTM_preparation.r), OPTIONAL and off by
   default: over the forested lowlands SRTM is noisy (canopy steps, pits over water) and the
   rule flags lakes and channels, which the paper removed by hand. SRTM 1 arc-second
   elevation is aggregated by 3 with the minimum (~90 m), the slope is computed with
   8 neighbours, slopes >= 20 degrees are kept, converted to polygons without holes
   and replaced by their convex hulls. The manual edits of the paper are not reproduced.
   SRTM tiles are read from the public AWS Terrain Tiles bucket (no account needed).

Output coding of the final mask (uint8): 0 = not water, 1 = water, 255 = no data, used only
outside the valid footprint (scene border), in the 64 px image margin, and in the optional
shade mask. The original files used 254 for water.
"""

from __future__ import annotations

import gzip
import logging
import math
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage

from .preprocess import NODATA_8BIT

log = logging.getLogger(__name__)

COARSE_FACTOR = 16
BORDER_RANGE_M = 1000.0  # left / right borders (R: st_buffer(left_border_utm, 1000))
BORDER_AZIMUTH_M = 500.0  # top / bottom borders (R: st_buffer(top_border_utm, 500))
SLOPE_THRESHOLD_DEG = 20.0
SRTM_AGGREGATE = 3
SKADI_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}/{name}.hgt.gz"
M_PER_DEG = 111320.0


def _pixel_size_m(transform, crs, lat: float) -> tuple[float, float]:
    """Approximate pixel size in metres (x, y) for geographic or projected grids."""
    if crs is not None and crs.is_geographic:
        return abs(transform.a) * M_PER_DEG * math.cos(math.radians(lat)), abs(transform.e) * M_PER_DEG
    return abs(transform.a), abs(transform.e)


def valid_data_mask(img8: np.ndarray, transform, crs) -> np.ndarray:
    """Boolean mask (H, W) of valid pixels after removal of SNAP no-data and scene borders.

    ``img8`` is the (2, H, W) or (H, W) 8-bit image; band 1 (VV) is used, as in the R code.
    """
    band = img8[0] if img8.ndim == 3 else img8
    h, w = band.shape
    f = COARSE_FACTOR
    hc, wc = math.ceil(h / f), math.ceil(w / f)
    pad = np.full((hc * f, wc * f), 255, dtype=np.uint8)  # 255 = outside image (no-data of the original files)
    pad[:h, :w] = band
    blocks = pad.reshape(hc, f, wc, f).transpose(0, 2, 1, 3).reshape(hc, wc, f * f).astype(np.float32)
    # gdalwarp -r average ignores the no-data value (255) of the source
    inside = blocks != 255
    n = inside.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(n > 0, (blocks * inside).sum(-1) / np.maximum(n, 1), np.nan)
    # R: r[is.na(r)] = 250; r[r == 250] = 0; r[r != 0] = 1
    coarse_valid = ~np.isnan(avg) & (avg != NODATA_8BIT)
    # cells outside the image are not scene borders: treat them as valid while eroding
    outside = n == 0
    coarse_valid = ndimage.binary_fill_holes(coarse_valid | outside) & ~outside  # nngeo::st_remove_holes

    lat = transform.f + transform.e * h / 2
    px, py = _pixel_size_m(transform, crs, lat)
    rx = int(round(BORDER_RANGE_M / (px * f)))
    ry = int(round(BORDER_AZIMUTH_M / (py * f)))
    if rx > 0 or ry > 0:
        yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
        ell = ((xx / max(rx, 1e-9)) ** 2 + (yy / max(ry, 1e-9)) ** 2) <= 1.0
        nodata = ~(coarse_valid | outside)
        coarse_valid = coarse_valid & ~ndimage.binary_dilation(nodata, structure=ell)
    # single pixels equal to 250 inside the footprint stay valid (~0 dB targets), as in the original
    return np.repeat(np.repeat(coarse_valid, f, axis=0), f, axis=1)[:h, :w]


# --------------------------------------------------------------------------- SRTM shade


def _skadi_name(lat_floor: int, lon_floor: int) -> tuple[str, str]:
    ns = f"{'N' if lat_floor >= 0 else 'S'}{abs(lat_floor):02d}"
    ew = f"{'E' if lon_floor >= 0 else 'W'}{abs(lon_floor):03d}"
    return ns, ns + ew


def read_srtm(bounds: tuple[float, float, float, float], cache_dir: str | Path | None = None):
    """Mosaic of SRTM 1 arc-second tiles covering ``bounds`` (lon/lat).

    Returns (elevation float32 array with NaN for voids/ocean, rasterio Affine transform).
    """
    from rasterio.transform import from_origin

    west, south, east, north = bounds
    lat0, lat1 = math.floor(south), math.floor(north - 1e-9)
    lon0, lon1 = math.floor(west), math.floor(east - 1e-9)
    n = 3600
    mosaic = np.full(((lat1 - lat0 + 1) * n, (lon1 - lon0 + 1) * n), np.nan, dtype=np.float32)
    cache = Path(cache_dir) if cache_dir else None
    for lat in range(lat0, lat1 + 1):
        for lon in range(lon0, lon1 + 1):
            ns, name = _skadi_name(lat, lon)
            data = None
            cached = cache / f"{name}.hgt.gz" if cache else None
            if cached and cached.exists():
                data = cached.read_bytes()
            else:
                try:
                    with urllib.request.urlopen(SKADI_URL.format(ns=ns, name=name), timeout=120) as r:
                        data = r.read()
                    if cached:
                        cached.parent.mkdir(parents=True, exist_ok=True)
                        cached.write_bytes(data)
                except Exception as err:  # noqa: BLE001 - missing tile = ocean
                    log.warning("SRTM tile %s not available (%s): treated as flat", name, err)
            if data is None:
                continue
            arr = np.frombuffer(gzip.decompress(data), dtype=">i2").reshape(3601, 3601).astype(np.float32)
            arr[arr == -32768] = np.nan
            r0 = (lat1 - lat) * n
            c0 = (lon - lon0) * n
            mosaic[r0 : r0 + n, c0 : c0 + n] = arr[:n, :n]
    transform = from_origin(lon0 - 0.5 / n, lat1 + 1 + 0.5 / n, 1 / n, 1 / n)
    return mosaic, transform


def slope_degrees(dem: np.ndarray, res_x_m: float, res_y_m: float) -> np.ndarray:
    """Slope in degrees with 8 neighbours (Horn 1981), like terra::terrain(v='slope')."""
    z = np.pad(dem, 1, mode="edge")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res_x_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res_y_m)
    s = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    s[np.isnan(dem)] = np.nan
    return s


def shade_mask(shape: tuple[int, int], transform, crs, cache_dir: str | Path | None = None) -> np.ndarray:
    """Boolean mask (H, W) of mountain shade (convex hulls of SRTM slopes >= 20 degrees)."""
    import rasterio.features
    from rasterio.warp import transform_bounds
    from shapely.geometry import MultiPoint, box
    from shapely.geometry import shape as shp_shape
    from shapely.ops import unary_union

    h, w = shape
    left, top = transform.c, transform.f
    right, bottom = left + transform.a * w, top + transform.e * h
    bounds = (left, bottom, right, top)
    if crs is not None and not crs.is_geographic:
        bounds = transform_bounds(crs, "EPSG:4326", *bounds)
    margin = 0.02
    bounds = (bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin)
    dem, dem_tr = read_srtm(bounds, cache_dir)
    # aggregate(r, fact = 3, fun = "min")
    k = SRTM_AGGREGATE
    hh, ww = dem.shape[0] // k * k, dem.shape[1] // k * k
    agg = dem[:hh, :ww].reshape(hh // k, k, ww // k, k)
    with np.errstate(all="ignore"):
        agg = np.nanmin(agg, axis=(1, 3))
    agg_tr = dem_tr * dem_tr.scale(k, k)
    lat = (bounds[1] + bounds[3]) / 2
    rx = abs(agg_tr.a) * M_PER_DEG * math.cos(math.radians(lat))
    ry = abs(agg_tr.e) * M_PER_DEG
    steep = np.nan_to_num(slope_degrees(agg, rx, ry), nan=0.0) >= SLOPE_THRESHOLD_DEG
    steep = ndimage.binary_fill_holes(steep)
    labels, n = ndimage.label(steep)  # 4-connectivity, like polygons cast to POLYGON
    if n == 0:
        return np.zeros(shape, dtype=bool)
    hulls = []
    for sl, lab in zip(ndimage.find_objects(labels), range(1, n + 1), strict=True):
        rr, cc = np.nonzero(labels[sl] == lab)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        # pixel corners in lon/lat
        xs = np.concatenate([cc, cc + 1, cc, cc + 1])
        ys = np.concatenate([rr, rr, rr + 1, rr + 1])
        lon = agg_tr.c + xs * agg_tr.a
        lat_ = agg_tr.f + ys * agg_tr.e
        hulls.append(MultiPoint(np.column_stack([lon, lat_])).convex_hull)
    geom = unary_union(hulls)
    if crs is not None and not crs.is_geographic:
        from rasterio.warp import transform_geom

        geom = shp_shape(transform_geom("EPSG:4326", crs, geom.__geo_interface__))
    geom = geom.intersection(box(left, bottom, right, top))
    if geom.is_empty:
        return np.zeros(shape, dtype=bool)
    return rasterio.features.rasterize([geom], out_shape=shape, transform=transform, fill=0, default_value=1).astype(bool)


def apply_masks(pred: np.ndarray, valid: np.ndarray, shade: np.ndarray | None = None) -> np.ndarray:
    """Combine the raw prediction (0/1/255) with the valid-data and shade masks."""
    out = pred.copy()
    out[~valid] = 255
    if shade is not None:
        out[shade] = 255
    return out
