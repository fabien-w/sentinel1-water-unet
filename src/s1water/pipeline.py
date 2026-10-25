"""End-to-end helpers: weights from Hugging Face, prediction of a preprocessed scene, GeoTIFF export."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from . import config
from .model import load_model
from .postprocess import apply_masks, shade_mask, valid_data_mask
from .predict import CROP, get_device, predict_image

log = logging.getLogger(__name__)

WATER, LAND, NODATA = 1, 0, 255
COLORMAP = {LAND: (0, 0, 0, 0), WATER: (30, 110, 255, 255), NODATA: (0, 0, 0, 0)}


def aoi_wkt(lon: float, lat: float, half_size: float = config.DEFAULT_AOI_HALF_SIZE) -> str:
    """Square area of interest (WKT, lon/lat) centred on a point."""
    w, e, s, n = lon - half_size, lon + half_size, lat - half_size, lat + half_size
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def download_weights(repo_id: str = config.HF_REPO_ID, filename: str = config.HF_WEIGHTS_FILE,
                     revision: str | None = None) -> str:
    """Download (and cache) the PyTorch weights from the Hugging Face Hub.

    Set the environment variable ``S1WATER_WEIGHTS`` to a local file to work offline.
    """
    import os

    if os.environ.get("S1WATER_WEIGHTS"):
        return os.environ["S1WATER_WEIGHTS"]
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def download_example_scene(repo_id: str = config.HF_REPO_ID, filename: str = config.HF_EXAMPLE_FILE) -> str:
    """Download the preprocessed Anavilhanas example (fallback when SNAP is not available)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def read_8bit(path: str | Path):
    """Read the 2-band 8-bit preprocessed image (VV, VH). Returns (array, profile)."""
    import rasterio

    with rasterio.open(path) as src:
        img = src.read([1, 2])
        prof = src.profile.copy()
    if img.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8, got {img.dtype}")
    return img, prof


def write_mask(path: str | Path, mask: np.ndarray, profile: dict, probability: np.ndarray | None = None) -> Path:
    """Write the water mask as a Cloud Optimized GeoTIFF (0 land, 1 water, 255 no data)."""
    import rasterio

    path = Path(path)
    prof = {k: profile[k] for k in ("crs", "transform", "width", "height")}
    prof.update(driver="COG", dtype="uint8", count=1, nodata=NODATA, compress="deflate", predictor=2)
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(mask, 1)
        dst.write_colormap(1, COLORMAP)
        dst.set_band_description(1, "water (1), not water (0), no data (255)")
        dst.update_tags(model="Sentinel-1 water U-Net (Wagner et al. 2024)", doi=config.PAPER_DOI,
                        threshold="0.5", values="0=not water,1=water,255=no data")
    tmp.replace(path)
    if probability is not None:
        pprof = dict(prof, dtype="float32", nodata=np.nan)
        with rasterio.open(path.with_name(path.stem + "_probability.tif"), "w", **pprof) as dst:
            dst.write(probability.astype(np.float32), 1)
    return path


def predict_scene(img8_tif: str | Path, out_tif: str | Path, weights: str | None = None, device: str | None = None,
                  apply_shade_mask: bool = False, srtm_cache: str | Path | None = None,
                  save_probability: bool = False) -> dict:
    """Prediction + post-processing of a preprocessed 8-bit scene, written as a GeoTIFF.

    Returns a summary dictionary (areas in km2, timing, device).
    """
    t0 = time.time()
    img, prof = read_8bit(img8_tif)
    dev = get_device(device)
    model = load_model(weights or download_weights(), dev)
    prob = predict_image(model, img, dev, return_probability=True)
    raw = np.full(prob.shape, NODATA, dtype=np.uint8)
    ok = ~np.isnan(prob)
    raw[ok] = (prob[ok] >= 0.5).astype(np.uint8)
    # the last 64 rows/cols lack the 64 px context of the original tiling when the image
    # was cut to an area of interest; they are no data, like the first 64 rows/cols
    raw[-CROP:, :] = NODATA
    raw[:, -CROP:] = NODATA
    t_pred = time.time() - t0
    valid = valid_data_mask(img, prof["transform"], prof["crs"])
    shade = shade_mask(raw.shape, prof["transform"], prof["crs"], srtm_cache) if apply_shade_mask else None
    mask = apply_masks(raw, valid, shade)
    write_mask(out_tif, mask, prof, prob if save_probability else None)

    from .postprocess import _pixel_size_m

    lat = prof["transform"].f + prof["transform"].e * prof["height"] / 2
    px, py = _pixel_size_m(prof["transform"], prof["crs"], lat)
    km2 = px * py / 1e6
    summary = {
        "device": str(dev),
        "seconds_prediction": round(t_pred, 1),
        "seconds_total": round(time.time() - t0, 1),
        "water_km2": round(float((mask == WATER).sum() * km2), 2),
        "land_km2": round(float((mask == LAND).sum() * km2), 2),
        "nodata_km2": round(float((mask == NODATA).sum() * km2), 2),
        "shade_km2": round(float(shade.sum() * km2), 2) if shade is not None else None,
        "output": str(out_tif),
    }
    log.info("%s", summary)
    return summary
