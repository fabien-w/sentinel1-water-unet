"""Sentinel-1 IW GRD preprocessing with ESA SNAP (esa_snappy), as in Wagner et al. (2024).

Processing chain (same operators and parameters as the original snappy notebooks):
  1. Apply-Orbit-File       orbitType='Sentinel Precise (Auto Download)', polyDegree=3, continueOnFail=false
  2. Remove-GRD-Border-Noise selectedPolarisations='VH,VV' (defaults otherwise)
  3. ThermalNoiseRemoval    removeThermalNoise=true
  4. Calibration            outputSigmaBand=true, outputImageScaleInDb=false (sigma0, linear)
  5. Terrain-Correction     demName='GETASSE30', imgResamplingMethod='BILINEAR_INTERPOLATION',
                            default map projection (WGS84 lat/lon) and default pixel spacing
                            (10 m, i.e. 8.983e-5 degree)
  6. LinearToFromdB         Sigma0_VH, Sigma0_VV -> dB
  7. Subset                 scene footprint (original) or an area of interest (option)
  8. 8-bit scaling          (maximum(-49, minimum(1, dB)) + 50) * 5   (gdal_calc --type=Byte)

The 8-bit model input is a 2-band image (band 1 = VV, band 2 = VH). SNAP no-data
(0 dB after conversion) becomes 250 in the 8-bit image, which is used later to build
the valid-data mask.

SNAP itself is a Java program; this module only needs ``esa_snappy`` (bundled with
SNAP 10+, configured with ``<snap>/bin/snappy-conf <python>``) for steps 1-7.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Parameters of the original processing (do not change: the model was trained on them)
ORBIT_TYPE = "Sentinel Precise (Auto Download)"
ORBIT_FALLBACK = "Sentinel Restituted (Auto Download)"
DEM_NAME = "GETASSE30"
RESAMPLING = "BILINEAR_INTERPOLATION"
DB_MIN, DB_MAX = -49.0, 1.0
NODATA_8BIT = 250  # SNAP no-data (0 dB) after the 8-bit scaling: (0 + 50) * 5


def _snappy():
    """Import esa_snappy (SNAP >= 10) or the old snappy module (SNAP <= 9)."""
    try:
        import esa_snappy as snappy
    except ImportError:  # pragma: no cover - old SNAP versions
        import snappy  # type: ignore
    return snappy


def polarisations_from_name(name: str) -> str:
    """Return SNAP polarisation list from the product name (as in the original notebook)."""
    pol = name.split("_")[3][2:4]
    table = {"DV": "VH,VV", "DH": "HH,HV", "SH": "HH", "HH": "HH", "SV": "VV"}
    if pol not in table:
        raise ValueError(f"Unknown polarisation code {pol!r} in {name}")
    return table[pol]


def footprint_wkt(safe_dir: str | Path) -> str:
    """Scene footprint polygon (WKT, lon/lat) read from preview/map-overlay.kml."""
    kml = Path(safe_dir) / "preview" / "map-overlay.kml"
    text = kml.read_text()
    coords = re.search(r"<coordinates>(.*?)</coordinates>", text, re.S).group(1).split()
    pts = [tuple(map(float, c.split(",")[:2])) for c in coords]
    pts.append(pts[0])
    return "POLYGON((" + ", ".join(f"{x} {y}" for x, y in pts) + "))"


def product_name(safe_dir: str | Path) -> str:
    """Product name (e.g. S1A_IW_GRDH_1SDV_...) from the folder name or the manifest."""
    name = Path(safe_dir).name.replace(".SAFE", "")
    if name.startswith("S1"):
        return name
    root = ET.parse(Path(safe_dir) / "manifest.safe").getroot()  # pragma: no cover
    return root.find(".//{*}productName").text  # pragma: no cover


def run_snap_chain(
    safe_dir: str | Path,
    out_tif: str | Path,
    aoi_wkt: str | None = None,
    orbit_fallback: bool = True,
) -> Path:
    """Run steps 1-7 with esa_snappy and write a 2-band float32 GeoTIFF in dB.

    Band order of the output: Sigma0_VV_db, Sigma0_VH_db.

    Args:
        safe_dir: unzipped Sentinel-1 product folder (manifest.safe inside).
        out_tif: output GeoTIFF path.
        aoi_wkt: optional WKT polygon (lon/lat) to subset after terrain correction.
            None reproduces the original processing (subset on the scene footprint).
        orbit_fallback: if precise orbits are not yet published (scenes < ~20 days old),
            use restituted orbits instead of failing. The original code failed.
    """
    snappy = _snappy()
    GPF, HashMap, ProductIO = snappy.GPF, snappy.HashMap, snappy.ProductIO
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    safe_dir = Path(safe_dir)
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    manifest = safe_dir / "manifest.safe"
    product = ProductIO.readProduct(str(manifest if manifest.exists() else safe_dir))
    name = product.getName()
    pols = polarisations_from_name(name)
    if pols != "VH,VV":
        raise ValueError(f"The model needs dual VV+VH scenes, got {pols} for {name}")

    def op(name_op, src, **params):
        p = HashMap()
        for k, v in params.items():
            p.put(k, v)
        return GPF.createProduct(name_op, p, src)

    log.info("1/7 Apply-Orbit-File")
    try:
        prod = op("Apply-Orbit-File", product, orbitType=ORBIT_TYPE, polyDegree="3", continueOnFail="false")
        prod.getBandNames()  # forces initialisation, raises if the orbit file is missing
    except Exception as err:  # noqa: BLE001 - Java exceptions come through jpy
        if not orbit_fallback:
            raise
        log.warning("Precise orbit not available (%s); using restituted orbit", err)
        prod = op("Apply-Orbit-File", product, orbitType=ORBIT_FALLBACK, polyDegree="3", continueOnFail="false")
    log.info("2/7 Remove-GRD-Border-Noise")
    prod = op("Remove-GRD-Border-Noise", prod, selectedPolarisations=pols)
    log.info("3/7 ThermalNoiseRemoval")
    prod = op("ThermalNoiseRemoval", prod, removeThermalNoise=True)
    log.info("4/7 Calibration (sigma0, linear)")
    prod = op("Calibration", prod, outputSigmaBand=True, selectedPolarisations=pols, outputImageScaleInDb=False)
    log.info("5/7 Terrain-Correction (GETASSE30, bilinear, WGS84)")
    prod = op(
        "Terrain-Correction",
        prod,
        demName=DEM_NAME,
        imgResamplingMethod=RESAMPLING,
        saveProjectedLocalIncidenceAngle=False,
        saveSelectedSourceBand=True,
    )
    log.info("6/7 LinearToFromdB")
    prod = op("LinearToFromdB", prod, sourceBands="Sigma0_VH,Sigma0_VV")
    log.info("7/7 Subset (%s)", "area of interest" if aoi_wkt else "scene footprint")
    geom = snappy.WKTReader().read(aoi_wkt or footprint_wkt(safe_dir))
    prod = op("Subset", prod, sourceBands="Sigma0_VV_db,Sigma0_VH_db", copyMetadata=True, geoRegion=geom)
    # Subset keeps the band order of the product (VH before VV), not the order requested:
    # record the real band names in the GeoTIFF so that later steps select bands by name.
    band_names = list(prod.getBandNames())
    log.info("Writing %s (bands %s)", out_tif, band_names)
    ProductIO.writeProduct(prod, str(out_tif.with_suffix("")), "GeoTIFF-BigTIFF")
    product.dispose()
    written = out_tif.with_suffix(".tif")
    if written != out_tif:  # pragma: no cover
        written.rename(out_tif)
    import rasterio

    with rasterio.open(out_tif, "r+") as dst:
        for i, b in enumerate(band_names, start=1):
            dst.set_band_description(i, b)
    return out_tif


def band_index(src, polarisation: str) -> int:
    """1-based index of the Sigma0_<pol>_db band, from the band descriptions."""
    for i, d in enumerate(src.descriptions, start=1):
        if d and polarisation in d:
            return i
    raise ValueError(f"no {polarisation} band in {src.name} (descriptions: {src.descriptions})")


def db_to_uint8(db: np.ndarray) -> np.ndarray:
    """8-bit scaling of the original code: (maximum(-49, minimum(1, x)) + 50) * 5.

    The original used ``gdal_calc.py --type=Byte`` on float32 input. gdal_calc computes
    in the numpy type of the input (float32) and GDAL converts the result to Byte by
    rounding to the nearest integer (GDALCopyWords). NaN (no data) -> 250 like 0 dB.
    """
    x = np.asarray(db, dtype=np.float32)
    x = np.where(np.isnan(x), np.float32(0), x)
    y = (np.maximum(np.float32(DB_MIN), np.minimum(np.float32(DB_MAX), x)) + np.float32(50)) * np.float32(5)
    return np.clip(np.floor(y + np.float32(0.5)), 0, 255).astype(np.uint8)


def to_8bit_geotiff(db_tif: str | Path, out_tif: str | Path, block_rows: int = 2048) -> Path:
    """Convert the 2-band dB GeoTIFF (VV, VH) into the 2-band 8-bit model input GeoTIFF."""
    import rasterio
    from rasterio.windows import Window

    db_tif, out_tif = Path(db_tif), Path(out_tif)
    with rasterio.open(db_tif) as src:
        bands = [band_index(src, "VV"), band_index(src, "VH")]
        prof = src.profile.copy()
        prof.update(dtype="uint8", count=2, nodata=None, compress="deflate", predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        prof.pop("photometric", None)
        with rasterio.open(out_tif, "w", **prof) as dst:
            for row in range(0, src.height, block_rows):
                h = min(block_rows, src.height - row)
                w = Window(0, row, src.width, h)
                dst.write(db_to_uint8(src.read(bands, window=w)), window=w)
            dst.set_band_description(1, "VV (8-bit, (max(-49,min(1,dB))+50)*5)")
            dst.set_band_description(2, "VH (8-bit, (max(-49,min(1,dB))+50)*5)")
            dst.update_tags(nodata_value_8bit=str(NODATA_8BIT), source=db_tif.name)
    return out_tif


def preprocess_scene(safe_dir: str | Path, out_dir: str | Path, aoi_wkt: str | None = None,
                     keep_db: bool = False, orbit_fallback: bool = True) -> Path:
    """Full preprocessing: SNAP chain + 8-bit scaling. Returns the 8-bit VV/VH GeoTIFF."""
    out_dir = Path(out_dir)
    name = product_name(safe_dir)
    db_tif = out_dir / f"{name}_VV_VH_db.tif"
    out_tif = out_dir / f"{name}_VV_VH_8bits.tif"
    run_snap_chain(safe_dir, db_tif, aoi_wkt=aoi_wkt, orbit_fallback=orbit_fallback)
    to_8bit_geotiff(db_tif, out_tif)
    if not keep_db:
        db_tif.unlink()
    return out_tif
