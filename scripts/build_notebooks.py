"""Build the Colab and JupyterLab notebooks from shared cells (run: python scripts/build_notebooks.py).

Both notebooks run the same steps a-g; only the setup cells differ:
  - Colab: installs the package from GitHub, optionally installs SNAP, reads AWS keys from Colab secrets
  - JupyterLab: uses the installed package, reads AWS keys from environment variables or a git-ignored .env file
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "fabien-w/sentinel1-water-unet"
HF = "Fabien-Ctrees/sentinel1-water-unet"
COLAB_URL = f"https://colab.research.google.com/github/{GITHUB}/blob/main/notebooks/water_sentinel1_colab.ipynb"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def intro(colab: bool):
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})\n\n" if colab else ""
    where = (
        "This notebook runs on **Google Colab**. Select a GPU first: *Runtime → Change runtime type → T4 GPU*."
        if colab
        else "This notebook runs in **JupyterLab** on your computer or a cloud machine (e.g. AWS EC2). "
        "Install the package and SNAP first, see the [README](../README.md#installation)."
    )
    return md(f"""
{badge}# Surface water from Sentinel-1 with a U-Net: Anavilhanas, Rio Negro

This tutorial maps surface water from a Sentinel-1 radar image with the U-Net model of
[Wagner et al. (2024), *The Amazon's 2023 Drought: Sentinel-1 Reveals Extreme Rio Negro River Contraction*, Remote Sensing 16(6):1056](https://doi.org/10.3390/rs16061056).
The model was trained on Sentinel-1 images of the Amazon and high-resolution reference water masks (F1-score 0.93).

{where}

**Steps**
- **a.** choose a point on an interactive map (default: Anavilhanas archipelago, Rio Negro, Brazil)
- **b.** search the Sentinel-1 IW GRD scenes over this point (default: August 2026)
- **c.** download the scene from the `sentinel-s1-l1c` AWS bucket (*requester pays*, see below)
- **d.** preprocess it with ESA SNAP (`esa_snappy`): orbit, border and thermal noise removal, calibration, terrain correction, dB, 8 bits
- **e.** download the PyTorch weights from [Hugging Face](https://huggingface.co/{HF}) and predict on the GPU (4224 px tiles, as in the paper)
- **f.** post-processing of the paper: scene-border mask and SRTM mountain-shade mask
- **g.** display the water mask over the radar image and a satellite basemap, and export a GeoTIFF

**About data access and costs.** The Sentinel-1 bucket `s3://sentinel-s1-l1c` (AWS Open Data) is
*requester pays*: you need an AWS account, and your account pays the transfer, about **US$0.12 per scene**
(1.3 GB × US$0.09/GB). Without AWS credentials, the notebook uses a **preprocessed Anavilhanas scene
(28 August 2026) hosted on Hugging Face** and skips steps c and d.
""")


COLAB_SETUP = [
    md("""
## 0. Setup (Colab)

Installs the `s1water` package from GitHub (about 1 minute). Then, if `INSTALL_SNAP` is checked,
installs ESA SNAP 14 and `esa_snappy` (about 1 GB, 4-6 minutes). SNAP is needed only to preprocess a
new scene (step d); unchecked, the notebook uses the preprocessed example scene.
"""),
    code(f"""
INSTALL_SNAP = True  #@param {{type:"boolean"}}

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "s1water[data,viz] @ git+https://github.com/{GITHUB}.git"], check=True)
import torch
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if not torch.cuda.is_available():
    print("No GPU: select Runtime > Change runtime type > T4 GPU, then Runtime > Restart session. "
          "The notebook still works on CPU, but the prediction takes ~10 times longer.")
"""),
    code("""
# ESA SNAP 14 + esa_snappy (only if INSTALL_SNAP). Errors here are not fatal: the notebook falls back
# to the preprocessed example scene from Hugging Face.
import os, shutil, site, urllib.request
SNAP_DIR = "/content/snap"
SNAP_OK = False
if INSTALL_SNAP:
    try:
        if not os.path.exists(f"{SNAP_DIR}/bin/gpt"):
            url = "https://download.esa.int/step/snap/14.0/installers/esa-snap_sentinel_linux-14.0.0.sh"
            print("Downloading SNAP installer (~1 GB)...")
            urllib.request.urlretrieve(url, "/content/snap_installer.sh")
            subprocess.run(["sh", "/content/snap_installer.sh", "-q", "-dir", SNAP_DIR], check=True)
            os.remove("/content/snap_installer.sh")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "esa-snappy"], check=True)
        sp = site.getsitepackages()[0]
        subprocess.run([f"{SNAP_DIR}/bin/snappy-conf", sys.executable, sp], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ini = f"{sp}/esa_snappy/esa_snappy.ini"
        txt = open(ini).read().replace("java_max_mem: ", "#java_max_mem: ")
        open(ini, "w").write(txt + "\\njava_max_mem: 8G\\n")  # Colab has ~12 GB of RAM
        import esa_snappy  # noqa: F401
        SNAP_OK = True
        print("SNAP and esa_snappy are ready")
    except Exception as err:
        print("SNAP installation failed, the preprocessed example scene will be used:", err)
"""),
    md("""
### AWS credentials (Colab secrets)

To download new scenes, add two secrets in the **🔑 Secrets** panel on the left: `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` (an IAM user with `s3:GetObject` and `s3:ListBucket` is enough), and give this
notebook access. Never paste keys into a cell.
"""),
    code("""
import os
try:
    from google.colab import userdata
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        try:
            os.environ[k] = userdata.get(k)
        except Exception:
            pass
except ImportError:
    pass
"""),
]

JUPYTER_SETUP = [
    md("""
## 0. Setup

The package and its dependencies are installed with the environment of the repository
(`conda env create -f environment.yml`, then `pip install -e ".[notebook]"`), and SNAP with
[docs/INSTALL_SNAP.md](../docs/INSTALL_SNAP.md).

**AWS credentials** are read, in this order, from environment variables (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` or `AWS_PROFILE`), from a `.env` file at the root of the repository
(git-ignored, copy `.env.example`), or from `~/.aws/credentials`. On an EC2 instance with an IAM role,
nothing is needed. Never write keys in the notebook.
"""),
    code("""
import os
from pathlib import Path

# load a git-ignored .env file (KEY=value lines) if present, without overriding existing variables
for env_file in (Path.cwd() / ".env", Path.cwd().parent / ".env"):
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        print("Loaded settings from", env_file)

import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found: the prediction runs on CPU (it works, but ~10 times slower).")
try:
    import esa_snappy  # noqa: F401
    SNAP_OK = True
    print("esa_snappy found: new scenes can be preprocessed")
except Exception as err:
    SNAP_OK = False
    print("esa_snappy not available (see docs/INSTALL_SNAP.md): the preprocessed example scene will be used.", err)
"""),
]

COMMON = [
    code("""
import json, logging, time
from pathlib import Path
import numpy as np
import rasterio
from s1water import config
from s1water.data import download_scene, scene_in_bucket, search_scenes
from s1water.model import load_model
from s1water.pipeline import aoi_wkt, download_example_scene, download_weights, read_8bit, write_mask
from s1water.postprocess import apply_masks, shade_mask, valid_data_mask
from s1water.predict import CROP, get_device, predict_image
from s1water import viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
for noisy in ("botocore", "boto3", "urllib3", "httpx", "huggingface_hub", "rasterio", "asf_search"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
WORK = Path("s1water_work")
WORK.mkdir(exist_ok=True)


def aws_credentials_available():
    try:
        import boto3
        return boto3.session.Session().get_credentials() is not None
    except Exception:
        return False


HAVE_AWS = aws_credentials_available()
print("AWS credentials found:", HAVE_AWS, "| SNAP available:", SNAP_OK)
if not (HAVE_AWS and SNAP_OK):
    print("-> steps c and d will use the preprocessed Anavilhanas scene from Hugging Face")
"""),
    md("""
## a. Choose a point

The default point is on the **Anavilhanas archipelago** (Rio Negro, Brazil), a maze of ~400 forested
islands and channels upstream of Manaus. Click on the map to read other coordinates, then change
`LAT` and `LON` and run the cells again. `HALF_SIZE` sets the size of the area processed around the
point (0.25° gives a ~55 × 55 km square; the whole scene is ~250 × 170 km).
"""),
    code("""
LAT = -2.70  #@param {type:"number"}
LON = -60.80  #@param {type:"number"}
HALF_SIZE = 0.25  #@param {type:"number"}
AOI = aoi_wkt(LON, LAT, HALF_SIZE)
viz.point_map(LAT, LON, zoom=9, aoi_wkt=AOI)
"""),
    md("""
## b. Search Sentinel-1 scenes

Sentinel-1 IW GRD (high resolution, dual polarisation VV + VH) scenes intersecting the point, found with
the Alaska Satellite Facility catalogue (`asf_search`, no account needed), as in the original workflow.
By default the most recent scene of the period is selected. Since 2025, the scenes over Anavilhanas
come from **Sentinel-1C/D** (the model was trained with Sentinel-1A images; the sensors are identical by design).
"""),
    code("""
START = "2026-08-01"  #@param {type:"string"}
END = "2026-08-31"  #@param {type:"string"}
# dates as YYYY-MM-DD (or just YYYY-MM); a day that does not exist, e.g. 2026-06-31, is corrected
scenes = search_scenes(LON, LAT, START, END)
if not scenes:
    raise RuntimeError(f"No Sentinel-1 scene found over ({LAT}, {LON}) between {START} and {END}. "
                       "Sentinel-1 passes every 6-12 days: try a longer period, e.g. a whole month.")
for i, s in enumerate(scenes):
    print(i, s.name, s.start, s.orbit_direction, "orbit", s.relative_orbit)
scene = scenes[-1]
print("\\nSelected:", scene.name)
viz.point_map(LAT, LON, zoom=7, footprints={scene.name[17:32]: scene.footprint_wkt}, aoi_wkt=AOI)
"""),
    md("""
## c. Download the scene

From `s3://sentinel-s1-l1c/GRD/YYYY/M/D/IW/DV/<scene>/` (region eu-central-1), the bucket used in the
original workflow. **Requester pays**: about US$0.12 per scene, billed to your AWS account.
"""),
    code("""
EXAMPLE_NAME = "S1D_IW_GRDH_1SDV_20260828T094657_20260828T094722_004324_007F97_2C40"
use_example = not (HAVE_AWS and SNAP_OK)
if not use_example:
    t = time.time()
    safe_dir = download_scene(scene.name, WORK / "safe")
    print(f"Downloaded {safe_dir} in {time.time() - t:.0f} s")
else:
    if scene.name != EXAMPLE_NAME or abs(LAT - config.DEFAULT_LAT) > 0.2 or abs(LON - config.DEFAULT_LON) > 0.2:
        print("Without AWS credentials and SNAP only the example scene can be used: switching to it.")
    print("Skipping download: using the preprocessed example scene", EXAMPLE_NAME)
"""),
    md("""
## d. Preprocess with ESA SNAP

Same chain and parameters as the paper (SNAP Python API):
Apply-Orbit-File (precise orbits) → Remove-GRD-Border-Noise → ThermalNoiseRemoval → Calibration (σ⁰) →
Terrain-Correction (GETASSE30 DEM, bilinear, WGS84, 10 m) → dB → subset to the area of interest →
8 bits with `(max(-49, min(1, dB)) + 50) × 5`. Takes 3-10 minutes for the default area.
"""),
    code("""
if not use_example:
    from s1water.preprocess import preprocess_scene
    t = time.time()
    img8_path = preprocess_scene(safe_dir, WORK, aoi_wkt=AOI)
    print(f"Preprocessed in {time.time() - t:.0f} s:", img8_path)
else:
    img8_path = download_example_scene()
    print("Preprocessed example scene:", img8_path)
img8, profile = read_8bit(img8_path)
print("Image size:", img8.shape, "| CRS:", profile["crs"], "| pixel:", profile["transform"].a, "deg")
"""),
    md("""
## e. Prediction on the GPU

The PyTorch weights (port of the original Keras model, validated: identical masks on the 2,379
validation tiles of the paper) are downloaded from Hugging Face. The image is cut into 4224 × 4224 tiles
every 4096 pixels; the model drops 64 pixels on each side, so the 4096 × 4096 predictions join without
overlap, as in the original workflow. Water = probability ≥ 0.5, computed in float16 on GPU like the original.
"""),
    code("""
device = get_device()
weights = download_weights()
model = load_model(weights, device)
t = time.time()
prob = predict_image(model, img8, device, return_probability=True)
print(f"Prediction: {time.time() - t:.1f} s on {device}")
raw = np.full(prob.shape, 255, np.uint8)
ok = ~np.isnan(prob)
raw[ok] = prob[ok] >= 0.5
raw[-CROP:, :] = 255  # image edges without the 64 px context
raw[:, -CROP:] = 255
"""),
    md("""
## f. Post-processing

As in the paper: (1) scene-border mask (no-data pixels and the noisy scene edges, 1000 m on the range
sides and 500 m on the azimuth sides) and (2) mountain-shade mask (SRTM slopes ≥ 20°, convex hulls),
because radar shadows look like water. Masked pixels are set to no data (255).
"""),
    code("""
valid = valid_data_mask(img8, profile["transform"], profile["crs"])
shade = shade_mask(raw.shape, profile["transform"], profile["crs"], cache_dir=WORK / "srtm")
water = apply_masks(raw, valid, shade)
lat_c = profile["transform"].f + profile["transform"].e * water.shape[0] / 2
px_km2 = (profile["transform"].a * 111.32 * np.cos(np.radians(lat_c))) * (abs(profile["transform"].e) * 111.32)
print(f"Water: {(water == 1).sum() * px_km2:,.1f} km2 | not water: {(water == 0).sum() * px_km2:,.1f} km2 | "
      f"no data: {(water == 255).sum() * px_km2:,.1f} km2 (of which mountain shade: {shade.sum() * px_km2:,.1f} km2)")
"""),
    md("""
## g. Results and GeoTIFF export

Layers: Esri satellite basemap, Sentinel-1 VV image, water mask (blue). Use the layer control (top right)
to switch them on and off. The GeoTIFF (Cloud Optimized, EPSG:4326, 10 m) codes 0 = not water,
1 = water, 255 = no data.
"""),
    code("""
viz.result_map(img8, water, profile["transform"])
"""),
    code("""
fig = viz.figure(img8, water, title=Path(img8_path).name)
out_tif = write_mask(WORK / (Path(img8_path).name.replace("_VV_VH_8bits.tif", "") + "_water.tif"), water, profile)
print("GeoTIFF written:", out_tif)
"""),
]

COLAB_END = [
    code("""
# download the GeoTIFF to your computer (also visible in the Files panel on the left)
try:
    from google.colab import files
    files.download(str(out_tif))
except Exception as err:
    print("Browser download not available here:", err, "| file:", out_tif.resolve())
"""),
]

CITE = md("""
## Citation

Wagner, F.H.; Favrichon, S.; Dalagnol, R.; Hirye, M.C.M.; Mullissa, A.; Saatchi, S. The Amazon's 2023
Drought: Sentinel-1 Reveals Extreme Rio Negro River Contraction. *Remote Sensing* **2024**, 16, 1056.
https://doi.org/10.3390/rs16061056

**Limitations.** Trained on the Amazon (2021-2022, Sentinel-1A); small rivers of 1-2 pixels (10-20 m)
are often missed; flooded vegetation is not mapped as water; strong convective rain cells can hide water
(the paper corrected them with a time series, not done here).
""")


def build(colab: bool):
    nb = nbf.v4.new_notebook()
    nb.cells = [intro(colab)] + (COLAB_SETUP if colab else JUPYTER_SETUP) + COMMON + (COLAB_END if colab else []) + [CITE]
    # fixed cell ids, so that rebuilding gives identical files (checked in CI)
    nb.cells = [nbf.from_dict({**c, "id": f"cell-{i:02d}"}) for i, c in enumerate(nb.cells)]
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    if colab:
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks"
    nbf.write(build(True), out / "water_sentinel1_colab.ipynb")
    nbf.write(build(False), out / "water_sentinel1_jupyterlab.ipynb")
    print("written", list(out.glob("*.ipynb")))
