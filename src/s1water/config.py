"""Project-wide settings (repository names, defaults of the tutorial)."""

import os

# Hugging Face model repository holding the PyTorch weights and the model card.
HF_REPO_ID = os.environ.get("S1WATER_HF_REPO", "Fabien-Ctrees/sentinel1-water-unet")
HF_WEIGHTS_FILE = "s1_water_unet.safetensors"
# Pre-processed Anavilhanas scene (fallback when SNAP cannot be installed, e.g. on Colab)
HF_EXAMPLE_FILE = "examples/anavilhanas_S1D_20260828_VV_VH_8bits.tif"

GITHUB_REPO = os.environ.get("S1WATER_GITHUB_REPO", "fabien-w/sentinel1-water-unet")

# Default point of the tutorial: Anavilhanas archipelago, Rio Negro, Brazil (lat, lon).
# Checked with JRC Global Surface Water: the 4 x 4 km box around it mixes permanent
# water channels (48 %) and islands (35 %).
DEFAULT_LAT, DEFAULT_LON = -2.70, -60.80
DEFAULT_START, DEFAULT_END = "2026-08-01", "2026-08-31"
# Half size (degrees) of the default area of interest processed around the point
DEFAULT_AOI_HALF_SIZE = 0.25

PAPER_DOI = "10.3390/rs16061056"
