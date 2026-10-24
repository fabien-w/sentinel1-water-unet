"""Water prediction on a full Sentinel-1 image, with the tiling of the original workflow.

Original workflow (R scripts 29.03 and 30 of the paper):
  - the 2-band 8-bit image (VV, VH) is padded on the right and bottom to a multiple of
    4096 pixels plus a 128-pixel border (gdal_translate -srcwin 0 0 W' H')
  - it is cut into 4224 x 4224 tiles every 4096 pixels (gdal_retile -ps 4224 4224 -overlap 128)
  - each tile is fed to the U-Net as 4 bands VV, VH, VV, VH scaled to [0, 1] (x / 255)
  - the model crops 64 pixels on each side, so each tile gives a 4096 x 4096 prediction
    that covers image pixels [64, 4160) of the tile; tiles do not overlap after cropping
  - water = probability >= 0.5, computed in mixed precision (float16) on the GPU
    (here the whole model is converted to float16, which also halves the GPU memory)

Here the predictions are written on the same pixel grid as the input image; the first
64 rows and columns, never predicted by the original workflow, are set to no data.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from .model import UNetS1Water

log = logging.getLogger(__name__)

TILE = 4224
STEP = 4096
CROP = 64
THRESHOLD = 0.5
# Fill value of the padding: gdal_translate -srcwin outside the image writes the band
# no-data value, which was 255 for the original 8-bit files written by gdal_calc.
PAD_VALUE = 255


def get_device(prefer: str | None = None) -> torch.device:
    """Return the GPU if available, else the CPU (with a clear message)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    log.warning("No GPU found: running on CPU. This works but is slow (several minutes per 4224 px tile).")
    return torch.device("cpu")


def tile_origins(height: int, width: int, step: int = STEP) -> list[tuple[int, int]]:
    """Top-left corners (row, col) of the 4224 px tiles, as gdal_retile produced them."""
    hp = math.ceil(height / step) * step
    wp = math.ceil(width / step) * step
    return [(r, c) for r in range(0, hp, step) for c in range(0, wp, step)]


@torch.inference_mode()
def predict_image(
    model: UNetS1Water,
    img8: np.ndarray,
    device: torch.device | str = "cpu",
    return_probability: bool = False,
    progress: bool = True,
    tile: int = TILE,
    step: int = STEP,
    crop: int = CROP,
) -> np.ndarray:
    """Predict water on a 2-band 8-bit image (2, H, W) with band order VV, VH.

    Returns a float32 probability array (H, W) if ``return_probability``, else a uint8
    mask with 1 = water, 0 = not water and 255 = not predicted (first 64 rows/cols).
    The computation runs in the dtype of the model (float16 on GPU with ``load_model``).
    ``tile``, ``step`` and ``crop`` must keep tile = step + 2 * crop; change them only for tests.
    """
    if tile != step + 2 * crop:
        raise ValueError("tile must equal step + 2 * crop")
    if img8.ndim != 3 or img8.shape[0] != 2 or img8.dtype != np.uint8:
        raise ValueError(f"expected a (2, H, W) uint8 array (VV, VH), got {img8.shape} {img8.dtype}")
    device = torch.device(device)
    dtype = next(model.parameters(), torch.zeros(1)).dtype
    _, h, w = img8.shape
    origins = tile_origins(h, w, step)
    hp = origins[-1][0] + tile
    wp = origins[-1][1] + tile
    padded = np.full((2, hp, wp), PAD_VALUE, dtype=np.uint8)
    padded[:, :h, :w] = img8

    prob = np.full((hp, wp), np.nan, dtype=np.float32)
    iterator = origins
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(origins, desc="tiles")
        except ImportError:  # pragma: no cover
            pass
    model.eval()
    for r, c in iterator:
        t = torch.from_numpy(padded[:, r : r + tile, c : c + tile]).to(device)
        x = t.float().div_(255.0).to(dtype)  # convert_image_dtype(uint8 -> float) then model dtype
        x = torch.cat([x, x], dim=0)[None]  # VV, VH, VV, VH
        p = model(x, crop=crop)
        prob[r + crop : r + crop + step, c + crop : c + crop + step] = p[0, 0].float().cpu().numpy()
        del t, x, p
    prob = prob[:h, :w]
    if return_probability:
        return prob
    mask = np.full((h, w), 255, dtype=np.uint8)
    ok = ~np.isnan(prob)
    mask[ok] = (prob[ok] >= THRESHOLD).astype(np.uint8)
    return mask
