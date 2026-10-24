"""s1water: surface water segmentation from Sentinel-1 SAR with a U-Net (Wagner et al. 2024)."""

__version__ = "1.0.1"

from .model import UNetS1Water, load_model  # noqa: F401
