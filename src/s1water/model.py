"""U-Net for surface water segmentation from Sentinel-1 (PyTorch port).

This is a line-by-line port of the R/Keras ``get_unet_128()`` model used in
Wagner et al. (2024), Remote Sensing 16(6):1056, https://doi.org/10.3390/rs16061056.
The Keras code was adapted from the RStudio AI blog U-Net post
(https://blogs.rstudio.com/ai/posts/2019-08-23-unet/), itself based on
Ronneberger et al. (2015).

Architecture (identical to the Keras original):
- encoder: 5 levels of [conv3x3-BN-ReLU] x2 (32, 64, 128, 256, 512 filters),
  each followed by 2x2 max pooling and dropout(0.5)
- centre: [conv3x3-BN-ReLU] x2 with 1024 filters
- decoder: nearest upsampling x2, concatenation [skip, upsampled], then
  [conv3x3-BN-ReLU] x3 (512, 256, 128, 64, 32 filters)
- optional cropping of the border (64 px for scene prediction, 0 for 256 px tiles)
- 1x1 conv + sigmoid -> water probability

Input: float tensor (N, 4, H, W) with values in [0, 1] (8-bit image / 255),
band order VV, VH, VV, VH. H and W must be multiples of 32.
"""

from __future__ import annotations

import torch
from torch import nn

# Keras BatchNormalization default epsilon (PyTorch default is 1e-5).
KERAS_BN_EPS = 1e-3

ENCODER_FILTERS = (32, 64, 128, 256, 512)
CENTER_FILTERS = 1024


class ConvBNReLU(nn.Sequential):
    """3x3 convolution ('same' padding) + batch normalisation + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch, eps=KERAS_BN_EPS, momentum=0.01),
            nn.ReLU(inplace=True),
        )


class UNetS1Water(nn.Module):
    """U-Net returning the probability of surface water for each pixel."""

    def __init__(self, in_channels: int = 4, num_classes: int = 1, dropout: float = 0.5):
        super().__init__()
        self.down = nn.ModuleList()
        ch = in_channels
        for f in ENCODER_FILTERS:
            self.down.append(nn.Sequential(ConvBNReLU(ch, f), ConvBNReLU(f, f)))
            ch = f
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(dropout)
        self.center = nn.Sequential(ConvBNReLU(ch, CENTER_FILTERS), ConvBNReLU(CENTER_FILTERS, CENTER_FILTERS))
        ch = CENTER_FILTERS
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.ModuleList()
        for f in reversed(ENCODER_FILTERS):
            # concatenation of the skip connection (f channels) and the upsampled map (ch channels)
            self.up.append(nn.Sequential(ConvBNReLU(f + ch, f), ConvBNReLU(f, f), ConvBNReLU(f, f)))
            ch = f
        self.classify = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, crop: int = 0) -> torch.Tensor:
        """Return water probabilities (N, 1, H - 2*crop, W - 2*crop)."""
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.drop(self.pool(x))
        x = self.center(x)
        for block in self.up:
            # pop the skip connection so its memory is released as soon as it is used
            # (a 4224 px tile then fits in the 16 GB of a T4 GPU in float16)
            x = torch.cat([skips.pop(), self.upsample(x)], dim=1)
            x = block(x)
        if crop > 0:
            x = x[..., crop:-crop, crop:-crop]
        # the sigmoid is computed in float32, as in the Keras model (dtype='float32')
        return torch.sigmoid(self.classify(x).float())


def load_model(weights_path: str, device: str | torch.device = "cpu", half: bool | None = None) -> UNetS1Water:
    """Build the model, load PyTorch weights (.pt or .safetensors) and set eval mode.

    ``half`` (default: True on GPU) converts the model to float16, like the original
    mixed-precision prediction; a 4224 px tile then needs ~10 GB of GPU memory instead of ~15 GB.
    """
    model = UNetS1Water()
    if str(weights_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(str(weights_path))
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device(device)
    if half is None:
        half = device.type == "cuda"
    model = model.to(device).eval()
    return model.half() if half else model
