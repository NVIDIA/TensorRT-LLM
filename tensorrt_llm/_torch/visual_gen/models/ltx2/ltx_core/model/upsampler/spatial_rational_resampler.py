from typing import Tuple

import torch
from einops import rearrange

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.upsampler.blur_downsample import BlurDownsample
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.upsampler.pixel_shuffle import PixelShuffleND


def _rational_for_scale(scale: float) -> Tuple[int, int]:
    mapping = {0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)}
    if float(scale) not in mapping:
        raise ValueError(f"Unsupported scale {scale}. Choose from {list(mapping.keys())}")
    return mapping[float(scale)]


class SpatialRationalResampler(torch.nn.Module):
    """
    Fully-learned rational spatial scaling: up by 'num' via PixelShuffle, then anti-aliased
    downsample by 'den' using fixed blur + stride. Operates on H,W only.
    For dims==3, work per-frame for spatial scaling (temporal axis untouched).
    Args:
        mid_channels (`int`): Number of intermediate channels for the convolution layer
        scale (`float`): Spatial scaling factor. Supported values are:
            - 0.75: Downsample by 3/4 (reduce spatial size)
            - 1.5: Upsample by 3/2 (increase spatial size)
            - 2.0: Upsample by 2x (double spatial size)
            - 4.0: Upsample by 4x (quadruple spatial size)
            Any other value will raise a ValueError.
    """

    def __init__(self, mid_channels: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.num, self.den = _rational_for_scale(self.scale)
        self.conv = torch.nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        return x
