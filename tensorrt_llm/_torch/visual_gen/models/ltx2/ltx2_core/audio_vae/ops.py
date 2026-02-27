# Ported from https://github.com/Lightricks/LTX-2
# packages/ltx-core/src/ltx_core/model/audio_vae/ops.py

import torch
from torch import nn


class PerChannelStatistics(nn.Module):
    """Per-channel statistics for normalizing/denormalizing audio latents."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer(
            "mean-of-means"
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer(
            "std-of-means"
        ).to(x)
