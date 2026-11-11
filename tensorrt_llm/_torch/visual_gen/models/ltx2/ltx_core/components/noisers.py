from dataclasses import replace
from typing import Protocol

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import LatentState


class Noiser(Protocol):
    """Protocol for adding noise to a latent state during diffusion."""

    def __call__(self, latent_state: LatentState, noise_scale: float) -> LatentState: ...


class GaussianNoiser(Noiser):
    """Adds Gaussian noise to a latent state, scaled by the denoise mask."""

    def __init__(self, generator: torch.Generator):
        super().__init__()
        self.generator = generator

    def _sample_noise(self, latent_state: LatentState) -> torch.Tensor:
        return torch.randn(
            *latent_state.latent.shape,
            device=latent_state.latent.device,
            dtype=latent_state.latent.dtype,
            generator=self.generator,
        )

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        noise = self._sample_noise(latent_state)
        latent = torch.lerp(latent_state.latent.float(), noise.float(), noise_scale)
        latent = torch.lerp(latent_state.clean_latent.float(), latent, latent_state.denoise_mask.tensor)
        return replace(
            latent_state,
            latent=latent.to(latent_state.latent.dtype),
        )
