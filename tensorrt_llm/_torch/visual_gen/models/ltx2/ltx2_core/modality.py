# Ported from https://github.com/Lightricks/LTX-2
# packages/ltx-core/src/ltx_core/model/transformer/modality.py

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """Input data for a single modality (video or audio) in the transformer.

    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    """

    latent: torch.Tensor  # (B, T, D): packed latent tokens
    timesteps: torch.Tensor  # (B,) or (B, T): per-batch or per-token timesteps
    positions: torch.Tensor  # (B, n_dims, T) or (B, n_dims, T, 2): index grid
    context: torch.Tensor  # Text embeddings
    enabled: bool = True
    context_mask: torch.Tensor | None = None
