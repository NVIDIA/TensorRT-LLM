"""Mask-based conditioning for inpainting and spatial conditioning."""

from dataclasses import replace

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.conditioning.item import ConditioningItem
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tools import LatentTools
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import DenoiseMask, LatentState


class VideoConditionByMask(ConditioningItem):
    """Condition video generation using a binary mask over latent frames.
    Masked positions (mask=1) receive the provided clean latent values and are
    excluded from denoising (denoise_mask set to ``1 - strength``). Unmasked
    positions (mask=0) are left unchanged and denoised normally.
    The mask operates in **unpatchified latent** space — it should have shape
    ``[B, F, H, W]`` matching the latent dimensions (after VAE encoding,
    before patchification). This is consistent with the latent input format
    used by all other conditioning items.
    Args:
        latent: Clean conditioning latents in unpatchified format [B, C, F, H, W].
            Must match the target shape of the latent tools.
        mask: Binary mask [B, F, H, W] in unpatchified latent space.
            1 = conditioning position (clean, excluded from denoising),
            0 = generated position (noised, denoised normally).
        strength: Conditioning strength for masked positions. 1.0 = fully clean
            (no denoising), 0.0 = no conditioning effect. Default 1.0.
    """

    def __init__(self, latent: torch.Tensor, mask: torch.Tensor, strength: float = 1.0):
        self.latent = latent
        self.mask = mask
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        """Apply mask-based conditioning to the latent state."""
        tokens = latent_tools.patchifier.patchify(self.latent)

        mask = latent_tools.patchifier.patchify(self.mask.unsqueeze(1))

        m = mask.to(dtype=latent_state.latent.dtype)
        inv = 1 - m

        return replace(
            latent_state,
            clean_latent=latent_state.clean_latent * inv + tokens * m,
            denoise_mask=DenoiseMask(latent_state.denoise_mask.tensor * inv.cpu() + (1.0 - self.strength) * m.cpu()),
        )
