"""Audio reference conditioning items."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.conditioning.mask_utils import extend_keyframes_mask, update_attention_mask
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tools import LatentTools
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import DenoiseMask, LatentState


class AudioConditionByReferenceLatent:
    """Append patchified reference audio tokens after the target audio sequence.
    Mirrors :class:`ltx_core.conditioning.types.reference_video_cond.VideoConditionByReferenceLatent`
    but for audio. The reference tokens are appended so the target audio tokens stay
    in the first ``num_noisy_tokens`` positions and can be kept by
    :meth:`ltx_core.tools.LatentTools.clear_conditioning`.
    Args:
        patchified: Patchified reference latent ``[B, T_ref, C]``.
        positions: RoPE positions for reference tokens, ``[B, 1, T_ref, 2]``.
        strength: 1.0 keeps reference clean; 0.0 would fully denoise it.
    """

    def __init__(
        self,
        patchified: torch.Tensor,
        positions: torch.Tensor,
        strength: float = 1.0,
    ) -> None:
        self.patchified = patchified
        self.positions = positions.to(dtype=torch.float32)
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        tokens = self.patchified
        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device="cpu",
            dtype=tokens.dtype,
        )

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.patchifier.get_token_count(latent_tools.target_shape),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=tokens.device,
            dtype=tokens.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, torch.zeros_like(tokens)], dim=1),
            denoise_mask=DenoiseMask(torch.cat([latent_state.denoise_mask.tensor, denoise_mask], dim=1)),
            positions=torch.cat([latent_state.positions, self.positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
            # Audio states carry no keyframe marker, so this resolves to None; called anyway to keep
            # the invariant that every appending conditioning item extends the per-token fields.
            keyframes_mask=extend_keyframes_mask(latent_state, tokens.shape[1], marked=False),
            generated_keyframe_layout=latent_state.generated_keyframe_layout,
            generated_keyframes=latent_state.generated_keyframes,
            frozen=latent_state.frozen,
        )
