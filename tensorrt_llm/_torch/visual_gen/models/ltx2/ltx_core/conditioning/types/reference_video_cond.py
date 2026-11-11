"""Reference video conditioning for IC-LoRA inference."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.components.patchifiers import get_pixel_coords
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.conditioning.item import ConditioningItem
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.conditioning.mask_utils import extend_keyframes_mask, update_attention_mask
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tools import VideoLatentTools
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import DenoiseMask, LatentState, VideoLatentShape


class VideoConditionByReferenceLatent(ConditioningItem):
    """
    Conditions video generation on a reference video latent for IC-LoRA inference.
    IC-LoRAs are trained by concatenating reference (control signal) and target tokens,
    learning to attend across both. This class replicates that setup at inference by
    appending the reference tokens to the sequence as clean latents (with placeholder zeros
    in the noisy latent).
    IC-LoRAs can be trained with lower-resolution references than the target (e.g., 384px
    reference for 768px output) for efficiency and better generalization. The
    `downscale_factor` scales reference positions to match target coordinates, preserving
    the learned positional relationships. This must match the factor used during training
    (stored in LoRA metadata).
    To add attention masking, wrap with :class:`ConditioningItemAttentionStrengthWrapper`.
    Args:
        latent: Reference video latents [B, C, F, H, W].
        downscale_factor: Target/reference spatial ratio (e.g. 2 = half-res ref).
        temporal_scale_factor: Target/reference temporal ratio S (e.g. 4 = ref at 1/4 fps).
        first_latent_frame: Temporal-grid index assigned to the first frame of ``latent``.
        strength: Conditioning strength. 1.0 = full (reference kept clean),
            0.0 = none (reference denoised). Default 1.0.
    """

    def __init__(
        self,
        latent: torch.Tensor,
        downscale_factor: int = 1,
        temporal_scale_factor: int = 1,
        strength: float = 1.0,
        first_latent_frame: int = 0,
    ):
        if first_latent_frame < 0:
            raise ValueError(f"first_latent_frame must be >= 0, got {first_latent_frame}")
        self.latent = latent
        self.downscale_factor = downscale_factor
        self.temporal_scale_factor = temporal_scale_factor
        self.strength = strength
        self.first_latent_frame = first_latent_frame

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """Append reference video tokens with positions translated into the target frame."""
        tokens = latent_tools.patchifier.patchify(self.latent)

        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.latent.shape),
            device=self.latent.device,
        )
        if self.first_latent_frame:
            latent_coords = latent_coords.clone()
            latent_coords[:, 0, :, :] += self.first_latent_frame
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        )
        positions = positions.to(dtype=torch.float32)

        # Place ref tokens on their own time spacing (= target_fps / S).
        positions[:, 0, ...] /= latent_tools.fps / self.temporal_scale_factor

        # Translate into the target's frame so ref's last patch ends with target's last
        # patch; clamp the causal patch's negative start back to [0, 1/target_fps).
        if self.temporal_scale_factor != 1:
            t_target = latent_state.positions[:, 0, 0:1, 1:2].to(dtype=torch.float32)  # = 1/target_fps
            positions[:, 0, ...] = torch.clamp(
                positions[:, 0, ...] - (self.temporal_scale_factor - 1) * t_target,
                min=0,
            )
        if self.downscale_factor != 1:
            positions[:, 1, ...] *= self.downscale_factor  # height axis
            positions[:, 2, ...] *= self.downscale_factor  # width axis

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device="cpu",
            dtype=self.latent.dtype,
        )

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=self.latent.device,
            dtype=self.latent.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, torch.zeros_like(tokens)], dim=1),
            denoise_mask=DenoiseMask(torch.cat([latent_state.denoise_mask.tensor, denoise_mask], dim=1)),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
            # Reference tokens are never keyframes. Their own first latent frame also spans a single
            # pixel frame, so a position-derived marker would wrongly claim them.
            keyframes_mask=extend_keyframes_mask(latent_state, tokens.shape[1], marked=False),
            generated_keyframe_layout=latent_state.generated_keyframe_layout,
            generated_keyframes=latent_state.generated_keyframes,
            frozen=latent_state.frozen,
        )
