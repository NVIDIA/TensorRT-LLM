import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.components.patchifiers import get_pixel_coords
from tensorrt_llm._torch.visual_gen.models.ltx_core.conditioning.item import ConditioningItem
from tensorrt_llm._torch.visual_gen.models.ltx_core.conditioning.mask_utils import extend_keyframes_mask, update_attention_mask
from tensorrt_llm._torch.visual_gen.models.ltx_core.tools import VideoLatentTools
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import DenoiseMask, LatentState, VideoLatentShape


class VideoConditionByKeyframeIndex(ConditioningItem):
    """
    Conditions video generation on keyframe latents at a specific frame index.
    Appends keyframe tokens to the sequence with positions offset by frame_idx: the keyframe
    latents become clean-latent tokens (placeholder zeros in the noisy latent) and the denoise
    mask is set from the strength parameter.
    To add attention masking, wrap with :class:`ConditioningItemAttentionStrengthWrapper`.
    Args:
        keyframes: Keyframe latents [B, C, F, H, W].
        frame_idx: Frame index offset for positional encoding.
        strength: Conditioning strength (1.0 = clean, 0.0 = fully denoised).
        num_pixel_frames: Number of pixel frames the keyframe latent originally encodes.
    """

    def __init__(
        self,
        keyframes: torch.Tensor,
        frame_idx: int,
        strength: float,
        num_pixel_frames: int = 1,
    ):
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength
        self.num_pixel_frames = num_pixel_frames

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.keyframes)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.keyframes.shape),
            device=self.keyframes.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix if self.frame_idx == 0 else False,
        )

        positions[:, 0, ...] += self.frame_idx
        # If the keyframe latent encodes a single pixel frame,
        # narrow the temporal end to [start, start + 1) instead of the
        # VAE-scaled range.
        if self.num_pixel_frames == 1:
            positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= latent_tools.fps

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device="cpu",
            dtype=self.keyframes.dtype,
        )

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=self.keyframes.device,
            dtype=self.keyframes.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, torch.zeros_like(tokens)], dim=1),
            denoise_mask=DenoiseMask(torch.cat([latent_state.denoise_mask.tensor, denoise_mask], dim=1)),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
            # Given keyframe content is ordinary image guidance, not a generated keyframe slot, so
            # it carries no keyframe marker.
            keyframes_mask=extend_keyframes_mask(latent_state, tokens.shape[1], marked=False),
            generated_keyframe_layout=latent_state.generated_keyframe_layout,
            generated_keyframes=latent_state.generated_keyframes,
            frozen=latent_state.frozen,
        )
