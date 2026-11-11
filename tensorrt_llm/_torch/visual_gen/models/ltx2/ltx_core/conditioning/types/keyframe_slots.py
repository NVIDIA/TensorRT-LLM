"""Generated keyframe slots: extra single-pixel-frame tokens denoised with the target.
Unlike every other conditioning item, this one supplies no content by default. It appends empty,
fully-denoised token slots whose RoPE position marks a single interior pixel frame, letting the
model generate additional frames at those positions and condition the surrounding video on them.
That relaxes the effective temporal compression ratio at those positions, at the cost of one
latent frame's worth of tokens per keyframe (which buys 1 pixel frame instead of the usual 8).
Optional ``initial_keyframes`` seeds the appended ``latent`` tokens (not ``clean_latent``) so the
noiser can lerp from that content when ``denoise_mask=1`` (``clean`` is ignored at mask 1).
Requires a checkpoint trained for it -- one whose transformer config sets
``use_keyframes_abs_pos_embedding``. On any other checkpoint the learned marker does not exist,
so the slots would be denoised as unmarked tokens and the extra compute would be wasted;
pipelines validate this up front rather than silently degrading.
"""

import itertools
from collections.abc import Sequence

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.components.patchifiers import get_pixel_coords
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.conditioning.item import ConditioningItem
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.conditioning.mask_utils import extend_keyframes_mask, update_attention_mask
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tools import VideoLatentTools
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import DenoiseMask, GeneratedKeyframeLayout, LatentState


class VideoGeneratedKeyframeSlots(ConditioningItem):
    """Appends fully-denoised keyframe slots at the given target pixel-frame indices.
    Each slot occupies one latent frame's worth of tokens at the target's spatial resolution, with
    ``denoise_mask = 1`` so the noiser fills it from the slot ``latent`` (zeros, or
    ``initial_keyframes`` when provided) and the denoising loop generates its content. The slot is
    marked in ``keyframes_mask`` so it receives the model's learned keyframe absolute-position
    embedding.
    Deliberately separate from :class:`VideoConditionByKeyframeIndex`, which appends *given*
    keyframe content for ordinary image guidance and must not be marked: the reference
    implementation only sets the keyframe marker for the keyframe conditioning methods, not for
    plain image / first-and-last-frame conditioning.
    All slots are appended by a single item so the resulting token range is contiguous and can be
    recorded as one :class:`GeneratedKeyframeLayout`.
    Args:
        pixel_frame_indices: Target pixel-frame index for each slot. Must be non-empty, strictly
            increasing, and within the target's frame range.
        initial_keyframes: Optional ``(B, C, K, H, W)`` latent content written into the appended
            ``latent`` tokens before noising. ``K`` must equal ``len(pixel_frame_indices)``.
    """

    def __init__(
        self,
        pixel_frame_indices: Sequence[int],
        initial_keyframes: torch.Tensor | None = None,
    ):
        indices = tuple(int(index) for index in pixel_frame_indices)
        if not indices:
            raise ValueError("pixel_frame_indices must be non-empty")
        if any(index < 0 for index in indices):
            raise ValueError(f"pixel_frame_indices must be non-negative, got {indices}")
        if any(b <= a for a, b in itertools.pairwise(indices)):
            raise ValueError(f"pixel_frame_indices must be strictly increasing, got {indices}")
        if initial_keyframes is not None:
            if initial_keyframes.ndim != 5:
                raise ValueError(
                    f"initial_keyframes must have shape (B, C, K, H, W), got {tuple(initial_keyframes.shape)}"
                )
            if initial_keyframes.shape[2] != len(indices):
                raise ValueError(
                    f"initial_keyframes K={initial_keyframes.shape[2]} must match {len(indices)} pixel_frame_indices"
                )
        self.pixel_frame_indices = indices
        self.initial_keyframes = initial_keyframes

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        num_pixel_frames = latent_tools.target_shape.upscale(latent_tools.scale_factors).frames
        if self.pixel_frame_indices[-1] >= num_pixel_frames:
            raise ValueError(
                f"Generated keyframe at pixel frame {self.pixel_frame_indices[-1]} is outside the target's "
                f"{num_pixel_frames} frames"
            )

        frame_shape = latent_tools.target_shape._replace(frames=1)
        tokens_per_keyframe = latent_tools.patchifier.get_token_count(frame_shape)
        device = latent_state.latent.device
        dtype = latent_state.latent.dtype

        positions = torch.cat(
            [self._slot_positions(index, latent_tools, device) for index in self.pixel_frame_indices],
            dim=2,
        )

        num_new_tokens = tokens_per_keyframe * len(self.pixel_frame_indices)
        batch_size = latent_state.latent.shape[0]
        if self.initial_keyframes is None:
            slot_tokens = torch.zeros(
                batch_size, num_new_tokens, latent_state.latent.shape[2], device=device, dtype=dtype
            )
        else:
            initials = self.initial_keyframes.to(device=device, dtype=dtype)
            if initials.shape[0] != batch_size:
                raise ValueError(
                    f"initial_keyframes batch {initials.shape[0]} does not match latent batch {batch_size}"
                )
            expected_hw = (latent_tools.target_shape.height, latent_tools.target_shape.width)
            if (initials.shape[3], initials.shape[4]) != expected_hw:
                raise ValueError(
                    f"initial_keyframes spatial size {(initials.shape[3], initials.shape[4])} does not match "
                    f"target latent spatial size {expected_hw}"
                )
            slot_tokens = torch.cat(
                [
                    latent_tools.patchifier.patchify(initials[:, :, index : index + 1])
                    for index in range(initials.shape[2])
                ],
                dim=1,
            )
        # denoise_mask 1 => the noiser lerps slot latent toward noise (clean_latent is ignored).
        denoise_mask = torch.ones(batch_size, num_new_tokens, 1, dtype=latent_state.denoise_mask.tensor.dtype)

        keyframes_mask = extend_keyframes_mask(latent_state, num_new_tokens, marked=True)

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=num_new_tokens,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        if latent_state.generated_keyframe_layout is not None:
            raise ValueError("Generated keyframe slots were already applied to this state; use a single item")

        return LatentState(
            latent=torch.cat([latent_state.latent, slot_tokens], dim=1),
            denoise_mask=DenoiseMask(torch.cat([latent_state.denoise_mask.tensor, denoise_mask], dim=1)),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, torch.zeros_like(slot_tokens)], dim=1),
            attention_mask=new_attention_mask,
            keyframes_mask=keyframes_mask,
            generated_keyframe_layout=GeneratedKeyframeLayout(
                pixel_frame_indices=self.pixel_frame_indices,
                tokens_per_keyframe=tokens_per_keyframe,
                first_token=latent_state.latent.shape[1],
            ),
            generated_keyframes=latent_state.generated_keyframes,
            frozen=latent_state.frozen,
        )

    @staticmethod
    def _slot_positions(
        pixel_frame_index: int,
        latent_tools: VideoLatentTools,
        device: torch.device,
    ) -> torch.Tensor:
        """Positions for one slot: full spatial grid, temporal span exactly ``[t, t + 1)``.
        The single-pixel-frame temporal extent is what distinguishes a slot from a regular latent
        frame (which spans ``temporal_scale_factor`` pixel frames) in RoPE space.
        """
        frame_shape = latent_tools.target_shape._replace(frames=1)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(output_shape=frame_shape, device=device)
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            # causal_fix collapses frame 0's span to a single frame; here the span is set
            # explicitly below, so it must not also be applied.
            causal_fix=False,
        ).to(dtype=torch.float32)
        positions[:, 0, ...] = float(pixel_frame_index)
        positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
        positions[:, 0, ...] /= latent_tools.fps
        return positions
