# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2x
"""Native IC-LoRA latent state operations, adapted from LTX-2 ICLoraPipeline.

Keep the reference's BF16 rounding boundaries and random draw order. In
particular, reference tokens participate in noise draws even when kept clean.
"""

from dataclasses import dataclass, replace

import torch

from ..ltx2.ltx2_core.patchifier import VideoLatentPatchifier, get_pixel_coords
from ..ltx2.ltx2_core.types import VIDEO_SCALE_FACTORS, VideoLatentShape

STAGE_1_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
STAGE_2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)


@dataclass(frozen=True)
class ICLoraState:
    """Patchified latent and conditioning tensors for one modality."""

    latent: torch.Tensor
    clean: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor
    keyframes_mask: torch.Tensor | None = None


def video_state(latent: torch.Tensor, fps: float) -> ICLoraState:
    """Create target state from BF16 ``[B, C, F, H, W]`` latents."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    patchifier = VideoLatentPatchifier(patch_size=1)
    shape = VideoLatentShape.from_torch_shape(latent.shape)
    positions = get_pixel_coords(
        patchifier.get_patch_grid_bounds(shape, device=latent.device),
        VIDEO_SCALE_FACTORS,
        causal_fix=True,
    ).float()
    positions[:, 0] /= fps
    tokens = patchifier.patchify(latent)
    mask = torch.ones((*tokens.shape[:2], 1), device=latent.device, dtype=torch.float32)
    keyframes = torch.zeros_like(mask)
    keyframes[:, : shape.height * shape.width] = 1
    return ICLoraState(tokens, tokens.clone(), mask, positions, keyframes)


def append_reference(
    state: ICLoraState,
    latent: torch.Tensor,
    fps: float,
    spatial_scale: int = 1,
    temporal_scale: int = 1,
    strength: float = 1.0,
) -> ICLoraState:
    """Append clean reference tokens; never mark their causal frame as a keyframe."""
    if spatial_scale < 1 or temporal_scale < 1:
        raise ValueError("Reference scale factors must be positive")
    if not 0 <= strength <= 1:
        raise ValueError("Reference strength must be in [0, 1]")
    reference = video_state(latent, fps / temporal_scale)
    positions = reference.positions
    if temporal_scale != 1:
        first_frame_end = state.positions[:, 0, 0:1, 1:2]
        positions[:, 0] = (positions[:, 0] - (temporal_scale - 1) * first_frame_end).clamp(min=0)
    positions[:, 1:] *= spatial_scale
    mask = torch.full(
        (*reference.latent.shape[:2], 1),
        1.0 - strength,
        device=latent.device,
        dtype=latent.dtype,
    )
    keyframes = state.keyframes_mask
    if keyframes is not None:
        keyframes = torch.cat([keyframes, torch.zeros_like(mask)], dim=1)
    return ICLoraState(
        torch.cat([state.latent, torch.zeros_like(reference.latent)], dim=1),
        torch.cat([state.clean, reference.latent], dim=1),
        torch.cat([state.mask, mask], dim=1),
        torch.cat([state.positions, positions], dim=2),
        keyframes,
    )


def add_noise(state: ICLoraState, generator: torch.Generator, scale: float) -> ICLoraState:
    """Noise all tokens before blending in clean reference conditioning."""
    noise = torch.randn(
        state.latent.shape,
        device=state.latent.device,
        dtype=state.latent.dtype,
        generator=generator,
    )
    latent = torch.lerp(state.latent.float(), noise.float(), scale)
    latent = torch.lerp(state.clean.float(), latent, state.mask.float())
    return replace(state, latent=latent.to(state.latent.dtype))


def euler_step(
    state: ICLoraState,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    *,
    sigma_value: float | None = None,
) -> ICLoraState:
    """Match X0Model -> conditioning blend -> EulerDiffusionStep, including casts."""
    dtype = state.latent.dtype
    denoised = (state.latent.float() - velocity.float() * (state.mask * sigma)).to(dtype)
    denoised = (denoised * state.mask + state.clean.float() * (1 - state.mask)).to(dtype)
    # Reference to_velocity divides by a Python scalar, not a CUDA tensor.
    # CUDA scalar division uses a reciprocal multiply; tensor division can
    # differ by an FP32 ULP and cross a subsequent BF16 rounding boundary.
    if sigma_value is None:
        sigma_value = sigma.item()
    derivative = ((state.latent.float() - denoised.float()) / sigma_value).to(dtype)
    latent = (state.latent.float() + derivative.float() * (next_sigma - sigma)).to(dtype)
    return replace(state, latent=latent)
