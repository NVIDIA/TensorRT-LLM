from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch._prims_common import DeviceLikeType

from tensorrt_llm._torch.visual_gen.models.ltx_core.components.patchifiers import (
    AudioLatentShape,
    AudioPatchifier,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.components.protocols import Patchifier
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import DenoiseMask, LatentState, SpatioTemporalScaleFactors

DEFAULT_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class LatentTools(Protocol):
    """
    Tools for building latent states.
    """

    patchifier: Patchifier
    target_shape: VideoLatentShape | AudioLatentShape

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        """
        Create an initial latent state. If initial_latent is provided, it will be used to create the latent state.
        """
        ...

    def patchify(self, latent_state: LatentState) -> LatentState:
        """
        Patchify the latent state.
        """
        if latent_state.latent.shape != self.target_shape.to_torch_shape():
            raise ValueError(
                f"Latent state has shape {latent_state.latent.shape}, expected shape is "
                f"{self.target_shape.to_torch_shape()}"
            )
        latent_state = latent_state.clone()
        latent = self.patchifier.patchify(latent_state.latent)
        clean_latent = self.patchifier.patchify(latent_state.clean_latent)
        denoise_mask = latent_state.denoise_mask.with_same_values(
            self.patchifier.patchify(latent_state.denoise_mask.tensor)
        )
        # keyframes_mask shares denoise_mask's layout, so it converts alongside it; leaving it
        # behind would produce a state mixing patchified and unpatchified per-token tensors.
        keyframes_mask = (
            self.patchifier.patchify(latent_state.keyframes_mask) if latent_state.keyframes_mask is not None else None
        )
        return replace(
            latent_state,
            latent=latent,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
            keyframes_mask=keyframes_mask,
        )

    def unpatchify(self, latent_state: LatentState) -> LatentState:
        """
        Unpatchify the latent state.
        """
        latent_state = latent_state.clone()
        latent = self.patchifier.unpatchify(latent_state.latent, output_shape=self.target_shape)
        clean_latent = self.patchifier.unpatchify(latent_state.clean_latent, output_shape=self.target_shape)
        denoise_mask = self.patchifier.unpatchify(
            latent_state.denoise_mask.tensor, output_shape=self.target_shape.mask_shape()
        )
        denoise_mask = latent_state.denoise_mask.with_same_values(denoise_mask)
        keyframes_mask = (
            self.patchifier.unpatchify(latent_state.keyframes_mask, output_shape=self.target_shape.mask_shape())
            if latent_state.keyframes_mask is not None
            else None
        )
        return replace(
            latent_state,
            latent=latent,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
            keyframes_mask=keyframes_mask,
        )

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        """
        Clear the conditioning from the latent state. This method removes extra tokens from the end of the latent.
        Therefore, conditioning items should add extra tokens ONLY to the end of the latent.
        Generated keyframe slots are *not* conditioning: they are generated output that happens to
        live outside the target grid. When ``generated_keyframe_layout`` is set, their denoised
        content is extracted into ``generated_keyframes`` before the extra tokens are dropped, so
        it survives into the returned state instead of being discarded here.
        """
        generated_keyframes = self.extract_generated_keyframes(latent_state)

        latent_state = latent_state.clone()

        num_tokens = self.patchifier.get_token_count(self.target_shape)
        latent = latent_state.latent[:, :num_tokens]
        clean_latent = latent_state.clean_latent[:, :num_tokens]
        denoise_mask = torch.ones_like(latent_state.denoise_mask.tensor)[:, :num_tokens]
        positions = latent_state.positions[:, :, :num_tokens]

        return LatentState(
            latent=latent,
            denoise_mask=DenoiseMask(denoise_mask, denoise_mask.new_ones((1,))),
            positions=positions,
            clean_latent=clean_latent,
            attention_mask=None,
            keyframes_mask=None,
            generated_keyframe_layout=latent_state.generated_keyframe_layout,
            generated_keyframes=generated_keyframes,
            frozen=latent_state.frozen,
        )

    def extract_generated_keyframes(self, latent_state: LatentState) -> torch.Tensor | None:
        """Extract generated keyframe slot content as a ``(B, C, K, H, W)`` latent.
        Returns ``None`` when the state carries no generated keyframe slots. Implementations decide;
        the protocol has no meaningful default.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class VideoLatentTools(LatentTools):
    """
    Tools for building video latent states.
    """

    patchifier: VideoLatentPatchifier
    target_shape: VideoLatentShape
    fps: float
    scale_factors: SpatioTemporalScaleFactors = DEFAULT_SCALE_FACTORS
    causal_fix: bool = True

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_torch_shape(), (
                f"Latent shape {initial_latent.shape} does not match target shape {self.target_shape.to_torch_shape()}"
            )
        else:
            initial_latent = torch.zeros(
                *self.target_shape.to_torch_shape(),
                device=device,
                dtype=dtype,
            )

        clean_latent = initial_latent.clone()

        denoise_mask = torch.ones(*self.target_shape.mask_shape().to_torch_shape(), dtype=torch.float32)

        latent_coords = self.patchifier.get_patch_grid_bounds(
            output_shape=self.target_shape,
            device=device,
        )

        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.scale_factors,
            causal_fix=self.causal_fix,
        ).float()
        positions[:, 0, ...] = positions[:, 0, ...] / self.fps

        state = self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=DenoiseMask(denoise_mask),
                positions=positions,
                clean_latent=clean_latent,
            )
        )
        return replace(state, keyframes_mask=self._first_frame_keyframes_mask(state))

    def _first_frame_keyframes_mask(self, state: LatentState) -> torch.Tensor:
        """Mark the target's first latent frame, which encodes a single pixel frame.
        Because the video encoder is causal, the first temporal latent frame covers 1 pixel frame
        while every later one covers ``temporal_scale_factor``. That makes it the same token class
        as a generated keyframe slot, and the reference implementation marks it unconditionally --
        independently of whether any keyframe slots exist. This is plain data about the latent, so
        it is always populated; models without the keyframe embedding ignore it.
        """
        mask = torch.zeros_like(state.denoise_mask.tensor, device=state.latent.device)
        mask[:, : self.tokens_per_latent_frame] = 1.0
        return mask

    @property
    def tokens_per_latent_frame(self) -> int:
        """Token count of a single latent frame at the target's spatial resolution."""
        return self.patchifier.get_token_count(self.target_shape._replace(frames=1))

    def extract_generated_keyframes(self, latent_state: LatentState) -> torch.Tensor | None:
        layout = latent_state.generated_keyframe_layout
        if layout is None:
            return None

        available = latent_state.latent.shape[1]
        if layout.first_token + layout.num_tokens > available:
            raise ValueError(
                f"Generated keyframe layout spans tokens [{layout.first_token}, "
                f"{layout.first_token + layout.num_tokens}) but the state has only {available} tokens. "
                "The layout was recorded against a different token sequence."
            )
        if layout.tokens_per_keyframe != self.tokens_per_latent_frame:
            raise ValueError(
                f"Generated keyframe layout has {layout.tokens_per_keyframe} tokens per keyframe, but this "
                f"target shape uses {self.tokens_per_latent_frame}. The layout belongs to a different resolution."
            )

        frame_shape = self.target_shape._replace(frames=1)
        tokens = latent_state.latent[:, layout.token_slice]
        frames = [
            self.patchifier.unpatchify(
                tokens[:, index * layout.tokens_per_keyframe : (index + 1) * layout.tokens_per_keyframe],
                output_shape=frame_shape,
            )
            for index in range(layout.num_keyframes)
        ]
        return torch.cat(frames, dim=2)


@dataclass(frozen=True)
class AudioLatentTools(LatentTools):
    """
    Tools for building audio latent states.
    """

    patchifier: AudioPatchifier
    target_shape: AudioLatentShape

    def extract_generated_keyframes(self, latent_state: LatentState) -> torch.Tensor | None:  # noqa: ARG002
        """Audio states never carry generated keyframe slots."""
        return None

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_torch_shape(), (
                f"Latent shape {initial_latent.shape} does not match target shape {self.target_shape.to_torch_shape()}"
            )
        else:
            initial_latent = torch.zeros(
                *self.target_shape.to_torch_shape(),
                device=device,
                dtype=dtype,
            )

        clean_latent = initial_latent.clone()

        denoise_mask = torch.ones(*self.target_shape.mask_shape().to_torch_shape(), dtype=torch.float32)

        latent_coords = self.patchifier.get_patch_grid_bounds(
            output_shape=self.target_shape,
            device=device,
        )

        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=DenoiseMask(denoise_mask),
                positions=latent_coords,
                clean_latent=clean_latent,
            )
        )
