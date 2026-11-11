import math
from typing import Optional, Tuple

import einops
import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.components.protocols import Patchifier
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape


class VideoLatentPatchifier(Patchifier):
    def __init__(self, patch_size: int):
        # Patch sizes for video latents.
        self._patch_size = (
            1,  # temporal dimension
            patch_size,  # height dimension
            patch_size,  # width dimension
        )

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: VideoLatentShape) -> int:
        return math.prod(tgt_shape.to_torch_shape()[2:]) // math.prod(self._patch_size)

    def patchify(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        latents = einops.rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )

        return latents

    def unpatchify(
        self,
        latents: torch.Tensor,
        output_shape: VideoLatentShape,
    ) -> torch.Tensor:
        assert self._patch_size[0] == 1, "Temporal patch size must be 1 for symmetric patchifier"

        patch_grid_frames = output_shape.frames // self._patch_size[0]
        patch_grid_height = output_shape.height // self._patch_size[1]
        patch_grid_width = output_shape.width // self._patch_size[2]

        latents = einops.rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q)",
            f=patch_grid_frames,
            h=patch_grid_height,
            w=patch_grid_width,
            p=self._patch_size[1],
            q=self._patch_size[2],
        )

        return latents

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the per-dimension bounds [inclusive start, exclusive end) for every
        patch produced by `patchify`. The bounds are expressed in the original
        video grid coordinates: frame/time, height, and width.
        The resulting tensor is shaped `[batch_size, 3, num_patches, 2]`, where:
            - axis 1 (size 3) enumerates (frame/time, height, width) dimensions
            - axis 3 (size 2) stores `[start, end)` indices within each dimension
        Args:
            output_shape: Video grid description containing frames, height, and width.
            device: Device of the latent tensor.
        """
        if not isinstance(output_shape, VideoLatentShape):
            raise ValueError("VideoLatentPatchifier expects VideoLatentShape when computing coordinates")

        frames = output_shape.frames
        height = output_shape.height
        width = output_shape.width
        batch_size = output_shape.batch

        # Validate inputs to ensure positive dimensions
        assert frames > 0, f"frames must be positive, got {frames}"
        assert height > 0, f"height must be positive, got {height}"
        assert width > 0, f"width must be positive, got {width}"
        assert batch_size > 0, f"batch_size must be positive, got {batch_size}"

        # Generate grid coordinates for each dimension (frame, height, width)
        # We use torch.arange to create the starting coordinates for each patch.
        # indexing='ij' ensures the dimensions are in the order (frame, height, width).
        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=frames, step=self._patch_size[0], device=device),
            torch.arange(start=0, end=height, step=self._patch_size[1], device=device),
            torch.arange(start=0, end=width, step=self._patch_size[2], device=device),
            indexing="ij",
        )

        # Stack the grid coordinates to create the start coordinates tensor.
        # Shape becomes (3, grid_f, grid_h, grid_w)
        patch_starts = torch.stack(grid_coords, dim=0)

        # Calculate end coordinates: start + patch_size, per axis with Python-int patch sizes so
        # no host-built tensor is copied to the device.
        # Shape becomes (3, grid_f, grid_h, grid_w)
        patch_ends = torch.stack(
            [start + size for start, size in zip(grid_coords, self._patch_size, strict=True)], dim=0
        )

        # Stack start and end coordinates together along the last dimension
        # Shape becomes (3, grid_f, grid_h, grid_w, 2), where the last dimension is [start, end]
        latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)

        # Broadcast to batch size and flatten all spatial/temporal dimensions into one sequence.
        # Final Shape: (batch_size, 3, num_patches, 2)
        latent_coords = einops.repeat(
            latent_coords,
            "c f h w bounds -> b c (f h w) bounds",
            b=batch_size,
            bounds=2,
        )

        return latent_coords


def get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool = False,
) -> torch.Tensor:
    """
    Map latent-space `[start, end)` coordinates to their pixel-space equivalents by scaling
    each axis (frame/time, height, width) with the corresponding VAE downsampling factors.
    Optionally compensate for causal encoding that keeps the first frame at unit temporal scale.
    Args:
        latent_coords: Tensor of latent bounds shaped `(batch, 3, num_patches, 2)`.
        scale_factors: SpatioTemporalScaleFactors tuple `(temporal, height, width)` with integer scale factors applied
        per axis.
        causal_fix: When True, rewrites the temporal axis of the first frame so causal VAEs
            that treat frame zero differently still yield non-negative timestamps.
    """
    # Axis 1 of `latent_coords` is ordered (frame/time, height, width) -- match that explicitly by
    # pulling fields from the NamedTuple rather than relying on tuple iteration order. The factors
    # stay Python ints: a host-built factor tensor would be a blocking copy to the device.
    pixel_coords = torch.stack(
        [
            latent_coords[:, 0] * scale_factors.time,
            latent_coords[:, 1] * scale_factors.height,
            latent_coords[:, 2] * scale_factors.width,
        ],
        dim=1,
    )

    if causal_fix:
        # VAE temporal stride for the very first frame is 1 instead of `scale_factors.time`.
        # Shift and clamp to keep the first-frame timestamps causal and non-negative.
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + 1 - scale_factors.time).clamp(min=0)

    return pixel_coords


class AudioPatchifier(Patchifier):
    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        """
        Patchifier tailored for spectrogram/audio latents.
        Args:
            patch_size: Number of mel bins combined into a single patch. This
                controls the resolution along the frequency axis.
            sample_rate: Original waveform sampling rate. Used to map latent
                indices back to seconds so downstream consumers can align audio
                and video cues.
            hop_length: Window hop length used for the spectrogram. Determines
                how many real-time samples separate two consecutive latent frames.
            audio_latent_downsample_factor: Ratio between spectrogram frames and
                latent frames; compensates for additional downsampling inside the
                VAE encoder.
            is_causal: When True, timing is shifted to account for causal
                receptive fields so timestamps do not peek into the future.
            shift: Integer offset applied to the latent indices. Enables
                constructing overlapping windows from the same latent sequence.
        """
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        return tgt_shape.frames

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Converts latent indices into real-time seconds while honoring causal
        offsets and the configured hop length.
        Args:
            start_latent: Inclusive start index inside the latent sequence. This
                sets the first timestamp returned.
            end_latent: Exclusive end index. Determines how many timestamps get
                generated.
            dtype: Floating-point dtype used for the returned tensor, allowing
                callers to control precision.
            device: Target device for the timestamp tensor. When omitted the
                computation occurs on CPU to avoid surprising GPU allocations.
        """
        if device is None:
            device = torch.device("cpu")

        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)

        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor

        if self.is_causal:
            # Frame offset for causal alignment.
            # The "+1" ensures the timestamp corresponds to the first sample that is fully available.
            causal_offset = 1
            audio_mel_frame = (audio_mel_frame + causal_offset - self.audio_latent_downsample_factor).clip(min=0)

        return audio_mel_frame * self.hop_length / self.sample_rate

    def _compute_audio_timings(
        self,
        batch_size: int,
        num_steps: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Builds a `(B, 1, T, 2)` tensor containing timestamps for each latent frame.
        This helper method underpins `get_patch_grid_bounds` for the audio patchifier.
        Args:
            batch_size: Number of sequences to broadcast the timings over.
            num_steps: Number of latent frames (time steps) to convert into timestamps.
            device: Device on which the resulting tensor should reside.
        """
        resolved_device = device
        if resolved_device is None:
            resolved_device = torch.device("cpu")

        start_timings = self._get_audio_latent_time_in_sec(
            self.shift,
            num_steps + self.shift,
            torch.float32,
            resolved_device,
        )
        start_timings = start_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        end_timings = self._get_audio_latent_time_in_sec(
            self.shift + 1,
            num_steps + self.shift + 1,
            torch.float32,
            resolved_device,
        )
        end_timings = end_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        return torch.stack([start_timings, end_timings], dim=-1)

    def patchify(
        self,
        audio_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flattens the audio latent tensor along time. Use `get_patch_grid_bounds`
        to derive timestamps for each latent frame based on the configured hop
        length and downsampling.
        Args:
            audio_latents: Latent tensor to patchify.
        Returns:
            Flattened patch tokens tensor. Use `get_patch_grid_bounds` to compute the
            corresponding timing metadata when needed.
        """
        audio_latents = einops.rearrange(
            audio_latents,
            "b c t f -> b t (c f)",
        )

        return audio_latents

    def unpatchify(
        self,
        audio_latents: torch.Tensor,
        output_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """
        Restores the `(B, C, T, F)` spectrogram tensor from flattened patches.
        Use `get_patch_grid_bounds` to recompute the timestamps that describe each
        frame's position in real time.
        Args:
            audio_latents: Latent tensor to unpatchify.
            output_shape: Shape of the unpatched output tensor.
        Returns:
            Unpatched latent tensor. Use `get_patch_grid_bounds` to compute the timing
            metadata associated with the restored latents.
        """
        # audio_latents shape: (batch, time, freq * channels)
        audio_latents = einops.rearrange(
            audio_latents,
            "b t (c f) -> b c t f",
            c=output_shape.channels,
            f=output_shape.mel_bins,
        )

        return audio_latents

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the temporal bounds `[inclusive start, exclusive end)` for every
        patch emitted by `patchify`. For audio this corresponds to timestamps in
        seconds aligned with the original spectrogram grid.
        The returned tensor has shape `[batch_size, 1, time_steps, 2]`, where:
            - axis 1 (size 1) represents the temporal dimension
            - axis 3 (size 2) stores the `[start, end)` timestamps per patch
        Args:
            output_shape: Audio grid specification describing the number of time steps.
            device: Target device for the returned tensor.
        """
        if not isinstance(output_shape, AudioLatentShape):
            raise ValueError("AudioPatchifier expects AudioLatentShape when computing coordinates")

        return self._compute_audio_timings(output_shape.batch, output_shape.frames, device)
