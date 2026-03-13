# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import NamedTuple

import torch


class VideoPixelShape(NamedTuple):
    """Shape of video pixel array (B, C, T, H, W)."""

    batch: int
    frames: int
    height: int
    width: int
    fps: float


class SpatioTemporalScaleFactors(NamedTuple):
    """Spatiotemporal downscaling between pixel and VAE latent grid."""

    time: int
    width: int
    height: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, width=32, height=32)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """Shape of video in VAE latent space (B, C, F, H, W)."""

    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    @staticmethod
    def from_pixel_shape(
        shape: "VideoPixelShape",
        latent_channels: int = 128,
        scale_factors: "SpatioTemporalScaleFactors" = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        frames = (shape.frames - 1) // scale_factors[0] + 1
        height = shape.height // scale_factors[1]
        width = shape.width // scale_factors[2]
        return VideoLatentShape(
            batch=shape.batch,
            channels=latent_channels,
            frames=frames,
            height=height,
            width=width,
        )

    def upscale(
        self,
        scale_factors: "SpatioTemporalScaleFactors" = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )


class AudioLatentShape(NamedTuple):
    """Shape of audio in VAE latent space (B, C, frames, mel_bins)."""

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = (
            float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)
        )
        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: "VideoPixelShape",
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )
