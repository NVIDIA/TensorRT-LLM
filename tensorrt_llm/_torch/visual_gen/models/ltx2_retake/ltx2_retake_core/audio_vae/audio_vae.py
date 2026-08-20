# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from types import ModuleType

import torch

from tensorrt_llm.inputs.multimodal_data import AudioData

from ....ltx2.ltx2_core.audio_vae.audio_vae import (
    LATENT_DOWNSAMPLE_FACTOR,
    build_mid_block,
    run_mid_block,
)
from ....ltx2.ltx2_core.audio_vae.causal_conv_2d import make_conv2d
from ....ltx2.ltx2_core.audio_vae.causality_axis import CausalityAxis
from ....ltx2.ltx2_core.audio_vae.ops import PerChannelStatistics
from ....ltx2.ltx2_core.normalization import NormType, build_normalization_layer
from ....ltx2.ltx2_core.patchifier import AudioPatchifier
from ....ltx2.ltx2_core.types import AudioLatentShape
from .downsample import build_downsampling_path


def _require_torchaudio() -> ModuleType:
    """Import the optional audio encoder dependency on demand."""
    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The LTX-2 native audio encoder needs `torchaudio` for waveform "
            "resampling and mel preprocessing. Install a torchaudio build matching "
            "the installed torch in the VisualGen runtime environment."
        ) from exc
    return torchaudio


class AudioEncoder(torch.nn.Module):
    """Compresses a log-mel spectrogram into normalized audio latents.

    Structural mirror of :class:`AudioDecoder`: a downsampling path of residual
    blocks, the shared mid block, then a final normalization + convolution.
    """

    def __init__(
        self,
        *,
        ch: int,
        ch_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: set[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        mid_block_add_attention: bool = True,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        is_causal: bool = True,
        mel_bins: int = 64,
    ) -> None:
        super().__init__()

        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft
        self.mel_bins = mel_bins
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.double_z = double_z

        self.conv_in = make_conv2d(
            in_channels,
            ch,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        self.non_linearity = torch.nn.SiLU()
        self.down, block_in = build_downsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            dropout=dropout,
            norm_type=norm_type,
            causality_axis=causality_axis,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
        )
        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=0,
            dropout=dropout,
            norm_type=norm_type,
            causality_axis=causality_axis,
            add_attention=mid_block_add_attention,
        )
        self.norm_out = build_normalization_layer(block_in, normtype=norm_type)
        self.conv_out = make_conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Encode a ``(B, C, time, mel_bins)`` spectrogram into ``(B, C, frames, mel_bins)``."""
        h = self.conv_in(spectrogram)
        h = self._run_downsampling_path(h)
        h = run_mid_block(self.mid, h)
        h = self.conv_out(self.non_linearity(self.norm_out(h)))
        return self._normalize_latents(h)

    def _run_downsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                h = stage.block[block_idx](h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)
            if level != self.num_resolutions - 1:
                h = stage.downsample(h)
        return h

    def _normalize_latents(self, latent_output: torch.Tensor) -> torch.Tensor:
        """Normalize the mean half of the encoder output with the per-channel statistics.

        With ``double_z=True`` the final convolution emits twice the latent
        channels -- mean and log-variance concatenated. Only the first (mean)
        half is normalized and returned; the variance half is deliberately
        dropped, matching the deterministic (mean-only) encode the pipeline uses.
        """
        means = torch.chunk(latent_output, 2, dim=1)[0] if self.double_z else latent_output
        latent_shape = AudioLatentShape(
            batch=means.shape[0],
            channels=means.shape[1],
            frames=means.shape[2],
            mel_bins=means.shape[3],
        )
        latent_patched = self.patchifier.patchify(means)
        latent_normalized = self.per_channel_statistics.normalize(latent_patched)
        return self.patchifier.unpatchify(latent_normalized, latent_shape)


def _waveform_to_mel(
    samples: torch.Tensor,
    source_sample_rate: int,
    audio_encoder: AudioEncoder,
) -> torch.Tensor:
    torchaudio = _require_torchaudio()
    if source_sample_rate != audio_encoder.sample_rate:
        resampled = torchaudio.functional.resample(
            samples,
            source_sample_rate,
            audio_encoder.sample_rate,
        )
        samples = resampled.to(device=samples.device, dtype=samples.dtype)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_encoder.sample_rate,
        n_fft=audio_encoder.n_fft,
        win_length=audio_encoder.n_fft,
        hop_length=audio_encoder.mel_hop_length,
        f_min=0.0,
        f_max=audio_encoder.sample_rate / 2.0,
        n_mels=audio_encoder.mel_bins,
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect",
        power=1.0,
        mel_scale="slaney",
        norm="slaney",
    ).to(device=samples.device)
    mel = torch.log(torch.clamp(mel_transform(samples), min=1e-5))
    return mel.to(dtype=samples.dtype).permute(0, 1, 3, 2).contiguous()


def encode_audio(
    audio: AudioData,
    audio_encoder: AudioEncoder,
) -> torch.Tensor:
    """Encode an audio waveform into audio VAE latents."""
    dtype = next(audio_encoder.parameters()).dtype
    device = next(audio_encoder.parameters()).device

    samples = torch.as_tensor(audio.samples, device=device)
    if samples.ndim != 3:
        raise ValueError(f"Expected audio samples with shape (B, C, T); got {samples.shape}.")
    input_channels = audio_encoder.conv_in.conv.in_channels
    if samples.shape[1] == 1 and input_channels == 2:
        samples = samples.expand(-1, 2, -1).contiguous()
    elif samples.shape[1] != input_channels:
        raise ValueError(
            f"Audio encoder expects {input_channels} channel(s), got {samples.shape[1]}."
        )
    mel_spectrogram = _waveform_to_mel(samples, audio.sample_rate, audio_encoder)
    return audio_encoder(mel_spectrogram.to(dtype=dtype))
