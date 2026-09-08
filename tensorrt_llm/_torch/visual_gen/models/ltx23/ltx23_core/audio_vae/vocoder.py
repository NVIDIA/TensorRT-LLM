# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import List

import einops
import torch
from torch import nn

from .activations import Activation1d, SnakeBeta
from .amp_block import AMPBlock1
from .bw import MelSTFT, VocoderWithBWE


class Vocoder(nn.Module):
    """LTX-2.3 BigVGAN AMP1 mel-spectrogram generator."""

    def __init__(
        self,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        upsample_rates: List[int] = [6, 5, 2, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 15, 8, 4, 4],
        resblock_dilation_sizes: List[List[int]] = [
            [1, 3, 5],
            [1, 3, 5],
            [1, 3, 5],
        ],
        upsample_initial_channel: int = 1024,
        output_sampling_rate: int = 24000,
        use_tanh_at_final: bool = True,
        apply_final_activation: bool = True,
        use_bias_at_final: bool = True,
    ) -> None:
        super().__init__()
        self.output_sampling_rate = output_sampling_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation

        self.conv_pre = nn.Conv1d(
            in_channels=128,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                kernel_size,
                stride,
                padding=(kernel_size - stride) // 2,
            )
            for i, (stride, kernel_size) in enumerate(
                zip(upsample_rates, upsample_kernel_sizes, strict=True)
            )
        )

        final_channels = upsample_initial_channel // (2 ** len(upsample_rates))
        self.resblocks = nn.ModuleList()
        for i in range(len(upsample_rates)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(AMPBlock1(channels, kernel_size, tuple(dilations)))

        self.act_post = Activation1d(SnakeBeta(final_channels))
        self.conv_post = nn.Conv1d(
            in_channels=final_channels,
            out_channels=2,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=use_bias_at_final,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 3)
        if x.dim() == 4:
            assert x.shape[1] == 2, "Input must have 2 channels for stereo"
            x = einops.rearrange(x, "b s c t -> b (s c) t")

        x = self.conv_pre(x)
        for i, upsample in enumerate(self.ups):
            x = upsample(x)
            start = i * self.num_kernels
            outputs = torch.stack(
                [self.resblocks[index](x) for index in range(start, start + self.num_kernels)]
            )
            x = outputs.mean(dim=0)

        x = self.conv_post(self.act_post(x))
        if self.apply_final_activation:
            x = torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)
        return x


def _vocoder_from_config(
    cfg: dict,
    apply_final_activation: bool = True,
    output_sampling_rate: int | None = None,
) -> Vocoder:
    keys = (
        "resblock_kernel_sizes",
        "upsample_rates",
        "upsample_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_initial_channel",
        "output_sampling_rate",
        "use_tanh_at_final",
        "use_bias_at_final",
    )
    kwargs = {key: cfg.get(key) for key in keys}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if output_sampling_rate is not None:
        kwargs["output_sampling_rate"] = output_sampling_rate
    return Vocoder(
        **kwargs,
        apply_final_activation=apply_final_activation,
    )


class LTX23VocoderConfigurator:
    """Build an LTX-2.3 vocoder from checkpoint configuration."""

    @classmethod
    def from_config(cls, config: dict):
        def check_config_value(cfg: dict, key: str, expected) -> None:
            actual = cfg.get(key)
            if actual != expected:
                raise ValueError(
                    f"LTX-2.3 vocoder config mismatch: expected {key}={expected!r}, got {actual!r}"
                )

        cfg = config.get("vocoder", {})
        if "bwe" not in cfg:
            check_config_value(cfg, "resblock", "1")
            check_config_value(cfg, "stereo", True)
            from ....ltx2.ltx2_core.audio_vae.model_configurator import VocoderConfigurator

            return VocoderConfigurator.from_config(config)

        vocoder_cfg = cfg.get("vocoder", {})
        bwe_cfg = cfg["bwe"]
        for section in (vocoder_cfg, bwe_cfg):
            check_config_value(section, "resblock", "AMP1")
            check_config_value(section, "stereo", True)
            check_config_value(section, "activation", "snakebeta")

        vocoder = _vocoder_from_config(
            vocoder_cfg,
            output_sampling_rate=bwe_cfg["input_sampling_rate"],
        )
        bwe_generator = _vocoder_from_config(
            bwe_cfg,
            apply_final_activation=False,
            output_sampling_rate=bwe_cfg["output_sampling_rate"],
        )
        mel_stft = MelSTFT(
            filter_length=bwe_cfg["n_fft"],
            hop_length=bwe_cfg["hop_length"],
            win_length=bwe_cfg["n_fft"],
            n_mel_channels=bwe_cfg["num_mels"],
        )
        return VocoderWithBWE(
            vocoder=vocoder,
            bwe_generator=bwe_generator,
            mel_stft=mel_stft,
            input_sampling_rate=bwe_cfg["input_sampling_rate"],
            output_sampling_rate=bwe_cfg["output_sampling_rate"],
            hop_length=bwe_cfg["hop_length"],
        )
