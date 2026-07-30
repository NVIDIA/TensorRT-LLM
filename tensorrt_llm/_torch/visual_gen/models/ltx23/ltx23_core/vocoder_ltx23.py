# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 vocoder: BigVGAN-v2 (AMP1) + bandwidth-extension (BWE) -> 48 kHz.

LTX-2 ships a HiFi-GAN vocoder (``ResBlock1`` + LeakyReLU, 24 kHz). LTX-2.3
replaces it with a BigVGAN-v2 generator using SnakeBeta activations and
anti-aliased ``Activation1d`` resampling, wrapped in a ``VocoderWithBWE`` that
re-analyzes the 16 kHz output with a causal mel-STFT, predicts a residual with a
second BigVGAN generator, and adds a Hann-sinc-resampled skip to reach 48 kHz.

This is a faithful port of the upstream Lightricks ``ltx_core`` vocoder
(``packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py``) restricted to
the AMP1 path LTX-2.3 uses. Every module name matches the checkpoint's
``vocoder.vocoder.*`` / ``vocoder.bwe_generator.*`` / ``vocoder.mel_stft.*``
keys so weights load without remapping.

Numerical contract (do not "optimize" away):
* SnakeBeta is log-scale: ``alpha=exp(alpha)``, ``beta=exp(beta)``.
* The whole ``VocoderWithBWE`` forward runs in fp32 (bf16 accumulation
  degrades spectral metrics 40-90% across the ~108 sequential convs).
* The BWE generator has ``apply_final_activation=False`` (residual, no clamp).
"""

import contextlib
import math
from collections.abc import Iterator
from typing import List

import einops
import torch
import torch.nn.functional as F
from torch import nn

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _check_config_value(cfg: dict, key: str, expected) -> None:
    actual = cfg.get(key)
    if actual != expected:
        raise ValueError(
            f"LTX-2.3 vocoder config mismatch: expected {key}={expected!r}, got {actual!r}"
        )


@contextlib.contextmanager
def _module_in_fp32(module: nn.Module, *, enabled: bool) -> Iterator[None]:
    """Temporarily cast *module* to float32, restoring its dtype on exit (MPS path)."""
    if not enabled:
        yield
        return
    module_dtype = next(module.parameters()).dtype
    module.float()
    try:
        yield
    finally:
        module.to(module_dtype)


# ---------------------------------------------------------------------------
# Anti-aliased resampling helpers (Kaiser-sinc filters) for BigVGAN v2.
# Adopted from https://github.com/NVIDIA/BigVGAN
# ---------------------------------------------------------------------------


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (
        torch.arange(-half_size, half_size) + 0.5
        if even
        else torch.arange(kernel_size) - half_size
    )
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n_channels, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x, self.filter.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels
        )


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        persistent: bool = True,
        window_type: str = "kaiser",
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            # Hann-windowed sinc (equivalent to torchaudio.functional.resample);
            # used for the BWE skip connection. Filter is not stored in checkpoint.
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            time_axis = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            sinc_filter = (torch.sinc(time_axis) * window * rolloff / ratio).view(1, 1, -1)
        else:
            # Kaiser-windowed sinc (BigVGAN default).
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            sinc_filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )

        self.register_buffer("filter", sinc_filter, persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n_channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        filt = self.filter.to(dtype=x.dtype, device=x.device).expand(n_channels, -1, -1)
        x = self.ratio * F.conv_transpose1d(x, filt, stride=self.stride, groups=n_channels)
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


class Snake(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(
            torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha
        )
        self.alpha.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.eps)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(
            torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha
        )
        self.alpha.requires_grad = alpha_trainable
        self.beta = nn.Parameter(
            torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha
        )
        self.beta.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.eps)) * torch.sin(x * alpha).pow(2)


class AMPBlock1(nn.Module):
    """BigVGAN anti-aliased multi-periodicity residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        activation: str = "snake",
    ) -> None:
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[i],
                    padding=get_padding(kernel_size, dilation[i]),
                )
                for i in range(3)
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)
                )
                for _ in range(3)
            ]
        )
        self.acts1 = nn.ModuleList([Activation1d(act_cls(channels)) for _ in range(len(self.convs1))])
        self.acts2 = nn.ModuleList([Activation1d(act_cls(channels)) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2, strict=True):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
        return x


def _make_resblock(
    resblock: str,
    ch: int,
    kernel_size: int,
    dilations: List[int],
    activation: str,
) -> nn.Module:
    if resblock == "AMP1":
        return AMPBlock1(ch, kernel_size, dilations, activation=activation)
    # HiFi-GAN fallback (unused by LTX-2.3, kept for parity); lazily import to
    # avoid a hard dependency on the ltx2 audio_vae package.
    from ...ltx2.ltx2_core.audio_vae.resnet import ResBlock1

    return ResBlock1(ch, kernel_size, dilations)


class Vocoder(nn.Module):
    """Mel-spectrogram -> waveform generator (HiFi-GAN "1" or BigVGAN "AMP1")."""

    def __init__(  # noqa: PLR0913
        self,
        resblock_kernel_sizes: List[int] | None = None,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_dilation_sizes: List[List[int]] | None = None,
        upsample_initial_channel: int = 1024,
        resblock: str = "1",
        output_sampling_rate: int = 24000,
        activation: str = "snake",
        use_tanh_at_final: bool = True,
        apply_final_activation: bool = True,
        use_bias_at_final: bool = True,
    ) -> None:
        super().__init__()

        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sampling_rate = output_sampling_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation
        self.is_amp = resblock == "AMP1"

        # Stereo checkpoints: 128 input channels (2 stereo x 64 mel), 2 out.
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
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(
                    _make_resblock(resblock, ch, kernel_size, dilations, activation)
                )

        if self.is_amp:
            self.act_post: nn.Module = Activation1d(SnakeBeta(final_channels))
        else:
            self.act_post = nn.LeakyReLU()

        self.conv_post = nn.Conv1d(
            in_channels=final_channels,
            out_channels=2,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=use_bias_at_final,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, time, mel_bins) -> (B, C, mel_bins, time)
        x = x.transpose(2, 3)
        if x.dim() == 4:  # stereo
            assert x.shape[1] == 2, "Input must have 2 channels for stereo"
            x = einops.rearrange(x, "b s c t -> b (s c) t")

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            if not self.is_amp:
                x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels
            block_outputs = torch.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                dim=0,
            )
            x = block_outputs.mean(dim=0)

        x = self.act_post(x)
        x = self.conv_post(x)

        if self.apply_final_activation:
            x = torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)
        return x


class _STFTFn(nn.Module):
    """STFT as a convolution with precomputed DFT x Hann-window bases (from checkpoint)."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length))

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.dim() == 2:
            y = y.unsqueeze(1)  # (B, 1, T)
        left_pad = max(0, self.win_length - self.hop_length)  # causal: left-only
        y = F.pad(y, (left_pad, 0))
        spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        magnitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag.float(), real.float()).to(real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    """Causal log-mel spectrogram; buffers (mel_basis, stft_fn.*) loaded from checkpoint."""

    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
    ) -> None:
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(n_mel_channels, n_freqs))

    def mel_spectrogram(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        magnitude, phase = self.stft_fn(y)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class VocoderWithBWE(nn.Module):
    """BigVGAN vocoder + residual bandwidth extension to a higher sample rate."""

    def __init__(
        self,
        vocoder: Vocoder,
        bwe_generator: Vocoder,
        mel_stft: MelSTFT,
        input_sampling_rate: int,
        output_sampling_rate: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        # Sinc skip filter is not stored in the checkpoint (persistent=False);
        # materialize on CPU so it exists even under meta-device construction.
        with torch.device("cpu"):
            self.resampler = UpSample1d(
                ratio=output_sampling_rate // input_sampling_rate,
                persistent=False,
                window_type="hann",
            )

    @property
    def conv_pre(self) -> nn.Conv1d:
        return self.vocoder.conv_pre

    @property
    def conv_post(self) -> nn.Conv1d:
        return self.vocoder.conv_post

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        batch, n_channels, _ = audio.shape
        flat = audio.reshape(batch * n_channels, -1)  # (B*C, T)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)  # (B*C, n_mels, T_frames)
        return mel.reshape(batch, n_channels, mel.shape[1], mel.shape[2])

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        input_dtype = mel_spec.dtype
        device_type = mel_spec.device.type
        module_dtype = next(self.parameters()).dtype
        fp32_ctx = (
            _module_in_fp32(self, enabled=module_dtype != torch.float32)
            if device_type == "mps"
            else torch.autocast(device_type=device_type, dtype=torch.float32)
        )

        with fp32_ctx:
            x = self.vocoder(mel_spec.float())
            _, _, length_low_rate = x.shape
            output_length = (
                length_low_rate * self.output_sampling_rate // self.input_sampling_rate
            )

            remainder = length_low_rate % self.hop_length
            if remainder != 0:
                x = F.pad(x, (0, self.hop_length - remainder))

            mel = self._compute_mel(x)  # (B, C, n_mels, T_frames)
            mel_for_bwe = mel.transpose(2, 3)  # (B, C, T_frames, mel_bins)
            residual = self.bwe_generator(mel_for_bwe)
            skip = self.resampler(x)
            assert residual.shape == skip.shape, f"residual {residual.shape} != skip {skip.shape}"

            return torch.clamp(residual + skip, -1, 1)[..., :output_length].to(input_dtype)


def _vocoder_from_config(
    cfg: dict,
    apply_final_activation: bool = True,
    output_sampling_rate: int | None = None,
) -> Vocoder:
    return Vocoder(
        resblock_kernel_sizes=cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        upsample_rates=cfg.get("upsample_rates", [6, 5, 2, 2, 2]),
        upsample_kernel_sizes=cfg.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
        resblock_dilation_sizes=cfg.get(
            "resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        ),
        upsample_initial_channel=cfg.get("upsample_initial_channel", 1024),
        resblock=cfg.get("resblock", "1"),
        output_sampling_rate=(
            output_sampling_rate
            if output_sampling_rate is not None
            else cfg.get("output_sampling_rate", 24000)
        ),
        activation=cfg.get("activation", "snake"),
        use_tanh_at_final=cfg.get("use_tanh_at_final", True),
        apply_final_activation=apply_final_activation,
        use_bias_at_final=cfg.get("use_bias_at_final", True),
    )


class LTX23VocoderConfigurator:
    """Build the LTX-2.3 ``VocoderWithBWE`` from the native checkpoint config.

    Auto-detects: a nested ``vocoder`` + ``bwe`` config => VocoderWithBWE (2.3);
    a flat config => plain HiFi-GAN Vocoder (pre-2.3, kept for parity).
    """

    @classmethod
    def from_config(cls, config: dict):
        cfg = config.get("vocoder", {})

        if "bwe" not in cfg:
            _check_config_value(cfg, "resblock", "1")
            _check_config_value(cfg, "stereo", True)
            return _vocoder_from_config(cfg)

        vocoder_cfg = cfg.get("vocoder", {})
        bwe_cfg = cfg["bwe"]

        _check_config_value(vocoder_cfg, "resblock", "AMP1")
        _check_config_value(vocoder_cfg, "stereo", True)
        _check_config_value(vocoder_cfg, "activation", "snakebeta")
        _check_config_value(bwe_cfg, "resblock", "AMP1")
        _check_config_value(bwe_cfg, "stereo", True)
        _check_config_value(bwe_cfg, "activation", "snakebeta")

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
