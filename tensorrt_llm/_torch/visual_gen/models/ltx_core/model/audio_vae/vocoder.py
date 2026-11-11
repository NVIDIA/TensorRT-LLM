import contextlib
import math
from collections.abc import Iterator
from typing import List

import einops
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.audio_vae.resnet import LRELU_SLOPE, ResBlock1
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.disposable import Disposable


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


@contextlib.contextmanager
def _module_in_fp32(module: nn.Module, *, enabled: bool) -> Iterator[None]:
    """Temporarily cast *module* to float32, restoring its original dtype on exit.
    Used for the MPS vocoder path where fp32 autocast is unavailable, so the
    weights must be materialized in float32 for the forward pass. Restores to the
    module's original weight dtype (captured here), not the input dtype. When
    *enabled* is False this is a no-op, so callers can wrap unconditionally.
    """
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
# Anti-aliased resampling helpers (kaiser-sinc filters) for BigVGAN v2
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
    time = torch.arange(-half_size, half_size) + 0.5 if even else torch.arange(kernel_size) - half_size
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
        return F.conv1d(x, self.filter.expand(n_channels, -1, -1), stride=self.stride, groups=n_channels)


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
            # Hann-windowed sinc filter equivalent to torchaudio.functional.resample
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
            # Kaiser-windowed sinc filter (BigVGAN default).
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
        self.alpha = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
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
        self.alpha = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta = nn.Parameter(torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha)
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
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)),
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)),
                nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)),
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


class Vocoder(torch.nn.Module, Disposable):
    """
    Vocoder model for synthesizing audio from Mel spectrograms.
    Args:
        resblock_kernel_sizes: List of kernel sizes for the residual blocks.
                               This value is read from the checkpoint at `config.vocoder.resblock_kernel_sizes`.
        upsample_rates: List of upsampling rates.
                               This value is read from the checkpoint at `config.vocoder.upsample_rates`.
        upsample_kernel_sizes: List of kernel sizes for the upsampling layers.
                               This value is read from the checkpoint at `config.vocoder.upsample_kernel_sizes`.
        resblock_dilation_sizes: List of dilation sizes for the residual blocks.
                               This value is read from the checkpoint at `config.vocoder.resblock_dilation_sizes`.
        upsample_initial_channel: Initial number of channels for the upsampling layers.
                               This value is read from the checkpoint at `config.vocoder.upsample_initial_channel`.
        resblock: Type of residual block to use ("1", "2", or "AMP1").
                                This value is read from the checkpoint at `config.vocoder.resblock`.
        output_sampling_rate: Waveform sample rate.
                               This value is read from the checkpoint at `config.vocoder.output_sampling_rate`.
        activation: Activation type for BigVGAN v2 ("snake" or "snakebeta"). Only used when resblock="AMP1".
        use_tanh_at_final: Apply tanh at the output (when apply_final_activation=True).
        apply_final_activation: Whether to apply the final tanh/clamp activation.
        use_bias_at_final: Whether to use bias in the final conv layer.
    """

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

        # Mutable default values are not supported as default arguments.
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

        # All production checkpoints are stereo: 128 input channels (2 stereo channels x 64 mel
        # bins each), 2 output channels.
        self.conv_pre = nn.Conv1d(
            in_channels=128,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        resblock_cls = ResBlock1 if resblock == "1" else AMPBlock1

        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                kernel_size,
                stride,
                padding=(kernel_size - stride) // 2,
            )
            for i, (stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True))
        )

        final_channels = upsample_initial_channel // (2 ** len(upsample_rates))
        self.resblocks = nn.ModuleList()

        for i in range(len(upsample_rates)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                if self.is_amp:
                    self.resblocks.append(resblock_cls(ch, kernel_size, dilations, activation=activation))
                else:
                    self.resblocks.append(resblock_cls(ch, kernel_size, dilations))

        if self.is_amp:
            self.act_post: nn.Module = Activation1d(SnakeBeta(final_channels))
        else:
            self.act_post = nn.LeakyReLU()

        # All production checkpoints are stereo: this final conv maps `final_channels` to 2 output channels (stereo).
        self.conv_post = nn.Conv1d(
            in_channels=final_channels,
            out_channels=2,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=use_bias_at_final,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vocoder.
        Args:
            x: Input Mel spectrogram tensor. Can be either:
               - 3D: (batch_size, time, mel_bins) for mono
               - 4D: (batch_size, 2, time, mel_bins) for stereo
        Returns:
            Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """
        x = x.transpose(2, 3)  # (batch, channels, time, mel_bins) -> (batch, channels, mel_bins, time)

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

            # Evaluate all resblocks with the same input tensor so they can run
            # independently (and thus in parallel on accelerator hardware) before
            # aggregating their outputs via mean.
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
    """Implements STFT as a convolution with precomputed DFT x Hann-window bases.
    The DFT basis rows (real and imaginary parts interleaved) multiplied by the causal
    Hann window are stored as buffers and loaded from the checkpoint. Using the exact
    bfloat16 bases from training ensures the mel values fed to the BWE generator are
    bit-identical to what it was trained on.
    """

    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length))

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute magnitude and phase spectrogram from a batch of waveforms.
        Applies causal (left-only) padding of win_length - hop_length samples so that
        each output frame depends only on past and present input — no lookahead.
        Args:
            y: Waveform tensor of shape (B, T).
        Returns:
            magnitude: Linear amplitude spectrogram, shape (B, n_freqs, T_frames).
            phase:     Phase spectrogram in radians, shape (B, n_freqs, T_frames).
        """
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
    """Causal log-mel spectrogram module whose buffers are loaded from the checkpoint.
    Computes a log-mel spectrogram by running the causal STFT (_STFTFn) on the input
    waveform and projecting the linear magnitude spectrum onto the mel filterbank.
    The module's state dict layout matches the 'mel_stft.*' keys stored in the checkpoint
    (mel_basis, stft_fn.forward_basis, stft_fn.inverse_basis).
    """

    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
    ) -> None:
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)

        # Initialized to zeros; load_state_dict overwrites with the checkpoint's
        # exact bfloat16 filterbank (vocoder.mel_stft.mel_basis, shape [n_mels, n_freqs]).
        n_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(n_mel_channels, n_freqs))

    def mel_spectrogram(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-mel spectrogram and auxiliary spectral quantities.
        Args:
            y: Waveform tensor of shape (B, T).
        Returns:
            log_mel:   Log-compressed mel spectrogram, shape (B, n_mel_channels, T_frames).
            magnitude: Linear amplitude spectrogram, shape (B, n_freqs, T_frames).
            phase:     Phase spectrogram in radians, shape (B, n_freqs, T_frames).
            energy:    Per-frame energy (L2 norm over frequency), shape (B, T_frames).
        """
        magnitude, phase = self.stft_fn(y)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class VocoderWithBWE(nn.Module, Disposable):
    """Vocoder with bandwidth extension (BWE) upsampling.
    Chains a mel-to-wav vocoder with a BWE module that upsamples the output
    to a higher sample rate. The BWE computes a mel spectrogram from the
    vocoder output, runs it through a second generator to predict a residual,
    and adds it to a sinc-resampled skip connection.
    The forward pass runs in fp32 via autocast to avoid bfloat16 accumulation
    errors that degrade spectral metrics by 40-90%.
    """

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
        # Compute the resampler on CPU so the sinc filter is materialized even when
        # the model is constructed on the meta device by the unit builder.
        # The filter is not stored in the checkpoint (persistent=False).
        with torch.device("cpu"):
            self.resampler = UpSample1d(
                ratio=output_sampling_rate // input_sampling_rate, persistent=False, window_type="hann"
            )

    @property
    def conv_pre(self) -> nn.Conv1d:
        return self.vocoder.conv_pre

    @property
    def conv_post(self) -> nn.Conv1d:
        return self.vocoder.conv_post

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram from waveform using causal STFT bases.
        Args:
            audio: Waveform tensor of shape (B, C, T).
        Returns:
            mel: Log-mel spectrogram of shape (B, C, n_mels, T_frames).
        """
        batch, n_channels, _ = audio.shape
        flat = audio.reshape(batch * n_channels, -1)  # (B*C, T)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)  # (B*C, n_mels, T_frames)
        return mel.reshape(batch, n_channels, mel.shape[1], mel.shape[2])  # (B, C, n_mels, T_frames)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Run the full vocoder + BWE forward pass.
        Runs in float32 regardless of weight or input dtype. bfloat16 arithmetic
        causes 40-90% spectral metric degradation due to accumulation errors
        compounding through 108 sequential convolutions in the BigVGAN v2 architecture.
        Args:
            mel_spec: Mel spectrogram of shape (B, 2, T, mel_bins) for stereo
                      or (B, T, mel_bins) for mono. Same format as Vocoder.forward.
        Returns:
            Waveform tensor of shape (B, out_channels, T_out) clipped to [-1, 1].
        """
        input_dtype = mel_spec.dtype
        # Run the entire forward pass in fp32.  bfloat16 accumulation errors
        # compound through 108 sequential convolutions and degrade spectral
        # metrics (mel_l1, MRSTFT) by 40-90% while perceptual quality (CDPAM)
        # is unaffected.  fp32 eliminates this degradation.
        # On CUDA/CPU we use autocast(dtype=float32) rather than self.float()
        # because it upcasts bf16 weights per-op at kernel level, avoiding the
        # temporary memory spike of self.float() / self.to(original_dtype).
        # Benchmarked on H100 (128.5M-param model):
        #   autocast fp32: +70 MB peak VRAM, 123 ms  (vs 482 MB / 95 ms for bf16)
        #   model.float(): +324 MB peak VRAM, 149 ms
        # Tested: both approaches produce bit-identical output.
        # MPS autocast does not upcast conv weights to fp32 (it only supports
        # lower-precision autocast dtypes), which would leave the float32 input
        # running against bf16 conv weights and raise a dtype mismatch. There we
        # fall back to materializing the weights in fp32 for the pass (bit-identical
        # per the note above; the memory spike is negligible for this small model).
        # The vocoder is normally built in fp32 on MPS, so this fallback is then a
        # no-op -- it only triggers if a bf16 module is run on MPS directly.
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
            output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate

            # Pad to multiple of hop_length for exact mel frame count
            remainder = length_low_rate % self.hop_length
            if remainder != 0:
                x = F.pad(x, (0, self.hop_length - remainder))

            # Compute mel spectrogram from vocoder output: (B, C, n_mels, T_frames)
            mel = self._compute_mel(x)

            # Vocoder.forward expects (B, C, T, mel_bins) — transpose before calling bwe_generator
            mel_for_bwe = mel.transpose(2, 3)  # (B, C, T_frames, mel_bins)
            residual = self.bwe_generator(mel_for_bwe)
            skip = self.resampler(x)
            assert residual.shape == skip.shape, f"residual {residual.shape} != skip {skip.shape}"

            return torch.clamp(residual + skip, -1, 1)[..., :output_length].to(input_dtype)
