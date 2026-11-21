# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""L2 BWE wrapper parity against an FP32 FFT and windowed-sinc oracle."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2_5.audio.bw import MelSTFT, VocoderWithBWE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _WaveformGenerator(nn.Module):
    """Controlled low-rate waveform at the vocoder/BWE boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.empty(1, 2, 1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform * self.gain


class _ResidualGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.proj(mel).squeeze(-1).repeat_interleave(8, dim=-1)


@pytest.mark.parametrize("length", [16, 19])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_bwe_waveform_against_independent_reference(length: int, dtype: torch.dtype) -> None:
    """T1: loaded stereo BWE, including gain, residual, clipping and hop padding.

    Controlled generators isolate the BWE wrapper; the oracle uses FFT analysis
    and direct sinc interpolation, not the production STFT/resampler helpers.
    """
    device = "cuda"
    model = (
        VocoderWithBWE(
            _WaveformGenerator(), _ResidualGenerator(), MelSTFT(8, 4, 8, 3), 24000, 48000, 4
        )
        .to(device)
        .eval()
    )
    time = torch.arange(8, device=device, dtype=torch.float32)
    frequency = torch.arange(5, device=device, dtype=torch.float32)[:, None]
    angle = 2 * math.pi * frequency * time / 8
    window = torch.hann_window(8, device=device)
    mel_basis = torch.tensor(
        [[0.5, 0.3, 0.2, 0, 0], [0, 0.2, 0.5, 0.3, 0], [0, 0, 0.2, 0.3, 0.5]],
        device=device,
    )
    gain = torch.tensor([[[0.75], [1.25]]], device=device)
    weight = torch.tensor([[0.03, -0.02, 0.04]], device=device)
    bias = torch.tensor([0.12], device=device)
    model.load_state_dict(
        {
            "vocoder.gain": gain,
            "bwe_generator.proj.weight": weight,
            "bwe_generator.proj.bias": bias,
            "mel_stft.stft_fn.forward_basis": torch.cat(
                (angle.cos() * window, -angle.sin() * window)
            ).unsqueeze(1),
            "mel_stft.mel_basis": mel_basis,
        },
        strict=True,
    )
    t = torch.arange(length, device=device, dtype=torch.float32)
    waveform = torch.stack((0.8 * torch.sin(t * 0.7), 0.9 * torch.cos(t * 0.4)))[None].to(dtype)
    low = waveform.float() * gain
    padded = F.pad(low, (0, (-length) % 4))
    frames = F.pad(padded, (4, 0)).unfold(-1, 8, 4)
    magnitude = torch.fft.rfft(frames * window, dim=-1).abs()
    mel = (magnitude @ mel_basis.T).clamp_min(1e-5).log()
    residual = (mel @ weight.T + bias).squeeze(-1).repeat_interleave(8, dim=-1)

    # Evaluate the interpolation sum directly, with replicated endpoint samples.
    positions = torch.arange(padded.shape[-1] * 2, device=device) / 2
    indices = torch.arange(-7, padded.shape[-1] + 7, device=device)
    distance = (positions[:, None] - indices[None, :]) * 0.99
    kernel = torch.sinc(distance) * torch.cos(distance.clamp(-6, 6) * math.pi / 12).square() * 0.99
    kernel = torch.where(distance.abs() <= 6, kernel, 0)
    skip = padded[..., indices.clamp(0, padded.shape[-1] - 1)] @ kernel.T
    summed = skip + residual
    expected = summed.clamp(-1, 1)[..., : length * 2].to(dtype)
    actual = model(waveform)
    assert actual.shape == (1, 2, length * 2)
    assert actual.dtype == dtype
    assert residual.norm() > 0.1
    assert (summed.abs() > 1).any(), "Exercise clipping as well as residual addition"
    delta = actual.float() - expected.float()
    relative_l2 = delta.norm() / expected.float().norm()
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    metrics = {
        "relative_l2": relative_l2.item(),
        "cosine": cosine.item(),
        "max_abs": delta.abs().max().item(),
    }
    print(f"BWE length={length} dtype={dtype}: {metrics}")
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert relative_l2 <= 1e-2 and cosine >= 0.9999, metrics
