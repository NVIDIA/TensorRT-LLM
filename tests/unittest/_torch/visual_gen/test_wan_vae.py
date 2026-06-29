# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Wan VAE implementation."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.visual_gen.models.wan.vae_loader import (
    TRTLLM_USE_DIFFUSER_VAE_ENV,
    _use_native_wan_vae,
    load_wan_vae,
)
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE, WanVAEConfig

DEVICE = "cuda"
# Parity runs in fp32 to check whether our implementation matches diffusers'
# computation, isolated from bf16 rounding noise that differs by memory layout
# (our channels_last vs diffusers' contiguous). Production runs the VAE in bf16.
DTYPE = torch.float32


def _wan22_ti2v_checkpoint() -> Path:
    override = os.environ.get("WAN22_TI2V_MODEL_PATH")
    if override:
        return Path(override)
    return Path(llm_models_root(check=True)) / "Wan2.2-TI2V-5B-Diffusers"


def _require_wan22_ti2v_checkpoint() -> Path:
    checkpoint_dir = _wan22_ti2v_checkpoint()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Wan2.2 TI2V checkpoint not found at {checkpoint_dir}. "
            "Set WAN22_TI2V_MODEL_PATH or LLM_MODELS_ROOT."
        )
    return checkpoint_dir


def _make_reference_and_wan_vae(
    checkpoint_dir: Path,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    vae_dir = checkpoint_dir / "vae"
    reference_vae = (
        AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=DTYPE).to(DEVICE).eval()
    )
    wan_vae = WanVAE(WanVAEConfig.from_json_file(vae_dir / "config.json"))
    wan_vae.load_state_dict(reference_vae.state_dict(), strict=True)
    wan_vae = wan_vae.to(device=DEVICE, dtype=DTYPE).eval()

    return reference_vae, wan_vae


def _assert_close_metrics(
    actual: torch.Tensor, expected: torch.Tensor, *, max_abs: float, relative_mean: float
) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    diff = (actual_float - expected_float).abs()
    denom = expected_float.abs().mean().clamp_min(1e-6)
    assert diff.max().item() <= max_abs
    assert (diff.mean() / denom).item() <= relative_mean


def _conv_weight_layout_counts(model: torch.nn.Module) -> dict[str, int]:
    counts = {
        "conv3d_total": 0,
        "conv3d_channels_last_3d": 0,
        "conv2d_total": 0,
        "conv2d_channels_last": 0,
    }
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            counts["conv3d_total"] += 1
            if module.weight.is_contiguous(memory_format=torch.channels_last_3d):
                counts["conv3d_channels_last_3d"] += 1
        elif isinstance(module, torch.nn.Conv2d):
            counts["conv2d_total"] += 1
            if module.weight.is_contiguous(memory_format=torch.channels_last):
                counts["conv2d_channels_last"] += 1
    return counts


def test_load_wan_vae_checkpoint_selection(monkeypatch):
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    device = torch.device("cpu")

    monkeypatch.delenv(TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
    wan_vae = load_wan_vae(str(checkpoint_dir), device)
    assert isinstance(wan_vae, WanVAE)
    del wan_vae

    monkeypatch.setenv(TRTLLM_USE_DIFFUSER_VAE_ENV, "1")
    reference_vae = load_wan_vae(str(checkpoint_dir), device)
    assert isinstance(reference_vae, AutoencoderKLWan)


def test_use_native_wan_vae_selects_native_unless_parallel(monkeypatch):
    single_gpu_mapping = MagicMock(world_size=1, parallel_vae_size=1)
    multi_gpu_mapping = MagicMock(world_size=2, parallel_vae_size=1)
    parallel_vae_mapping = MagicMock(world_size=2, parallel_vae_size=2)

    monkeypatch.delenv(TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
    # Selection depends only on parallel_vae_size, not world_size.
    assert _use_native_wan_vae(single_gpu_mapping)
    assert _use_native_wan_vae(multi_gpu_mapping)
    assert not _use_native_wan_vae(parallel_vae_mapping)


@pytest.mark.parametrize("fallback_value", ["1", "2", "-1"])
def test_use_diffuser_vae_env_forces_diffusers(monkeypatch, fallback_value):
    single_gpu_mapping = MagicMock(world_size=1, parallel_vae_size=1)

    monkeypatch.setenv(TRTLLM_USE_DIFFUSER_VAE_ENV, fallback_value)
    assert not _use_native_wan_vae(single_gpu_mapping)


def test_use_diffuser_vae_env_zero_keeps_native(monkeypatch):
    single_gpu_mapping = MagicMock(world_size=1, parallel_vae_size=1)

    monkeypatch.setenv(TRTLLM_USE_DIFFUSER_VAE_ENV, "0")
    assert _use_native_wan_vae(single_gpu_mapping)


@pytest.mark.parametrize(
    ("case_name", "height", "width"),
    [
        ("360p", 352, 640),
        ("720p", 704, 1280),
    ],
)
def test_wan_vae_matches_diffusers_decode_checkpoint(
    case_name: str,
    height: int,
    width: int,
):
    del case_name
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    reference_vae, wan_vae = _make_reference_and_wan_vae(checkpoint_dir)

    torch.manual_seed(1)
    frames = 81
    latent_frames = 1 + (frames - 1) // wan_vae.config.scale_factor_temporal
    latents = torch.randn(
        1,
        wan_vae.config.z_dim,
        latent_frames,
        height // wan_vae.config.scale_factor_spatial,
        width // wan_vae.config.scale_factor_spatial,
        device=DEVICE,
        dtype=DTYPE,
    ).to(memory_format=torch.channels_last_3d)

    with torch.inference_mode():
        reference_decoded = reference_vae.decode(latents).sample
        wan_decoded = wan_vae.decode(latents).sample

    # fp32 parity; the residual gap is only channels_last vs contiguous conv
    # reduction order.
    _assert_close_metrics(
        wan_decoded,
        reference_decoded,
        max_abs=4e-3,
        relative_mean=1e-3,
    )

    assert wan_decoded.is_contiguous(memory_format=torch.channels_last_3d)
    counts = _conv_weight_layout_counts(wan_vae)
    assert counts["conv3d_channels_last_3d"] == counts["conv3d_total"]
    assert counts["conv2d_channels_last"] == counts["conv2d_total"]


@pytest.mark.parametrize(
    ("case_name", "height", "width"),
    [
        ("360p", 352, 640),
        ("720p", 704, 1280),
    ],
)
def test_wan_vae_matches_diffusers_encode_checkpoint(
    case_name: str,
    height: int,
    width: int,
):
    del case_name
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    reference_vae, wan_vae = _make_reference_and_wan_vae(checkpoint_dir)

    torch.manual_seed(2)
    video = (
        torch.rand(
            1,
            wan_vae.config.public_video_channels,
            81,
            height,
            width,
            device=DEVICE,
            dtype=DTYPE,
        )
        .mul_(2.0)
        .sub_(1.0)
    ).to(memory_format=torch.channels_last_3d)

    with torch.inference_mode():
        reference_latents = reference_vae.encode(video).latent_dist.mode()
        wan_latents = wan_vae.encode(video).latent_dist.mode()

    # fp32 parity; the residual gap is only channels_last vs contiguous conv
    # reduction order.
    _assert_close_metrics(
        wan_latents,
        reference_latents,
        max_abs=2e-3,
        relative_mean=1e-3,
    )
