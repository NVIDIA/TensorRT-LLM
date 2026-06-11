# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Wan VAE implementation."""

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.vae_loader import (
    TRTLLM_WAN_VAE_BACKEND_FALLBACK_ENV,
    load_wan_vae,
)
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE, WanVAEConfig

DEVICE = "cuda"
DTYPE = torch.bfloat16
DIFFUSERS_WAN_VAE_REFERENCE_VERSION = "0.37.1"


def _llm_models_root() -> Path:
    root = Path(os.environ.get("LLM_MODELS_ROOT", "/home/scratch.trt_llm_data_ci/llm-models/"))
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    return root


def _wan22_ti2v_checkpoint() -> Path:
    return Path(
        os.environ.get("WAN22_TI2V_MODEL_PATH", _llm_models_root() / "Wan2.2-TI2V-5B-Diffusers")
    )


def _require_wan22_ti2v_checkpoint() -> Path:
    checkpoint_dir = _wan22_ti2v_checkpoint()
    if not checkpoint_dir.exists():
        pytest.skip(
            f"Wan2.2 TI2V checkpoint not found at {checkpoint_dir}. "
            "Set WAN22_TI2V_MODEL_PATH or LLM_MODELS_ROOT."
        )
    return checkpoint_dir


def _require_cuda_memory(min_gb: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Wan VAE checkpoint parity requires CUDA.")
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb < min_gb:
        pytest.skip(f"Wan VAE checkpoint parity requires at least {min_gb} GB GPU memory.")


def _require_diffusers_wan_vae_reference_version() -> None:
    try:
        diffusers_version = version("diffusers")
    except PackageNotFoundError:
        pytest.skip("Wan VAE checkpoint parity requires diffusers.")

    if diffusers_version != DIFFUSERS_WAN_VAE_REFERENCE_VERSION:
        pytest.skip(
            "Wan2.2-TI2V VAE checkpoint parity is tied to diffusers "
            f"{DIFFUSERS_WAN_VAE_REFERENCE_VERSION}; diffusers {diffusers_version} "
            "is not a compatible reference for this test."
        )


def _make_reference_and_wan_vae(
    layout_mode: str, checkpoint_dir: Path
) -> tuple[torch.nn.Module, torch.nn.Module]:
    _require_diffusers_wan_vae_reference_version()
    wan_autoencoder = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")

    vae_dir = checkpoint_dir / "vae"
    reference_vae = (
        wan_autoencoder.AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=DTYPE)
        .to(DEVICE)
        .eval()
    )
    wan_vae = WanVAE(
        WanVAEConfig.from_json_file(vae_dir / "config.json"),
        layout_mode=layout_mode,
    )
    wan_vae.load_diffusers_state_dict(reference_vae.state_dict(), strict=True)
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


def _apply_layout_mode(tensor: torch.Tensor, layout_mode: str) -> torch.Tensor:
    if layout_mode == "channels_last":
        return tensor.to(memory_format=torch.channels_last_3d)
    return tensor


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
    wan_autoencoder = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    device = torch.device("cpu")

    monkeypatch.delenv(TRTLLM_WAN_VAE_BACKEND_FALLBACK_ENV, raising=False)
    wan_vae = load_wan_vae(str(checkpoint_dir), device)
    assert isinstance(wan_vae, WanVAE)
    assert wan_vae.layout_mode == "channels_last"
    del wan_vae

    monkeypatch.setenv(TRTLLM_WAN_VAE_BACKEND_FALLBACK_ENV, "1")
    reference_vae = load_wan_vae(str(checkpoint_dir), device)
    assert isinstance(reference_vae, wan_autoencoder.AutoencoderKLWan)


@pytest.mark.parametrize(
    ("case_name", "height", "width", "min_gpu_memory_gb"),
    [
        ("360p", 352, 640, 40),
        ("720p", 704, 1280, 80),
    ],
)
@pytest.mark.parametrize("layout_mode", ["contiguous", "channels_last"])
def test_wan_vae_matches_diffusers_decode_checkpoint(
    layout_mode: str,
    case_name: str,
    height: int,
    width: int,
    min_gpu_memory_gb: int,
):
    del case_name
    _require_cuda_memory(min_gpu_memory_gb)
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    reference_vae, wan_vae = _make_reference_and_wan_vae(layout_mode, checkpoint_dir)

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
    )

    latents = _apply_layout_mode(latents, layout_mode)

    with torch.inference_mode():
        reference_decoded = reference_vae.decode(latents).sample
        wan_decoded = wan_vae.decode(latents).sample

    if layout_mode == "contiguous":
        _assert_close_metrics(
            wan_decoded,
            reference_decoded,
            max_abs=1e-2,
            relative_mean=1e-4,
        )
    else:
        # channels_last uses a different BF16 F.normalize/RMSNorm reduction order.
        # See wan_vae_project_notes/experiments.md for the numerical analysis.
        _assert_close_metrics(
            wan_decoded,
            reference_decoded,
            max_abs=8e-2,
            relative_mean=1e-3,
        )

    if layout_mode == "channels_last":
        assert wan_decoded.is_contiguous(memory_format=torch.channels_last_3d)
        counts = _conv_weight_layout_counts(wan_vae)
        assert counts["conv3d_channels_last_3d"] == counts["conv3d_total"]
        assert counts["conv2d_channels_last"] == counts["conv2d_total"]


@pytest.mark.parametrize(
    ("case_name", "height", "width", "min_gpu_memory_gb"),
    [
        ("360p", 352, 640, 40),
        ("720p", 704, 1280, 80),
    ],
)
@pytest.mark.parametrize("layout_mode", ["contiguous", "channels_last"])
def test_wan_vae_matches_diffusers_encode_checkpoint(
    layout_mode: str,
    case_name: str,
    height: int,
    width: int,
    min_gpu_memory_gb: int,
):
    del case_name
    _require_cuda_memory(min_gpu_memory_gb)
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    reference_vae, wan_vae = _make_reference_and_wan_vae(layout_mode, checkpoint_dir)

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
    )
    video = _apply_layout_mode(video, layout_mode)

    with torch.inference_mode():
        reference_latents = reference_vae.encode(video).latent_dist.mode()
        wan_latents = wan_vae.encode(video).latent_dist.mode()

    if layout_mode == "contiguous":
        _assert_close_metrics(
            wan_latents,
            reference_latents,
            max_abs=2e-2,
            relative_mean=1e-4,
        )
    else:
        # channels_last uses a different BF16 F.normalize/RMSNorm reduction order.
        # See wan_vae_project_notes/experiments.md for the numerical analysis.
        _assert_close_metrics(
            wan_latents,
            reference_latents,
            max_abs=8e-2,
            relative_mean=3e-3,
        )


def test_wan_vae_tiling_fails_early():
    wan_vae = WanVAE(layout_mode="contiguous")

    with pytest.raises(NotImplementedError, match="does not support tiled encode/decode"):
        wan_vae.enable_tiling()
