# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Wan VAE implementation."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.visual_gen.models.wan import vae_loader
from tensorrt_llm._torch.visual_gen.models.wan.parallel_vae import (
    TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE,
    ParallelVAE_TrtllmWan,
    WanCausalConvHalo,
    _native_decode_chunk_size,
)
from tensorrt_llm._torch.visual_gen.models.wan.vae_loader import (
    TRTLLM_USE_DIFFUSER_VAE_ENV,
    _is_nvfp4_vae_ckpt,
    _select_dynamic_fp4_convs,
    _use_native_wan_vae,
    load_wan_vae,
)
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import (
    NVFP4WanCausalConv3d,
    WanCausalConv3d,
    WanConv2d,
    WanResidualBlock,
    WanVAE,
    WanVAEConfig,
    _decode_chunk_slices,
    _fp4_align_input_channels,
    _fp4_align_output_channels,
    _supports_nvfp4_conv3d,
    swap_wan_convs_to_fp4,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

DEVICE = "cuda"
# Parity runs in fp32 to check whether our implementation matches diffusers'
# computation, isolated from bf16 rounding noise that differs by memory layout
# (our channels_last vs diffusers' contiguous). Production runs the VAE in bf16.
DTYPE = torch.float32


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, False),
        ({"quantization_config": None}, False),
        ({"quantization_config": {"quant_algo": "FP8"}}, False),
        ({"quantization_config": {"quant_algo": "NVFP4"}}, True),
    ],
)
def test_detect_nvfp4_checkpoint(tmp_path, config, expected):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert _is_nvfp4_vae_ckpt(tmp_path) is expected


def test_fp4_conv_composes_with_parallel_vae_halo():
    base = WanCausalConv3d(8, 8, 3, padding=1)
    fp4_conv = NVFP4WanCausalConv3d(
        base,
        input_scale=1.0 / 50.0,
        absorb_norm=True,
        norm_gamma=torch.ones(8),
        norm_scale=8**0.5,
    )

    # ParallelVAE_TrtllmWan targets the native base class, which intentionally
    # includes its NVFP4 subclass when replacing convs with halo wrappers.
    assert isinstance(fp4_conv, ParallelVAE_TrtllmWan._conv3d_cls)
    halo = WanCausalConvHalo(fp4_conv, chunk_dim=4, adj_groups=[None], rank=0, world_size=2)
    assert halo.module is fp4_conv
    assert halo.absorbs_silu
    assert halo.absorbs_norm
    # The mainline halo path computes an expanded output and strips it after
    # the convolution, so its rank-local residual cannot enter the epilogue.
    assert not getattr(halo, "supports_residual_fusion", False)


def test_fp4_conv_rejects_unsupported_geometry():
    base = WanCausalConv3d(8, 8, 1)

    with pytest.raises(ValueError, match="stride-1 3x3x3"):
        NVFP4WanCausalConv3d(base)


def test_fp4_conv_capability_uses_geometry_not_parent_type():
    assert _supports_nvfp4_conv3d(WanCausalConv3d(3, 8, 3, padding=1))
    assert not _supports_nvfp4_conv3d(WanCausalConv3d(8, 8, 1))
    assert not _supports_nvfp4_conv3d(WanCausalConv3d(8, 8, (3, 1, 1), padding=(1, 0, 0)))
    assert not _supports_nvfp4_conv3d(WanConv2d(8, 8, 3, padding=1))


@pytest.mark.parametrize("use_cache", [False, True])
def test_residual_block_routes_residual_to_supported_conv2(use_cache):
    class _FakeConv(torch.nn.Module):
        supports_residual_fusion = True

        def forward(self, x, cache_x=None, *, residual=None):
            assert residual is not None
            return x + residual

    class _FakeConv1(torch.nn.Module):
        def forward(self, x, cache_x=None):
            return x

    block = WanResidualBlock(8, 8).eval()
    block.conv_shortcut = torch.nn.Identity()
    block.norm1 = torch.nn.Identity()
    block.norm2 = torch.nn.Identity()
    block.nonlinearity = torch.nn.Identity()
    block.conv1 = _FakeConv1()
    block.conv2 = _FakeConv()
    x = torch.randn(1, 8, 2, 4, 4)

    if use_cache:
        output = block(x, feat_cache=[None, None], feat_idx=[0])
    else:
        output = block(x)

    torch.testing.assert_close(output, 2 * x)


def test_swap_fp4_configures_static_norm_fusion_by_default():
    model = torch.nn.Sequential(WanResidualBlock(8, 8).eval())
    input_scales = {"0.conv1": 1.0 / 50.0, "0.conv2": 1.0 / 50.0}

    replaced, static = swap_wan_convs_to_fp4(model, input_scales)

    conv1 = model[0].conv1
    conv2 = model[0].conv2
    assert replaced == static == 2
    assert isinstance(conv1, NVFP4WanCausalConv3d)
    assert isinstance(conv2, NVFP4WanCausalConv3d)
    assert not conv1.training
    assert not conv2.training
    assert conv1.absorbs_norm
    assert conv2.absorbs_norm


def test_fp4_derived_parameters_are_invalidated_by_module_updates():
    conv = NVFP4WanCausalConv3d(WanCausalConv3d(8, 8, 3, padding=1))
    conv._fp4_pq = {}  # type: ignore[typeddict-item]
    conv._fp4_static_gs = torch.ones(1)

    conv.to(torch.float64)

    assert conv._fp4_pq is None
    assert conv._fp4_static_gs is None

    conv._fp4_pq = {}  # type: ignore[typeddict-item]
    conv.load_state_dict(conv.state_dict())
    assert conv._fp4_pq is None


def test_swap_fp4_respects_checkpoint_module_names():
    model = torch.nn.Sequential(WanResidualBlock(8, 8).eval())

    replaced, static = swap_wan_convs_to_fp4(
        model,
        {"0.conv1": 1.0 / 50.0, "0.conv2": 1.0 / 50.0},
        only_names={"0.conv2"},
    )

    assert replaced == static == 1
    assert not isinstance(model[0].conv1, NVFP4WanCausalConv3d)
    assert isinstance(model[0].conv2, NVFP4WanCausalConv3d)


def test_dynamic_fp4_selection_uses_geometry_and_config_exclusions():
    class _MixedConvs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = WanCausalConv3d(64, 64, 3, padding=1)
            self.small_input = WanCausalConv3d(16, 64, 3, padding=1)
            self.small_output = WanCausalConv3d(64, 3, 3, padding=1)
            self.temporal = WanCausalConv3d(64, 64, (3, 1, 1), padding=(1, 0, 0))
            self.pointwise = WanCausalConv3d(64, 64, 1)
            self.image = WanConv2d(64, 64, 3, padding=1)
            self.block = WanResidualBlock(64, 64).eval()

    model = _MixedConvs()
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        exclude_modules=["block.conv1"],
    )

    selected = _select_dynamic_fp4_convs(model, quant_config)
    assert selected == {"spatial", "block.conv2"}

    replaced, static = swap_wan_convs_to_fp4(model, only_names=selected)

    assert replaced == 2
    assert static == 0
    assert isinstance(model.spatial, NVFP4WanCausalConv3d)
    assert isinstance(model.block.conv2, NVFP4WanCausalConv3d)
    assert not isinstance(model.block.conv1, NVFP4WanCausalConv3d)
    assert not isinstance(model.small_input, NVFP4WanCausalConv3d)
    assert not isinstance(model.small_output, NVFP4WanCausalConv3d)
    assert not isinstance(model.temporal, NVFP4WanCausalConv3d)
    assert not isinstance(model.pointwise, NVFP4WanCausalConv3d)
    assert isinstance(model.image, WanConv2d)


def test_vae_quant_config_enables_dynamic_fp4_from_bf16(monkeypatch):
    model = torch.nn.Sequential(WanResidualBlock(64, 64).eval())
    monkeypatch.setattr(vae_loader, "_use_native_wan_vae", lambda: True)
    monkeypatch.setattr(vae_loader, "_is_nvfp4_vae_ckpt", lambda _: False)
    monkeypatch.setattr(vae_loader, "_load_native_wan_vae", lambda *args: model)

    loaded = load_wan_vae(
        "/unused",
        torch.device("cpu"),
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            exclude_modules=["0.conv1"],
        ),
    )

    assert loaded is model
    assert not isinstance(model[0].conv1, NVFP4WanCausalConv3d)
    assert isinstance(model[0].conv2, NVFP4WanCausalConv3d)


def test_explicit_bf16_config_dequantizes_fp4_checkpoint(monkeypatch):
    sentinel = torch.nn.Identity()
    call: dict[str, object] = {}
    monkeypatch.setattr(vae_loader, "_use_native_wan_vae", lambda: True)
    monkeypatch.setattr(vae_loader, "_is_nvfp4_vae_ckpt", lambda _: True)

    def _fake_load(*args, **kwargs):
        call.update(kwargs)
        return sentinel

    monkeypatch.setattr(vae_loader, "_load_nvfp4_wan_vae", _fake_load)

    loaded = load_wan_vae(
        "/unused",
        torch.device("cpu"),
        quant_config=QuantConfig(),
    )

    assert loaded is sentinel
    assert call["enable_fp4"] is False


@pytest.mark.parametrize(
    ("enable_fp4", "selected", "expected_count"),
    [
        (True, {"decoder.conv1"}, 1),
        (False, {"decoder.conv1", "decoder.conv2"}, 2),
        (True, {"decoder.conv1", "decoder.conv2"}, None),
    ],
)
def test_nvfp4_checkpoint_bf16_operator_warning(enable_fp4, selected, expected_count):
    quantized = {"decoder.conv1", "decoder.conv2"}

    with patch.object(vae_loader.logger, "warning") as warning:
        vae_loader._warn_dequantized_nvfp4_weights(quantized, selected, enable_fp4)

    if expected_count is None:
        warning.assert_not_called()
        return
    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert message.startswith(f"{expected_count} VAE convolution(s)")
    assert "cannot recover the original BF16 weights" in message


def test_wan_vae_rejects_unsupported_quantization():
    with pytest.raises(ValueError, match="supports only NVFP4"):
        load_wan_vae(
            "/unused",
            torch.device("cpu"),
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
        )


@pytest.mark.parametrize(
    ("channels", "expected"),
    [(8, 64), (64, 64), (96, 128), (192, 192), (257, 512), (512, 512)],
)
def test_fp4_input_channel_alignment(channels, expected):
    assert _fp4_align_input_channels(channels) == expected


@pytest.mark.parametrize(
    ("channels", "expected"),
    [(3, 8), (8, 8), (96, 96), (129, 256), (257, 512), (512, 512)],
)
def test_fp4_output_channel_alignment(channels, expected):
    assert _fp4_align_output_channels(channels) == expected


def _require_checkpoint(model_dir: str, env_var: str) -> Path:
    override = os.environ.get(env_var)
    checkpoint_dir = Path(override) if override else Path(llm_models_root(check=True)) / model_dir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"{model_dir} checkpoint not found at {checkpoint_dir}. Set {env_var} or LLM_MODELS_ROOT."
        )
    return checkpoint_dir


def _require_wan22_ti2v_checkpoint() -> Path:
    # Wan2.2-TI2V-5B: is_residual=True -> exercises the residual encoder/decoder path.
    return _require_checkpoint("Wan2.2-TI2V-5B-Diffusers", "DIFFUSION_MODEL_PATH_WAN22_TI2V_5B")


def _require_wan21_t2v_1p3b_checkpoint() -> Path:
    # Wan2.1-T2V-1.3B: is_residual=False -> exercises the non-residual encoder/decoder path.
    return _require_checkpoint("Wan2.1-T2V-1.3B-Diffusers", "DIFFUSION_MODEL_PATH_WAN21_1_3B")


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


def test_load_wan_vae_defaults_to_native(monkeypatch):
    checkpoint_dir = _require_wan22_ti2v_checkpoint()

    monkeypatch.delenv(TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
    wan_vae = load_wan_vae(str(checkpoint_dir), torch.device("cpu"))
    assert isinstance(wan_vae, WanVAE)


def test_load_wan_vae_honors_diffusers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_dir = _require_wan22_ti2v_checkpoint()

    monkeypatch.setenv(TRTLLM_USE_DIFFUSER_VAE_ENV, "1")
    wan_vae = load_wan_vae(str(checkpoint_dir), torch.device("cpu"))
    assert isinstance(wan_vae, AutoencoderKLWan)


def test_use_native_wan_vae_default(monkeypatch):
    """Native VAE is the default when the diffusers-fallback env is unset."""
    monkeypatch.delenv(TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
    assert _use_native_wan_vae()


@pytest.mark.parametrize("fallback_value", ["1", "2", "-1"])
def test_use_diffuser_vae_env_forces_diffusers(monkeypatch, fallback_value):
    monkeypatch.setenv(TRTLLM_USE_DIFFUSER_VAE_ENV, fallback_value)
    assert not _use_native_wan_vae()


def test_use_diffuser_vae_env_zero_keeps_native(monkeypatch):
    monkeypatch.setenv(TRTLLM_USE_DIFFUSER_VAE_ENV, "0")
    assert _use_native_wan_vae()


@pytest.mark.parametrize(
    ("num_frames", "chunk_size", "expected"),
    [
        (0, 3, []),
        (1, 4, [(0, 1)]),
        (5, 1, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        (5, 2, [(0, 1), (1, 3), (3, 5)]),
        (6, 4, [(0, 1), (1, 5), (5, 6)]),
    ],
)
def test_decode_chunk_slices_preserve_first_frame(
    num_frames: int,
    chunk_size: int,
    expected: list[tuple[int, int]],
) -> None:
    actual = [(chunk.start, chunk.stop) for chunk in _decode_chunk_slices(num_frames, chunk_size)]
    assert actual == expected


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_decode_chunk_slices_reject_invalid_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _decode_chunk_slices(num_frames=5, chunk_size=chunk_size)


@pytest.mark.parametrize(
    ("parallel_size", "dtype", "expected"),
    [
        (1, torch.bfloat16, 1),
        (1, torch.float32, 1),
        (2, torch.bfloat16, 2),
        (4, torch.bfloat16, 4),
        (4, torch.float32, 2),
        (8, torch.bfloat16, 2),
    ],
)
def test_native_decode_chunk_size_uses_tuned_or_conservative_value(
    monkeypatch: pytest.MonkeyPatch,
    parallel_size: int,
    dtype: torch.dtype,
    expected: int,
) -> None:
    monkeypatch.delenv(TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE, raising=False)
    assert _native_decode_chunk_size(parallel_size, dtype) == expected


def test_native_decode_chunk_size_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE, "5")
    assert _native_decode_chunk_size(1, torch.bfloat16) == 5


@pytest.mark.parametrize("override", ["0", "-1", "invalid"])
def test_native_decode_chunk_size_rejects_invalid_env_override(
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    monkeypatch.setenv(TLLM_WAN_VAE_DECODE_TEMPORAL_CHUNK_SIZE, override)
    with pytest.raises(ValueError, match="must be a positive integer"):
        _native_decode_chunk_size(4, torch.bfloat16)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        pytest.param(352, 640, id="360p"),
        pytest.param(704, 1280, id="720p"),
    ],
)
def test_wan22_ti2v_vae_matches_diffusers_decode_checkpoint(
    height: int,
    width: int,
):
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    reference_vae, wan_vae = _make_reference_and_wan_vae(checkpoint_dir)
    assert wan_vae.config.is_residual is True

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
        wan_decoded = wan_vae.decode(latents, temporal_chunk_size=4).sample

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
    ("height", "width"),
    [
        pytest.param(352, 640, id="360p"),
        pytest.param(704, 1280, id="720p"),
    ],
)
def test_wan22_ti2v_vae_matches_diffusers_encode_checkpoint(
    height: int,
    width: int,
):
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    reference_vae, wan_vae = _make_reference_and_wan_vae(checkpoint_dir)
    assert wan_vae.config.is_residual is True

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


# Wan2.1-T2V-1.3B has is_residual=False, so it exercises the non-residual
# encoder/decoder path (WanResidualBlock loop + WanResample / WanUpBlock) that
# the Wan2.2-TI2V-5B tests above (is_residual=True) never reach.
def test_wan21_t2v_vae_matches_diffusers_decode_checkpoint():
    checkpoint_dir = _require_wan21_t2v_1p3b_checkpoint()
    reference_vae, wan_vae = _make_reference_and_wan_vae(checkpoint_dir)
    assert wan_vae.config.is_residual is False

    torch.manual_seed(1)
    frames, height, width = 81, 480, 832
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
        wan_decoded = wan_vae.decode(latents, temporal_chunk_size=4).sample

    # fp32 parity; the residual gap is only channels_last vs contiguous conv
    # reduction order.
    _assert_close_metrics(wan_decoded, reference_decoded, max_abs=4e-3, relative_mean=1e-3)


def test_wan22_temporal_chunk4_matches_chunk1_checkpoint() -> None:
    checkpoint_dir = _require_wan22_ti2v_checkpoint()
    _, wan_vae = _make_reference_and_wan_vae(checkpoint_dir)

    torch.manual_seed(3)
    latents = torch.randn(
        1,
        wan_vae.config.z_dim,
        5,
        8,
        8,
        device=DEVICE,
        dtype=DTYPE,
    ).to(memory_format=torch.channels_last_3d)

    with torch.inference_mode():
        framewise = wan_vae.decode(latents, temporal_chunk_size=1).sample
        batched = wan_vae.decode(latents, temporal_chunk_size=4).sample

    _assert_close_metrics(batched, framewise, max_abs=4e-3, relative_mean=1e-3)


def test_wan21_t2v_vae_matches_diffusers_encode_checkpoint():
    checkpoint_dir = _require_wan21_t2v_1p3b_checkpoint()
    reference_vae, wan_vae = _make_reference_and_wan_vae(checkpoint_dir)
    assert wan_vae.config.is_residual is False

    torch.manual_seed(2)
    height, width = 480, 832
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
    _assert_close_metrics(wan_latents, reference_latents, max_abs=3e-3, relative_mean=1e-3)
