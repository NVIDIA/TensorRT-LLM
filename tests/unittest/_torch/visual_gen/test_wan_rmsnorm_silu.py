# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch/lifecycle and numerical tests for automatic BF16 Wan decoder selection."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.wan import vae_loader, wan_vae


def _norm(channels: int = 256) -> wan_vae.WanRMSNorm:
    return wan_vae.WanRMSNorm(channels, images=False).to(dtype=torch.bfloat16).eval()


def _prepared_norm() -> wan_vae.WanRMSNorm:
    norm = _norm()
    norm._silu_zero_bias = torch.zeros(256, dtype=torch.bfloat16)
    return norm


def _metadata_input(norm: wan_vae.WanRMSNorm) -> SimpleNamespace:
    # CPU-only dispatch fixture: metadata stands in for a CUDA activation;
    # no kernel is called and real CPU tensors supply parameter/buffer metadata.
    return SimpleNamespace(
        is_cuda=True,
        requires_grad=False,
        dtype=torch.bfloat16,
        ndim=5,
        shape=(1, 256, 1, 2, 3),
        device=norm.gamma.device,
        is_contiguous=lambda **_kwargs: True,
        stride=lambda: (1536, 1, 1536, 768, 256),
    )


def test_dispatch_metadata_and_singleton_strides() -> None:
    norm = _prepared_norm()
    x = _metadata_input(norm)
    with (
        torch.no_grad(),
        mock.patch.object(torch.cuda, "current_device", return_value=x.device.index),
        mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
    ):
        assert wan_vae._can_fuse_wan_norm_silu(norm, x, F.silu)
        x.stride = lambda: (999, 1, 777, 768, 256)
        assert wan_vae._can_fuse_wan_norm_silu(norm, x, F.silu)
        x.stride = lambda: (999, 1, 777, 769, 256)
        assert not wan_vae._can_fuse_wan_norm_silu(norm, x, F.silu)


@pytest.mark.parametrize(
    "case",
    [
        "unprepared",
        "training",
        "grad",
        "compiling",
        "capturing",
        "other_device",
        "sm90",
        "sm103",
        "requires_grad",
        "cpu",
        "fp32",
        "four_dimensional",
        "empty",
        "channels",
        "strided",
        "channel_last_norm",
        "bias",
        "negative_zero_bias",
        "scale",
        "gamma_dtype",
        "zero_dtype",
        "zero_shape",
        "activation",
        "forward",
        "pre_hook",
        "hook",
        "global_hook",
    ],
)
def test_unsupported_dispatch_uses_native(case: str) -> None:
    norm = _prepared_norm()
    x = _metadata_input(norm)
    activation = F.silu
    handle = None
    if case == "unprepared":
        norm._silu_zero_bias = None
    elif case == "training":
        norm.train()
    elif case == "requires_grad":
        x.requires_grad = True
    elif case == "cpu":
        x.is_cuda = False
    elif case == "fp32":
        x.dtype = torch.float32
    elif case == "four_dimensional":
        x.ndim = 4
    elif case == "empty":
        x.shape = (0, 256, 1, 2, 3)
    elif case == "channels":
        x.shape = (1, 128, 1, 2, 3)
    elif case == "strided":
        x.is_contiguous = lambda **_kwargs: False
    elif case == "channel_last_norm":
        norm.channel_first = False
    elif case == "bias":
        norm.bias = nn.Parameter(torch.zeros_like(norm.gamma))
    elif case == "negative_zero_bias":
        norm.bias = -0.0
    elif case == "scale":
        norm.scale = 1.0
    elif case == "gamma_dtype":
        norm.gamma = nn.Parameter(norm.gamma.float())
    elif case == "zero_dtype":
        norm._silu_zero_bias = norm._silu_zero_bias.float()
    elif case == "zero_shape":
        norm._silu_zero_bias = norm._silu_zero_bias[:-1]
    elif case == "activation":
        activation = F.relu
    elif case == "forward":
        norm.forward = lambda value: value
    elif case == "pre_hook":
        handle = norm.register_forward_pre_hook(lambda *_args: None)
    elif case == "hook":
        handle = norm.register_forward_hook(lambda *_args: None)
    elif case == "global_hook":
        handle = torch.nn.modules.module.register_module_forward_hook(lambda *_args: None)
    try:
        with (
            torch.set_grad_enabled(case == "grad"),
            mock.patch.object(torch.compiler, "is_compiling", return_value=case == "compiling"),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=case == "capturing"
            ),
            mock.patch.object(
                torch.cuda,
                "current_device",
                return_value=-1 if case == "other_device" else x.device.index,
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_capability",
                return_value={"sm90": (9, 0), "sm103": (10, 3)}.get(case, (10, 0)),
            ),
        ):
            assert not wan_vae._can_fuse_wan_norm_silu(norm, x, activation)
    finally:
        if handle is not None:
            handle.remove()


def test_cpu_fallback_preserves_hooks_and_gradients() -> None:
    norm = _prepared_norm()
    x = torch.randn(1, 256, 1, 2, 3, dtype=torch.bfloat16, requires_grad=True)
    calls = []
    handle = norm.register_forward_hook(lambda *_args: calls.append("norm"))
    expected = F.silu(norm(x))
    actual = wan_vae._wan_norm_silu(norm, x, F.silu)
    handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert calls == ["norm", "norm"]
    assert x.grad is not None and norm.gamma.grad is not None


def _residual_owner() -> wan_vae.WanResidualBlock:
    # Only ownership is under test; avoid allocating convolution weights.
    block = wan_vae.WanResidualBlock.__new__(wan_vae.WanResidualBlock)
    nn.Module.__init__(block)
    block.norm1 = _norm()
    block.norm2 = _norm()
    return block


def _vae_ownership_fixture() -> nn.Module:
    vae = nn.Module()
    vae.encoder = _residual_owner()
    vae.decoder = nn.Module()
    vae.decoder.block = _residual_owner()
    vae.decoder.norm_out = _norm()
    vae.decoder.attention = nn.Module()
    vae.decoder.attention.norm = wan_vae.WanRMSNorm(256).to(dtype=torch.bfloat16)
    return vae.eval()


def test_decoder_buffers_are_nonpersistent_and_reused() -> None:
    vae = _vae_ownership_fixture()
    state_keys = set(vae.state_dict())
    # Exercise allocation/registration with real CPU tensors, standing in for
    # final CUDA placement. This does not claim actual CUDA lifecycle coverage.
    with (
        mock.patch.object(
            torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True
        ),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
    ):
        wan_vae._prepare_wan_decoder_norm_silu(vae)
        selected = (vae.decoder.block.norm1, vae.decoder.block.norm2, vae.decoder.norm_out)
        buffers = [norm._silu_zero_bias for norm in selected]
        assert len({id(buffer) for buffer in buffers}) == 3
        for buffer in buffers:
            assert torch.count_nonzero(buffer.view(torch.int16)).item() == 0
        wan_vae._prepare_wan_decoder_norm_silu(vae)
        assert all(norm._silu_zero_bias is buffer for norm, buffer in zip(selected, buffers))
    assert vae.encoder.norm1._silu_zero_bias is None
    assert vae.encoder.norm2._silu_zero_bias is None
    assert vae.decoder.attention.norm._silu_zero_bias is None
    assert set(vae.state_dict()) == state_keys
    vae.load_state_dict(vae.state_dict(), strict=True)
    vae.to(dtype=torch.float32)
    assert all(norm._silu_zero_bias.dtype == torch.float32 for norm in selected)


def test_unprepared_cpu_model_does_not_allocate_buffers() -> None:
    vae = _vae_ownership_fixture()
    wan_vae._prepare_wan_decoder_norm_silu(vae)
    assert all(
        norm._silu_zero_bias is None
        for norm in vae.modules()
        if isinstance(norm, wan_vae.WanRMSNorm)
    )


@pytest.mark.parametrize("channels", [256, 512, 1024])
@pytest.mark.parametrize("frames", [1, 4])
@pytest.mark.parametrize("case", ["random", "zero", "near_clamp", "subnormal", "silu_tails"])
def test_cuda_matches_native_error_bound(channels: int, frames: int, case: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BF16 fused kernel")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Automatic fused selection is limited to SM100")
    norm = wan_vae.WanRMSNorm(channels, images=False).to(device="cuda", dtype=torch.bfloat16).eval()
    norm._silu_zero_bias = torch.zeros(channels, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(42)
    x = torch.randn(1, channels, frames, 2, 3, device="cuda", dtype=torch.bfloat16)
    if case == "zero":
        x.zero_()
    elif case == "near_clamp":
        x.fill_(1e-12 / channels**0.5)
        x[..., 0, :] *= 0.5
        x[..., 1, :] *= 2.0
    elif case == "subnormal":
        x.fill_(2.0**-133)
    elif case == "silu_tails":
        norm.gamma.data.fill_(8.0)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    original, gamma = x.clone(), norm.gamma.detach().clone()
    with torch.no_grad():
        native = F.silu(norm(x))
        # FP32 oracle has no BF16 intermediate rounding; the dynamic bound
        # scales with the native implementation's error for the same input.
        reference = F.silu(F.normalize(x.float(), dim=1) * norm.scale * norm.gamma.float())
        from tensorrt_llm._torch.visual_gen.models.wan import rmsnorm_silu

        with mock.patch.object(
            rmsnorm_silu, "rmsnorm_silu", wraps=rmsnorm_silu.rmsnorm_silu
        ) as fused:
            actual = wan_vae._wan_norm_silu(norm, x, F.silu)
            fused.assert_called_once()
    assert torch.isfinite(actual).all()
    native_error = (native.float() - reference).abs().max()
    candidate_error = (actual.float() - reference).abs().max()
    assert candidate_error <= 2 * native_error + 1e-6
    assert actual.shape == x.shape and actual.stride() == x.stride()
    assert actual.dtype == x.dtype
    assert actual.data_ptr() != x.data_ptr()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(norm.gamma, gamma, rtol=0, atol=0)
    # Instrumentation remains observable through the native fallback.
    with torch.no_grad(), mock.patch.object(norm, "forward", wraps=norm.forward) as forward:
        # Instrumented norms deliberately take the native path.
        fallback = wan_vae._wan_norm_silu(norm, x, F.silu)
        forward.assert_called_once()
    torch.testing.assert_close(fallback, native, rtol=0, atol=0)


@pytest.mark.parametrize("route", ["ordinary", "dynamic_fp4", "packed_fp4", "packed_dequant"])
def test_loader_prepares_only_ordinary_bf16(route: str) -> None:
    checkpoint_is_fp4 = route in ("packed_fp4", "packed_dequant")
    algo = vae_loader.QuantAlgo.NVFP4 if route in ("dynamic_fp4", "packed_fp4") else None
    quant = SimpleNamespace(quant_algo=algo)
    loaded = mock.sentinel.loaded_vae
    with (
        mock.patch.object(vae_loader, "_use_native_wan_vae", return_value=True),
        mock.patch.object(vae_loader, "_is_nvfp4_vae_ckpt", return_value=checkpoint_is_fp4),
        mock.patch.object(vae_loader, "_load_native_wan_vae", return_value=loaded),
        mock.patch.object(vae_loader, "_load_nvfp4_wan_vae", return_value=loaded),
        mock.patch.object(vae_loader, "_select_dynamic_fp4_convs", return_value=set()),
        mock.patch.object(vae_loader, "_validate_dynamic_weight_request"),
        mock.patch.object(vae_loader, "_resolve_input_scales"),
        mock.patch.object(vae_loader, "_resolve_nvfp4_device_support"),
        mock.patch.object(wan_vae, "swap_wan_convs_to_fp4", return_value=(0, 0)),
        mock.patch.object(vae_loader, "_prepare_wan_decoder_norm_silu") as prepare,
    ):
        assert (
            vae_loader.load_wan_vae("unused-checkpoint", torch.device("cpu"), quant_config=quant)
            is loaded
        )
    if route == "ordinary":
        prepare.assert_called_once_with(loaded)
    else:
        prepare.assert_not_called()


@pytest.mark.parametrize("capability", [(9, 0), (10, 3)])
def test_other_architectures_do_not_prepare(capability: tuple[int, int]) -> None:
    vae = _vae_ownership_fixture()
    with (
        mock.patch.object(
            torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True
        ),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=capability),
    ):
        wan_vae._prepare_wan_decoder_norm_silu(vae)
    assert all(
        norm._silu_zero_bias is None
        for norm in vae.modules()
        if isinstance(norm, wan_vae.WanRMSNorm)
    )
