# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WinoVae component tests: baseline equals the native WanVAE; each option stays within tolerance."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE
from tensorrt_llm._torch.visual_gen.models.wan.wino_vae import WinoVae, WinoVaeOptions

# Cosmos3-shaped (residual blocks, patch 2, 12 in/out channels) but tiny.
SMALL_CONFIG = {
    "base_dim": 16,
    "decoder_base_dim": 24,
    "z_dim": 8,
    "dim_mult": [1, 2, 2, 2],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def test_rejects_temporal_factor_mismatch():
    with pytest.raises(ValueError, match="scale_factor_temporal=8"):
        WinoVae({**SMALL_CONFIG, "scale_factor_temporal": 8})


def _pair(options: WinoVaeOptions | None = None) -> tuple[WanVAE, WinoVae]:
    torch.manual_seed(0)
    native = WanVAE(SMALL_CONFIG)
    fast = WinoVae(SMALL_CONFIG, options)
    fast.load_state_dict(native.state_dict(), strict=True)
    return (
        native.to(device=DEVICE, dtype=DTYPE).eval(),
        fast.to(device=DEVICE, dtype=DTYPE).eval(),
    )


def _inputs(frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    pixels = (torch.rand(1, 3, frames, 64, 96, generator=gen) * 2 - 1).to(DEVICE, DTYPE)
    latents = torch.randn(1, SMALL_CONFIG["z_dim"], (frames - 1) // 4 + 1, 4, 6, generator=gen).to(
        DEVICE, DTYPE
    )
    return pixels, latents


@pytest.mark.parametrize("frames", [1, 5, 9])
def test_wino_vae_baseline_matches_native_exactly(frames: int) -> None:
    native, fast = _pair()
    pixels, latents = _inputs(frames)
    with torch.inference_mode():
        assert torch.equal(
            fast.encode(pixels, return_dict=False)[0].mode(),
            native.encode(pixels, return_dict=False)[0].mode(),
        )
        assert torch.equal(
            fast.decode(latents, return_dict=False)[0],
            native.decode(latents, return_dict=False)[0],
        )


def test_wino_vae_state_dict_keys_match_native() -> None:
    native, fast = _pair()
    assert native.state_dict().keys() == fast.state_dict().keys()


def test_wino_vae_options_reject_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown WinoVaeOptions"):
        WinoVaeOptions.from_dict({"nope": True})


OPTION_SETS = [
    {"fused_norm": True, "norm_mode": "fp32"},
    {"fused_norm": True, "norm_mode": "reference"},
    {"fused_norm": True, "fused_residual": True},
    {"fused_norm": True, "fused_residual": True, "norm_mode": "reference"},
    {"fused_upsample": True},
    {"fused_norm": True, "fused_residual": True, "fused_upsample": True},
    {"fused_norm": True, "fused_residual": True, "fused_bias": True},
    {"fused_norm": True, "fused_residual": True, "fused_bias": True, "norm_mode": "reference"},
    {"fused_norm": True, "fused_residual": True, "fused_upsample": True, "fused_bias": True},
    {"fused_norm": True, "fused_upsample": True, "fused_upsample_norm": True},
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_upsample": True,
        "fused_upsample_norm": True,
        "fused_bias": True,
        "norm_mode": "reference",
    },
    {
        "fused_norm": True,
        "winograd_temporal": True,
        "fused_upsample": True,
        "fused_upsample_norm": True,
        "fused_bias": True,
    },
    {"conv_out_frame_split": 3},
    {"first_frame_conv2d": True},
    {"fold_encoder_tail": True},
    {"encoder_chunk_frames": 8},
    {"fused_output": True},
    {"fused_norm": True, "fused_residual": True, "fold_encoder_tail": True, "fused_output": True},
    {
        "fused_norm": True,
        "first_frame_conv2d": True,
        "conv_slabs": True,
        "conv_out_frame_split": 3,
        "fold_encoder_tail": True,
    },
    {"fused_norm": True, "winograd_temporal": True},
    {"fused_norm": True, "winograd_temporal": True, "fused_upsample": True},
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_upsample": True,
        "fused_upsample_norm": True,
        "fused_bias": True,
        "first_frame_conv2d": True,
        "winograd_temporal": True,
        "winograd_spatial": "all",
        "conv_slabs": True,
        "fused_attention": True,
    },
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_bias": True,
        "winograd_temporal": True,
        "norm_mode": "reference",
    },
    {"fused_norm": True, "conv_slabs": True, "cudnn_frontend": True},
    {"fused_norm": True, "winograd_temporal": True, "winograd_temporal_order": 2},
    {
        "fused_norm": True,
        "winograd_temporal": True,
        "winograd_temporal_order": 2,
        "winograd_slab_dtype": "bf16",
    },
    {"fused_norm": True, "winograd_temporal": True, "winograd_spatial": "all"},
    {
        "fused_norm": True,
        "winograd_temporal": True,
        "winograd_temporal_order": 2,
        "winograd_slab_dtype": "bf16",
        "winograd_spatial": "all",
    },
    {
        "fused_norm": True,
        "winograd_temporal": True,
        "winograd_spatial": "all",
        "winograd_spatial_kernel": "sm100",
    },
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_upsample": True,
        "fused_upsample_norm": True,
        "fused_bias": True,
        "first_frame_conv2d": True,
        "winograd_temporal": True,
        "winograd_spatial": "all",
        "conv_slabs": True,
        "fused_attention": True,
    },
    {"fused_norm": True, "conv_slabs": True},
    {"fused_norm": True, "fused_attention": True},
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_bias": True,
        "fused_attention": True,
        "norm_mode": "reference",
    },
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_bias": True,
        "conv_slabs": True,
        "first_frame_conv2d": True,
    },
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_upsample": True,
        "fused_upsample_norm": True,
        "fused_bias": True,
        "first_frame_conv2d": True,
        "winograd_temporal": True,
        "conv_slabs": True,
        "cudnn_benchmark": "auto",
        "conv_out_frame_split": 3,
    },
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_bias": True,
        "winograd_temporal": True,
        "norm_mode": "reference",
    },
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_upsample": True,
        "fused_upsample_norm": True,
        "fused_bias": True,
        "first_frame_conv2d": True,
        "winograd_temporal": True,
    },
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_upsample": True,
        "fused_upsample_norm": True,
        "fused_bias": True,
        "first_frame_conv2d": True,
    },
    {"cudnn_benchmark": "auto"},
    {
        "fused_norm": True,
        "fused_residual": True,
        "fused_upsample": True,
        "fused_bias": True,
        "conv_out_frame_split": 3,
        "cudnn_benchmark": "auto",
    },
]


@pytest.mark.parametrize(
    "option", OPTION_SETS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items())
)
@pytest.mark.parametrize("frames", [1, 9])
def test_each_option_stays_close_to_native(option: dict, frames: int) -> None:
    native, fast = _pair(WinoVaeOptions(**option))
    pixels, latents = _inputs(frames)
    with torch.inference_mode():
        for actual, expected in (
            (
                fast.encode(pixels, return_dict=False)[0].mode(),
                native.encode(pixels, return_dict=False)[0].mode(),
            ),
            (
                fast.decode(latents, return_dict=False)[0],
                native.decode(latents, return_dict=False)[0],
            ),
        ):
            diff = actual.float() - expected.float()
            rel_l2 = diff.norm() / expected.float().norm().clamp_min(1e-12)
            assert torch.isfinite(actual.float()).all()
            # Both modes drift from native only through last-bit effects (fp32 mode: fewer
            # roundings; reference mode: fp32 reduction order), amplified by the random-weight
            # network. Bitwise equality of the kernel itself is checked in test_wino_vae_kernels.
            assert rel_l2.item() < 2e-2, f"{option}: rel_l2={rel_l2.item():.3e}"


def test_encode_mode_matches_posterior_mode() -> None:
    _, fast = _pair(WinoVaeOptions(fold_encoder_tail=True))
    pixels, _ = _inputs(9)
    with torch.inference_mode():
        mean_only = fast.encode_mode(pixels)
        full = fast.encode(pixels, return_dict=False)[0]
        assert torch.equal(mean_only.float(), full.mode().float())
        # The lazy posterior still serves the sampling path.
        assert full.sample().shape == full.mode().shape
        assert torch.isfinite(full.std.float()).all()


def test_fold_latent_affine_matches_eager() -> None:
    _, fast = _pair(WinoVaeOptions())
    _, latents = _inputs(5)
    z = SMALL_CONFIG["z_dim"]
    gen = torch.Generator().manual_seed(3)
    mean = (torch.randn(z, generator=gen) * 0.5).to(DEVICE)
    std = (torch.rand(z, generator=gen) + 0.5).to(DEVICE)
    with torch.inference_mode():
        eager = fast.decode(
            (latents.float() * std.view(1, z, 1, 1, 1) + mean.view(1, z, 1, 1, 1)).to(DTYPE),
            return_dict=False,
        )[0]
        fast.fold_latent_affine(mean, std)
        folded = fast.decode(latents, return_dict=False)[0]
        fast.clear_latent_affine()
        again = fast.decode(latents, return_dict=False)[0]
    diff = (folded.float() - eager.float()).norm() / eager.float().norm().clamp_min(1e-12)
    assert diff.item() < 2e-2, f"rel_l2={diff.item():.3e}"
    assert not torch.equal(again.float(), folded.float())  # the fold is really off again


@pytest.mark.skipif(DEVICE != "cuda", reason="frame buffers need the CUDA norm kernel")
def test_frame_buffer_grows_for_a_longer_chunk_in_a_new_clip() -> None:
    """A clip decoded in short chunks sizes the convs' frame buffers; the next clip's longer
    chunk must get a deeper buffer, not a window clipped at the end of the old one."""
    native, fast = _pair(
        WinoVaeOptions(
            fused_norm=True,
            fused_residual=True,
            conv_slabs=True,
            conv_out_frame_split=3,
            first_frame_conv2d=True,
        )
    )
    assert fast.options.use_slabs
    _, latents = _inputs(21)
    with torch.inference_mode():
        expected = native.decode(latents, return_dict=False)[0].float()
        fast.decode(latents, return_dict=False, temporal_chunk_size=1)
        actual = fast.decode(latents, return_dict=False, temporal_chunk_size=5)[0].float()
    rel_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    assert torch.isfinite(actual).all()
    assert rel_l2.item() < 2e-2, f"rel_l2={rel_l2.item():.3e}"


@pytest.mark.skipif(DEVICE != "cuda", reason="frame buffers and Winograd need the CUDA norm kernel")
@pytest.mark.parametrize("ffc2d,frames,chunk", [(False, 9, 2), (True, 21, 3), (False, 17, 4)])
def test_frame_buffer_to_winograd_chunk_transition(ffc2d: bool, frames: int, chunk: int) -> None:
    """Odd-length chunks (the first one, a tail) run the frame-buffer convs, even-length chunks
    the temporal Winograd convs; the history frames must carry over in both directions."""
    native, fast = _pair(
        WinoVaeOptions(
            fused_norm=True, conv_slabs=True, winograd_temporal=True, first_frame_conv2d=ffc2d
        )
    )
    assert fast.options.use_slabs and fast.options.use_winograd
    _, latents = _inputs(frames)
    with torch.inference_mode():
        expected = native.decode(latents, return_dict=False)[0].float()
        actual = fast.decode(latents, return_dict=False, temporal_chunk_size=chunk)[0].float()
    rel_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    assert torch.isfinite(actual).all()
    assert rel_l2.item() < 2e-2, f"rel_l2={rel_l2.item():.3e}"


@pytest.mark.skipif(DEVICE != "cuda", reason="the deferred bias needs the CUDA frame ops")
def test_non_residual_encoder_keeps_downsample_bias() -> None:
    """Without residual down blocks nothing consumes a deferred downsample bias, so the conv
    must keep it."""
    cfg = {**SMALL_CONFIG, "is_residual": False}
    torch.manual_seed(0)
    native = WanVAE(cfg)
    fast = WinoVae(cfg, WinoVaeOptions(fused_norm=True, fused_residual=True, fused_bias=True))
    fast.load_state_dict(native.state_dict(), strict=True)
    native = native.to(DEVICE, DTYPE).eval()
    fast = fast.to(DEVICE, DTYPE).eval()
    pixels, _ = _inputs(9)
    with torch.inference_mode():
        expected = native.encode(pixels, return_dict=False)[0].mode().float()
        actual = fast.encode(pixels, return_dict=False)[0].mode().float()
    rel_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    assert rel_l2.item() < 2e-2, f"rel_l2={rel_l2.item():.3e}"


@pytest.mark.skipif(DEVICE != "cuda", reason="the dtype contract applies to the CUDA kernels")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_optimized_paths_reject_non_bf16(dtype: torch.dtype) -> None:
    """The optimized kernels read and write bf16; any other dtype is refused up front instead
    of being reinterpreted. The all-off model stays dtype-agnostic."""
    _, fast = _pair(WinoVaeOptions(fused_norm=True, winograd_temporal=True))
    fast = fast.to(dtype)
    pixels, latents = _inputs(5)
    with torch.inference_mode(), pytest.raises(TypeError, match="bf16"):
        fast.decode(latents.to(dtype), return_dict=False)
    with torch.inference_mode(), pytest.raises(TypeError, match="bf16"):
        fast.encode(pixels.to(dtype), return_dict=False)
    _, plain = _pair()
    plain = plain.to(dtype)
    with torch.inference_mode():
        out = plain.decode(latents.to(dtype), return_dict=False)[0]
    assert out.dtype is dtype
