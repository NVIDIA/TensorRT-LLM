# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic FLUX.3 fixtures and component comparisons."""

import torch
import torch.nn.functional as F


def small_model_params(hidden_size=256, flow_map=False):
    return dict(
        in_channels={"image": 128, "image_cond": 128},
        sequence={"x_image": "image", "x_image_cond": "image_cond"},
        vec_in_dim=None,
        context_in_dim=48,
        hidden_size=hidden_size,
        num_heads=hidden_size // 128,
        depth=1,
        depth_single_blocks=2,
        depth_late_blocks=0,
        axes_dim=[32, 32, 32, 32],
        theta=10000,
        mlp_ratio=3.0,
        out_modes=["image"],
        qkv_bias=False,
        flow_map=flow_map,
    )


def assert_parity(actual, expected, *, relative_l2=1e-2, cosine=0.9999, zero_atol=0.0):
    assert actual.shape == expected.shape
    shape = tuple(expected.shape)
    channel_mean_shift = (
        (actual.float() - expected.float()).reshape(-1, shape[-1]).mean(0).abs().max().item()
    )
    actual, expected = actual.float().flatten(), expected.float().flatten()
    delta = actual - expected
    norm = expected.norm().item()
    stats = dict(
        shape=shape,
        max_channel_mean_shift=channel_mean_shift,
        p99_abs=delta.abs().quantile(0.99).item(),
        relative_l2=delta.norm().item() / norm if norm else delta.norm().item(),
        cosine=F.cosine_similarity(actual, expected, dim=0).item()
        if norm
        else float(actual.norm() == 0),
        max_abs=delta.abs().max().item(),
    )
    print(stats)
    if norm == 0:
        torch.testing.assert_close(actual, expected, rtol=0, atol=zero_atol, msg=str(stats))
    else:
        assert stats["relative_l2"] <= relative_l2 and stats["cosine"] >= cosine, stats


def small_weights(hidden_size=256, flow_map=False):
    """BF16 package tensors for the small image-only DiT, before runtime remapping."""
    h = hidden_size
    shapes = {
        "emb_in.image.weight": (h, 128),
        "emb_in.image_cond.weight": (h, 128),
        "txt_in.weight": (h, 48),
        "time_in.in_layer.weight": (h, 256),
        "time_in.out_layer.weight": (h, h),
        "final_layer.image.linear.weight": (128, h),
        "final_layer.image.adaLN_modulation.1.weight": (2 * h, h),
    }
    for stage in ("early", "single"):
        for mode in ("image", "image_cond", "txt"):
            shapes[f"{stage}_stream_modulations.{mode}.lin.weight"] = (3 * h, h)
    for prefix, depth in (
        ("content_mode_blocks.image", 1),
        ("content_mode_blocks.image_cond", 1),
        ("txt_mode_blocks", 1),
        ("single_blocks", 2),
    ):
        for index in range(depth):
            for name, shape in {
                "q_proj.weight": (h, h),
                "k_proj.weight": (h, h),
                "v_proj.weight": (h, h),
                "attn_out.weight": (h, h),
                "mlp_in.weight": (6 * h, h),
                "mlp_out.weight": (h, 3 * h),
                "norm.query_norm.scale": (128,),
                "norm.key_norm.scale": (128,),
            }.items():
                shapes[f"{prefix}.{index}.{name}"] = shape
    if flow_map:
        shapes["ref_time_in.in_layer.weight"] = (h, 256)
        shapes["ref_time_in.out_layer.weight"] = (h, h)
    rng = torch.Generator().manual_seed(42)
    return {
        name: (
            torch.ones(shape)
            if name.endswith(".scale")
            else torch.randn(shape, generator=rng) * 0.02
        ).bfloat16()
        for name, shape in shapes.items()
    }
