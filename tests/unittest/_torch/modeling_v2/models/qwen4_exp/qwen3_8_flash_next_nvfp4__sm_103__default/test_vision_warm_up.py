# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the default target's vision tower warm-up: after `VisionTower.warm_up`, which
`post_load_weights` runs, a media request of any grid compiles no Triton kernel.

The tower is one block deep (the block's kernels are every block's) over random weights at the
checkpoint's shapes, declared by the tower itself (`weights.VISION_WEIGHTS`).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.configs import Qwen4ExpVisionConfig
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    vision as V,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.attention.vision_rope_2d_packed_qk import (  # noqa: E501
    _rotary_neox_inplace_kernel,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.vision.triton_pos_embed_interpolate import (  # noqa: E501
    _bilinear_pos_embed_kernel,
)
from tensorrt_llm._torch.modules.fla.layernorm_gated import _layer_norm_fwd_1pass_kernel

assert torch.cuda.is_available(), "the vision tower runs on a CUDA device"

# Qwen3.8-Flash-Next's vision config, one block deep.
_VISION = dict(
    deepstack_visual_indexes=[],
    depth=1,
    dtype="bfloat16",
    hidden_act="gelu_pytorch_tanh",
    hidden_size=1152,
    in_channels=3,
    intermediate_size=4304,
    num_heads=16,
    num_position_embeddings=2304,
    out_hidden_size=2560,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
)
HIDDEN_OUT = 2560

#: Every Triton kernel the tower launches.
_KERNELS = {
    "rope": _rotary_neox_inplace_kernel,
    "pos_embed": _bilinear_pos_embed_kernel,
    "layer_norm": _layer_norm_fwd_1pass_kernel,
}


def _tower() -> V.VisionTower:
    vcfg = Qwen4ExpVisionConfig(**_VISION)
    with torch.device("cuda"):
        tower = V.VisionTower(vcfg, torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(7)
    for param in tower.w.values():
        param.data.copy_(torch.randn(param.shape, generator=gen, device="cuda") * 0.02)
    tower._w = V._vision_views(tower.w, tower.depth)
    return tower


def _variants() -> dict:
    """Compiled variants per kernel on this device (through any `heuristics` wrappers)."""
    device = torch.cuda.current_device()
    counts = {}
    for name, kernel in _KERNELS.items():
        while not hasattr(kernel, "device_caches"):
            kernel = kernel.fn
        counts[name] = len(kernel.device_caches[device][0])
    return counts


def test_after_warm_up_no_grid_compiles_a_kernel():
    tower = _tower()
    tower.warm_up()
    torch.cuda.synchronize()
    warmed = _variants()
    assert all(warmed.values()), f"the warm-up compiled nothing: {warmed}"
    gen = torch.Generator(device="cuda").manual_seed(11)
    for grid in ([(1, 16, 16)], [(1, 6, 10)], [(2, 8, 12)], [(1, 62, 90)], [(1, 2, 34), (1, 4, 4)]):
        rows = sum(t * h * w for t, h, w in grid)
        pixels = torch.randn(rows, tower.pixel_width, generator=gen, device="cuda").to(tower.dtype)
        out = tower(pixels, torch.tensor(grid, dtype=torch.int64))
        torch.cuda.synchronize()
        assert out.shape == (rows // 4, HIDDEN_OUT) and bool(torch.isfinite(out).all()), grid
        assert _variants() == warmed, f"{grid} compiled a kernel after the warm-up"
