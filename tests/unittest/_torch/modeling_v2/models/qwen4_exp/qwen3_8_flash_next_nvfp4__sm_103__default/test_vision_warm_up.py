# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests of the default target's vision warm-up and concurrent execution.

After `VisionTower.warm_up`, which `post_load_weights` runs, a media request of any grid
compiles no Triton kernel. Concurrent forwards use independent attention scratch.

The tower is one block deep (the block's kernels are every block's) over random weights at the
checkpoint's shapes, built from the weight table's own `vis_*` declarations.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    vision as V,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.attention.vision_rope_2d_packed_qk import (  # noqa: E501
    _rotary_neox_inplace_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.vision.triton_pos_embed_interpolate import (  # noqa: E501
    _bilinear_pos_embed_kernel,
)
from tensorrt_llm._torch.configs import Qwen4ExpVisionConfig
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
    d = SimpleNamespace(
        hidden=HIDDEN_OUT,
        dtype=torch.bfloat16,
        v_depth=vcfg.depth,
        v_hidden=vcfg.hidden_size,
        v_inter=vcfg.intermediate_size,
        v_channels=vcfg.in_channels,
        v_tpatch=vcfg.temporal_patch_size,
        v_patch=vcfg.patch_size,
        v_pos_rows=vcfg.num_position_embeddings,
        v_merge_dim=vcfg.hidden_size * vcfg.spatial_merge_size**2,
    )
    gen = torch.Generator(device="cuda").manual_seed(7)
    w = {"embed": torch.empty(8, HIDDEN_OUT, dtype=torch.bfloat16, device="cuda")}
    for entry in W.MODEL_WEIGHTS.WEIGHTS:
        if not entry.name.startswith("vis_"):
            continue
        for key, _, _ in W.MODEL_WEIGHTS._targets(entry, d):
            shape, dtype = entry.shape(d), entry.dtype or d.dtype
            w[key] = (torch.randn(*shape, generator=gen, device="cuda") * 0.02).to(dtype)
    return V.VisionTower(w, vcfg)


def _variants() -> dict:
    """Compiled variants per kernel on this device (through any `heuristics` wrappers)."""
    device = torch.cuda.current_device()
    counts = {}
    for name, kernel in _KERNELS.items():
        while not hasattr(kernel, "device_caches"):
            kernel = kernel.fn
        counts[name] = len(kernel.device_caches[device][0])
    return counts


def test_attention_workspace_is_owned_by_each_cuda_stream():
    tower = _tower()
    device = torch.device("cuda", torch.cuda.current_device())
    first = torch.cuda.Stream(device=device)
    second = torch.cuda.Stream(device=device)
    with torch.cuda.stream(first):
        first_workspace = tower._attention_workspace(device)
        first_workspace.resize_(64).fill_(1)
        assert tower._attention_workspace(device) is first_workspace
    with torch.cuda.stream(second):
        second_workspace = tower._attention_workspace(device)
        second_workspace.resize_(64).fill_(2)
        assert tower._attention_workspace(device) is second_workspace
    first.synchronize()
    second.synchronize()
    assert first_workspace is not second_workspace
    assert torch.equal(first_workspace.cpu(), torch.ones(64, dtype=torch.int8))
    assert torch.equal(second_workspace.cpu(), torch.full((64,), 2, dtype=torch.int8))


def test_concurrent_vision_calls_match_serial_outputs():
    tower = _tower()
    grids = [torch.tensor([[1, 4, 4]]), torch.tensor([[1, 8, 4]])]
    pixels = [
        torch.randn(rows, tower.pixel_width, dtype=tower.dtype, device="cuda") for rows in (16, 32)
    ]
    expected = [tower(values, grid) for values, grid in zip(pixels, grids, strict=True)]
    producer = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    actual = []
    for stream, values, grid in zip(streams, pixels, grids, strict=True):
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            actual.append(tower(values, grid))
    for stream in streams:
        stream.synchronize()
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-4)


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
