# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details.layout import HopperMXScaleLayout, HopperMXValueLayout

from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import (
    convert_layout_expert_chunked,
    update_weight_stride,
)


def _assert_tensors_identical(chunked, reference):
    assert chunked.storage.data.shape == reference.storage.data.shape
    assert chunked.storage.data.stride() == reference.storage.data.stride()
    assert chunked.storage.data.dtype == reference.storage.data.dtype
    assert torch.equal(chunked.storage.data, reference.storage.data)
    assert type(chunked.storage.layout) is type(reference.storage.layout)
    assert chunked.dtype == reference.dtype
    assert list(chunked.shape) == list(reference.shape)
    assert list(chunked.shape_max) == list(reference.shape_max)


# Shapes: (num_experts, in_dim // 2, out_dim). The last two trigger the
# 64-alignment padding inside HopperMXValueLayout.swizzle_data.
@pytest.mark.parametrize("shape", [(8, 64, 128), (5, 128, 64), (3, 72, 96), (8, 40, 200)])
@pytest.mark.parametrize("max_chunk_bytes", [1, 16 * 1024])
def test_chunked_value_swizzle_matches_one_shot(shape, max_chunk_bytes):
    torch.manual_seed(1234)
    w = torch.randint(0, 256, shape, dtype=torch.uint8)
    # Match the (num_experts, K, N) stride layout used at the real call site.
    w = update_weight_stride(w)

    reference = convert_layout(wrap_torch_tensor(w, dtype=FP4), HopperMXValueLayout, mx_axis=1)
    chunked = convert_layout_expert_chunked(
        w, FP4, HopperMXValueLayout, {"mx_axis": 1}, max_chunk_bytes=max_chunk_bytes
    )
    _assert_tensors_identical(chunked, reference)


@pytest.mark.parametrize("shape", [(8, 4, 128), (5, 6, 64), (3, 5, 96)])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("max_chunk_bytes", [1, 4 * 1024])
def test_chunked_scale_swizzle_matches_one_shot(shape, num_warps, max_chunk_bytes):
    torch.manual_seed(1234)
    w_scale = torch.randint(0, 256, shape, dtype=torch.uint8)

    reference = convert_layout(
        wrap_torch_tensor(w_scale), HopperMXScaleLayout, mx_axis=1, num_warps=num_warps
    )
    chunked = convert_layout_expert_chunked(
        w_scale,
        None,
        HopperMXScaleLayout,
        {"mx_axis": 1, "num_warps": num_warps},
        max_chunk_bytes=max_chunk_bytes,
    )
    _assert_tensors_identical(chunked, reference)
