# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import mxfp8_quantize
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.mxfp8__sm_103__default.weights import (
    _load_projection,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="interleaving requires a CUDA device")
@pytest.mark.parametrize("expert_bank", [False, True], ids=["segmented_rows", "sharded_expert_k"])
def test_projection_windows_select_corresponding_scale_blocks(expert_bank):
    name = "projection"
    if expert_bank:
        shape = (4, 7, 192)
        # Physical slots can hold non-contiguous logical experts and replicas.
        experts = [3, 0, 3]
        bank = (len(experts),)
        entries = [(name, (experts, slice(None), slice(32, 128)), None, None)]
        selected = (experts, slice(None), slice(32, 128))
        scale_selected = (experts, slice(None), slice(1, 4))
    else:
        shape = (15, 96)
        bank = ()
        entries = [
            (name, (slice(2, 5),), (slice(0, 3),), None),
            (name, (slice(9, 13),), (slice(3, 7),), None),
        ]
        selected = scale_selected = [2, 3, 4, 9, 10, 11, 12]
    weight = torch.arange(math.prod(shape)).reshape(shape)
    packed = (weight % 113).to(torch.float8_e4m3fn)
    blocks = (weight[..., ::32] // 32 % 251 + 1).to(torch.uint8)
    checkpoint = {name: packed, name + "_scale": blocks}
    target = torch.empty((*bank, 128, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((*bank, 512), device="cuda", dtype=torch.uint8)
    _load_projection(checkpoint, entries, target, scales, (7, 96))
    values = target.view(torch.uint8).cpu()
    expected = torch.zeros_like(values)
    expected[..., :7, :96] = packed.view(torch.uint8)[selected]
    torch.testing.assert_close(values, expected)
    expected_blocks = blocks[scale_selected].reshape(-1, 7, 3)
    expected_scales = torch.stack([mxfp8_quantize._swizzle(b) for b in expected_blocks])
    torch.testing.assert_close(scales.cpu().reshape(-1, 512), expected_scales)
