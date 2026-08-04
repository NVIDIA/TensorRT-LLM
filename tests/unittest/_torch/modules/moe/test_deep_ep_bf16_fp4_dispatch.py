# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.deep_ep_utils import deep_ep_installed
from tensorrt_llm._utils import get_sm_version

_SM_VERSION = get_sm_version() if torch.cuda.is_available() else 0

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not deep_ep_installed, reason="requires DeepEP"),
    pytest.mark.skipif(_SM_VERSION not in (100, 103), reason="requires SM100/SM103"),
]


def _make_bf16_input(case: str, num_tokens: int, hidden_size: int) -> torch.Tensor:
    if case == "zeros":
        return torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    if case == "finite_extremes":
        finfo = torch.finfo(torch.bfloat16)
        pattern = torch.tensor(
            [finfo.min, -448.0, -6.0, -0.5, 0.0, 0.5, 6.0, 448.0, finfo.max],
            dtype=torch.bfloat16,
            device="cuda",
        )
        num_elements = num_tokens * hidden_size
        repeat_count = (num_elements + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(repeat_count)[:num_elements].view(num_tokens, hidden_size)

    generator = torch.Generator(device="cuda").manual_seed(20260722 + num_tokens)
    return (
        torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        * 32
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("case", ["zeros", "finite_extremes", "random"])
def test_deep_ep_nvfp4_pack_matches_trtllm_bitwise(num_tokens: int, case: str):
    from tensorrt_llm.deep_ep.buffer import Buffer

    hidden_size = 4096
    bf16_input = _make_bf16_input(case, num_tokens, hidden_size)
    static_scale = torch.tensor(0.75, dtype=torch.float32, device="cuda")

    reference_values, reference_scales = torch.ops.trtllm.fp4_quantize(
        bf16_input, static_scale, 16, False, False
    )
    deep_ep_values, deep_ep_scales = Buffer.quantize_bf16_to_nvfp4(
        bf16_input,
        static_scale.expand(num_tokens, 1).contiguous(),
    )

    assert torch.equal(
        reference_values.view(torch.uint8).reshape(-1),
        deep_ep_values.view(torch.uint8).reshape(-1),
    )
    assert torch.equal(
        reference_scales.view(torch.uint8).reshape(-1),
        deep_ep_scales.view(torch.uint8).reshape(-1),
    )
