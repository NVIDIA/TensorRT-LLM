# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.attention.cute_fmha import cute_fmha


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="CuTe FMHA needs SM100, SM103 or SM107",
)
@pytest.mark.parametrize("cell", cute_fmha.CELLS, ids=lambda cell: cell.why)
def test_cute_fmha(cell):
    spec = cell.spec
    rng = torch.Generator(device="cuda").manual_seed(371)

    q, k, v = [
        torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16, generator=rng) * 0.5
        for _ in range(3)
    ]
    kwargs = dict(
        sm_scale=0.078125,
        qk_dtype=spec["qk_dtype"],
        v_block_size=spec.get("v_block_size"),
    )
    cute_fmha.is_valid(q, k, v, **kwargs)
    out, lse = cute_fmha(q, k, v, **kwargs)
    expected_out, expected_lse = cute_fmha.reference(q, k, v, **kwargs)
    # Blockscaled Q/K and FP8 softmax probabilities round before the output reduction.
    atol = {None: 0.005, "bf16": 0.015, "fp8": 0.015, "mxfp8": 0.015, "nvfp4": 0.035}[
        spec["qk_dtype"]
    ]
    torch.testing.assert_close(out, expected_out, rtol=0.01, atol=atol)
    torch.testing.assert_close(lse, expected_lse, rtol=0.001, atol=atol)


@pytest.mark.cpu_only
def test_kv_head_count_matches_queries():
    q = torch.empty(1, 8, 2, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 8, 1, 128, dtype=q.dtype)
    with pytest.raises(AssertionError):
        cute_fmha.is_valid(q, k, k, sm_scale=0.125)
