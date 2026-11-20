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
    if "skip_softmax_threshold_scale_factor" in spec:
        _check_skipped_tile(dict(spec))
        return
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


def _check_skipped_tile(kwargs):
    q, k, v = [torch.zeros(1, 384, 1, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q[..., 0] = 1
    k[:, 128:256, :, 0], k[:, 256:, :, 0] = -4, -5
    v[:, 128:256], v[:, 256:] = 4, 32
    cute_fmha.is_valid(q, k, v, sm_scale=1.0, **kwargs)
    out, lse = cute_fmha(q, k, v, sm_scale=1.0, **kwargs)

    # Every query has tile weights [1, exp(-4), exp(-5)]. Only exp(-5) < 4/384.
    expected_out, expected_lse = cute_fmha.reference(
        q[:, :256], k[:, :256], v[:, :256], sm_scale=1.0
    )
    torch.testing.assert_close(out, expected_out[:, :1].expand_as(q), rtol=0, atol=1e-3)
    torch.testing.assert_close(lse, expected_lse[:, :, :1].expand(1, 1, 384), rtol=0, atol=1e-3)


@pytest.mark.cpu_only
def test_kv_head_count_matches_queries():
    q = torch.empty(1, 8, 2, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 8, 1, 128, dtype=q.dtype)
    with pytest.raises(AssertionError):
        cute_fmha.is_valid(q, k, k, sm_scale=0.125)
