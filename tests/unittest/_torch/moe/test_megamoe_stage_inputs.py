# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-launch MegaMoE input staging against per-tensor copies and tail fills."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE,
    reason="CUDA and the CuTe DSL are required",
)

# DeepSeek-V4.1-Flash: top-6, hidden 5120 (2560 NVFP4 bytes, 320 scale bytes).
MAX_T, TOP_K, ACT_BYTES, SF_BYTES = 64, 6, 2560, 320


def _buffers(idx_dtype):
    torch.manual_seed(1)
    return {
        "idx_buffer": torch.randint(-5, 400, (MAX_T, TOP_K), device="cuda").to(idx_dtype),
        "weights_buffer": torch.randn(MAX_T, TOP_K, device="cuda"),
        "act_buffer": torch.randint(0, 256, (MAX_T, ACT_BYTES), device="cuda", dtype=torch.uint8),
        "sf_buffer": torch.randint(0, 256, (MAX_T, SF_BYTES), device="cuda", dtype=torch.uint8),
    }


def _inputs(num_tokens):
    return (
        torch.randint(0, 384, (num_tokens, TOP_K), device="cuda", dtype=torch.int32),
        torch.rand(num_tokens, TOP_K, device="cuda"),
        torch.randint(0, 256, (num_tokens, ACT_BYTES), device="cuda", dtype=torch.uint8),
        torch.randint(0, 256, (num_tokens, SF_BYTES), device="cuda", dtype=torch.uint8),
    )


def _reference(buffers, idx, weights, x, x_sf, stage_activation):
    out = {name: value.clone() for name, value in buffers.items()}
    n = idx.shape[0]
    out["idx_buffer"][:n] = idx.to(out["idx_buffer"].dtype)
    out["weights_buffer"][:n] = weights
    if stage_activation:
        out["act_buffer"][:n] = x
        out["sf_buffer"][:n] = x_sf
    out["idx_buffer"][n:] = -1
    out["weights_buffer"][n:] = 0
    return out


@pytest.mark.parametrize("num_tokens", [0, 1, 48, MAX_T])
@pytest.mark.parametrize("stage_activation", [True, False])
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
@torch.inference_mode()
def test_stage_inputs_matches_per_tensor_staging(num_tokens, stage_activation, idx_dtype):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.megamoe_stage_inputs import (
        stage_megamoe_inputs,
    )

    buffers = _buffers(idx_dtype)
    idx, weights, x, x_sf = _inputs(num_tokens)
    expected = _reference(buffers, idx, weights, x, x_sf, stage_activation)
    if stage_activation:
        stage_megamoe_inputs(idx, weights, x=x, x_sf=x_sf, **buffers)
    else:
        stage_megamoe_inputs(idx, weights, buffers["idx_buffer"], buffers["weights_buffer"])
    for name, value in expected.items():
        torch.testing.assert_close(buffers[name], value, atol=0, rtol=0)


@torch.inference_mode()
def test_stage_inputs_replays_in_a_cuda_graph():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.megamoe_stage_inputs import (
        stage_megamoe_inputs,
    )

    buffers = _buffers(torch.int64)
    idx, weights, x, x_sf = _inputs(48)
    stage_megamoe_inputs(idx, weights, x=x, x_sf=x_sf, **buffers)  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage_megamoe_inputs(idx, weights, x=x, x_sf=x_sf, **buffers)
    idx.random_(0, 384)
    x.random_(0, 256)
    buffers["idx_buffer"].fill_(7)
    buffers["weights_buffer"].fill_(3.0)
    expected = _reference(buffers, idx, weights, x, x_sf, True)
    graph.replay()
    torch.cuda.synchronize()
    for name, value in expected.items():
        torch.testing.assert_close(buffers[name], value, atol=0, rtol=0)
