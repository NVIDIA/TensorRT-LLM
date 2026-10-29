# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU regression coverage for the HELIX NCCL input layout."""

import pytest
import torch
import torch._inductor.config as inductor_config

from tensorrt_llm._torch.modules.attention import _helix_nccl_pre_alltoall


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("num_tokens", [512, 1008, 1016, 1020, 1024])
@pytest.mark.parametrize("cp_size", [8, 16])
@pytest.mark.parametrize("mask_kind", ["none", "empty", "mixed", "all"])
def test_helix_nccl_chunks_are_contiguous(num_tokens: int, cp_size: int, mask_kind: str) -> None:
    # Isolate specializations so a cache-limit eager fallback cannot hide the bug.
    torch._dynamo.reset()
    try:
        with torch.no_grad(), inductor_config.patch(comprehensive_padding=True):
            partial_o = torch.randn(num_tokens, 96 * 512, device="cuda", dtype=torch.bfloat16)
            stats = torch.randn(num_tokens, 96, 2, device="cuda", dtype=torch.float32)
            mask = None
            if mask_kind != "none":
                mask = torch.zeros(num_tokens, device="cuda", dtype=torch.bool)
                if mask_kind == "mixed":
                    mask[::3] = True
                elif mask_kind == "all":
                    mask.fill_(True)
                # Empty-KV attention can produce NaNs; those rows must be sanitized.
                partial_o[mask] = float("nan")
                stats[mask] = float("nan")

            expected_o = partial_o.clone()
            expected_stats = stats.clone()
            if mask is not None:
                expected_o[mask] = 0
                expected_stats[mask, :, 0] = float("-inf")
                expected_stats[mask, :, 1] = 0
            expected = []
            for tensor in (expected_o, expected_stats):
                transposed = tensor.transpose(0, 1).contiguous()
                expected.extend(transposed.chunk(cp_size))

            # Compile outside capture, on a side stream as required by CUDA graphs.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _helix_nccl_pre_alltoall(partial_o, stats, mask, cp_size)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = _helix_nccl_pre_alltoall(partial_o, stats, mask, cp_size)
            graph.replay()
            torch.cuda.synchronize()

            assert len(actual) == 2 * cp_size
            for output, reference in zip(actual, expected):
                assert output.is_contiguous(), (output.shape, output.stride())
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
