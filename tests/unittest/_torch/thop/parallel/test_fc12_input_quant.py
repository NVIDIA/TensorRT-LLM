# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input quantization/reset equivalence and replay ordering for Rubin FC12."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_FUSED_FC12_AVAILABLE
from tensorrt_llm._utils import get_sm_version


@pytest.mark.skipif(
    not IS_CUTLASS_DSL_FUSED_FC12_AVAILABLE or get_sm_version() != 107,
    reason="Requires the Rubin fused FC12 CuTe DSL build",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
# (7, 224) = 1568 elements: with either 8 or 16 values/lane, the last warp
# is partial, exercising convergent scale-group shuffles with inactive lanes.
@pytest.mark.parametrize(
    "num_tokens,hidden", [(0, 512), (1, 512), (7, 224), (21, 4096), (256, 4096)]
)
@pytest.mark.parametrize("use_pdl,zero_output", [(False, False), (False, True), (True, True)])
def test_fc12_quantize_reset(dtype, num_tokens, hidden, use_pdl, zero_output):
    """Check native bytes, empty input, reset coverage, and changing graph input."""
    from cuda.bindings import driver as cuda

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.moe.fused_fc12_workspace import (
        reset_fc12_sync_workspace,
    )

    torch.manual_seed(87)
    x = torch.randn((num_tokens, hidden), dtype=dtype, device="cuda")
    if num_tokens:
        # Exercise zeros, tiny values, scale boundaries, and FP16/BF16 maxima.
        row = x[0].view(-1, 32)
        for i, value in enumerate((0.0, 2.0**-127, 1.0, 448.0, 449.0, torch.finfo(dtype).max)):
            row[i].fill_(value)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sf = torch.empty((num_tokens, hidden // 32), dtype=torch.uint8, device="cuda")
    ready = torch.empty(777, dtype=torch.int32, device="cuda")
    fc1_counter = torch.empty(1, dtype=torch.int32, device="cuda")
    fc2_counter = torch.empty_like(fc1_counter)
    output = torch.empty_like(x, dtype=torch.bfloat16)

    def ptr(tensor, element_type):
        return make_ptr(element_type, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    args = [
        ptr(ready, cutlass.Int32),
        cutlass.Int32(ready.numel()),
        ptr(fc1_counter, cutlass.Int32),
        ptr(fc2_counter, cutlass.Int32),
        ptr(output, cutlass.BFloat16),
        cutlass.Int32(output.numel()),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    quant_args = dict(
        input_raw_ptr=ptr(x, cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16),
        input_quant_ptr=ptr(q, cutlass.Float8E4M3FN),
        input_scale_ptr=ptr(sf, cutlass.Float8E8M0FNU),
        input_numel=cutlass.Int32(x.numel()),
    )
    compiled = cute.compile(
        reset_fc12_sync_workspace, *args, use_pdl=use_pdl, zero_output=zero_output, **quant_args
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    args[-1] = cuda.CUstream(stream.cuda_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compiled(*args, **quant_args)
    for iteration in range(3):
        if iteration == 1:
            x.zero_()
        elif iteration == 2:
            x.fill_(-0.375)
        q.view(torch.uint8).fill_(255)
        sf.fill_(255)
        ready.fill_(19)
        fc1_counter.fill_(23)
        fc2_counter.fill_(29)
        output.fill_(7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(ready).item() == 0
        assert fc1_counter.item() == fc2_counter.item() == 0
        torch.testing.assert_close(output, torch.full_like(output, 0 if zero_output else 7))
        if num_tokens:
            # alignment=32 keeps the reference K unpadded for hidden sizes below 512.
            expected, expected_sf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=32)
            torch.testing.assert_close(
                q.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
            )
            torch.testing.assert_close(
                sf, expected_sf.view(torch.uint8).reshape_as(sf), atol=0, rtol=0
            )


@pytest.mark.skipif(
    not IS_CUTLASS_DSL_FUSED_FC12_AVAILABLE or get_sm_version() != 107,
    reason="Requires the Rubin fused FC12 CuTe DSL build",
)
@pytest.mark.parametrize("num_ready,num_output", [(0, 0), (777, 512), (8193, 65536)])
@pytest.mark.parametrize("use_pdl,zero_output", [(False, False), (False, True), (True, True)])
def test_fc12_reset_only_replay(num_ready, num_output, use_pdl, zero_output):
    """Reset-only calls preserve all buffers and graph ordering without raw input."""
    from cuda.bindings import driver as cuda

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.moe.fused_fc12_workspace import (
        reset_fc12_sync_workspace,
    )

    def ptr(tensor, element_type):
        return make_ptr(element_type, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    ready = torch.empty(num_ready, dtype=torch.int32, device="cuda")
    fc1_counter = torch.empty(1, dtype=torch.int32, device="cuda")
    fc2_counter = torch.empty_like(fc1_counter)
    output = torch.empty(num_output, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    args = [
        ptr(ready, cutlass.Int32),
        cutlass.Int32(num_ready),
        ptr(fc1_counter, cutlass.Int32),
        ptr(fc2_counter, cutlass.Int32),
        ptr(output, cutlass.BFloat16),
        cutlass.Int32(num_output),
        cuda.CUstream(stream.cuda_stream),
    ]
    compiled = cute.compile(
        reset_fc12_sync_workspace, *args, use_pdl=use_pdl, zero_output=zero_output
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compiled(*args)
    for value in (3, 9, -5):
        ready.fill_(value)
        fc1_counter.fill_(value)
        fc2_counter.fill_(value)
        output.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(ready).item() == 0
        assert fc1_counter.item() == fc2_counter.item() == 0
        torch.testing.assert_close(
            output, torch.full_like(output, 0 if zero_output else value), atol=0, rtol=0
        )
