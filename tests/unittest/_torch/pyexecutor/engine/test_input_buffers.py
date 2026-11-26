# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.input_buffers import InputBuffers


@pytest.mark.parametrize("draft_capacity", [None, 0, 8])
def test_runner_input_buffers_support_repeated_cuda_graph_replay(
    draft_capacity: int | None,
) -> None:
    buffers = InputBuffers.allocate(
        max_num_tokens=16,
        max_batch_size=2,
        max_beam_width=2,
        max_seq_len=8,
        use_cache_indirection=True,
        max_num_draft_tokens=draft_capacity,
    )
    assert buffers.cache_indirection.shape == (2, 2, 8)
    if draft_capacity is None:
        assert buffers.gather_ids_cuda is None
        assert buffers.draft_tokens_cuda is None
    else:
        assert buffers.gather_ids_cuda.numel() == 16
        assert buffers.draft_tokens_cuda.numel() == draft_capacity
    input_ptr = buffers.input_ids_cuda.data_ptr()
    buffers.input_ids_cuda.fill_(1)
    buffers.position_ids_cuda.fill_(2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = buffers.input_ids_cuda + buffers.position_ids_cuda
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = buffers.input_ids_cuda + buffers.position_ids_cuda
    output_ptr = output.data_ptr()
    for token in (3, 7):
        buffers.input_ids_cuda.fill_(token)
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, token + 2))
        assert output.data_ptr() == output_ptr
        assert buffers.input_ids_cuda.data_ptr() == input_ptr
    graph.reset()
