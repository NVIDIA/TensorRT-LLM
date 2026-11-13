# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrefillPadRows on the GPU: a captured graph reads each replay's mask, written by the eager writer before it.

The writer runs in the model's outer forward (never captured) on the current stream, and the breakable graph replays
its segments on the same stream, so every replay reads the mask of its own step. Here a captured masked_fill stands in
for the route-stage kernel (ids -1 on masked rows).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
    prefill_pad_rows as ppr,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("bucket", [5120, 12287, 16384])
def test_each_replay_reads_its_own_steps_mask(bucket):
    state = ppr.PrefillPadRows(16384, in_graph=lambda: True)
    state.allocate("cuda")
    ids = torch.zeros((bucket, 4), dtype=torch.int32, device="cuda")
    state.begin(bucket, bucket, padded=False, graphed=False)  # the capture arms the rows only
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Warm-up outside the capture (torch allocates lazily), then the capture.
        mask = state.mask_for(bucket, capturing=True)
        ids.fill_(7).masked_fill_(mask.bool().unsqueeze(1), -1)
        with torch.cuda.graph(graph, stream=stream):
            ids.fill_(7).masked_fill_(mask.bool().unsqueeze(1), -1)
    torch.cuda.current_stream().wait_stream(stream)
    state.end()
    for live in (bucket, 0, bucket // 3, 4097 if bucket > 4097 else 1):
        state.begin(live, bucket, padded=live != bucket, graphed=True)
        graph.replay()
        state.end()
        want = torch.where(torch.arange(bucket, device="cuda") < live, 7, -1).to(torch.int32)
        assert torch.equal(ids, want.unsqueeze(1).expand(bucket, 4)), live


def test_the_writer_is_one_launch_and_the_buffer_never_moves():
    state = ppr.PrefillPadRows(16384)
    state.allocate("cuda")
    ptr = state._mask.data_ptr()
    state.begin(9000, 16384, padded=True, graphed=True)
    torch.cuda.synchronize()
    assert int(state._mask[:16384].sum()) == 16384 - 9000
    state.allocate("cuda")
    assert state._mask.data_ptr() == ptr and state.writes == 1
