# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for the embedded DSpark worker's scratch position."""

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker


def _make_worker(device: str) -> DSv4DSparkWorker:
    # Only the per-slot tensors are needed; avoid model and CUDA initialization.
    worker = object.__new__(DSv4DSparkWorker)
    worker._scratch_slot = 128
    worker._win = 8
    worker._ctx_len = torch.zeros(129, dtype=torch.long, device=device)
    worker._valid_len = torch.zeros(129, dtype=torch.long, device=device)
    worker._position_initialized = torch.zeros(129, dtype=torch.bool, device=device)
    return worker


@pytest.mark.parametrize("initialized", [False, True])
def test_scratch_position_stays_inside_rope_table(initialized: bool) -> None:
    worker = _make_worker("cpu")
    scratch = worker._scratch_slot
    # max_position_embeddings + block_size + 2 for a block-size-one drafter.
    table = torch.zeros(1048579, 1)
    worker._ctx_len[scratch] = table.shape[0] - 2
    worker._position_initialized[scratch] = initialized
    slots = torch.tensor([127, scratch, scratch, scratch])
    nacc = torch.tensor([2, 1, 2, 1])
    positions = torch.tensor([32000, table.shape[0], table.shape[0], table.shape[0]])

    for step in range(64):
        old, start = worker._advance_generation_state(slots, nacc, positions)
        # Draft attention and masked backfill gather before masking dummy rows.
        table[start]
        table[start + 1]
        table[old + 1]
        assert old.tolist() == [32000 + 2 * step, 0, 0, 0]
        assert start.tolist() == [32002 + 2 * step, 1, 2, 1]
        assert int(worker._ctx_len[scratch]) <= 2
        assert int(worker._valid_len[scratch]) <= worker._win
        positions[0] = 7  # An initialized real slot must ignore input positions.

    # A recycled real slot must bootstrap from the new request's position.
    worker._ctx_len[127] = 0
    worker._valid_len[127] = 0
    worker._position_initialized[127] = False
    positions[0] = 500
    old, start = worker._advance_generation_state(slots, nacc, positions)
    assert old.tolist() == [500, 0, 0, 0]
    assert start.tolist() == [502, 1, 2, 1]
    assert int(worker._valid_len[127]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph capture")
@pytest.mark.parametrize("batch", [16, 32, 48, 64, 96, 128])
def test_scratch_position_resets_on_graph_replay(batch: int) -> None:
    worker = _make_worker("cuda")
    scratch = worker._scratch_slot
    slots = torch.full((batch,), scratch, dtype=torch.long, device="cuda")
    nacc = torch.ones(batch, dtype=torch.long, device="cuda")
    positions = torch.full((batch,), 32000, dtype=torch.long, device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            worker._advance_generation_state(slots, nacc, positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        old, start = worker._advance_generation_state(slots, nacc, positions)

    for iteration in range(32):
        mapping = list(range(128 - batch // 2, 128)) + [scratch] * (batch // 2)
        if iteration % 2:
            mapping.reverse()
        slots.copy_(torch.tensor(mapping, device="cuda"))
        nacc.fill_(1 + iteration % 2)
        worker._ctx_len[:128] = 1000
        worker._position_initialized[:128] = iteration % 2 == 0
        worker._ctx_len[scratch] = 1048577
        worker._position_initialized[scratch] = True
        graph.replay()
        expected = [
            0 if slot == scratch else (1000 if iteration % 2 == 0 else 32000) for slot in mapping
        ]
        assert old.tolist() == expected
        assert start.tolist() == [p + 1 + iteration % 2 for p in expected]

    slots.fill_(scratch)
    for _ in range(1000):
        graph.replay()
    assert int(worker._ctx_len[scratch]) == 2
