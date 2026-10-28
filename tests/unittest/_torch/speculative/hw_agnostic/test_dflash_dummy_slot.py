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
"""DFlash's dummy slot (CUDA-graph padding, warmup and attention-DP dummies) is at context length 0 between steps.

Adapted from main's test of the same name (#19988). A dummy length that kept growing would put the padding rows'
context K / V past the padding request's own pages, and on a unified KV cache trip the drafter's allocation check.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the slot lengths live on the GPU"
)


def _worker(lengths, bases=None):
    w = SimpleNamespace(
        _dummy_slot=len(lengths) - 1,
        _max_ctx=1000,
        _ctx_len=torch.tensor(lengths, dtype=torch.long, device="cuda"),
        _ctx_base=None if bases is None else torch.tensor(bases, dtype=torch.long, device="cuda"),
    )
    w._advance_ctx_len = partial(DFlashWorker._advance_ctx_len, w)
    w._empty_dummy_slot = partial(DFlashWorker._empty_dummy_slot, w)
    return w


def _cuda(values):
    return torch.tensor(values, dtype=torch.long, device="cuda")


def test_padded_steps_keep_the_dummy_slot_empty():
    worker = _worker([40, 300, 7, 0, 0])
    slots = _cuda([0, 1, 2, 4, 4])  # three requests in a graph of five
    for step in range(1, 5):
        worker._advance_ctx_len(slots, _cuda([8] * 5))
        torch.cuda.synchronize()
        assert worker._ctx_len.tolist() == [40 + 8 * step, 300 + 8 * step, 7 + 8 * step, 0, 0], step


def test_dummy_slot_reset_replays_in_a_cuda_graph():
    worker = _worker([40, 300, 7, 0, 0])
    slots = _cuda([0, 1, 4])
    accepted = _cuda([2, 3, 5])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        worker._advance_ctx_len(slots, accepted)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        worker._advance_ctx_len(slots, accepted)
    worker._ctx_len.copy_(_cuda([40, 300, 7, 0, 0]))
    for step in range(1, 4):
        worker._ctx_len[worker._dummy_slot] = 99  # whatever the slot holds before the step
        graph.replay()
        torch.cuda.synchronize()
        assert worker._ctx_len.tolist() == [40 + 2 * step, 300 + 3 * step, 7, 0, 0], step


def test_windowed_arena_dummy_slot_is_empty_with_its_base():
    worker = _worker([40, 300, 0], bases=[8, 100, 64])
    worker._advance_ctx_len(_cuda([0, 2, 2]), _cuda([4, 4, 4]))
    torch.cuda.synchronize()
    assert worker._ctx_len.tolist() == [44, 300, 0]
    assert worker._ctx_base.tolist() == [8, 100, 0]


def test_warmup_reseeds_generation_slots_from_this_batch():
    # Slot 0 still holds an earlier warmup batch's length for a reused id; the
    # padding row (no slot of its own) is left alone.
    worker = _worker([500, 7, 0, 0])
    worker._req_to_slot = {11: 0, 12: 1}
    worker._ctx_len_host = [500, 7, 0, 0]
    worker._write_ctx_len = partial(DFlashWorker._write_ctx_len, worker)
    meta = SimpleNamespace(
        request_ids=[11, 12, 1 << 62],
        num_generations=3,
        seq_lens=[8, 8, 8],
        host_position_ids=[20] * 8 + [30] * 8 + [0] * 8,
    )

    DFlashWorker._reseed_warmup_generation_slots(worker, meta)
    torch.cuda.synchronize()

    assert worker._ctx_len.tolist() == [20, 30, 0, 0]
    assert worker._ctx_len_host == [20, 30, 0, 0]
