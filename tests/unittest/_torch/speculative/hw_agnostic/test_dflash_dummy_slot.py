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

Dummy rows add their accepted tokens to the dummy slot's length like any request, while a padding row's page-table row
maps past the padding request's own pages to page 0, which another request owns. A dummy length that kept growing
would put the padding rows' context K / V on that page. The drafting step's ``DFlashWorker._advance_ctx_len`` empties
the slot right after the add, inside the captured step, so ``DFlashSpecMetadata.prepare`` writes slot lengths only for
evicted requests.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the slot lengths live on the GPU"
)

FLOOR = 1 << 40  # request ids from here on are CUDA-graph padding dummies


def _worker(lengths):
    w = SimpleNamespace(
        _ctx_buf_inited=True,
        _req_to_slot={11: 0, 12: 1, 13: 2},
        _req_ctx_pos={},
        _free_slots=[3],
        _graph_dummy_id_floor=FLOOR,
        _dummy_slot=len(lengths) - 1,
        _max_ctx=1000,
        _ctx_len=torch.tensor(lengths, dtype=torch.long, device="cuda"),
        _ctx_len_host=list(lengths),
        _batch_to_slot=torch.zeros(8, dtype=torch.long, device="cuda"),
    )
    w._write_ctx_len = partial(DFlashWorker._write_ctx_len, w)
    w._advance_ctx_len = partial(DFlashWorker._advance_ctx_len, w)
    w._assign_slot = lambda rid, *a, **k: None
    return w


def _prepare(worker, request_ids, num_generations):
    meta = SimpleNamespace(
        request_ids=request_ids,
        num_generations=num_generations,
        batch_indices_cuda=torch.zeros(8, dtype=torch.int, device="cuda"),
        _dflash_worker=worker,
    )
    DFlashSpecMetadata.prepare(meta)
    torch.cuda.synchronize()


def _cuda(values):
    return torch.tensor(values, dtype=torch.long, device="cuda")


def test_padded_steps_keep_the_dummy_slot_empty():
    worker = _worker([40, 300, 7, 0, 0])
    # Three requests in a graph of five.
    _prepare(worker, [11, 12, 13, FLOOR + 1, FLOOR + 1], num_generations=5)
    slots = worker._batch_to_slot[:5].clone()
    assert slots.tolist() == [0, 1, 2, 4, 4]
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


def test_prepare_resets_only_evicted_slots():
    worker = _worker([40, 300, 7, 0, 96])
    written = []
    write = worker._write_ctx_len

    def record(updates):
        written.append(dict(updates))
        write(updates)

    worker._write_ctx_len = record
    _prepare(worker, [11, 12, 13, FLOOR + 1], num_generations=4)
    _prepare(worker, [11, 13, FLOOR + 1], num_generations=3)  # request 12 left
    assert written == [{}, {1: 0}]
    assert worker._ctx_len.tolist() == [40, 0, 7, 0, 96]
    assert 1 in worker._free_slots and 12 not in worker._req_to_slot
