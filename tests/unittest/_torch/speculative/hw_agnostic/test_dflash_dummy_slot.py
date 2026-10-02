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
"""DFlash's dummy slot (CUDA-graph padding and warmup dummies) starts every step at context length 0.

Padding rows add their accepted tokens to the dummy slot's length like any request, while their page-table rows are
the padding request's pages with the rest mapped to page 0, which another request owns. A dummy length that kept
growing would put the padding rows' context K / V on that page. ``DFlashSpecMetadata.prepare`` runs before every step
(eager or replayed graph); after it the dummy slot's length is 0 and every real slot's is untouched.
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
        _ctx_len=torch.tensor(lengths, dtype=torch.long, device="cuda"),
        _ctx_len_host=list(lengths),
        _batch_to_slot=torch.zeros(8, dtype=torch.long, device="cuda"),
    )
    w._write_ctx_len = partial(DFlashWorker._write_ctx_len, w)
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


def test_padding_steps_keep_the_dummy_slot_empty():
    worker = _worker([40, 300, 7, 0, 0])
    padded = [11, 12, 13, FLOOR + 1, FLOOR + 1]  # three requests in a graph of five
    for step in range(4):
        # What the step's acceptance does to the slots of its rows (k3_ctx_kv / the torch path): + accepted.
        _prepare(worker, padded, num_generations=5)
        assert worker._ctx_len.tolist() == [40 + 8 * step, 300 + 8 * step, 7 + 8 * step, 0, 0], step
        assert worker._batch_to_slot[:5].tolist() == [0, 1, 2, 4, 4]
        assert worker._ctx_len_host[4] == 0
        worker._ctx_len[[0, 1, 2]] += 8
        worker._ctx_len[4] += 2 * 8  # both padding rows land on the dummy slot


def test_evicted_request_and_dummy_reset_together():
    worker = _worker([40, 300, 7, 0, 96])
    _prepare(worker, [11, 13, FLOOR + 1], num_generations=3)  # request 12 left; one padding row
    assert worker._ctx_len.tolist() == [40, 0, 7, 0, 0]
    assert 1 in worker._free_slots and 12 not in worker._req_to_slot
