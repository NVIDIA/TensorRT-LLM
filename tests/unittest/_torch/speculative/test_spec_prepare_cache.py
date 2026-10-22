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
"""TRTLLM_Q35_SPEC_PREPARE_CACHE: Eagle3OneModelSpecMetadata.prepare() skips redundant batch-index / slot-id
uploads with the same buffer contents; changed slots, a larger batch or a new buffer upload again."""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative import eagle3
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelSpecMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Slots:
    def __init__(self, slots):
        self.slots = dict(slots)

    def get_slot(self, rid):
        return self.slots[rid]


def _metadata(slot_manager, batch_indices=None):
    md = object.__new__(Eagle3OneModelSpecMetadata)
    md.use_rejection_sampling = False
    md.batch_indices_cuda = (
        batch_indices
        if batch_indices is not None
        else torch.full((8,), -1, dtype=torch.int, device="cuda")
    )
    md.slot_ids = torch.full((8,), -1, dtype=torch.long, device="cuda")
    md.spec_resource_manager = types.SimpleNamespace(slot_manager=slot_manager)
    md.spec_dec_mode = types.SimpleNamespace(is_mtp_eagle_one_model=lambda: True)
    md.use_dynamic_tree = False
    md.num_tokens = 0
    md.num_generations = 0
    return md


def _uploads(monkeypatch):
    """Count the pinned staging tensors prepare() builds (one per upload)."""
    calls = []
    real_arange, real_tensor = torch.arange, torch.tensor

    def arange(*args, **kwargs):
        if kwargs.get("pin_memory"):
            calls.append("arange")
        return real_arange(*args, **kwargs)

    def tensor(*args, **kwargs):
        if kwargs.get("pin_memory"):
            calls.append("tensor")
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(eagle3.torch, "arange", arange)
    monkeypatch.setattr(eagle3.torch, "tensor", tensor)
    return calls


@pytest.mark.parametrize("cache", [False, True])
def test_same_buffers_fewer_uploads(monkeypatch, cache):
    if cache:
        monkeypatch.setenv(eagle3.SPEC_PREPARE_CACHE_ENV, "1")
    else:
        monkeypatch.delenv(eagle3.SPEC_PREPARE_CACHE_ENV, raising=False)
    slots = _Slots({5: 2, 6: 0, 7: 3})
    md = _metadata(slots)
    calls = _uploads(monkeypatch)
    steps = [[5, 6], [5, 6], [5, 6], [5], [5, 6, 7], [5, 6, 7]]
    for request_ids in steps:
        md.request_ids = request_ids
        md.prepare()
        torch.cuda.synchronize()
        n = len(request_ids)
        assert md.batch_indices_cuda[:n].tolist() == list(range(n))
        assert md.slot_ids[:n].tolist() == [slots.slots[r] for r in request_ids]
    if not cache:
        assert len(calls) == 2 * len(steps)
    else:
        # arange: n=2 once, n=3 once (n=1 is covered by the n=2 prefix); slot ids: [2,0], [2], [2,0,3]
        assert calls.count("arange") == 2
        assert calls.count("tensor") == 3


def test_reassigned_slot_and_shared_index_buffer(monkeypatch):
    monkeypatch.setenv(eagle3.SPEC_PREPARE_CACHE_ENV, "1")
    slots = _Slots({5: 2, 6: 0})
    shared = torch.full((8,), -1, dtype=torch.int, device="cuda")
    md, graph_md = _metadata(slots, shared), _metadata(slots, shared)
    calls = _uploads(monkeypatch)
    md.request_ids = [5, 6]
    md.prepare()
    graph_md.request_ids = [5, 6]
    graph_md.prepare()  # same shared index buffer: no index upload; its own slot buffer: one slot upload
    assert calls == ["arange", "tensor", "tensor"]
    slots.slots[6] = 4  # a request moved to another slot -> upload
    md.prepare()
    torch.cuda.synchronize()
    assert md.slot_ids[:2].tolist() == [2, 4]
    assert calls[-1] == "tensor" and len(calls) == 4


def test_new_buffer_is_not_an_old_one(monkeypatch):
    monkeypatch.setenv(eagle3.SPEC_PREPARE_CACHE_ENV, "1")
    slots = _Slots({5: 2})
    md = _metadata(slots)
    md.request_ids = [5]
    md.prepare()
    md.batch_indices_cuda = torch.full((8,), -1, dtype=torch.int, device="cuda")
    md.slot_ids = torch.full((8,), -1, dtype=torch.long, device="cuda")
    md.prepare()
    torch.cuda.synchronize()
    assert md.batch_indices_cuda[:1].tolist() == [0]
    assert md.slot_ids[:1].tolist() == [2]
