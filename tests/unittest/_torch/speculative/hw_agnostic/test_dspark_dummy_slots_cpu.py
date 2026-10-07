# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host slot-ownership regression with CPU tensors in place of CUDA allocations."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.speculative.dspark as dspark
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import cuda_graph_dummy_request_id
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata, DSv4DSparkWorker


@pytest.mark.parametrize("physical_k", range(1, 9))
def test_all_padding_families_preserve_real_slot_ownership(monkeypatch, physical_k):
    # Exercise the real initialization and host prepare methods. This deliberately
    # does not validate device copies, graph replay, or target KV-cache contents.
    original_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(dspark, "prefer_pinned", lambda: False)
    worker = object.__new__(DSv4DSparkWorker)
    worker.spec_config = SimpleNamespace(max_draft_len=physical_k)
    worker._win_inited = False
    worker._attention_warmup_attempted = True
    worker.return_confidence = False
    worker._confidence_logits = None
    draft = SimpleNamespace(
        block_size=physical_k,
        num_stages=1,
        _attn_params={"window_size": 4, "head_dim": 2},
    )
    DSv4DSparkWorker._lazy_init(worker, draft, SimpleNamespace(max_num_requests=4))
    meta = SimpleNamespace(
        _dspark_worker=worker, batch_indices_cuda=torch.empty(4, dtype=torch.int32)
    )
    for variant in (0, 1):
        for runtime_k in range(physical_k + 1):
            dummy = cuda_graph_dummy_request_id(
                runtime_k, variant=variant, max_draft_len=physical_k
            )
            # Include ADP idle, the secondary zero-real family, and a live row.
            meta.request_ids = [1000, 0, dummy]
            meta.num_generations = 3
            DSparkSpecMetadata.prepare(meta)
            assert worker._req_to_slot == {1000: 0}
            assert list(worker._free_slots) == [1, 2, 3]
            assert worker._batch_to_slot[:3].tolist() == [0, 4, 4]
    # Removing a real request releases exactly its slot, never a dummy's.
    meta.request_ids = [dummy, 0]
    meta.num_generations = 2
    DSparkSpecMetadata.prepare(meta)
    assert worker._req_to_slot == {}
    assert sorted(worker._free_slots) == [0, 1, 2, 3]
