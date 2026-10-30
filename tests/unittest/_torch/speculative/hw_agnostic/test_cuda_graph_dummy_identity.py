# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDA_GRAPH_DUMMY_REQUEST_ID,
    _padding_dummy_draft_token_capacity,
    cuda_graph_dummy_request_id,
)
from tensorrt_llm._torch.speculative import dspark as dspark_module
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata, DSv4DSparkWorker


@pytest.mark.parametrize("mode", ["Eagle", "MTP"])
def test_tree_speculation_keeps_native_dummy_identity(mode):
    config = SimpleNamespace(decoding_type=mode, max_draft_len=3)
    capacity = _padding_dummy_draft_token_capacity(config, 12)
    assert capacity == 12
    assert cuda_graph_dummy_request_id(12, variant=0, max_draft_len=capacity) == (
        CUDA_GRAPH_DUMMY_REQUEST_ID - 12
    )


def test_dspark_does_not_relax_the_physical_k_bound():
    config = SimpleNamespace(decoding_type="DSpark", max_draft_len=3)
    capacity = _padding_dummy_draft_token_capacity(config, 12)
    assert capacity == 3
    with pytest.raises(ValueError, match="invalid CUDA padding dummy identity"):
        cuda_graph_dummy_request_id(12, variant=0, max_draft_len=capacity)


def test_secondary_dspark_dummy_ids_are_disjoint():
    native = {cuda_graph_dummy_request_id(k, variant=0, max_draft_len=3) for k in range(4)}
    secondary = {cuda_graph_dummy_request_id(k, variant=1, max_draft_len=3) for k in range(4)}
    assert native.isdisjoint(secondary)


def test_target_only_dummy_identity_is_unchanged():
    capacity = _padding_dummy_draft_token_capacity(None, 0)
    assert cuda_graph_dummy_request_id(0, variant=0, max_draft_len=capacity) == (
        CUDA_GRAPH_DUMMY_REQUEST_ID
    )


@pytest.mark.parametrize("physical_k", [1, 3, 8])
def test_padding_families_never_consume_real_worker_slots(monkeypatch, physical_k):
    def cpu_allocator(original):
        def allocate(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original(*args, **kwargs)

        return allocate

    for name in ("empty", "zeros", "full"):
        monkeypatch.setattr(torch, name, cpu_allocator(getattr(torch, name)))
    monkeypatch.setattr(dspark_module, "prefer_pinned", lambda: False)
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
    worker._lazy_init(draft, SimpleNamespace(max_num_requests=4))
    metadata = SimpleNamespace(
        _dspark_worker=worker,
        batch_indices_cuda=torch.empty(4, dtype=torch.int32),
    )

    for variant in (0, 1):
        for runtime_k in range(physical_k + 1):
            dummy_id = cuda_graph_dummy_request_id(
                runtime_k, variant=variant, max_draft_len=physical_k
            )
            metadata.request_ids = [1000, 0, dummy_id]
            metadata.num_generations = 3
            DSparkSpecMetadata.prepare(metadata)
            assert worker._req_to_slot == {1000: 0}
            assert list(worker._free_slots) == [1, 2, 3]
            assert worker._batch_to_slot[:3].tolist() == [0, 4, 4]

    metadata.request_ids = [0, dummy_id]
    metadata.num_generations = 2
    DSparkSpecMetadata.prepare(metadata)
    assert worker._req_to_slot == {}
    assert sorted(worker._free_slots) == [0, 1, 2, 3]
