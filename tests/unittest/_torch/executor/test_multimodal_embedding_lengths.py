# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (
    PyResult,
    get_request_multimodal_embedding_lengths,
)
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.sampler import (
    EarlyStopWithMMResult,
    MultimodalResult,
    SampleStateWithMMResult,
)


class _FakeRequest:
    def __init__(self, multimodal_lengths=None, multimodal_item_runs=None):
        self.py_request_id = 0
        self.multimodal_lengths = multimodal_lengths
        self.multimodal_item_runs = multimodal_item_runs
        self.py_result = PyResult(prompt_len=5, max_new_tokens=1)
        self.state = None
        self.finished_reason = None

    def set_finished_reason(self, finish_reason, beam_idx):
        self.finished_reason = (finish_reason, beam_idx)


def _make_request(multimodal_lengths=None, multimodal_item_runs=None):
    return _FakeRequest(
        multimodal_lengths=multimodal_lengths,
        multimodal_item_runs=multimodal_item_runs,
    )


def test_request_multimodal_embedding_lengths_prefer_item_runs():
    request = _make_request(
        multimodal_lengths=[5],
        multimodal_item_runs=[[(1, 5, [2, 4])]],
    )

    assert get_request_multimodal_embedding_lengths(request) == [3]


def test_request_multimodal_embedding_lengths_fall_back_to_legacy_lengths():
    request = _make_request(multimodal_lengths=[5])

    assert get_request_multimodal_embedding_lengths(request) == [5]


def test_mm_result_splitting_uses_item_run_embedding_lengths():
    request = _make_request(
        multimodal_lengths=[5],
        multimodal_item_runs=[[(1, 5, [2, 4])]],
    )
    mm_embedding = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    state = SampleStateWithMMResult(
        requests=[request],
        data=MultimodalResult(mm_embeddings=[mm_embedding]),
    )

    EarlyStopWithMMResult().update_requests(state)

    handles = request.py_result.mm_embedding_handles
    assert handles is not None
    assert len(handles) == 1
    assert handles[0]["tensor_size"] == [3, 2]


class _FakeVisionEncoder:
    def __init__(self, output):
        self.output = output

    def forward(self, multimodal_params):
        assert len(multimodal_params) == 2
        return [self.output]


class _FakeEngine:
    def __init__(self, output):
        self.model = _FakeVisionEncoder(output)


class _FakeScheduledRequests:
    def __init__(self, context_requests):
        self.context_requests = context_requests
        self.num_context_requests = len(context_requests)


def test_mm_encoder_only_splits_by_item_run_embedding_lengths():
    requests = [
        _make_request(
            multimodal_lengths=[5],
            multimodal_item_runs=[[(0, 5, [1, 3])]],
        ),
        _make_request(
            multimodal_lengths=[4],
            multimodal_item_runs=[[(0, 4, [2])]],
        ),
    ]
    engine = _FakeEngine(torch.arange(12, dtype=torch.float32).reshape(6, 2))

    result = PyTorchModelEngine._forward_step_mm_encoder_only(
        engine,
        {
            "multimodal_params": [
                SimpleNamespace(multimodal_data={}),
                SimpleNamespace(multimodal_data={}),
            ]
        },
        _FakeScheduledRequests(requests),
    )

    assert [embedding.shape for embedding in result["mm_embeddings"]] == [
        torch.Size([3, 2]),
        torch.Size([3, 2]),
    ]
