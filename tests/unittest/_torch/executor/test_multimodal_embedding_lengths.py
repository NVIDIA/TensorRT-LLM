# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    PyResult,
    get_request_multimodal_embedding_lengths,
)
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.sampler import (
    EarlyStopWithMMResult,
    MultimodalResult,
    SampleStateWithMMResult,
)
from tensorrt_llm.sampling_params import SamplingParams

HASH_INTS = [
    0x01020304,
    0x05060708,
    0x11121314,
    0x15161718,
    0x21222324,
    0x25262728,
    0x31323334,
    0x35363738,
]


class _FakeRequest:
    def __init__(
        self,
        multimodal_embedding_lengths=None,
        multimodal_item_runs=None,
    ):
        self.py_request_id = 0
        self.multimodal_embedding_lengths = multimodal_embedding_lengths
        self.multimodal_item_runs = multimodal_item_runs
        self.py_result = PyResult(prompt_len=5, max_new_tokens=1)
        self.state = None
        self.finished_reason = None

    def set_finished_reason(self, finish_reason, beam_idx):
        self.finished_reason = (finish_reason, beam_idx)


def _make_request(
    multimodal_embedding_lengths=None,
    multimodal_item_runs=None,
):
    return _FakeRequest(
        multimodal_embedding_lengths=multimodal_embedding_lengths,
        multimodal_item_runs=multimodal_item_runs,
    )


def _sampling_config():
    return tensorrt_llm.bindings.SamplingConfig(SamplingParams()._get_sampling_config())


def _make_llm_request(
    multimodal_item_runs,
    *,
    multimodal_embedding_lengths=None,
    multimodal_prompt_lengths=None,
):
    return LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=[10, 999, 777, 999, 11],
        sampling_config=_sampling_config(),
        is_streaming=False,
        end_id=2,
        pad_id=0,
        multimodal_hashes=[HASH_INTS],
        multimodal_uuids=None,
        multimodal_item_runs=multimodal_item_runs,
        multimodal_embedding_lengths=multimodal_embedding_lengths,
        multimodal_prompt_lengths=multimodal_prompt_lengths,
    )


def test_llm_request_caches_multimodal_runtime_metadata_once():
    request = _make_llm_request([[(0, 5, [1, 3])]])

    assert request.multimodal_embedding_lengths == [3]
    assert request.multimodal_prompt_lengths == [5]
    assert request.py_multimodal_item_run_spans == [[(0, 5)]]


def test_llm_request_rejects_inconsistent_cached_embedding_lengths():
    with pytest.raises(ValueError, match="multimodal_embedding_lengths"):
        _make_llm_request(
            [[(0, 5, [1, 3])]],
            multimodal_embedding_lengths=[4],
        )


def test_request_multimodal_embedding_lengths_use_cached_field():
    request = _make_request(
        multimodal_embedding_lengths=[3],
        multimodal_item_runs=[[(1, 5, [2, 4])]],
    )

    assert get_request_multimodal_embedding_lengths(request) == [3]


def test_request_multimodal_embedding_lengths_returns_none_without_cached_field():
    request = _make_request()

    assert get_request_multimodal_embedding_lengths(request) is None


def test_mm_result_splitting_uses_cached_embedding_lengths():
    request = _make_request(
        multimodal_embedding_lengths=[3],
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
            multimodal_embedding_lengths=[3],
            multimodal_item_runs=[[(0, 5, [1, 3])]],
        ),
        _make_request(
            multimodal_embedding_lengths=[3],
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
