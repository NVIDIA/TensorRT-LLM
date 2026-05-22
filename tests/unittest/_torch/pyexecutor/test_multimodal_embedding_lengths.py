# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequestState,
    LlmResponse,
    LlmResult,
    PyResult,
    get_multimodal_embedding_lengths,
)
from tensorrt_llm._torch.pyexecutor.sampler import EarlyStopWithMMResult, MultimodalResult
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.executor import GenerationRequest, GenerationResult
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.sampling_params import SamplingParams


@pytest.mark.parametrize(
    "req,expected",
    [
        (SimpleNamespace(multimodal_lengths=[3, 5]), None),
        (
            SimpleNamespace(
                multimodal_lengths=[6, 5],
                py_multimodal_data={"multimodal_embedding_lengths": [5, 3]},
            ),
            [5, 3],
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[6, 5],
                py_multimodal_data={
                    "layout_metadata": {
                        "multimodal_embedding_lengths": [6, 4],
                    },
                },
            ),
            None,
        ),
    ],
)
def test_multimodal_embedding_lengths_returns_top_level_metadata(req, expected):
    """Getter reads top-level lengths and ignores layout metadata."""
    assert get_multimodal_embedding_lengths(req) == expected


@pytest.mark.parametrize(
    "req,exception,match",
    [
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={"multimodal_embedding_lengths": [3, 1]},
            ),
            ValueError,
            "length must match",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={"multimodal_embedding_lengths": [5]},
            ),
            ValueError,
            "exceeds",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={"multimodal_embedding_lengths": [-1]},
            ),
            ValueError,
            "non-negative",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={
                    "multimodal_embedding_lengths": torch.tensor([4]),
                },
            ),
            TypeError,
            "must be a list",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data={"multimodal_embedding_lengths": (4,)},
            ),
            TypeError,
            "must be a list",
        ),
        (
            SimpleNamespace(
                multimodal_lengths=[4],
                py_multimodal_data=["multimodal_embedding_lengths", [4]],
            ),
            TypeError,
            "py_multimodal_data must be a dict",
        ),
    ],
)
def test_multimodal_embedding_lengths_rejects_invalid_metadata(req, exception, match):
    """Bad length metadata is rejected by the getter."""
    with pytest.raises(exception, match=match):
        get_multimodal_embedding_lengths(req)


class _FakePyResult:
    def __init__(self):
        self.mm_embeddings = []

    def append_mm_embeddings(self, mm_embedding, mm_embedding_lengths):
        self.mm_embeddings.append((mm_embedding, mm_embedding_lengths))


class _FakeRequest:
    def __init__(self, multimodal_lengths=None):
        self.multimodal_lengths = multimodal_lengths
        self.py_result = _FakePyResult()
        self.state = None
        self.finished_reason = None

    def set_finished_reason(self, reason, beam):
        self.finished_reason = (reason, beam)


def test_mm_encoder_sampler_aligns_mixed_batch_by_request_index():
    """Sparse MM encoder outputs attach to the original request index."""
    text_request = _FakeRequest()
    mm_request = _FakeRequest(multimodal_lengths=[4])
    sampler = EarlyStopWithMMResult()
    state = sampler.SampleState(
        requests=[text_request, mm_request],
        data=MultimodalResult(
            mm_embeddings=[torch.ones(4, 2)],
            mm_embedding_request_indices=[1],
            mm_embedding_lengths=[[4]],
            extra_data={},
        ),
    )

    sampler.update_requests(state)

    assert text_request.state == LlmRequestState.GENERATION_COMPLETE
    assert mm_request.state == LlmRequestState.GENERATION_COMPLETE
    assert text_request.finished_reason == (FinishReason.LENGTH, 0)
    assert mm_request.finished_reason == (FinishReason.LENGTH, 0)
    assert len(text_request.py_result.mm_embeddings) == 0
    [(mm_embedding, mm_embedding_lengths)] = mm_request.py_result.mm_embeddings
    assert mm_embedding.shape == (4, 2)
    assert mm_embedding_lengths == [4]


def test_py_result_mm_embedding_handles_use_cpu_shared_memory():
    """MM encoder result handles should be CPU-backed shared-memory handles."""
    result = PyResult(prompt_len=1, max_new_tokens=1)
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    result.append_mm_embeddings(source, [1, 3])

    handles = result.mm_embedding_handles
    assert handles is not None
    assert [handle["method_key"] for handle in handles] == [2, 2]
    restored = [SharedTensorContainer.from_dict(handle).get_local_view() for handle in handles]
    assert torch.equal(torch.cat(restored, dim=0), source)


def test_py_result_mm_embedding_handles_outlive_source_tensor():
    result = PyResult(prompt_len=1, max_new_tokens=1)

    def append_from_local_tensor():
        source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        result.append_mm_embeddings(source, [4])

    append_from_local_tensor()
    gc.collect()

    restored = SharedTensorContainer.from_dict(result.mm_embedding_handles[0]).get_local_view()
    assert torch.equal(restored, torch.arange(8, dtype=torch.float32).reshape(4, 2))


def test_generation_result_owns_mm_embedding_handles_for_disagg_handoff():
    py_result = PyResult(prompt_len=1, max_new_tokens=1)
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    py_result.append_mm_embeddings(source, [4])

    result = tllm.Result()
    result.output_token_ids = [[1]]
    result.context_logits = None
    result.generation_logits = None
    result.log_probs = None
    result.cum_log_probs = None
    result.finish_reasons = [tllm.FinishReason.END_ID]
    result.is_final = True
    result.sequence_index = 0
    response = LlmResponse(
        request_id=0,
        result=LlmResult(result, py_result, is_final=True),
        client_id=0,
    )

    request = GenerationRequest(
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
        multimodal_params=MultimodalParams(
            multimodal_input=MultimodalInput(
                multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
                multimodal_positions=[0],
                multimodal_lengths=[4],
            )
        ),
    ).set_id(0)
    generation_result = GenerationResult(request)
    generation_result._handle_response(response)
    disagg_params = generation_result.disaggregated_params

    del response, py_result, source
    gc.collect()

    restored = SharedTensorContainer.from_dict(
        disagg_params.multimodal_embedding_handles[0]
    ).get_local_view()
    assert torch.equal(restored, torch.arange(8, dtype=torch.float32).reshape(4, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_py_result_cuda_mm_embedding_handles_are_cpu_backed():
    """CUDA MM encoder outputs are copied to CPU before creating handoff handles."""
    result = PyResult(prompt_len=1, max_new_tokens=1)
    source = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)

    result.append_mm_embeddings(source, [2, 2])

    handles = result.mm_embedding_handles
    assert handles is not None
    assert [handle["method_key"] for handle in handles] == [2, 2]
    restored = [SharedTensorContainer.from_dict(handle).get_local_view() for handle in handles]
    assert all(tensor.device.type == "cpu" for tensor in restored)
    assert torch.equal(torch.cat(restored, dim=0).to(source.device), source)


class _FakeScheduledRequests:
    def __init__(self, num_context_requests):
        self.generation_requests = []
        self.context_requests = [_FakeRequest() for _ in range(num_context_requests)]

    @property
    def num_context_requests(self):
        return len(self.context_requests)


def test_mm_encoder_sampler_builds_typed_result_from_model_outputs():
    """Sampler converts raw model-output dicts into typed MM results."""
    sampler = EarlyStopWithMMResult()
    state = sampler.sample_async(
        _FakeScheduledRequests(2),
        {
            "mm_embeddings": [torch.ones(4, 2)],
            "mm_embedding_request_indices": [1],
            "mm_embedding_lengths": [[4]],
            "mrope_position_ids": ["pos"],
        },
        [],
    )

    assert state.data.mm_embedding_request_indices == [1]
    assert state.data.mm_embedding_lengths == [[4]]
    assert state.data.extra_data == {"mrope_position_ids": ["pos"]}


def test_mm_encoder_sampler_rejects_typed_result_batch_mismatch():
    """MM embedding arrays must stay length-aligned with request indices."""
    sampler = EarlyStopWithMMResult()

    with pytest.raises(ValueError, match="batch size"):
        sampler.sample_async(
            _FakeScheduledRequests(2),
            {
                "mm_embeddings": [torch.ones(4, 2)],
                "mm_embedding_request_indices": [1],
                "mm_embedding_lengths": [],
            },
            [],
        )


def test_mm_encoder_sampler_rejects_invalid_request_index():
    """MM encoder output cannot target a request outside the scheduled batch."""
    sampler = EarlyStopWithMMResult()

    with pytest.raises(ValueError, match="invalid request index"):
        sampler.sample_async(
            _FakeScheduledRequests(1),
            {
                "mm_embeddings": [torch.ones(4, 2)],
                "mm_embedding_request_indices": [1],
                "mm_embedding_lengths": [[4]],
            },
            [],
        )


def test_multimodal_result_rejects_embedding_shape_mismatch():
    """Per-item lengths must sum to the attached embedding rows."""
    with pytest.raises(ValueError, match="shape mismatch"):
        MultimodalResult(
            mm_embeddings=[torch.ones(4, 2)],
            mm_embedding_request_indices=[0],
            mm_embedding_lengths=[[3]],
            extra_data={},
        )
