# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for cross-iteration MM encoder prefetch.

Exercises ``maybe_prefetch_mm_encoder_for_next_iter`` in isolation against a
stub model and real ``LlmRequest`` objects. End-to-end perf validation is
out of scope here; this just verifies the plumbing.
"""

import pytest
import torch

from tensorrt_llm._torch.models import modeling_multimodal_mixin as mm_mixin
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalModelMixin,
    _get_mm_aux_stream,
    maybe_prefetch_mm_encoder_for_next_iter,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig


class _StubModel(MultimodalModelMixin):
    def __init__(self, hidden_size: int, tokens_per_image: int):
        self._hidden_size = hidden_size
        self._tokens_per_image = tokens_per_image
        self.encoder_call_count = 0
        self.encoder_call_stream_id = None

    def encode_multimodal_inputs(self, multimodal_params, **kwargs) -> torch.Tensor:
        self.encoder_call_count += 1
        self.encoder_call_stream_id = torch.cuda.current_stream().cuda_stream
        pv = multimodal_params[0].multimodal_data["image"]["pixel_values"]
        assert pv.device.type == "cuda", (
            "pixel_values should be on CUDA after to_device in the helper."
        )
        embeddings = torch.randn(
            len(multimodal_params) * self._tokens_per_image,
            self._hidden_size,
            device=pv.device,
        )
        return embeddings


def _make_request(request_id: int, num_tokens: int) -> LlmRequest:
    pixel_values = torch.randn(3, 32, 32)  # CPU tensor, simulating unscheduled
    cumsum = torch.arange(1, num_tokens + 1, dtype=torch.int64)
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[0] * num_tokens,
        sampling_config=SamplingConfig(beam_width=1),
        is_streaming=False,
        py_multimodal_data={
            "image": {
                "pixel_values": pixel_values,
            },
            "multimodal_embed_mask_cumsum": cumsum,
        },
    )


def _make_metadata_only_request(request_id: int, num_tokens: int) -> LlmRequest:
    cumsum = torch.arange(1, num_tokens + 1, dtype=torch.int64)
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[0] * num_tokens,
        sampling_config=SamplingConfig(beam_width=1),
        is_streaming=False,
        py_multimodal_data={
            "layout_metadata": {"special_token_offsets": []},
            "mrope_config": {"mrope_position_ids": torch.zeros(3, 1, num_tokens)},
            "multimodal_embed_mask_cumsum": cumsum,
            "special_token_offsets": [],
        },
    )


@pytest.fixture(autouse=True)
def _reset_env_and_streams(monkeypatch):
    monkeypatch.delenv("TLLM_MM_SIDE_STREAM", raising=False)
    yield


def test_cross_iter_prefetch_max_ahead_counts_cached_requests(monkeypatch):
    monkeypatch.setattr(mm_mixin, "_get_mm_aux_stream", lambda *args, **kwargs: object())

    dispatches = []
    monkeypatch.setattr(
        mm_mixin,
        "_dispatch_cross_iter_prefetch",
        lambda model, candidates, aux_stream: dispatches.append(candidates),
    )

    model = _StubModel(hidden_size=8, tokens_per_image=4)
    cached_req = _make_request(request_id=0, num_tokens=4)
    cached_req.py_multimodal_data["multimodal_embedding"] = torch.zeros((4, 8))
    fresh_req = _make_request(request_id=1, num_tokens=4)

    n = maybe_prefetch_mm_encoder_for_next_iter(
        model, [cached_req, fresh_req], max_prefetch_ahead=1
    )

    assert n == 0
    assert dispatches == []


def test_cross_iter_prefetch_max_ahead_config_override(monkeypatch):
    monkeypatch.setattr(mm_mixin, "_get_mm_aux_stream", lambda *args, **kwargs: object())

    dispatches = []
    monkeypatch.setattr(
        mm_mixin,
        "_dispatch_cross_iter_prefetch",
        lambda model, candidates, aux_stream: dispatches.append(candidates),
    )

    model = _StubModel(hidden_size=8, tokens_per_image=4)
    cached_req = _make_request(request_id=0, num_tokens=4)
    cached_req.py_multimodal_data["multimodal_embedding"] = torch.zeros((4, 8))
    fresh_req = _make_request(request_id=1, num_tokens=4)

    n = maybe_prefetch_mm_encoder_for_next_iter(
        model,
        [cached_req, fresh_req],
        max_prefetch=4,
        max_prefetch_ahead=2,
    )

    assert n == 1
    assert len(dispatches) == 1
    assert [req.py_request_id for req, _, _ in dispatches[0]] == [1]


def test_cross_iter_prefetch_max_ahead_counts_pending_event(monkeypatch):
    monkeypatch.setattr(mm_mixin, "_get_mm_aux_stream", lambda *args, **kwargs: object())

    dispatches = []
    monkeypatch.setattr(
        mm_mixin,
        "_dispatch_cross_iter_prefetch",
        lambda model, candidates, aux_stream: dispatches.append(candidates),
    )

    model = _StubModel(hidden_size=8, tokens_per_image=4)
    in_progress_req = _make_request(request_id=0, num_tokens=4)
    in_progress_req.py_mm_encoder_event = object()
    fresh_req = _make_request(request_id=1, num_tokens=4)

    n = maybe_prefetch_mm_encoder_for_next_iter(
        model, [in_progress_req, fresh_req], max_prefetch_ahead=1
    )

    assert n == 0
    assert dispatches == []


def test_cross_iter_prefetch_skips_metadata_only_requests(monkeypatch):
    monkeypatch.setattr(mm_mixin, "_get_mm_aux_stream", lambda *args, **kwargs: object())

    dispatches = []
    monkeypatch.setattr(
        mm_mixin,
        "_dispatch_cross_iter_prefetch",
        lambda model, candidates, aux_stream: dispatches.append(candidates),
    )

    model = _StubModel(hidden_size=8, tokens_per_image=4)
    metadata_with_event_req = _make_metadata_only_request(request_id=0, num_tokens=4)
    metadata_with_event_req.py_mm_encoder_event = object()
    metadata_req = _make_metadata_only_request(request_id=1, num_tokens=4)
    fresh_req = _make_request(request_id=2, num_tokens=4)

    n = maybe_prefetch_mm_encoder_for_next_iter(
        model,
        [metadata_with_event_req, metadata_req, fresh_req],
        max_prefetch=4,
        max_prefetch_ahead=1,
    )

    assert n == 1
    assert len(dispatches) == 1
    assert [req.py_request_id for req, _, _ in dispatches[0]] == [2]


def test_cross_iter_prefetch_ignores_old_side_stream_env(monkeypatch):
    monkeypatch.setenv("TLLM_MM_SIDE_STREAM", "1")
    monkeypatch.setattr(
        mm_mixin,
        "_get_mm_aux_stream",
        lambda *args, **kwargs: pytest.fail("old env var should not enable prefetch"),
    )

    model = _StubModel(hidden_size=8, tokens_per_image=4)
    req = _make_request(request_id=0, num_tokens=4)

    n = maybe_prefetch_mm_encoder_for_next_iter(model, [req])

    assert n == 0
    assert "multimodal_embedding" not in req.py_multimodal_data


def test_cross_iter_prefetch_no_op_when_config_off():
    model = _StubModel(hidden_size=8, tokens_per_image=4)
    req = _make_request(request_id=0, num_tokens=4)
    n = maybe_prefetch_mm_encoder_for_next_iter(model, [req])
    assert n == 0
    assert "multimodal_embedding" not in req.py_multimodal_data
    assert req.py_mm_encoder_event is None


def test_cross_iter_prefetch_materializes_on_side_stream(monkeypatch):
    max_prefetch_ahead = 1
    aux_stream = _get_mm_aux_stream(max_prefetch_ahead)
    assert aux_stream is not None

    model = _StubModel(hidden_size=8, tokens_per_image=4)
    req = _make_request(request_id=0, num_tokens=4)

    n = maybe_prefetch_mm_encoder_for_next_iter(model, [req], max_prefetch_ahead=max_prefetch_ahead)

    assert n == 1
    assert model.encoder_call_count == 1
    assert model.encoder_call_stream_id == aux_stream.cuda_stream
    embedding = req.py_multimodal_data.get("multimodal_embedding")
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (4, 8)
    assert isinstance(req.py_mm_encoder_event, torch.cuda.Event)
    req.py_mm_encoder_event.synchronize()


def test_cross_iter_prefetch_skips_in_flight_and_cached(monkeypatch):
    model = _StubModel(hidden_size=8, tokens_per_image=4)
    in_flight_req = _make_request(request_id=0, num_tokens=4)
    cached_req = _make_request(request_id=1, num_tokens=4)
    # Simulate already cached.
    cached_req.py_multimodal_data["multimodal_embedding"] = torch.zeros((4, 8), device="cuda")
    fresh_req = _make_request(request_id=2, num_tokens=4)

    n = maybe_prefetch_mm_encoder_for_next_iter(
        model,
        [in_flight_req, cached_req, fresh_req],
        in_flight_request_ids=[0],
        max_prefetch_ahead=2,
    )
    assert n == 1
    assert "multimodal_embedding" not in in_flight_req.py_multimodal_data
    assert fresh_req.py_multimodal_data.get("multimodal_embedding") is not None
