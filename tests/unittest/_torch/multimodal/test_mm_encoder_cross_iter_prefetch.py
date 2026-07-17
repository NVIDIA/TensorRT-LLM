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

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_multimodal_mixin as mm_mixin
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalModelMixin,
    _get_mm_aux_stream,
    maybe_prefetch_mm_encoder_for_next_iter,
)
from tensorrt_llm._torch.models.modeling_multimodal_utils import get_multimodal_embeddings
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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


class _CacheStubModel(_StubModel):
    supports_encoder_cache = True

    def __init__(self, hidden_size: int, tokens_per_image: int):
        super().__init__(hidden_size, tokens_per_image)
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(
                encoder_cache_max_bytes=4096,
                encoder_side_stream_max_ahead=2,
            )
        )
        self.last_encoder_batch_size = 0

    @property
    def embedding_dim(self) -> int:
        return self._hidden_size

    @property
    def embedding_dtype(self) -> torch.dtype:
        return torch.float32

    def encode_multimodal_inputs(self, multimodal_params, **kwargs) -> torch.Tensor:
        self.encoder_call_count += 1
        self.last_encoder_batch_size = len(multimodal_params)
        self.encoder_call_stream_id = torch.cuda.current_stream().cuda_stream
        return torch.full(
            (len(multimodal_params) * self._tokens_per_image, self._hidden_size),
            float(self.encoder_call_count),
            device="cuda",
        )


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


def _make_cacheable_request(
    request_id: int,
    num_tokens: int,
    *,
    item_hash: list[int] | None = None,
) -> LlmRequest:
    if item_hash is None:
        item_hash = [1, 2, 3, 4, 5, 6, 7, 8]
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[0] * num_tokens,
        sampling_config=SamplingConfig(beam_width=1),
        is_streaming=False,
        py_multimodal_data={
            "image": {"pixel_values": torch.randn(3, 32, 32)},
            "multimodal_embed_mask_cumsum": torch.arange(1, num_tokens + 1, dtype=torch.int64),
            "multimodal_embedding_lengths": [num_tokens],
            "mm_processor_kwargs_hash": "kwargs-a",
        },
        multimodal_hashes=[item_hash],
        multimodal_positions=[0],
        multimodal_lengths=[num_tokens],
        multimodal_uuids=[None],
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


@requires_cuda
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


@requires_cuda
def test_cross_iter_prefetch_populates_and_reuses_persistent_cache():
    max_prefetch_ahead = 2
    aux_stream = _get_mm_aux_stream(max_prefetch_ahead)
    assert aux_stream is not None
    model = _CacheStubModel(hidden_size=8, tokens_per_image=4)

    first = _make_cacheable_request(request_id=0, num_tokens=4)
    assert (
        maybe_prefetch_mm_encoder_for_next_iter(
            model, [first], max_prefetch_ahead=max_prefetch_ahead
        )
        == 1
    )
    first.py_mm_encoder_event.synchronize()
    first_embedding = first.py_multimodal_data["multimodal_embedding"]

    cache = model._multimodal_encoder_cache
    assert cache is not None
    assert len(cache) == 1
    assert model.encoder_call_count == 1
    assert model.encoder_call_stream_id == aux_stream.cuda_stream

    second = _make_cacheable_request(request_id=1, num_tokens=4)
    assert (
        maybe_prefetch_mm_encoder_for_next_iter(
            model, [second], max_prefetch_ahead=max_prefetch_ahead
        )
        == 1
    )
    second.py_mm_encoder_event.synchronize()

    assert model.encoder_call_count == 1
    torch.testing.assert_close(second.py_multimodal_data["multimodal_embedding"], first_embedding)


@requires_cuda
def test_cross_iter_prefetch_mixed_cache_hit_and_miss_encodes_only_miss():
    max_prefetch_ahead = 2
    model = _CacheStubModel(hidden_size=8, tokens_per_image=4)
    first = _make_cacheable_request(request_id=0, num_tokens=4)
    maybe_prefetch_mm_encoder_for_next_iter(model, [first], max_prefetch_ahead=max_prefetch_ahead)
    first.py_mm_encoder_event.synchronize()

    hit = _make_cacheable_request(request_id=1, num_tokens=4)
    miss = _make_cacheable_request(request_id=2, num_tokens=4, item_hash=[9] * 8)
    assert (
        maybe_prefetch_mm_encoder_for_next_iter(
            model,
            [hit, miss],
            max_prefetch=2,
            max_prefetch_ahead=max_prefetch_ahead,
        )
        == 2
    )
    hit.py_mm_encoder_event.synchronize()

    assert model.encoder_call_count == 2
    assert model.last_encoder_batch_size == 1
    torch.testing.assert_close(
        hit.py_multimodal_data["multimodal_embedding"],
        torch.ones((4, 8), device="cuda"),
    )
    torch.testing.assert_close(
        miss.py_multimodal_data["multimodal_embedding"],
        torch.full((4, 8), 2.0, device="cuda"),
    )


@requires_cuda
def test_cross_iter_prefetch_cache_model_preserves_uncacheable_fallbacks():
    max_prefetch_ahead = 2
    model = _CacheStubModel(hidden_size=8, tokens_per_image=4)
    unkeyable = _make_request(request_id=0, num_tokens=4)
    mixed_modality = _make_cacheable_request(request_id=1, num_tokens=4)
    mixed_modality.py_multimodal_data["audio"] = {"input_features": torch.empty(1)}

    assert (
        maybe_prefetch_mm_encoder_for_next_iter(
            model,
            [unkeyable, mixed_modality],
            max_prefetch=2,
            max_prefetch_ahead=max_prefetch_ahead,
        )
        == 2
    )
    unkeyable.py_mm_encoder_event.synchronize()

    assert model.encoder_call_count == 1
    assert model.last_encoder_batch_size == 2
    cache = model._multimodal_encoder_cache
    assert cache is not None
    assert len(cache) == 0


@requires_cuda
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


@requires_cuda
def test_cross_iter_prefetch_does_not_synchronize_main_stream(monkeypatch):
    """Routing through the cache must not block the calling (main) stream.

    The prefetch path may only enqueue aux-stream work and cross-stream waits; a
    blocking host/stream synchronization would defeat the overlap the side stream
    exists for. Spy on the synchronization entry points and assert none fire while
    the routed dispatch runs.
    """
    model = _CacheStubModel(hidden_size=8, tokens_per_image=4)
    req = _make_cacheable_request(request_id=0, num_tokens=4)
    original_encoder = model.encode_multimodal_inputs

    def delayed_encoder(multimodal_params, **kwargs):
        torch.cuda._sleep(100_000_000)
        return original_encoder(multimodal_params, **kwargs)

    monkeypatch.setattr(model, "encode_multimodal_inputs", delayed_encoder)

    sync_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: sync_calls.append("device"))
    monkeypatch.setattr(
        torch.cuda.Stream, "synchronize", lambda self, *a, **k: sync_calls.append("stream")
    )

    n = maybe_prefetch_mm_encoder_for_next_iter(model, [req], max_prefetch_ahead=2)

    assert n == 1
    assert sync_calls == []
    assert not req.py_mm_encoder_event.query()
    # Drain the queued aux-stream work via the (unpatched) event before teardown.
    req.py_mm_encoder_event.synchronize()


@requires_cuda
def test_cross_iter_prefetch_does_not_rewrite_request_local_embedding():
    """A present request-local embedding skips both the cache lookup and the write.

    Reproduces the next-iteration in-iter consume of a request whose embedding was
    already produced by a prefetch miss: the encoder must not run again and the
    persistent cache must not be rewritten.
    """
    model = _CacheStubModel(hidden_size=8, tokens_per_image=4)
    req = _make_cacheable_request(request_id=0, num_tokens=4)
    assert maybe_prefetch_mm_encoder_for_next_iter(model, [req], max_prefetch_ahead=2) == 1
    req.py_mm_encoder_event.synchronize()

    cache = model._multimodal_encoder_cache
    stats_before = cache.stats()
    assert stats_before.insertions == 1

    cumsum = req.py_multimodal_data["multimodal_embed_mask_cumsum"]
    param = MultimodalParams(
        multimodal_input=mm_mixin._build_request_multimodal_input(req, cache_enabled=True),
        multimodal_data=req.py_multimodal_data,
        multimodal_runtime=MultimodalRuntimeData(
            past_seen_token_num=0,
            chunk_end_pos=cumsum.numel(),
            embed_mask_cumsum=cumsum,
        ),
    )
    model._get_or_encode_multimodal_embeddings([param])

    assert model.encoder_call_count == 1  # no re-encode
    stats_after = cache.stats()
    assert stats_after.insertions == stats_before.insertions
    assert stats_after.replacements == stats_before.replacements


@requires_cuda
@pytest.mark.parametrize("embedding_count", [1, 2])
def test_prefetched_embedding_records_main_consumer_stream(monkeypatch, embedding_count):
    """The request may release an aux-produced embedding before its main-stream gather completes."""
    aux_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()
    producer_event = torch.cuda.Event()
    with torch.cuda.stream(aux_stream):
        embeddings = [torch.ones(4, device="cuda") for _ in range(embedding_count)]
        producer_event.record(aux_stream)

    record_stream_calls = []
    original_record_stream = torch.Tensor.record_stream

    def record_stream(tensor, stream):
        record_stream_calls.append((tensor.data_ptr(), stream.cuda_stream))
        return original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    param = MultimodalParams(
        multimodal_data={
            "multimodal_embedding": embeddings[0] if embedding_count == 1 else embeddings
        },
        encoder_event=producer_event,
    )

    with torch.cuda.stream(consumer_stream):
        gathered = get_multimodal_embeddings(
            encoder_forward_fn=lambda _: pytest.fail("attached embeddings must skip the encoder"),
            multimodal_params=[param],
        )

    consumer_stream.synchronize()
    expected_calls = {
        (embedding.data_ptr(), consumer_stream.cuda_stream) for embedding in embeddings
    }
    assert expected_calls.issubset(record_stream_calls)
    torch.testing.assert_close(gathered[0], torch.ones(4 * embedding_count, device="cuda"))
