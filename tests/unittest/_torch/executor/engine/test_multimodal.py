# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _torch.executor.multimodal_utils import (
    bare_mm_item_scheduler,
    make_llm_request,
    make_mm_request,
)

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
)
from tensorrt_llm._torch.pyexecutor.engine.multimodal import (
    MultimodalItemScheduler,
    resolve_mm_encoder_output_budget,
    validate_mm_encoder_scheduling_compatibility,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    MultimodalEncoderRequestError,
    initialize_multimodal_encoder_request,
    is_multimodal_encoder_ready,
)
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalParams
from tensorrt_llm.inputs.registry import MultimodalEncoderItemMetadata
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderSchedulingPolicy

# The item-scheduling surface is pure logic: no kernels, no device transfers.
pytestmark = pytest.mark.cpu_only


def _cache_request(
    request_id: int,
    *,
    hashes: list[list[int]] | None,
    embedding_lengths: Sequence[int],
    kwargs_hash: str | None = "kw",
) -> LlmRequest:
    """A cache-keyable item-scheduling request with raw image payload."""
    num_items = len(embedding_lengths)
    multimodal_data = {
        "image": {
            "pixel_values": torch.arange(sum(embedding_lengths)).unsqueeze(1),
            "image_grid_thw": torch.tensor([[1, 1, length] for length in embedding_lengths]),
        },
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
            item_refs=[("image", item_idx) for item_idx in range(num_items)],
            encoder_token_lengths=list(embedding_lengths),
            output_embedding_lengths=list(embedding_lengths),
        ),
        "multimodal_embedding_lengths": list(embedding_lengths),
    }
    if kwargs_hash is not None:
        multimodal_data["mm_processor_kwargs_hash"] = kwargs_hash
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        py_multimodal_data=multimodal_data,
        multimodal_hashes=hashes,
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=1 << 30)
    return request


def _cache_mm_item_scheduler(
    cache: TensorLRUCache[Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    supports_encoder_cache: bool = True,
) -> MultimodalItemScheduler:
    class _Model(MultimodalModelMixin):
        supports_encoder_cache = False

        def __init__(self):
            self.encoded_item_counts = []

        def _get_multimodal_encoder_cache(self):
            return cache

        def forward_multimodal_encoder_items(self, encoder_inputs):
            # Items, not input tuples: adjacent same-request same-modality
            # items are sliced into one tuple.
            self.encoded_item_counts.append(sum(len(lengths) for _, lengths, _ in encoder_inputs))
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_lengths, _ in encoder_inputs
                for embedding_length in embedding_lengths
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    model = _Model()
    model.supports_encoder_cache = supports_encoder_cache
    return bare_mm_item_scheduler(model)


def test_qwen3_output_budget_uses_post_merge_embedding_capacity() -> None:
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    # 16384 rows of fp16 -> 32768 bytes per embedding, via the mixin's explicit
    # embedding_dim/embedding_dtype contract.
    model = SimpleNamespace(embedding_dim=16384, embedding_dtype=torch.float16)

    budget, bytes_per_embedding = resolve_mm_encoder_output_budget(processor, 65536, model)

    assert bytes_per_embedding == 32768
    assert budget == 512 * 1024**2


def test_output_budget_requires_processor_embedding_capacity() -> None:
    processor = SimpleNamespace(get_max_mm_encoder_output_embeddings=lambda *_: None)

    # The embedding capacity is validated before the model is consulted, so
    # `model=None` never gets there.
    with pytest.raises(ValueError, match="get_max_mm_encoder_output_embeddings"):
        resolve_mm_encoder_output_budget(processor, 65536, None)


def test_eager_compatibility_is_checked_only_for_item_scheduled_models() -> None:
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
            encoder_side_stream_max_ahead=0,
        ),
        enable_attention_dp=True,
        cache_transceiver_config=SimpleNamespace(backend="NIXL"),
    )

    validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="attention DP"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)

    args.enable_attention_dp = False
    with pytest.raises(ValueError, match="disaggregated"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_side_stream_compatibility_is_checked_only_for_item_scheduled_models() -> None:
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
            encoder_side_stream_max_ahead=1,
        ),
        enable_attention_dp=False,
        cache_transceiver_config=None,
    )

    validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="side-stream prefetch"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_item_encoder_classifies_request_state_contract_errors() -> None:
    class _Model(MultimodalModelMixin):
        pass

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_llm_request(1)

    with pytest.raises(MultimodalEncoderRequestError, match="no longer active"):
        mm_item_scheduler.forward_items([], {request.request_id: [0]})

    with pytest.raises(MultimodalEncoderRequestError, match="no encoder item state"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


def test_item_encoder_classifies_output_count_contract_error() -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            return []

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])

    with pytest.raises(MultimodalEncoderRequestError, match="one output per item"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


@pytest.mark.parametrize("failure_stage", ["prepare", "forward"])
def test_item_encoder_translates_model_contract_errors(failure_stage: str) -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            if failure_stage == "prepare":
                raise MultimodalEncoderContractError("bad request metadata")
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            raise MultimodalEncoderContractError("bad encoder output rows")

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])

    expected = "bad request metadata" if failure_stage == "prepare" else "bad encoder output rows"
    with pytest.raises(MultimodalEncoderRequestError, match=expected):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


@pytest.mark.parametrize("failure_stage", ["prepare", "forward"])
def test_item_encoder_does_not_translate_system_errors(failure_stage: str) -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            if failure_stage == "prepare":
                raise torch.cuda.OutOfMemoryError("encoder OOM")
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            raise torch.cuda.OutOfMemoryError("encoder OOM")

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])

    with pytest.raises(torch.cuda.OutOfMemoryError, match="encoder OOM"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


def test_item_outputs_accumulate_on_request_and_release_raw_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model(MultimodalModelMixin):
        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_lengths, _ in encoder_inputs
                for embedding_length in embedding_lengths
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_llm_request(
        1,
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(5).unsqueeze(1),
                "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", 0), ("image", 1)],
                encoder_token_lengths=[2, 3],
                output_embedding_lengths=[2, 3],
            ),
            "multimodal_embedding_lengths": [2, 3],
        },
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=8)
    assert request.py_mm_encoder_state.embedding_lengths == [2, 3]

    mm_item_scheduler.forward_items([request], {1: [0]})

    # Items encoded across iterations land in their own rows of one buffer
    # sized for the whole request, so the charge is already the full
    # footprint. Raw inputs stay until every item is in.
    state = request.py_mm_encoder_state
    assert state.embeddings.shape == (2 + 3, 2)
    assert state.recorded == [True, False]
    assert state.resident_output_bytes(8) == (2 + 3) * 8
    assert "image" in request.py_multimodal_data

    mm_item_scheduler.forward_items([request], {1: [1]})

    # Publishing is by reference: the buffer is already the contiguous form
    # prefill consumes, so nothing is copied or concatenated here.
    published = request.py_multimodal_data["multimodal_embedding"]
    assert published is state.embeddings
    assert published.tolist() == [
        [2.0, 2.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [3.0, 3.0],
        [3.0, 3.0],
    ]
    assert "image" not in request.py_multimodal_data
    assert state.resident_output_bytes(8) == (2 + 3) * 8
    assert is_multimodal_encoder_ready(request)


def test_duplicate_request_hits_cache_and_skips_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TensorLRUCache(1 << 20, name="test")
    mm_item_scheduler = _cache_mm_item_scheduler(cache, monkeypatch)
    first = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    mm_item_scheduler.forward_items([first], {1: [0, 1]})
    assert mm_item_scheduler.model.encoded_item_counts == [2]

    second = _cache_request(2, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    mm_item_scheduler.forward_items([second], {2: [0, 1]})

    # Read-through: every item hit, so the encoder never ran again, and each
    # request owns an independent copy (no cross-request aliasing).
    assert mm_item_scheduler.model.encoded_item_counts == [2]
    assert is_multimodal_encoder_ready(second)
    published = second.py_multimodal_data["multimodal_embedding"]
    first_published = first.py_multimodal_data["multimodal_embedding"]
    assert torch.equal(published, first_published)
    assert published.untyped_storage().data_ptr() != first_published.untyped_storage().data_ptr()


def test_cache_hit_at_encode_skips_only_hit_items(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TensorLRUCache(1 << 20, name="test")
    mm_item_scheduler = _cache_mm_item_scheduler(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    key0, _ = MultimodalModelMixin.build_encoder_cache_item_keys(
        [[1, 2], [3, 4]], [("image", 0), ("image", 1)], [2, 3], "kw"
    )
    cache.put(key0, torch.full((2, 2), 7.0))  # entry from an earlier request
    mm_item_scheduler.forward_items([request], {1: [0, 1]})

    assert mm_item_scheduler.model.encoded_item_counts == [1]  # only the miss encoded
    published = request.py_multimodal_data["multimodal_embedding"]
    assert torch.equal(published[:2], torch.full((2, 2), 7.0))
    assert is_multimodal_encoder_ready(request)


def test_cache_eviction_leaves_recorded_slots_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TensorLRUCache(2 * 2 * 4, name="test")  # holds exactly one 2-row item
    mm_item_scheduler = _cache_mm_item_scheduler(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2]], embedding_lengths=[2])
    mm_item_scheduler.forward_items([request], {1: [0]})
    recorded = request.py_multimodal_data["multimodal_embedding"][:2]
    assert len(cache) == 1

    cache.clear()  # simulate eviction of the entry the request came from

    assert torch.equal(recorded, torch.full((2, 2), 2.0))  # owned clone, untouched
    assert is_multimodal_encoder_ready(request)


def test_cache_off_encodes_every_item_without_touching_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # supports_encoder_cache=False -> mm_encoder_cache is None -> pure encode.
    cache = TensorLRUCache(1 << 20, name="test")
    mm_item_scheduler = _cache_mm_item_scheduler(cache, monkeypatch, supports_encoder_cache=False)
    assert mm_item_scheduler.encoder_cache is None
    request = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])

    mm_item_scheduler.forward_items([request], {1: [0, 1]})

    assert mm_item_scheduler.model.encoded_item_counts == [2]
    assert is_multimodal_encoder_ready(request)
    assert len(cache) == 0  # never populated


@pytest.mark.parametrize("case", ["no_hashes", "no_kwargs_hash", "count_mismatch"])
def test_key_guards_bypass_cache(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TensorLRUCache(1 << 20, name="test")
    mm_item_scheduler = _cache_mm_item_scheduler(cache, monkeypatch)
    request = _cache_request(
        1,
        hashes=None
        if case == "no_hashes"
        else ([[1, 2]] if case == "count_mismatch" else [[1, 2], [3, 4]]),
        embedding_lengths=[2, 3],
        kwargs_hash=None if case == "no_kwargs_hash" else "kw",
    )

    assert mm_item_scheduler.item_keys(request) is None

    mm_item_scheduler.forward_items([request], {1: [0, 1]})
    assert is_multimodal_encoder_ready(request)
    assert len(cache) == 0  # unkeyable items never populate the cache
