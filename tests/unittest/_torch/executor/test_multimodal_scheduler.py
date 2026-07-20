# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_mistral import Mistral3InputProcessor
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    MultimodalEncoderProgress,
    MultimodalEncoderRequestState,
    get_multimodal_encoder_token_lengths,
    initialize_multimodal_encoder_request,
    is_multimodal_encoder_ready,
)
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine,
    _resolve_mm_encoder_token_budget,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    MultimodalEagerEncoderScheduler,
    MultimodalScheduler,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalParams,
    strip_mm_encoder_inputs,
)
from tensorrt_llm.inputs.registry import MultimodalEncoderItemMetadata


class _CapacityScheduler:
    def schedule_request(self, requests):
        return list(requests), [], []


class _RejectMultimodalCapacityScheduler:
    def schedule_request(self, requests):
        fitting = [request for request in requests if request.py_mm_encoder_state is None]
        return fitting, [], []


class _MicroBatchScheduler:
    def schedule(self, requests, inflight_request_ids):
        del inflight_request_ids
        return [], list(requests), []


class _BaseScheduler:
    def __init__(self):
        self.capacity_scheduler = _CapacityScheduler()
        self.micro_batch_scheduler = _MicroBatchScheduler()

    def can_schedule(self, requests):
        return bool(requests)


def _llm_request(request_id, multimodal_data=None):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        py_multimodal_data=multimodal_data,
    )


def _request(request_id, costs, *, ready=()):
    request = _llm_request(
        request_id,
        multimodal_data={
            "image": {"pixel_values": torch.empty(len(costs), 1)},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", item_idx) for item_idx in range(len(costs))],
                encoder_token_lengths=costs,
                output_embedding_lengths=[1] * len(costs),
            ),
            "multimodal_embedding_lengths": [1] * len(costs),
        },
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=1 << 30)
    for item_idx in ready:
        request.py_mm_encoder_state.outputs[item_idx] = torch.empty(1)
    return request


def test_mm_encoder_token_lengths_distinguishes_missing_and_invalid_data():
    request = _llm_request(1)

    assert get_multimodal_encoder_token_lengths(request) is None

    request.py_multimodal_data = []
    with pytest.raises(TypeError, match="multimodal_data must be a dict"):
        get_multimodal_encoder_token_lengths(request)


def test_mm_encoder_readiness_is_derived_from_request_local_outputs():
    request = _request(1, [4, 4])
    assert request.py_mm_encoder_state.progress is MultimodalEncoderProgress.PENDING
    assert not is_multimodal_encoder_ready(request)

    request.py_mm_encoder_state.outputs[0] = torch.empty(1)
    assert request.py_mm_encoder_state.progress is MultimodalEncoderProgress.PARTIAL
    assert not is_multimodal_encoder_ready(request)

    request.py_mm_encoder_state.outputs[1] = torch.empty(1)
    assert is_multimodal_encoder_ready(request)

    # A precomputed-embedding request never gets item state in the first
    # place (initialize skips it), and post-prefill strip drops the state:
    # both report ready through the state-absence branch.
    request.py_mm_encoder_state = None
    assert is_multimodal_encoder_ready(request)


def test_multimodal_scheduler_keeps_items_atomic_and_backfills_requests():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=2, max_num_tokens=10)
    first = _request(1, [7, 7])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0], 2: [0]}
    assert output.context_requests == [second]


def test_multimodal_scheduler_selects_all_items_and_admits_request_when_batch_fits():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=2, max_num_tokens=10)
    request = _request(1, [6, 4])

    output = scheduler.schedule_request([request], set())

    # The encoder step is the single encode site: an in-budget batch simply
    # has every pending item selected, and the request still enters the LLM
    # batch in the same iteration (encode runs before the LLM forward).
    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == [request]


def test_multimodal_scheduler_withholds_request_on_budget_overflow():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=3, max_num_tokens=10)
    request = _request(1, [6, 4, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []


def test_multimodal_scheduler_preserves_non_multimodal_requests():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=1, max_num_tokens=1)
    request = _llm_request(1)
    initialize_multimodal_encoder_request(request, max_num_tokens=1)

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]


def test_encoder_token_budget_auto_raises_for_atomic_item():
    assert _resolve_mm_encoder_token_budget(8192, 65536) == 65536


def test_request_rejects_item_above_effective_startup_maximum():
    request = _request(1, [9])
    request.py_multimodal_data["image"] = {"pixel_values": torch.empty(1)}

    with pytest.raises(ValueError, match="exceeding the effective startup maximum 8"):
        initialize_multimodal_encoder_request(request, max_num_tokens=8)


def test_eager_scheduler_encodes_request_rejected_by_llm_capacity():
    base_scheduler = _BaseScheduler()
    base_scheduler.capacity_scheduler = _RejectMultimodalCapacityScheduler()
    scheduler = MultimodalEagerEncoderScheduler(base_scheduler, max_num_items=1, max_num_tokens=8)
    multimodal_request = _request(1, [8])
    text_request = _llm_request(2)
    initialize_multimodal_encoder_request(text_request, max_num_tokens=8)

    output = scheduler.schedule_request([multimodal_request, text_request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [text_request]


def test_forward_multimodal_encoder_step_delegates_to_model_engine():
    calls = []
    executor = object.__new__(PyExecutor)
    executor.active_requests = [SimpleNamespace(request_id=1)]
    executor.model_engine = SimpleNamespace(
        forward_multimodal_encoder_items=lambda requests, items: calls.append((requests, items))
    )
    scheduled_items = {1: [0]}
    scheduled_requests = SimpleNamespace(scheduled_mm_encoder_items=scheduled_items)

    executor._forward_multimodal_encoder_step(scheduled_requests)

    assert calls == [(executor.active_requests, scheduled_items)]


def _executor_for_mm_admission(active_requests, *, max_num_tokens=8):
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(tp_size=1)
    executor.max_num_active_requests = 8
    executor.is_benchmark_disagg = False
    executor._supports_mm_encoder_item_scheduling = True
    executor.model_engine = SimpleNamespace(
        encoder_max_num_items=8,
        encoder_max_num_tokens=max_num_tokens,
    )
    executor.active_requests = active_requests
    return executor


def _waiting_item(request_id, costs=None):
    multimodal_data = None
    if costs is not None:
        multimodal_data = {
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("image", item_idx) for item_idx in range(len(costs))],
                encoder_token_lengths=costs,
                output_embedding_lengths=[1] * len(costs),
            ),
            "multimodal_embedding_lengths": [1] * len(costs),
        }
    return RequestQueueItem(
        request_id,
        _llm_request(request_id, multimodal_data=multimodal_data),
    )


def test_mm_admission_does_not_charge_ready_active_request():
    active = _request(1, [8], ready=(0,))
    waiting = FCFSWaitingQueue([_waiting_item(2, [8])])
    executor = _executor_for_mm_admission([active])

    admitted = executor._pop_from_waiting_queue(waiting, 1)

    assert [item.id for item in admitted] == [2]
    assert not waiting


def test_mm_admission_passes_oversized_request_to_validation():
    waiting = FCFSWaitingQueue([_waiting_item(1, [9]), _waiting_item(2, None)])
    executor = _executor_for_mm_admission([], max_num_tokens=8)

    admitted = executor._pop_from_waiting_queue(waiting, 0)

    assert [item.id for item in admitted] == [1, 2]
    assert not waiting


def test_item_encoder_slices_and_restores_selected_item_order():
    class _Model(MultimodalModelMixin):
        def encode_multimodal_inputs(self, multimodal_params):
            return torch.cat(
                [param.multimodal_data["image"]["pixel_values"] for param in multimodal_params]
            )

    multimodal_param = MultimodalParams(
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
        }
    )

    model = _Model()
    encoder_inputs = model.prepare_multimodal_encoder_inputs(
        [(multimodal_param, 1), (multimodal_param, 0)]
    )
    outputs = model.forward_multimodal_encoder_items(encoder_inputs)

    assert [output.squeeze(1).tolist() for output in outputs] == [
        [2, 3, 4],
        [0, 1],
    ]


def test_prepare_multimodal_encoder_inputs_slices_before_device_transfer():
    multimodal_param = MultimodalParams(
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
        }
    )

    encoder_inputs = MultimodalModelMixin.prepare_multimodal_encoder_inputs(
        MultimodalModelMixin(), [(multimodal_param, 1)]
    )

    item_param, embedding_length, modality = encoder_inputs[0]
    assert modality == "image"
    assert embedding_length == 3
    assert item_param.multimodal_data["image"]["pixel_values"].squeeze(1).tolist() == [2, 3, 4]
    assert multimodal_param.multimodal_data["image"]["pixel_values"].shape[0] == 5


def test_prepare_multimodal_encoder_inputs_rejects_invalid_metadata_types():
    multimodal_param = MultimodalParams(
        multimodal_data={
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: ("image", 0),
            "multimodal_embedding_lengths": [1],
        }
    )

    with pytest.raises(TypeError, match="must be a MultimodalEncoderItemMetadata"):
        MultimodalModelMixin().prepare_multimodal_encoder_inputs([(multimodal_param, 0)])


def test_strip_mm_encoder_inputs_preserves_embedding_and_runtime_metadata():
    embedding = torch.empty(3, 4)
    mm_data = {
        "image": {"pixel_values": torch.empty(2, 3)},
        "video": {"pixel_values_videos": torch.empty(2, 3)},
        "multimodal_embedding": embedding,
        "multimodal_embed_mask_cumsum": torch.tensor([0, 1]),
    }

    strip_mm_encoder_inputs(mm_data)

    assert "image" not in mm_data
    assert "video" not in mm_data
    assert mm_data["multimodal_embedding"] is embedding
    assert "multimodal_embed_mask_cumsum" in mm_data


def test_item_outputs_fill_one_contiguous_buffer_and_release_raw_data(monkeypatch):
    class _Model(MultimodalModelMixin):
        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_length, _ in encoder_inputs
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    request = _llm_request(
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
    assert request.py_mm_encoder_state.output_offsets == [0, 2, 5]

    engine.forward_multimodal_encoder_items([request], {1: [0]})

    output_buffer = request.py_mm_encoder_state.output_buffer
    assert output_buffer.shape == (5, 2)
    assert request.py_mm_encoder_state.outputs[0].data_ptr() == output_buffer.data_ptr()
    assert request.py_mm_encoder_state.outputs[1] is None
    assert "image" in request.py_multimodal_data

    engine.forward_multimodal_encoder_items([request], {1: [1]})

    assert request.py_multimodal_data["multimodal_embedding"] is output_buffer
    assert (
        request.py_mm_encoder_state.outputs[1].untyped_storage().data_ptr()
        == output_buffer.untyped_storage().data_ptr()
    )
    assert output_buffer.tolist() == [[2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0]]
    assert "image" not in request.py_multimodal_data


def test_qwen_item_metadata_uses_prompt_order_and_pre_merger_costs():
    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=11,
        video_token_id=12,
        vision_start_token_id=10,
        vision_end_token_id=13,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    prompt_token_ids = [10, 12, 12, 13, 1, 10, 11, 11, 13]
    multimodal_data = {
        "image": {"image_grid_thw": torch.tensor([[1, 4, 4]])},
        "video": {"video_grid_thw": torch.tensor([[2, 4, 4]])},
    }

    metadata = processor.get_mm_encoder_item_metadata(prompt_token_ids, multimodal_data)

    assert isinstance(metadata, MultimodalEncoderItemMetadata)
    assert metadata.item_refs == [("video", 0), ("image", 0)]
    assert metadata.encoder_token_lengths == [32, 16]
    assert metadata.output_embedding_lengths == [8, 4]


def test_qwen_item_metadata_collapses_frame_spans_into_original_video():
    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=11,
        video_token_id=12,
        vision_start_token_id=10,
        vision_end_token_id=13,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    prompt_token_ids = [10, 12, 13, 100, 10, 12, 13]
    multimodal_data = {
        "video": {"video_grid_thw": torch.tensor([[2, 4, 4]])},
    }

    metadata = processor.get_mm_encoder_item_metadata(prompt_token_ids, multimodal_data)

    assert metadata.item_refs == [("video", 0)]
    assert metadata.encoder_token_lengths == [32]
    assert metadata.output_embedding_lengths == [8]


def test_mistral_item_metadata_separates_patch_and_embedding_units():
    processor = object.__new__(Mistral3InputProcessor)
    processor._vision_geometry = lambda: (14, 2, 3, 1024)

    metadata = processor.get_mm_encoder_item_metadata(
        [], {"image": {"image_sizes": [[28, 56], [56, 56]]}}
    )

    assert metadata.item_refs == [("image", 0), ("image", 1)]
    assert metadata.encoder_token_lengths == [8, 16]
    assert metadata.output_embedding_lengths == [2, 4]


# ---------------------------------------------------------------------------
# Encoder cache x item scheduling integration
# ---------------------------------------------------------------------------


def _cache_request(request_id, *, hashes, embedding_lengths, kwargs_hash="kw"):
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


def _cache_engine(cache, monkeypatch):
    class _Model(MultimodalModelMixin):
        def __init__(self):
            self.encoded_item_counts = []

        def _get_multimodal_encoder_cache(self):
            return cache

        def forward_multimodal_encoder_items(self, encoder_inputs):
            self.encoded_item_counts.append(len(encoder_inputs))
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_length, _ in encoder_inputs
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    engine.supports_mm_encoder_item_scheduling = True
    return engine


def _sweep_executor(engine, active_requests):
    executor = object.__new__(PyExecutor)
    executor.active_requests = active_requests
    executor.model_engine = engine
    executor._supports_mm_encoder_item_scheduling = True
    return executor


def test_item_cache_keys_pin_the_full_request_path_format():
    hashes = [[1, 2], [3, 4]]

    keys = MultimodalModelMixin.build_encoder_cache_item_keys(
        hashes, [("image", 0), ("video", 0)], [2, 3], "kw"
    )

    # Must stay identical to `_encoder_cache_item_key` / `_encoder_cache_keys`
    # so both encode paths hit each other's entries.
    assert keys == [("image", (1, 2), 2, "kw"), ("video", (3, 4), 3, "kw")]


def test_item_encode_writes_independent_cache_entries(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])

    engine.forward_multimodal_encoder_items([request], {1: [0, 1]})

    key0, key1 = MultimodalModelMixin.build_encoder_cache_item_keys(
        [[1, 2], [3, 4]], [("image", 0), ("image", 1)], [2, 3], "kw"
    )
    assert torch.equal(cache.get(key0), request.py_mm_encoder_state.outputs[0])
    assert torch.equal(cache.get(key1), request.py_mm_encoder_state.outputs[1])
    # Entries are clones: they must not alias the request's shared buffer.
    buffer_storage_ptr = request.py_mm_encoder_state.output_buffer.untyped_storage().data_ptr()
    assert cache.get(key0).untyped_storage().data_ptr() != buffer_storage_ptr
    assert cache.get(key1).untyped_storage().data_ptr() != buffer_storage_ptr


def test_sweep_completes_duplicate_request_without_encoding_or_budget(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    first = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    engine.forward_multimodal_encoder_items(
        [
            first,
        ],
        {1: [0, 1]},
    )
    assert engine.model.encoded_item_counts == [2]

    second = _cache_request(2, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    executor = _sweep_executor(engine, [second])
    executor._attach_mm_encoder_cache_hits()

    assert engine.model.encoded_item_counts == [2]  # no further encoding
    assert is_multimodal_encoder_ready(second)
    assert (
        second.py_multimodal_data["multimodal_embedding"]
        is second.py_mm_encoder_state.output_buffer
    )
    assert "image" not in second.py_multimodal_data

    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=8, max_num_tokens=1 << 20)
    output = scheduler.schedule_request([second], set())
    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [second]


def test_partial_hit_routes_to_item_path_and_schedules_only_misses(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    key0, _ = MultimodalModelMixin.build_encoder_cache_item_keys(
        [[1, 2], [3, 4]], [("image", 0), ("image", 1)], [2, 3], "kw"
    )
    cache.put(key0, torch.full((2, 2), 7.0))

    executor = _sweep_executor(engine, [request])
    executor._attach_mm_encoder_cache_hits()

    assert request.py_mm_encoder_state.progress is MultimodalEncoderProgress.PARTIAL

    # A PARTIAL request must take the item path even though the whole batch
    # would fit the budgets, and only the miss is scheduled.
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=8, max_num_tokens=1 << 20)
    output = scheduler.schedule_request([request], set())
    assert output.scheduled_mm_encoder_items == {1: [1]}

    engine.forward_multimodal_encoder_items([request], {1: [1]})
    assert engine.model.encoded_item_counts == [1]
    assert is_multimodal_encoder_ready(request)


def test_eviction_after_commit_does_not_regress_progress(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2]], embedding_lengths=[2])
    engine.forward_multimodal_encoder_items([request], {1: [0]})

    cache.clear()

    assert is_multimodal_encoder_ready(request)
    assert request.py_multimodal_data["multimodal_embedding"].shape == (2, 2)


@pytest.mark.parametrize(
    "case",
    ["no_hashes", "no_kwargs_hash", "count_mismatch", "no_cache", "flag_off"],
)
def test_cache_context_guards_disable_cache_per_request(case, monkeypatch):
    cache = None if case == "no_cache" else TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    if case == "flag_off":
        engine.supports_mm_encoder_item_scheduling = False
    request = _cache_request(
        1,
        hashes=None
        if case == "no_hashes"
        else ([[1, 2]] if case == "count_mismatch" else [[1, 2], [3, 4]]),
        embedding_lengths=[2, 3],
        kwargs_hash=None if case == "no_kwargs_hash" else "kw",
    )

    assert engine.get_mm_encoder_cache_and_keys(request) is None

    # The request still runs the item path exactly as without a cache.
    engine.forward_multimodal_encoder_items([request], {1: [0, 1]})
    assert is_multimodal_encoder_ready(request)


# ---------------------------------------------------------------------------
# MultimodalEncoderRequestState unit behavior
# ---------------------------------------------------------------------------


def test_mm_encoder_state_enforces_offsets_slot_invariant():
    with pytest.raises(ValueError, match="one more entry"):
        MultimodalEncoderRequestState(output_offsets=[0, 2], outputs=[None, None])


def test_mm_encoder_state_progress_and_pending_transitions():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])

    assert state.progress is MultimodalEncoderProgress.PENDING
    assert state.pending_item_indices() == [0, 1]

    state.fill(1, torch.ones(3, 2))
    assert state.progress is MultimodalEncoderProgress.PARTIAL
    assert state.pending_item_indices() == [0]

    state.fill(0, torch.zeros(2, 2))
    assert state.progress is MultimodalEncoderProgress.READY
    assert state.output_buffer.tolist() == [[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]


def test_mm_encoder_state_fill_rejects_mismatched_outputs():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])

    with pytest.raises(ValueError, match="expected 2"):
        state.fill(0, torch.ones(5, 2))

    state.fill(0, torch.ones(2, 2))
    with pytest.raises(ValueError, match="matching"):
        state.fill(1, torch.ones(3, 4))  # hidden dim mismatch vs buffer


def test_mm_encoder_state_finalize_into_is_a_conditional_no_op():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2])
    multimodal_data = {"image": {"pixel_values": torch.empty(2, 1)}}

    assert state.finalize_into(multimodal_data) is False
    assert "multimodal_embedding" not in multimodal_data

    state.fill(0, torch.ones(2, 2))
    assert state.finalize_into(multimodal_data) is True
    assert multimodal_data["multimodal_embedding"] is state.output_buffer
    assert "image" not in multimodal_data
