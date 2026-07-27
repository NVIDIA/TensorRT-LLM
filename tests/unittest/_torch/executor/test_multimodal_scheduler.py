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
    _validate_mm_encoder_scheduling_compatibility,
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
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderSchedulingPolicy


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


def test_item_scheduling_rejects_raw_payload_without_item_metadata():
    request = _llm_request(
        1,
        multimodal_data={"image": {"pixel_values": torch.empty(1, 1)}},
    )

    with pytest.raises(ValueError, match="requires multimodal_encoder_item_metadata"):
        initialize_multimodal_encoder_request(request, max_num_tokens=8)


def test_multimodal_scheduler_keeps_items_atomic_and_backfills_requests():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_num_items=2, max_num_tokens=10)
    first = _request(1, [7, 7])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0], 2: [0]}
    assert output.context_requests == [second]


def test_scheduler_defers_items_beyond_output_byte_budget():
    # Budget hosts exactly one 1-row item (4 bytes): the second request's
    # item must wait even though the token/item budgets would admit it
    # (allocate-before-compute).
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_num_items=8,
        max_num_tokens=1 << 20,
        output_budget_bytes=4,
        bytes_per_encoder_embedding=4,
    )
    first = _request(1, [3])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [first]


def test_resident_outputs_of_active_requests_block_new_admissions():
    # A request that already holds recorded-but-unconsumed outputs (e.g.
    # mid-chunked-prefill) occupies the budget purely through its live
    # state — no counter, no release call — deferring new encoder work.
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_num_items=8,
        max_num_tokens=1 << 20,
        output_budget_bytes=4,
        bytes_per_encoder_embedding=4,
    )
    holder = _request(1, [3], ready=(0,))  # 1 row resident = full budget
    newcomer = _request(2, [3])

    output = scheduler.schedule_request([holder, newcomer], set())
    assert output.scheduled_mm_encoder_items is None

    # Consumption (post-prefill strip clears the state) frees the budget on
    # the next pass with no further bookkeeping — same for an aborted
    # request, which simply leaves the active list.
    holder.py_mm_encoder_state = None
    output = scheduler.schedule_request([holder, newcomer], set())
    assert output.scheduled_mm_encoder_items == {2: [0]}


def test_head_of_line_reservation_blocks_later_requests():
    # Token budget splits the head request across iterations; its unencoded
    # remainder reserves budget bytes, so the request behind it cannot
    # squat the space the head needs to ever complete (deadlock avoidance).
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_num_items=8,
        max_num_tokens=5,
        output_budget_bytes=8,
        bytes_per_encoder_embedding=4,
    )
    head = _request(1, [5, 5])  # second item exceeds this iteration's tokens
    follower = _request(2, [3])

    output = scheduler.schedule_request([head, follower], set())

    # 8-byte budget: head's item 0 claims 4, its pending item 1 reserves the
    # other 4, leaving nothing for the follower despite free space.
    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == []


def test_admission_rejects_requests_larger_than_output_budget():
    # A long-video request whose total embedding footprint can never fit
    # the output budget fails at admission (failing only that request),
    # with guidance to raise encoder_max_num_tokens. Reachable once LLM
    # chunked prefill admits prompts longer than max_num_tokens.
    request = _llm_request(
        1,
        multimodal_data={
            "video": {"pixel_values_videos": torch.empty(3, 1)},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalEncoderItemMetadata(
                item_refs=[("video", 0)],
                encoder_token_lengths=[12],
                output_embedding_lengths=[3],
            ),
            "multimodal_embedding_lengths": [3],
        },
    )
    with pytest.raises(ValueError, match="raise encoder_max_num_tokens"):
        initialize_multimodal_encoder_request(
            request,
            max_num_tokens=1 << 30,
            max_output_bytes=2 * 4,  # fits 2 rows; the video needs 3
            bytes_per_encoder_embedding=4,
        )


def test_oversized_request_fails_fast_instead_of_starving():
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_num_items=8,
        max_num_tokens=1 << 20,
        output_budget_bytes=4,
        bytes_per_encoder_embedding=4,
    )
    request = _request(1, [3, 3])  # 2 rows = 8 bytes > 4-byte budget

    with pytest.raises(RuntimeError, match="raise encoder_max_num_tokens"):
        scheduler.schedule_request([request], set())


def test_scheduler_requires_bytes_per_embedding_alongside_budget():
    with pytest.raises(ValueError, match="bytes_per_encoder_embedding"):
        MultimodalScheduler(
            _BaseScheduler(),
            max_num_items=1,
            max_num_tokens=1,
            output_budget_bytes=4,
        )


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


def test_qwen_output_budget_uses_post_merge_embedding_capacity():
    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    engine = object.__new__(PyTorchModelEngine)
    engine.max_num_tokens = 8192
    engine.encoder_max_num_items = 8
    engine.encoder_max_num_tokens = 65536
    engine.input_processor = processor
    engine._resolve_bytes_per_mm_encoder_embedding = lambda: 32768

    budget = engine._resolve_mm_encoder_output_budget_bytes()

    assert engine.max_mm_encoder_output_embeddings == 16384
    assert engine.bytes_per_mm_encoder_embedding == 32768
    assert budget == 512 * 1024**2


def test_output_budget_requires_processor_embedding_capacity():
    engine = object.__new__(PyTorchModelEngine)
    engine.encoder_max_num_items = 8
    engine.encoder_max_num_tokens = 65536
    engine.input_processor = SimpleNamespace(get_max_mm_encoder_output_embeddings=lambda *_: None)

    with pytest.raises(ValueError, match="get_max_mm_encoder_output_embeddings"):
        engine._resolve_mm_encoder_output_budget_bytes()


def test_eager_compatibility_is_checked_only_for_item_scheduled_models():
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER
        ),
        enable_attention_dp=True,
        cache_transceiver_config=SimpleNamespace(backend="NIXL"),
    )

    _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="attention DP"):
        _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)

    args.enable_attention_dp = False
    with pytest.raises(ValueError, match="disaggregated"):
        _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


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
    executor._mm_encoder_item_scheduling_enabled = True
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


def test_item_outputs_accumulate_on_request_and_release_raw_data(monkeypatch):
    class _Model(MultimodalModelMixin):
        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_length, _ in encoder_inputs
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    engine.mm_encoder_item_scheduling_enabled = True
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
    assert request.py_mm_encoder_state.embedding_lengths == [2, 3]

    engine.forward_multimodal_encoder_items([request], {1: [0]})

    # Items encoded across iterations accumulate as request-owned slots;
    # raw inputs stay until the request completes.
    assert request.py_mm_encoder_state.outputs[0].shape == (2, 2)
    assert request.py_mm_encoder_state.outputs[1] is None
    assert request.py_mm_encoder_state.resident_output_bytes(8) == 2 * 8
    assert "image" in request.py_multimodal_data

    engine.forward_multimodal_encoder_items([request], {1: [1]})

    published = request.py_multimodal_data["multimodal_embedding"]
    assert published == request.py_mm_encoder_state.outputs
    assert [slot.tolist() for slot in published] == [
        [[2.0, 2.0], [2.0, 2.0]],
        [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]],
    ]
    assert "image" not in request.py_multimodal_data


# ---------------------------------------------------------------------------
# Item-path read-through against the encoder cache (supports_encoder_cache)
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


def _cache_engine(cache, monkeypatch, *, supports_encoder_cache=True):
    class _Model(MultimodalModelMixin):
        supports_encoder_cache = False

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
    model = _Model()
    model.supports_encoder_cache = supports_encoder_cache
    engine.model = model
    engine.mm_encoder_item_scheduling_enabled = True
    return engine


def test_item_encode_populates_cache_with_independent_copies(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])

    engine.forward_multimodal_encoder_items([request], {1: [0, 1]})

    key0, key1 = MultimodalModelMixin.build_encoder_cache_item_keys(
        [[1, 2], [3, 4]], [("image", 0), ("image", 1)], [2, 3], "kw"
    )
    # The request records owned clones; the cache keeps its own copies —
    # eviction can never invalidate a recorded slot.
    assert torch.equal(cache.get(key0), request.py_mm_encoder_state.outputs[0])
    assert cache.get(key0) is not request.py_mm_encoder_state.outputs[0]
    assert torch.equal(cache.get(key1), request.py_mm_encoder_state.outputs[1])


def test_duplicate_request_hits_cache_and_skips_encoding(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    first = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    engine.forward_multimodal_encoder_items([first], {1: [0, 1]})
    assert engine.model.encoded_item_counts == [2]

    second = _cache_request(2, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    engine.forward_multimodal_encoder_items([second], {2: [0, 1]})

    # Read-through: every item hit, so the encoder never ran again, and each
    # request owns an independent copy (no cross-request aliasing).
    assert engine.model.encoded_item_counts == [2]
    assert is_multimodal_encoder_ready(second)
    published = second.py_multimodal_data["multimodal_embedding"]
    assert torch.equal(published[0], first.py_mm_encoder_state.outputs[0])
    assert published[0] is not first.py_mm_encoder_state.outputs[0]


def test_cache_hit_at_encode_skips_only_hit_items(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])
    key0, _ = MultimodalModelMixin.build_encoder_cache_item_keys(
        [[1, 2], [3, 4]], [("image", 0), ("image", 1)], [2, 3], "kw"
    )
    cache.put(key0, torch.full((2, 2), 7.0))  # entry from an earlier request
    engine.forward_multimodal_encoder_items([request], {1: [0, 1]})

    assert engine.model.encoded_item_counts == [1]  # only the miss encoded
    assert torch.equal(request.py_mm_encoder_state.outputs[0], torch.full((2, 2), 7.0))
    assert is_multimodal_encoder_ready(request)


def test_cache_eviction_leaves_recorded_slots_intact(monkeypatch):
    cache = TensorLRUCache(2 * 2 * 4, name="test")  # holds exactly one 2-row item
    engine = _cache_engine(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2]], embedding_lengths=[2])
    engine.forward_multimodal_encoder_items([request], {1: [0]})
    recorded = request.py_mm_encoder_state.outputs[0]

    cache.clear()  # simulate eviction of the entry the request came from

    assert torch.equal(recorded, torch.full((2, 2), 2.0))  # owned clone, untouched
    assert is_multimodal_encoder_ready(request)


def test_cache_off_encodes_every_item_without_touching_cache(monkeypatch):
    # supports_encoder_cache=False -> mm_encoder_cache is None -> pure encode.
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch, supports_encoder_cache=False)
    assert engine.mm_encoder_cache is None
    request = _cache_request(1, hashes=[[1, 2], [3, 4]], embedding_lengths=[2, 3])

    engine.forward_multimodal_encoder_items([request], {1: [0, 1]})

    assert engine.model.encoded_item_counts == [2]
    assert is_multimodal_encoder_ready(request)
    assert len(cache) == 0  # never populated


def test_item_cache_keys_share_the_full_request_path_format():
    keys = MultimodalModelMixin.build_encoder_cache_item_keys(
        [[1, 2], [3, 4]], [("image", 0), ("video", 0)], [2, 3], "kw"
    )
    assert keys == [("image", (1, 2), 2, "kw"), ("video", (3, 4), 3, "kw")]


@pytest.mark.parametrize("case", ["no_hashes", "no_kwargs_hash", "count_mismatch"])
def test_key_guards_bypass_cache(case, monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch)
    request = _cache_request(
        1,
        hashes=None
        if case == "no_hashes"
        else ([[1, 2]] if case == "count_mismatch" else [[1, 2], [3, 4]]),
        embedding_lengths=[2, 3],
        kwargs_hash=None if case == "no_kwargs_hash" else "kw",
    )

    assert engine.get_mm_encoder_item_keys(request) is None

    engine.forward_multimodal_encoder_items([request], {1: [0, 1]})
    assert is_multimodal_encoder_ready(request)
    assert len(cache) == 0  # unkeyable items never populate the cache


def test_qwen_get_mm_encoder_cache_is_none_without_supports_flag(monkeypatch):
    # Item-scheduled model that does not opt into the cache (e.g. Qwen) gets
    # no read-through even with a positive encoder_cache_max_bytes.
    cache = TensorLRUCache(1 << 20, name="test")
    engine = _cache_engine(cache, monkeypatch, supports_encoder_cache=False)
    assert engine.mm_encoder_cache is None


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
# MultimodalEncoderRequestState unit behavior
# ---------------------------------------------------------------------------


def test_mm_encoder_state_enforces_lengths_slot_invariant():
    with pytest.raises(ValueError, match="one entry per item slot"):
        MultimodalEncoderRequestState(embedding_lengths=[2], outputs=[None, None])


def test_mm_encoder_state_progress_and_pending_transitions():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])

    assert state.progress is MultimodalEncoderProgress.PENDING
    assert state.pending_item_indices() == [0, 1]

    state.record(1, torch.ones(3, 2))
    assert state.progress is MultimodalEncoderProgress.PARTIAL
    assert state.pending_item_indices() == [0]

    state.record(0, torch.zeros(2, 2))
    assert state.progress is MultimodalEncoderProgress.READY
    assert [slot.tolist() for slot in state.outputs] == [
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1], [1, 1]],
    ]


def test_mm_encoder_state_record_rejects_mismatched_outputs():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])

    with pytest.raises(ValueError, match="expected 2"):
        state.record(0, torch.ones(5, 2))

    state.record(0, torch.ones(2, 2))
    with pytest.raises(ValueError, match="matching"):
        state.record(1, torch.ones(3, 4))  # hidden dim mismatch vs items
    with pytest.raises(ValueError, match="already recorded"):
        state.record(0, torch.ones(2, 2))  # items encode at most once


def test_mm_encoder_state_record_takes_an_owned_clone():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2])
    batch = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    view = batch[1:3]  # a view into a larger batched encoder output

    state.record(0, view)

    recorded = state.outputs[0]
    assert torch.equal(recorded, view)
    # The clone neither aliases the batch storage (which a view would pin
    # in full) nor can be invalidated by whoever owns the source tensor.
    assert recorded.data_ptr() != view.data_ptr()
    assert recorded.untyped_storage().nbytes() == 2 * 2 * 4


def test_mm_encoder_state_resident_output_bytes_counts_recorded_slots():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])
    assert state.resident_output_bytes(4) == 0

    state.record(1, torch.ones(3, 2))
    assert state.resident_output_bytes(4) == 3 * 4

    state.record(0, torch.ones(2, 2))
    assert state.resident_output_bytes(4) == 5 * 4


def test_mm_encoder_state_finalize_into_is_a_conditional_no_op():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2])
    multimodal_data = {"image": {"pixel_values": torch.empty(2, 1)}}

    assert state.finalize_into(multimodal_data) is False
    assert "multimodal_embedding" not in multimodal_data

    state.record(0, torch.ones(2, 2))
    assert state.finalize_into(multimodal_data) is True
    assert multimodal_data["multimodal_embedding"] == state.outputs
    assert "image" not in multimodal_data
