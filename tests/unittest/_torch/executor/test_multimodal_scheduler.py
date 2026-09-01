# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
)
from tensorrt_llm._torch.pyexecutor.engine.multimodal import (
    MultimodalItemScheduler,
    resolve_mm_encoder_output_budget,
    validate_mm_encoder_scheduling_compatibility,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    MultimodalEncoderProgress,
    MultimodalEncoderRequestError,
    MultimodalEncoderRequestState,
    get_multimodal_encoder_token_lengths,
    initialize_multimodal_encoder_request,
    is_multimodal_encoder_ready,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    MultimodalEagerEncoderScheduler,
    MultimodalScheduler,
    ScheduledRequests,
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


def _bare_mm_item_scheduler(model, input_processor=None):
    """A scheduler with no budget resolution -- the engine only builds one when item
    scheduling is engaged, so these tests skip `create()` and exercise the item path."""
    return MultimodalItemScheduler(model=model, input_processor=input_processor)


def _llm_request(request_id, multimodal_data=None):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        py_multimodal_data=multimodal_data,
    )


def _record(state, item_idx, *, hidden=1, fill=0.0):
    """Write one item the way the encoder step does, sized from its declaration."""
    state.record(
        item_idx,
        torch.full((state.embedding_lengths[item_idx], hidden), fill),
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
        _record(request.py_mm_encoder_state, item_idx)
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

    _record(request.py_mm_encoder_state, 0)
    assert request.py_mm_encoder_state.progress is MultimodalEncoderProgress.PARTIAL
    assert not is_multimodal_encoder_ready(request)

    _record(request.py_mm_encoder_state, 1)
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
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=2, max_num_tokens=10)
    first = _request(1, [7, 7])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0], 2: [0]}
    assert output.context_requests == [second]


def test_scheduler_defers_items_beyond_output_byte_budget():
    # Budget hosts exactly one 1-row item (4 bytes): the second request's
    # item must wait even though the token budget would admit it
    # (allocate-before-compute).
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_batch_size=8,
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
        max_batch_size=8,
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


def test_started_request_holds_its_whole_footprint_across_iterations():
    # The token budget splits the head request across iterations, but its
    # first item already allocates storage for all of them, so the bytes it
    # still needs are charged from the start. A request behind it cannot
    # squat that space and leave the head unable to finish.
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_batch_size=8,
        max_num_tokens=5,
        output_budget_bytes=8,
        bytes_per_encoder_embedding=4,
    )
    head = _request(1, [5, 5])  # second item exceeds this iteration's tokens
    follower = _request(2, [3])

    output = scheduler.schedule_request([head, follower], set())

    # 8-byte budget: the head charges all 8 (both of its 1-row items) when it
    # starts, leaving nothing for the follower even though only item 0 is
    # encoded this iteration.
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
    with pytest.raises(ValueError, match="raise encoder_max_num_tokens") as exc_info:
        initialize_multimodal_encoder_request(
            request,
            max_num_tokens=1 << 30,
            max_output_bytes=2 * 4,  # fits 2 rows; the video needs 3
            bytes_per_encoder_embedding=4,
        )
    assert "Multimodal request 1" in str(exc_info.value)
    assert "effective encoder_max_num_tokens is 1073741824" in str(exc_info.value)


def test_oversized_request_fails_fast_instead_of_starving():
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_batch_size=8,
        max_num_tokens=1 << 20,
        output_budget_bytes=4,
        bytes_per_encoder_embedding=4,
    )
    request = _request(1, [3, 3])  # 2 rows = 8 bytes > 4-byte budget

    with pytest.raises(RuntimeError, match="raise encoder_max_num_tokens") as exc_info:
        scheduler.schedule_request([request], set())
    assert "Multimodal request 1" in str(exc_info.value)
    assert "effective encoder_max_num_tokens is 1048576" in str(exc_info.value)


def test_scheduler_requires_bytes_per_embedding_alongside_budget():
    with pytest.raises(ValueError, match="bytes_per_encoder_embedding"):
        MultimodalScheduler(
            _BaseScheduler(),
            max_batch_size=1,
            max_num_tokens=1,
            output_budget_bytes=4,
        )


def test_multimodal_scheduler_selects_all_items_and_admits_request_when_batch_fits():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=2, max_num_tokens=10)
    request = _request(1, [6, 4])

    output = scheduler.schedule_request([request], set())

    # The encoder step is the single encode site: an in-budget batch simply
    # has every pending item selected, and the request still enters the LLM
    # batch in the same iteration (encode runs before the LLM forward).
    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == [request]


def test_multimodal_scheduler_respects_encoder_batch_size():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=2, max_num_tokens=4)
    request = _request(1, [1, 1, 1, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []


def test_multimodal_scheduler_withholds_request_on_budget_overflow():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=3, max_num_tokens=10)
    request = _request(1, [6, 4, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []


def test_multimodal_scheduler_preserves_non_multimodal_requests():
    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=1, max_num_tokens=1)
    request = _llm_request(1)
    initialize_multimodal_encoder_request(request, max_num_tokens=1)

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]


def test_qwen3_output_budget_uses_post_merge_embedding_capacity():
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    # 16384 rows of fp16 -> 32768 bytes per embedding, via the mixin's explicit
    # embedding_dim/embedding_dtype contract.
    model = SimpleNamespace(embedding_dim=16384, embedding_dtype=torch.float16)

    budget, bytes_per_embedding = resolve_mm_encoder_output_budget(processor, 65536, model)

    assert bytes_per_embedding == 32768
    assert budget == 512 * 1024**2


def test_output_budget_requires_processor_embedding_capacity():
    processor = SimpleNamespace(get_max_mm_encoder_output_embeddings=lambda *_: None)

    # The embedding capacity is validated before the model is consulted, so
    # `model=None` never gets there.
    with pytest.raises(ValueError, match="get_max_mm_encoder_output_embeddings"):
        resolve_mm_encoder_output_budget(processor, 65536, None)


def test_eager_compatibility_is_checked_only_for_item_scheduled_models():
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


def test_side_stream_compatibility_is_checked_only_for_item_scheduled_models():
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


def test_request_rejects_item_above_effective_startup_maximum():
    request = _request(1, [9])
    request.py_multimodal_data["image"] = {"pixel_values": torch.empty(1)}

    with pytest.raises(ValueError, match="exceeding the effective startup maximum 8"):
        initialize_multimodal_encoder_request(request, max_num_tokens=8)


def test_eager_scheduler_encodes_request_rejected_by_llm_capacity():
    base_scheduler = _BaseScheduler()
    base_scheduler.capacity_scheduler = _RejectMultimodalCapacityScheduler()
    scheduler = MultimodalEagerEncoderScheduler(base_scheduler, max_batch_size=1, max_num_tokens=8)
    multimodal_request = _request(1, [8])
    text_request = _llm_request(2)
    initialize_multimodal_encoder_request(text_request, max_num_tokens=8)

    output = scheduler.schedule_request([multimodal_request, text_request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [text_request]


def test_forward_multimodal_encoder_step_scopes_failure_to_item_owners():
    failed = _request(1, [4])
    unrelated_context = _llm_request(2)
    unrelated_generation = _llm_request(3)
    handled = []

    def fail_encoder(*_):
        raise MultimodalEncoderRequestError("bad MM output")

    executor = object.__new__(PyExecutor)
    executor.active_requests = [failed, unrelated_context, unrelated_generation]
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(world_size=1)
    executor.model_engine = SimpleNamespace(forward_multimodal_encoder_items=fail_encoder)
    executor._handle_errors = lambda error_msg, **kwargs: handled.append((error_msg, kwargs))

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests([failed, unrelated_context])
    scheduled_requests.append_generation_request(unrelated_generation)
    scheduled_requests.scheduled_mm_encoder_items = {failed.request_id: [0]}

    executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [unrelated_context]
    assert scheduled_requests.generation_requests == [unrelated_generation]
    assert scheduled_requests.scheduled_mm_encoder_items is None
    assert handled == [("bad MM output", {"requests": [failed], "charge_budget": False})]


def test_forward_multimodal_encoder_step_contains_model_contract_error():
    failed = _request(1, [4])
    failed.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = ("image", 0)
    unrelated = _llm_request(2)
    handled = []

    engine = SimpleNamespace(
        forward_multimodal_encoder_items=_bare_mm_item_scheduler(
            MultimodalModelMixin()
        ).forward_items
    )

    executor = object.__new__(PyExecutor)
    executor.active_requests = [failed, unrelated]
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(world_size=1)
    executor.model_engine = engine
    executor._handle_errors = lambda error_msg, **kwargs: handled.append((error_msg, kwargs))

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests([failed, unrelated])
    scheduled_requests.scheduled_mm_encoder_items = {failed.request_id: [0]}

    executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [unrelated]
    assert scheduled_requests.scheduled_mm_encoder_items is None
    assert len(handled) == 1
    assert "must be a MultimodalEncoderItemMetadata" in handled[0][0]
    assert handled[0][1] == {"requests": [failed], "charge_budget": False}


def test_forward_multimodal_encoder_step_contains_stale_schedule():
    unrelated = _llm_request(2)
    handled = []

    engine = SimpleNamespace(
        forward_multimodal_encoder_items=_bare_mm_item_scheduler(
            MultimodalModelMixin()
        ).forward_items
    )

    executor = object.__new__(PyExecutor)
    executor.active_requests = [unrelated]
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(world_size=1)
    executor.model_engine = engine
    executor._handle_errors = lambda error_msg, **kwargs: handled.append((error_msg, kwargs))

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests([unrelated])
    scheduled_requests.scheduled_mm_encoder_items = {1: [0]}

    executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [unrelated]
    assert scheduled_requests.scheduled_mm_encoder_items is None
    assert handled == [
        (
            "Scheduled MM request 1 is no longer active",
            {
                "requests": [],
                "charge_budget": False,
            },
        )
    ]


def test_forward_multimodal_encoder_step_propagates_system_errors():
    failed = _request(1, [4])

    def fail_encoder(*_):
        raise torch.cuda.OutOfMemoryError("encoder OOM")

    executor = object.__new__(PyExecutor)
    executor.active_requests = [failed]
    executor.model_engine = SimpleNamespace(forward_multimodal_encoder_items=fail_encoder)

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests([failed])
    scheduled_requests.scheduled_mm_encoder_items = {failed.request_id: [0]}

    with pytest.raises(torch.cuda.OutOfMemoryError, match="encoder OOM"):
        executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [failed]
    assert scheduled_requests.scheduled_mm_encoder_items == {failed.request_id: [0]}


def _executor_for_mm_admission(active_requests, *, max_batch_size=8, max_num_tokens=8):
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = False
    executor.dist = SimpleNamespace(tp_size=1)
    executor.max_num_active_requests = 8
    executor.is_benchmark_disagg = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.model_engine = SimpleNamespace(
        encoder_batch_size=max_batch_size,
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


def test_mm_admission_uses_active_request_state_snapshot():
    active = _request(1, [4])
    active.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = object()
    waiting = FCFSWaitingQueue([_waiting_item(2, [4])])
    executor = _executor_for_mm_admission([active])

    admitted = executor._pop_from_waiting_queue(waiting, 1)

    assert [item.id for item in admitted] == [2]
    assert not waiting


def test_mm_admission_passes_malformed_metadata_to_validation():
    malformed = _waiting_item(1, [4])
    malformed.request.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = object()
    waiting = FCFSWaitingQueue([malformed, _waiting_item(2, [8])])
    executor = _executor_for_mm_admission([])

    admitted = executor._pop_from_waiting_queue(waiting, 0)

    assert [item.id for item in admitted] == [1, 2]
    assert not waiting


def test_mm_admission_respects_encoder_batch_size():
    waiting = FCFSWaitingQueue([_waiting_item(1, [1, 1]), _waiting_item(2, [1])])
    executor = _executor_for_mm_admission([], max_batch_size=1)

    admitted = executor._pop_from_waiting_queue(waiting, 0)

    assert [item.id for item in admitted] == [1]
    assert [item.id for item in waiting] == [2]


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

    item_param, embedding_lengths, modality = encoder_inputs[0]
    assert modality == "image"
    assert embedding_lengths == [3]
    assert item_param.multimodal_data["image"]["pixel_values"].squeeze(1).tolist() == [2, 3, 4]
    assert multimodal_param.multimodal_data["image"]["pixel_values"].shape[0] == 5


def test_prepare_multimodal_encoder_inputs_rejects_invalid_metadata_types():
    multimodal_param = MultimodalParams(
        multimodal_data={
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: ("image", 0),
            "multimodal_embedding_lengths": [1],
        }
    )

    with pytest.raises(
        MultimodalEncoderContractError, match="must be a MultimodalEncoderItemMetadata"
    ):
        MultimodalModelMixin().prepare_multimodal_encoder_inputs([(multimodal_param, 0)])


def test_item_encoder_classifies_request_state_contract_errors():
    class _Model(MultimodalModelMixin):
        pass

    mm_item_scheduler = _bare_mm_item_scheduler(_Model())
    request = _llm_request(1)

    with pytest.raises(MultimodalEncoderRequestError, match="no longer active"):
        mm_item_scheduler.forward_items([], {request.request_id: [0]})

    with pytest.raises(MultimodalEncoderRequestError, match="no encoder item state"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


def test_item_encoder_classifies_output_count_contract_error():
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            return []

    mm_item_scheduler = _bare_mm_item_scheduler(_Model())
    request = _request(1, [4])

    with pytest.raises(MultimodalEncoderRequestError, match="one output per item"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


@pytest.mark.parametrize("failure_stage", ["prepare", "forward"])
def test_item_encoder_translates_model_contract_errors(failure_stage):
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            if failure_stage == "prepare":
                raise MultimodalEncoderContractError("bad request metadata")
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            raise MultimodalEncoderContractError("bad encoder output rows")

    mm_item_scheduler = _bare_mm_item_scheduler(_Model())
    request = _request(1, [4])

    expected = "bad request metadata" if failure_stage == "prepare" else "bad encoder output rows"
    with pytest.raises(MultimodalEncoderRequestError, match=expected):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


@pytest.mark.parametrize("failure_stage", ["prepare", "forward"])
def test_item_encoder_does_not_translate_system_errors(failure_stage):
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(self, _):
            if failure_stage == "prepare":
                raise torch.cuda.OutOfMemoryError("encoder OOM")
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(self, _):
            raise torch.cuda.OutOfMemoryError("encoder OOM")

    mm_item_scheduler = _bare_mm_item_scheduler(_Model())
    request = _request(1, [4])

    with pytest.raises(torch.cuda.OutOfMemoryError, match="encoder OOM"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


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


def test_terminate_request_releases_partial_multimodal_encoder_state():
    request = _request(1, [4, 4])
    _record(request.py_mm_encoder_state, 0)
    freed = []

    executor = object.__new__(PyExecutor)
    executor.resource_manager = SimpleNamespace(free_resources=freed.append)
    executor._prefetched_request_ids = {request.py_request_id}
    executor._disagg_timed_out_ctx_cancelled_ids = {request.py_request_id}
    executor._disagg_timed_out_gen_cancelled_ids = {request.py_request_id}
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=1)

    executor._do_terminate_request(request)

    assert freed == [request]
    assert request.py_mm_encoder_state is None
    assert request.py_multimodal_data == {}
    assert executor._prefetched_request_ids == set()
    assert executor._disagg_timed_out_ctx_cancelled_ids == set()
    assert executor._disagg_timed_out_gen_cancelled_ids == set()


def test_item_outputs_accumulate_on_request_and_release_raw_data(monkeypatch):
    class _Model(MultimodalModelMixin):
        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_lengths, _ in encoder_inputs
                for embedding_length in embedding_lengths
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    mm_item_scheduler = _bare_mm_item_scheduler(_Model())
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


def _cache_mm_item_scheduler(cache, monkeypatch, *, supports_encoder_cache=True):
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
    return _bare_mm_item_scheduler(model)


def test_duplicate_request_hits_cache_and_skips_encoding(monkeypatch):
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


def test_cache_hit_at_encode_skips_only_hit_items(monkeypatch):
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


def test_cache_eviction_leaves_recorded_slots_intact(monkeypatch):
    cache = TensorLRUCache(2 * 2 * 4, name="test")  # holds exactly one 2-row item
    mm_item_scheduler = _cache_mm_item_scheduler(cache, monkeypatch)
    request = _cache_request(1, hashes=[[1, 2]], embedding_lengths=[2])
    mm_item_scheduler.forward_items([request], {1: [0]})
    recorded = request.py_multimodal_data["multimodal_embedding"][:2]
    assert len(cache) == 1

    cache.clear()  # simulate eviction of the entry the request came from

    assert torch.equal(recorded, torch.full((2, 2), 2.0))  # owned clone, untouched
    assert is_multimodal_encoder_ready(request)


def test_cache_off_encodes_every_item_without_touching_cache(monkeypatch):
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
def test_key_guards_bypass_cache(case, monkeypatch):
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


# ---------------------------------------------------------------------------
# MultimodalEncoderRequestState unit behavior
# ---------------------------------------------------------------------------


def test_mm_encoder_state_enforces_lengths_slot_invariant():
    with pytest.raises(ValueError, match="one entry per item slot"):
        MultimodalEncoderRequestState(
            embedding_lengths=[2], encoder_token_lengths=[4], recorded=[False, False]
        )


def test_mm_encoder_state_copies_validated_scheduler_costs_at_admission():
    request = _request(1, [4, 7])

    assert request.py_mm_encoder_state.encoder_token_lengths == [4, 7]

    metadata = request.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    metadata.encoder_token_lengths[0] = 100
    assert request.py_mm_encoder_state.encoder_token_lengths == [4, 7]

    scheduler = MultimodalScheduler(_BaseScheduler(), max_batch_size=2, max_num_tokens=11)
    output = scheduler.schedule_request([request], set())
    assert output.scheduled_mm_encoder_items == {1: [0, 1]}


def test_mm_encoder_state_progress_and_pending_transitions():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])

    assert state.progress is MultimodalEncoderProgress.PENDING
    assert state.pending_item_indices() == [0, 1]

    state.record(1, torch.ones(3, 2))
    assert state.progress is MultimodalEncoderProgress.PARTIAL
    assert state.pending_item_indices() == [0]

    state.record(0, torch.zeros(2, 2))
    assert state.progress is MultimodalEncoderProgress.READY
    # Items land in prompt order in their own row ranges of one buffer.
    assert state.embeddings.tolist() == [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
    ]


def test_mm_encoder_state_record_rejects_mismatched_outputs():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])

    with pytest.raises(MultimodalEncoderRequestError, match="expected 2"):
        state.record(0, torch.ones(5, 2))

    state.record(0, torch.ones(2, 2))
    with pytest.raises(MultimodalEncoderRequestError, match="matching"):
        state.record(1, torch.ones(3, 4))  # hidden dim mismatch vs items
    with pytest.raises(MultimodalEncoderRequestError, match="already recorded"):
        state.record(0, torch.ones(2, 2))  # items encode at most once


def test_mm_encoder_state_record_copies_into_owned_storage():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2])
    batch = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    view = batch[1:3]  # a view into a larger batched encoder output

    state.record(0, view)

    assert torch.equal(state.embeddings, view)
    # The buffer neither aliases the batch storage (which a view would pin in
    # full) nor can be invalidated by whoever owns the source tensor, and it
    # is sized for this request alone.
    assert state.embeddings.untyped_storage().data_ptr() != batch.untyped_storage().data_ptr()
    assert state.embeddings.untyped_storage().nbytes() == 2 * 2 * 4


def test_mm_encoder_state_charges_the_whole_request_from_its_first_item():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])
    assert state.resident_output_bytes(4) == 0
    assert not state.has_storage

    # The first item allocates storage for every item, so the charge is the
    # full footprint immediately -- which is what the request occupies.
    state.record(1, torch.ones(3, 2))
    assert state.has_storage
    assert state.resident_output_bytes(4) == (2 + 3) * 4

    state.record(0, torch.ones(2, 2))
    assert state.resident_output_bytes(4) == (2 + 3) * 4


def test_mm_encoder_state_finalize_is_a_conditional_no_op():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2])
    multimodal_data = {"image": {"pixel_values": torch.empty(2, 1)}}

    assert state.finalize(multimodal_data) is False
    assert "multimodal_embedding" not in multimodal_data

    state.record(0, torch.ones(2, 2))
    assert state.finalize(multimodal_data) is True
    assert multimodal_data["multimodal_embedding"] is state.embeddings
    assert "image" not in multimodal_data


def test_mm_encoder_state_publishes_the_buffer_without_copying():
    """Publishing must hand over the buffer itself, not a second materialization.

    The buffer is already the contiguous form the prefill path consumes, so a
    copy here (or a per-item list that prefill has to concatenate) would put a
    second full copy of the request's embeddings on the device that the byte
    budget does not account for.
    """
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])
    multimodal_data = {"image": {"pixel_values": torch.empty(2, 1)}}
    state.record(0, torch.ones(2, 2))
    state.record(1, torch.ones(3, 2))
    buffer_ptr = state.embeddings.untyped_storage().data_ptr()

    assert state.finalize(multimodal_data) is True

    published = multimodal_data["multimodal_embedding"]
    assert published is state.embeddings
    assert published.untyped_storage().data_ptr() == buffer_ptr
    assert published.shape == (2 + 3, 2)
    # Readiness and byte accounting are unchanged by publishing: the rows stay
    # resident until the request is stripped.
    assert state.progress is MultimodalEncoderProgress.READY
    assert state.pending_item_indices() == []
    assert state.resident_output_bytes(4) == (2 + 3) * 4
