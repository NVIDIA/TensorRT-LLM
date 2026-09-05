# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from _torch.executor.multimodal_utils import (
    bare_mm_item_scheduler,
    make_llm_request,
    make_mm_request,
    record_output,
)

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import (
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


def _item_cache_keys(request):
    state = request.py_mm_encoder_state
    return [("test_mm", request.request_id, item_idx) for item_idx in range(state.num_items)]


def _scheduler(
    *,
    max_batch_size,
    max_num_tokens,
    cache_capacity=1 << 20,
    base_scheduler=None,
    scheduler_cls=MultimodalScheduler,
):
    return scheduler_cls(
        base_scheduler or _BaseScheduler(),
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        encoder_cache=TensorLRUCache(cache_capacity),
        get_item_cache_keys=_item_cache_keys,
        bytes_per_encoder_embedding=4,
        retain_cache_entries=False,
    )


def test_mm_encoder_token_lengths_distinguishes_missing_and_invalid_data():
    request = make_llm_request(1)

    assert get_multimodal_encoder_token_lengths(request) is None

    request.py_multimodal_data = []
    with pytest.raises(TypeError, match="multimodal_data must be a dict"):
        get_multimodal_encoder_token_lengths(request)


def test_mm_encoder_readiness_is_derived_from_item_state():
    request = make_mm_request(1, [4, 4])
    assert request.py_mm_encoder_state.progress is MultimodalEncoderProgress.PENDING
    assert not is_multimodal_encoder_ready(request)

    record_output(request.py_mm_encoder_state, 0)
    assert request.py_mm_encoder_state.progress is MultimodalEncoderProgress.PARTIAL
    assert not is_multimodal_encoder_ready(request)

    record_output(request.py_mm_encoder_state, 1)
    assert is_multimodal_encoder_ready(request)

    # A precomputed-embedding request never gets item state in the first
    # place (initialize skips it), and post-prefill strip drops the state:
    # both report ready through the state-absence branch.
    request.py_mm_encoder_state = None
    assert is_multimodal_encoder_ready(request)


def test_item_scheduling_rejects_raw_payload_without_item_metadata():
    request = make_llm_request(
        1,
        multimodal_data={"image": {"pixel_values": torch.empty(1, 1)}},
    )

    with pytest.raises(ValueError, match="requires multimodal_encoder_item_metadata"):
        initialize_multimodal_encoder_request(request, max_num_tokens=8)


def test_multimodal_scheduler_keeps_items_atomic_and_backfills_requests():
    scheduler = _scheduler(max_batch_size=2, max_num_tokens=10)
    first = make_mm_request(1, [7, 7])
    second = make_mm_request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0], 2: [0]}
    assert output.context_requests == [second]


def test_multimodal_scheduler_encodes_shared_cache_key_once():
    cache = TensorLRUCache(8)
    scheduler = MultimodalScheduler(
        _BaseScheduler(),
        max_batch_size=2,
        max_num_tokens=8,
        encoder_cache=cache,
        get_item_cache_keys=lambda _request: [("stable", 0)],
        bytes_per_encoder_embedding=4,
        retain_cache_entries=True,
    )
    first = make_mm_request(1, [4])
    second = make_mm_request(2, [4])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {first.request_id: [0]}
    assert output.context_requests == [first, second]
    assert first.py_mm_encoder_state.item_cache_keys == second.py_mm_encoder_state.item_cache_keys
    assert cache.stats().inflight_deduplications == 1


def test_scheduler_defers_items_beyond_output_byte_budget():
    # Budget hosts exactly one 1-row item (4 bytes): the second request's
    # item must wait even though the token budget would admit it
    # (allocate-before-compute).
    scheduler = _scheduler(max_batch_size=8, max_num_tokens=1 << 20, cache_capacity=4)
    first = make_mm_request(1, [3])
    second = make_mm_request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [first]


def test_referenced_outputs_block_new_admissions_until_explicit_release():
    scheduler = _scheduler(max_batch_size=8, max_num_tokens=1 << 20, cache_capacity=4)
    holder = make_mm_request(1, [3])
    newcomer = make_mm_request(2, [3])

    first_output = scheduler.schedule_request([holder], set())
    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    holder_cache_key = holder.py_mm_encoder_state.item_cache_keys[0]
    assert holder_cache_key is not None
    scheduler.encoder_cache.commit(holder_cache_key, torch.ones(1, dtype=torch.float32))
    holder.py_mm_encoder_state.mark_cache_key_ready(holder_cache_key)

    output = scheduler.schedule_request([holder, newcomer], set())
    assert output.scheduled_mm_encoder_items is None

    drained = holder.py_mm_encoder_state.pop_all_cache_keys()
    assert drained == [holder_cache_key]
    scheduler.encoder_cache.release(holder_cache_key)
    assert scheduler.encoder_cache.get(holder_cache_key, record_stats=False) is None
    holder.py_mm_encoder_state = None
    output = scheduler.schedule_request([holder, newcomer], set())
    assert output.scheduled_mm_encoder_items == {2: [0]}


def test_started_request_holds_its_whole_footprint_across_iterations():
    # The token budget splits the head request across iterations, but its
    # first item already allocates storage for all of them, so the bytes it
    # still needs are charged from the start. A request behind it cannot
    # squat that space and leave the head unable to finish.
    scheduler = _scheduler(max_batch_size=8, max_num_tokens=5, cache_capacity=8)
    head = make_mm_request(1, [5, 5])  # second item exceeds this iteration's tokens
    follower = make_mm_request(2, [3])

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
    request = make_llm_request(
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


def test_multimodal_scheduler_selects_all_items_and_admits_request_when_batch_fits():
    scheduler = _scheduler(max_batch_size=2, max_num_tokens=10)
    request = make_mm_request(1, [6, 4])

    output = scheduler.schedule_request([request], set())

    # The encoder step is the single encode site: an in-budget batch simply
    # has every pending item selected, and the request still enters the LLM
    # batch in the same iteration (encode runs before the LLM forward).
    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == [request]


def test_multimodal_scheduler_respects_encoder_batch_size():
    scheduler = _scheduler(max_batch_size=2, max_num_tokens=4)
    request = make_mm_request(1, [1, 1, 1, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []


def test_multimodal_scheduler_withholds_request_on_budget_overflow():
    scheduler = _scheduler(max_batch_size=3, max_num_tokens=10)
    request = make_mm_request(1, [6, 4, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []


def test_multimodal_scheduler_preserves_non_multimodal_requests():
    scheduler = _scheduler(max_batch_size=1, max_num_tokens=1)
    request = make_llm_request(1)
    initialize_multimodal_encoder_request(request, max_num_tokens=1)

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]


def test_request_rejects_item_above_effective_startup_maximum():
    request = make_mm_request(1, [9])
    request.py_multimodal_data["image"] = {"pixel_values": torch.empty(1)}

    with pytest.raises(ValueError, match="exceeding the effective startup maximum 8"):
        initialize_multimodal_encoder_request(request, max_num_tokens=8)


def test_eager_scheduler_encodes_request_rejected_by_llm_capacity():
    base_scheduler = _BaseScheduler()
    base_scheduler.capacity_scheduler = _RejectMultimodalCapacityScheduler()
    scheduler = _scheduler(
        max_batch_size=1,
        max_num_tokens=8,
        base_scheduler=base_scheduler,
        scheduler_cls=MultimodalEagerEncoderScheduler,
    )
    multimodal_request = make_mm_request(1, [8])
    text_request = make_llm_request(2)
    initialize_multimodal_encoder_request(text_request, max_num_tokens=8)

    output = scheduler.schedule_request([multimodal_request, text_request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [text_request]


def test_forward_multimodal_encoder_step_scopes_failure_to_item_owners():
    failed = make_mm_request(1, [4])
    unrelated_context = make_llm_request(2)
    unrelated_generation = make_llm_request(3)
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
    failed = make_mm_request(1, [4])
    failed.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = ("image", 0)
    unrelated = make_llm_request(2)
    handled = []

    def fail_encoder(*_):
        raise MultimodalEncoderRequestError(
            "multimodal_encoder_item_metadata must be a MultimodalEncoderItemMetadata"
        )

    engine = SimpleNamespace(forward_multimodal_encoder_items=fail_encoder)

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
    unrelated = make_llm_request(2)
    handled = []

    engine = SimpleNamespace(
        forward_multimodal_encoder_items=bare_mm_item_scheduler(
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
    failed = make_mm_request(1, [4])

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
        make_llm_request(request_id, multimodal_data=multimodal_data),
    )


def test_mm_admission_does_not_charge_ready_active_request():
    active = make_mm_request(1, [8], ready=(0,))
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
    active = make_mm_request(1, [4])
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


def test_terminate_request_releases_multimodal_cache_references_idempotently():
    request = make_mm_request(1, [4, 4])
    state = request.py_mm_encoder_state
    cache = TensorLRUCache(16)
    cache_key = ("mm_transient", request.request_id, 0)
    cache.acquire(cache_key, 4, retain_after_release=False)
    cache.commit(cache_key, torch.ones(1))
    state.set_item_cache_key(0, cache_key, ready=True)
    freed = []

    executor = object.__new__(PyExecutor)
    executor._mm_encoder_item_scheduling_enabled = True
    executor.enable_attention_dp = False
    executor.global_rank = 0
    executor.model_engine = SimpleNamespace(mm_encoder_cache=cache)
    executor.resource_manager = SimpleNamespace(free_resources=freed.append)
    executor._prefetched_request_ids = {request.py_request_id}
    executor._disagg_timed_out_ctx_cancelled_ids = {request.py_request_id}
    executor._disagg_timed_out_gen_cancelled_ids = {request.py_request_id}
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0)
    executor.result_wait_queues = {}

    executor._do_terminate_request(request)
    executor._release_multimodal_resources(request)

    assert freed == [request]
    assert request.py_mm_encoder_state is None
    assert request.py_multimodal_data == {}
    assert executor._prefetched_request_ids == set()
    assert executor._disagg_timed_out_ctx_cancelled_ids == set()
    assert executor._disagg_timed_out_gen_cancelled_ids == set()


def test_weight_invalidation_rejects_live_references():
    invalidations = []
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor.model_engine = SimpleNamespace(
        invalidate_multimodal_encoder_cache=lambda: invalidations.append(True)
    )
    executor.invalidate_multimodal_encoder_cache()

    assert invalidations == [True]

    request = make_mm_request(1, [4])
    request.py_mm_encoder_state.set_item_cache_key(0, ("cache", 0), ready=False)
    executor.active_requests = [request]
    with pytest.raises(RuntimeError, match="live multimodal cache references"):
        executor.invalidate_multimodal_encoder_cache()
    assert invalidations == [True]


# ---------------------------------------------------------------------------
# MultimodalEncoderRequestState unit behavior
# ---------------------------------------------------------------------------


def test_mm_encoder_state_enforces_lengths_slot_invariant():
    with pytest.raises(ValueError, match="one cache key per item slot"):
        MultimodalEncoderRequestState(
            embedding_lengths=[2], encoder_token_lengths=[4], item_ready=[False, False]
        )


def test_mm_encoder_state_copies_validated_scheduler_costs_at_admission():
    request = make_mm_request(1, [4, 7])

    assert request.py_mm_encoder_state.encoder_token_lengths == [4, 7]

    metadata = request.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    metadata.encoder_token_lengths[0] = 100
    assert request.py_mm_encoder_state.encoder_token_lengths == [4, 7]

    scheduler = _scheduler(max_batch_size=2, max_num_tokens=11)
    output = scheduler.schedule_request([request], set())
    assert output.scheduled_mm_encoder_items == {1: [0, 1]}


def test_mm_encoder_state_tracks_prompt_ordered_cache_key_readiness():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2, 3])
    first_cache_key = ("cache", 0)
    second_cache_key = ("cache", 1)

    state.set_item_cache_key(0, first_cache_key, ready=False)
    state.set_item_cache_key(1, second_cache_key, ready=False)
    state.mark_cache_key_ready(second_cache_key)
    assert state.progress is MultimodalEncoderProgress.PARTIAL
    assert state.pending_item_indices() == [0]

    state.mark_cache_key_ready(first_cache_key)
    assert state.progress is MultimodalEncoderProgress.READY
    assert state.pop_all_cache_keys() == [first_cache_key, second_cache_key]
    assert state.item_cache_keys == [None, None]
    assert state.progress is MultimodalEncoderProgress.PENDING
