# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
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
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine,
    _validate_mm_encoder_scheduling_compatibility,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    MultimodalEagerEncoderScheduler,
    MultimodalScheduler,
    ScheduledRequests,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm._torch.tensor_lru_cache import CacheEntryState, TensorLRUCache
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
    del hidden, fill
    state.item_ready[item_idx] = True


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
    scheduler = _scheduler(max_batch_size=2, max_num_tokens=10)
    first = _request(1, [7, 7])
    second = _request(2, [3])

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
    first = _request(1, [4])
    second = _request(2, [4])

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
    first = _request(1, [3])
    second = _request(2, [3])

    output = scheduler.schedule_request([first, second], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [first]


def test_pinned_outputs_block_new_admissions_until_explicit_release():
    scheduler = _scheduler(max_batch_size=8, max_num_tokens=1 << 20, cache_capacity=4)
    holder = _request(1, [3])
    newcomer = _request(2, [3])

    first_output = scheduler.schedule_request([holder], set())
    assert first_output.scheduled_mm_encoder_items == {1: [0]}
    holder_cache_key = holder.py_mm_encoder_state.item_cache_keys[0]
    assert holder_cache_key is not None
    assert scheduler.encoder_cache.put(
        holder_cache_key,
        torch.ones(1, dtype=torch.float32),
        expected_state=CacheEntryState.RESERVED,
    )
    holder.py_mm_encoder_state.mark_cache_key_ready(holder_cache_key)

    output = scheduler.schedule_request([holder, newcomer], set())
    assert output.scheduled_mm_encoder_items is None

    drained = holder.py_mm_encoder_state.pop_all_cache_keys()
    assert drained == [holder_cache_key]
    assert scheduler.encoder_cache.release(holder_cache_key) == holder_cache_key
    holder.py_mm_encoder_state = None
    output = scheduler.schedule_request([holder, newcomer], set())
    assert output.scheduled_mm_encoder_items == {2: [0]}


def test_started_request_holds_its_whole_footprint_across_iterations():
    # The token budget splits the head request across iterations, but its
    # first item already allocates storage for all of them, so the bytes it
    # still needs are charged from the start. A request behind it cannot
    # squat that space and leave the head unable to finish.
    scheduler = _scheduler(max_batch_size=8, max_num_tokens=5, cache_capacity=8)
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
    scheduler = _scheduler(max_batch_size=8, max_num_tokens=1 << 20, cache_capacity=4)
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
            encoder_cache=TensorLRUCache(4),
            get_item_cache_keys=_item_cache_keys,
            bytes_per_encoder_embedding=0,
            retain_cache_entries=False,
        )


def test_multimodal_scheduler_selects_all_items_and_admits_request_when_batch_fits():
    scheduler = _scheduler(max_batch_size=2, max_num_tokens=10)
    request = _request(1, [6, 4])

    output = scheduler.schedule_request([request], set())

    # The encoder step is the single encode site: an in-budget batch simply
    # has every pending item selected, and the request still enters the LLM
    # batch in the same iteration (encode runs before the LLM forward).
    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == [request]


def test_multimodal_scheduler_respects_encoder_batch_size():
    scheduler = _scheduler(max_batch_size=2, max_num_tokens=4)
    request = _request(1, [1, 1, 1, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []


def test_multimodal_scheduler_withholds_request_on_budget_overflow():
    scheduler = _scheduler(max_batch_size=3, max_num_tokens=10)
    request = _request(1, [6, 4, 1])

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items == {1: [0, 1]}
    assert output.context_requests == []


def test_multimodal_scheduler_preserves_non_multimodal_requests():
    scheduler = _scheduler(max_batch_size=1, max_num_tokens=1)
    request = _llm_request(1)
    initialize_multimodal_encoder_request(request, max_num_tokens=1)

    output = scheduler.schedule_request([request], set())

    assert output.scheduled_mm_encoder_items is None
    assert output.context_requests == [request]


def test_qwen_output_budget_uses_post_merge_embedding_capacity():
    from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase

    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    engine = object.__new__(PyTorchModelEngine)
    engine.max_num_tokens = 8192
    engine.encoder_max_num_tokens = 65536
    engine.input_processor = processor
    engine._get_mm_encoder_embedding_size_bytes = lambda: 32768

    budget = engine._compute_mm_encoder_output_budget_bytes()

    assert engine.max_mm_encoder_output_embeddings == 16384
    assert engine.bytes_per_mm_encoder_embedding == 32768
    assert budget == 512 * 1024**2


def test_output_row_bytes_use_config_dtype_without_pp_embedding_weight():
    engine = object.__new__(PyTorchModelEngine)
    engine.model = SimpleNamespace(
        embedding_dim=16384,
        model_config=SimpleNamespace(torch_dtype=torch.bfloat16),
    )

    assert engine._get_mm_encoder_embedding_size_bytes() == 32768


def test_downstream_pp_item_scheduler_does_not_create_encoder_store():
    class _CacheModel(MultimodalModelMixin):
        supports_encoder_cache = True

    model = _CacheModel()
    model.model_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(encoder_cache_max_bytes=64)
    )
    model._multimodal_encoder_cache = None
    engine = object.__new__(PyTorchModelEngine)
    engine.model = model
    engine.mm_encoder_item_scheduling_enabled = True
    engine.mapping = SimpleNamespace(is_first_pp_rank=lambda: False)

    assert engine.mm_encoder_cache is None
    assert model._multimodal_encoder_cache is None


def test_output_budget_requires_processor_embedding_capacity():
    engine = object.__new__(PyTorchModelEngine)
    engine.encoder_max_num_tokens = 65536
    engine.input_processor = SimpleNamespace(get_max_mm_encoder_output_embeddings=lambda *_: None)

    with pytest.raises(ValueError, match="get_max_mm_encoder_output_embeddings"):
        engine._compute_mm_encoder_output_budget_bytes()


def test_eager_compatibility_is_checked_only_for_item_scheduled_models():
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
            encoder_side_stream_max_ahead=0,
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


def test_side_stream_compatibility_is_checked_only_for_item_scheduled_models():
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
            encoder_side_stream_max_ahead=1,
        ),
        enable_attention_dp=False,
        cache_transceiver_config=None,
    )

    _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="side-stream prefetch"):
        _validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_request_rejects_item_above_effective_startup_maximum():
    request = _request(1, [9])
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
    multimodal_request = _request(1, [8])
    text_request = _llm_request(2)
    initialize_multimodal_encoder_request(text_request, max_num_tokens=8)

    output = scheduler.schedule_request([multimodal_request, text_request], set())

    assert output.scheduled_mm_encoder_items == {1: [0]}
    assert output.context_requests == [text_request]


def test_multimodal_scheduler_remote_result_skips_local_cache_policy():
    scheduler = _scheduler(max_batch_size=1, max_num_tokens=4)
    blocked = _request(1, [4])
    text_request = _llm_request(2)

    output = scheduler.schedule_request_with_mm_decisions(
        [blocked, text_request],
        set(),
        blocked_request_ids=[blocked.request_id],
        scheduled_items={blocked.request_id: [0]},
        cache_removals=[("old", 0)],
    )

    assert output.context_requests == [text_request]
    assert output.scheduled_mm_encoder_items == {blocked.request_id: [0]}
    assert output.mm_encoder_cache_removals == [("old", 0)]
    assert blocked.py_mm_encoder_state.item_cache_keys == [None]
    assert len(scheduler.encoder_cache) == 0


def test_forward_multimodal_encoder_step_scopes_failure_to_item_owners():
    failed = _request(1, [4])
    follower = _request(4, [4])
    unrelated_mm_context = _request(5, [4])
    unrelated_context = _llm_request(2)
    unrelated_generation = _llm_request(3)
    handled = []

    def fail_encoder(*_, **__):
        raise MultimodalEncoderRequestError(
            "bad MM output", request_ids={failed.request_id, follower.request_id}
        )

    executor = object.__new__(PyExecutor)
    executor.active_requests = [
        failed,
        follower,
        unrelated_mm_context,
        unrelated_context,
        unrelated_generation,
    ]
    executor.enable_attention_dp = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.global_rank = 0
    executor.dist = SimpleNamespace(world_size=1, is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(run_multimodal_encoder_schedule=fail_encoder)
    executor._handle_errors = lambda error_msg, **kwargs: handled.append((error_msg, kwargs))

    scheduled_requests = ScheduledRequests()
    scheduled_requests.reset_context_requests(
        [failed, follower, unrelated_mm_context, unrelated_context]
    )
    scheduled_requests.append_generation_request(unrelated_generation)
    scheduled_requests.scheduled_mm_encoder_items = {failed.request_id: [0]}

    result = executor._forward_multimodal_encoder_step(scheduled_requests)

    assert scheduled_requests.context_requests == [unrelated_mm_context, unrelated_context]
    assert scheduled_requests.generation_requests == [unrelated_generation]
    assert scheduled_requests.scheduled_mm_encoder_items is None
    assert result == ("bad MM output", [failed.request_id, follower.request_id])
    assert handled == [
        (
            "bad MM output",
            {"requests": [failed, follower], "charge_budget": False},
        )
    ]


def test_forward_multimodal_encoder_step_contains_model_contract_error():
    failed = _request(1, [4])
    failed.py_multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = ("image", 0)
    unrelated = _llm_request(2)
    handled = []

    def fail_run(*_, **__):
        raise MultimodalEncoderRequestError(
            "multimodal_encoder_item_metadata must be a MultimodalEncoderItemMetadata"
        )

    executor = object.__new__(PyExecutor)
    executor.active_requests = [failed, unrelated]
    executor.enable_attention_dp = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.global_rank = 0
    executor.dist = SimpleNamespace(world_size=1, is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(run_multimodal_encoder_schedule=fail_run)
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

    def fail_run(*_, **__):
        raise MultimodalEncoderRequestError("Scheduled MM request 1 is no longer active")

    executor = object.__new__(PyExecutor)
    executor.active_requests = [unrelated]
    executor.enable_attention_dp = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.global_rank = 0
    executor.dist = SimpleNamespace(world_size=1, is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(run_multimodal_encoder_schedule=fail_run)
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

    def fail_encoder(*_, **__):
        raise torch.cuda.OutOfMemoryError("encoder OOM")

    executor = object.__new__(PyExecutor)
    executor.active_requests = [failed]
    executor.enable_attention_dp = False
    executor._mm_encoder_item_scheduling_enabled = True
    executor.global_rank = 0
    executor.dist = SimpleNamespace(is_first_pp_rank=True, pp_size=1)
    executor.model_engine = SimpleNamespace(run_multimodal_encoder_schedule=fail_encoder)

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


@pytest.mark.parametrize(
    "pp_size, expected_pending_removals", [(1, []), (2, [("mm_transient", 1, 0)])]
)
def test_terminate_request_releases_multimodal_cache_references_idempotently(
    pp_size, expected_pending_removals
):
    request = _request(1, [4, 4])
    state = request.py_mm_encoder_state
    cache = TensorLRUCache(16)
    cache_key = ("mm_transient", request.request_id, 0)
    cache.allocate(cache_key, 4, retain_after_release=False)
    cache.put(cache_key, torch.ones(1), expected_state=CacheEntryState.RESERVED, expected_bytes=4)
    state.set_item_cache_key(0, cache_key, ready=True)
    freed = []

    executor = object.__new__(PyExecutor)
    executor._mm_encoder_item_scheduling_enabled = True
    executor.enable_attention_dp = False
    executor.global_rank = 0
    executor.model_engine = SimpleNamespace(mm_encoder_cache=cache)
    executor._pending_mm_encoder_cache_removals = []
    executor.resource_manager = SimpleNamespace(free_resources=freed.append)
    executor._prefetched_request_ids = {request.py_request_id}
    executor._disagg_timed_out_ctx_cancelled_ids = {request.py_request_id}
    executor._disagg_timed_out_gen_cancelled_ids = {request.py_request_id}
    executor.gather_all_responses = False
    executor.dist = SimpleNamespace(rank=0, is_first_pp_rank=True, pp_size=pp_size)
    executor.result_wait_queues = {}

    executor._do_terminate_request(request)
    executor._release_multimodal_resources(request)

    assert freed == [request]
    assert request.py_mm_encoder_state is None
    assert request.py_multimodal_data == {}
    assert executor._pending_mm_encoder_cache_removals == expected_pending_removals
    assert executor._prefetched_request_ids == set()
    assert executor._disagg_timed_out_ctx_cancelled_ids == set()
    assert executor._disagg_timed_out_gen_cancelled_ids == set()


def test_weight_invalidation_clears_old_removal_delta_and_rejects_live_references():
    invalidations = []
    executor = object.__new__(PyExecutor)
    executor.active_requests = []
    executor.model_engine = SimpleNamespace(
        invalidate_multimodal_encoder_cache=lambda: invalidations.append(True)
    )
    executor._pending_mm_encoder_cache_removals = [("old", 0)]

    executor.invalidate_multimodal_encoder_cache()

    assert invalidations == [True]
    assert executor._pending_mm_encoder_cache_removals == []

    request = _request(1, [4])
    request.py_mm_encoder_state.set_item_cache_key(0, ("cache", 0), ready=False)
    executor.active_requests = [request]
    with pytest.raises(RuntimeError, match="live multimodal cache references"):
        executor.invalidate_multimodal_encoder_cache()
    assert invalidations == [True]


def test_item_outputs_commit_to_prompt_ordered_cache_keys(monkeypatch):
    cache = TensorLRUCache(1 << 20, name="test")

    class _Model(MultimodalModelMixin):
        def _get_multimodal_encoder_cache(self):
            return cache

        def forward_multimodal_encoder_items(self, encoder_inputs):
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_lengths, _ in encoder_inputs
                for embedding_length in embedding_lengths
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    engine = object.__new__(PyTorchModelEngine)
    engine.model = _Model()
    engine.mm_encoder_item_scheduling_enabled = True
    engine.mapping = SimpleNamespace(is_first_pp_rank=lambda: True)
    engine.bytes_per_mm_encoder_embedding = 8
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
    state = request.py_mm_encoder_state
    item_cache_keys = [
        ("mm_transient", request.request_id, item_idx) for item_idx in range(state.num_items)
    ]
    for item_idx, (cache_key, rows) in enumerate(
        zip(item_cache_keys, state.embedding_lengths, strict=True)
    ):
        cache.allocate(cache_key, rows * 8, retain_after_release=False)
        state.set_item_cache_key(item_idx, cache_key, ready=False)

    engine.forward_multimodal_encoder_items([request], {1: [0]})

    assert state.item_ready == [True, False]
    torch.testing.assert_close(cache.get(item_cache_keys[0]), torch.full((2, 2), 2.0))
    assert "image" in request.py_multimodal_data

    engine.forward_multimodal_encoder_items([request], {1: [1]})

    torch.testing.assert_close(cache.get(item_cache_keys[1]), torch.full((3, 2), 3.0))
    assert "multimodal_embedding" not in request.py_multimodal_data
    assert "image" not in request.py_multimodal_data
    assert is_multimodal_encoder_ready(request)


# ---------------------------------------------------------------------------
# MultimodalEncoderRequestState unit behavior
# ---------------------------------------------------------------------------


def test_mm_encoder_state_enforces_lengths_slot_invariant():
    with pytest.raises(ValueError, match="one cache key per item slot"):
        MultimodalEncoderRequestState(
            embedding_lengths=[2], encoder_token_lengths=[4], item_ready=[False, False]
        )


def test_mm_encoder_state_copies_validated_scheduler_costs_at_admission():
    request = _request(1, [4, 7])

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


def test_mm_encoder_state_rejects_replacing_an_item_cache_key():
    state = MultimodalEncoderRequestState.from_embedding_lengths([2])
    state.set_item_cache_key(0, ("cache", 0), ready=False)

    with pytest.raises(RuntimeError, match="already has a cache key"):
        state.set_item_cache_key(0, ("cache", 1), ready=False)
