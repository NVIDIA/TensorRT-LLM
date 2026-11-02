# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for request_utils.py functions.

This module tests:
- Request merging functions (merge_requests, merge_helix_requests)
- Waiting queue functions (get_from_waiting_queue, can_process_attention_dp_request)
- Rank-0 clock piggyback in RequestBroadcaster

"""

import datetime
import gc
import pickle
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.request_utils import (
    RequestBroadcaster,
    attach_py_objects_to_requests,
    can_process_attention_dp_request,
    derive_attention_dp_per_rank_request_cap,
    executor_request_to_llm_request,
    get_from_waiting_queue,
    merge_helix_requests,
    merge_requests,
    partition_context_for_helix,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampling_resolution import SlotSamplingParams
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest as CppLlmRequest
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalItemPartition,
)
from tensorrt_llm.mapping import CpType
from tensorrt_llm.sampling_params import LogprobMode

pytestmark = pytest.mark.cpu_only


def test_multimodal_payload_broadcast_and_children_keep_the_same_contract():
    import pickle

    metadata = MultimodalItemMetadata(
        [("audio", 0), ("image", 0)],
        output_embedding_lengths=[1, 2],
        input_partitions={"audio": {"features": MultimodalItemPartition(1, (3,))}},
    )
    payload = {
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
        "audio": {"features": torch.arange(6).reshape(2, 3)},
        "image": {},
    }
    source = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=2,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
    )
    source.py_multimodal_data = payload
    source.py_conversation_params = None
    objects = RequestBroadcaster._collect_py_objects(None, [RequestQueueItem(42, source)])
    # Reproduce the existing Python-object rank handoff, without adding the
    # request aggregate, CUDA event, or cache state to its wire format.
    restored_objects = pickle.loads(pickle.dumps(objects))
    restored = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=2,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
    )
    queue_items = [RequestQueueItem(42, restored)]
    attach_py_objects_to_requests(queue_items, restored_objects)
    request = executor_request_to_llm_request(
        42, restored, child_req_ids=[43], exclude_last_generation_logits=False
    )
    params = request.py_multimodal_params
    child = request.child_requests[0].py_multimodal_params
    assert params.item_metadata == child.item_metadata == metadata
    assert child is not params
    assert child.item_metadata is not params.item_metadata
    assert params.encoder_state is child.encoder_state is None
    assert params.multimodal_runtime is child.multimodal_runtime is None
    assert params.item_metadata.item_refs == (("audio", 0), ("image", 0))
    torch.testing.assert_close(
        child.multimodal_data["audio"]["features"], payload["audio"]["features"]
    )


@pytest.fixture
def attention_dp_config():
    """Create a config dict for attention DP testing."""
    return {
        "tp_size": 4,
        "max_num_active_requests": 8,
    }


@pytest.fixture
def all_ranks_num_active_requests():
    return [2, 1, 3, 0]  # 4 ranks


def create_mock_request_with_py_schedule_params(attention_dp_rank=None, attention_dp_relax=False):
    mock_request = Mock()

    if attention_dp_rank is not None:
        mock_schedule_params = Mock()
        mock_schedule_params.attention_dp_rank = attention_dp_rank
        mock_schedule_params.attention_dp_relax = attention_dp_relax

        mock_schedule_params.configure_mock(
            attention_dp_rank=attention_dp_rank, attention_dp_relax=attention_dp_relax
        )

        mock_request.py_scheduling_params = mock_schedule_params
    else:
        mock_request.py_scheduling_params = None

    mock_request.input_token_ids = [1, 2, 3]

    return mock_request


def test_request_broadcaster_collects_conversation_params_with_none():
    conversation_params = ConversationParams(conversation_id="conv")
    source_items = [
        RequestQueueItem(1, SimpleNamespace(py_conversation_params=conversation_params)),
        RequestQueueItem(2, SimpleNamespace(py_conversation_params=None)),
    ]

    py_request_objects = RequestBroadcaster._collect_py_objects(None, source_items)

    py_objects = dict(py_request_objects)
    assert py_objects["py_conversation_params"] == {
        1: conversation_params,
        2: None,
    }

    target_items = [
        RequestQueueItem(1, SimpleNamespace()),
        RequestQueueItem(2, SimpleNamespace()),
    ]
    attach_py_objects_to_requests(target_items, py_request_objects)

    assert target_items[0].request.py_conversation_params is conversation_params
    assert hasattr(target_items[1].request, "py_conversation_params")
    assert target_items[1].request.py_conversation_params is None


def test_request_broadcaster_requires_conversation_params_attr():
    source_items = [RequestQueueItem(1, SimpleNamespace())]

    with pytest.raises(AttributeError):
        RequestBroadcaster._collect_py_objects(None, source_items)


def _broadcast_round_trip(request: trtllm.Request) -> trtllm.Request:
    request.py_conversation_params = None
    items = [RequestQueueItem(42, request)]
    sidecar = RequestBroadcaster._collect_py_objects(None, items)
    restored_items, restored_sidecar = pickle.loads(pickle.dumps((items, sidecar)))
    attach_py_objects_to_requests(restored_items, restored_sidecar)
    return restored_items[0].request


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("original_end_id", [None, 99])
def test_request_broadcast_preserves_python_inputs(enabled, original_end_id) -> None:
    request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        end_id=-1,
        output_config=trtllm.OutputConfig(return_perf_metrics=True),
    )
    request.py_logprobs_mode = LogprobMode.PROCESSED if enabled else LogprobMode.RAW
    request.py_logprobs_simple_format = enabled
    request.py_return_routed_experts = enabled
    request.py_original_end_id = original_end_id
    request.py_arrival_time = 123.0

    restored = _broadcast_round_trip(request)
    for attribute in (
        "py_logprobs_mode",
        "py_logprobs_simple_format",
        "py_return_routed_experts",
        "py_original_end_id",
        "py_arrival_time",
    ):
        assert getattr(restored, attribute) == getattr(request, attribute)

    translated = executor_request_to_llm_request(42, restored, [], False)
    assert translated.py_logprobs_mode == request.py_logprobs_mode
    assert translated.py_logprobs_simple_format is enabled
    assert translated.py_return_routed_experts is enabled
    assert translated.py_original_end_id == original_end_id
    translated.finish_by_reason(trtllm.FinishReason.CANCELLED)
    response = translated.create_response()
    assert response is not None
    metrics = response.result.get_result().request_perf_metrics
    assert metrics.timing_metrics.arrival_time == datetime.timedelta(seconds=123)


def test_request_broadcast_preserves_absent_python_input_defaults() -> None:
    restored = _broadcast_round_trip(
        trtllm.Request(input_token_ids=[1, 2, 3], max_tokens=10, end_id=7)
    )
    assert not hasattr(restored, "py_original_end_id")
    translated = executor_request_to_llm_request(42, restored, [], False)
    assert translated.py_logprobs_mode == LogprobMode.RAW
    assert translated.py_logprobs_simple_format is False
    assert translated.py_return_routed_experts is False
    assert translated.py_original_end_id == 7


@pytest.mark.parametrize("position_ids", [None, [4, 5, 6]])
@pytest.mark.parametrize("override", [None, [], [7, 8, 9]])
def test_request_translation_preserves_position_ids_after_broadcast(position_ids, override) -> None:
    restored = _broadcast_round_trip(
        trtllm.Request(input_token_ids=[1, 2, 3], max_tokens=10, position_ids=position_ids)
    )
    translated = executor_request_to_llm_request(42, restored, [], False, position_ids=override)
    assert translated.py_position_ids == (position_ids if override is None else override)


@pytest.mark.parametrize("output_length", [None, 6])
def test_request_translation_preserves_direct_encoder_inputs_after_broadcast(output_length) -> None:
    features = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    restored = _broadcast_round_trip(
        trtllm.Request(
            input_token_ids=[1, 2, 3],
            max_tokens=10,
            encoder_input_features=features,
            encoder_output_length=output_length,
            sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        )
    )
    assert restored.encoder_output_length == output_length
    translated = executor_request_to_llm_request(42, restored, [43], False)
    for request in (translated, *translated.child_requests):
        torch.testing.assert_close(request.py_encoder_input_features, features)
        assert request.py_encoder_input_features is translated.py_encoder_input_features
        assert request.encoder_output_len == (12 if output_length is None else output_length)
        assert request.state == LlmRequestState.ENCODER_INIT
        request.pause(20)
        assert request.state == LlmRequestState.ENCODER_INIT


@pytest.mark.parametrize("output_length", [None, 6])
def test_request_translation_preserves_token_encoder_output_length(output_length) -> None:
    restored = _broadcast_round_trip(
        trtllm.Request(
            input_token_ids=[1, 2, 3],
            max_tokens=10,
            encoder_input_token_ids=[4, 5, 6, 7],
            encoder_output_length=output_length,
        )
    )
    translated = executor_request_to_llm_request(42, restored, [], False)
    assert translated.encoder_output_len == (4 if output_length is None else output_length)
    assert translated.state == LlmRequestState.ENCODER_INIT


@pytest.mark.parametrize("direct_output_length", [None, 6, 7])
@pytest.mark.parametrize("direct_features", [False, True])
def test_request_translation_resolves_encoder_input_routes(
    direct_features, direct_output_length
) -> None:
    features = torch.ones(12, 8)
    request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        encoder_input_features=features if direct_features else None,
        encoder_output_length=direct_output_length,
    )
    request.py_multimodal_data = {
        "audio": {"encoder_input_features": features, "encoder_output_len": 6}
    }
    restored = _broadcast_round_trip(request)
    if direct_features:
        with pytest.raises(ValueError, match="Specify encoder_input_features either directly"):
            executor_request_to_llm_request(42, restored, [], False)
    elif direct_output_length == 7:
        with pytest.raises(ValueError, match="encoder_output_length conflicts"):
            executor_request_to_llm_request(42, restored, [], False)
    else:
        translated = executor_request_to_llm_request(42, restored, [], False)
        torch.testing.assert_close(translated.py_encoder_input_features, features)
        assert translated.encoder_output_len == 6
        assert translated.state == LlmRequestState.ENCODER_INIT


def test_request_translation_requires_multimodal_encoder_output_length() -> None:
    request = trtllm.Request(input_token_ids=[1, 2, 3], max_tokens=10)
    request.py_multimodal_data = {"audio": {"encoder_input_features": torch.ones(12, 8)}}
    with pytest.raises(ValueError, match="without encoder_output_len"):
        executor_request_to_llm_request(42, _broadcast_round_trip(request), [], False)


@pytest.mark.parametrize("seed", [None, 11])
def test_child_factory_owns_inputs_and_resets_execution_state(seed) -> None:
    features = torch.ones(12, 8)
    parent = LlmRequest(
        request_id=42,
        input_tokens=[1, 2, 3],
        max_new_tokens=10,
        sampling_config=trtllm.SamplingConfig(seed=seed, num_return_sequences=3),
        is_streaming=True,
        client_id=7,
        return_log_probs=True,
        return_generation_logits=True,
        exclude_last_generation_logits=True,
        stop_words_list=[[4, 5]],
        embedding_bias=torch.ones(8),
        position_ids=[7, 8, 9],
        encoder_input_features=features,
        encoder_output_len=6,
        py_multimodal_data={"embedding": torch.ones(2, 8)},
        py_lora_ckpt_source="hf",
        draft_tokens=[10, 11],
    )
    parent.py_bad_words = [[6, 7]]
    parent.py_original_end_id = 99
    parent.seq_slot = 4
    parent.py_seq_slot = 4
    parent.py_batch_idx = 3
    parent.py_decoding_iter = 5
    parent.py_total_draft_tokens = 8
    parent.py_per_pos_drafted[0] = 8
    parent.py_draft_tokens = [99]
    parent.py_encoder_output_ready_event = object()
    parent.py_encoder_output = torch.ones(6, 8)
    parent.py_perf_timing = object()
    parent.py_kv_transfer_timed_out = True
    parent.py_connector_allocation_reported = True
    parent.py_unclassified_state = object()
    parent.py_result.diff.context_logits_list.append(torch.ones(3, 8))

    for child_id in (43, 44):
        parent.create_child_request(child_id)
    child, sibling = parent.child_requests
    assert parent._native_child_requests[0] is child
    assert parent._native_child_requests[1] is sibling
    assert parent.sampling_config.seed == (seed or 0)
    assert child.sampling_config.seed == (seed or 0) + 1
    assert sibling.sampling_config.seed == (seed or 0) + 2
    assert child.parent_request_id == 42
    assert child.py_request_id == 43
    assert child.py_client_id == 7
    assert child.py_lora_ckpt_source == "hf"
    assert child.child_requests == []
    assert child._native_child_requests == []
    assert child.seq_slot is None
    assert child.py_seq_slot is None
    assert child.py_batch_idx is None
    assert child.py_decoding_iter == 0
    assert child.py_total_draft_tokens == 0
    assert child.py_per_pos_drafted[0] == 0
    assert child.py_draft_tokens == [10, 11]
    assert child.py_mm_encoder_event is None
    assert child.py_mm_encoder_state is None
    assert child.py_encoder_output_ready_event is None
    assert child.py_encoder_output is None
    assert child.py_perf_timing is None
    assert child.py_kv_transfer_timed_out is False
    assert child.py_connector_allocation_reported is False
    assert not hasattr(child, "py_unclassified_state")
    assert child.py_result is not parent.py_result
    assert child.py_result.diff.context_logits_list == []
    assert child.py_result._exclude_last_generation_logits is True
    assert child.py_result._generation_logits is not None
    assert child.py_result._log_probs is not None
    assert child.py_result._log_probs is not parent.py_result._log_probs
    assert child.py_encoder_input_features is features
    assert sibling.py_encoder_input_features is features
    assert child.encoder_output_len == 6
    assert child.py_original_end_id == 99
    assert (
        child.py_multimodal_data["embedding"].data_ptr()
        != parent.py_multimodal_data["embedding"].data_ptr()
    )
    child.py_embedding_bias.add_(1)
    torch.testing.assert_close(parent.py_embedding_bias, torch.ones(8))
    child.py_position_ids[0] = 100
    child.py_stop_words_list[0][0] = 100
    child.py_bad_words[0][0] = 100
    assert parent.py_position_ids == [7, 8, 9]
    assert sibling.py_position_ids == [7, 8, 9]
    assert parent.py_stop_words_list == [[4, 5]]
    assert parent.py_bad_words == [[6, 7]]
    child.seq_slot = 9
    assert parent._native_child_requests[0].seq_slot == 9


@pytest.mark.parametrize("processor_count", [None, 0, 2])
def test_child_factory_copies_processor_list_and_preserves_callables(processor_count) -> None:
    class StatefulProcessor:
        def __init__(self):
            self.state = []

        def __call__(self, logits):
            return logits

        def __deepcopy__(self, memo):
            raise AssertionError("User-supplied processors must not be deep-copied")

    processors = (
        None if processor_count is None else [StatefulProcessor() for _ in range(processor_count)]
    )
    parent = LlmRequest(
        request_id=42,
        input_tokens=[1, 2, 3],
        max_new_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=3),
        is_streaming=False,
        py_logits_post_processors=processors,
    )
    parent.create_child_request(43)
    parent.create_child_request(44)
    child, sibling = parent.child_requests
    if processors is None:
        assert child.py_logits_post_processors is None
        assert sibling.py_logits_post_processors is None
        return

    child_processors = child.py_logits_post_processors
    sibling_processors = sibling.py_logits_post_processors
    assert parent.py_logits_post_processors is processors
    assert child_processors is not processors
    assert sibling_processors is not processors
    assert child_processors is not sibling_processors
    for original, child_processor, sibling_processor in zip(
        processors, child_processors, sibling_processors, strict=True
    ):
        assert child_processor is original
        assert sibling_processor is original
        assert child_processor.state is original.state

    child_processors.append(StatefulProcessor())
    assert len(processors) == processor_count
    assert len(sibling_processors) == processor_count


@pytest.mark.parametrize("runtime_field", ["py_mm_encoder_state", "py_mm_encoder_event"])
def test_child_factory_rejects_multimodal_encoder_ownership(runtime_field) -> None:
    parent = LlmRequest(
        request_id=42,
        input_tokens=[1, 2, 3],
        max_new_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        is_streaming=False,
        py_multimodal_data={"embedding": torch.ones(2, 8)},
    )
    runtime_value = object()
    setattr(parent, runtime_field, runtime_value)
    with pytest.raises(RuntimeError, match="before MM encoder admission"):
        parent.create_child_request(43)
    assert parent.child_requests == []
    assert parent._native_child_requests == []
    assert parent.sampling_config.seed is None
    assert getattr(parent, runtime_field) is runtime_value

    setattr(parent, runtime_field, None)
    parent.create_child_request(43)
    assert parent.child_requests[0].sampling_config.seed == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_child_factory_shares_only_sequence_group_completion(streaming) -> None:
    parent = LlmRequest(
        request_id=42,
        input_tokens=[1, 2, 3],
        max_new_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=3),
        is_streaming=streaming,
    )
    parent.create_child_request(43)
    parent.create_child_request(44)
    child, sibling = parent.child_requests
    assert not parent.is_finished
    for request, expected_final in ((child, False), (parent, False), (sibling, True)):
        request.finish_by_reason(trtllm.FinishReason.CANCELLED)
        response = request.create_response()
        assert response is not None
        assert response.request_id == 42
        assert response.result.get_result().is_sequence_final
        assert response.result.is_final is expected_final
    assert child.is_finished
    assert sibling.is_finished


@pytest.mark.parametrize("input_field", ["py_multimodal_data", "py_disaggregated_params"])
def test_child_factory_failure_does_not_register_a_partial_child(input_field) -> None:
    class UncopyableInput:
        def __deepcopy__(self, memo):
            raise ValueError("input copy failed")

    parent = LlmRequest(
        request_id=42,
        input_tokens=[1, 2, 3],
        max_new_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        is_streaming=False,
    )
    setattr(parent, input_field, UncopyableInput())
    with pytest.raises(ValueError, match="input copy failed"):
        parent.create_child_request(43)
    assert parent.child_requests == []
    assert parent._native_child_requests == []
    assert parent.sampling_config.seed is None
    setattr(parent, input_field, None)
    parent.create_child_request(43)
    assert parent.child_requests[0].sampling_config.seed == 1
    with pytest.raises(RuntimeError, match="A child request cannot create its own child"):
        parent.child_requests[0].create_child_request(44)
    with pytest.raises(RuntimeError, match="Cannot create child requests more than"):
        parent.create_child_request(44)


def test_native_parent_retains_child_python_inputs_without_a_reference_cycle() -> None:
    parent = LlmRequest(
        request_id=42,
        input_tokens=[1, 2, 3],
        max_new_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        is_streaming=False,
        position_ids=[7, 8, 9],
    )
    parent.create_child_request(43)
    child_ref = weakref.ref(parent.child_requests[0])
    parent_ref = weakref.ref(parent)
    parent.child_requests.clear()
    gc.collect()
    assert child_ref() is not None
    assert child_ref().py_position_ids == [7, 8, 9]
    del parent
    gc.collect()
    assert parent_ref() is None
    assert child_ref() is None


def test_executor_request_to_llm_request_adopts_context_phase_draft_tokens() -> None:
    request_id = 42
    first_gen_tokens = [100]
    draft_tokens = [101, 102, 103]
    context_phase_params = trtllm.ContextPhaseParams(
        first_gen_tokens,
        request_id,
        None,
        draft_tokens,
        None,
        None,
    )
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        type=trtllm.RequestType.REQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=context_phase_params,
    )

    llm_request = executor_request_to_llm_request(
        request_id,
        executor_request,
        child_req_ids=[],
        exclude_last_generation_logits=False,
    )

    assert llm_request.is_generation_only_request
    assert llm_request.has_draft_tokens()
    assert llm_request.num_draft_tokens == len(draft_tokens)
    assert llm_request.draft_tokens == draft_tokens
    assert llm_request.py_draft_tokens == draft_tokens
    assert llm_request.context_phase_params.draft_tokens == draft_tokens


_I32_PROMPT = [7, 0, 2**31 - 1, 3, 151_643]


def test_executor_request_to_llm_request_ingests_int32_tokens() -> None:
    executor_request = trtllm.Request(input_token_ids=_I32_PROMPT, max_tokens=4)
    tokens = executor_request.get_input_token_ids_array()
    assert tokens.dtype == np.int32 and tokens.tolist() == _I32_PROMPT
    tokens[0] = -5  # an owned copy
    assert executor_request.input_token_ids == _I32_PROMPT

    llm_request = executor_request_to_llm_request(
        3, executor_request, child_req_ids=[], exclude_last_generation_logits=False
    )
    assert llm_request.get_tokens(0) == _I32_PROMPT
    assert llm_request.orig_prompt_len == len(_I32_PROMPT)


@pytest.mark.parametrize(
    "dtype, stride", [(np.int32, 1), (np.int64, 1), (np.int32, 2)], ids=["i32", "i64", "strided"]
)
def test_llm_request_ctor_copies_token_ndarray(dtype, stride) -> None:
    # Only a contiguous int32 array is memcpy'd; any other array takes the sequence cast.
    tokens = np.repeat(np.asarray(_I32_PROMPT, dtype=dtype), stride)[::stride]
    request = CppLlmRequest(
        request_id=1,
        max_new_tokens=4,
        input_tokens=tokens,
        sampling_config=trtllm.SamplingConfig(),
        is_streaming=False,
    )
    tokens[:] = 0
    assert request.get_tokens(0) == _I32_PROMPT


def test_executor_request_to_llm_request_owns_end_id_in_python() -> None:
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        end_id=7,
    )

    llm_request = executor_request_to_llm_request(
        42,
        executor_request,
        child_req_ids=[43],
        exclude_last_generation_logits=False,
    )

    assert llm_request.py_end_id == 7
    assert llm_request.child_requests[0].py_end_id == 7


@pytest.mark.parametrize("shape", [(12, 8), (12, 3, 2, 2)])
@pytest.mark.parametrize("output_length", [None, 6])
def test_encoder_feature_metadata_drives_native_request(shape, output_length) -> None:
    features = torch.ones(shape)
    request = LlmRequest(
        request_id=42,
        max_new_tokens=10,
        input_tokens=[1, 2, 3],
        sampling_config=trtllm.SamplingConfig(),
        is_streaming=False,
        encoder_input_tokens=[4, 5],
        encoder_input_features=features,
        encoder_output_len=output_length,
    )
    # Feature length takes precedence over token length, while explicit output
    # length accounts for encoder downsampling.
    assert request.encoder_output_len == (12 if output_length is None else output_length)
    assert request.py_encoder_input_features is features
    request.validate(10, 30, 0)
    request.pause(20)
    assert request.state.name == "ENCODER_INIT"
    assert request.py_encoder_input_features is features


def test_executor_request_to_llm_request_owns_encoder_features_in_python() -> None:
    features = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=3),
    )
    executor_request.py_multimodal_data = {
        "audio": {"encoder_input_features": features, "encoder_output_len": 6}
    }
    llm_request = executor_request_to_llm_request(
        42, executor_request, child_req_ids=[43, 44], exclude_last_generation_logits=False
    )
    child, sibling = llm_request.child_requests
    assert llm_request.py_encoder_input_features is features
    for request in (llm_request, child, sibling):
        assert request.py_encoder_input_features is features
        assert request.py_encoder_input_features.data_ptr() == features.data_ptr()
        assert request.encoder_output_len == 6
        assert request.state.name == "ENCODER_INIT"
        request.validate(10, 30, 0)
        request.pause(20)
        assert request.state.name == "ENCODER_INIT"
        assert request.encoder_output_len == 6

        assert request.py_encoder_input_features is features

    child.py_encoder_input_features = child.py_encoder_input_features.clone()
    child.py_encoder_input_features.add_(1)
    assert child.py_encoder_input_features.data_ptr() != features.data_ptr()
    torch.testing.assert_close(child.py_encoder_input_features, features + 1)
    assert llm_request.py_encoder_input_features is features
    assert sibling.py_encoder_input_features is features
    torch.testing.assert_close(features, torch.arange(96, dtype=torch.float32).reshape(12, 8))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", [LogprobMode.RAW, LogprobMode.PROCESSED])
def test_logprob_flag_is_python_owned(enabled, mode) -> None:
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        output_config=trtllm.OutputConfig(return_log_probs=enabled),
    )
    request = executor_request_to_llm_request(
        42, executor_request, child_req_ids=[43], exclude_last_generation_logits=False
    )
    assert request.py_return_log_probs is enabled
    assert request.child_requests[0].py_return_log_probs is enabled
    assert not hasattr(request, "return_log_probs")
    assert (request.py_result._log_probs is not None) is enabled
    if enabled:
        assert request.child_requests[0].py_result._log_probs is not request.py_result._log_probs
    request.py_logprobs_mode = mode
    request.py_seq_slot = 0
    slot_params = SlotSamplingParams(num_slots=1, device="cpu")
    slot_params.admit([request], torch.tensor([0]))
    assert slot_params.need_raw_logprobs == [enabled and mode == LogprobMode.RAW]
    assert slot_params.need_processed_logprobs == [enabled and mode == LogprobMode.PROCESSED]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("beam_width", [1, 2])
@pytest.mark.parametrize("simple_format", [False, True])
def test_python_logprob_response_snapshots(streaming, beam_width, simple_format) -> None:
    request = LlmRequest(
        request_id=42,
        max_new_tokens=10,
        input_tokens=[1, 2, 3],
        sampling_config=trtllm.SamplingConfig(beam_width=beam_width),
        is_streaming=streaming,
        return_log_probs=True,
        logprobs_simple_format=simple_format,
    )
    request.state = (
        LlmRequestState.GENERATION_IN_PROGRESS if streaming else LlmRequestState.GENERATION_COMPLETE
    )
    first_probs = [[-0.5] if simple_format else [{4: Logprob(-0.5)}] for _ in range(beam_width)]
    next_probs = [[-0.25] if simple_format else [{5: Logprob(-0.25)}] for _ in range(beam_width)]
    request.add_new_tokens([4] * beam_width)
    request.py_result.set_log_probs(first_probs, [-0.5] * beam_width)
    first_response = request.create_response()
    assert first_response is not None
    # Exercise the IPC snapshot boundary for beam-search results, which replace
    # all logprobs each iteration rather than emitting greedy streaming deltas.
    snapshot = pickle.loads(pickle.dumps(first_response))
    native_result = first_response.result.get_result()
    assert not hasattr(native_result, "log_probs")
    assert not hasattr(native_result, "cum_log_probs")
    assert snapshot.result.log_probs == first_probs
    assert snapshot.result.cum_log_probs == [-0.5] * beam_width
    request.add_new_tokens([5] * beam_width)
    if beam_width > 1:
        request.py_result.set_log_probs(
            [first + next_ for first, next_ in zip(first_probs, next_probs)],
            [-0.75] * beam_width,
        )
    else:
        request.py_result.append_log_probs(next_probs)
    request.state = LlmRequestState.GENERATION_COMPLETE
    final_response = request.create_response()
    assert final_response is not None
    expected = (
        next_probs
        if streaming and beam_width == 1
        else [first + next_ for first, next_ in zip(first_probs, next_probs)]
    )
    assert final_response.result.log_probs == expected
    assert final_response.result.cum_log_probs == [-0.75] * beam_width
    assert snapshot.result.log_probs == first_probs
    assert snapshot.result.cum_log_probs == [-0.5] * beam_width
    if streaming and beam_width == 1:
        assert first_response.result.log_probs == first_probs
        assert first_response.result.cum_log_probs == [-0.5]


@pytest.mark.parametrize("streaming", [False, True])
def test_optional_tensor_results_are_python_owned(streaming) -> None:
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=4,
        streaming=streaming,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        output_config=trtllm.OutputConfig(
            return_context_logits=True,
            return_generation_logits=True,
            return_encoder_output=True,
            additional_model_outputs=[trtllm.AdditionalModelOutput("hidden", True)],
        ),
    )
    request = executor_request_to_llm_request(
        42, executor_request, child_req_ids=[43], exclude_last_generation_logits=False
    )
    assert request.py_return_context_logits
    assert request.py_return_generation_logits
    assert request.py_return_encoder_output
    assert request.child_requests[0].py_return_encoder_output
    for name in ("return_context_logits", "return_generation_logits", "alloc_context_logits"):
        assert not hasattr(request, name)
    context = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    generation = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
    encoder = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    # Preallocate CPU storage so these response tests do not require CUDA or
    # pinned-memory allocation, while exercising the real append/get methods.
    request.py_result._context_logits._storage = torch.empty(3, 1, 4)
    request.py_result._context_logits.beam_width = 1
    request.py_result._generation_logits.use_chunked_generation_logits = False
    request.py_result._generation_logits._storage = torch.empty(4, 1, 4)
    request.py_result._generation_logits.beam_width = 1
    request.py_result.append_context_logits(context)
    request.py_result.append_generation_logits(generation)
    request.py_result.set_encoder_output(encoder)
    request.py_result.append_additional_context_outputs("hidden", context)
    request.py_result.append_additional_generation_outputs("hidden", generation)
    request.add_new_token(4, 0)
    request.state = LlmRequestState.GENERATION_COMPLETE
    response = request.create_response()
    assert response is not None
    native = response.result.get_result()
    for name in (
        "context_logits",
        "generation_logits",
        "encoder_output",
        "additional_outputs",
        "spec_dec_fast_logits_info",
    ):
        assert not hasattr(native, name)
    torch.testing.assert_close(response.result.context_logits, context)
    torch.testing.assert_close(response.result.generation_logits, generation.transpose(0, 1))
    torch.testing.assert_close(response.result.encoder_output, encoder)
    torch.testing.assert_close(response.result.additional_context_outputs["hidden"], context)
    torch.testing.assert_close(response.result.additional_generation_outputs["hidden"], generation)


@pytest.mark.parametrize("estimate", [None, 0, 5000])
def test_executor_request_to_llm_request_copies_uncached_prefill_estimate(estimate) -> None:
    """Router estimates survive conversion and fan-out, with independent aging."""
    request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=3),
    )
    if estimate is not None:
        request.py_uncached_prefill_tokens = estimate
    parent = executor_request_to_llm_request(
        42, request, child_req_ids=[43, 44], exclude_last_generation_logits=False
    )
    child, sibling = parent.child_requests
    assert parent.py_uncached_prefill_tokens == estimate
    assert child.py_uncached_prefill_tokens == estimate
    assert sibling.py_uncached_prefill_tokens == estimate

    # The scheduler clears an aged-out request's estimate independently.
    child.py_uncached_prefill_tokens = None
    assert parent.py_uncached_prefill_tokens == estimate
    assert sibling.py_uncached_prefill_tokens == estimate


def test_executor_request_to_llm_request_owns_position_ids_in_python() -> None:
    position_ids = [4, 7, 9]
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
    )

    llm_request = executor_request_to_llm_request(
        42,
        executor_request,
        child_req_ids=[43],
        exclude_last_generation_logits=False,
        position_ids=position_ids,
    )

    assert llm_request.py_position_ids == position_ids
    assert llm_request.child_requests[0].py_position_ids == position_ids
    assert llm_request.child_requests[0].py_position_ids is not llm_request.py_position_ids


def test_executor_request_to_llm_request_owns_guided_decoding_params_in_python() -> None:
    params = trtllm.GuidedDecodingParams(trtllm.GuidedDecodingParams.GuideType.REGEX, guide=r"\d+")
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        guided_decoding_params=params,
    )

    llm_request = executor_request_to_llm_request(
        42,
        executor_request,
        child_req_ids=[43],
        exclude_last_generation_logits=False,
    )

    assert llm_request.py_guided_decoding_params.guide == params.guide
    child_params = llm_request.child_requests[0].py_guided_decoding_params
    assert child_params.guide == params.guide
    assert child_params is not llm_request.py_guided_decoding_params


def test_executor_request_to_llm_request_owns_mrope_data_in_python() -> None:
    rotary_cos_sin = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    position_deltas = torch.tensor([-3], dtype=torch.int32)
    position_ids = torch.arange(9, dtype=torch.int32).reshape(3, 1, 3)
    executor_request = trtllm.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=10,
        sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        mrope_config=trtllm.MropeConfig(rotary_cos_sin, position_deltas),
    )
    executor_request.py_multimodal_data = {"mrope_config": {"mrope_position_ids": position_ids}}

    llm_request = executor_request_to_llm_request(
        42,
        executor_request,
        child_req_ids=[43],
        exclude_last_generation_logits=False,
    )

    mrope_config = llm_request.py_multimodal_data["mrope_config"]
    assert mrope_config["mrope_position_ids"] is position_ids
    torch.testing.assert_close(mrope_config["mrope_rotary_cos_sin"], rotary_cos_sin)
    torch.testing.assert_close(mrope_config["mrope_position_deltas"], position_deltas)
    child_config = llm_request.child_requests[0].py_multimodal_data["mrope_config"]
    torch.testing.assert_close(child_config["mrope_rotary_cos_sin"], rotary_cos_sin)
    assert child_config["mrope_rotary_cos_sin"].data_ptr() != rotary_cos_sin.data_ptr()
    torch.testing.assert_close(child_config["mrope_position_deltas"], position_deltas)
    assert child_config["mrope_position_deltas"].data_ptr() != position_deltas.data_ptr()
    torch.testing.assert_close(child_config["mrope_position_ids"], position_ids)
    assert child_config["mrope_position_ids"].data_ptr() != position_ids.data_ptr()


def _make_broadcast_item(req_id):
    return RequestQueueItem(
        req_id,
        SimpleNamespace(
            py_logits_post_processors=None,
            py_multimodal_data=None,
            py_scheduling_params=None,
            py_num_logprobs=None,
            py_disaggregated_params=None,
            py_conversation_params=None,
            py_lora_path=None,
        ),
    )


class _SingleRoundDist:
    """Fake transport recording the collectives RequestBroadcaster issues.

    ``broadcast_or_none`` behaves like the real one: the root's object comes
    back on the root, ``received`` stands for what a non-root rank gets.
    """

    def __init__(self, rank, *, world_size=2, has_pp=False, supports=True, received=None):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = world_size
        self.cp_size = 1
        self.has_pp = has_pp
        # Single PP stage that is both first and last: the PP route then only
        # runs its intra-stage tp_cp_broadcast (no send/recv chain).
        self.pp_size = 2 if has_pp else 1
        self.is_first_pp_rank = True
        self.is_last_pp_rank = True
        self.supports_single_round_broadcast = supports
        self._received = received
        self.calls = []

    def tp_cp_broadcast(self, obj, root=0, **kwargs):
        self.calls.append(("tp_cp_broadcast", obj))
        return obj

    # Resolved (not called) by the PP route on a first-and-last stage.
    def send_object(self, *args, **kwargs):
        raise AssertionError("no PP send expected on a first-and-last stage")

    isend_object = recv_object = send_object

    def broadcast_or_none(self, obj, root=0):
        self.calls.append(("broadcast_or_none", obj))
        return obj if self.rank == root else self._received

    def broadcast(self, obj, root=0, prefer_cpu=False):
        self.calls.append(("broadcast", obj, prefer_cpu))
        return obj

    # Scalar request-count collectives (main routes them through the int64
    # helpers instead of the object broadcasts).
    def broadcast_int64(self, values, root=0, prefer_cpu=False):
        values = list(values)
        self.calls.append(("broadcast_int64", values, prefer_cpu))
        return values

    def tp_cp_broadcast_int64(self, values, root=0, **kwargs):
        values = list(values)
        self.calls.append(("tp_cp_broadcast_int64", values))
        return values


def test_single_round_empty_iteration_is_one_collective():
    dist = _SingleRoundDist(rank=0)
    hang_detector = MagicMock()
    broadcaster = RequestBroadcaster(dist, hang_detector)

    new_requests, py_request_objects = broadcaster.broadcast([])

    assert (new_requests, py_request_objects) == ([], None)
    assert dist.calls == [("broadcast_or_none", None)]
    hang_detector.pause.assert_called_once()


def test_single_round_busy_iteration_sends_requests_and_py_objects_once():
    dist = _SingleRoundDist(rank=0)
    broadcaster = RequestBroadcaster(dist, MagicMock())
    items = [_make_broadcast_item(1), _make_broadcast_item(2)]

    new_requests, py_request_objects = broadcaster.broadcast(items)

    assert new_requests == items
    assert py_request_objects == broadcaster._collect_py_objects(items)
    assert len(dist.calls) == 1
    kind, payload = dist.calls[0]
    assert kind == "broadcast_or_none"
    assert payload == (items, py_request_objects)


def test_single_round_non_root_receives_payload_or_nothing():
    items = [_make_broadcast_item(7)]
    py_objects = RequestBroadcaster(_SingleRoundDist(rank=0), MagicMock())._collect_py_objects(
        items
    )

    dist = _SingleRoundDist(rank=1, received=(items, py_objects))
    assert RequestBroadcaster(dist, MagicMock()).broadcast([]) == (items, py_objects)
    assert dist.calls == [("broadcast_or_none", None)]

    dist = _SingleRoundDist(rank=1, received=None)
    assert RequestBroadcaster(dist, MagicMock()).broadcast([]) == ([], None)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"supports": False}, "transport without the single-round path"),
        ({"has_pp": True}, "pipeline route keeps probe + send/recv chain"),
        ({"world_size": 1}, "single rank never broadcasts"),
    ],
)
def test_single_round_falls_back_to_probe_when_not_applicable(kwargs, reason):
    dist = _SingleRoundDist(rank=0, **kwargs)
    broadcaster = RequestBroadcaster(dist, MagicMock())
    items = [_make_broadcast_item(1)]

    new_requests, _ = broadcaster.broadcast(items)

    assert new_requests == items, reason
    assert all(kind != "broadcast_or_none" for kind, *_ in dist.calls), reason


def test_single_round_idle_path_keeps_cpu_probe():
    """prefer_cpu marks the idle path: non-root ranks may park in the probe for
    minutes, so it must stay on the CPU transport (probe first, not the
    GPU-staged single round)."""
    dist = _SingleRoundDist(rank=0)
    broadcaster = RequestBroadcaster(dist, MagicMock())

    new_requests, py_request_objects = broadcaster.broadcast([], prefer_cpu=True)

    assert (new_requests, py_request_objects) == ([], None)
    assert dist.calls == [("broadcast_int64", [0], True)]


def test_merge_helix_requests_with_padding():
    """Test merge_helix_requests with basic valid input."""

    tokens_per_block = 2

    # Create request item with 13 tokens to get exactly 7 blocks for 4 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=5,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    for rank in [0, 1, 2, 3]:
        # Test merge_helix_requests with 4 CP ranks.
        result = merge_helix_requests(
            [request_item],
            cp_rank=rank,
            cp_size=4,
            tokens_per_block=tokens_per_block,
            exclude_last_generation_logits=False,
        )

        # Verify the result.
        assert len(result) == 1
        llm_request = result[0]
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        # Round-robin block distribution across 4 CP ranks (7 blocks total, 2 tokens/block):
        #   rank 0 owns blocks {0, 4} -> tokens [1,2, 9,10]
        #   rank 1 owns blocks {1, 5} -> tokens [3,4, 11,12]
        #   rank 2 owns blocks {2, 6} -> tokens [5,6, 13]  (block 6 is the last block; padding stripped)
        #   rank 3 owns block  {3}    -> tokens [7,8]
        if rank == 0:
            assert llm_request.get_tokens(0) == [1, 2, 9, 10]
        elif rank == 1:
            assert llm_request.get_tokens(0) == [3, 4, 11, 12]
        elif rank == 2:
            assert llm_request.get_tokens(0) == [5, 6, 13]
        else:
            assert llm_request.get_tokens(0) == [7, 8]


def test_merge_helix_requests_without_padding():
    """Test merge_helix_requests with evenly divisible tokens (no padding)."""

    tokens_per_block = 4

    # Create request item with 12 tokens to get exactly 3 blocks for 2 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=5,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    for rank in [0, 1]:
        # Test merge_helix_requests with 2 CP ranks.
        result = merge_helix_requests(
            [request_item],
            cp_rank=rank,
            cp_size=2,
            tokens_per_block=tokens_per_block,
            exclude_last_generation_logits=False,
        )

        # Verify the result.
        assert len(result) == 1
        llm_request = result[0]
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        # Round-robin block distribution across 2 CP ranks (3 blocks total, 4 tokens/block):
        #   rank 0 owns blocks {0, 2} -> tokens [1,2,3,4, 9,10,11,12]
        #   rank 1 owns block  {1}    -> tokens [5,6,7,8]
        if rank == 0:
            assert llm_request.get_tokens(0) == [1, 2, 3, 4, 9, 10, 11, 12]
        else:
            assert llm_request.get_tokens(0) == [5, 6, 7, 8]


def test_merge_helix_requests_empty_ranks():
    """When num_total_blocks < cp_size, the highest CP ranks own no blocks.

    Such "empty" ranks must produce an empty token list (and seqlen_this_rank_cp
    == 0), while total_input_len_cp still reflects the full prompt length so the
    global position ids stay correct. They are no longer rejected.
    """
    tokens_per_block = 4

    # 12 tokens -> 3 blocks, which is fewer than 4 CP ranks, so rank 3 is empty.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=12,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

    # Round-robin block distribution across 4 CP ranks (3 blocks total, 4 tokens/block):
    #   rank 0 owns block {0} -> tokens [1,2,3,4]
    #   rank 1 owns block {1} -> tokens [5,6,7,8]
    #   rank 2 owns block {2} -> tokens [9,10,11,12]
    #   rank 3 owns no blocks -> [] (empty rank)
    expected_tokens = {
        0: [1, 2, 3, 4],
        1: [5, 6, 7, 8],
        2: [9, 10, 11, 12],
        3: [],
    }
    for rank in range(4):
        result = merge_helix_requests(
            [request_item],
            cp_rank=rank,
            cp_size=4,
            tokens_per_block=tokens_per_block,
            exclude_last_generation_logits=False,
        )

        assert len(result) == 1
        llm_request = result[0]
        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        assert llm_request.get_tokens(0) == expected_tokens[rank]
        # total_input_len_cp is always the full prompt length.
        assert llm_request.total_input_len_cp == len(input_tokens)
        assert llm_request.seqlen_this_rank_cp == len(expected_tokens[rank])


def test_merge_helix_requests_empty_ranks_with_padding():
    """Exercise padding-strip-on-last-owner and empty ranks together.

    With 10 tokens and tokens_per_block=4 there are 3 blocks, the last of which
    is partially filled (tokens [9, 10]). Distributed round-robin over 4 CP
    ranks, the last block owner (rank 2) must strip the block padding while the
    block-less rank (rank 3) must be an empty rank.
    """
    tokens_per_block = 4

    # 10 tokens -> 3 blocks (last block half-full), fewer than 4 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=12,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

    # Round-robin block distribution across 4 CP ranks (3 blocks, 4 tokens/block):
    #   rank 0 owns block {0} -> tokens [1,2,3,4]
    #   rank 1 owns block {1} -> tokens [5,6,7,8]
    #   rank 2 owns block {2} -> tokens [9,10]  (last block; padding stripped)
    #   rank 3 owns no blocks -> [] (empty rank)
    expected_tokens = {
        0: [1, 2, 3, 4],
        1: [5, 6, 7, 8],
        2: [9, 10],
        3: [],
    }
    for rank in range(4):
        result = merge_helix_requests(
            [request_item],
            cp_rank=rank,
            cp_size=4,
            tokens_per_block=tokens_per_block,
            exclude_last_generation_logits=False,
        )

        assert len(result) == 1
        llm_request = result[0]
        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        assert llm_request.get_tokens(0) == expected_tokens[rank]
        # total_input_len_cp is always the full prompt length.
        assert llm_request.total_input_len_cp == len(input_tokens)
        assert llm_request.seqlen_this_rank_cp == len(expected_tokens[rank])


def _partition_by_blocks(input_tokens, cp_rank, cp_size, tokens_per_block):
    """Round-robin block ownership spelled out block by block."""
    input_len = len(input_tokens)
    num_blocks = -(-input_len // tokens_per_block)
    token_ids, position_ids = [], []
    for block in range(cp_rank, num_blocks, cp_size):
        for position in range(
            block * tokens_per_block, min((block + 1) * tokens_per_block, input_len)
        ):
            token_ids.append(input_tokens[position])
            position_ids.append(position)
    return token_ids, position_ids, input_len, -input_len % tokens_per_block


@pytest.mark.parametrize("tokens_per_block", [1, 4, 32])
@pytest.mark.parametrize("cp_size", [1, 3, 4, 8])
@pytest.mark.parametrize("input_len", [1, 31, 32, 33, 257, 4097])
def test_partition_context_for_helix_matches_block_layout(input_len, cp_size, tokens_per_block):
    rng = np.random.default_rng(input_len)
    input_tokens = rng.integers(0, 1 << 20, size=input_len, dtype=np.int32)
    for cp_rank in range(cp_size):
        expected = _partition_by_blocks(input_tokens.tolist(), cp_rank, cp_size, tokens_per_block)
        assert (
            partition_context_for_helix(input_tokens, cp_rank, cp_size, tokens_per_block)
            == expected
        )


def test_request_input_token_ids_array_matches_prompt():
    input_tokens = [5, 7, 11, 13, (1 << 31) - 1]
    executor_request = trtllm.Request(input_token_ids=input_tokens, max_tokens=1)

    tokens = executor_request.get_input_token_ids_array()

    assert tokens.dtype == np.int32
    assert tokens.tolist() == input_tokens == executor_request.input_token_ids
    # The array owns its tokens.
    del executor_request
    assert tokens.tolist() == input_tokens


@patch("tensorrt_llm._torch.pyexecutor.request_utils.executor_request_to_llm_request")
def test_merge_requests_default(mock_convert):
    """Test merging requests with default configuration."""
    mock_llm_request = Mock(child_requests=[])
    mock_convert.return_value = mock_llm_request

    requests = [RequestQueueItem(1, Mock()), RequestQueueItem(2, Mock())]
    result = merge_requests(
        requests,
        cp_config={},
        cp_rank=0,
        cp_size=1,
        exclude_last_generation_logits=False,
        is_disagg=False,
    )

    assert len(result) == 2
    assert mock_convert.call_count == 2


def test_merge_requests_with_helix_cp_config():
    """Test merge_requests routes to merge_helix_requests with HELIX cp_config."""
    tokens_per_block = 2

    # Create request item with 13 tokens to get exactly 7 blocks for 4 CP ranks.
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=5,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(
        id=1,
        request=executor_request,
    )

    cp_config = {
        "cp_type": CpType.HELIX,
        "tokens_per_block": tokens_per_block,
    }

    for rank in [0, 1, 2, 3]:
        # Test merge_requests with HELIX cp_config and 4 CP ranks.
        result = merge_requests(
            [request_item],
            cp_config=cp_config,
            cp_rank=rank,
            cp_size=4,
            exclude_last_generation_logits=False,
            is_disagg=True,
        )

        # Verify the result.
        assert len(result) == 1
        llm_request = result[0]
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

        assert isinstance(llm_request, LlmRequest)
        assert llm_request.request_id == 1
        # Round-robin block distribution across 4 CP ranks (7 blocks total, 2 tokens/block):
        #   rank 0 owns blocks {0, 4} -> tokens [1,2, 9,10]
        #   rank 1 owns blocks {1, 5} -> tokens [3,4, 11,12]
        #   rank 2 owns blocks {2, 6} -> tokens [5,6, 13]  (block 6 is the last block; padding stripped)
        #   rank 3 owns block  {3}    -> tokens [7,8]
        if rank == 0:
            assert llm_request.get_tokens(0) == [1, 2, 9, 10]
        elif rank == 1:
            assert llm_request.get_tokens(0) == [3, 4, 11, 12]
        elif rank == 2:
            assert llm_request.get_tokens(0) == [5, 6, 13]
        else:
            assert llm_request.get_tokens(0) == [7, 8]


def test_merge_requests_agg_helix_keeps_full_prompt():
    """Aggregated helix keeps the full prompt on every CP rank, so every rank
    sees identical request lengths."""
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    executor_request = trtllm.Request(
        input_token_ids=input_tokens,
        max_tokens=5,
        streaming=False,
        sampling_config=trtllm.SamplingConfig(),
        output_config=trtllm.OutputConfig(),
    )
    request_item = RequestQueueItem(id=1, request=executor_request)
    cp_config = {"cp_type": CpType.HELIX, "tokens_per_block": 2}

    for rank in range(4):
        result = merge_requests(
            [request_item],
            cp_config=cp_config,
            cp_rank=rank,
            cp_size=4,
            exclude_last_generation_logits=False,
            is_disagg=False,
        )

        assert len(result) == 1
        llm_request = result[0]
        assert llm_request.get_tokens(0) == input_tokens
        assert llm_request.prompt_len == len(input_tokens)
        assert llm_request.total_input_len_cp == len(input_tokens)


def test_get_from_waiting_queue():
    """Test getting items from waiting queue."""
    # Add items to waiting queue
    waiting_queue = FCFSWaitingQueue()
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    waiting_queue.extend(items)

    # Get 3 items
    result = get_from_waiting_queue(
        waiting_queue, 3, enable_attention_dp=False, max_num_active_requests=16
    )

    assert len(result) == 3
    assert result == items[:3]
    assert len(waiting_queue) == 2


@pytest.mark.parametrize(
    "queue_size,request_count,expected_result,expected_remaining",
    [
        (0, 5, 0, 0),  # Empty queue
        (3, -1, 0, 3),  # Negative count
        (3, 0, 0, 3),  # Zero count
        (3, 10, 3, 0),  # Request more than available
    ],
)
def test_get_from_waiting_queue_edge_cases(
    queue_size, request_count, expected_result, expected_remaining
):
    """Test edge cases for getting items from waiting queue."""
    # Setup queue
    waiting_queue = FCFSWaitingQueue()
    if queue_size > 0:
        items = [RequestQueueItem(i, Mock()) for i in range(queue_size)]
        waiting_queue.extend(items)

    result = get_from_waiting_queue(
        waiting_queue, request_count, enable_attention_dp=False, max_num_active_requests=16
    )

    assert len(result) == expected_result
    assert len(waiting_queue) == expected_remaining


def test_get_from_waiting_queue_with_attention_dp(
    attention_dp_config, all_ranks_num_active_requests
):
    waiting_queue = FCFSWaitingQueue()
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    waiting_queue.extend(items)

    result = get_from_waiting_queue(
        waiting_queue,
        3,
        True,
        attention_dp_config["max_num_active_requests"],
        all_ranks_num_active_requests,
    )

    assert len(result) == 3
    assert result == items[:3]
    assert len(waiting_queue) == 2


def test_get_from_waiting_queue_with_attention_dp_filtering(
    attention_dp_config, all_ranks_num_active_requests
):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    req2 = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=True)
    )
    req3 = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=None)
    )  # No scheduling params

    waiting_queue = FCFSWaitingQueue()
    waiting_queue.extend([req1, req2, req3])

    # Set rank 0 to full capacity to test filtering
    all_ranks_num_active_requests[0] = 8

    result = get_from_waiting_queue(
        waiting_queue,
        3,
        True,
        attention_dp_config["max_num_active_requests"],
        all_ranks_num_active_requests,
    )

    assert len(result) == 2
    assert req2 in result
    assert req3 in result
    assert req1 not in result


def test_can_process_attention_dp_request(attention_dp_config):
    max_num_active_requests = attention_dp_config["max_num_active_requests"]

    req_no_params = RequestQueueItem(1, Mock())
    assert can_process_attention_dp_request(req_no_params, [0, 0, 0, 0], max_num_active_requests)

    req_relax = RequestQueueItem(
        2, create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=True)
    )
    assert can_process_attention_dp_request(req_relax, [0, 0, 0, 0], max_num_active_requests)

    req_target = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1, attention_dp_relax=False),
    )
    all_ranks = [0, 0, 0, 0]
    assert can_process_attention_dp_request(req_target, all_ranks, max_num_active_requests)
    assert all_ranks[1] == 1

    req_no_capacity = RequestQueueItem(
        4,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0, attention_dp_relax=False),
    )
    all_ranks_full = [8, 0, 0, 0]  # Rank 0 is at capacity
    assert not can_process_attention_dp_request(
        req_no_capacity, all_ranks_full, max_num_active_requests
    )


# --------------------------------------------------------------------------
# nvbug-6133201: per-rank gen-phase step-token cap via tightened
# per-rank request cap.
# --------------------------------------------------------------------------
# Under enable_attention_dp the global Python scheduler caps tokens
# cluster-wide and the ADP router caps per-rank requests, but no
# component caps per-rank gen-phase step-tokens.  PyExecutor tightens
# the per-rank request cap to
# ``max_num_tokens // (1 + max_total_draft_tokens)`` so per-rank step-
# token load cannot exceed max_num_tokens by construction.


class TestDeriveAttentionDpPerRankRequestCap:
    """Unit tests for ``derive_attention_dp_per_rank_request_cap``.

    The fix for nvbug-6133201 is the cap arithmetic implemented in this
    helper; PyExecutor calls it once at ``__init__`` and the result
    flows through the existing ``max_num_active_requests`` plumbing.
    """

    def test_no_tightening_when_max_num_tokens_is_none(self):
        # LlmArgs.max_num_tokens == None -> helper is a no-op.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=None, max_total_draft_tokens=3
            )
            == 128
        )

    def test_nvbug_6133201_failing_config(self):
        # nvbug-6133201 numbers: max_batch_size=128,
        # max_total_draft_tokens=3 (MTP3), max_num_tokens=256.
        # Per-rank step-token cost per req = 1 + 3 = 4.
        # Effective cap = 256 // 4 = 64; per-rank load at saturation
        # 64 * 4 = 256 = max_num_tokens, so the per-rank assert in
        # model_engine.py cannot trip on gen-phase accumulation.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=256, max_total_draft_tokens=3
            )
            == 64
        )

    def test_no_tightening_when_arithmetic_already_fits(self):
        # Correctly-sized LlmArgs (max_batch_size * (1+max_total_draft_tokens)
        # <= max_num_tokens): cap == base_cap, no behavioral change.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=512, max_total_draft_tokens=3
            )
            == 128
        )
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=4096, max_total_draft_tokens=3
            )
            == 128
        )

    def test_no_spec_decoding(self):
        # max_total_draft_tokens == 0: step-token cost per req == 1,
        # effective cap == max_num_tokens.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=64, max_total_draft_tokens=0
            )
            == 64
        )

    def test_negative_max_total_draft_tokens_clamped(self):
        # Defensive: a stray negative value must not yield div-by-zero
        # or a negative cap.
        assert (
            derive_attention_dp_per_rank_request_cap(
                base_cap=128, max_num_tokens=256, max_total_draft_tokens=-5
            )
            == 128
        )


def test_executor_request_to_llm_request_adopts_return_routed_experts() -> None:
    def _build(flag):
        executor_request = trtllm.Request(input_token_ids=[1, 2, 3], max_tokens=4)
        if flag is not None:
            # Set by BaseWorker._enqueue_request from SamplingParams (py_* side channel).
            executor_request.py_return_routed_experts = flag
        return executor_request_to_llm_request(
            7, executor_request, child_req_ids=[], exclude_last_generation_logits=False
        )

    assert _build(None).py_return_routed_experts is False  # default: opted out
    assert _build(True).py_return_routed_experts is True


class TestBroadcasterClock:
    """RequestBroadcaster(sync_clock=True): rank 0's clock and aux ride along
    with the request count."""

    def _dist(self, rank, world_size=2):
        dist = Mock()
        dist.rank = rank
        dist.world_size = world_size
        dist.has_pp = False
        dist.supports_single_round_broadcast = False
        return dist

    def _hang_detector(self):
        hd = Mock()
        hd.pause.return_value.__enter__ = Mock(return_value=None)
        hd.pause.return_value.__exit__ = Mock(return_value=False)
        return hd

    def test_default_probe_unchanged(self):
        dist = self._dist(rank=1)
        dist.broadcast_int64 = Mock(return_value=[0])
        b = RequestBroadcaster(dist, self._hang_detector())
        assert b.broadcast([]) == ([], None)
        (values,), _ = dist.broadcast_int64.call_args
        assert list(values) == [0]
        assert b.rank0_now is None

    def test_sync_clock_keeps_the_count_probe(self):
        """The single-round broadcast has no room for the clock."""
        for sync_clock, probes in ((False, 0), (True, 1)):
            dist = self._dist(rank=1)
            dist.supports_single_round_broadcast = True
            dist.broadcast_or_none = Mock(return_value=None)
            dist.broadcast_int64 = Mock(return_value=[0, 1_000_000, -1])
            b = RequestBroadcaster(dist, self._hang_detector(), sync_clock=sync_clock)
            assert b.broadcast([]) == ([], None)
            assert dist.broadcast_int64.call_count == probes
            assert dist.broadcast_or_none.call_count == 1 - probes

    def test_default_probe_pp_forwards_count(self):
        dist = self._dist(rank=4, world_size=8)
        dist.has_pp = True
        dist.is_first_pp_rank = False
        dist.is_last_pp_rank = False
        dist.pp_size = 2
        dist.prev_pp_rank, dist.next_pp_rank = 0, 1
        dist.recv_object = Mock(return_value=[0])
        dist.send_object = Mock()
        b = RequestBroadcaster(dist, self._hang_detector())
        assert b.broadcast([]) == ([], None)
        dist.send_object.assert_called_once_with([0], 1, 3)
        assert b.rank0_now is None

    def test_sync_clock_single_rank(self):
        dist = self._dist(rank=0, world_size=1)
        b = RequestBroadcaster(dist, self._hang_detector(), sync_clock=True)
        assert b.broadcast([]) == ([], None)
        assert b.rank0_now is not None and b.rank0_now > 0

    def test_sync_clock_pp_forwards_header(self):
        dist = self._dist(rank=4, world_size=8)
        dist.has_pp = True
        dist.is_first_pp_rank = False
        dist.is_last_pp_rank = False
        dist.pp_size = 2
        dist.prev_pp_rank, dist.next_pp_rank = 0, 1
        dist.recv_object = Mock(return_value=[0, 5_000_000, -1])
        dist.send_object = Mock()
        b = RequestBroadcaster(dist, self._hang_detector(), sync_clock=True)
        assert b.broadcast([]) == ([], None)
        dist.send_object.assert_called_once_with([0, 5_000_000, -1], 1, 3)
        assert b.rank0_now == pytest.approx(5.0)

    def test_sync_clock_aux_rides_along(self):
        root = self._dist(rank=0)
        root.broadcast_int64 = Mock(side_effect=lambda header, root, prefer_cpu: header)
        b = RequestBroadcaster(root, self._hang_detector(), sync_clock=True)
        b.aux = 0.25
        b.broadcast([])
        (values,), _ = root.broadcast_int64.call_args
        assert len(values) == 3 and values[2] == 250_000
        assert b.rank0_aux == pytest.approx(0.25)
        b.aux = None  # nothing to send yet
        b.broadcast([])
        assert b.rank0_aux is None

        other = self._dist(rank=1)
        other.broadcast_int64 = Mock(return_value=[0, 1_000_000, 170_000])
        b = RequestBroadcaster(other, self._hang_detector(), sync_clock=True)
        b.aux = 9.0  # ignored off rank 0
        b.broadcast([])
        (values,), _ = other.broadcast_int64.call_args
        assert list(values) == [0, 0, -1]
        assert b.rank0_aux == pytest.approx(0.17) and b.rank0_now == pytest.approx(1.0)
