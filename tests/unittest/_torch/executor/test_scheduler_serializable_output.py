import pickle

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests, SerializableSchedulerOutput

pytestmark = pytest.mark.cpu_only


def _make_request(request_id: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=5,
        input_tokens=[request_id],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _request_ids(requests):
    return [req.request_id for req in requests]


def test_serializable_scheduler_output_round_trip():
    # Create all requests and put them in a pool
    request_pool = {idx: _make_request(idx) for idx in range(1, 8)}

    # Create scheduler result: scheduled_requests, fitting_disagg_gen_init_requests, num_fitting_requests
    scheduled_requests = ScheduledRequests()
    scheduled_requests.encoder_requests = [request_pool[7]]
    scheduled_requests.context_requests_last_chunk = [request_pool[1], request_pool[2]]
    scheduled_requests.generation_requests = [request_pool[3]]
    scheduled_requests.paused_requests = [request_pool[4]]
    scheduled_requests.scheduled_mm_encoder_items = {1: [0, 2], 7: [1]}
    scheduled_requests.recompute_paused_requests = [request_pool[7]]
    fitting_disagg_gen_init_requests = [request_pool[5], request_pool[6]]
    num_fitting_requests = 3

    # Create serializable scheduler output from scheduler result
    serializable_output = SerializableSchedulerOutput.from_scheduler_result(
        scheduled_requests,
        fitting_disagg_gen_init_requests,
        num_fitting_requests,
        wait_for_disagg_gen_transfer_progress=True,
    )

    # Serialize and deserialize the serializable scheduler output
    serialized_bytes = pickle.dumps(serializable_output)
    restored_output: SerializableSchedulerOutput = pickle.loads(serialized_bytes)

    # Restore the scheduler result from the deserialized serializable scheduler output
    active_requests = list(request_pool.values())
    restored_schedule, restored_fitting, restored_num_fitting = restored_output.to_scheduler_result(
        active_requests
    )

    # Verify the restored scheduler result is correct
    assert restored_num_fitting == num_fitting_requests
    assert restored_output.wait_for_disagg_gen_transfer_progress
    assert _request_ids(restored_schedule.encoder_requests) == _request_ids(
        scheduled_requests.encoder_requests
    )
    assert _request_ids(restored_schedule.context_requests_chunking) == _request_ids(
        scheduled_requests.context_requests_chunking
    )
    assert _request_ids(restored_schedule.context_requests_last_chunk) == _request_ids(
        scheduled_requests.context_requests_last_chunk
    )
    assert _request_ids(restored_schedule.generation_requests) == _request_ids(
        scheduled_requests.generation_requests
    )
    assert _request_ids(restored_schedule.paused_requests) == _request_ids(
        scheduled_requests.paused_requests
    )
    assert (
        restored_schedule.scheduled_mm_encoder_items
        == scheduled_requests.scheduled_mm_encoder_items
    )
    assert _request_ids(restored_schedule.recompute_paused_requests) == _request_ids(
        scheduled_requests.recompute_paused_requests
    )
    assert _request_ids(restored_fitting) == _request_ids(fitting_disagg_gen_init_requests)


# The instance state of a schedule message without the pipeline-parallel knobs:
# exactly the fields the message had before those knobs existed, in this order.
_BASE_STATE_KEYS = [
    "encoder_requests",
    "context_requests_chunking",
    "context_requests_last_chunk",
    "generation_requests",
    "paused_requests",
    "fitting_disagg_gen_init_requests",
    "num_fitting_requests",
    "wait_for_disagg_gen_transfer_progress",
    "scheduled_mm_encoder_items",
    "recompute_paused_requests",
]
_PP_STATE_KEYS = [
    "context_chunks",
    "context_first_chunks",
    "reuse_proposals",
    "context_admission_order",
    "subagent_idle_drops",
    "longchain_ops",
]
_LEADER_ATTRS = [
    "leader_context_chunks",
    "leader_first_context_chunks",
    "leader_reuse_proposals",
    "leader_context_admission_order",
    "leader_subagent_idle_drops",
    "leader_longchain_ops",
]


def _context_batch():
    request_pool = {idx: _make_request(idx) for idx in range(1, 4)}
    scheduled = ScheduledRequests()
    scheduled.context_requests_last_chunk = [request_pool[1], request_pool[2]]
    scheduled.generation_requests = [request_pool[3]]
    return scheduled, list(request_pool.values())


def test_message_without_pp_knobs_carries_only_the_base_fields():
    scheduled, active = _context_batch()
    message = SerializableSchedulerOutput.from_scheduler_result(scheduled, [], 2)
    assert list(vars(message)) == _BASE_STATE_KEYS
    restored = pickle.loads(pickle.dumps(message))
    assert list(vars(restored)) == _BASE_STATE_KEYS
    assert not restored.carries_pp_fields()
    assert restored.context_chunks is None and restored.longchain_ops is None
    batch, _, _ = restored.to_scheduler_result(active)
    assert not any(name in vars(batch) for name in _LEADER_ATTRS)
    assert all(getattr(batch, name) is None for name in _LEADER_ATTRS)
    assert not any(name in vars(ScheduledRequests()) for name in _LEADER_ATTRS)


def test_message_with_a_pp_knob_carries_all_six_fields_in_order():
    scheduled, active = _context_batch()
    message = SerializableSchedulerOutput.from_scheduler_result(
        scheduled, [], 2, record_context_chunks=True, attach_pp_fields=True
    )
    assert list(vars(message)) == _BASE_STATE_KEYS + _PP_STATE_KEYS
    restored = pickle.loads(pickle.dumps(message))
    assert restored.carries_pp_fields()
    assert set(restored.context_chunks) == {1, 2}
    assert restored.context_first_chunks is None and restored.subagent_idle_drops is None
    batch, _, _ = restored.to_scheduler_result(active)
    assert set(_LEADER_ATTRS) <= set(vars(batch))
    assert batch.leader_context_chunks == restored.context_chunks
    assert batch.leader_first_context_chunks is None


def test_knob_on_pass_with_nothing_to_send_still_carries_the_fields():
    scheduled, _ = _context_batch()
    message = SerializableSchedulerOutput.from_scheduler_result(
        scheduled, [], 2, attach_pp_fields=True
    )
    assert list(vars(message)) == _BASE_STATE_KEYS + _PP_STATE_KEYS
    assert all(getattr(message, name) is None for name in _PP_STATE_KEYS)
