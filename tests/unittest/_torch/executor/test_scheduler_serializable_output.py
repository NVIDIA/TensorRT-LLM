import pickle

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests, SerializableSchedulerOutput


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
    scheduled_requests.context_requests = [request_pool[1], request_pool[2]]
    scheduled_requests.generation_requests = [request_pool[3]]
    scheduled_requests.paused_requests = [request_pool[4]]
    fitting_disagg_gen_init_requests = [request_pool[5], request_pool[6]]
    num_fitting_requests = 3

    # Create serializable scheduler output from scheduler result
    serializable_output = SerializableSchedulerOutput.from_scheduler_result(
        scheduled_requests, fitting_disagg_gen_init_requests, num_fitting_requests
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
    assert _request_ids(restored_schedule.context_requests) == _request_ids(
        scheduled_requests.context_requests
    )
    assert _request_ids(restored_schedule.generation_requests) == _request_ids(
        scheduled_requests.generation_requests
    )
    assert _request_ids(restored_schedule.paused_requests) == _request_ids(
        scheduled_requests.paused_requests
    )
    assert _request_ids(restored_fitting) == _request_ids(fitting_disagg_gen_init_requests)
