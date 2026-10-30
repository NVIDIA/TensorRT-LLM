# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for request batching, independent RNG streams and response routing."""

import asyncio
import queue
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import zmq
from diffusers.utils.torch_utils import randn_tensor
from pydantic import ValidationError

from tensorrt_llm._torch.visual_gen.batching import (
    BatchInfo,
    DynamicBatchScheduler,
    randn_tensor_for_batch,
    request_batch_key,
    split_batch_output,
)
from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionExecutor,
    DiffusionRemoteClient,
    DiffusionRequest,
    DiffusionResponse,
)
from tensorrt_llm._torch.visual_gen.output import PipelineOutput
from tensorrt_llm.visual_gen import BatchingConfig, MediaRef, VisualGenArgs, VisualGenParams

pytestmark = pytest.mark.cpu_only


def _request(request_id=1, *, prompts=None, **kwargs):
    return DiffusionRequest(
        request_id=request_id,
        prompt=prompts or [f"prompt {request_id}"],
        params=VisualGenParams(seed=request_id, **kwargs),
    )


class _Queue:
    """Virtual-time receive queue, including requests that arrive during a poll."""

    def __init__(self, requests):
        self.requests = deque(requests)
        self.now = 0.0
        self.poll_timeouts = []

    def receive(self):
        arrival, req = self.requests.popleft()
        self.now = max(self.now, arrival)
        return req

    def poll(self, timeout):
        self.poll_timeouts.append(timeout)
        if self.requests and self.requests[0][0] <= self.now + timeout:
            self.now = max(self.now, self.requests[0][0])
            return True
        self.now += timeout
        return False

    def clock(self):
        return self.now


def _next(scheduler, requests, *, enabled=True):
    return scheduler.next_request(requests.receive, requests.poll, enabled=enabled)


def test_defaults_and_yaml_round_trip(tmp_path):
    defaults = VisualGenArgs()
    assert defaults.batching_config.max_batch_size == 1
    assert defaults.batching_config.max_queue_delay_ms == 0
    path = tmp_path / "batching.yaml"
    path.write_text("batching_config:\n  max_batch_size: 4\n  max_queue_delay_ms: 5\n")
    args = VisualGenArgs.from_yaml(path)
    assert args.batching_config == BatchingConfig(max_batch_size=4, max_queue_delay_ms=5)
    assert VisualGenArgs.model_validate(args.model_dump()) == args


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_batch_size": 0},
        {"max_batch_size": -1},
        {"max_batch_size": 1.5},
        {"max_queue_delay_ms": -1},
        {"max_queue_delay_ms": float("inf")},
        {"max_queue_delay_ms": float("nan")},
        {"unknown": 4},
    ],
)
def test_invalid_batching_config(kwargs):
    with pytest.raises(ValidationError):
        BatchingConfig(**kwargs)


@pytest.mark.parametrize("enabled,limit", [(False, 4), (True, 1)])
def test_disabled_does_not_poll_or_change_request(enabled, limit):
    req = _request()
    requests = _Queue([(0, req), (0, None)])
    scheduler = DynamicBatchScheduler(limit, 500, clock=requests.clock)
    assert _next(scheduler, requests, enabled=enabled) is req
    assert requests.poll_timeouts == []
    assert _next(scheduler, requests, enabled=enabled) is None


def test_full_batch_dispatches_without_waiting_and_preserves_inputs():
    original = [_request(i) for i in range(3)]
    requests = _Queue([(0, req) for req in original])
    scheduler = DynamicBatchScheduler(3, 100, clock=requests.clock)
    merged = _next(scheduler, requests)
    assert merged.prompt == ["prompt 0", "prompt 1", "prompt 2"]
    assert merged.batch_info == BatchInfo((0, 1, 2), (1, 1, 1), (0, 1, 2))
    assert requests.now == 0
    assert original[0].prompt == ["prompt 0"]
    assert original[0].batch_info is None
    assert merged.prepared_inputs is not original[0].prepared_inputs
    merged.params.height = 512
    assert original[0].params.height is None


def test_zero_delay_merges_only_already_queued_requests():
    requests = _Queue([(0, _request(1)), (0, _request(2)), (0.001, _request(3))])
    scheduler = DynamicBatchScheduler(4, 0, clock=requests.clock)
    merged = _next(scheduler, requests)
    assert merged.batch_info.request_ids == (1, 2)
    assert requests.poll_timeouts == [0, 0]
    assert requests.now == 0


def test_arrivals_share_one_deadline():
    requests = _Queue([(0, _request(1)), (0.003, _request(2)), (0.006, _request(3))])
    scheduler = DynamicBatchScheduler(4, 5, clock=requests.clock)
    merged = _next(scheduler, requests)
    assert merged.batch_info.request_ids == (1, 2)
    assert requests.now == pytest.approx(0.005)
    assert requests.poll_timeouts == pytest.approx([0.005, 0.002])


def test_incompatible_requests_keep_fifo_order():
    first, second, third = _request(1), _request(2, width=512), _request(3)
    requests = _Queue([(0, first), (0, second), (0, third), (0, None)])
    scheduler = DynamicBatchScheduler(4, 5, clock=requests.clock)
    assert _next(scheduler, requests) is first
    assert _next(scheduler, requests) is second
    assert _next(scheduler, requests) is third
    assert _next(scheduler, requests) is None


def test_pending_request_does_not_start_a_new_wait_window():
    first, second = _request(1), _request(2, width=512)
    requests = _Queue([(0, first), (0, second)])
    scheduler = DynamicBatchScheduler(4, 5, clock=requests.clock)
    assert _next(scheduler, requests) is first
    requests.now = 1.0  # Time spent running the first pipeline call.
    assert _next(scheduler, requests) is second
    assert requests.poll_timeouts[-1] == 0


def test_limit_counts_images_including_prompt_lists():
    first = _request(1, prompts=["a", "b"], num_images_per_prompt=2)
    second = _request(2, num_images_per_prompt=2)
    third = _request(3, num_images_per_prompt=2)
    requests = _Queue([(0, req) for req in [first, second, third, None]])
    scheduler = DynamicBatchScheduler(7, clock=requests.clock)
    merged = _next(scheduler, requests)
    assert merged.batch_info.sample_counts == (4, 2)
    assert merged.prompt == ["a", "b", "prompt 2"]
    assert _next(scheduler, requests) is third
    assert _next(scheduler, requests) is None


def test_oversized_single_request_runs_unchanged():
    req = _request(num_images_per_prompt=8)
    requests = _Queue([(0, req)])
    assert _next(DynamicBatchScheduler(4), requests) is req
    assert requests.poll_timeouts == []


def test_shutdown_flushes_partial_batch_before_stopping():
    requests = _Queue([(0, _request(1)), (0, _request(2)), (0, None)])
    scheduler = DynamicBatchScheduler(4, 100, clock=requests.clock)
    assert _next(scheduler, requests).batch_info.request_ids == (1, 2)
    assert _next(scheduler, requests) is None
    assert not requests.requests


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 512},
        {"width": 512},
        {"num_inference_steps": 4},
        {"guidance_scale": 2.0},
        {"max_sequence_length": 128},
        {"negative_prompt": "blurry"},
        {"num_frames": 1},
        {"frame_rate": 16.0},
        {"num_images_per_prompt": 2},
        {"extra_params": {"foo": [1, 2]}},
    ],
)
def test_sampling_parameters_are_part_of_compatibility_key(kwargs):
    assert request_batch_key(_request(1)) != request_batch_key(_request(2, **kwargs))


def test_prompt_and_seed_are_independent_but_caller_intent_is_preserved():
    assert request_batch_key(_request(1)) == request_batch_key(_request(2))
    assert request_batch_key(_request(1)) != request_batch_key(_request(2, height=None))


@pytest.mark.parametrize("seed", [-(1 << 63) - 1, 1 << 64])
def test_invalid_seed_does_not_fail_compatible_peers(seed):
    req = _request()
    req.params.seed = seed
    assert request_batch_key(req) is None


@pytest.mark.parametrize("slot", ["image_reference", "video_reference", "audio_reference"])
def test_references_are_never_merged(slot):
    req = _request(**{slot: MediaRef(content=b"content", format="bytes")})
    requests = _Queue([(0, req)])
    assert _next(DynamicBatchScheduler(4, 100), requests) is req
    assert requests.poll_timeouts == []


@pytest.mark.parametrize("reason", ["unseeded", "prepared", "handles", "opaque_extra", "nested"])
def test_unsafe_requests_remain_singletons(reason):
    req = _request()
    if reason == "unseeded":
        req.params.seed = None
    elif reason == "prepared":
        req.prepared_inputs["latents"] = torch.zeros(1)
    elif reason == "handles":
        req.ref_handles = [{"handle": "opaque"}]
    elif reason == "opaque_extra":
        req.params.extra_params = {"opaque": object()}
    else:
        req.batch_info = BatchInfo((1, 2), (1, 1), (1, 2))
    assert request_batch_key(req) is None


@pytest.mark.parametrize("counts", [(1, 1), (2, 3), (4, 1)])
def test_rng_matches_original_request_shape_and_is_order_independent(counts):
    seeds = (42, 17)
    info = BatchInfo((1, 2), counts, seeds)
    shape = (sum(counts), 4, 8, 8)
    expected = [
        randn_tensor((count, *shape[1:]), generator=torch.Generator().manual_seed(seed))
        for count, seed in zip(counts, seeds)
    ]
    actual = randn_tensor_for_batch(
        shape,
        generator=torch.Generator().manual_seed(999),
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_info=info,
    )
    assert torch.equal(actual, torch.cat(expected))
    reverse = randn_tensor_for_batch(
        shape,
        generator=torch.Generator(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_info=BatchInfo((2, 1), counts[::-1], seeds[::-1]),
    )
    assert torch.equal(reverse, torch.cat(expected[::-1]))


def test_rng_rejects_inconsistent_sample_counts():
    with pytest.raises(ValueError, match="sample counts"):
        randn_tensor_for_batch(
            (3, 4),
            generator=torch.Generator(),
            device="cpu",
            dtype=torch.float32,
            batch_info=BatchInfo((1, 2), (1, 1), (42, 17)),
        )


def test_split_preserves_batch_axes_metadata_and_independent_storage():
    out = PipelineOutput(
        image=torch.arange(12).reshape(3, 2, 2, 1),
        video=torch.zeros(3, 2, 2, 2, 3),
        audio=torch.ones(3, 2, 8),
        action=torch.ones(3, 4, 6),
        frame_rate=16,
        denoise=0.25,
    )
    parts = split_batch_output(out, BatchInfo((1, 2), (1, 2), (42, 17)))
    assert parts[0].image.shape == (1, 2, 2, 1)
    assert torch.equal(parts[1].image, out.image[1:])
    assert parts[1].audio.shape == (2, 2, 8)
    assert parts[1].video.shape[0] == parts[1].action.shape[0] == 2
    assert parts[0].denoise == parts[1].denoise == 0.25
    assert parts[0].frame_rate == 16
    parts[0].image.zero_()
    assert out.image.sum() > 0
    assert parts[0].image.data_ptr() != out.image.data_ptr()


def test_split_rejects_bad_output_before_delivery():
    with pytest.raises(ValueError, match="image batch dimension"):
        split_batch_output(
            PipelineOutput(image=torch.zeros(1, 4, 4, 3)), BatchInfo((1, 2), (1, 1), (1, 2))
        )


@pytest.mark.parametrize("failure", [None, ValueError("bad request"), MemoryError("full"), "ipc"])
def test_executor_routes_each_success_or_failure_and_continues(failure):
    executor = DiffusionExecutor.__new__(DiffusionExecutor)
    executor.rank = 0
    executor.device_id = 0
    executor.in_client_process = True
    executor.response_queue = queue.Queue()
    executor.pipeline = SimpleNamespace(
        default_generation_params={},
        extra_param_specs={},
        _warmed_up_shapes=set(),
        prepare_request=Mock(),
        request_warmup_cache_key=Mock(return_value=(256, 256)),
        run_inference=Mock(
            return_value=PipelineOutput(image=torch.zeros(3, 4, 4, 3)),
            side_effect=None if failure == "ipc" else failure,
        ),
        classify_request_failure=lambda exc: (
            "capacity" if isinstance(exc, MemoryError) else "client"
        ),
    )
    req = _request(7)
    req.batch_info = BatchInfo((7, 9), (1, 2), (7, 9))
    with patch.object(
        PipelineOutput,
        "to_handle",
        side_effect=[None, RuntimeError("IPC failure")] if failure == "ipc" else None,
    ):
        executor.process_request(req)
    first, second = executor.response_queue.get_nowait(), executor.response_queue.get_nowait()
    assert (first.request_id, second.request_id) == (7, 9)
    assert executor.response_queue.empty()
    if failure == "ipc":
        assert first.error_msg is None
        assert second.error_msg == "IPC failure"
    elif failure is None:
        assert first.output.image.shape[0] == 1
        assert second.output.image.shape[0] == 2
        assert first.generation == second.generation
    else:
        assert first.error_msg == second.error_msg == str(failure)
        assert first.error_type == second.error_type
        executor.pipeline.run_inference.side_effect = None
        with patch.object(PipelineOutput, "to_handle"):
            executor.process_request(_request(10))
        assert executor.response_queue.get_nowait().error_msg is None


def test_receive_loop_broadcasts_a_single_merged_request():
    requests = _Queue([(0, _request(1)), (0, _request(2)), (0, None)])
    executor = DiffusionExecutor.__new__(DiffusionExecutor)
    executor.rank, executor.world_size, executor.device_id = 0, 2, 0
    executor.sender_thread = None
    executor.visual_gen_args = VisualGenArgs(batching_config={"max_batch_size": 2})
    executor.pipeline = SimpleNamespace(supports_dynamic_batching=True)
    executor.requests_ipc = SimpleNamespace(get=requests.receive, poll=requests.poll)
    executor._broadcast_request = Mock(side_effect=lambda req: req)
    executor.process_request = Mock()
    executor.serve_forever()
    assert executor.process_request.call_count == 1
    merged = executor.process_request.call_args.args[0]
    assert merged.batch_info.request_ids == (1, 2)
    assert executor._broadcast_request.call_count == 2
    assert executor._broadcast_request.call_args.args == (None,)


@pytest.mark.parametrize("supports_batching,cache", [(False, None), (True, object())])
def test_unsupported_pipeline_or_cache_uses_singletons(supports_batching, cache):
    requests = _Queue([(0, _request(1)), (0, _request(2)), (0, None)])
    executor = DiffusionExecutor.__new__(DiffusionExecutor)
    executor.rank, executor.world_size, executor.device_id = 0, 1, 0
    executor.sender_thread = None
    executor.visual_gen_args = SimpleNamespace(
        batching_config=BatchingConfig(max_batch_size=4, max_queue_delay_ms=100),
        cache_config=cache,
    )
    executor.pipeline = SimpleNamespace(supports_dynamic_batching=supports_batching)
    executor.requests_ipc = SimpleNamespace(get=requests.receive, poll=requests.poll)
    executor.process_request = Mock()
    executor.serve_forever()
    assert executor.process_request.call_count == 2
    assert requests.poll_timeouts == []


def test_client_sends_a_bounded_burst_and_retries_backpressure_in_order():
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client.pending_requests = queue.Queue()
    original = [_request(i) for i in range(3)]
    for req in original:
        client.pending_requests.put(req)
    client._request_to_send = None
    client.requests_ipc = SimpleNamespace(
        put_nowait=Mock(side_effect=[None, zmq.Again(), None, None])
    )
    client._iter_stats = Mock()
    client._check_worker_liveness = Mock(return_value=None)
    client._process_requests(max_requests=3)
    assert client._request_to_send is original[1]
    assert client.pending_requests.qsize() == 1
    client._process_requests(max_requests=1)
    assert client.pending_requests.qsize() == 1
    assert client._request_to_send is None
    client._process_requests(max_requests=3)
    assert client.pending_requests.empty()
    sent_ids = [call.args[0].request_id for call in client.requests_ipc.put_nowait.call_args_list]
    assert sent_ids == [0, 1, 1, 2]
    assert client._iter_stats.record_request_started.call_count == 3


def test_abandoning_one_request_keeps_its_batch_peers():
    async def check():
        client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
        client.lock = asyncio.Lock()
        client.response_event = asyncio.Event()
        client.completed_responses = {}
        client._abandoned_request_ids = set()
        client._iter_stats = Mock()
        client.pending_requests = queue.Queue()
        await client.abandon_request_id(1)
        await client._store_response(
            DiffusionResponse(1, output=PipelineOutput(image=torch.ones(1)))
        )
        peer = DiffusionResponse(2, output=PipelineOutput(image=torch.ones(1)))
        await client._store_response(peer)
        assert client.completed_responses == {2: peer}
        assert not client._abandoned_request_ids
        assert client._iter_stats.record_request_completed.call_count == 2

    asyncio.run(check())
