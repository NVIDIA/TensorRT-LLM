# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time

import msgspec
import pytest
import ray
import zmq
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.executor.ray.utils import control_action_decorator
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import KVEventsConfig
from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension
from tensorrt_llm.scheduling_params import SchedulingParams


class ResetEventsWorkerExtension(WorkerExtension):
    @control_action_decorator
    def reset_and_wait_for_observer(self, endpoint):
        # Keep the real executor parked until the driver observes the events;
        # resuming it early could hide a missing reset-time flush.
        assert self.engine.control_request_barrier.is_set()
        assert not self.engine.control_action_done.is_set()
        WorkerExtension.reset_prefix_cache.__wrapped__(self)
        rank = self.engine.dist.rank
        with zmq.Context.instance().socket(zmq.DEALER) as observer:
            observer.setsockopt(zmq.LINGER, 0)
            observer.setsockopt(zmq.IDENTITY, str(rank).encode())
            observer.setsockopt(zmq.RCVTIMEO, 30_000)
            observer.setsockopt(zmq.SNDTIMEO, 30_000)
            observer.connect(endpoint)
            observer.send(b"reset")
            assert observer.recv() == b"observed"
        return rank


def _replay_until(replay, start_seq, event_type):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        replay.send_multipart([b"", start_seq.to_bytes(8, "big")])
        batches = []
        while True:
            _, _, sequence, payload = replay.recv_multipart()
            if not payload:
                break
            batches.append((int.from_bytes(sequence, "big"), msgspec.msgpack.decode(payload)))
        if any(event["type"] == event_type for _, batch in batches for event in batch[1]):
            return batches
    pytest.fail(f"No {event_type} event published while the executor was parked")


def _buffered_events(llm):
    events = llm._executor.rpc_client.fetch_kv_cache_events_wait_async(timeout=1).remote()
    return [json.loads(event) if isinstance(event, str) else event for event in events]


def _check_cache_reset_events_through_control_rpc(streaming, tmp_path):
    ranks = 1 if streaming else 2
    endpoint = f"ipc://{tmp_path}/events"
    replay_endpoint = f"ipc://{tmp_path}/replay"
    observer_endpoint = f"ipc://{tmp_path}/observer"
    config = KvCacheConfig(
        enable_block_reuse=True,
        use_kv_cache_manager_v2=True,
        max_tokens=1024,
        tokens_per_block=32,
        event_buffer_max_size=0 if streaming else 1024,
        kv_events_config=(
            KVEventsConfig(
                enable_kv_cache_events=True,
                publisher="zmq",
                endpoint=endpoint,
                replay_endpoint=replay_endpoint,
            )
            if streaming
            else None
        ),
    )
    with (
        zmq.Context.instance().socket(zmq.ROUTER) as observer,
        zmq.Context.instance().socket(zmq.DEALER) as replay,
    ):
        for socket in (observer, replay):
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 30_000)
            socket.setsockopt(zmq.SNDTIMEO, 30_000)
        observer.bind(observer_endpoint)
        if streaming:
            replay.connect(replay_endpoint)
        with LLM(
            model=llm_models_root() / "Qwen3/Qwen3-0.6B",
            orchestrator_type="ray",
            ray_worker_extension_cls=(
                "_torch.ray_orchestrator.test_kv_cache_reset_events.ResetEventsWorkerExtension"
            ),
            tensor_parallel_size=ranks,
            enable_attention_dp=not streaming,
            enable_lm_head_tp_in_adp=not streaming,
            kv_cache_config=config,
            load_format="dummy",
            model_kwargs={"num_hidden_layers": 1},
            max_seq_len=256,
            max_batch_size=2,
            max_num_tokens=256,
            enable_autotuner=False,
        ) as llm:
            # Leave rank 1 empty: it must still participate in the buffered
            # reset gather while rank 0 has removal events to publish.
            output = llm.generate(
                [1] * 65,
                SamplingParams(max_tokens=1, ignore_eos=True),
                scheduling_params=(
                    SchedulingParams(attention_dp_rank=0, attention_dp_relax=False)
                    if not streaming
                    else None
                ),
            )
            assert output.outputs[0].token_ids

            stored = {rank: set() for rank in range(ranks)}
            if streaming:
                before = _replay_until(replay, 0, "BlockStored")
                start_seq = before[-1][0] + 1
            else:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not stored[0]:
                    for event in _buffered_events(llm):
                        if event and event["data"]["type"] == "stored":
                            stored[event["attention_dp_rank"]].update(
                                block["block_hash"] for block in event["data"]["blocks"]
                            )
                assert stored[0], "Rank 0 must have cached blocks"
                assert not stored[1], "Rank 1 must exercise the empty-cache gather path"

            refs = llm._collective_rpc(
                "reset_and_wait_for_observer", args=(observer_endpoint,), non_block=True
            )
            parked = []
            try:
                for _ in range(ranks):
                    identity, message = observer.recv_multipart()
                    parked.append(identity)
                    assert message == b"reset"
                assert set(parked) == {str(rank).encode() for rank in range(ranks)}
                if streaming:
                    after = _replay_until(replay, start_seq, "AllBlocksCleared")
                    assert after[0][0] == start_seq
                    assert after[-1][1][1][-1] == {"type": "AllBlocksCleared"}
                else:
                    removed = {rank: set() for rank in range(ranks)}
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline and removed != stored:
                        for event in _buffered_events(llm):
                            if event and event["data"]["type"] == "removed":
                                removed[event["attention_dp_rank"]].update(
                                    event["data"]["block_hashes"]
                                )
                    assert removed == stored
            finally:
                for identity in parked:
                    observer.send_multipart([identity, b"observed"])
                assert sorted(ray.get(refs, timeout=40)) == list(range(ranks))


@pytest.mark.ray
@pytest.mark.timeout(240)
def test_streaming_reset_is_published_before_control_rpc_resumes(tmp_path):
    _check_cache_reset_events_through_control_rpc(True, tmp_path)


@pytest.mark.ray
@pytest.mark.gpu2
@pytest.mark.timeout(240)
def test_buffered_attention_dp_reset_with_an_empty_rank(tmp_path):
    _check_cache_reset_events_through_control_rpc(False, tmp_path)
