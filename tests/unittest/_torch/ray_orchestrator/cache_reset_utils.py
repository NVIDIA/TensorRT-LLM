# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from contextlib import ExitStack

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
    def reset_for_subscription_probe(self):
        publisher = self.engine.kv_cache_manager.event_manager._publisher
        previous = publisher.enqueued_batches
        WorkerExtension.reset_prefix_cache.__wrapped__(self)
        assert publisher.enqueued_batches == previous + 1
        assert publisher.dropped_batches == 0
        return self.engine.dist.rank, publisher.enqueued_batches - 1

    @control_action_decorator
    def reset_and_wait_for_observer(self, endpoint):
        """Reset reuse state and hold every executor until its events are observed."""
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
    """Read retained batches until the requested event appears, without resuming inference."""
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


def check_cache_reset_events_through_control_rpc(tmp_path, ranks=1):
    """Observe reset events through public transports while the control RPC is parked."""
    endpoint = f"ipc://{tmp_path}/events"
    replay_endpoint = f"ipc://{tmp_path}/replay"
    observer_endpoint = f"ipc://{tmp_path}/observer"
    config = KvCacheConfig(
        enable_block_reuse=True,
        use_kv_cache_manager_v2=True,
        max_tokens=1024,
        tokens_per_block=32,
        kv_events_config=KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=endpoint,
            replay_endpoint=replay_endpoint,
        ),
    )
    with ExitStack() as stack:
        observer = stack.enter_context(zmq.Context.instance().socket(zmq.ROUTER))
        replays = [
            stack.enter_context(zmq.Context.instance().socket(zmq.DEALER)) for _ in range(ranks)
        ]
        subscribers = [
            stack.enter_context(zmq.Context.instance().socket(zmq.SUB)) for _ in range(ranks)
        ]
        for socket in (observer, *replays, *subscribers):
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 30_000)
            socket.setsockopt(zmq.SNDTIMEO, 30_000)
        observer.bind(observer_endpoint)
        for rank, replay in enumerate(replays):
            replay.connect(replay_endpoint if rank == 0 else f"{replay_endpoint}_dp{rank}")
        for rank, subscriber in enumerate(subscribers):
            subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            subscriber.connect(endpoint if rank == 0 else f"{endpoint}_dp{rank}")
        with LLM(
            model=llm_models_root() / "Qwen3/Qwen3-0.6B",
            orchestrator_type="ray",
            ray_worker_extension_cls=(
                "_torch.ray_orchestrator.cache_reset_utils.ResetEventsWorkerExtension"
            ),
            tensor_parallel_size=ranks,
            enable_attention_dp=ranks > 1,
            kv_cache_config=config,
            load_format="dummy",
            model_kwargs={"num_hidden_layers": 1},
            max_seq_len=256,
            max_batch_size=2,
            max_num_tokens=256,
            enable_autotuner=False,
        ) as llm:
            # A PUB socket does not acknowledge subscriptions. Use idle resets
            # to establish that every rank's live subscriber is receiving.
            ready = set()
            last_seen = [-1] * ranks
            deadline = time.monotonic() + 30
            while len(ready) < ranks and time.monotonic() < deadline:
                refs = llm._collective_rpc("reset_for_subscription_probe", non_block=True)
                cursors = dict(ray.get(refs, timeout=40))
                assert set(cursors) == set(range(ranks))
                for rank, subscriber in enumerate(subscribers):
                    if subscriber.poll(250):
                        _, frame, payload = subscriber.recv_multipart()
                        last_seen[rank] = int.from_bytes(frame, "big")
                        batch = msgspec.msgpack.decode(payload)
                        if batch[2] == rank and batch[1][-1] == {"type": "AllBlocksCleared"}:
                            ready.add(rank)
            assert len(ready) == ranks, "Not all live event subscriptions became ready"

            output = llm.generate(
                [1] * 65,
                SamplingParams(max_tokens=1, ignore_eos=True),
                scheduling_params=(
                    SchedulingParams(attention_dp_rank=0, attention_dp_relax=False)
                    if ranks > 1
                    else None
                ),
            )
            assert output.outputs[0].token_ids
            before = _replay_until(replays[0], 0, "BlockStored")
            start_seqs = [before[-1][0] + 1]
            start_seqs.extend(cursors[rank] + 1 for rank in range(1, ranks))

            # Consume all earlier batches so the next observed marker must be
            # the reset performed while the control request is parked.
            for subscriber, start_seq, sequence in zip(subscribers, start_seqs, last_seen):
                while sequence < start_seq - 1:
                    _, frame, _ = subscriber.recv_multipart()
                    sequence = int.from_bytes(frame, "big")

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
                for rank, (replay, subscriber, start_seq) in enumerate(
                    zip(replays, subscribers, start_seqs)
                ):
                    sequence = start_seq
                    while True:
                        _, frame, payload = subscriber.recv_multipart()
                        assert int.from_bytes(frame, "big") == sequence
                        sequence += 1
                        live = msgspec.msgpack.decode(payload)
                        assert live[2] == rank
                        if live[1][-1] == {"type": "AllBlocksCleared"}:
                            break
                    after = _replay_until(replay, start_seq, "AllBlocksCleared")
                    assert after[0][0] == start_seq
                    assert all(batch[2] == rank for _, batch in after)
                    assert after[-1][1][1][-1] == {"type": "AllBlocksCleared"}
            finally:
                for identity in parked:
                    observer.send_multipart([identity, b"observed"])
                assert sorted(ray.get(refs, timeout=40)) == list(range(ranks))
