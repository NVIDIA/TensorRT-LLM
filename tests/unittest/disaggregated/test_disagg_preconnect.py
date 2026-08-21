# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from tensorrt_llm._torch.disaggregation.native.messenger import ZMQMessenger
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    MessageType,
    RankInfoServer,
    Sender,
    preconnect_instances,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.cluster_storage import WatchEventType
from tensorrt_llm.serve.disagg_auto_scaling import WorkerInfo
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.router import KvCacheAwareRouter, RoundRobinRouter


def _rank_info(
    instance_name: str,
    instance_rank: int,
    sender_endpoints: list[str] | None = None,
) -> RankInfo:
    return RankInfo(
        instance_name=instance_name,
        instance_rank=instance_rank,
        tp_size=2,
        tp_rank=instance_rank,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[1],
        sender_endpoints=sender_endpoints or [],
        server_endpoint="",
        self_endpoint="",
        transfer_engine_info=f"agent-{instance_rank}".encode("ascii"),
    )


class _RegistrationServer:
    def __init__(self, ack: list[bytes] | None = None) -> None:
        self.messenger = ZMQMessenger(mode="ROUTER")
        self.received_ranks: list[int] = []
        self._lock = threading.Lock()
        self._ack = ack or [
            MessageType.PRECONNECT_RANK_INFO_ACK,
            AgentResult.SUCCESS.value.encode("ascii"),
        ]

        def handle_message(messages: list[bytes]) -> bool:
            send_id = messages[0]
            assert messages[1] == MessageType.PRECONNECT_RANK_INFO
            rank_info = RankInfo.from_bytes(messages[2])
            with self._lock:
                self.received_ranks.append(rank_info.instance_rank)
            self.messenger.send([send_id, *self._ack])
            return True

        self.messenger.start_listener(handle_message)

    @property
    def endpoint(self) -> str:
        return self.messenger.endpoint

    def close(self) -> None:
        self.messenger.stop()


class _FakeRouter:
    def __init__(self, servers: list[str]) -> None:
        self.servers = list(servers)
        self._prepared = set(servers)

    async def prepare_server(self, server: str) -> bool:
        self._prepared.add(server)
        return True

    async def add_server(self, server: str) -> bool:
        if server not in self.servers:
            self.servers.append(server)
        return True

    async def remove_server(self, server: str) -> None:
        if server in self.servers:
            self.servers.remove(server)
        self._prepared.discard(server)

    async def discard_prepared_server(self, server: str) -> None:
        if server not in self.servers:
            self._prepared.discard(server)

    def get_server_info(self, _server: str) -> dict:
        return {}


def test_preconnect_instances_registers_full_rank_matrix() -> None:
    context_senders = [_RegistrationServer(), _RegistrationServer()]
    context_info = _rank_info(
        "context",
        0,
        sender_endpoints=[sender.endpoint for sender in context_senders],
    )
    generation_infos = [_rank_info("generation", rank) for rank in range(2)]
    context_info_server = RankInfoServer(context_info)
    generation_info_server = RankInfoServer(generation_infos[0])
    generation_info_server.set_instance_rank_infos(
        [rank_info.to_bytes() for rank_info in generation_infos]
    )

    try:
        preconnect_instances(context_info_server.endpoint, generation_info_server.endpoint)
        for sender in context_senders:
            assert sorted(sender.received_ranks) == [0, 1]
    finally:
        context_info_server.shutdown()
        generation_info_server.shutdown()
        for sender in context_senders:
            sender.close()


def test_preconnect_control_receive_has_a_finite_timeout() -> None:
    silent_server = ZMQMessenger(mode="ROUTER")
    try:
        with pytest.raises(TimeoutError, match="Preconnect control transaction"):
            preconnect_instances(
                silent_server.endpoint,
                silent_server.endpoint,
                control_timeout_ms=10,
            )
    finally:
        silent_server.stop()


@pytest.mark.parametrize(
    ("ack", "error_match"),
    [
        (
            [
                MessageType.PRECONNECT_RANK_INFO_ACK,
                AgentResult.FAILED.value.encode("ascii"),
                b"receiver rejected registration",
            ],
            "PRECONNECT_RANK_INFO failed.*receiver rejected registration",
        ),
        (
            [b"WRONG_ACK", AgentResult.SUCCESS.value.encode("ascii")],
            "Invalid PRECONNECT_RANK_INFO ACK",
        ),
    ],
)
def test_preconnect_instances_rejects_unsuccessful_ack(ack: list[bytes], error_match: str) -> None:
    context_sender = _RegistrationServer(ack=ack)
    context_info_server = RankInfoServer(
        _rank_info("context", 0, sender_endpoints=[context_sender.endpoint])
    )
    generation_info_server = RankInfoServer(_rank_info("generation", 0))

    try:
        with pytest.raises(RuntimeError, match=error_match):
            preconnect_instances(context_info_server.endpoint, generation_info_server.endpoint)
    finally:
        context_info_server.shutdown()
        generation_info_server.shutdown()
        context_sender.close()


def test_sender_registration_is_idempotent(monkeypatch) -> None:
    sender = Sender.__new__(Sender)
    sender._shutdown = False
    sender._device_id = 0
    sender._registrar = Mock()
    sender._agent = Mock()
    sender._loaded_remote_agents = set()
    sender._loaded_remote_agents_lock = threading.Lock()
    rank_info = _rank_info("generation", 1)

    monkeypatch.setattr("torch.cuda.set_device", Mock())
    monkeypatch.setattr("tensorrt_llm._torch.disaggregation.native.transfer.CUASSERT", Mock())
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.native.transfer.cudart.cudaSetDevice",
        Mock(),
    )

    message = [MessageType.REGISTER_RANK_INFO, rank_info.to_bytes()]
    sender._register_peer_rank(b"sender", message)
    sender._register_peer_rank(b"sender", message)

    assert sender._registrar.register.call_count == 2
    sender._agent.load_remote_agent.assert_called_once_with(
        "generation1", rank_info.transfer_engine_info
    )


def test_sender_preconnect_rejects_registration_during_shutdown() -> None:
    sender = Sender.__new__(Sender)
    sender._shutdown = True
    sender._messenger = Mock()
    sender._start_listener()
    handle_message = sender._messenger.start_listener.call_args.args[0]
    rank_info = _rank_info("generation", 1)

    handle_message([b"sender", MessageType.PRECONNECT_RANK_INFO, rank_info.to_bytes()])

    sender._messenger.send.assert_called_once_with(
        [
            b"sender",
            MessageType.PRECONNECT_RANK_INFO_ACK,
            AgentResult.FAILED.value.encode("ascii"),
            b"Sender is shutting down",
        ]
    )


def test_add_server_promotes_prepared_server_without_preparing_again() -> None:
    router = RoundRobinRouter(server_role=ServerRole.CONTEXT, servers=[])
    router._prepared_ready_servers.add("ctx1:8000")
    router.prepare_server = AsyncMock(return_value=True)

    async def exercise() -> None:
        try:
            assert await router.add_server("ctx1:8000") is True
        finally:
            await router.close()

    asyncio.run(exercise())

    router.prepare_server.assert_not_awaited()
    assert router.servers == ["ctx1:8000"]


def test_discovered_multi_instance_topology_stays_unready_until_preconnect() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = False
    coordinator._preconnected_pairs = set()
    coordinator._preconnect_lock = asyncio.Lock()
    coordinator._desired_workers = {}
    coordinator._ctx_router = Mock(
        num_prepared_servers=2,
        servers=["ctx1", "ctx2"],
    )
    coordinator._gen_router = Mock(
        num_prepared_servers=2,
        servers=["gen1", "gen2"],
    )
    coordinator._disagg_cluster_manager = Mock()
    coordinator._disagg_cluster_manager.is_ready_with_router = AsyncMock(return_value=True)
    coordinator._attempt_preconnect_pairs = AsyncMock(return_value=True)

    async def exercise() -> None:
        assert await coordinator.is_ready() is False
        await coordinator._maybe_complete_initial_preconnect()
        assert await coordinator.is_ready() is True

    asyncio.run(exercise())

    pairs, phase = coordinator._attempt_preconnect_pairs.await_args.args
    assert set(pairs) == {
        ("ctx1", "gen1"),
        ("ctx1", "gen2"),
        ("ctx2", "gen1"),
        ("ctx2", "gen2"),
    }
    assert phase == "initial discovered topology"


def test_initial_preconnect_retries_without_a_new_worker_event() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = False
    coordinator._preconnected_pairs = set()
    coordinator._preconnect_lock = asyncio.Lock()
    coordinator._health_check_interval_secs = 0
    coordinator._desired_workers = {}
    coordinator._ctx_router = Mock(num_prepared_servers=1, servers=["ctx1"])
    coordinator._gen_router = Mock(num_prepared_servers=1, servers=["gen1"])
    coordinator._disagg_cluster_manager = Mock()
    coordinator._disagg_cluster_manager.is_ready_with_router = AsyncMock(
        side_effect=[False, True, True]
    )
    coordinator._attempt_preconnect_pairs = AsyncMock(return_value=True)

    asyncio.run(coordinator._maybe_complete_initial_preconnect())

    assert coordinator._preconnect_completed is True
    assert coordinator._disagg_cluster_manager.is_ready_with_router.await_count == 3
    coordinator._attempt_preconnect_pairs.assert_awaited_once_with(
        [("ctx1", "gen1")], "initial discovered topology"
    )


def test_dynamic_worker_attempts_preconnect_before_routing_and_falls_back_on_failure() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = True
    coordinator._preconnected_pairs = {("ctx1", "gen1")}
    coordinator._preconnect_lock = asyncio.Lock()
    coordinator._desired_workers = {}
    coordinator._worker_onboarding_tasks = {}
    coordinator._initial_preconnect_task = None
    coordinator._ctx_router = Mock(servers=["ctx1", "ctx2"])
    coordinator._gen_router = Mock(servers=["gen1"])
    events = []
    coordinator._gen_router.prepare_server = AsyncMock(
        side_effect=lambda server: events.append(("prepare", server)) or True
    )
    coordinator._gen_router.add_server = AsyncMock(
        side_effect=lambda server: events.append(("add", server)) or True
    )
    coordinator._gen_router.discard_prepared_server = AsyncMock()
    coordinator._attempt_preconnect_pairs = AsyncMock(
        side_effect=lambda pairs, phase: events.append(("preconnect", set(pairs))) or False
    )
    coordinator._disagg_cluster_manager = Mock()

    worker = WorkerInfo(
        worker_id="gen2-worker",
        host="gen2",
        port=8000,
        role=ServerRole.GENERATION,
    )

    async def exercise() -> None:
        await coordinator._on_worker_event(worker, WatchEventType.SET)
        await asyncio.gather(*coordinator._worker_onboarding_tasks.values())

    asyncio.run(exercise())

    coordinator._gen_router.prepare_server.assert_awaited_once_with("gen2:8000")
    pairs, phase = coordinator._attempt_preconnect_pairs.await_args.args
    assert set(pairs) == {("ctx1", "gen2:8000"), ("ctx2", "gen2:8000")}
    assert phase == "onboarding worker gen2:8000"
    # Even a failed optimization attempt preserves the original lazy-registration path.
    coordinator._gen_router.add_server.assert_awaited_once_with("gen2:8000")
    assert events == [
        ("prepare", "gen2:8000"),
        ("preconnect", {("ctx1", "gen2:8000"), ("ctx2", "gen2:8000")}),
        ("add", "gen2:8000"),
    ]


def test_attempt_preconnect_failure_returns_false_without_raising() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_pairs = AsyncMock(side_effect=RuntimeError("unsupported backend"))

    pairs = [("ctx1", "gen1")]
    succeeded = asyncio.run(coordinator._attempt_preconnect_pairs(pairs, "test phase"))

    assert succeeded is False
    coordinator._preconnect_pairs.assert_awaited_once_with(pairs)


def test_preconnect_pairs_settles_all_attempts_and_retains_success_before_raising() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnected_pairs = set()
    coordinator._desired_workers = {}
    coordinator._ctx_router = Mock(servers=["ctx1"])
    coordinator._gen_router = Mock(servers=["gen1", "gen2"])
    completed_pairs: list[tuple[str, str]] = []

    async def preconnect_pair(context_server: str, generation_server: str) -> bool:
        if generation_server == "gen1":
            raise RuntimeError("gen1 failed")
        await asyncio.sleep(0.01)
        completed_pairs.append((context_server, generation_server))
        return True

    coordinator._preconnect_pair = AsyncMock(side_effect=preconnect_pair)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="1 of 2 instance pairs failed"):
            await coordinator._preconnect_pairs([("ctx1", "gen1"), ("ctx1", "gen2")])

    asyncio.run(exercise())

    assert completed_pairs == [("ctx1", "gen2")]
    assert coordinator._preconnected_pairs == {("ctx1", "gen2")}


def test_deleted_worker_is_removed_from_preconnected_pair_bookkeeping() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = True
    coordinator._preconnected_pairs = {
        ("ctx1:8000", "gen1"),
        ("ctx1:8000", "gen2"),
        ("ctx2", "gen1"),
    }
    coordinator._desired_workers = {(ServerRole.CONTEXT, "ctx1:8000"): "ctx1-worker"}
    coordinator._worker_onboarding_tasks = {}
    coordinator._ctx_router = Mock(servers=["ctx1:8000", "ctx2"])
    coordinator._ctx_router.remove_server = AsyncMock()
    coordinator._ctx_router.discard_prepared_server = AsyncMock()
    coordinator._gen_router = Mock(servers=["gen1", "gen2"])

    worker = WorkerInfo(
        worker_id="ctx1-worker",
        host="ctx1",
        port=8000,
        role=ServerRole.CONTEXT,
    )
    asyncio.run(coordinator._on_worker_event(worker, WatchEventType.DELETE))

    coordinator._ctx_router.remove_server.assert_awaited_once_with("ctx1:8000")
    coordinator._ctx_router.discard_prepared_server.assert_awaited_once_with("ctx1:8000")
    assert coordinator._preconnected_pairs == {("ctx2", "gen1")}


def test_outdated_delete_does_not_override_newer_desired_worker_state() -> None:
    # This verifies worker-event ordering only: an event for an older worker ID
    # must not tear down state tracked for the newer desired worker.
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = True
    coordinator._preconnected_pairs = {("ctx1", "gen1:8000")}
    coordinator._desired_workers = {(ServerRole.GENERATION, "gen1:8000"): "current-worker"}
    coordinator._worker_onboarding_tasks = {}
    coordinator._ctx_router = Mock(servers=["ctx1"])
    coordinator._gen_router = Mock(servers=["gen1:8000"])
    coordinator._gen_router.remove_server = AsyncMock()
    coordinator._gen_router.discard_prepared_server = AsyncMock()
    stale_worker = WorkerInfo(
        worker_id="old-worker",
        host="gen1",
        port=8000,
        role=ServerRole.GENERATION,
    )

    asyncio.run(coordinator._on_worker_event(stale_worker, WatchEventType.DELETE))

    coordinator._gen_router.remove_server.assert_not_awaited()
    assert coordinator._desired_workers[(ServerRole.GENERATION, "gen1:8000")] == "current-worker"
    assert coordinator._preconnected_pairs == {("ctx1", "gen1:8000")}


def test_kv_aware_dynamic_worker_is_staged_before_hash_handshake() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = True
    coordinator._preconnected_pairs = set()
    coordinator._preconnect_lock = asyncio.Lock()
    coordinator._desired_workers = {}
    coordinator._worker_onboarding_tasks = {}
    coordinator._initial_preconnect_task = None
    coordinator._disagg_cluster_manager = Mock()
    coordinator._ctx_router = Mock(servers=["ctx1"])
    coordinator._gen_router = KvCacheAwareRouter(
        server_role=ServerRole.GENERATION,
        servers=[],
        tokens_per_block=32,
    )
    coordinator._gen_router._fetch_server_info = AsyncMock(
        return_value={
            "tokens_per_block": 32,
            "kv_cache_hash_algo": "v1_block_key",
        }
    )

    async def assert_staged_before_preconnect(_pairs, _phase):
        assert coordinator._gen_router.servers == []
        assert coordinator._gen_router.num_prepared_servers == 0
        assert "gen1:8000" in coordinator._gen_router._server_state
        return True

    coordinator._attempt_preconnect_pairs = AsyncMock(side_effect=assert_staged_before_preconnect)

    worker = WorkerInfo(
        worker_id="gen1-worker",
        host="gen1",
        port=8000,
        role=ServerRole.GENERATION,
    )

    async def exercise() -> None:
        try:
            await coordinator._on_worker_event(worker, WatchEventType.SET)
            await asyncio.gather(*coordinator._worker_onboarding_tasks.values())
            assert coordinator._gen_router.servers == ["gen1:8000"]
            assert coordinator._gen_router.num_prepared_servers == 1
            assert coordinator._gen_router._server_state["gen1:8000"].hash_algo == ("v1_block_key")
        finally:
            await coordinator._gen_router.close()

    asyncio.run(exercise())


def test_slow_onboarding_does_not_block_delete() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = True
    coordinator._preconnected_pairs = {("ctx1", "gen1:8000")}
    coordinator._preconnect_lock = asyncio.Lock()
    coordinator._desired_workers = {}
    coordinator._worker_onboarding_tasks = {}
    coordinator._initial_preconnect_task = None
    coordinator._disagg_cluster_manager = Mock()
    coordinator._ctx_router = Mock(servers=["ctx1"])
    coordinator._gen_router = Mock(servers=[])
    coordinator._gen_router.prepare_server = AsyncMock(return_value=True)
    coordinator._gen_router.add_server = AsyncMock(return_value=True)
    coordinator._gen_router.remove_server = AsyncMock()
    coordinator._gen_router.discard_prepared_server = AsyncMock()
    preconnect_started = asyncio.Event()
    release_preconnect = asyncio.Event()

    async def slow_preconnect(_pairs, _phase):
        preconnect_started.set()
        await release_preconnect.wait()
        return True

    coordinator._attempt_preconnect_pairs = AsyncMock(side_effect=slow_preconnect)
    worker = WorkerInfo(
        worker_id="gen1-worker",
        host="gen1",
        port=8000,
        role=ServerRole.GENERATION,
    )

    async def exercise() -> None:
        await coordinator._on_worker_event(worker, WatchEventType.SET)
        await asyncio.wait_for(preconnect_started.wait(), timeout=1)
        await asyncio.wait_for(
            coordinator._on_worker_event(worker, WatchEventType.DELETE), timeout=1
        )
        release_preconnect.set()
        await asyncio.sleep(0)

    asyncio.run(exercise())

    coordinator._gen_router.add_server.assert_not_awaited()
    coordinator._gen_router.remove_server.assert_awaited_once_with("gen1:8000")
    assert (ServerRole.GENERATION, "gen1:8000") not in coordinator._desired_workers
    assert coordinator._preconnected_pairs == set()


def test_concurrent_context_and_generation_onboarding_covers_cross_pair() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = True
    coordinator._preconnected_pairs = {("ctx1:8000", "gen1:8000")}
    coordinator._preconnect_lock = asyncio.Lock()
    coordinator._desired_workers = {}
    coordinator._worker_onboarding_tasks = {}
    coordinator._initial_preconnect_task = None
    coordinator._disagg_cluster_manager = Mock()
    coordinator._ctx_router = _FakeRouter(["ctx1:8000"])
    coordinator._gen_router = _FakeRouter(["gen1:8000"])
    attempted_pairs: list[tuple[str, str]] = []

    async def record_preconnect(pairs, _phase):
        attempted_pairs.extend(pairs)
        return True

    coordinator._attempt_preconnect_pairs = AsyncMock(side_effect=record_preconnect)
    context_worker = WorkerInfo(
        worker_id="ctx2-worker",
        host="ctx2",
        port=8000,
        role=ServerRole.CONTEXT,
    )
    generation_worker = WorkerInfo(
        worker_id="gen2-worker",
        host="gen2",
        port=8000,
        role=ServerRole.GENERATION,
    )

    async def exercise() -> None:
        await coordinator._on_worker_event(context_worker, WatchEventType.SET)
        await coordinator._on_worker_event(generation_worker, WatchEventType.SET)
        await asyncio.gather(*coordinator._worker_onboarding_tasks.values())

    asyncio.run(exercise())

    assert set(attempted_pairs) == {
        ("ctx2:8000", "gen1:8000"),
        ("ctx1:8000", "gen2:8000"),
        ("ctx2:8000", "gen2:8000"),
    }
    assert coordinator._ctx_router.servers == ["ctx1:8000", "ctx2:8000"]
    assert coordinator._gen_router.servers == ["gen1:8000", "gen2:8000"]


def test_generation_only_topology_completes_with_no_pairs() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = False
    coordinator._preconnected_pairs = set()
    coordinator._preconnect_lock = asyncio.Lock()
    coordinator._desired_workers = {}
    coordinator._ctx_router = Mock(num_prepared_servers=0, servers=[])
    coordinator._gen_router = Mock(num_prepared_servers=2, servers=["gen1", "gen2"])
    coordinator._disagg_cluster_manager = Mock()
    coordinator._disagg_cluster_manager.is_ready_with_router = AsyncMock(return_value=True)
    coordinator._attempt_preconnect_pairs = AsyncMock(return_value=True)

    asyncio.run(coordinator._maybe_complete_initial_preconnect())

    coordinator._attempt_preconnect_pairs.assert_awaited_once_with(
        [], "initial discovered topology"
    )
    assert coordinator._preconnect_completed is True


def test_initial_preconnect_retries_when_readiness_is_lost_after_attempt() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnect_completed = False
    coordinator._preconnected_pairs = set()
    coordinator._preconnect_lock = asyncio.Lock()
    coordinator._health_check_interval_secs = 0
    coordinator._desired_workers = {}
    coordinator._ctx_router = Mock(num_prepared_servers=1, servers=["ctx1"])
    coordinator._gen_router = Mock(num_prepared_servers=1, servers=["gen1"])
    coordinator._disagg_cluster_manager = Mock()
    coordinator._disagg_cluster_manager.is_ready_with_router = AsyncMock(
        side_effect=[True, False, True, True]
    )
    coordinator._attempt_preconnect_pairs = AsyncMock(return_value=True)

    asyncio.run(coordinator._maybe_complete_initial_preconnect())

    assert coordinator._preconnect_completed is True
    assert coordinator._disagg_cluster_manager.is_ready_with_router.await_count == 4
    assert coordinator._attempt_preconnect_pairs.await_count == 2


def test_pair_removed_during_preconnect_is_not_recorded() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnected_pairs = set()
    coordinator._desired_workers = {}
    coordinator._ctx_router = Mock(servers=["ctx1"])
    coordinator._gen_router = Mock(servers=["gen1"])

    async def preconnect_then_remove_context(_context: str, _generation: str) -> bool:
        coordinator._ctx_router.servers = []
        return True

    coordinator._preconnect_pair = AsyncMock(side_effect=preconnect_then_remove_context)

    asyncio.run(coordinator._preconnect_pairs([("ctx1", "gen1")]))

    assert coordinator._preconnected_pairs == set()


def test_pair_without_python_nixl_endpoints_is_skipped_cleanly() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._ctx_router = Mock()
    coordinator._gen_router = Mock()
    coordinator._ctx_router.get_server_info.return_value = {}
    coordinator._gen_router.get_server_info.return_value = {
        "disaggregated_params": {"ctx_info_endpoint": "tcp://gen:1234"}
    }

    performed = asyncio.run(coordinator._preconnect_pair("ctx1", "gen1"))

    assert performed is False


def test_skipped_pair_is_not_marked_preconnected_and_can_retry() -> None:
    coordinator = DisaggCoordinatorService.__new__(DisaggCoordinatorService)
    coordinator._preconnected_pairs = set()
    coordinator._desired_workers = {}
    coordinator._ctx_router = Mock(servers=["ctx1"])
    coordinator._gen_router = Mock(servers=["gen1"])
    coordinator._preconnect_pair = AsyncMock(side_effect=[False, True])

    async def exercise() -> None:
        await coordinator._preconnect_pairs([("ctx1", "gen1")])
        assert coordinator._preconnected_pairs == set()
        await coordinator._preconnect_pairs([("ctx1", "gen1")])

    asyncio.run(exercise())

    assert coordinator._preconnect_pair.await_count == 2
    assert coordinator._preconnected_pairs == {("ctx1", "gen1")}


def test_rank_info_allgather_is_always_published() -> None:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._kv_cache_manager = Mock(pp_layers=[0])
    transceiver._mapping = Mock()
    transceiver._context_info_endpoint = "tcp://context:1234"
    transceiver._transfer_worker = Mock(sender_endpoint="tcp://sender:1234")
    transceiver._transfer_worker.rank_info.to_bytes.return_value = b"rank-info"
    transceiver._dist = Mock()
    transceiver._dist.pp_allgather.return_value = [1]
    transceiver._dist.allgather.side_effect = [
        ["tcp://sender:1234"],
        [b"rank-info"],
    ]

    transceiver._exchange_rank_info()

    assert transceiver._dist.allgather.call_count == 2
    transceiver._transfer_worker.populate_instance_and_rank_info.assert_called_once_with(
        endpoints=["tcp://sender:1234"], layer_num_per_pp=[1]
    )
    transceiver._transfer_worker.publish_instance_rank_infos.assert_called_once_with([b"rank-info"])
