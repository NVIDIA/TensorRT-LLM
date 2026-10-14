# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node-local discovery must not become a second inference endpoint."""

from types import SimpleNamespace

import grpc
import pytest

from tensorrt_llm.grpc.openengine import node  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import (  # noqa: E402
    generation_pb2,
    kv_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

pytestmark = pytest.mark.cpu_only


def _source(rank, host):
    return {
        "transport": "zmq",
        "encoding": "msgpack",
        "schema_version": 1,
        "data_parallel_rank": rank,
        "endpoint_addr": {"host": host, "port": 5557 + rank, "protocol": "tcp"},
    }


def test_publisher_descriptor_advertises_connectable_endpoints():
    config = SimpleNamespace(topic="kv", buffer_steps=7, hwm=9, max_queue_size=11)
    descriptor = {
        "rank": 1,
        "endpoint": "tcp://*:5558",
        "replay_endpoint": "tcp://*:5658",
    }
    source = node._source(descriptor, config, "node-b")
    assert source["endpoint_addr"] == {"host": "node-b", "port": 5558, "protocol": "tcp"}
    assert source["replay_endpoint"] == "tcp://node-b:5658"


def test_follower_real_rpc_is_local_and_metadata_only():
    info = server_pb2.ServerInfo(instance_id="engine-incarnation")
    info.extra.update({"trtllm_node": {"leader": False, "local_dp_ranks": [1]}})
    server = node.NodeMetadataServer("127.0.0.1", 0, info, [_source(1, "follower")])
    try:
        with grpc.insecure_channel(f"127.0.0.1:{server.port}") as channel:
            control = openengine_pb2_grpc.ControlStub(channel)
            discovered = control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=3)
            assert discovered == info
            sources = control.GetKvEventSources(kv_pb2.GetKvEventSourcesRequest(), timeout=3)
            assert [source.data_parallel_rank for source in sources.sources] == [1]
            with pytest.raises(grpc.RpcError) as invalid:
                control.GetKvEventSources(
                    kv_pb2.GetKvEventSourcesRequest(data_parallel_ranks=[0]), timeout=3
                )
            assert invalid.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            with pytest.raises(grpc.RpcError) as inference:
                next(
                    openengine_pb2_grpc.InferenceStub(channel).Generate(
                        generation_pb2.GenerateRequest(), timeout=3
                    )
                )
            assert inference.value.code() == grpc.StatusCode.UNIMPLEMENTED
    finally:
        server.close()
    with grpc.insecure_channel(f"127.0.0.1:{server.port}") as channel:
        with pytest.raises(grpc.RpcError) as stopped:
            openengine_pb2_grpc.ControlStub(channel).GetServerInfo(
                server_pb2.GetServerInfoRequest(), timeout=1
            )
        assert stopped.value.code() == grpc.StatusCode.UNAVAILABLE


def test_startup_owns_actual_local_sources_once(monkeypatch):
    sources = [_source(0, "leader"), _source(1, "follower"), _source(2, "follower")]
    calls = []

    class Server:
        def __init__(self, host, port, info, local):
            calls.append((info, local))

        def close(self):
            pass

    monkeypatch.setattr(node, "NodeMetadataServer", Server)
    monkeypatch.setattr(node.socket, "getfqdn", lambda: "follower")
    for rank in [1, 2]:
        rows = iter(
            [
                [
                    (0, "leader", sources[0], None),
                    (1, "follower", sources[1], None),
                    (2, "follower", sources[2], None),
                ],
                [None, None, None],
            ]
        )
        executor = SimpleNamespace(
            llm_args=SimpleNamespace(
                _openengine_discovery={
                    "engine_id": "incarnation",
                    "host": "127.0.0.1",
                    "port": 15051,
                },
                kv_cache_config=SimpleNamespace(kv_events_config=None),
            ),
            dist=SimpleNamespace(rank=rank, tp_size=3, allgather=lambda _: next(rows)),
            kv_cache_manager=SimpleNamespace(tokens_per_block=32),
            _kv_cache_capacity={},
            enable_attention_dp=True,
        )
        node.initialize_node_discovery(executor)
        metadata = executor._openengine_discovery["node"]
        assert metadata["engine_id"] == "incarnation"
        assert metadata["local_dp_ranks"] == [1, 2]
        assert metadata["leader"] is False
        assert (executor._openengine_node_server is not None) == (rank == 1)
    assert len(calls) == 1
    assert calls[0][1] == sources[1:]


def test_startup_rejects_missing_attention_dp_publisher(monkeypatch):
    monkeypatch.setattr(node.socket, "getfqdn", lambda: "follower")
    rows = [(0, "leader", _source(0, "leader"), None), (1, "follower", None, None)]
    executor = SimpleNamespace(
        llm_args=SimpleNamespace(
            _openengine_discovery={"engine_id": "incarnation", "host": "127.0.0.1", "port": 15051},
            kv_cache_config=SimpleNamespace(kv_events_config=None),
        ),
        dist=SimpleNamespace(rank=1, tp_size=2, allgather=lambda _: rows),
        kv_cache_manager=SimpleNamespace(tokens_per_block=32),
        _kv_cache_capacity={},
        enable_attention_dp=True,
    )
    with pytest.raises(RuntimeError, match="publishers are incomplete"):
        node.initialize_node_discovery(executor)


def test_bind_failure_rolls_back_successful_node(monkeypatch):
    closed = []

    class Server:
        def __init__(self, *args):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(node, "NodeMetadataServer", Server)
    monkeypatch.setattr(node.socket, "getfqdn", lambda: "follower")
    rows = iter(
        [
            [
                (0, "leader", _source(0, "leader"), None),
                (1, "follower", _source(1, "follower"), None),
            ],
            [None, "another node could not bind"],
        ]
    )
    executor = SimpleNamespace(
        llm_args=SimpleNamespace(
            _openengine_discovery={"engine_id": "incarnation", "host": "127.0.0.1", "port": 15051},
            kv_cache_config=SimpleNamespace(kv_events_config=None),
        ),
        dist=SimpleNamespace(rank=1, tp_size=2, allgather=lambda _: next(rows)),
        kv_cache_manager=SimpleNamespace(tokens_per_block=32),
        _kv_cache_capacity={},
        enable_attention_dp=True,
    )
    with pytest.raises(RuntimeError, match="another node could not bind"):
        node.initialize_node_discovery(executor)
    assert closed == [True]


def test_unexpected_listener_error_reaches_the_error_collective(monkeypatch):
    """A local startup KeyError must not strand peer ranks at the next collective."""
    monkeypatch.setattr(node.socket, "getfqdn", lambda: "follower")
    gathered = []
    placements = [
        (0, "leader", _source(0, "leader"), None),
        (1, "follower", _source(1, "follower"), None),
    ]

    def allgather(value):
        gathered.append(value)
        return placements if len(gathered) == 1 else [None, value]

    executor = SimpleNamespace(
        llm_args=SimpleNamespace(
            _openengine_discovery={"engine_id": "incarnation", "port": 15051},
            kv_cache_config=SimpleNamespace(kv_events_config=None),
        ),
        dist=SimpleNamespace(rank=1, tp_size=2, allgather=allgather),
        kv_cache_manager=SimpleNamespace(tokens_per_block=32),
        enable_attention_dp=True,
    )

    with pytest.raises(RuntimeError, match="OpenEngine node discovery startup failed"):
        node.initialize_node_discovery(executor)
    assert len(gathered) == 2
    assert "host" in gathered[1]
