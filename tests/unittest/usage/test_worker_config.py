# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker observations use real processes but never contact a telemetry service."""

import hmac
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field, PrivateAttr

from tensorrt_llm.usage import llmapi_config
from tensorrt_llm.usage import worker_config as wc
from tensorrt_llm.usage.llmapi_config import collect_llm_api_config_payloads

pytestmark = pytest.mark.cpu_only
HOST_CACHE_FIELD = "kv_cache_config.host_cache_size"
DISK_CACHE_FIELD = "kv_cache_config.disk_cache_size"


class CacheConfig(BaseModel):
    host_cache_size: int | None = 8192
    disk_cache_size: int | None = 16384
    enable_partial_reuse: bool = True
    disk_cache_path: str = "/private/customer/cache"


class Args(BaseModel):
    kv_cache_config: CacheConfig = Field(default_factory=CacheConfig)
    unrelated: bool = True
    _worker_config_endpoint: object = PrivateAttr(None)
    _worker_config_observation: object = PrivateAttr(None)


def snapshot(args=None):
    config, meta = map(json.loads, collect_llm_api_config_payloads(args or Args()))
    return {
        "manifest": meta["capture_manifest_digest"],
        "policy": meta["field_policy_version"],
        "values": {key: config[key] for key in wc.FIELDS if key in config},
    }


def captured(observations, expected=2):
    args = Args()
    args._worker_config_observation = {"expected": expected, "snapshots": observations}
    return tuple(map(json.loads, collect_llm_api_config_payloads(args)))


def test_two_process_snapshots_override_parent_values():
    collector = wc.WorkerConfigCollector(2, host="127.0.0.1")
    child = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('worker_config', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
host, port, key, rank, snapshot = json.loads(sys.argv[2])
snapshot['values'] = {path: (False if path == module.PARTIAL_REUSE_FIELD else 0)
                      for path in module.FIELDS}
module.send_snapshot((host, port, bytes.fromhex(key)), rank, snapshot)
"""
    try:
        host, port, key = collector.endpoint
        children = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child,
                    wc.__file__,
                    json.dumps([host, port, key.hex(), rank, snapshot()]),
                ]
            )
            for rank in range(2)
        ]
        for process in children:
            assert process.wait(timeout=10) == 0
    finally:
        observation = collector.finish()
    args = Args()
    args._worker_config_observation = observation
    config, meta = map(json.loads, collect_llm_api_config_payloads(args))
    assert config[HOST_CACHE_FIELD] == config[DISK_CACHE_FIELD] == 0
    assert config[wc.PARTIAL_REUSE_FIELD] is False
    assert config["unrelated"] is True
    assert meta["worker_capture"]["expected"] == meta["worker_capture"]["received"] == 2
    assert meta["worker_capture"]["verified_fields"] == list(wc.FIELDS)
    assert "private" not in json.dumps([config, meta])
    assert key.hex() not in json.dumps([config, meta])
    assert not collector._thread.is_alive()


def test_conflicting_values_not_null_or_parent_fallback():
    first, second = snapshot(), snapshot()
    second["values"][wc.PARTIAL_REUSE_FIELD] = False
    config, meta = captured([first, second])
    assert wc.PARTIAL_REUSE_FIELD not in config
    assert meta["worker_capture"]["conflicting_fields"] == [wc.PARTIAL_REUSE_FIELD]
    assert config[HOST_CACHE_FIELD] == 8192


@pytest.mark.parametrize("remaining_bytes", [0, -1])
def test_payload_budget_applies_after_worker_merge(monkeypatch, remaining_bytes):
    observed = snapshot(
        Args(
            kv_cache_config=CacheConfig(
                host_cache_size=0, disk_cache_size=0, enable_partial_reuse=False
            )
        )
    )
    expected = {**observed["values"], "unrelated": True}
    budget = len(llmapi_config._canonical_json(expected).encode()) + remaining_bytes
    monkeypatch.setattr(llmapi_config, "MAX_CONFIG_BYTES", budget)

    config, meta = captured([observed, observed])

    if remaining_bytes < 0:
        del expected["unrelated"]
    assert config == expected
    assert meta["payload_truncated"] is (remaining_bytes < 0)
    assert meta["captured_field_count"] == len(expected)
    assert len(llmapi_config._canonical_json(config).encode()) <= budget


@pytest.mark.parametrize("observations", [[], [snapshot()]])
def test_missing_worker_is_not_agreement(observations):
    config, meta = captured(observations)
    assert not set(wc.FIELDS) & config.keys()
    assert meta["worker_capture"]["unavailable_fields"] == list(wc.FIELDS)
    assert config["unrelated"] is True


def test_policy_mismatch_is_unavailable():
    other = snapshot()
    other["policy"] = "future"
    config, meta = captured([snapshot(), other])
    assert meta["worker_capture"]["received"] == 1
    assert not set(wc.FIELDS) & config.keys()


def test_null_is_a_value_missing_is_not():
    first, second = snapshot(), snapshot()
    first["values"][HOST_CACHE_FIELD] = second["values"][HOST_CACHE_FIELD] = None
    del second["values"][DISK_CACHE_FIELD]
    config, meta = captured([first, second])
    assert config[HOST_CACHE_FIELD] is None
    assert DISK_CACHE_FIELD not in config
    assert meta["worker_capture"]["unavailable_fields"] == [DISK_CACHE_FIELD]


def packet(collector, rank, value, key=None):
    body = json.dumps({"protocol": 1, "rank": rank, "snapshot": value}).encode()
    return hmac.digest(key or collector._key, body, "sha256") + body


def test_bad_packets_duplicates_and_rank_bounds():
    collector = wc.WorkerConfigCollector(2, host="127.0.0.1")
    try:
        for bad in (
            b"invalid",
            b"x" * 4096,
            packet(collector, 0, snapshot(), b"wrong-key"),
            packet(collector, 2, snapshot()),
            packet(collector, True, snapshot()),
        ):
            collector._accept(bad)
        invalid = snapshot()
        invalid["values"][HOST_CACHE_FIELD] = True
        collector._accept(packet(collector, 0, invalid))
        invalid = snapshot()
        invalid["values"]["kv_cache_config.disk_cache_path"] = "/private"
        collector._accept(packet(collector, 0, invalid))
        assert collector._snapshots == {}
        good = packet(collector, 0, snapshot())
        collector._accept(good)
        collector._accept(good)
        assert len(collector._snapshots) == 1
        different = snapshot()
        different["values"][HOST_CACHE_FIELD] = 0
        collector._accept(packet(collector, 0, different))
        collector._accept(good)
        assert collector._snapshots == {}
    finally:
        collector.finish(wait=False)


def test_missing_receiver_and_timeout_are_bounded():
    start = time.monotonic()
    wc.send_snapshot(("127.0.0.1", 9, b"key"), 0, snapshot())
    collector = wc.WorkerConfigCollector(2, host="127.0.0.1")
    observation = collector.finish()
    assert observation == {"expected": 2, "snapshots": []}
    assert time.monotonic() - start < 2


def test_opt_out_does_not_create_collector(monkeypatch):
    monkeypatch.setattr(wc, "_enabled", lambda args: False)
    monkeypatch.setattr(
        wc, "WorkerConfigCollector", lambda *a, **k: pytest.fail("collector started")
    )
    with wc.observe_workers(SimpleNamespace()):
        pass


def test_cleanup_does_not_swallow_engine_error(monkeypatch):
    monkeypatch.setattr(wc, "_enabled", lambda args: True)
    args = SimpleNamespace(encode_only=False, parallel_config=SimpleNamespace(world_size=2))
    with pytest.raises(RuntimeError, match="engine failed"):
        with wc.observe_workers(args):
            raise RuntimeError("engine failed")
    assert args._worker_config_endpoint is None
    assert args._worker_config_observation["expected"] == 2


def test_worker_opt_out_does_not_send(monkeypatch):
    args = Args()
    args._worker_config_endpoint = ("127.0.0.1", 9, b"key")
    monkeypatch.setattr(wc, "_enabled", lambda args: False)
    monkeypatch.setattr(wc, "send_snapshot", lambda *a: pytest.fail("sent"))
    wc.publish_worker_config(args, 0)


@pytest.mark.parametrize("mode", ["attached", "deferred", "bind_failure"])
def test_unavailable_collection_does_not_fail_startup(monkeypatch, mode):
    monkeypatch.setattr(wc, "_enabled", lambda args: True)
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    args = SimpleNamespace(encode_only=False, parallel_config=SimpleNamespace(world_size=2))
    if mode == "attached":
        monkeypatch.setenv("TLLM_EXECUTOR_ATTACH_INFO", "attached")
    elif mode == "deferred":
        args.ray_placement_config = SimpleNamespace(defer_workers_init=True)

    def start_collector(*args):
        if mode != "bind_failure":
            pytest.fail("unsupported startup must not open a receiver")
        raise OSError("cannot bind")

    monkeypatch.setattr(wc, "WorkerConfigCollector", start_collector)
    with wc.observe_workers(args):
        pass
    assert args._worker_config_endpoint is None
    assert args._worker_config_observation["expected"] == 2
    assert args._worker_config_observation["snapshots"] == []
    assert args._worker_config_observation["pending"] is False


def test_new_construction_does_not_reuse_an_old_observation(monkeypatch):
    monkeypatch.setattr(wc, "_enabled", lambda args: True)
    monkeypatch.setenv("TLLM_EXECUTOR_ATTACH_INFO", "attached")
    args = SimpleNamespace(
        encode_only=False,
        parallel_config=SimpleNamespace(world_size=2),
        _worker_config_observation={"expected": 2, "snapshots": [snapshot(), snapshot()]},
    )
    with wc.observe_workers(args):
        assert args._worker_config_observation is None
    assert args._worker_config_observation["snapshots"] == []


def test_worker_uses_existing_sanitizer(monkeypatch):
    args = Args(kv_cache_config=CacheConfig(enable_partial_reuse=False))
    args._worker_config_endpoint = ("127.0.0.1", 9, b"key")
    monkeypatch.setattr(wc, "_enabled", lambda args: True)
    sent = []
    complete = threading.Event()

    def send(*args):
        sent.append(args)
        complete.set()

    monkeypatch.setattr(wc, "send_snapshot", send)
    wc.publish_worker_config(args, 1)
    assert complete.wait(2)
    assert sent[0][2]["values"][wc.PARTIAL_REUSE_FIELD] is False
    assert "private" not in json.dumps(sent[0][2])


def test_opt_out_during_capture_prevents_worker_send(monkeypatch):
    args = Args()
    args._worker_config_endpoint = ("127.0.0.1", 9, b"key")
    monkeypatch.setattr(wc, "_enabled", lambda args: True)

    def capture_then_disable(args):
        payload = collect_llm_api_config_payloads(args)
        monkeypatch.setattr(wc, "_enabled", lambda args: False)
        return payload

    monkeypatch.setattr(llmapi_config, "collect_llm_api_config_payloads", capture_then_disable)
    monkeypatch.setattr(wc, "send_snapshot", lambda *a: pytest.fail("sent after opt-out"))
    wc.publish_worker_config(args, 0)


def test_late_packet_cannot_cross_instances():
    first = wc.WorkerConfigCollector(1, host="127.0.0.1")
    second = wc.WorkerConfigCollector(1, host="127.0.0.1")
    try:
        second._accept(packet(first, 0, snapshot()))
        assert second._snapshots == {}
    finally:
        first.finish(wait=False)
        second.finish(wait=False)


def test_executor_hook_observes_after_runtime_mutation(monkeypatch):
    pytest.importorskip("torch")
    from tensorrt_llm._torch.pyexecutor import model_loader, py_executor_creator
    from tensorrt_llm.executor.base_worker import BaseWorker

    args = SimpleNamespace(
        backend="pytorch",
        parallel_config=SimpleNamespace(to_mapping=lambda: None),
        is_partial_model_loading=False,
        checkpoint_loader=None,
        checkpoint_format=None,
        mx_config=None,
        checkpoint_io_policy="auto",
        load_format="auto",
        max_seq_len=1024,
        kv_cache_config=CacheConfig(),
    )
    worker = object.__new__(BaseWorker)
    worker._engine = None
    worker.rank = 0
    worker.llm_args = args
    worker._backend = "pytorch"
    worker._hf_model_dir = None
    worker._tokenizer = None
    worker._resource_governor_queue = None
    worker._lora_config = None
    worker.doing_shutdown = True
    monkeypatch.setattr(worker, "_get_comm_ranks_device_id", lambda: None)

    def create_executor(**kwargs):
        kwargs["llm_args"].kv_cache_config.enable_partial_reuse = False
        return SimpleNamespace(max_seq_len=1024)

    monkeypatch.setattr(py_executor_creator, "create_py_executor", create_executor)
    monkeypatch.setattr(model_loader, "_construct_checkpoint_loader", lambda *a, **k: None)
    observed = []
    monkeypatch.setattr(
        wc,
        "publish_worker_config",
        lambda args, rank: observed.append(args.kv_cache_config.enable_partial_reuse),
    )
    worker.setup_engine()
    assert observed == [False]


def test_startup_never_waits_for_receiver_or_snapshots(monkeypatch):
    monkeypatch.setattr(wc, "_enabled", lambda args: True)
    monkeypatch.setattr(wc.socket, "gethostbyname", lambda *a: pytest.fail("startup used DNS"))
    monkeypatch.setattr(
        wc.WorkerConfigCollector, "finish", lambda *a: pytest.fail("startup waited")
    )
    args = SimpleNamespace(encode_only=False, parallel_config=SimpleNamespace(world_size=2))
    with wc.observe_workers(args) as collector:
        assert collector.endpoint is not None
    assert args._worker_config_observation["pending"] is True
    assert len(args._worker_config_observation["capture_id"]) == 32
    collector.cancel()
    assert collector._finished.wait(2)


def test_pending_initial_capture_never_uses_parent_worker_values():
    args = Args()
    args._worker_config_observation = {
        "expected": 2,
        "snapshots": [],
        "pending": True,
        "capture_id": "a" * 32,
    }
    config, meta = map(json.loads, collect_llm_api_config_payloads(args))
    assert not set(wc.FIELDS) & config.keys()
    assert meta["worker_capture"]["status"] == "pending"
    assert meta["capture_id"] == "a" * 32
