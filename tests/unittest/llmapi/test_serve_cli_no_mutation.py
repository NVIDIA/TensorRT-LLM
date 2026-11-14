# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The trtllm-serve config path must not mutate caller-owned config mappings.

Each test pins two things for one function on the serve config path: the
mapping the caller passed in is unchanged after the call, and the value handed
downstream is the one today's code hands down. ``_pop_bool_config_option`` is
the one declared exception (removing the key from the caller's mapping is the
effect both callers need) and has no test here.
"""

import copy
import socket
from types import SimpleNamespace
from unittest import mock

import pytest

import tensorrt_llm.commands.serve as serve_module
import tensorrt_llm.serve.router as router_module
from tensorrt_llm.commands.serve import main as serve_main
from tensorrt_llm.llmapi import SleepConfig
from tensorrt_llm.llmapi.disagg_utils import (
    DisaggClusterConfig,
    MinimalInstances,
    RouterConfig,
    extract_disagg_cluster_config,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, update_llm_args_with_extra_dict
from tensorrt_llm.serve._frontend_processes import _init_multi_frontend_mode
from tensorrt_llm.usage.config import TelemetryConfig, UsageContext

pytestmark = pytest.mark.cpu_only

_MISSING = object()


def _assert_unchanged(function: str, mapping: dict, snapshot: dict) -> None:
    changed = sorted(
        k
        for k in set(mapping) | set(snapshot)
        if mapping.get(k, _MISSING) != snapshot.get(k, _MISSING)
    )
    assert mapping == snapshot, (
        f"{function} changed the mapping its caller passed in "
        f"({', '.join(changed)}). Copy it first, or return the new mapping "
        "and let the caller rebind (parameter objects need side-effect-free "
        "callees)."
    )


class _Stop(Exception):
    """Raised by a constructor stand-in to stop a launcher after the merge."""


class _Recorder:
    """Records the constructor call a launcher makes, then stops it."""

    def __init__(self):
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        raise _Stop


class _FakeSocket:
    """A bind-free stand-in for the listening socket of launch_server."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def setsockopt(self, *args, **kwargs):
        pass

    def bind(self, address):
        pass

    def getsockname(self):
        return ("127.0.0.1", 12345)

    def close(self):
        pass


def test_apply_effective_telemetry_config_does_not_mutate_llm_args(monkeypatch):
    applied = []
    monkeypatch.setattr(
        serve_module, "apply_usage_session_config", lambda config, **kwargs: applied.append(config)
    )
    telemetry_config = TelemetryConfig()
    llm_args = {"model": "m", "telemetry_config": telemetry_config}
    snapshot = dict(llm_args)

    effective = serve_module._apply_effective_telemetry_config(llm_args, telemetry=False)

    _assert_unchanged("_apply_effective_telemetry_config", llm_args, snapshot)
    assert llm_args["telemetry_config"] is telemetry_config
    assert telemetry_config.disabled is False
    assert effective["telemetry_config"].disabled is True
    assert applied and applied[0].disabled is True

    # With telemetry enabled the mapping passes through as it is.
    effective = serve_module._apply_effective_telemetry_config(llm_args, telemetry=True)
    assert effective["telemetry_config"] is telemetry_config


def test_provision_kv_cache_pool_does_not_mutate_llm_args():
    llm_args = {
        "model": "m",
        "kv_connector_config": {
            "connector_module": "mod",
            "connector_scheduler_class": "Scheduler",
            "connector_worker_class": "Worker",
        },
    }
    snapshot = copy.deepcopy(llm_args)

    with serve_module._provision_kv_cache_pool(llm_args) as effective:
        # The validated model is handed on so it is not parsed twice.
        validated = effective["kv_connector_config"]
        assert isinstance(validated, KvCacheConnectorConfig)
        assert validated.connector_module == "mod"

    _assert_unchanged("_provision_kv_cache_pool", llm_args, snapshot)
    assert isinstance(llm_args["kv_connector_config"], dict)

    # A process that does not own the engine gets the mapping as it is.
    with serve_module._provision_kv_cache_pool(llm_args, owns_engine=False) as effective:
        assert effective is llm_args
    _assert_unchanged("_provision_kv_cache_pool", llm_args, snapshot)


def test_launch_server_does_not_mutate_llm_args(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 12345)),
        ],
    )
    monkeypatch.setattr(socket, "socket", _FakeSocket)
    recorder = _Recorder()
    monkeypatch.setattr(serve_module, "PyTorchLLM", recorder)

    llm_args = {
        "model": "m",
        "backend": "pytorch",
        "build_config": {"max_batch_size": 8},
        "num_serve_frontends": 2,
    }
    snapshot = copy.deepcopy(llm_args)

    with pytest.raises(_Stop):
        serve_module.launch_server(
            host="127.0.0.1", port=12345, llm_args=llm_args, multi_frontend_enabled=False
        )

    _assert_unchanged("launch_server", llm_args, snapshot)
    assert "build_config" not in recorder.kwargs
    assert "num_serve_frontends" not in recorder.kwargs
    assert recorder.kwargs["model"] == "m"
    assert recorder.kwargs["backend"] == "pytorch"


@pytest.mark.parametrize("enabled", [False, True])
def test_init_multi_frontend_mode_does_not_mutate_llm_args(monkeypatch, enabled):
    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    llm_args = {"num_serve_frontends": 3}
    snapshot = dict(llm_args)

    mode = _init_multi_frontend_mode(llm_args, enabled=enabled)

    _assert_unchanged("_init_multi_frontend_mode", llm_args, snapshot)
    assert mode.num_frontends == (3 if enabled else 1)
    assert not mode.is_attached_frontend


def test_launch_mm_encoder_server_does_not_mutate_encoder_args(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(serve_module, "MultimodalEncoder", recorder)
    encoder_args = {"model": "m", "build_config": {"max_batch_size": 8}}
    snapshot = copy.deepcopy(encoder_args)

    with pytest.raises(_Stop):
        serve_module.launch_mm_encoder_server(
            host="127.0.0.1", port=12345, encoder_args=encoder_args
        )

    _assert_unchanged("launch_mm_encoder_server", encoder_args, snapshot)
    assert "build_config" not in recorder.kwargs
    assert recorder.kwargs["model"] == "m"


def test_launch_embedding_server_does_not_mutate_llm_args(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(serve_module, "PyTorchLLM", recorder)
    monkeypatch.setattr(
        serve_module,
        "_resolve_embedding_architecture_override",
        lambda *args, **kwargs: {"architectures": ["FakeForTextEmbedding"]},
    )
    llm_args = {
        "model": "m",
        "build_config": {"max_batch_size": 8},
        "encode_only": False,
    }
    snapshot = copy.deepcopy(llm_args)

    with pytest.raises(_Stop):
        serve_module.launch_embedding_server(
            host="127.0.0.1", port=12345, llm_args=llm_args, max_queue_delay=0.0, max_queue_size=1
        )

    _assert_unchanged("launch_embedding_server", llm_args, snapshot)
    assert "build_config" not in recorder.kwargs
    assert recorder.kwargs["encode_only"] is True
    assert recorder.kwargs["model_kwargs"] == {"architectures": ["FakeForTextEmbedding"]}


def test_resolve_default_num_serve_frontends_does_not_mutate_llm_args():
    llm_args = {"model": "m", "num_serve_frontends": 8}
    snapshot = dict(llm_args)

    effective = serve_module._resolve_default_num_serve_frontends(
        llm_args, requested=False, port=8000, grpc=False, report_addr=None, enable_sleep_mode=True
    )

    _assert_unchanged("_resolve_default_num_serve_frontends", llm_args, snapshot)
    # The single-frontend fallback lands in the returned mapping only.
    assert effective["num_serve_frontends"] == 1

    # Without a conflict the mapping passes through as it is.
    effective = serve_module._resolve_default_num_serve_frontends(
        llm_args, requested=False, port=8000, grpc=False, report_addr=None, enable_sleep_mode=False
    )
    assert effective is llm_args


def test_serve_llm_sleep_config_default_reaches_launcher(monkeypatch):
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret")
    with (
        mock.patch("tensorrt_llm.commands.serve.get_is_diffusion_only_model", return_value=False),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        mock.patch("tensorrt_llm.commands.serve.launch_server") as launch,
    ):
        serve_main(args=["dummy/model", "--enable_sleep_mode"], standalone_mode=False)

    assert isinstance(launch.call_args.kwargs["llm_args"]["sleep_config"], SleepConfig)


def test_serve_llm_disagg_cluster_key_leaves_launcher_llm_args(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("disagg_cluster:\n  cluster_uri: etcd://cfg:2379\n", encoding="utf-8")
    with (
        mock.patch("tensorrt_llm.commands.serve.get_is_diffusion_only_model", return_value=False),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        mock.patch("tensorrt_llm.commands.serve.launch_server") as launch,
    ):
        serve_main(
            args=[
                "dummy/model",
                "--config",
                str(config_path),
                "--server_role",
                "context",
                "--disagg_cluster_uri",
                "etcd://flag:2379",
            ],
            standalone_mode=False,
        )

    kwargs = launch.call_args.kwargs
    assert "disagg_cluster" not in kwargs["llm_args"]
    cluster = kwargs["disagg_cluster_config"]
    assert isinstance(cluster, DisaggClusterConfig)
    # A cluster_uri in the YAML wins over --disagg_cluster_uri today; the flag
    # only fills a missing one. The precedence change belongs to #20490.
    assert cluster.cluster_uri == "etcd://cfg:2379"


def test_disaggregated_mpi_worker_does_not_mutate_built_llm_args(monkeypatch):
    monkeypatch.setenv(
        serve_module.DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT, "1"
    )
    monkeypatch.setenv(serve_module.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX, "0")
    # The worker resolves the shared draft KV mode into the environment; the
    # test restores it afterwards.
    monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", "auto")
    built = {"model": "m", "router": {"type": "round_robin"}}
    snapshot = copy.deepcopy(built)
    launched = {}

    def fake_launch(config_file, llm_args):
        launched["llm_args"] = llm_args

    stub_cfg = SimpleNamespace(
        server_configs=[SimpleNamespace(type="ctx", other_args={})], ctx_frontend_affinity=None
    )
    with (
        mock.patch.object(serve_module._command_telemetry, "apply_disaggregated_telemetry_config"),
        mock.patch.object(serve_module, "parse_disagg_config_file", return_value=stub_cfg),
        mock.patch.object(
            serve_module, "_build_llm_args_from_disagg_server_cfg", return_value=built
        ),
        mock.patch.object(serve_module, "_launch_disaggregated_server", side_effect=fake_launch),
    ):
        serve_module.disaggregated_mpi_worker.callback(
            config_file="cfg.yaml", log_level="info", telemetry=True
        )

    _assert_unchanged("disaggregated_mpi_worker", built, snapshot)
    assert launched["llm_args"] == {"model": "m"}


def test_create_router_does_not_mutate_router_config_args(monkeypatch):
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(router_module, "RoundRobinRouter", recorder)
    router_config = RouterConfig(type="round_robin", args={"foo": 1})
    snapshot = dict(router_config.args)

    router_module.create_router(router_config, ["http://s1"], disagg_node_id=7)

    _assert_unchanged("create_router", router_config.args, snapshot)
    _args, kwargs = calls[0]
    assert kwargs["foo"] == 1
    assert kwargs["disagg_node_id"] == 7


def test_openengine_launch_server_does_not_mutate_llm_args(monkeypatch):
    from tensorrt_llm.grpc.openengine import server as oe_server

    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    recorder = _Recorder()
    monkeypatch.setattr(oe_server, "PyTorchLLM", recorder)
    llm_args = {
        "model": "m",
        "backend": "pytorch",
        "build_config": {"max_batch_size": 8},
    }
    snapshot = copy.deepcopy(llm_args)

    with pytest.raises(_Stop):
        oe_server.launch_server(
            host="127.0.0.1", port=12345, llm_args=llm_args, enable_load_metrics=True
        )

    _assert_unchanged("launch_server", llm_args, snapshot)
    assert "build_config" not in recorder.kwargs
    assert recorder.kwargs["_enable_routing_load"] is True
    assert recorder.kwargs["_openengine_discovery"]["host"] == "127.0.0.1"
    assert recorder.kwargs["_openengine_discovery"]["port"] == 12345


def test_openengine_cli_passes_provisioned_connector_config(monkeypatch):
    from tensorrt_llm.grpc.openengine import server as oe_server

    monkeypatch.delenv("TLLM_EXECUTOR_ATTACH_INFO", raising=False)
    llm_args = {
        "model": "m",
        "backend": "pytorch",
        "num_serve_frontends": 1,
        "kv_connector_config": {
            "connector_module": "mod",
            "connector_scheduler_class": "Scheduler",
            "connector_worker_class": "Worker",
        },
    }
    snapshot = copy.deepcopy(llm_args)
    with (
        mock.patch.object(serve_module, "get_is_diffusion_only_model", return_value=False),
        mock.patch.object(serve_module, "device_count", return_value=1),
        mock.patch.object(serve_module, "get_llm_args", return_value=(llm_args, None)),
        mock.patch.object(oe_server, "launch_server") as launch,
    ):
        serve_main(
            args=["dummy/model", "--grpc", "--grpc-protocol", "openengine"],
            standalone_mode=False,
        )

    _assert_unchanged("OpenEngine CLI", llm_args, snapshot)
    launch.assert_called_once()
    connector = launch.call_args.kwargs["llm_args"]["kv_connector_config"]
    assert isinstance(connector, KvCacheConnectorConfig)
    assert connector.connector_module == "mod"


def test_smg_launch_server_does_not_mutate_llm_args(monkeypatch):
    pytest.importorskip("smg_grpc_proto")
    from tensorrt_llm.grpc.smg import server as smg_server

    recorder = _Recorder()
    monkeypatch.setattr(smg_server, "PyTorchLLM", recorder)
    llm_args = {
        "model": "m",
        "backend": "pytorch",
        "build_config": {"max_batch_size": 8},
        "kv_connector_config": {
            "connector_module": "mod",
            "connector_scheduler_class": "Scheduler",
            "connector_worker_class": "Worker",
        },
    }
    snapshot = copy.deepcopy(llm_args)

    with pytest.raises(_Stop):
        smg_server.launch_smg_server(host="127.0.0.1", port=12346, llm_args=llm_args)

    _assert_unchanged("launch_smg_server", llm_args, snapshot)
    assert "build_config" not in recorder.kwargs
    assert recorder.kwargs["model"] == "m"
    assert isinstance(recorder.kwargs["kv_connector_config"], KvCacheConnectorConfig)


def test_update_llm_args_with_extra_dict_does_not_mutate_yaml_telemetry():
    yaml_tc = {"disabled": True, "usage_context": "cli_serve"}
    yaml_dict = {"telemetry_config": yaml_tc}
    llm_args = {"model": "m", "telemetry_config": {"disabled": False}}
    snapshot = copy.deepcopy(yaml_dict)

    merged = update_llm_args_with_extra_dict(llm_args, yaml_dict)

    _assert_unchanged("update_llm_args_with_extra_dict", yaml_dict, snapshot)
    assert yaml_dict["telemetry_config"] is yaml_tc
    assert yaml_tc["usage_context"] == "cli_serve"
    merged_tc = merged["telemetry_config"]
    assert merged_tc.disabled is True
    # The YAML usage_context is dropped: the CLI entry point owns it.
    assert merged_tc.usage_context == UsageContext.UNKNOWN


def test_extract_disagg_cluster_config_does_not_mutate_cluster_config_dict():
    cluster_config_dict = {
        "cluster_uri": "etcd://cfg:2379",
        "minimal_instances": {"context_servers": 2, "generation_servers": 3},
    }
    snapshot = copy.deepcopy(cluster_config_dict)

    first = extract_disagg_cluster_config(cluster_config_dict, "etcd://flag:2379")

    _assert_unchanged("extract_disagg_cluster_config", cluster_config_dict, snapshot)
    assert isinstance(cluster_config_dict["minimal_instances"], dict)
    # Preserve current precedence here; #20490 changes it separately.
    assert first.cluster_uri == "etcd://cfg:2379"
    assert first.minimal_instances == MinimalInstances(context_servers=2, generation_servers=3)

    # A second call on the same mapping gives an equal result.
    second = extract_disagg_cluster_config(cluster_config_dict, "etcd://flag:2379")
    assert second == first
