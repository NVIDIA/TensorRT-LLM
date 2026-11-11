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
"""trtllm-serve passes each option of its subcommands to the right launcher parameter.

The CLI stability gate checks that the options exist; these tests check where
their values go. Every value is distinct, so an option passed to the wrong
parameter, or a positional argument that shifted, fails by parameter name.
"""

import contextlib
import importlib
import inspect
import os
import re
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest

import tensorrt_llm.commands.serve as serve_module
from tensorrt_llm.commands.serve import (
    get_llm_args,
    launch_embedding_server,
    launch_mm_encoder_server,
    launch_server,
    launch_visual_gen_server,
    main,
)
from tensorrt_llm.commands.serve_cli import _disagg_fleet as fleet_module
from tensorrt_llm.llmapi import SleepConfig
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.tool_parser import ToolParserFactory

pytestmark = pytest.mark.cpu_only

# What the patched metadata-server config parser returns for a given path.
_METADATA_SERVER_CFG = object()


def _lookup_module(name: str) -> ModuleType:
    """The module where the subcommand's callback looks names up."""
    return inspect.getmodule(main.commands[name].callback)


class _Recorder:
    """Stands in for a launcher; keeps each call's arguments by parameter name."""

    def __init__(self, launcher) -> None:
        self._signature = inspect.signature(launcher)
        self.calls = []

    def __call__(self, *args, **kwargs) -> None:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        self.calls.append(dict(bound.arguments))


class _Wiring:
    """Asserts, by parameter name, what a subcommand passed to its launcher."""

    def __init__(self, subcommand: str, launcher: str, module: ModuleType, recorder: _Recorder):
        assert len(recorder.calls) == 1, (
            f"trtllm-serve {subcommand} called {launcher} {len(recorder.calls)} times, expected once"
        )
        self.subcommand = subcommand
        self.launcher = launcher
        self.module = module
        self.arguments = recorder.calls[0]

    def _check(self, where: str, actual, expected, *, same_object: bool) -> None:
        matches = actual is expected if same_object else actual == expected
        assert matches, (
            f"trtllm-serve {self.subcommand} passed {actual!r} to {self.launcher}({where}), "
            f"expected {expected!r}. An option is wired to the wrong parameter; check the call "
            f"in {self.module.__name__}. Positional arguments shift silently, so bind them by "
            "keyword."
        )

    def expect(self, parameter: str, expected, *, same_object: bool = False) -> None:
        self._check(
            f"{parameter}=...", self.arguments[parameter], expected, same_object=same_object
        )

    def expect_item(self, parameter: str, key: str, expected) -> None:
        self._check(
            f"{parameter}[{key!r}]=...", self.arguments[parameter][key], expected, same_object=False
        )

    def expect_attribute(self, parameter: str, attribute: str, expected) -> None:
        actual = getattr(self.arguments[parameter], attribute)
        self._check(f"{parameter}.{attribute}=...", actual, expected, same_object=False)


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Patch what the subcommands touch on their way to a launcher.

    The metadata-server config parser returns a sentinel for the file the test
    passes (and None without a file, as the real parser does); model inspection
    says "not a diffusion model"; one GPU is reported for the LLM arguments.
    """
    metadata_file = tmp_path / "metadata_server.yaml"
    metadata_file.write_text("{}\n")

    def parse_metadata_server_config_file(path):
        return _METADATA_SERVER_CFG if path is not None else None

    for subcommand in ("serve", "mm_embedding_serve", "embeddings", "disaggregated"):
        monkeypatch.setattr(
            _lookup_module(subcommand),
            "parse_metadata_server_config_file",
            parse_metadata_server_config_file,
        )
    monkeypatch.setattr(_lookup_module("serve"), "get_is_diffusion_only_model", lambda model: False)
    monkeypatch.setattr(inspect.getmodule(get_llm_args), "device_count", lambda: 1)
    return SimpleNamespace(metadata_file=str(metadata_file))


def _record(
    monkeypatch: pytest.MonkeyPatch, subcommand: str, launcher_name: str, launcher
) -> _Recorder:
    recorder = _Recorder(launcher)
    monkeypatch.setattr(_lookup_module(subcommand), launcher_name, recorder)
    return recorder


def test_serve_wires_every_server_option_into_launch_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cli: SimpleNamespace
) -> None:
    module = _lookup_module("serve")
    recorder = _record(monkeypatch, "serve", "launch_server", launch_server)
    config = tmp_path / "serve.yaml"
    config.write_text("internal_request_auth_key: key-x\n")
    report_addr = str(tmp_path / "server.addr")
    tool_parser = next(iter(ToolParserFactory.parsers))

    main(
        args=[
            "serve",
            "org/model-x",
            "--host", "h.test",
            "--port", "12345",
            "--tool_parser", tool_parser,
            "--middleware", "m.mw_a",
            "--chat_template", "tmpl-x",
            "--metadata_server_config_file", cli.metadata_file,
            "--server_role", "context",
            "--disagg_cluster_uri", "etcd://c:2379",
            "--media_io_kwargs", '{"video": {"num_frames": 3}}',
            "--served_model_name", "served-x",
            "--allow_request_chat_template",
            "--num_input_processor_workers", "3",
            "--num_media_load_workers", "5",
            "--report_addr", report_addr,
            "--tp_size", "2",
            "--max_batch_size", "7",
            "--config", str(config),
        ],
        standalone_mode=False,
    )  # fmt: skip

    wiring = _Wiring("serve", "launch_server", module, recorder)
    wiring.expect("host", "h.test")
    wiring.expect("port", 12345)
    wiring.expect("tool_parser", tool_parser)
    wiring.expect("middleware", ("m.mw_a",))
    wiring.expect("chat_template", "tmpl-x")
    wiring.expect("metadata_server_cfg", _METADATA_SERVER_CFG, same_object=True)
    wiring.expect("server_role", ServerRole.CONTEXT, same_object=True)
    wiring.expect_attribute("disagg_cluster_config", "cluster_uri", "etcd://c:2379")
    wiring.expect_attribute(
        "multimodal_server_config", "media_io_kwargs", {"video": {"num_frames": 3}}
    )
    wiring.expect("served_model_name", "served-x")
    wiring.expect("allow_request_chat_template", True, same_object=True)
    wiring.expect("num_input_processor_workers", 3)
    wiring.expect("num_media_load_workers", 5)
    wiring.expect("internal_disagg_auth_key", "key-x")
    wiring.expect("report_addr", report_addr)
    wiring.expect("enable_runtime_control_endpoints", False, same_object=True)
    wiring.expect("runtime_control_api_key", None, same_object=True)
    wiring.expect_item("llm_args", "model", "org/model-x")
    wiring.expect_item("llm_args", "tensor_parallel_size", 2)
    wiring.expect_item("llm_args", "max_batch_size", 7)


def test_serve_sleep_mode_reaches_launch_server(
    monkeypatch: pytest.MonkeyPatch, cli: SimpleNamespace
) -> None:
    module = _lookup_module("serve")
    recorder = _record(monkeypatch, "serve", "launch_server", launch_server)
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret-x")

    main(
        args=["serve", "org/model-x", "--port", "12345", "--enable_sleep_mode"],
        standalone_mode=False,
    )

    wiring = _Wiring("serve", "launch_server", module, recorder)
    wiring.expect("enable_runtime_control_endpoints", True, same_object=True)
    wiring.expect("runtime_control_api_key", "secret-x")
    assert isinstance(wiring.arguments["llm_args"]["sleep_config"], SleepConfig), (
        "trtllm-serve serve --enable_sleep_mode reached launch_server without a SleepConfig "
        f"in llm_args['sleep_config']: {wiring.arguments['llm_args'].get('sleep_config')!r}"
    )


def test_default_subcommand_is_serve(monkeypatch: pytest.MonkeyPatch, cli: SimpleNamespace) -> None:
    module = _lookup_module("serve")
    recorder = _record(monkeypatch, "serve", "launch_server", launch_server)

    main(args=["org/model-x", "--port", "12346"], standalone_mode=False)

    wiring = _Wiring("serve", "launch_server", module, recorder)
    wiring.expect("port", 12346)
    wiring.expect_item("llm_args", "model", "org/model-x")


def test_serve_routes_visual_gen_to_its_launcher(
    monkeypatch: pytest.MonkeyPatch, cli: SimpleNamespace
) -> None:
    module = _lookup_module("serve")
    http_recorder = _record(monkeypatch, "serve", "launch_server", launch_server)
    recorder = _record(monkeypatch, "serve", "launch_visual_gen_server", launch_visual_gen_server)

    main(
        args=[
            "serve",
            "org/model-x",
            "--host", "h.test",
            "--port", "12345",
            "--enable_visual_gen",
            "--middleware", "m.mw_b",
            "--metadata_server_config_file", cli.metadata_file,
        ],
        standalone_mode=False,
    )  # fmt: skip

    assert http_recorder.calls == [], (
        "trtllm-serve serve --enable_visual_gen must not start the OpenAI HTTP server, but "
        f"called launch_server with {http_recorder.calls}"
    )
    wiring = _Wiring("serve", "launch_visual_gen_server", module, recorder)
    wiring.expect("host", "h.test")
    wiring.expect("port", 12345)
    wiring.expect("model", "org/model-x")
    wiring.expect("visual_gen_args", None, same_object=True)
    wiring.expect("metadata_server_cfg", _METADATA_SERVER_CFG, same_object=True)
    wiring.expect("middleware", ("m.mw_b",))


def test_mm_embedding_serve_wiring(monkeypatch: pytest.MonkeyPatch, cli: SimpleNamespace) -> None:
    module = _lookup_module("mm_embedding_serve")
    recorder = _record(
        monkeypatch, "mm_embedding_serve", "launch_mm_encoder_server", launch_mm_encoder_server
    )

    main(
        args=[
            "mm_embedding_serve",
            "org/enc-x",
            "--host", "h.test",
            "--port", "12347",
            "--allow_request_chat_template",
            "--metadata_server_config_file", cli.metadata_file,
        ],
        standalone_mode=False,
    )  # fmt: skip

    wiring = _Wiring("mm_embedding_serve", "launch_mm_encoder_server", module, recorder)
    wiring.expect("host", "h.test")
    wiring.expect("port", 12347)
    wiring.expect_item("encoder_args", "model", "org/enc-x")
    wiring.expect("metadata_server_cfg", _METADATA_SERVER_CFG, same_object=True)
    wiring.expect("allow_request_chat_template", True, same_object=True)


def test_embeddings_wiring(monkeypatch: pytest.MonkeyPatch, cli: SimpleNamespace) -> None:
    module = _lookup_module("embeddings")
    recorder = _record(
        monkeypatch, "embeddings", "launch_embedding_server", launch_embedding_server
    )

    main(
        args=[
            "embeddings",
            "org/emb-x",
            "--host", "h.test",
            "--port", "12348",
            "--max_queue_delay", "0.25",
            "--max_queue_size", "17",
            "--metadata_server_config_file", cli.metadata_file,
        ],
        standalone_mode=False,
    )  # fmt: skip

    wiring = _Wiring("embeddings", "launch_embedding_server", module, recorder)
    wiring.expect("host", "h.test")
    wiring.expect("port", 12348)
    wiring.expect("max_queue_delay", 0.25)
    wiring.expect("max_queue_size", 17)
    wiring.expect("metadata_server_cfg", _METADATA_SERVER_CFG, same_object=True)
    wiring.expect_item("llm_args", "model", "org/emb-x")


# --- the disaggregated subcommands -------------------------------------------

_DISAGG_CONFIG_FILE = "disagg.yaml"
_ROLE_ENV = fleet_module.DisaggLauncherEnvs.TLLM_DISAGG_ROLE


class _ServerRecorder(_Recorder):
    """Stands in for a server class; the instance it hands back records the serve call."""

    def __init__(self, server_class) -> None:
        super().__init__(server_class)
        self.served = []

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        return lambda host, port, sockets=None: self.served.append((host, port))


def _disagg_config(num_workers: int, coordinator_url: Optional[str]) -> SimpleNamespace:
    """What the patched disaggregated config parser returns."""
    return SimpleNamespace(
        hostname="h.test",
        port=12350,
        num_workers=num_workers,
        disagg_coordinator_url=coordinator_url,
        schedule_style=None,
    )


@pytest.fixture
def disagg_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cli: SimpleNamespace
) -> SimpleNamespace:
    """Patch what the disaggregated subcommands touch on their way to a launcher.

    The telemetry configuration and the lifecycle phase are no-ops, the
    single-server path keeps the garbage collector on, and the variables the
    launchers publish into the environment are restored afterwards.
    """
    module = _lookup_module("disaggregated")
    monkeypatch.setattr(
        module._command_telemetry,
        "apply_disaggregated_telemetry_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(module, "set_lifecycle_phase", lambda phase: None)
    # set_prometheus_multiproc_dir() deletes the directory its own variable
    # points at on its second call in a process; a directory that outlives the
    # call keeps these tests independent of each other.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    monkeypatch.setenv("TRTLLM_DISAGG_SERVER_DISABLE_GC", "0")
    for name in (
        _ROLE_ENV,
        fleet_module.DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID,
        "TRTLLM_DISAGG_SHARED_DRAFT_KV",
    ):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(module=module, metadata_file=cli.metadata_file)


@pytest.mark.parametrize(
    "num_workers,coordinator_url,launcher_name",
    [
        pytest.param(3, "http://coord.test:9000", "_serve_disagg_fleet", id="external-coordinator"),
        pytest.param(2, None, "_serve_coordinator_and_fleet", id="implicit-coordinator"),
        pytest.param(1, None, "OpenAIDisaggServer", id="single-server"),
    ],
)
def test_disaggregated_dispatches_by_topology(
    monkeypatch: pytest.MonkeyPatch,
    disagg_cli: SimpleNamespace,
    num_workers: int,
    coordinator_url: Optional[str],
    launcher_name: str,
) -> None:
    module = disagg_cli.module
    config = _disagg_config(num_workers, coordinator_url)
    monkeypatch.setattr(module, "parse_disagg_config_file", lambda path: config)
    launcher = getattr(module, launcher_name)
    recorder = (
        _ServerRecorder(launcher) if launcher_name == "OpenAIDisaggServer" else _Recorder(launcher)
    )
    monkeypatch.setattr(module, launcher_name, recorder)
    # The single server binds its socket before it serves; neither reaches the network here.
    bound = SimpleNamespace(getsockname=lambda: (config.hostname, config.port))
    monkeypatch.setattr(
        module, "_bind_listener", lambda *args, **kwargs: contextlib.nullcontext(bound)
    )
    served = []
    monkeypatch.setattr(module, "uvloop", SimpleNamespace(run=served.append))

    main(
        args=[
            "disaggregated",
            "-c", _DISAGG_CONFIG_FILE,
            "-m", disagg_cli.metadata_file,
            "-r", "17",
            "-t", "23",
            "--schedule_style", "generation_first",
        ],
        standalone_mode=False,
    )  # fmt: skip

    wiring = _Wiring("disaggregated", launcher_name, module, recorder)
    assert config.schedule_style == "generation_first", (
        "trtllm-serve disaggregated --schedule_style must reach disagg_cfg.schedule_style; "
        f"found {config.schedule_style!r}"
    )
    if launcher_name == "OpenAIDisaggServer":
        wiring.expect("config", config, same_object=True)
        wiring.expect("req_timeout_secs", 17)
        wiring.expect("server_start_timeout_secs", 23)
        wiring.expect("metadata_server_cfg", _METADATA_SERVER_CFG, same_object=True)
        wiring.expect("metrics_interval_secs", 0)
        assert os.environ[_ROLE_ENV] == "server_coordinator"
        assert recorder.served == [(config.hostname, config.port)] and len(served) == 1, (
            "the single disaggregated server must be served once on its configured address; "
            f"recorded {recorder.served} and {len(served)} event loop runs"
        )
    else:
        wiring.expect("disagg_cfg", config, same_object=True)
        wiring.expect("config_file", _DISAGG_CONFIG_FILE)
        wiring.expect("metadata_server_config_file", disagg_cli.metadata_file)
        wiring.expect("request_timeout", 17)
        wiring.expect("server_start_timeout", 23)
        wiring.expect("num_workers", num_workers)
        if launcher_name == "_serve_disagg_fleet":
            wiring.expect("coordinator_url", coordinator_url)
            assert _ROLE_ENV not in os.environ, (
                "a fleet that delegates to an external coordinator publishes no role of its own; "
                f"found {_ROLE_ENV}={os.environ.get(_ROLE_ENV)!r}"
            )
        else:
            wiring.expect("metadata_server_cfg", _METADATA_SERVER_CFG, same_object=True)
            assert os.environ[_ROLE_ENV] == "coordinator"


def test_disaggregated_mpi_worker_remote_client_wiring(
    monkeypatch: pytest.MonkeyPatch, disagg_cli: SimpleNamespace
) -> None:
    module = disagg_cli.module
    server = SimpleNamespace(
        type="ctx",
        hostname="h.test",
        port=12351,
        other_args={
            "model": "org/model-x",
            "tensor_parallel_size": 2,
            "router": {"type": "round_robin"},
        },
    )
    config = SimpleNamespace(server_configs=[server], ctx_frontend_affinity=None)
    monkeypatch.setattr(module, "parse_disagg_config_file", lambda path: config)
    monkeypatch.setenv(
        fleet_module.DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT, "1"
    )
    monkeypatch.setenv(fleet_module.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX, "0")
    recorder = _record(
        monkeypatch,
        "disaggregated_mpi_worker",
        "_launch_disaggregated_server",
        module._launch_disaggregated_server,
    )

    main(args=["disaggregated_mpi_worker", "-c", _DISAGG_CONFIG_FILE], standalone_mode=False)

    wiring = _Wiring("disaggregated_mpi_worker", "_launch_disaggregated_server", module, recorder)
    wiring.expect("disagg_config_file", _DISAGG_CONFIG_FILE)
    wiring.expect_item("llm_args", "model", "org/model-x")
    wiring.expect_item("llm_args", "tensor_parallel_size", 2)
    assert "router" not in wiring.arguments["llm_args"], (
        "trtllm-serve disaggregated_mpi_worker passed the server's router setting on as an LLM "
        f"argument: {wiring.arguments['llm_args'].get('router')!r}. The router belongs to the "
        "disaggregated server, not to the LLM; drop it before launching the server."
    )


@pytest.mark.parametrize("server_type,expected_role", [("ctx", "context"), ("gen", "generation")])
@pytest.mark.parametrize("auth_key", [None, "key-x"], ids=["no-auth-key", "auth-key"])
def test_launch_disaggregated_server_wiring(
    monkeypatch: pytest.MonkeyPatch,
    disagg_cli: SimpleNamespace,
    server_type: str,
    expected_role: str,
    auth_key: Optional[str],
) -> None:
    module = disagg_cli.module
    server = SimpleNamespace(type=server_type, hostname="h.test", port=12352)
    config = SimpleNamespace(server_configs=[server], allow_request_chat_template=True)
    if auth_key is not None:
        config.internal_request_auth_key = auth_key
    monkeypatch.setattr(module, "parse_disagg_config_file", lambda path: config)
    monkeypatch.setattr(module, "mpi_rank", lambda: 0)
    monkeypatch.setenv(fleet_module.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX, "0")
    recorder = _Recorder(launch_server)
    # The launcher is looked up in tensorrt_llm.commands.serve while it lives there, and in
    # the module of _launch_disaggregated_server once it moves there: patch both.
    monkeypatch.setattr(serve_module, "launch_server", recorder)
    monkeypatch.setattr(module, "launch_server", recorder, raising=False)
    llm_args = {"backend": "pytorch", "model": "org/model-x"}

    module._launch_disaggregated_server(_DISAGG_CONFIG_FILE, llm_args)

    wiring = _Wiring("disaggregated_mpi_worker", "launch_server", module, recorder)
    wiring.expect("host", "h.test")
    wiring.expect("port", 12352)
    wiring.expect("llm_args", llm_args, same_object=True)
    wiring.expect("allow_request_chat_template", True, same_object=True)
    wiring.expect("multi_frontend_enabled", False, same_object=True)
    wiring.expect("internal_disagg_auth_key", auth_key)
    assert os.environ[_ROLE_ENV] == expected_role, (
        f"a {server_type} server must publish the {expected_role} role in {_ROLE_ENV} before "
        f"it launches; found {os.environ.get(_ROLE_ENV)!r}"
    )


def test_fleet_worker_command_imports_the_fleet_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = inspect.getmodule(fleet_module._launch_disagg_fleet)
    commands = []

    class _FakePopen:
        pid = 4242

        def __init__(self, command, **kwargs) -> None:
            commands.append(command)

        def poll(self):
            return None

    monkeypatch.setattr(module, "subprocess", SimpleNamespace(Popen=_FakePopen))
    monkeypatch.setattr(module, "atexit", SimpleNamespace(register=lambda cleanup: None))
    monkeypatch.setattr(
        module,
        "signal",
        SimpleNamespace(
            signal=lambda signum, handler: None, SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT
        ),
    )
    config = SimpleNamespace(hostname="h.test", port=12353, schedule_style="context_first")

    module._launch_disagg_fleet(
        config, _DISAGG_CONFIG_FILE, None, 180, 180, 1, "http://coord.test:9000"
    )

    expected = [
        sys.executable,
        "-c",
        "from tensorrt_llm.commands.serve_cli._disagg_fleet import _run_fleet_worker; "
        "_run_fleet_worker()",
    ]
    assert commands == [expected], f"the fleet worker command changed: {commands}"
    match = re.fullmatch(
        r"from (\S+) import _run_fleet_worker; _run_fleet_worker\(\)", commands[0][2]
    )
    assert match is not None, commands[0][2]
    worker_module = importlib.import_module(match.group(1))
    assert hasattr(worker_module, "_run_fleet_worker"), (
        f"The fleet worker command imports {match.group(1)}, which has no _run_fleet_worker. Keep "
        "the -c string in _launch_disagg_fleet and the module that defines _run_fleet_worker in "
        "step: no import check sees a module path inside a string."
    )
