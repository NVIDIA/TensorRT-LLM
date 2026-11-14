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
"""trtllm-serve passes each option of its LLM subcommands to the right launcher parameter.

The CLI stability gate checks that the options exist; these tests check where
their values go. Every value is distinct, so an option passed to the wrong
parameter, or a positional argument that shifted, fails by parameter name.
"""

import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tensorrt_llm.commands.serve import (
    get_llm_args,
    launch_embedding_server,
    launch_mm_encoder_server,
    launch_server,
    launch_visual_gen_server,
    main,
)
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

    for subcommand in ("serve", "mm_embedding_serve", "embeddings"):
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
