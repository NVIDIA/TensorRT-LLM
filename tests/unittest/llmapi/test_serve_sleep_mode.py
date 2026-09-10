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

from unittest import mock

import click
import pytest

from tensorrt_llm.commands.serve import main as serve_main
from tensorrt_llm.llmapi import ExecutorMemoryType, SleepConfig

pytestmark = pytest.mark.cpu_only


def _invoke_sleep_mode(extra_args=None):
    args = ["dummy/model", "--enable_sleep_mode"]
    if extra_args:
        args.extend(extra_args)
    return serve_main(args=args, standalone_mode=False)


def test_sleep_mode_requires_environment_key(monkeypatch):
    monkeypatch.delenv("TRTLLM_RUNTIME_CONTROL_API_KEY", raising=False)

    with pytest.raises(click.UsageError, match="TRTLLM_RUNTIME_CONTROL_API_KEY"):
        _invoke_sleep_mode()


def test_sleep_mode_rejects_autodeploy(monkeypatch):
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret")

    with pytest.raises(click.UsageError, match="PyTorch backend"):
        _invoke_sleep_mode(["--backend", "_autodeploy"])


def test_sleep_mode_rejects_grpc(monkeypatch):
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret")

    with pytest.raises(click.UsageError, match="HTTP server"):
        _invoke_sleep_mode(["--grpc"])


def test_sleep_mode_rejects_visual_gen(monkeypatch):
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret")

    with (
        mock.patch("tensorrt_llm.commands.serve.get_is_diffusion_only_model", return_value=False),
        pytest.raises(click.UsageError, match="text-generation server"),
    ):
        _invoke_sleep_mode(["--enable_visual_gen"])


def test_sleep_mode_injects_default_config_and_server_auth(monkeypatch):
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret")

    with (
        mock.patch(
            "tensorrt_llm.commands.serve.get_is_diffusion_only_model",
            return_value=False,
        ),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        mock.patch("tensorrt_llm.commands.serve.launch_server") as launch,
    ):
        _invoke_sleep_mode()

    llm_args = launch.call_args.args[2]
    assert isinstance(llm_args["sleep_config"], SleepConfig)
    assert launch.call_args.kwargs["enable_runtime_control_endpoints"]
    assert launch.call_args.kwargs["runtime_control_api_key"] == "secret"


def test_sleep_mode_preserves_yaml_config(monkeypatch, tmp_path):
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret")
    config = tmp_path / "config.yaml"
    config.write_text(
        "sleep_config:\n  restore_modes:\n    model: CPU\n",
        encoding="utf-8",
    )

    with (
        mock.patch(
            "tensorrt_llm.commands.serve.get_is_diffusion_only_model",
            return_value=False,
        ),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        mock.patch("tensorrt_llm.commands.serve.launch_server") as launch,
    ):
        _invoke_sleep_mode(["--config", str(config)])

    sleep_config = launch.call_args.args[2]["sleep_config"]
    if isinstance(sleep_config, SleepConfig):
        restore_modes = sleep_config.restore_modes
    else:
        restore_modes = sleep_config["restore_modes"]
    mode = restore_modes[ExecutorMemoryType.MODEL_ENGINE_MAIN]
    assert getattr(mode, "name", mode) == "CPU"
