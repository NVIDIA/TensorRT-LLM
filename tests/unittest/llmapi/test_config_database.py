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
"""L0 tests for validating curated and database YAML configs against TorchLlmArgs."""

import asyncio
from pathlib import Path
from unittest import mock

import pytest

from tensorrt_llm.commands.serve import main as serve_main
from tensorrt_llm.llmapi import llm_args as llm_args_module

from . import yaml_validation_harness as yaml_harness

CONFIG_ROOT = Path(__file__).parents[3] / "examples" / "configs"
CURATED_DIR = CONFIG_ROOT / "curated"
DATABASE_DIR = CONFIG_ROOT / "database"

CURATED_CONFIGS = yaml_harness.collect_yaml_files(CURATED_DIR, "**/*.yaml")
DATABASE_CONFIGS = yaml_harness.collect_yaml_files(
    DATABASE_DIR, "**/*.yaml", exclude_names={"lookup.yaml"}
)
ALL_CONFIGS = sorted(CURATED_CONFIGS + DATABASE_CONFIGS)


@pytest.fixture(autouse=True)
def mock_gpu_environment():
    """Mock GPU functions for CPU-only schema test execution."""
    with yaml_harness.mock_cuda_for_schema_validation():
        yield


def get_config_id(config_path: Path) -> str:
    return str(config_path.relative_to(CONFIG_ROOT))


def _assert_kv_cache_block_reuse_policy(config_dict: dict) -> None:
    assert config_dict.get("kv_cache_config", {}).get("enable_block_reuse") is not False


@pytest.mark.part0
@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=get_config_id)
def test_database_yaml_config_validates_against_llm_args(config_path: Path):
    config_dict = yaml_harness.load_yaml_dict(config_path)
    yaml_harness.validate_torch_llm_args_config(config_dict)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=get_config_id)
def test_database_yaml_config_does_not_disable_kv_cache_block_reuse(config_path: Path):
    config_dict = yaml_harness.load_yaml_dict(config_path)
    yaml_harness.assert_custom_policy(config_dict, _assert_kv_cache_block_reuse_policy)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=get_config_id)
def test_database_yaml_config_does_not_set_default_leaves(config_path: Path):
    config_dict = yaml_harness.load_yaml_dict(config_path)
    default_cfg = llm_args_module.TorchLlmArgs(
        model="dummy/model", skip_tokenizer_init=True
    ).model_dump(mode="json")
    yaml_harness.assert_no_default_valued_leaves(config_dict, default_cfg)


@pytest.mark.part0
def test_database_yaml_config_count():
    assert len(ALL_CONFIGS) > 0, "No curated or database config files found"


def _serve_cli_args(config_path: Path, port: int = 17999):
    """CLI argv for serve, like: trtllm-serve <model> --config <config.yaml> --port <port>."""
    return [
        "dummy/model",
        "--config",
        str(config_path),
        "--port",
        str(port),
    ]


async def _noop_serve(_host, _port, sockets=None):
    pass


class _MockOpenAIServer:
    def __init__(self, generator, model, **kwargs):
        self.generator = generator
        self.model = model

    def __call__(self, host, port, sockets=None):
        return _noop_serve(host, port, sockets)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=get_config_id)
def test_database_yaml_config_serve_cli(config_path: Path):
    """Invoke serve via CLI (as a user would); Click supplies defaults; server/LLM mocked."""
    mock_llm = mock.Mock()
    mock_pytorch_llm = mock.Mock(return_value=mock_llm)

    # Run the coroutine (mock server's _noop_serve) so it is awaited and we avoid
    # "coroutine was never awaited". Must capture the real asyncio.run before
    # patching: we patch tensorrt_llm.commands.serve.asyncio.run, which is the
    # same asyncio module this test uses, so asyncio.run would recurse otherwise.
    _real_asyncio_run = asyncio.run

    def _run_coroutine(coroutine):
        return _real_asyncio_run(coroutine)

    with (
        mock.patch("tensorrt_llm.commands.serve.get_is_diffusion_model", return_value=False),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        mock.patch("tensorrt_llm.commands.serve.PyTorchLLM", mock_pytorch_llm),
        mock.patch("tensorrt_llm.commands.serve.OpenAIServer", _MockOpenAIServer),
        mock.patch("tensorrt_llm.commands.serve.asyncio.run", side_effect=_run_coroutine),
    ):
        serve_main(args=_serve_cli_args(config_path), standalone_mode=False)
        mock_pytorch_llm.assert_called_once()
    call_kwargs = mock_pytorch_llm.call_args[1]
    llm_args_module.TorchLlmArgs(**call_kwargs)
