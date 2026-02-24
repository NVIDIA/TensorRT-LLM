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
import yaml

from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
from tensorrt_llm.commands.serve import main as serve_main
from tensorrt_llm.llmapi import llm_args as llm_args_module

from .yaml_validation_harness import (
    assert_no_default_valued_leaves,
    collect_yaml_files,
    load_yaml_dict,
    mock_cuda_for_schema_validation,
    validate_torch_llm_args_config,
)

CONFIG_ROOT = Path(__file__).parents[3] / "examples" / "configs"
REPO_ROOT = CONFIG_ROOT.parent.parent
CURATED_DIR = CONFIG_ROOT / "curated"
DATABASE_DIR = CONFIG_ROOT / "database"

CURATED_CONFIGS = collect_yaml_files(CURATED_DIR, "**/*.yaml", exclude_names={"lookup.yaml"})
DATABASE_CONFIGS = collect_yaml_files(DATABASE_DIR, "**/*.yaml", exclude_names={"lookup.yaml"})
ALL_CONFIGS = sorted(CURATED_CONFIGS + DATABASE_CONFIGS)
ALL_LOOKUP_PATHS = sorted((DATABASE_DIR / "lookup.yaml", CURATED_DIR / "lookup.yaml"))


def _load_config_path_to_lookup_entry() -> dict[str, dict]:
    """Build config_path -> {model, arch?} from database and curated lookup.yaml files."""
    result = {}
    for lookup_path in ALL_LOOKUP_PATHS:
        if not lookup_path.exists():
            continue
        with open(lookup_path, encoding="utf-8") as f:
            entries = yaml.safe_load(f)
        if not entries:
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "config_path" not in entry:
                continue
            key = entry["config_path"]
            result[key] = {"model": entry.get("model"), "arch": entry.get("arch")}
    return result


_CONFIG_PATH_TO_ENTRY = _load_config_path_to_lookup_entry()


def _get_default_values_for_config(config_path: Path) -> dict:
    """Get the default values for a config path.

    Some models may have custom default values that are different from the global defaults.
    Based on the model architecture, get the default values that are actually applied.
    """
    try:
        config_key = config_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        config_key = None
    entry = _CONFIG_PATH_TO_ENTRY.get(config_key) if config_key else None
    global_default = llm_args_module.TorchLlmArgs(
        model="dummy/model", skip_tokenizer_init=True
    ).model_dump(mode="json")
    if not entry or not entry.get("arch") or not entry.get("model"):
        return global_default

    model = entry["model"]
    arch = entry["arch"]
    model_cls = MODEL_CLASS_MAPPING.get(arch)
    if not model_cls or not hasattr(model_cls, "get_model_defaults"):
        return global_default

    base_args = llm_args_module.TorchLlmArgs(model=model, skip_tokenizer_init=True).model_dump(
        mode="json"
    )

    model_defaults = model_cls.get_model_defaults(base_args)
    base_args.update(model_defaults)
    return base_args


@pytest.fixture(autouse=True)
def mock_gpu_environment():
    """Mock GPU functions for CPU-only schema test execution."""
    with mock_cuda_for_schema_validation():
        yield


def get_config_id(config_path: Path) -> str:
    return str(config_path.relative_to(CONFIG_ROOT))


def _assert_kv_cache_block_reuse_policy(config_dict: dict) -> None:
    assert config_dict.get("kv_cache_config", {}).get("enable_block_reuse") is not False, (
        "KV cache block reuse should not be disabled"
    )


@pytest.mark.part0
@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=get_config_id)
def test_database_yaml_config_validates_against_llm_args(config_path: Path):
    config_dict = load_yaml_dict(config_path)
    validate_torch_llm_args_config(config_dict)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=get_config_id)
def test_database_yaml_config_does_not_disable_kv_cache_block_reuse(config_path: Path):
    config_dict = load_yaml_dict(config_path)
    _assert_kv_cache_block_reuse_policy(config_dict)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=get_config_id)
def test_database_yaml_config_does_not_set_default_leaves(config_path: Path):
    config_dict = load_yaml_dict(config_path)
    default_cfg = _get_default_values_for_config(config_path)
    assert_no_default_valued_leaves(config_dict, default_cfg)


@pytest.mark.part0
def test_database_yaml_config_count():
    assert len(ALL_CONFIGS) > 0, "No curated or database config files found"


@pytest.mark.part0
def test_all_configs_have_lookup_entry():
    """Every config in ALL_CONFIGS must have an entry in the lookup files (ALL_LOOKUP_PATHS)."""
    config_keys = {p.relative_to(REPO_ROOT).as_posix() for p in ALL_CONFIGS}
    lookup_keys = set(_CONFIG_PATH_TO_ENTRY)
    missing = config_keys - lookup_keys
    assert not missing, (
        "The following configs have no entry in the lookup files. "
        "Add each to the appropriate lookup.yaml file:\n" + "\n".join(sorted(missing))
    )


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
