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
"""Test that all config YAMLs under examples/auto_deploy/ can be ingested by build_and_run_ad.py.

This test recursively discovers YAML config files and validates that each one can be loaded
as a yaml_extra for LlmArgs, which is the dry-run equivalent of:
    python examples/auto_deploy/build_and_run_ad.py --model <model> --args.yaml-extra <config> --dry-run
"""

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

# Root directory for the example configs
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_AD_EXAMPLES_DIR = _REPO_ROOT / "examples" / "auto_deploy"

# Files that are not LlmArgs configs and should be excluded from validation
_EXCLUDED_FILES = {
    "models.yaml",  # model registry index, not an LlmArgs config
    "flux_transforms.yaml",  # for build_and_run_flux.py, different schema
    "nemotron_fp8_ir_test.yaml",  # build_and_run_ad ExperimentConfig, not an LlmArgs config
}

# Dummy model name used during validation (model path is not resolved during construction)
_DUMMY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _find_config_yamls():
    """Recursively find all config YAML files under examples/auto_deploy/."""
    assert _AD_EXAMPLES_DIR.is_dir(), f"Expected directory: {_AD_EXAMPLES_DIR}"
    yaml_files = sorted(_AD_EXAMPLES_DIR.rglob("*.yaml"))
    return [f for f in yaml_files if f.name not in _EXCLUDED_FILES]


@pytest.fixture(
    params=_find_config_yamls(),
    ids=lambda p: str(p.relative_to(_AD_EXAMPLES_DIR)),
)
def config_yaml_path(request):
    return request.param


def test_config_yaml_is_valid_yaml(config_yaml_path):
    """Each config YAML must be syntactically valid and contain a top-level dict."""
    with open(config_yaml_path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"Expected top-level dict, got {type(data)}"


def _is_cross_validation_error(e: ValidationError) -> bool:
    """Check if all errors are cross-field validation failures expected for fragment configs.

    Fragment configs (e.g., model_registry/configs/) are designed to be composed with other
    configs. When loaded in isolation, the default CudaGraphConfig (max_batch_size=128) may
    conflict with a fragment's small max_batch_size. This cross-validation failure is expected
    and should not fail the test.
    """
    for err in e.errors():
        msg = str(err.get("msg", ""))
        if "max_batch_size" in msg and "cuda_graph_config" in msg:
            continue
        if "max_num_tokens" in msg and "max_batch_size" in msg:
            continue
        return False
    return True


def test_config_yaml_dry_run_ingestion(config_yaml_path):
    """Each config YAML must be ingestible as an LlmArgs yaml_extra without errors.

    This is the programmatic equivalent of:
        python build_and_run_ad.py --model <dummy> --args.yaml-extra <config> --dry-run

    Cross-field validation errors between max_batch_size and cuda_graph_config are tolerated
    because these configs are fragments meant to be composed, not used standalone.
    """
    try:
        LlmArgs(model=_DUMMY_MODEL, yaml_extra=[str(config_yaml_path)])
    except ValidationError as e:
        if not _is_cross_validation_error(e):
            raise


def test_functional_single_request_cuda_graph_overrides_deepseek_v4_dashboard_config():
    """Functional gates can stack a final config to avoid capturing dashboard batch sizes."""
    config_dir = _AD_EXAMPLES_DIR / "model_registry" / "configs"
    args = LlmArgs(
        model=_DUMMY_MODEL,
        yaml_extra=[
            str(config_dir / "dashboard_default.yaml"),
            str(config_dir / "world_size_8.yaml"),
            str(config_dir / "deepseek_v4_flash.yaml"),
            str(config_dir / "num_hidden_layers_5.yaml"),
            str(config_dir / "functional_single_request_cuda_graph.yaml"),
        ],
    )

    assert args.model_kwargs["num_hidden_layers"] == 5
    assert args.max_batch_size == 1
    assert args.cuda_graph_config is not None
    assert args.cuda_graph_config.max_batch_size == 1
    assert args.cuda_graph_config.batch_sizes == [1]
    assert args.transforms["compile_model"]["cuda_graph_batch_sizes"] == [1]


def test_deepseek_v4_flash_uses_attention_and_moe_ir_sharding_for_full_model_config():
    """Full DeepSeek V4 runs shard attention and packed MoE weights before checkpoint load."""
    config_dir = _AD_EXAMPLES_DIR / "model_registry" / "configs"
    args = LlmArgs(
        model=_DUMMY_MODEL,
        yaml_extra=[
            str(config_dir / "dashboard_default.yaml"),
            str(config_dir / "world_size_8.yaml"),
            str(config_dir / "deepseek_v4_flash.yaml"),
            str(config_dir / "functional_single_request_cuda_graph.yaml"),
        ],
    )

    assert args.world_size == 8
    assert args.transforms["detect_sharding"]["enabled"] is False
    assert args.transforms["sharding_transform_executor"]["enabled"] is False
    assert args.transforms["apply_sharding_hints"]["enabled"] is True
    assert args.transforms["apply_sharding_hints"]["dist_mapping"] == {
        "tp": 8,
        "moe_ep": 8,
        "moe_tp": 1,
        "moe_cluster": 1,
    }
    assert args.transforms["apply_sharding_hints"]["enable_attention_dp"] is False
    assert args.transforms["apply_sharding_hints"]["shard_layers"] == ["mla", "moe"]
