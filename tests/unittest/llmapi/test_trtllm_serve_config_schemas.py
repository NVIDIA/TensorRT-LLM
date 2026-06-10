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
"""Tests for generated trtllm-serve YAML JSON Schemas."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from jsonschema.exceptions import ValidationError

from .yaml_validation_harness import collect_yaml_files, load_yaml_dict

REPO_ROOT = Path(__file__).parents[3]
CONFIG_ROOT = REPO_ROOT / "examples" / "configs"
CURATED_DIR = CONFIG_ROOT / "curated"
DATABASE_DIR = CONFIG_ROOT / "database"

ALL_SERVE_CONFIGS = sorted(
    collect_yaml_files(CURATED_DIR, "**/*.yaml", exclude_names={"lookup.yaml"})
    + collect_yaml_files(DATABASE_DIR, "**/*.yaml", exclude_names={"lookup.yaml"})
)

# Disagg orchestrator example YAMLs. etcd_config.yaml is a metadata-server config
# (different YAML schema, passed via --metadata_server_config_file), so exclude it.
DISAGG_EXAMPLES_DIR = REPO_ROOT / "examples" / "disaggregated"
ALL_DISAGG_CONFIGS = sorted(
    collect_yaml_files(DISAGG_EXAMPLES_DIR, "disagg_config.yaml")
    + collect_yaml_files(DISAGG_EXAMPLES_DIR, "slurm/simple_example/disagg_config.yaml")
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCHEMA_GENERATOR = _load_module(
    "generate_trtllm_serve_schemas",
    REPO_ROOT / "scripts" / "generate_trtllm_serve_schemas.py",
)
SERVE_CONFIG_SCHEMA_FILENAME = SCHEMA_GENERATOR.SERVE_CONFIG_SCHEMA_FILENAME
AUTODEPLOY_CONFIG_SCHEMA_FILENAME = SCHEMA_GENERATOR.AUTODEPLOY_CONFIG_SCHEMA_FILENAME
DISAGG_CONFIG_SCHEMA_FILENAME = SCHEMA_GENERATOR.DISAGG_CONFIG_SCHEMA_FILENAME
VISUAL_GEN_CONFIG_SCHEMA_FILENAME = SCHEMA_GENERATOR.VISUAL_GEN_CONFIG_SCHEMA_FILENAME


def _validator(schema: dict) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture(scope="module")
def serve_config_validator() -> jsonschema.Draft202012Validator:
    return _validator(SCHEMA_GENERATOR.generate_serve_config_schema())


@pytest.fixture(scope="module")
def autodeploy_config_validator() -> jsonschema.Draft202012Validator:
    return _validator(SCHEMA_GENERATOR.generate_autodeploy_config_schema())


@pytest.fixture(scope="module")
def disagg_config_validator() -> jsonschema.Draft202012Validator:
    return _validator(SCHEMA_GENERATOR.generate_disagg_config_schema())


@pytest.fixture(scope="module")
def visual_gen_config_validator() -> jsonschema.Draft202012Validator:
    return _validator(SCHEMA_GENERATOR.generate_visual_gen_config_schema())


def test_serve_config_schema_accepts_minimal_yaml(serve_config_validator):
    serve_config_validator.validate({"max_batch_size": 8})


def test_serve_config_schema_accepts_serve_loader_aliases(serve_config_validator):
    serve_config_validator.validate(
        {
            "backend": "pytorch",
            "hf_revision": "main",
        }
    )


def test_serve_config_schema_rejects_typo(serve_config_validator):
    with pytest.raises(ValidationError):
        serve_config_validator.validate({"max_batch_szie": 8})


def test_serve_config_schema_accepts_disagg_cluster_for_worker_registration(
    serve_config_validator,
):
    # trtllm-serve serve pops disagg_cluster from the YAML before constructing
    # the LLM (serve.py), so the regular config must accept it.
    serve_config_validator.validate(
        {
            "disagg_cluster": {
                "cluster_uri": "etcd://localhost:2379",
                "cluster_name": "team-cluster",
                "heartbeat_interval_sec": 5,
            }
        }
    )


def test_serve_config_schema_rejects_disagg_cluster_missing_uri(serve_config_validator):
    with pytest.raises(ValidationError):
        serve_config_validator.validate({"disagg_cluster": {"cluster_name": "no-uri"}})


def test_autodeploy_config_schema_accepts_disagg_cluster(autodeploy_config_validator):
    # Same disagg_cluster path applies to autodeploy workers.
    autodeploy_config_validator.validate(
        {
            "backend": "_autodeploy",
            "disagg_cluster": {"cluster_uri": "etcd://localhost:2379"},
        }
    )


def test_serve_config_schema_accepts_unquoted_env_overrides_scalars(serve_config_validator):
    # env_overrides values are coerced to strings at runtime; the static schema
    # must accept unquoted scalars so YAML configs like `TRTLLM_ENABLE_PDL: 1`
    # don't trip the IDE.
    serve_config_validator.validate(
        {
            "env_overrides": {
                "TRTLLM_ENABLE_PDL": 1,
                "NCCL_GRAPH_REGISTER": 0,
                "SOME_FLAG": True,
            }
        }
    )


@pytest.mark.parametrize(
    "config_path",
    ALL_SERVE_CONFIGS,
    ids=lambda path: str(path.relative_to(CONFIG_ROOT)),
)
def test_existing_serve_configs_validate_against_schema(serve_config_validator, config_path: Path):
    serve_config_validator.validate(load_yaml_dict(config_path))


def test_autodeploy_config_schema_accepts_minimal_yaml(autodeploy_config_validator):
    autodeploy_config_validator.validate(
        {
            "world_size": 1,
            "compile_backend": "torch-opt",
        }
    )


def test_autodeploy_config_schema_accepts_backend_marker_and_aliases(autodeploy_config_validator):
    autodeploy_config_validator.validate(
        {
            "backend": "_autodeploy",
            "hf_revision": "main",
        }
    )


def test_autodeploy_config_schema_rejects_wrong_backend(autodeploy_config_validator):
    with pytest.raises(ValidationError):
        autodeploy_config_validator.validate({"backend": "pytorch"})


def test_autodeploy_config_schema_rejects_typo(autodeploy_config_validator):
    with pytest.raises(ValidationError):
        autodeploy_config_validator.validate({"compile_backennd": "torch-opt"})


def test_disagg_config_schema_accepts_minimal_yaml(disagg_config_validator):
    disagg_config_validator.validate(
        {
            "context_servers": {"urls": ["ctx:8001"]},
            "generation_servers": {"urls": ["gen:8002"]},
        }
    )


def test_disagg_config_schema_accepts_inherited_torch_args(disagg_config_validator):
    # Top-level TorchLlmArgs-style fields should validate; they're inherited
    # into each server block by extract_disagg_cfg at runtime.
    disagg_config_validator.validate(
        {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "backend": "pytorch",
            "disable_overlap_scheduler": True,
            "free_gpu_memory_fraction": 0.25,
            "context_servers": {
                "num_instances": 1,
                "tensor_parallel_size": 1,
                "kv_cache_config": {"free_gpu_memory_fraction": 0.2},
                "urls": ["localhost:8001"],
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"],
            },
        }
    )


def test_disagg_config_schema_accepts_gen_only_with_zero_instances(disagg_config_validator):
    # gen-only configs set context_servers.num_instances = 0.
    disagg_config_validator.validate(
        {
            "context_servers": {"num_instances": 0},
            "generation_servers": {"urls": ["localhost:8002"]},
        }
    )


def test_disagg_config_schema_rejects_top_level_typo(disagg_config_validator):
    with pytest.raises(ValidationError):
        disagg_config_validator.validate({"contextservers": {}})


def test_disagg_config_schema_rejects_nested_typo(disagg_config_validator):
    with pytest.raises(ValidationError):
        disagg_config_validator.validate({"context_servers": {"num_instnces": 1}})


def test_disagg_config_schema_rejects_bad_schedule_style(disagg_config_validator):
    with pytest.raises(ValidationError):
        disagg_config_validator.validate({"schedule_style": "round_robin"})


def test_disagg_config_schema_accepts_unquoted_env_overrides_in_server_block(
    disagg_config_validator,
):
    # The same runtime str-coercion that applies to a worker config applies to
    # env_overrides inside context_servers / generation_servers blocks.
    disagg_config_validator.validate(
        {
            "env_overrides": {"TRTLLM_ENABLE_PDL": 1},
            "context_servers": {
                "env_overrides": {"TRTLLM_ENABLE_PDL": 1, "SOME_FLAG": True},
                "urls": ["localhost:8001"],
            },
            "generation_servers": {"urls": ["localhost:8002"]},
        }
    )


def test_disagg_config_schema_enforces_node_id_bounds(disagg_config_validator):
    disagg_config_validator.validate({"node_id": 0})
    disagg_config_validator.validate({"node_id": 1023})
    disagg_config_validator.validate({"node_id": None})
    with pytest.raises(ValidationError):
        disagg_config_validator.validate({"node_id": 1024})
    with pytest.raises(ValidationError):
        disagg_config_validator.validate({"node_id": -1})


@pytest.mark.parametrize(
    "config_path",
    ALL_DISAGG_CONFIGS,
    ids=lambda path: str(path.relative_to(REPO_ROOT / "examples")),
)
def test_existing_disagg_configs_validate_against_schema(
    disagg_config_validator, config_path: Path
):
    disagg_config_validator.validate(load_yaml_dict(config_path))


def test_visual_gen_config_schema_accepts_representative_yaml(visual_gen_config_validator):
    visual_gen_config_validator.validate(
        {
            "parallel": {
                "dit_cfg_size": 1,
                "dit_ulysses_size": 1,
            },
            "attention": {
                "backend": "TRTLLM",
            },
        }
    )


def test_visual_gen_config_schema_rejects_nested_typo(visual_gen_config_validator):
    with pytest.raises(ValidationError):
        visual_gen_config_validator.validate(
            {
                "parallel": {
                    "dit_cfg_szie": 1,
                }
            }
        )


def _load_schema_assets_extension():
    extension_path = REPO_ROOT / "docs" / "source" / "_ext" / "trtllm_schema_assets.py"
    return _load_module("trtllm_schema_assets", extension_path)


def test_docs_extension_writes_schema_assets(tmp_path):
    extension = _load_schema_assets_extension()
    app = SimpleNamespace(builder=SimpleNamespace(name="html", outdir=str(tmp_path)))

    extension._write_schema_assets(app)

    schema_dir = tmp_path / "_static" / "schemas"
    assert (schema_dir / SERVE_CONFIG_SCHEMA_FILENAME).is_file()
    assert (schema_dir / AUTODEPLOY_CONFIG_SCHEMA_FILENAME).is_file()
    assert (schema_dir / DISAGG_CONFIG_SCHEMA_FILENAME).is_file()
    assert (schema_dir / VISUAL_GEN_CONFIG_SCHEMA_FILENAME).is_file()
