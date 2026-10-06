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
"""CPU-only schema generation, validation, and documentation asset checks."""

import copy
import importlib.util
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

pytestmark = pytest.mark.cpu_only
_REPO_ROOT = Path(__file__).resolve().parents[3]
generator = SimpleNamespace(
    **runpy.run_path(str(_REPO_ROOT / "scripts/generate_trtllm_serve_schemas.py"))
)


@pytest.fixture(scope="module")
def schemas(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    paths = generator.write_schemas(tmp_path_factory.mktemp("schemas"), version="test-release")
    result = {path.name: json.loads(path.read_text()) for path in paths}
    for name, schema in result.items():
        Draft202012Validator.check_schema(schema)
        assert schema["$id"].endswith(f"/test-release/_static/schemas/{name}")
    return result


@pytest.mark.parametrize(
    ("filename", "config", "valid"),
    [
        (generator.SERVE_SCHEMA, {}, True),
        (generator.SERVE_SCHEMA, {"max_batch_size": 8}, True),
        (generator.SERVE_SCHEMA, {"max_batch_szie": 8}, False),
        (generator.SERVE_SCHEMA, {"max_batch_size": "eight"}, False),
        (generator.SERVE_SCHEMA, {"moe_config": {"backend": "CUTLASS"}}, True),
        (generator.SERVE_SCHEMA, {"moe_config": {"backend_typo": "CUTLASS"}}, False),
        (generator.SERVE_SCHEMA, {"moe_config": {"backend": "not-a-backend"}}, False),
        (generator.SERVE_SCHEMA, {"backend": "_autodeploy"}, False),
        (
            generator.SERVE_SCHEMA,
            {"hf_revision": "main", "allow_request_chat_template": True},
            True,
        ),
        (generator.SERVE_SCHEMA, {"internal_request_auth_key": "example-key"}, True),
        (generator.SERVE_SCHEMA, {"internal_request_auth_key": ""}, False),
        (
            generator.SERVE_SCHEMA,
            {"disagg_cluster": {"cluster_uri": "etcd://localhost:2379"}},
            True,
        ),
        (generator.SERVE_SCHEMA, {"disagg_cluster": {"cluster_name": "missing-uri"}}, False),
        (
            generator.SERVE_SCHEMA,
            {"disagg_cluster": {"cluster_uri": "etcd://localhost:2379", "cluster_nmae": "bad"}},
            False,
        ),
        (generator.SERVE_SCHEMA, {"env_overrides": {"FLAG": 1, "OTHER": True}}, True),
        (generator.SERVE_SCHEMA, {"telemetry_config": {"disabled": True}}, True),
        (generator.SERVE_SCHEMA, {"multimodal_config": None}, True),
        (generator.SERVE_SCHEMA, {"num_serve_frontends": 0}, False),
        (generator.SERVE_SCHEMA, {"cuda_graph_config": {}}, True),
        (generator.SERVE_SCHEMA, {"cuda_graph_config": {"batch_sizes": [1, 2, 4]}}, True),
        (generator.SERVE_SCHEMA, {"checkpoint_loader": {}}, False),
        (generator.SERVE_SCHEMA, {"tokenizer": 42}, False),
        (generator.DISAGG_SCHEMA, {"context_servers": {"urls": ["ctx:8001"]}}, True),
        (generator.DISAGG_SCHEMA, {"context_servers": {"num_instnces": 1}}, False),
        (generator.DISAGG_SCHEMA, {"context_servers": {"env_overrides": {"FLAG": 1}}}, True),
        (
            generator.DISAGG_SCHEMA,
            {"context_servers": {"moe_config": {"backend_typo": "CUTLASS"}}},
            False,
        ),
        (generator.DISAGG_SCHEMA, {"schedule_style": "invalid"}, False),
        (generator.DISAGG_SCHEMA, {"num_workers": 2, "server_keep_alive_timeout": 20}, True),
        (generator.VISUAL_GEN_SCHEMA, {"parallel_config": {"cfg_size": 1}}, True),
        (generator.VISUAL_GEN_SCHEMA, {"parallel_config": {"cfg_szie": 1}}, False),
        (generator.VISUAL_GEN_SCHEMA, {"parallel_config": {"cfg_size": 0}}, False),
        (generator.VISUAL_GEN_SCHEMA, {"attention_config": {"backend": "not-a-backend"}}, False),
    ],
)
def test_config_validation(
    schemas: dict[str, dict], filename: str, config: dict, valid: bool
) -> None:
    errors = list(Draft202012Validator(schemas[filename]).iter_errors(config))
    assert (not errors) == valid, [error.message for error in errors]


def test_disagg_node_id_matches_runtime(schemas: dict[str, dict]) -> None:
    from tensorrt_llm.llmapi.disagg_utils import DISAGG_NODE_ID_BITS

    validator = Draft202012Validator(schemas[generator.DISAGG_SCHEMA])
    maximum = (1 << DISAGG_NODE_ID_BITS) - 1
    for value in (None, 0, maximum):
        validator.validate({"node_id": value})
    for value in (-1, maximum + 1):
        assert not validator.is_valid({"node_id": value})


def test_runtime_only_union_does_not_disable_validation() -> None:
    class RuntimeObject:
        pass

    class Config(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        tokenizer: str | RuntimeObject | None = None
        count: int = Field(1, json_schema_extra={"type": "Python int"})

    original = copy.deepcopy(Config.model_fields["count"].json_schema_extra)
    schema = generator._schema_for(Config)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate({"tokenizer": "name", "count": 2})
    for value in (1, [], {}, True):
        assert not validator.is_valid({"tokenizer": value})
    assert not validator.is_valid({"count": "two"})
    assert Config.model_fields["count"].json_schema_extra == original


def test_nested_validation_alias(schemas: dict[str, dict]) -> None:
    config = {"speculative_config": {"decoding_type": "Eagle3", "speculative_model_dir": "model"}}
    Draft202012Validator(schemas[generator.SERVE_SCHEMA]).validate(config)


@pytest.mark.parametrize(
    ("filename", "directory", "pattern"),
    [
        (generator.SERVE_SCHEMA, "examples/configs/curated", "*.yaml"),
        (generator.SERVE_SCHEMA, "examples/configs/database", "**/*.yaml"),
        (generator.DISAGG_SCHEMA, "examples/disaggregated", "**/disagg_config.yaml"),
    ],
)
def test_existing_configs(
    schemas: dict[str, dict], filename: str, directory: str, pattern: str
) -> None:
    validator = Draft202012Validator(schemas[filename])
    # curated/eplb contains load-balancer sidecars, not standalone serving configs.
    paths = sorted((_REPO_ROOT / directory).glob(pattern))
    assert paths
    failures = []
    for path in paths:
        if path.name == "lookup.yaml":
            continue
        config = yaml.safe_load(path.read_text()) or {}
        errors = list(validator.iter_errors(config))
        if errors:
            failures.append(
                (str(path.relative_to(_REPO_ROOT)), [error.message for error in errors])
            )
    assert not failures, failures


@pytest.mark.parametrize(
    ("builder_format", "failed"), [("html", False), ("html", True), ("latex", False)]
)
def test_docs_publishes_only_after_successful_html_build(
    tmp_path: Path,
    schemas: dict[str, dict],
    monkeypatch: pytest.MonkeyPatch,
    builder_format: str,
    failed: bool,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "trtllm_schema_assets", _REPO_ROOT / "docs/source/_ext/trtllm_schema_assets.py"
    )
    assert spec is not None and spec.loader is not None
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    calls = []
    monkeypatch.setattr(
        extension.runpy,
        "run_path",
        lambda _path: {"write_schemas": lambda *args, **kwargs: calls.append((args, kwargs))},
    )
    app = SimpleNamespace(
        builder=SimpleNamespace(format=builder_format),
        outdir=tmp_path,
        config=SimpleNamespace(version="test-release"),
    )
    extension._write_schema_assets(app, RuntimeError("failed") if failed else None)
    if builder_format == "html" and not failed:
        assert calls == [((tmp_path / "_static/schemas",), {"version": "test-release"})]
    else:
        assert not calls


def test_generation_is_deterministic(tmp_path: Path, schemas: dict[str, dict]) -> None:
    paths = generator.write_schemas(tmp_path, version="test-release")
    assert {path.name: json.loads(path.read_text()) for path in paths} == schemas


def test_checked_in_schemas_are_current() -> None:
    generator.write_schemas(_REPO_ROOT / "tensorrt_llm/schemas", check=True)


@pytest.mark.parametrize("missing", [False, True])
def test_check_detects_stale_files_without_writing(tmp_path: Path, missing: bool) -> None:
    paths = generator.write_schemas(tmp_path, version="test-release")
    generator.write_schemas(tmp_path, version="test-release", check=True)
    if missing:
        paths[0].unlink()
    else:
        paths[0].write_text("{}\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="Missing or stale configuration schemas"):
        generator.write_schemas(tmp_path, version="test-release", check=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
