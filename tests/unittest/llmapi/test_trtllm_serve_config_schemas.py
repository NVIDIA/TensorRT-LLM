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

import ast
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
    from tensorrt_llm.version import __version__

    paths = generator.write_schemas(tmp_path_factory.mktemp("schemas"))
    result = {path.name: json.loads(path.read_text()) for path in paths}
    for name, schema in result.items():
        Draft202012Validator.check_schema(schema)
        assert schema["$id"].endswith(f"/{__version__}/_static/schemas/{name}")
        assert schema["title"].startswith(f"TensorRT-LLM {__version__} ")
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
        (generator.SERVE_SCHEMA, {"backend": "pytorch"}, True),
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
        (generator.DISAGG_SCHEMA, {"backend": "pytorch"}, True),
        (generator.DISAGG_SCHEMA, {"backend": "_autodeploy"}, False),
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


def _assert_serving_fields_in_schema(source: str, function_name: str, properties: dict) -> None:
    """Check literal top-level config accesses, without tracing aliases or dynamic keys."""
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    config_names = {"llm_args", "llm_args_dict", "llm_args_extra_dict", "raw_llm_args_extra_dict"}
    helpers = {"_pop_bool_config_option", "_pop_optional_str_config_option"}
    fields = set()
    for node in ast.walk(function):
        mapping = key = None
        if isinstance(node, ast.Call) and node.args:
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "get",
                "pop",
                "setdefault",
            }:
                mapping, key = node.func.value, node.args[0]
            elif isinstance(node.func, ast.Name) and node.func.id in helpers and len(node.args) > 1:
                mapping, key = node.args[:2]
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            mapping, key = node.value, node.slice
        if (
            isinstance(mapping, ast.Name)
            and mapping.id in config_names
            and isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        ):
            fields.add(key.value)
    assert fields, f"No recognized config accesses in {function_name}; update the coverage guard."
    missing = fields - properties.keys()
    assert not missing, (
        f"Serving YAML fields missing from schema in {function_name}: {sorted(missing)}. "
        "Update generate_serve_schema() and regenerate the schemas."
    )


@pytest.mark.parametrize(
    ("filename", "function_name"),
    [
        ("tensorrt_llm/commands/serve.py", "serve"),
        ("tensorrt_llm/llmapi/llm_args.py", "update_llm_args_with_extra_dict"),
    ],
)
def test_serving_yaml_fields_have_schema(
    schemas: dict[str, dict], filename: str, function_name: str
) -> None:
    _assert_serving_fields_in_schema(
        (_REPO_ROOT / filename).read_text(encoding="utf-8"),
        function_name,
        schemas[generator.SERVE_SCHEMA]["properties"],
    )


@pytest.mark.parametrize(
    "access",
    [
        'raw_llm_args_extra_dict.get("new_serving_option")',
        'llm_args_extra_dict.pop("new_serving_option", None)',
        'llm_args_dict.setdefault("new_serving_option", False)',
        'llm_args["new_serving_option"]',
        '_pop_bool_config_option(llm_args_extra_dict, "new_serving_option")',
        '_pop_optional_str_config_option(llm_args_extra_dict, "new_serving_option")',
    ],
)
def test_serving_field_guard_detects_new_fields(access: str) -> None:
    source = f"def serve():\n    def _serve_llm():\n        {access}\n"
    with pytest.raises(AssertionError, match="Serving YAML fields missing.*new_serving_option"):
        _assert_serving_fields_in_schema(source, "serve", {})
    _assert_serving_fields_in_schema(source, "serve", {"new_serving_option": {}})


def test_serving_field_guard_ignores_unrelated_and_nested_keys() -> None:
    source = """
def serve():
    request.get("not_a_config_field")
    _pop_bool_config_option(other_mapping, "not_a_config_field")
    llm_args.get(dynamic_key)
    llm_args["internal_only_field"] = True
    llm_args["kv_cache_config"]["nested_field"]
"""
    _assert_serving_fields_in_schema(source, "serve", {"kv_cache_config": {}})


def test_serving_field_guard_rejects_empty_scan() -> None:
    with pytest.raises(AssertionError, match="No recognized config accesses"):
        _assert_serving_fields_in_schema("def serve(): pass", "serve", {})


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
        lambda _path: {"write_schemas": lambda output_dir: calls.append(output_dir)},
    )
    app = SimpleNamespace(
        builder=SimpleNamespace(format=builder_format),
        outdir=tmp_path,
    )
    extension._write_schema_assets(app, RuntimeError("failed") if failed else None)
    if builder_format == "html" and not failed:
        assert calls == [tmp_path / "_static/schemas"]
    else:
        assert not calls


def test_generation_is_deterministic(tmp_path: Path, schemas: dict[str, dict]) -> None:
    paths = generator.write_schemas(tmp_path)
    assert {path.name: json.loads(path.read_text()) for path in paths} == schemas


def test_checked_in_schemas_are_current() -> None:
    generator.write_schemas(_REPO_ROOT / "tensorrt_llm/schemas", check=True)


@pytest.mark.parametrize("missing", [False, True])
def test_check_detects_stale_files_without_writing(tmp_path: Path, missing: bool) -> None:
    paths = generator.write_schemas(tmp_path)
    generator.write_schemas(tmp_path, check=True)
    if missing:
        paths[0].unlink()
    else:
        paths[0].write_text("{}\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="Missing or stale configuration schemas"):
        generator.write_schemas(tmp_path, check=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
