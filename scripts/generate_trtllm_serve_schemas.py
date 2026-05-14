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
"""Generate JSON Schemas for trtllm-serve YAML config files."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# Each release's docs are also published at /<version>/, so pinning the schema
# $id to that path keeps a YAML config's `# yaml-language-server: $schema=...`
# directive resolving to the schema generated against the same TensorRT-LLM
# version, even after /latest/ rolls forward.
_DOCS_BASE_URL = "https://nvidia.github.io/TensorRT-LLM"


def _read_trtllm_version() -> str:
    # Read tensorrt_llm/version.py directly to avoid importing the package
    # (which would require the compiled bindings to be present).
    version_path = Path(__file__).resolve().parents[1] / "tensorrt_llm" / "version.py"
    try:
        spec = importlib.util.spec_from_file_location(
            "_trtllm_version", version_path)
        if spec is None or spec.loader is None:
            return "latest"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "__version__", "latest")
    except (FileNotFoundError, OSError):
        return "latest"


SCHEMA_BASE_URL = f"{_DOCS_BASE_URL}/{_read_trtllm_version()}/_static/schemas"

SERVE_CONFIG_SCHEMA_FILENAME = "trtllm-serve-config.schema.json"
AUTODEPLOY_CONFIG_SCHEMA_FILENAME = (
    "trtllm-serve-autodeploy-config.schema.json"
)
VISUAL_GEN_CONFIG_SCHEMA_FILENAME = (
    "trtllm-serve-visual-gen-config.schema.json"
)

_VALID_JSON_SCHEMA_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "null", "array", "object"}
)

# env_overrides values are coerced to strings at runtime
# (TorchLlmArgs.coerce_env_overrides_to_str); permit the same scalar shapes
# in the static schema so unquoted YAML scalars validate.
_ENV_OVERRIDES_VALUE_SCHEMA = {
    "type": ["string", "integer", "number", "boolean"]
}


class TRTLLMServeSchemaGenerator(GenerateJsonSchema):
    """Allow runtime-only Python types to remain permissive in IDE schemas."""

    def handle_invalid_for_json_schema(self, schema: Any,
                                       error_info: str) -> dict[str, Any]:
        return {
            "description": (
                "Accepted by TensorRT-LLM runtime validation; this field cannot "
                f"be fully represented in JSON Schema ({error_info})."
            )
        }


def _schema_id(filename: str) -> str:
    return f"{SCHEMA_BASE_URL}/{filename}"


def _ensure_repo_root_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _copy_property(schema: dict[str, Any], field_name: str) -> dict[str, Any]:
    properties = schema.setdefault("properties", {})
    field_schema = properties.get(field_name, {})
    return copy.deepcopy(field_schema)


def _strip_invalid_json_schema_types(node: Any) -> None:
    """Drop ``type`` keys whose value is not a JSON Schema primitive.

    Several fields in TorchLlmArgs set ``json_schema_extra={"type": "..."}``
    to a Python type-name string (e.g. ``"Union[MoeLoadBalancerConfig, dict,
    str]"``) for autodoc rendering. Those values flow through unchanged into
    the generated schema and make it fail ``Draft202012Validator.check_schema``.
    """
    if isinstance(node, dict):
        type_value = node.get("type")
        if isinstance(type_value, str) and type_value not in _VALID_JSON_SCHEMA_TYPES:
            node.pop("type")
        for value in node.values():
            _strip_invalid_json_schema_types(value)
    elif isinstance(node, list):
        for value in node:
            _strip_invalid_json_schema_types(value)


def _widen_env_overrides(schema: dict[str, Any]) -> None:
    """Mirror runtime str-coercion for ``env_overrides``.

    ``TorchLlmArgs.env_overrides`` is typed as ``Dict[str, str]``, but the
    ``coerce_env_overrides_to_str`` validator stringifies any scalar value.
    Real configs (e.g. ``TRTLLM_ENABLE_PDL: 1``) therefore validate at
    runtime but would otherwise trip the static schema; widen the value
    type to match what the runtime accepts.
    """
    env = schema.get("properties", {}).get("env_overrides")
    if not isinstance(env, dict):
        return

    def _widen(node: dict[str, Any]) -> None:
        if node.get("type") == "object":
            node["additionalProperties"] = copy.deepcopy(_ENV_OVERRIDES_VALUE_SCHEMA)

    _widen(env)
    for branch in env.get("anyOf", []):
        if isinstance(branch, dict):
            _widen(branch)


def _add_schema_metadata(
    schema: dict[str, Any],
    *,
    schema_id: str,
    title: str,
    description: str,
) -> dict[str, Any]:
    schema.pop("required", None)
    schema["$schema"] = SCHEMA_DRAFT
    schema["$id"] = schema_id
    schema["title"] = title
    schema["description"] = description
    schema.setdefault("type", "object")
    schema.setdefault("additionalProperties", False)
    _strip_invalid_json_schema_types(schema)
    _widen_env_overrides(schema)
    return schema


def _hf_revision_alias(schema: dict[str, Any]) -> dict[str, Any]:
    """Build an ``hf_revision`` property schema cloned from ``revision``.

    ``trtllm-serve`` translates ``hf_revision`` -> ``revision`` in
    ``update_llm_args_with_extra_dict`` so both spellings need to validate.
    """
    revision_schema = _copy_property(schema, "revision")
    if not revision_schema:
        revision_schema = {"type": ["string", "null"]}
    revision_schema["description"] = (
        "Alias for revision accepted by trtllm-serve --config YAML files."
    )
    return revision_schema


def _add_serve_config_aliases(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    properties["backend"] = {
        "type": "string",
        "const": "pytorch",
        "description": (
            "Optional backend marker accepted by trtllm-serve YAML configs. "
            "This schema covers the default PyTorch serve config surface."
        ),
    }
    properties["hf_revision"] = _hf_revision_alias(schema)


def _add_autodeploy_config_aliases(schema: dict[str, Any]) -> None:
    # AutoDeploy's LlmArgs already exposes `backend` as Literal["_autodeploy"],
    # so the const marker is emitted by Pydantic; only the hf_revision alias
    # needs to be added here.
    properties = schema.setdefault("properties", {})
    properties["hf_revision"] = _hf_revision_alias(schema)


def _model_schema(
    model_cls: type[BaseModel],
    *,
    schema_id: str,
    title: str,
    description: str,
) -> dict[str, Any]:
    schema = model_cls.model_json_schema(
        by_alias=True,
        mode="validation",
        schema_generator=TRTLLMServeSchemaGenerator,
    )
    return _add_schema_metadata(
        schema,
        schema_id=schema_id,
        title=title,
        description=description,
    )


def generate_serve_config_schema() -> dict[str, Any]:
    _ensure_repo_root_on_syspath()
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    schema = _model_schema(
        TorchLlmArgs,
        schema_id=_schema_id(SERVE_CONFIG_SCHEMA_FILENAME),
        title="TensorRT-LLM trtllm-serve Config",
        description=(
            "YAML fragment accepted by trtllm-serve serve <model> --config for "
            "the default PyTorch backend. The command line supplies model and "
            "other defaults; runtime validation remains authoritative."
        ),
    )
    _add_serve_config_aliases(schema)
    return schema


def generate_autodeploy_config_schema() -> dict[str, Any]:
    _ensure_repo_root_on_syspath()
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs as AutoDeployLlmArgs

    schema = _model_schema(
        AutoDeployLlmArgs,
        schema_id=_schema_id(AUTODEPLOY_CONFIG_SCHEMA_FILENAME),
        title="TensorRT-LLM trtllm-serve AutoDeploy Config",
        description=(
            "YAML fragment accepted by trtllm-serve serve <model> --config "
            "--backend _autodeploy. The command line supplies model and other "
            "defaults; runtime validation remains authoritative."
        ),
    )
    _add_autodeploy_config_aliases(schema)
    return schema


def generate_visual_gen_config_schema() -> dict[str, Any]:
    _ensure_repo_root_on_syspath()
    from tensorrt_llm._torch.visual_gen.config import VisualGenArgs

    return _model_schema(
        VisualGenArgs,
        schema_id=_schema_id(VISUAL_GEN_CONFIG_SCHEMA_FILENAME),
        title="TensorRT-LLM trtllm-serve VisualGen Config",
        description=(
            "YAML fragment accepted by trtllm-serve --extra_visual_gen_options "
            "for visual generation models. Runtime validation remains "
            "authoritative."
        ),
    )


def write_schemas(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    schemas = {
        SERVE_CONFIG_SCHEMA_FILENAME: generate_serve_config_schema(),
        AUTODEPLOY_CONFIG_SCHEMA_FILENAME: generate_autodeploy_config_schema(),
        VISUAL_GEN_CONFIG_SCHEMA_FILENAME: generate_visual_gen_config_schema(),
    }

    written_paths = []
    for filename, schema in schemas.items():
        output_path = output_dir / filename
        output_path.write_text(
            json.dumps(schema, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        written_paths.append(output_path)

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSON Schemas for trtllm-serve YAML configs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where schema JSON files should be written.",
    )
    args = parser.parse_args()

    for path in write_schemas(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
