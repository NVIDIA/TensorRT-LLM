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
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema

SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_BASE_URL = "https://nvidia.github.io/TensorRT-LLM/latest/_static/schemas"

SERVE_CONFIG_SCHEMA_FILENAME = "trtllm-serve-config.schema.json"
VISUAL_GEN_CONFIG_SCHEMA_FILENAME = (
    "trtllm-serve-visual-gen-config.schema.json"
)


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
    return schema


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

    revision_schema = _copy_property(schema, "revision")
    if not revision_schema:
        revision_schema = {"type": ["string", "null"]}
    revision_schema["description"] = (
        "Alias for revision accepted by trtllm-serve --config YAML files."
    )
    properties["hf_revision"] = revision_schema


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
