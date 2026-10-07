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
"""Export editor schemas from the configuration types, without constructing an LLM."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import PydanticOmit, core_schema

_DOCS_URL = "https://nvidia.github.io/TensorRT-LLM"
_JSON_TYPES = {"string", "number", "integer", "boolean", "null", "array", "object"}
SERVE_SCHEMA = "trtllm-serve-config.schema.json"
DISAGG_SCHEMA = "trtllm-serve-disagg-config.schema.json"
VISUAL_GEN_SCHEMA = "trtllm-serve-visual-gen-config.schema.json"


class ServeSchemaGenerator(GenerateJsonSchema):
    """Keep YAML-representable union branches and ignore autodoc-only type labels."""

    def handle_invalid_for_json_schema(
        self, schema: core_schema.CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        # An unconstrained fallback would make e.g. str | Tokenizer accept any YAML value.
        raise PydanticOmit

    def generate_inner(self, schema: core_schema.CoreSchema) -> JsonSchemaValue:
        metadata = schema.get("metadata", {})
        extra = metadata.get("pydantic_js_extra")
        if isinstance(extra, dict) and isinstance(extra.get("type"), str):
            if extra["type"] not in _JSON_TYPES:
                schema = {
                    **schema,
                    "metadata": {
                        **metadata,
                        "pydantic_js_extra": {
                            key: value for key, value in extra.items() if key != "type"
                        },
                    },
                }
        return super().generate_inner(schema)

    def model_fields_schema(self, schema: core_schema.ModelFieldsSchema) -> JsonSchemaValue:
        result = super().model_fields_schema(schema)
        properties = result["properties"]
        for field in schema["fields"].values():
            aliases = field.get("validation_alias")
            if isinstance(aliases, list):
                names = [
                    alias[0]
                    for alias in aliases
                    if isinstance(alias, list) and len(alias) == 1 and isinstance(alias[0], str)
                ]
                if names and names[0] in properties:
                    for name in names[1:]:
                        properties[name] = copy.deepcopy(properties[names[0]])
        return result

    def dataclass_schema(self, schema: core_schema.DataclassSchema) -> JsonSchemaValue:
        result = super().dataclass_schema(schema)
        # These configs are constructed with **yaml_dict, which rejects unknown arguments.
        self.resolve_ref_schema(result)["additionalProperties"] = False
        return result


def _schema_for(annotation: Any, parent: dict | None = None) -> dict:
    schema = TypeAdapter(annotation).json_schema(schema_generator=ServeSchemaGenerator)
    if parent is not None:
        definitions = parent.setdefault("$defs", {})
        for name, definition in schema.pop("$defs", {}).items():
            if name in definitions and definitions[name] != definition:
                raise ValueError(f"Conflicting schema definition: {name}")
            definitions[name] = definition
    return schema


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def generate_serve_schema() -> dict:
    """Describe the YAML fragment merged with ordinary serving CLI arguments."""
    from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig
    from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig, TorchLlmArgs

    schema = _schema_for(TorchLlmArgs)
    # MODEL is supplied on the command line; nested required fields still apply.
    schema["required"] = [name for name in schema.get("required", []) if name != "model"]
    properties = schema["properties"]
    properties.update(
        hf_revision=copy.deepcopy(properties["revision"]),
        allow_request_chat_template={"type": "boolean", "default": False},
        internal_request_auth_key=_nullable({"type": "string", "minLength": 1}),
        disagg_cluster=_nullable(_schema_for(DisaggClusterConfig, schema)),
    )
    # Serving and field validators normalize these values before model validation.
    properties["env_overrides"] = _nullable({"type": "object", "additionalProperties": True})
    properties["multimodal_config"] = _nullable(properties["multimodal_config"])
    properties["telemetry_config"] = _nullable(properties["telemetry_config"])
    # The runtime infers mode when it is omitted; both variants can match a partial config.
    for branch in properties["cuda_graph_config"]["anyOf"]:
        if "oneOf" in branch:
            branch["anyOf"] = branch.pop("oneOf")
            branch.pop("discriminator", None)
    schema["$defs"]["MoeConfig"]["properties"]["load_balancer"] = {
        "anyOf": [_schema_for(MoeLoadBalancerConfig, schema), {"type": ["string", "null"]}]
    }
    # These object/Any annotations are Python API hooks, not arbitrary YAML mappings.
    for name in ("batched_logits_processor", "checkpoint_loader", "_mpi_session"):
        properties[name] = {
            "type": "null",
            "description": "Live objects require the Python LLM API.",
        }
    schema["$defs"]["RayPlacementConfig"]["properties"]["placement_groups"] = {"type": "null"}
    user_draft = schema["$defs"]["UserProvidedDecodingConfig"]["properties"]
    user_draft["drafter"] = {"not": {}, "description": "A Drafter requires the Python LLM API."}
    user_draft["resource_manager"] = {"type": "null"}
    # The deprecated native decoding config is accepted as a mapping by the YAML loader.
    properties["decoding_config"] = {"type": ["object", "null"]}
    return schema


def generate_disagg_schema() -> dict:
    """Describe orchestrator options and the inherited per-role worker options."""
    from tensorrt_llm.llmapi.disagg_utils import (
        DISAGG_NODE_ID_BITS,
        ConditionalDisaggConfig,
        OtlpConfig,
        extract_ctx_gen_cfgs,
        extract_disagg_cfg,
    )

    schema = generate_serve_schema()
    properties = schema["properties"]
    properties.update(
        free_gpu_memory_fraction={"type": "number", "minimum": 0, "maximum": 1},
        kv_cache_dtype={"type": "string"},
        video_pruning_rate={"type": "number"},
    )
    worker = _schema_for(extract_ctx_gen_cfgs, schema)
    worker["properties"].pop("type")
    worker.pop("required", None)
    worker["properties"] = properties | worker["properties"]
    worker["properties"]["router"] = {
        "type": "object",
        "properties": {"type": {"type": "string", "default": "round_robin"}},
        "additionalProperties": True,
    }
    worker["additionalProperties"] = False
    schema["$defs"]["DisaggServerBlock"] = worker

    orchestrator = _schema_for(extract_disagg_cfg, schema)["properties"]
    for name in ("context_servers", "generation_servers"):
        orchestrator[name] = _nullable({"$ref": "#/$defs/DisaggServerBlock"})
    orchestrator["conditional_disagg_config"] = _nullable(
        _schema_for(ConditionalDisaggConfig, schema)
    )
    orchestrator["otlp_config"] = _nullable(_schema_for(OtlpConfig, schema))
    orchestrator["disagg_cluster"] = properties["disagg_cluster"]
    orchestrator["internal_request_auth_key"] = properties["internal_request_auth_key"]
    orchestrator["node_id"] = {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": (1 << DISAGG_NODE_ID_BITS) - 1,
    }
    properties.update(orchestrator)
    return schema


def generate_visual_gen_schema() -> dict:
    """Describe the separate VisualGen engine-args YAML file."""
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    return _schema_for(VisualGenArgs)


def write_schemas(output_dir: Path, *, check: bool = False) -> list[Path]:
    """Write or check self-contained schemas for the imported package version."""
    from tensorrt_llm.version import __version__

    schemas = {
        SERVE_SCHEMA: (generate_serve_schema(), "trtllm-serve --config"),
        DISAGG_SCHEMA: (generate_disagg_schema(), "trtllm-serve disaggregated --config"),
        VISUAL_GEN_SCHEMA: (generate_visual_gen_schema(), "trtllm-serve --visual_gen_args"),
    }
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    stale = []
    for filename, (schema, command) in schemas.items():
        schema.update(
            {
                "$comment": "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
                "All rights reserved. SPDX-License-Identifier: Apache-2.0. "
                "Generated by scripts/generate_trtllm_serve_schemas.py; do not edit by hand.",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": f"{_DOCS_URL}/{__version__}/_static/schemas/{filename}",
                "title": f"TensorRT-LLM {__version__} {command}",
                "description": "Static YAML checks only; runtime validation remains authoritative.",
            }
        )
        path = output_dir / filename
        content = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(filename)
        else:
            path.write_text(content, encoding="utf-8")
        paths.append(path)
    if stale:
        raise ValueError(
            f"Missing or stale configuration schemas: {', '.join(stale)}. "
            "Run python3 scripts/generate_trtllm_serve_schemas.py and commit the updated JSON."
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tensorrt_llm/schemas",
        help="Destination directory (defaults to the checked-in schemas).",
    )
    parser.add_argument(
        "--check", action="store_true", help="Fail on missing or stale schemas without writing."
    )
    args = parser.parse_args()
    for path in write_schemas(args.output_dir, check=args.check):
        print(path)


if __name__ == "__main__":
    main()
