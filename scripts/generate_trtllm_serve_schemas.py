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
import runpy
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import core_schema

_DOCS_URL = "https://nvidia.github.io/TensorRT-LLM"
_JSON_TYPES = {"string", "number", "integer", "boolean", "null", "array", "object"}
SERVE_SCHEMA = "trtllm-serve-config.schema.json"
DISAGG_SCHEMA = "trtllm-serve-disagg-config.schema.json"
VISUAL_GEN_SCHEMA = "trtllm-serve-visual-gen-config.schema.json"
_SCHEMA_COMMANDS = {
    SERVE_SCHEMA: "trtllm-serve --config",
    DISAGG_SCHEMA: "trtllm-serve disaggregated --config",
    VISUAL_GEN_SCHEMA: "trtllm-serve --visual_gen_args",
}


class _ServeSchemaGeneratorMixin:
    """Keep YAML-representable union branches and ignore autodoc-only type labels."""

    # Recheck these overrides on Pydantic upgrades; they inspect its schema internals.
    def handle_invalid_for_json_schema(
        self, schema: core_schema.CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        # Reject unsupported values without dropping the field or its nullable/union alternatives.
        return {"not": {}}

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


@cache
def _schema_generator() -> type:
    """Create the Pydantic generator only when full schema generation is requested."""
    from pydantic.json_schema import GenerateJsonSchema

    class ServeSchemaGenerator(_ServeSchemaGeneratorMixin, GenerateJsonSchema):
        pass

    return ServeSchemaGenerator


def _schema_for(annotation: Any, parent: dict | None = None) -> dict:
    from pydantic import TypeAdapter

    schema = TypeAdapter(annotation).json_schema(schema_generator=_schema_generator())
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
    from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, InternalRequestAuthKey
    from tensorrt_llm.llmapi.llm_args import (
        SERVE_CONFIG_ALIASES,
        MoeLoadBalancerConfig,
        MoePerLayerDynamicLoadBalancerConfig,
        TorchLlmArgs,
    )

    # Annotated fields are derived; config changes still require snapshot regeneration and CPU tests.
    schema = _schema_for(TorchLlmArgs)
    # MODEL is supplied on the command line; nested required fields still apply.
    schema["required"] = [name for name in schema.get("required", []) if name != "model"]
    properties = schema["properties"]
    for alias, name in SERVE_CONFIG_ALIASES.items():
        properties[alias] = copy.deepcopy(properties[name])
    # Keep server-only fields aligned with serve.py.
    properties.update(
        allow_request_chat_template={"type": "boolean", "default": False},
        internal_request_auth_key=_schema_for(InternalRequestAuthKey),
        disagg_cluster=_nullable(_schema_for(DisaggClusterConfig, schema)),
    )
    # Serving treats null as unset before constructing the non-optional telemetry model.
    properties["telemetry_config"] = _nullable(properties["telemetry_config"])
    # Keep accepted forms aligned with TorchLlmArgs.validate_load_balancer(): a mapping
    # without mode is standard EPLB, so only the per-layer dynamic form requires it.
    standard, per_layer_dynamic = (
        {"$ref": f"#/$defs/{model.__name__}"}
        for model in (MoeLoadBalancerConfig, MoePerLayerDynamicLoadBalancerConfig)
    )
    schema["$defs"]["MoeConfig"]["properties"]["load_balancer"]["anyOf"] = [
        standard,
        {**per_layer_dynamic, "required": ["mode"]},
        {"type": ["string", "null"]},
    ]
    # These object/Any annotations are Python API hooks, not arbitrary YAML mappings.
    # Update these restrictions when hooks are added or gain YAML construction support.
    for name in ("batched_logits_processor", "checkpoint_loader", "_mpi_session"):
        properties[name] = {
            "type": "null",
            "description": "Live objects require the Python LLM API.",
        }
    schema["$defs"]["RayPlacementConfig"]["properties"]["placement_groups"] = {"type": "null"}
    user_draft = schema["$defs"]["UserProvidedDecodingConfig"]["properties"]
    user_draft["drafter"] = {"not": {}, "description": "A Drafter requires the Python LLM API."}
    user_draft["resource_manager"] = {"type": "null"}
    # Update this mapping exception when the YAML loader changes legacy DecodingConfig support.
    properties["decoding_config"] = {"type": ["object", "null"]}
    return schema


def generate_disagg_schema() -> dict:
    """Describe orchestrator options and the inherited per-role worker options."""
    from tensorrt_llm.llmapi.disagg_utils import (
        DISAGG_NODE_ID_BITS,
        ConditionalDisaggConfig,
        OtlpConfig,
        RouterConfig,
        extract_ctx_gen_cfgs,
        extract_disagg_cfg,
    )
    from tensorrt_llm.llmapi.llm_args import SERVE_CLI_CONFIG_FIELDS, TorchLlmArgs

    schema = generate_serve_schema()
    properties = schema["properties"]
    for config_name, aliases in SERVE_CLI_CONFIG_FIELDS.items():
        config = _schema_for(TorchLlmArgs.model_fields[config_name].annotation, schema)
        for name, field in aliases.items():
            properties[name] = copy.deepcopy(config["properties"][field])
    worker = _schema_for(extract_ctx_gen_cfgs, schema)
    # Only type is required and supplied internally; revisit if worker inputs gain required fields.
    worker["properties"].pop("type")
    worker.pop("required", None)
    worker["properties"] = properties | worker["properties"]
    # YAML flattens router-specific args alongside the shared type field.
    worker["properties"]["router"] = {
        "type": "object",
        "properties": {"type": _schema_for(RouterConfig, schema)["properties"]["type"]},
        "additionalProperties": True,
    }
    worker["additionalProperties"] = False
    schema["$defs"]["DisaggServerBlock"] = worker

    orchestrator = _schema_for(extract_disagg_cfg, schema)["properties"]
    # Update these refinements when extract_disagg_cfg() changes its untyped mapping inputs.
    for name in ("context_servers", "generation_servers"):
        orchestrator[name] = _nullable({"$ref": "#/$defs/DisaggServerBlock"})
    orchestrator["conditional_disagg_config"] = _nullable(
        _schema_for(ConditionalDisaggConfig, schema)
    )
    orchestrator["otlp_config"] = _nullable(_schema_for(OtlpConfig, schema))
    orchestrator["disagg_cluster"] = properties["disagg_cluster"]
    orchestrator["internal_request_auth_key"] = properties["internal_request_auth_key"]
    # Mirror extract_disagg_cfg()'s node_id validation; the bit limit is shared.
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


def _source_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "tensorrt_llm/version.py"
    version = runpy.run_path(str(version_file)).get("__version__")
    if not isinstance(version, str) or not version:
        raise ValueError(f"Missing __version__ string in {version_file}")
    return version


def _set_schema_version_metadata(schema: dict, filename: str, command: str, version: str) -> None:
    schema["$id"] = f"{_DOCS_URL}/{version}/_static/schemas/{filename}"
    schema["title"] = f"TensorRT-LLM {version} {command}"


def _set_schema_metadata(schema: dict, filename: str, command: str, version: str) -> None:
    schema.update(
        {
            "$comment": "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
            "All rights reserved. SPDX-License-Identifier: Apache-2.0. "
            "Generated by scripts/generate_trtllm_serve_schemas.py; do not edit by hand.",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "description": "Static YAML checks only; runtime validation remains authoritative.",
        }
    )
    _set_schema_version_metadata(schema, filename, command, version)


def _raise_stale(stale: list[str], *, sync_version: bool = False) -> None:
    if stale:
        command = "python3 scripts/generate_trtllm_serve_schemas.py"
        if sync_version:
            command += " --sync-version"
        raise ValueError(
            f"Missing or stale configuration schemas: {', '.join(stale)}. "
            f"Run {command} and commit the updated JSON."
        )


def write_schemas(output_dir: Path, *, check: bool = False) -> list[Path]:
    """Write or check self-contained schemas for the source package version."""
    version = _source_version()

    schemas = {
        SERVE_SCHEMA: generate_serve_schema(),
        DISAGG_SCHEMA: generate_disagg_schema(),
        VISUAL_GEN_SCHEMA: generate_visual_gen_schema(),
    }
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    stale = []
    for filename, command in _SCHEMA_COMMANDS.items():
        schema = schemas[filename]
        _set_schema_metadata(schema, filename, command, version)
        path = output_dir / filename
        content = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(filename)
        else:
            path.write_text(content, encoding="utf-8")
        paths.append(path)
    _raise_stale(stale)
    return paths


def sync_schema_version_metadata(output_dir: Path, *, check: bool = False) -> list[Path]:
    """Update version-derived metadata without importing TensorRT-LLM or Pydantic."""
    version = _source_version()
    paths = []
    updates = {}
    missing = []
    for filename, command in _SCHEMA_COMMANDS.items():
        path = output_dir / filename
        paths.append(path)
        if not path.is_file():
            missing.append(filename)
            continue
        schema = json.loads(path.read_text(encoding="utf-8"))
        original_metadata = (schema.get("$id"), schema.get("title"))
        _set_schema_version_metadata(schema, filename, command, version)
        if original_metadata != (schema["$id"], schema["title"]):
            updates[path] = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    if missing:
        raise ValueError(
            f"Cannot sync version metadata for missing schemas: {', '.join(missing)}. "
            "Run python3 scripts/generate_trtllm_serve_schemas.py with a matching build first."
        )
    if check:
        _raise_stale([path.name for path in updates], sync_version=True)
    else:
        for path, content in updates.items():
            path.write_text(content, encoding="utf-8")
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
    parser.add_argument(
        "--sync-version",
        action="store_true",
        help="Update only version-derived metadata without importing project dependencies.",
    )
    args = parser.parse_args()
    writer = sync_schema_version_metadata if args.sync_version else write_schemas
    for path in writer(args.output_dir, check=args.check):
        print(path)


if __name__ == "__main__":
    main()
