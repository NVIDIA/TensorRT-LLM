# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional

from pydantic import BaseModel

from tensorrt_llm.sampling_params import GuidedDecodingParams

from .reasoning_parser import ReasoningParserFactory

GPT_OSS_FINAL_CHANNEL_TRIGGER = "<|start|>assistant<|channel|>final<|message|>"


def _normalize_json_schema_for_structural_tag(json_schema: Any) -> Any:
    if isinstance(json_schema, BaseModel):
        json_schema = json_schema.model_json_schema()
    if isinstance(json_schema, str):
        json_schema = json.loads(json_schema)
    if isinstance(json_schema, dict) and "schema" in json_schema:
        json_schema = json_schema["schema"]
    return json_schema


def _guided_decoding_content(
        guided_decoding_params: GuidedDecodingParams) -> Optional[dict]:
    if guided_decoding_params.json is not None:
        json_schema = _normalize_json_schema_for_structural_tag(
            guided_decoding_params.json)
        return {"type": "json_schema", "json_schema": json_schema}
    if guided_decoding_params.json_object:
        return {"type": "json_schema", "json_schema": {"type": "object"}}
    if guided_decoding_params.regex is not None:
        return {"type": "regex", "pattern": guided_decoding_params.regex}
    if guided_decoding_params.grammar is not None:
        return {"type": "grammar", "grammar": guided_decoding_params.grammar}
    return None


def adapt_guided_decoding_for_reasoning_parser(
    guided_decoding_params: Optional[GuidedDecodingParams],
    reasoning_parser: Optional[str],
) -> Optional[GuidedDecodingParams]:
    """Constrain only final answer content for reasoning-capable formats.

    Plain JSON/regex/grammar guides constrain generation from the first token.
    Reasoning models may emit an unconstrained reasoning section before the
    final answer, so wrap the user guide in a structural tag that starts when
    final answer content begins.
    """
    if guided_decoding_params is None or reasoning_parser is None:
        return guided_decoding_params
    if guided_decoding_params.structural_tag is not None:
        return guided_decoding_params

    content = _guided_decoding_content(guided_decoding_params)
    if content is None:
        return guided_decoding_params

    if reasoning_parser.lower() == "gpt_oss":
        format_ = {
            "type":
            "triggered_tags",
            "triggers": [GPT_OSS_FINAL_CHANNEL_TRIGGER],
            "tags": [{
                "begin": GPT_OSS_FINAL_CHANNEL_TRIGGER,
                "content": content,
                "end": "",
            }],
            "stop_after_first":
            True,
        }
    else:
        parser = ReasoningParserFactory.create_reasoning_parser(
            reasoning_parser)
        format_ = {
            "type":
            "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": parser.reasoning_start,
                    "content": {
                        "type": "any_text"
                    },
                    "end": parser.reasoning_end,
                },
                content,
            ],
        }

    structural_tag = {"type": "structural_tag", "format": format_}
    return GuidedDecodingParams(
        structural_tag=json.dumps(structural_tag, separators=(",", ":")))
