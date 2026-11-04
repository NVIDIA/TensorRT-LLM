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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py

import math
from typing import Any, Dict, Optional

import torch
from openai.types.responses import ResponseFormatTextConfig

from tensorrt_llm.llmapi import GuidedDecodingParams, SamplingParams
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.serve.openai_protocol.common import (EarlyStopping,
                                                       OpenAIBaseModel,
                                                       ResponseFormat)
from tensorrt_llm.serve.serving_extensions import structured_output_format_for

_LOGIT_BIAS_MIN = -100.0

_LOGIT_BIAS_MAX = 100.0


def _early_stopping_to_int(value: Optional[EarlyStopping]) -> Optional[int]:
    """Translate the HF-style tri-state into the engine's integer encoding.

    Mirrors ``BeamSearchEarlyStop``: ``False`` -> 0, ``True`` -> 1,
    ``"never"`` -> 2. ``None`` stays unset so the engine picks its default.
    """
    if value is None:
        return None
    if value == "never":
        return 2
    # NB: bool is a subclass of int, so this yields 1/0 for True/False.
    return int(value)


def _logit_bias_to_embedding_bias(
        logit_bias: Optional[Dict[str, float]],
        vocab_size: Optional[int]) -> Optional[torch.Tensor]:
    """Convert OpenAI logit_bias dict to embedding_bias tensor for sampling."""
    if logit_bias is None:
        return None
    if vocab_size is None:
        raise ValueError(
            "logit_bias requires a tokenizer, but the server was started "
            "without one (e.g. num_postprocess_workers > 0). "
            "Remove logit_bias from your request or set num_postprocess_workers=0."
        )
    elif vocab_size <= 0:
        raise ValueError("vocab_size must be positive when logit_bias is used")

    # Create 1D zeros tensor as expected by executor API (will be unsqueezed to [1, vocab_size] internally)
    embedding_bias = torch.zeros(vocab_size, dtype=torch.float32)

    # Apply biases for specified token IDs
    for token_str, bias in logit_bias.items():
        try:
            token_id = int(token_str)
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(
                    f"Invalid logit_bias key '{token_str}': must be a valid integer token ID"
                )
            raise
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"Token ID {token_id} out of vocabulary range [0, {vocab_size})"
            )
        try:
            bias_value = float(bias)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"logit_bias value for token ID {token_id} must be a number"
            ) from e
        if not math.isfinite(bias_value):
            raise ValueError(f"logit_bias value for token ID {token_id} "
                             "must be finite")
        if not _LOGIT_BIAS_MIN <= bias_value <= _LOGIT_BIAS_MAX:
            raise ValueError(
                f"logit_bias value for token ID {token_id} must be in "
                f"[{_LOGIT_BIAS_MIN:g}, {_LOGIT_BIAS_MAX:g}]")
        embedding_bias[token_id] = bias_value

    return embedding_bias


def _response_format_to_guided_decoding_params(
    response_format: Optional[ResponseFormat],
    reasoning_parser: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[GuidedDecodingParams]:
    if response_format is None:
        guided_decoding_params = None
    elif response_format.type == "text":
        guided_decoding_params = None
    elif response_format.type == "json":
        if response_format.schema is None:
            raise ValueError(
                f"response_format.schema is required for response_format.type == {response_format.type!r}, but got None."
            )
        guided_decoding_params = GuidedDecodingParams(
            json=response_format.schema)
    elif response_format.type == "json_schema":
        if response_format.json_schema is None:
            raise ValueError(
                f"response_format.json_schema is required for response_format.type == {response_format.type!r}, but got None."
            )
        # OpenAI API spec wraps the actual JSON schema under a "schema" key.
        # Extract the actual schema if the wrapper format is used.
        json_schema = response_format.json_schema
        if isinstance(json_schema, dict) and "schema" in json_schema:
            json_schema = json_schema["schema"]
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
    elif response_format.type == "json_object":
        guided_decoding_params = GuidedDecodingParams(json_object=True)
    elif response_format.type == "regex":
        if response_format.regex is None:
            raise ValueError(
                f"response_format.regex is required for response_format.type == {response_format.type!r}, but got None."
            )
        guided_decoding_params = GuidedDecodingParams(
            regex=response_format.regex)
    elif response_format.type == "ebnf":
        if response_format.ebnf is None:
            raise ValueError(
                f"response_format.ebnf is required for response_format.type == {response_format.type!r}, but got None."
            )
        guided_decoding_params = GuidedDecodingParams(
            grammar=response_format.ebnf)
    elif response_format.type == "structural_tag":
        guided_decoding_params = GuidedDecodingParams(
            structural_tag=response_format.model_dump_json(by_alias=True,
                                                           exclude_none=True))
    else:
        raise ValueError(f"Unsupported response format: {response_format.type}")

    if guided_decoding_params is None or reasoning_parser is None:
        return guided_decoding_params

    if guided_decoding_params.structural_tag is not None:
        return guided_decoding_params

    # Adapt guided_decoding_params for reasoning parser
    if guided_decoding_params.json is not None:
        content = {
            "type": "json_schema",
            "json_schema": guided_decoding_params.json
        }
    elif guided_decoding_params.json_object:
        content = {"type": "json_schema", "json_schema": {"type": "object"}}
    elif guided_decoding_params.regex is not None:
        content = {"type": "regex", "pattern": guided_decoding_params.regex}
    elif guided_decoding_params.grammar is not None:
        content = {"type": "grammar", "grammar": guided_decoding_params.grammar}

    extension_format = structured_output_format_for(reasoning_parser)
    if extension_format is not None:
        # A registered per-model serving extension owns the placement of the
        # constraint relative to this model's reasoning markup.
        stag_format = extension_format(content, chat_template_kwargs)
        if stag_format is None:
            return guided_decoding_params
    else:
        # Force thinking and then trigger user constraint
        parser = ReasoningParserFactory.create_reasoning_parser(
            reasoning_parser)
        stag_format = {
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

    stag_format = ResponseFormat(type="structural_tag", format=stag_format)
    return GuidedDecodingParams(structural_tag=stag_format.model_dump_json(
        by_alias=True, exclude_none=True))


def _response_format_text_config_to_guided_decoding_params(
    text_format: Optional[ResponseFormatTextConfig],
    reasoning_parser: Optional[str] = None,
) -> Optional[GuidedDecodingParams]:
    if text_format is None:
        return None

    resp_format = ResponseFormat(type=text_format.type,
                                 json_schema=getattr(text_format, "schema_",
                                                     None))
    return _response_format_to_guided_decoding_params(
        resp_format, reasoning_parser=reasoning_parser)


def _record_sampling_params_request_fields(
        request: OpenAIBaseModel, sampling_params: SamplingParams) -> None:
    """Preserve explicitly supplied fields across protocol defaulting."""
    sampling_params.set_request_provided_fields(
        field_name for field_name in request.model_fields_set
        if getattr(request, field_name, None) is not None)
