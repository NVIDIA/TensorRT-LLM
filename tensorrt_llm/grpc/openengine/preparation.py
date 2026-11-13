# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, portable CPU metadata for generation-only preparation reuse."""

import base64
import hashlib
import json
from dataclasses import asdict
from itertools import chain
from typing import Any

import numpy as np
from google.protobuf import struct_pb2

from tensorrt_llm.inputs.model_config import resolve_top_level_model_type
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.llmapi.llm import PreprocessedInputs

_DECODE_PREPARATION_KEY = "trtllm_decode_preparation"
_MAX_DECODE_PREPARATION_BYTES = 2 * 1024 * 1024


def _input_digest(request: Any) -> str:
    digest = hashlib.sha256()
    kind = request.WhichOneof("input")
    data = request.prompt.encode() if kind == "prompt" else request.token_ids.SerializeToString()
    for part in chain((kind.encode(), data), (item.SerializeToString() for item in request.media)):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _config_digest(llm: Any) -> str:
    processor = llm.input_processor
    config = processor.config.to_dict()
    config.pop("_name_or_path", None)
    image_processor = getattr(processor.processor, "image_processor", None)
    identity = {
        "model": config,
        "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "media_io": processor.get_preferred_media_io_kwargs(),
        "image_processor": image_processor.to_dict() if image_processor is not None else None,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _decode_preparation(
    request: Any, llm: Any, capture: dict, preprocessed: PreprocessedInputs | None = None
) -> str | None:
    if not capture:
        return None
    payload = {
        "version": 1,
        "model_type": resolve_top_level_model_type(llm.input_processor.config),
        "input_digest": _input_digest(request),
        "config_digest": _config_digest(llm),
        "state": capture["state"],
        "prompt_token_ids_u32le": base64.b64encode(
            np.asarray(capture["prompt_token_ids"], dtype="<u4").tobytes()
        ).decode("ascii"),
    }
    params = getattr(preprocessed, "multimodal_params", None)
    if params is not None and params.multimodal_input is not None:
        payload["multimodal_input"] = asdict(params.multimodal_input)
    wire_value = struct_pb2.Value(
        string_value=json.dumps(payload, separators=(",", ":"), allow_nan=False)
    )
    if wire_value.ByteSize() > _MAX_DECODE_PREPARATION_BYTES:
        return None
    return wire_value.string_value


def _prepared_decode_from_request(request: Any, llm: Any) -> PreprocessedInputs | None:
    if not request.HasField("kv") or not request.kv.HasField("session"):
        return None
    attrs = request.kv.session.attributes_struct
    if _DECODE_PREPARATION_KEY not in attrs.fields:
        return None
    value = attrs.fields[_DECODE_PREPARATION_KEY]
    if value.ByteSize() > _MAX_DECODE_PREPARATION_BYTES:
        raise ValueError("prepared decode metadata exceeds the size limit")
    if value.WhichOneof("kind") != "string_value":
        raise ValueError("prepared decode metadata must be a JSON string")
    try:
        payload = json.loads(value.string_value)
    except (ValueError, RecursionError) as error:
        raise ValueError("invalid prepared decode JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("prepared decode metadata must contain a JSON object")
    version = payload.get("version")
    if type(version) not in (int, float):
        raise ValueError("prepared decode metadata version must be a number")
    if version != 1:
        return None  # An unknown version retains the ordinary media preparation path.
    if payload.get("model_type") != resolve_top_level_model_type(llm.input_processor.config):
        raise ValueError("prepared decode metadata belongs to another model type")
    if payload.get("input_digest") != _input_digest(request):
        raise ValueError("prepared decode metadata does not match the request inputs")
    if payload.get("config_digest") != _config_digest(llm):
        raise ValueError("prepared decode metadata belongs to another model configuration")
    encoded_tokens = payload.get("prompt_token_ids_u32le")
    limit = getattr(llm.args, "max_seq_len", None) or 131072
    if (
        not isinstance(encoded_tokens, str)
        or not encoded_tokens
        or len(encoded_tokens) > ((limit * 4 + 2) // 3) * 4
    ):
        raise ValueError("invalid prepared prompt length")
    try:
        raw_tokens = base64.b64decode(encoded_tokens, validate=True)
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid prepared prompt encoding") from error
    if not raw_tokens or len(raw_tokens) % 4 or len(raw_tokens) > limit * 4:
        raise ValueError("invalid prepared prompt byte length")
    tokens = np.frombuffer(raw_tokens, dtype="<u4").tolist()
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("prepared decode model state must be an object")
    max_length = getattr(llm.args, "max_seq_len", None)
    if request.HasField("stopping") and request.stopping.HasField("max_tokens"):
        requested_length = len(tokens) + request.stopping.max_tokens
        max_length = min(max_length, requested_length) if max_length else requested_length
    restored = llm.input_processor.restore_decode_state(state, len(tokens), max_length=max_length)
    mm_input = payload.get("multimodal_input")
    if mm_input is not None:

        def normalize(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if value is None or isinstance(value, str):
                return value
            if (
                type(value) not in (int, float)
                or not -(2**31) <= value < 2**31
                or not float(value).is_integer()
            ):
                raise ValueError("invalid prepared multimodal identity")
            return int(value)

        if not isinstance(mm_input, dict):
            raise ValueError("prepared multimodal identity must be an object")
        try:
            mm_input = MultimodalInput(**normalize(mm_input))
        except (TypeError, RecursionError) as error:
            raise ValueError("invalid prepared multimodal identity fields") from error
        if any(
            len(digest) != 8 or any(type(value) is not int for value in digest)
            for digest in mm_input.multimodal_hashes
        ):
            raise ValueError("prepared multimodal hashes must contain eight int32 values")
        if any(
            position + length > len(tokens)
            for position, length in zip(mm_input.multimodal_positions, mm_input.multimodal_lengths)
        ):
            raise ValueError("prepared multimodal identity exceeds the prompt")
        if mm_input.multimodal_run_positions is not None and any(
            position + length > len(tokens)
            for position, length in zip(
                mm_input.multimodal_run_positions, mm_input.multimodal_run_lengths
            )
        ):
            raise ValueError("prepared multimodal runs exceed the prompt")
    params = MultimodalParams(multimodal_input=mm_input, multimodal_data=restored)
    try:
        params.to_handle("multimodal_data")
    except Exception:
        params.release_handles()
        raise
    return PreprocessedInputs(prompt_token_ids=tokens, multimodal_params=params)
