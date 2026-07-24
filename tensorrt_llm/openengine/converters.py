# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversions between OpenEngine messages and TensorRT-LLM API objects."""

import asyncio
import base64
import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any

from google.protobuf.json_format import MessageToDict
from openengine.v1 import generation_pb2, kv_pb2

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.inputs.media_io import MEDIA_IO_REGISTRY
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

HANDOFF_ATTRIBUTE = "tensorrt_llm.disaggregated_params.v1"
_MODALITY_NAMES = {
    generation_pb2.MODALITY_UNSPECIFIED: "image",
    generation_pb2.MODALITY_IMAGE: "image",
    generation_pb2.MODALITY_VIDEO: "video",
    generation_pb2.MODALITY_AUDIO: "audio",
}


def _optional(message: object, field: str) -> Any | None:
    return getattr(message, field) if message.HasField(field) else None


def _candidate_count(selection: object, enabled: bool) -> int | None:
    if not enabled:
        return None
    kind = selection.WhichOneof("selection")
    if kind in (None, "top_n"):
        return selection.top_n if kind == "top_n" else 0
    raise ValueError("TensorRT-LLM OpenEngine supports top_n logprob selection only")


def to_sampling_params(request: generation_pb2.GenerateRequest) -> SamplingParams:
    """Build TRT-LLM sampling params without inventing wire defaults."""
    sampling = request.sampling
    stopping = request.stopping
    response = request.response
    kwargs: dict[str, Any] = {
        "max_tokens": (32 if _optional(stopping, "max_tokens") is None else stopping.max_tokens),
        "detokenize": True,
    }
    field_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "repetition_penalty": "repetition_penalty",
        "seed": "seed",
    }
    for proto_name, trt_name in field_map.items():
        value = _optional(sampling, proto_name)
        if value is not None:
            kwargs[trt_name] = value
    num_sequences = _optional(sampling, "num_sequences")
    if num_sequences is not None:
        kwargs["n"] = num_sequences
        kwargs["best_of"] = num_sequences
    min_tokens = _optional(stopping, "min_tokens")
    if min_tokens is not None:
        kwargs["min_tokens"] = min_tokens
    ignore_eos = _optional(stopping, "ignore_eos")
    if ignore_eos is not None:
        kwargs["ignore_eos"] = ignore_eos
    include_stop = _optional(stopping, "include_stop_in_output")
    if include_stop is not None:
        kwargs["include_stop_str_in_output"] = include_stop
    stop_text: list[str] = []
    stop_ids: list[int] = []
    for condition in stopping.conditions:
        kind = condition.WhichOneof("condition")
        if kind == "stop_text":
            stop_text.append(condition.stop_text)
        elif kind == "stop_token_id":
            stop_ids.append(condition.stop_token_id)
    if stop_text:
        kwargs["stop"] = stop_text
    if stop_ids:
        kwargs["stop_token_ids"] = stop_ids

    prompt_requested = bool(_optional(response, "return_prompt_logprobs"))
    output_requested = bool(_optional(response, "return_output_logprobs"))
    prompt_count = _candidate_count(response.prompt_candidates, prompt_requested)
    output_count = _candidate_count(response.output_candidates, output_requested)
    if prompt_count is not None:
        kwargs["prompt_logprobs"] = prompt_count
    if output_count is not None:
        kwargs["logprobs"] = output_count
    prompt_start = _optional(response, "prompt_logprob_start")
    if prompt_start not in (None, 0):
        raise ValueError("TensorRT-LLM does not support prompt_logprob_start through OpenEngine")

    guide_kind = request.guided.WhichOneof("guide")
    if guide_kind:
        guide_kwargs: dict[str, Any]
        if guide_kind == "json_schema":
            guide_kwargs = {"json": request.guided.json_schema}
        elif guide_kind == "regex":
            guide_kwargs = {"regex": request.guided.regex}
        elif guide_kind == "ebnf_grammar":
            guide_kwargs = {"grammar": request.guided.ebnf_grammar}
        elif guide_kind == "structural_tag":
            guide_kwargs = {"structural_tag": request.guided.structural_tag}
        elif guide_kind == "choice":
            choices = [f"(?:{re.escape(choice)})" for choice in request.guided.choice.choices]
            guide_kwargs = {"regex": "^(?:" + "|".join(choices) + ")$"}
        elif guide_kind == "json_object":
            guide_kwargs = {"json_object": True}
        else:
            raise ValueError(f"Unsupported guided decoding mode {guide_kind!r}")
        kwargs["guided_decoding"] = GuidedDecodingParams(**guide_kwargs)
    return SamplingParams(**kwargs)


def to_priority(priority: int | None) -> float:
    """Map signed OpenEngine ordering into TRT-LLM's bounded priority domain."""
    if priority is None:
        return 0.5
    return 0.5 + 0.5 * priority / (1 + abs(priority))


def _message_struct(struct_message: object) -> dict[str, Any]:
    if not struct_message.fields:
        return {}
    return MessageToDict(struct_message, preserving_proto_field_name=True)


def validate_media_options(options: dict[str, Any]) -> None:
    unknown = set(options).difference(MEDIA_IO_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown media option modalities: {sorted(unknown)}")
    invalid = [name for name, value in options.items() if not isinstance(value, dict)]
    if invalid:
        raise ValueError(f"Media options must be objects for modalities: {sorted(invalid)}")


async def load_media(
    media: list[object],
    media_options: object,
    server_config: MultimodalServerConfig | None,
) -> dict[str, list[Any]] | None:
    """Decode ordered OpenEngine media with TRT-LLM's media-I/O merge rules."""
    if not media:
        return None
    request_options = _message_struct(media_options)
    validate_media_options(request_options)
    server_options = (server_config.media_io_kwargs if server_config is not None else None) or {}
    output: dict[str, list[Any]] = {}
    pending: list[tuple[str, asyncio.Future[Any] | asyncio.Task[Any] | Any]] = []
    for item in media:
        modality = _MODALITY_NAMES.get(item.modality)
        if modality is None:
            raise ValueError(f"Unsupported media modality {item.modality}")
        media_io = MEDIA_IO_REGISTRY[modality].create(
            server_options.get(modality), request_options.get(modality)
        )
        source = item.WhichOneof("source")
        if source in ("url", "data_uri"):
            pending.append((modality, media_io.async_load(getattr(item, source))))
        elif source == "raw_bytes":
            pending.append(
                (modality, media_io._run_in_executor(media_io.load_bytes, item.raw_bytes))
            )
        else:
            raise ValueError("Each media item must carry exactly one source")
    decoded = await asyncio.gather(*(awaitable for _, awaitable in pending))
    for (modality, _), value in zip(pending, decoded):
        output.setdefault(modality, []).append(value)
    return output


def media_uuids(media: list[object]) -> dict[str, list[str | None]] | None:
    """Preserve caller media IDs for engine KV-event key correlation."""
    output: dict[str, list[str | None]] = {}
    any_uuid = False
    for item in media:
        modality = _MODALITY_NAMES.get(item.modality)
        if modality is None:
            raise ValueError(f"Unsupported media modality {item.modality}")
        uuid = item.uuid or None
        any_uuid = any_uuid or uuid is not None
        output.setdefault(modality, []).append(uuid)
    return output if any_uuid else None


def stable_request_id(request_id: str) -> int:
    """Map arbitrary wire request IDs into TRT-LLM's positive int64 domain."""
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if hasattr(value, "logprob"):
        output = {"logprob": float(value.logprob)}
        if getattr(value, "rank", None) is not None:
            output["rank"] = int(value.rank)
        return output
    if hasattr(value, "item"):
        return value.item()
    return value


def encode_handoff(
    params: DisaggregatedParams, *, requires_decode_media: bool = False
) -> kv_pb2.KvSessionRef:
    """Encode the revisioned context-first TRT handoff profile."""
    unsupported = {
        "first-generation logits": params.first_gen_logits,
        "multimodal embedding handles": params.multimodal_embedding_handles,
        "multimodal hashes": params.multimodal_hashes,
    }
    if not requires_decode_media:
        unsupported.update(
            {
                "mrope position IDs": params.mrope_position_ids_handle,
                "mrope position deltas": params.mrope_position_deltas_handle,
            }
        )
    present = [name for name, value in unsupported.items() if value is not None]
    if present:
        raise ValueError("OpenEngine context-first handoff does not support " + ", ".join(present))
    if params.schedule_style not in (None, DisaggScheduleStyle.CONTEXT_FIRST):
        raise ValueError("Generation-first handoff is not supported")
    payload = {
        "first_gen_tokens": _json_value(params.first_gen_tokens),
        "first_gen_log_probs": _json_value(params.first_gen_log_probs),
        "ctx_request_id": None if params.ctx_request_id is None else str(params.ctx_request_id),
        "disagg_request_id": None
        if params.disagg_request_id is None
        else str(params.disagg_request_id),
        "ctx_dp_rank": params.ctx_dp_rank,
        "ctx_info_endpoint": params.ctx_info_endpoint,
        "draft_tokens": _json_value(params.draft_tokens),
        "ctx_usage": _json_value(params.ctx_usage),
        "conversation_id": params.conversation_id,
        "schedule_style": "context_first",
        "requires_decode_media": requires_decode_media,
        "opaque_state": (
            None
            if params.opaque_state is None
            else base64.b64encode(params.opaque_state).decode("ascii")
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    session = kv_pb2.KvSessionRef(
        session_id=str(params.disagg_request_id or params.ctx_request_id or ""),
        transfer_backend="tensorrt_llm",
        dp_rank=params.ctx_dp_rank or 0,
        handoff_profile=HANDOFF_ATTRIBUTE,
    )
    session.attributes_struct[HANDOFF_ATTRIBUTE] = canonical
    return session


def _decode_handoff_payload(session: kv_pb2.KvSessionRef) -> dict[str, Any]:
    if session.handoff_profile and session.handoff_profile != HANDOFF_ATTRIBUTE:
        raise ValueError(f"Unsupported TensorRT-LLM handoff profile {session.handoff_profile!r}")
    if HANDOFF_ATTRIBUTE not in session.attributes_struct:
        raise ValueError(f"KV session is missing {HANDOFF_ATTRIBUTE!r}")
    encoded = session.attributes_struct[HANDOFF_ATTRIBUTE]
    if not isinstance(encoded, str):
        raise ValueError("TensorRT-LLM handoff attribute must be a JSON string")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("TensorRT-LLM handoff is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("TensorRT-LLM handoff must contain a JSON object")
    return payload


def handoff_requires_decode_media(session: kv_pb2.KvSessionRef) -> bool:
    """Return whether decode must recompute transient MM state from raw media."""
    payload = _decode_handoff_payload(session)
    value = payload.get("requires_decode_media", False)
    if not isinstance(value, bool):
        raise ValueError("requires_decode_media must be a boolean")
    return value


def decode_handoff(session: kv_pb2.KvSessionRef) -> DisaggregatedParams:
    """Decode and validate the revisioned context-first TRT handoff profile."""
    payload = _decode_handoff_payload(session)
    if payload.get("schedule_style", "context_first") != "context_first":
        raise ValueError("Generation-first handoff is not supported")
    if any(
        payload.get(key) is not None
        for key in (
            "first_gen_logits",
            "multimodal_embedding_handles",
            "multimodal_hashes",
            "mrope_position_ids_handle",
            "mrope_position_deltas_handle",
        )
    ):
        raise ValueError("Encoder-stage and first-generation logits handles are not supported")

    def decimal_id(name: str) -> int | None:
        value = payload.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.isdecimal():
            raise ValueError(f"{name} must be a decimal string")
        return int(value)

    opaque = payload.get("opaque_state")
    try:
        opaque_state = None if opaque is None else base64.b64decode(opaque, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("opaque_state must be canonical base64") from error

    ctx_usage = payload.get("ctx_usage")
    if ctx_usage is not None:
        if not isinstance(ctx_usage, dict):
            raise ValueError("ctx_usage must be an object")
        prompt_tokens = ctx_usage.get("prompt_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens < 0
        ):
            raise ValueError("ctx_usage.prompt_tokens must be a non-negative integer")
        prompt_details = ctx_usage.get("prompt_tokens_details")
        if prompt_details is not None:
            if not isinstance(prompt_details, dict):
                raise ValueError("ctx_usage.prompt_tokens_details must be an object")
            cached_tokens = prompt_details.get("cached_tokens", 0)
            if (
                not isinstance(cached_tokens, int)
                or isinstance(cached_tokens, bool)
                or cached_tokens < 0
            ):
                raise ValueError(
                    "ctx_usage.prompt_tokens_details.cached_tokens must be a non-negative integer"
                )

    def logprobs() -> list[Any] | None:
        encoded_logprobs = payload.get("first_gen_log_probs")
        if encoded_logprobs is None:
            return None
        if not isinstance(encoded_logprobs, list):
            raise ValueError("first_gen_log_probs must be a list")
        decoded: list[Any] = []
        for position in encoded_logprobs:
            if isinstance(position, (int, float)):
                decoded.append(float(position))
                continue
            if not isinstance(position, dict):
                raise ValueError("first_gen_log_probs entries must be candidate objects or numbers")
            candidates: dict[int, Logprob] = {}
            for token_id, value in position.items():
                if not isinstance(value, dict) or "logprob" not in value:
                    raise ValueError("first_gen_log_probs contains an invalid candidate")
                try:
                    decoded_token_id = int(token_id)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "first_gen_log_probs contains a non-integer token ID"
                    ) from error
                candidates[decoded_token_id] = Logprob(
                    logprob=float(value["logprob"]),
                    rank=(None if value.get("rank") is None else int(value["rank"])),
                )
            decoded.append(candidates)
        return decoded

    return DisaggregatedParams(
        request_type="generation_only",
        first_gen_tokens=payload.get("first_gen_tokens"),
        first_gen_log_probs=logprobs(),
        ctx_request_id=decimal_id("ctx_request_id"),
        disagg_request_id=decimal_id("disagg_request_id"),
        ctx_dp_rank=payload.get("ctx_dp_rank", session.dp_rank),
        ctx_info_endpoint=payload.get("ctx_info_endpoint"),
        draft_tokens=payload.get("draft_tokens"),
        ctx_usage=ctx_usage,
        conversation_id=payload.get("conversation_id"),
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        opaque_state=opaque_state,
    )


def to_lora_request(adapter: object | None) -> LoRARequest | None:
    if adapter is None:
        return None
    return LoRARequest(adapter.lora_name, adapter.lora_id, adapter.source_path)
