# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine gRPC servicer backed by one TensorRT-LLM LLM instance."""

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import replace
from pathlib import Path
from typing import Any

import grpc
from openengine import MINIMUM_CLIENT_REVISION, SCHEMA_RELEASE, SCHEMA_REVISION
from openengine.v1 import (
    error_pb2,
    generation_pb2,
    input_pb2,
    kv_pb2,
    lifecycle_pb2,
    lora_pb2,
    model_pb2,
    observability_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

import tensorrt_llm
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_hash import (
    KV_CACHE_HASH_ALGO_V1,
    KV_CACHE_HASH_ALGO_V2_SHA256_64,
    get_effective_kv_cache_event_hash_algo,
)
from tensorrt_llm.scheduling_params import SchedulingParams
from tensorrt_llm.serve.kv_event_fanout import KvEventFanout
from tensorrt_llm.serve.request_tracker import RequestTracker
from tensorrt_llm.serve.stats_fanout import StatsFanout

from .converters import (
    HANDOFF_ATTRIBUTE,
    decode_handoff,
    encode_handoff,
    handoff_requires_decode_media,
    load_media,
    media_uuids,
    stable_request_id,
    to_priority,
    to_sampling_params,
)
from .lora_registry import LoraRegistry


def schema_release() -> str:
    """Return the immutable OpenEngine source identity configured at startup."""
    return os.getenv("OPENENGINE_SCHEMA_RELEASE", SCHEMA_RELEASE)


def _arg(llm: object, name: str, default: Any = None) -> Any:
    args = getattr(llm, "args", None)
    value = getattr(args, name, None)
    if value is not None:
        return value
    parallel = getattr(args, "parallel_config", None)
    return getattr(parallel, name, default)


def _token_text(llm: object, token_id: int) -> str:
    tokenizer = getattr(llm, "tokenizer", None)
    if tokenizer is None:
        return ""
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except (ValueError, TypeError, AttributeError):
        return ""


def _signed_i64(value: int) -> int:
    if value >= 1 << 63:
        return value - (1 << 64)
    if value < -(1 << 63):
        return ((value + (1 << 63)) % (1 << 64)) - (1 << 63)
    return value


def _seconds(value: object) -> float:
    converter = getattr(value, "total_seconds", None)
    if callable(converter):
        return float(converter())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _block_hash(value: object, hash_algo: str | None) -> kv_pb2.KvBlockHash:
    if isinstance(value, bytes):
        return kv_pb2.KvBlockHash(value=value, encoding=hash_algo or "bytes")
    if isinstance(value, int):
        return kv_pb2.KvBlockHash(
            value=str(_signed_i64(value)).encode("ascii"), encoding="decimal_int64"
        )
    text = str(value)
    try:
        encoded = bytes.fromhex(text)
    except ValueError:
        return kv_pb2.KvBlockHash(value=text.encode("utf-8"), encoding="utf8")
    return kv_pb2.KvBlockHash(value=encoded, encoding=hash_algo or "hex")


class OpenEngineServicer(
    openengine_pb2_grpc.InferenceServicer,
    openengine_pb2_grpc.ControlServicer,
):
    """Engine-neutral service surface over TensorRT-LLM's Python LLM API."""

    def __init__(
        self,
        llm: object,
        model: str,
        role: int,
        tracker: RequestTracker,
        media_config: MultimodalServerConfig | None = None,
        reasoning_parser: str | None = None,
        tool_parser: str | None = None,
        kv_event_fanout: KvEventFanout | None = None,
        stats_fanout: StatsFanout | None = None,
        instance_id: str | None = None,
        event_host: str = "127.0.0.1",
        event_port: int = 0,
        kv_session_ttl_seconds: float = 300.0,
        post_abort_cleanup_timeout_seconds: float = 1.0,
    ) -> None:
        self.llm = llm
        self.model = model
        model_id = _arg(llm, "model", model)
        self.model_id = str(model_id) if model_id else model
        tokenizer_source = _arg(llm, "tokenizer", None)
        self.tokenizer_source = str(tokenizer_source) if tokenizer_source else self.model_id
        tokenizer_mode = _arg(llm, "tokenizer_mode", "auto")
        self.tokenizer_mode = str(tokenizer_mode) if tokenizer_mode else "auto"
        self.model_aliases = [self.model_id] if self.model_id != self.model else []
        self._accepted_model_names = {self.model, *self.model_aliases}
        self.role = role
        self.tracker = tracker
        self.media_config = media_config
        self.reasoning_parser = reasoning_parser or ""
        self.tool_parser = tool_parser or ""
        processor = getattr(llm, "input_processor", None)
        identity_getter = getattr(processor, "get_model_owned_lora_identities", None)
        model_owned_loras = identity_getter() if callable(identity_getter) else {}
        self.loras = LoraRegistry(model_owned_loras)
        self._lora_names_by_id: dict[int, str] = {
            adapter_id: name for name, adapter_id in model_owned_loras.items()
        }
        self.kv_events = kv_event_fanout
        self.stats = stats_fanout
        self._kv_session_requests: dict[str, str] = {}
        self._kv_session_timers: dict[str, asyncio.TimerHandle] = {}
        if kv_session_ttl_seconds <= 0:
            raise ValueError("kv_session_ttl_seconds must be positive")
        self._kv_session_ttl_seconds = kv_session_ttl_seconds
        if post_abort_cleanup_timeout_seconds <= 0:
            raise ValueError("post_abort_cleanup_timeout_seconds must be positive")
        self._post_abort_cleanup_timeout_seconds = post_abort_cleanup_timeout_seconds
        self._partial_block_hashes: dict[int, set[object]] = {}
        self.instance_id = instance_id or str(uuid.uuid4())
        self.event_host = event_host
        self.event_port = event_port

    def _log_handoff(
        self,
        request: generation_pb2.GenerateRequest,
        phase: str,
        session: kv_pb2.KvSessionRef | None = None,
        result: object | None = None,
    ) -> None:
        if self.role == server_pb2.ENGINE_ROLE_AGGREGATED:
            return
        if session is None:
            if not request.kv.HasField("session"):
                return
            session = request.kv.session
        if not session.session_id:
            return

        record: dict[str, Any] = {
            "phase": phase,
            "role": server_pb2.EngineRole.Name(self.role),
            "request_id": request.request_id,
            "session_id": session.session_id,
            "handoff_profile": session.handoff_profile,
            "dp_rank": session.dp_rank,
        }
        try:
            handoff = decode_handoff(session)
        except (TypeError, ValueError):
            handoff = None
        if handoff is not None:
            record["tensorrt_llm"] = {
                "ctx_request_id": (
                    str(handoff.ctx_request_id) if handoff.ctx_request_id is not None else None
                ),
                "disagg_request_id": (
                    str(handoff.disagg_request_id)
                    if handoff.disagg_request_id is not None
                    else None
                ),
                "ctx_dp_rank": handoff.ctx_dp_rank,
                "ctx_info_endpoint": handoff.ctx_info_endpoint,
                "conversation_id": handoff.conversation_id,
                "schedule_style": (
                    handoff.schedule_style.name
                    if isinstance(handoff.schedule_style, DisaggScheduleStyle)
                    else str(handoff.schedule_style)
                ),
            }

        if result is not None:
            for output in getattr(result, "outputs", ()):
                metrics = getattr(output, "request_perf_metrics", None)
                timing = getattr(metrics, "timing_metrics", None)
                if timing is None:
                    continue
                kv_bytes = int(getattr(timing, "kv_cache_size", 0) or 0)
                start = getattr(timing, "kv_cache_transfer_start", None)
                end = getattr(timing, "kv_cache_transfer_end", None)
                start_seconds = _seconds(start)
                end_seconds = _seconds(end)
                record["kv_transfer"] = {
                    "bytes": kv_bytes,
                    "duration_seconds": max(0.0, end_seconds - start_seconds),
                }
                break

        logger.info("OpenEngine handoff %s", json.dumps(record, sort_keys=True))

    def _log_lora_selection(
        self,
        request: generation_pb2.GenerateRequest,
        phase: str,
        session: kv_pb2.KvSessionRef | None = None,
    ) -> None:
        if not request.lora_name:
            return
        session_id = ""
        if session is not None:
            session_id = session.session_id
        elif request.kv.HasField("session"):
            session_id = request.kv.session.session_id
        logger.info(
            "OpenEngine LoRA selection %s",
            json.dumps(
                {
                    "phase": phase,
                    "role": server_pb2.EngineRole.Name(self.role),
                    "request_id": request.request_id,
                    "session_id": session_id,
                    "lora_name": request.lora_name,
                },
                sort_keys=True,
            ),
        )

    @staticmethod
    def _request_metadata(context: grpc.aio.ServicerContext) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for item in context.invocation_metadata():
            try:
                key, value = item
            except (TypeError, ValueError):
                key, value = item.key, item.value
            normalized_key = str(key).lower()
            if normalized_key.startswith("openengine-") and normalized_key in metadata:
                raise ValueError(f"Duplicate reserved gRPC metadata key {normalized_key!r}")
            if isinstance(value, bytes):
                try:
                    normalized_value = value.decode("ascii")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"gRPC metadata value for {normalized_key!r} must be ASCII"
                    ) from error
            else:
                normalized_value = str(value)
            metadata[normalized_key] = normalized_value
        return metadata

    @staticmethod
    def _metadata_int(metadata: dict[str, str], key: str, minimum: int, maximum: int) -> int | None:
        if key not in metadata:
            return None
        value = metadata[key]
        digits = value[1:] if value.startswith("-") and minimum < 0 else value
        if not digits or not digits.isdecimal():
            raise ValueError(f"gRPC metadata {key!r} must be a base-10 integer")
        try:
            parsed = int(value, 10)
        except ValueError as error:
            raise ValueError(f"gRPC metadata {key!r} must be a base-10 integer") from error
        if not minimum <= parsed <= maximum:
            raise ValueError(f"gRPC metadata {key!r} must be in the range [{minimum}, {maximum}]")
        return parsed

    async def Generate(
        self, request: generation_pb2.GenerateRequest, context: grpc.aio.ServicerContext
    ) -> AsyncGenerator[generation_pb2.GenerateResponse, None]:
        result = None
        consumed_session_id = None
        produced_session = None
        admitted = False
        selected_lora = False
        selection_logged = False
        try:
            metadata = self._request_metadata(context)
            priority_value = self._metadata_int(
                metadata, "openengine-priority", -(1 << 31), (1 << 31) - 1
            )
            target_dp_rank = self._metadata_int(
                metadata, "openengine-target-dp-rank", 0, (1 << 32) - 1
            )
            if self.role == server_pb2.ENGINE_ROLE_DECODE and request.kv.HasField("session"):
                session_dp_rank = request.kv.session.dp_rank
                if target_dp_rank is not None and target_dp_rank != session_dp_rank:
                    raise ValueError(
                        "openengine-target-dp-rank does not match the KV session dp_rank"
                    )
                target_dp_rank = session_dp_rank
            self._validate_generate(request, target_dp_rank)
            params = to_sampling_params(request)
            media_items = list(request.media)
            media = await load_media(media_items, request.media_options, self.media_config)
            input_kind = request.WhichOneof("input")
            inputs: dict[str, Any]
            if input_kind == "prompt":
                inputs = {"prompt": request.prompt}
            else:
                inputs = {"prompt_token_ids": list(request.token_ids.ids)}
            if media:
                inputs["multi_modal_data"] = media
                uuids = media_uuids(media_items)
                if uuids is not None:
                    inputs["multi_modal_uuids"] = uuids
            lora_request = self._required_multimodal_lora(request)
            if lora_request is not None and request.lora_name:
                raise ValueError(
                    f"Modalities in this request require the model-owned "
                    f"{lora_request.lora_name!r} adapter and cannot be combined with "
                    "a user-selected LoRA"
                )
            if lora_request is None and request.lora_name:
                lora_request = await self.loras.request(request.lora_name)
            disaggregated = self._disaggregated_params(request, target_dp_rank, metadata)
            context_usage = (
                disaggregated.ctx_usage
                if disaggregated is not None and disaggregated.request_type == "generation_only"
                else None
            )
            scheduling = self._scheduling_params(target_dp_rank)
            priority = to_priority(priority_value)
            trace_headers = {
                key.lower(): value
                for key, value in metadata.items()
                if key.lower() in ("traceparent", "tracestate", "baggage")
            }
            result = self.llm.generate_async(
                inputs=inputs,
                sampling_params=params,
                lora_request=lora_request,
                streaming=True,
                disaggregated_params=disaggregated,
                scheduling_params=scheduling,
                cache_salt=(request.kv.cache_salt if request.kv.HasField("cache_salt") else None),
                trace_headers=trace_headers or None,
                priority=priority,
            )
            self.tracker.admit(request.request_id, result)
            admitted = True
            self._log_handoff(request, "admitted")
            if request.lora_name:
                selected_lora = True
            if self.stats is not None:
                self.stats.wake()
            if request.kv.HasField("session") and request.kv.session.session_id:
                consumed_session_id = request.kv.session.session_id
                self._track_kv_session(consumed_session_id, request.request_id)
        except KeyError as error:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, f"LoRA is not loaded: {error.args[0]}"
            )
            return
        except (ValueError, TypeError, RuntimeError) as error:
            if (
                result is not None
                and self.tracker.active_requests.get(request.request_id) is not result
            ):
                try:
                    result.abort()
                except (RuntimeError, AssertionError):
                    logger.warning(
                        "Failed to abort rejected duplicate request %s",
                        request.request_id,
                    )
            code = (
                grpc.StatusCode.UNAVAILABLE
                if self.tracker.draining
                else grpc.StatusCode.INVALID_ARGUMENT
            )
            await context.abort(code, str(error))
            return

        sent_tokens: dict[int, int] = {}
        sent_text: dict[int, int] = {}
        prompt_sent = False
        completed = False
        try:
            async for current in result:
                if context.cancelled():
                    await self.tracker.abort(request.request_id)
                    return
                if selected_lora and not selection_logged:
                    self._log_lora_selection(request, "selected")
                    selection_logged = True
                if not prompt_sent:
                    prompt = self._prompt_output(current)
                    if prompt is not None:
                        prompt_sent = True
                        yield generation_pb2.GenerateResponse(
                            request_id=request.request_id, prompt=prompt
                        )
                if self.role == server_pb2.ENGINE_ROLE_PREFILL:
                    if current.finished:
                        handoff = current.disaggregated_params
                        if handoff is None and current.outputs:
                            handoff = current.outputs[0].disaggregated_params
                        if handoff is None:
                            raise RuntimeError(
                                "Context-only result did not return disaggregated parameters"
                            )
                        handoff = replace(
                            handoff,
                            ctx_usage=self._usage_payload(current),
                            ctx_info_endpoint=(
                                handoff.ctx_info_endpoint or self._context_info_endpoint()
                            ),
                        )
                        session = encode_handoff(handoff, requires_decode_media=bool(request.media))
                        produced_session = session
                        self._track_kv_session(session.session_id, request.request_id)
                        self._log_handoff(request, "complete", session, current)
                        if selection_logged:
                            self._log_lora_selection(request, "complete", session)
                        yield generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            prefill_ready=generation_pb2.PrefillReady(kv_session=session),
                            usage=self._usage(current),
                        )
                        completed = True
                    continue
                for output in current.outputs:
                    token_start = sent_tokens.get(output.index, 0)
                    text_start = sent_text.get(output.index, 0)
                    token_ids = list(output.token_ids or [])
                    delta_ids = token_ids[token_start:]
                    delta_text = (output.text or "")[text_start:]
                    if delta_ids or delta_text:
                        logprobs = list(output.logprobs or [])[token_start:]
                        yield generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            token=generation_pb2.TokenOutput(
                                output_index=output.index,
                                tokens=self._token_infos(delta_ids, logprobs),
                                text=delta_text,
                            ),
                        )
                    sent_tokens[output.index] = len(token_ids)
                    sent_text[output.index] = len(output.text or "")
                if current.finished:
                    self._log_handoff(request, "complete", result=current)
                    if selection_logged:
                        self._log_lora_selection(request, "complete")
                    outputs = current.outputs
                    for index, output in enumerate(outputs):
                        response = generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            finished=self._finished(output),
                        )
                        if index == len(outputs) - 1:
                            response.usage.CopyFrom(self._usage(current, context_usage))
                        yield response
                    completed = True
        except asyncio.CancelledError:
            await self.tracker.abort(request.request_id)
            raise
        except (RuntimeError, ValueError, TypeError) as error:
            logger.error("OpenEngine request %s failed: %s", request.request_id, error)
            yield generation_pb2.GenerateResponse(
                request_id=request.request_id,
                error=error_pb2.EngineError(
                    code=error_pb2.ERROR_CODE_INTERNAL,
                    message=str(error),
                    retryable=False,
                ),
            )
        finally:
            if admitted and not completed:
                self._log_handoff(request, "aborted", produced_session)
                if selection_logged:
                    self._log_lora_selection(request, "aborted", produced_session)
            if consumed_session_id is not None:
                self._release_kv_session(consumed_session_id)
            if not completed and request.request_id in self.tracker.active_requests:
                await self.tracker.abort(request.request_id)
            else:
                await self.tracker.finish(request.request_id)
            if self.stats is not None:
                self.stats.wake()

    def _track_kv_session(self, session_id: str, request_id: str) -> None:
        self._release_kv_session(session_id)
        self._kv_session_requests[session_id] = request_id
        loop = asyncio.get_running_loop()
        self._kv_session_timers[session_id] = loop.call_later(
            self._kv_session_ttl_seconds, self._release_kv_session, session_id
        )

    def _release_kv_session(self, session_id: str) -> bool:
        released = self._kv_session_requests.pop(session_id, None) is not None
        timer = self._kv_session_timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()
        return released

    def _context_info_endpoint(self) -> str | None:
        """Return the context worker endpoint needed for direct NIXL discovery."""
        params = getattr(self.llm, "disaggregated_params", None)
        if not isinstance(params, dict):
            return None
        endpoint = params.get("ctx_info_endpoint")
        if isinstance(endpoint, str):
            return endpoint or None
        if isinstance(endpoint, (list, tuple)):
            return next(
                (item for item in endpoint if isinstance(item, str) and item),
                None,
            )
        return None

    def _release_all_kv_sessions(self) -> None:
        for timer in self._kv_session_timers.values():
            timer.cancel()
        self._kv_session_timers.clear()
        self._kv_session_requests.clear()

    def _active_kv_session_count(self) -> int:
        """Count sessions whose owning request is still engine-active.

        Prefill handoff handles remain abortable until their TTL expires, but
        their context request has already finished and must not inflate load.
        Drain continues to report and release all open handles separately.
        """
        return sum(
            request_id in self.tracker.active_requests
            for request_id in self._kv_session_requests.values()
        )

    def close(self) -> None:
        """Release protocol-owned timers without shutting down the shared LLM."""
        self._release_all_kv_sessions()

    def _validate_generate(
        self, request: generation_pb2.GenerateRequest, target_dp_rank: int | None = None
    ) -> None:
        if not request.request_id:
            raise ValueError("request_id must not be empty")
        if request.model and request.model not in self._accepted_model_names:
            raise ValueError(f"Unknown model {request.model!r}")
        if request.WhichOneof("input") is None:
            raise ValueError("Generate requires prompt or token_ids input")
        if request.request_id in self.tracker.active_requests:
            raise ValueError(f"Request {request.request_id!r} is already active")
        if self.tracker.draining:
            raise RuntimeError("TensorRT-LLM is draining")
        configured_backend = getattr(
            getattr(self.llm, "args", None), "guided_decoding_backend", None
        )
        if request.guided.backend and request.guided.backend != configured_backend:
            raise ValueError("Per-request guided decoding backend selection is not supported")
        if request.kv.HasField("bypass_prefix_cache") and request.kv.bypass_prefix_cache:
            raise ValueError("Prefix-cache bypass is not supported by TensorRT-LLM")
        if target_dp_rank is not None:
            self._scheduling_params(target_dp_rank)
        if request.media:
            processor = getattr(self.llm, "input_processor", None)
            aggregate = self._available_modalities(processor)
            prefill_decode = self._available_modalities(processor, prefill_decode=True)
            names = {
                input_pb2.MODALITY_UNSPECIFIED: "image",
                input_pb2.MODALITY_IMAGE: "image",
                input_pb2.MODALITY_VIDEO: "video",
                input_pb2.MODALITY_AUDIO: "audio",
            }
            allowed = (
                aggregate if self.role == server_pb2.ENGINE_ROLE_AGGREGATED else prefill_decode
            )
            unsupported = {
                names.get(item.modality, "unspecified")
                for item in request.media
                if names.get(item.modality) not in allowed
            }
            if unsupported:
                raise ValueError(
                    f"Media modalities are not supported for this role: {sorted(unsupported)}"
                )
            if isinstance(processor, BaseMultimodalInputProcessor):
                requested_modalities = tuple(
                    dict.fromkeys(names[item.modality] for item in request.media)
                )
                processor.get_required_lora_spec(requested_modalities)
        if self.role == server_pb2.ENGINE_ROLE_DECODE:
            if not request.kv.HasField("session"):
                raise ValueError("Decode requests require a prefill KV session")
            if request.kv.session.HasField("bootstrap"):
                raise ValueError("TensorRT-LLM does not support client-created KV bootstrap")
            requires_media = handoff_requires_decode_media(request.kv.session)
            if requires_media and not request.media:
                raise ValueError("Decode request must resend the ordered context-phase media")
            if request.media and not requires_media:
                raise ValueError("Decode KV session does not require raw media")
        elif self.role == server_pb2.ENGINE_ROLE_PREFILL:
            if request.kv.HasField("session"):
                raise ValueError("Prefill requests cannot consume a KV session")
        elif request.kv.HasField("session"):
            raise ValueError("Aggregated requests cannot consume a KV session")

    def _disaggregated_params(
        self,
        request: generation_pb2.GenerateRequest,
        target_dp_rank: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> DisaggregatedParams | None:
        if self.role == server_pb2.ENGINE_ROLE_AGGREGATED:
            return None
        if self.role == server_pb2.ENGINE_ROLE_PREFILL:
            return DisaggregatedParams(
                request_type="context_only",
                disagg_request_id=stable_request_id(request.request_id),
                ctx_dp_rank=target_dp_rank,
                schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
                conversation_id=(metadata or {}).get("conversation_id"),
            )
        return decode_handoff(request.kv.session)

    def _scheduling_params(self, target_dp_rank: int | None) -> SchedulingParams | None:
        if target_dp_rank is None:
            return None
        rank = target_dp_rank
        data_parallel_size = int(_arg(self.llm, "data_parallel_size", 1))
        if rank >= data_parallel_size:
            raise ValueError(
                f"data_parallel_rank {rank} is outside the configured DP size {data_parallel_size}"
            )
        attention_dp = bool(_arg(self.llm, "enable_attention_dp", False))
        if not attention_dp or getattr(self.llm, "_on_trt_backend", False):
            if rank == 0:
                return None
            raise ValueError(
                "A nonzero data_parallel_rank requires the PyTorch backend with attention DP "
                "enabled"
            )
        return SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False)

    def _required_multimodal_lora(
        self, request: generation_pb2.GenerateRequest
    ) -> LoRARequest | None:
        if not request.media:
            return None
        processor = getattr(self.llm, "input_processor", None)
        if not isinstance(processor, BaseMultimodalInputProcessor):
            return None
        names = {
            input_pb2.MODALITY_UNSPECIFIED: "image",
            input_pb2.MODALITY_IMAGE: "image",
            input_pb2.MODALITY_VIDEO: "video",
            input_pb2.MODALITY_AUDIO: "audio",
        }
        modalities = tuple(
            dict.fromkeys(names[item.modality] for item in request.media if item.modality in names)
        )
        spec = processor.get_required_lora_spec(modalities)
        if spec is None:
            return None
        if not self._supports_lora():
            raise ValueError(
                f"Modalities in this request require the model-owned {spec.name!r} adapter, "
                "but TensorRT-LLM was not configured with LoRA support"
            )
        self._lora_names_by_id[spec.adapter_id] = spec.name
        return LoRARequest(spec.name, spec.adapter_id, spec.path)

    def _available_modalities(
        self, processor: object, *, prefill_decode: bool = False
    ) -> tuple[str, ...]:
        if not isinstance(processor, BaseMultimodalInputProcessor):
            return ()
        declared = (
            processor.get_openengine_prefill_decode_modalities()
            if prefill_decode
            else processor.get_openengine_modalities()
        )
        available = []
        for modality in declared:
            spec = processor.get_required_lora_spec((modality,))
            if spec is not None and (not self._supports_lora() or not Path(spec.path).is_dir()):
                continue
            available.append(modality)
        return tuple(available)

    def _token_infos(
        self, token_ids: list[int], logprobs: list[Any]
    ) -> list[generation_pb2.TokenInfo]:
        infos: list[generation_pb2.TokenInfo] = []
        for index, token_id in enumerate(token_ids):
            info = generation_pb2.TokenInfo(
                token_id=token_id,
                token=_token_text(self.llm, token_id),
            )
            if index < len(logprobs):
                value = logprobs[index]
                if isinstance(value, dict):
                    sampled = value.get(token_id)
                    if sampled is not None:
                        info.logprob = sampled.logprob
                        if sampled.rank is not None:
                            info.rank = sampled.rank
                    for candidate_id, candidate in value.items():
                        candidate_proto = info.candidates.add(
                            token_id=candidate_id,
                            logprob=candidate.logprob,
                            token=_token_text(self.llm, candidate_id),
                        )
                        if candidate.rank is not None:
                            candidate_proto.rank = candidate.rank
                elif isinstance(value, (int, float)):
                    info.logprob = float(value)
            infos.append(info)
        return infos

    def _prompt_output(self, result: object) -> generation_pb2.PromptOutput | None:
        if not result.outputs:
            return None
        prompt_logprobs = result.outputs[0].prompt_logprobs
        if not prompt_logprobs:
            return None
        token_ids = list(result.prompt_token_ids)
        return generation_pb2.PromptOutput(
            tokens=self._token_infos(token_ids, list(prompt_logprobs))
        )

    @staticmethod
    def _usage(result: object, context_usage: dict[str, Any] | None = None) -> generation_pb2.Usage:
        prompt = len(result.prompt_token_ids)
        completion = sum(len(output.token_ids or []) for output in result.outputs)
        cached = getattr(result, "cached_tokens", None)
        if context_usage is not None:
            prompt = int(context_usage["prompt_tokens"])
            details = context_usage.get("prompt_tokens_details") or {}
            cached = int(details.get("cached_tokens", 0))
        usage = generation_pb2.Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )
        if cached is not None:
            usage.cached_prompt_tokens = cached
        return usage

    @classmethod
    def _usage_payload(cls, result: object) -> dict[str, Any]:
        usage = cls._usage(result)
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cached_prompt_tokens},
        }

    @staticmethod
    def _finished(output: object) -> generation_pb2.GenerationFinished:
        reason_map = {
            "stop": generation_pb2.FINISH_REASON_STOP,
            "length": generation_pb2.FINISH_REASON_LENGTH,
            "cancelled": generation_pb2.FINISH_REASON_CANCELLED,
            "timeout": generation_pb2.FINISH_REASON_CANCELLED,
        }
        finished = generation_pb2.GenerationFinished(
            output_index=output.index,
            reason=reason_map.get(output.finish_reason, generation_pb2.FINISH_REASON_STOP),
        )
        if isinstance(output.stop_reason, int):
            finished.stop_match.stop_token_id = output.stop_reason
        elif isinstance(output.stop_reason, str):
            finished.stop_match.stop_text = output.stop_reason
        return finished

    async def GetServerInfo(
        self, request: server_pb2.GetServerInfoRequest, context: grpc.aio.ServicerContext
    ) -> server_pb2.ServerInfo:
        del request, context
        tp = _arg(self.llm, "tensor_parallel_size", 1)
        pp = _arg(self.llm, "pipeline_parallel_size", 1)
        dp = _arg(self.llm, "data_parallel_size", 1)
        args = getattr(self.llm, "args", None)
        kv_capacity = self._kv_capacity()
        capacity = server_pb2.DeploymentCapacity()
        if kv_capacity.get("tokensPerBlock") is not None:
            capacity.kv_block_size = kv_capacity["tokensPerBlock"]
        if kv_capacity.get("maxNumBlocks") is not None:
            capacity.total_kv_blocks = kv_capacity["maxNumBlocks"]
        max_requests = getattr(args, "max_batch_size", None)
        if max_requests is not None:
            capacity.max_running_requests = max_requests
        max_tokens = getattr(args, "max_num_tokens", None)
        if max_tokens is not None:
            capacity.max_batched_tokens = max_tokens
        max_loras = getattr(getattr(args, "lora_config", None), "max_loras", None)
        if max_loras is not None:
            capacity.max_loras = max_loras
        return server_pb2.ServerInfo(
            engine_name="tensorrt_llm",
            engine_version=getattr(tensorrt_llm, "__version__", "unknown"),
            engine_role=self.role,
            instance_id=self.instance_id,
            supported_models=[self.model],
            parallelism=server_pb2.ParallelismInfo(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                data_parallel_size=dp,
                data_parallel_rank=_arg(self.llm, "data_parallel_rank", 0),
            ),
            kv_connector=self._kv_connector_info(),
            capacity=capacity,
            schema_revision=SCHEMA_REVISION,
            minimum_client_revision=MINIMUM_CLIENT_REVISION,
            schema_release=schema_release(),
        )

    async def GetModelInfo(
        self, request: model_pb2.GetModelInfoRequest, context: grpc.aio.ServicerContext
    ) -> model_pb2.ModelInfo:
        if request.model and request.model not in self._accepted_model_names:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model!r}")
            raise ValueError(f"Unknown model {request.model!r}")
        input_processor = getattr(self.llm, "input_processor", None)
        aggregate = self._available_modalities(input_processor)
        prefill_decode = self._available_modalities(input_processor, prefill_decode=True)
        modality_enum = {
            "image": input_pb2.MODALITY_IMAGE,
            "video": input_pb2.MODALITY_VIDEO,
            "audio": input_pb2.MODALITY_AUDIO,
        }
        args = getattr(self.llm, "args", None)
        guided_backend = getattr(args, "guided_decoding_backend", None)
        guided_modes = []
        if guided_backend is not None:
            guided_modes = [
                model_pb2.GUIDED_DECODING_MODE_JSON_SCHEMA,
                model_pb2.GUIDED_DECODING_MODE_REGEX,
                model_pb2.GUIDED_DECODING_MODE_EBNF_GRAMMAR,
                model_pb2.GUIDED_DECODING_MODE_CHOICE,
                model_pb2.GUIDED_DECODING_MODE_JSON_OBJECT,
            ]
            if guided_backend == "xgrammar":
                guided_modes.append(model_pb2.GUIDED_DECODING_MODE_STRUCTURAL_TAG)
        max_context = getattr(args, "max_seq_len", None) or getattr(args, "max_input_len", None)
        info = model_pb2.ModelInfo(
            model_id=self.model_id,
            served_model_name=self.model,
            served_model_aliases=self.model_aliases,
            tokenizer_modes=[self.tokenizer_mode],
            tokenizer=model_pb2.TokenizerInfo(
                source=self.tokenizer_source,
                mode=self.tokenizer_mode,
            ),
            supports_text_input=True,
            supports_token_ids_input=True,
            supports_lora=self._supports_lora(),
            supports_multimodal=bool(aggregate),
            reasoning_parser=self.reasoning_parser,
            tool_call_parser=self.tool_parser,
            generation=model_pb2.GenerationCapabilities(
                prompt_logprobs=model_pb2.LogprobCapabilities(
                    supported=True,
                    candidate_selection_modes=[model_pb2.CANDIDATE_TOKEN_SELECTION_MODE_TOP_N],
                    max_top_n=20,
                ),
                output_logprobs=model_pb2.LogprobCapabilities(
                    supported=True,
                    candidate_selection_modes=[model_pb2.CANDIDATE_TOKEN_SELECTION_MODE_TOP_N],
                    max_top_n=20,
                ),
                guided_decoding=model_pb2.GuidedDecodingCapabilities(
                    supported=guided_backend is not None,
                    modes=guided_modes,
                ),
                max_num_sequences=getattr(args, "max_beam_width", 1),
                supports_priority=True,
                supports_stop_in_output=True,
                supports_cache_salt=True,
                supports_prefix_cache_bypass=False,
            ),
            multimodal_capabilities=model_pb2.MultimodalCapabilities(
                aggregate_modalities=[modality_enum[name] for name in aggregate],
                prefill_decode_modalities=[modality_enum[name] for name in prefill_decode],
                source_types=[
                    input_pb2.MEDIA_SOURCE_TYPE_URL,
                    input_pb2.MEDIA_SOURCE_TYPE_DATA_URI,
                    input_pb2.MEDIA_SOURCE_TYPE_RAW_BYTES,
                ],
                supports_per_request_media_options=True,
            ),
        )
        routing_token_getter = getattr(
            input_processor, "get_openengine_routing_image_token_id", None
        )
        if "image" in aggregate and callable(routing_token_getter):
            routing_image_token_id = routing_token_getter()
            if routing_image_token_id is not None:
                info.multimodal_capabilities.routing_image_token_id = routing_image_token_id
        if max_context is not None:
            info.max_context_length = max_context
        return info

    def _kv_capacity(self) -> dict[str, int]:
        capacity: dict[str, int] = {}
        config = getattr(getattr(self.llm, "args", None), "kv_cache_config", None)
        tokens_per_block = getattr(config, "tokens_per_block", None)
        if tokens_per_block is not None:
            capacity["tokensPerBlock"] = int(tokens_per_block)
        getter = getattr(self.llm, "get_kv_cache_capacity", None)
        if callable(getter):
            try:
                discovered = getter()
            except (RuntimeError, AttributeError, TypeError) as error:
                logger.debug("KV capacity discovery unavailable: %s", error)
            else:
                if isinstance(discovered, dict):
                    for key in ("maxNumBlocks", "tokensPerBlock", "maxNumTokens"):
                        if discovered.get(key) is not None:
                            capacity[key] = int(discovered[key])
        return capacity

    async def GetLoad(
        self, request: observability_pb2.GetLoadRequest, context: grpc.aio.ServicerContext
    ) -> observability_pb2.LoadInfo:
        del context
        latest = self.stats.latest_by_rank() if self.stats is not None else {}
        rank_infos = []
        running_requests = 0
        queued_requests = 0
        used_kv_blocks = 0
        total_kv_blocks = 0
        prefill_batch_size = 0
        decode_batch_size = 0
        attributes = {"source": "shared_stats_fanout" if latest else "shared_request_tracker"}
        tokens_per_block: set[int] = set()
        for rank, stat in sorted(latest.items()):
            kv_stats = stat.get("kvCacheStats") or {}
            ifb_stats = stat.get("inflightBatchingStats") or {}
            rank_running = int(stat.get("numActiveRequests", 0))
            rank_queued = int(stat.get("numQueuedRequests", 0))
            rank_total = int(kv_stats.get("maxNumBlocks", 0))
            rank_used = int(
                kv_stats.get(
                    "usedNumBlocks",
                    max(0, rank_total - int(kv_stats.get("freeNumBlocks", rank_total))),
                )
            )
            rank_prefill = int(ifb_stats.get("numContextRequests", 0))
            rank_decode = int(ifb_stats.get("numGenRequests", 0))
            rank_tokens_per_block = kv_stats.get("tokensPerBlock")
            if rank_tokens_per_block is not None:
                rank_tokens_per_block = int(rank_tokens_per_block)
                tokens_per_block.add(rank_tokens_per_block)
                attributes[f"rank.{rank}.kv_tokens_per_block"] = str(rank_tokens_per_block)
            running_requests += rank_running
            queued_requests += rank_queued
            used_kv_blocks += rank_used
            total_kv_blocks += rank_total
            prefill_batch_size += rank_prefill
            decode_batch_size += rank_decode
            if request.include_per_rank:
                rank_infos.append(
                    observability_pb2.RankLoadInfo(
                        data_parallel_rank=rank,
                        running_requests=rank_running,
                        queued_requests=rank_queued,
                        used_kv_blocks=rank_used,
                        total_kv_blocks=rank_total,
                        prefill_batch_size=rank_prefill,
                        decode_batch_size=rank_decode,
                    )
                )

        capacity = self._kv_capacity()
        running_requests = max(running_requests, self.tracker.active_count)
        if not latest:
            total_kv_blocks = capacity.get("maxNumBlocks", 0)
            if capacity.get("tokensPerBlock") is not None:
                tokens_per_block.add(capacity["tokensPerBlock"])
        if len(tokens_per_block) == 1:
            attributes["kv_tokens_per_block"] = str(next(iter(tokens_per_block)))
        response = observability_pb2.LoadInfo(
            instance_id=self.instance_id,
            timestamp_unix_nanos=time.time_ns(),
            running_requests=running_requests,
            queued_requests=queued_requests,
            active_kv_sessions=self._active_kv_session_count(),
            ranks=rank_infos,
            attributes=attributes,
        )
        if used_kv_blocks or latest:
            response.used_kv_blocks = used_kv_blocks
        if total_kv_blocks:
            response.total_kv_blocks = total_kv_blocks
        if prefill_batch_size or latest:
            response.prefill_batch_size = prefill_batch_size
        if decode_batch_size or latest:
            response.decode_batch_size = decode_batch_size
        return response

    async def Health(
        self, request: lifecycle_pb2.HealthRequest, context: grpc.aio.ServicerContext
    ) -> lifecycle_pb2.HealthResponse:
        if request.model and request.model != self.model:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model!r}")
            raise ValueError(f"Unknown model {request.model!r}")
        if request.role not in (server_pb2.ENGINE_ROLE_UNSPECIFIED, self.role):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Requested health role does not match this engine",
            )
            raise ValueError("Requested health role does not match this engine")
        if request.include_inference_probe:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Role-safe inference probes are not implemented by this server",
            )
            raise NotImplementedError("Role-safe inference probes are not implemented")
        healthy, message = await self.tracker.health()
        state = (
            lifecycle_pb2.HEALTH_STATE_DRAINING
            if self.tracker.draining
            else lifecycle_pb2.HEALTH_STATE_READY
            if healthy
            else lifecycle_pb2.HEALTH_STATE_NOT_READY
        )
        checks = [lifecycle_pb2.HealthCheck(name="scheduler", state=state, message=message)]
        return lifecycle_pb2.HealthResponse(state=state, checks=checks)

    async def Abort(
        self, request: lifecycle_pb2.AbortRequest, context: grpc.aio.ServicerContext
    ) -> lifecycle_pb2.AbortResponse:
        target = request.WhichOneof("target")
        if target is None:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Abort target must be specified")
            raise ValueError("Abort target must be specified")
        if target == "all_requests":
            count = await self.tracker.abort_all()
            self._release_all_kv_sessions()
            return lifecycle_pb2.AbortResponse(
                status=lifecycle_pb2.ABORT_STATUS_ABORTED, message=f"Aborted {count} requests"
            )
        request_id = request.request_id
        released_session = False
        if target == "kv_session":
            session_id = request.kv_session.session_id
            request_id = self._kv_session_requests.get(session_id, session_id)
            released_session = self._release_kv_session(session_id)
        aborted = await self.tracker.abort(request_id)
        return lifecycle_pb2.AbortResponse(
            status=(
                lifecycle_pb2.ABORT_STATUS_ABORTED
                if aborted or released_session
                else lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED
            ),
            message=(
                f"Aborted {request_id}"
                if aborted
                else f"Released KV session {request.kv_session.session_id}"
                if released_session
                else f"Request {request_id} is not active"
            ),
        )

    async def Drain(
        self, request: lifecycle_pb2.DrainRequest, context: grpc.aio.ServicerContext
    ) -> AsyncGenerator[lifecycle_pb2.DrainResponse, None]:
        del context
        if request.stop_accepting_new_requests:
            await self.tracker.start_drain()
        yield lifecycle_pb2.DrainResponse(
            state=lifecycle_pb2.DRAIN_STATE_STARTED,
            in_flight_requests=self.tracker.active_count,
            open_kv_sessions=len(self._kv_session_requests),
            message="Process-wide drain started",
        )
        timeout = request.deadline_ms / 1000.0 if request.HasField("deadline_ms") else None
        empty = await self.tracker.wait_empty(timeout)
        if not empty and request.abort_after_deadline:
            await self.tracker.abort_all()
            empty = await self.tracker.wait_empty(self._post_abort_cleanup_timeout_seconds)
        if not empty:
            yield lifecycle_pb2.DrainResponse(
                error=error_pb2.EngineError(
                    code=error_pb2.ERROR_CODE_INTERNAL,
                    message="Drain deadline expired with active requests",
                    retryable=False,
                ),
                in_flight_requests=self.tracker.active_count,
                open_kv_sessions=len(self._kv_session_requests),
            )
            return
        self._release_all_kv_sessions()
        yield lifecycle_pb2.DrainResponse(
            state=lifecycle_pb2.DRAIN_STATE_COMPLETE,
            in_flight_requests=0,
            open_kv_sessions=0,
            message="Drain complete",
        )

    async def LoadLora(
        self, request: lora_pb2.LoadLoraRequest, context: grpc.aio.ServicerContext
    ) -> lora_pb2.LoadLoraResponse:
        if not self._supports_lora():
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "TensorRT-LLM was not configured with LoRA support",
            )
            raise RuntimeError("TensorRT-LLM was not configured with LoRA support")
        try:
            adapter, already_loaded = await self.loras.load(request.adapter)
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise
        self._lora_names_by_id[adapter.lora_id] = adapter.lora_name
        return lora_pb2.LoadLoraResponse(adapter=adapter, already_loaded=already_loaded)

    async def UnloadLora(
        self, request: lora_pb2.UnloadLoraRequest, context: grpc.aio.ServicerContext
    ) -> lora_pb2.UnloadLoraResponse:
        if not self._supports_lora():
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "TensorRT-LLM was not configured with LoRA support",
            )
            raise RuntimeError("TensorRT-LLM was not configured with LoRA support")
        try:
            adapter = await self.loras.unload(request.lora_name)
        except KeyError:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"LoRA {request.lora_name!r} is not loaded"
            )
            raise
        return lora_pb2.UnloadLoraResponse(adapter=adapter)

    async def ListLoras(
        self, request: lora_pb2.ListLorasRequest, context: grpc.aio.ServicerContext
    ) -> lora_pb2.ListLorasResponse:
        del request
        if not self._supports_lora():
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "TensorRT-LLM was not configured with LoRA support",
            )
            raise RuntimeError("TensorRT-LLM was not configured with LoRA support")
        return lora_pb2.ListLorasResponse(adapters=await self.loras.list())

    def _supports_lora(self) -> bool:
        args = getattr(self.llm, "args", None)
        if getattr(args, "lora_config", None) is not None or bool(
            getattr(args, "enable_lora", False)
        ):
            return True
        build_config = getattr(args, "build_config", None)
        plugin_config = getattr(build_config, "plugin_config", None)
        return bool(getattr(plugin_config, "lora_plugin", False))

    def _kv_connector_info(self) -> kv_pb2.KvConnectorInfo:
        enabled = self.role in (server_pb2.ENGINE_ROLE_PREFILL, server_pb2.ENGINE_ROLE_DECODE)
        return kv_pb2.KvConnectorInfo(
            enabled=enabled,
            transfer_backend="tensorrt_llm" if enabled else "",
            supported_protocols=["nixl"] if enabled else [],
            supports_remote_prefill=enabled,
            supports_decode_pull=enabled,
            supports_abort_cleanup=enabled,
            supports_drain=enabled,
            schema_version=1,
            handoff_profile=HANDOFF_ATTRIBUTE,
            supports_client_bootstrap=False,
        )

    def _kv_events_enabled(self) -> bool:
        args = getattr(self.llm, "args", None)
        config = getattr(args, "kv_cache_config", None)
        if self.kv_events is None or getattr(config, "event_buffer_max_size", 0) <= 0:
            return False
        effective_hash_algo = get_effective_kv_cache_event_hash_algo(
            getattr(config, "kv_cache_event_hash_algo", "auto"),
            bool(getattr(config, "use_kv_cache_manager_v2", False)),
        )
        return effective_hash_algo in (KV_CACHE_HASH_ALGO_V1, KV_CACHE_HASH_ALGO_V2_SHA256_64)

    async def GetKvConnectorInfo(
        self, request: kv_pb2.GetKvConnectorInfoRequest, context: grpc.aio.ServicerContext
    ) -> kv_pb2.KvConnectorInfo:
        del request, context
        return self._kv_connector_info()

    async def GetKvEventSources(
        self, request: kv_pb2.GetKvEventSourcesRequest, context: grpc.aio.ServicerContext
    ) -> kv_pb2.GetKvEventSourcesResponse:
        if not self._kv_events_enabled():
            return kv_pb2.GetKvEventSourcesResponse()
        data_parallel_size = int(_arg(self.llm, "data_parallel_size", 1))
        available_ranks = set(range(data_parallel_size))
        requested_ranks = set(request.data_parallel_ranks)
        invalid_ranks = requested_ranks - available_ranks
        if invalid_ranks:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unknown data-parallel ranks: {sorted(invalid_ranks)}",
            )
            return kv_pb2.GetKvEventSourcesResponse()
        selected_ranks = sorted(requested_ranks or available_ranks)
        return kv_pb2.GetKvEventSourcesResponse(
            sources=[
                kv_pb2.KvEventSource(
                    transport="grpc",
                    endpoint_addr=kv_pb2.KvEndpoint(
                        host=self.event_host, port=self.event_port, protocol="grpc"
                    ),
                    data_parallel_rank=rank,
                    encoding="protobuf",
                    schema_version=1,
                    max_queue_size=self.kv_events.buffer_size,
                )
                for rank in selected_ranks
            ]
        )

    async def SubscribeKvEvents(
        self, request: kv_pb2.SubscribeKvEventsRequest, context: grpc.aio.ServicerContext
    ) -> AsyncGenerator[kv_pb2.SubscribeKvEventsResponse, None]:
        if not self._kv_events_enabled():
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "KV event collection is disabled"
            )
            return
        if request.include_snapshot or request.start_sequence_number:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "KV event snapshot and replay subscriptions are not implemented",
            )
            return
        selected_ranks = set(request.data_parallel_ranks)
        available_ranks = set(range(int(_arg(self.llm, "data_parallel_size", 1))))
        invalid_ranks = selected_ranks - available_ranks
        if invalid_ranks:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unknown data-parallel ranks: {sorted(invalid_ranks)}",
            )
            return
        async for sequence, raw in self.kv_events.subscribe(selected_ranks):
            if context.cancelled():
                return
            rank = raw.get("attention_dp_rank", 0)
            if selected_ranks and rank not in selected_ranks:
                continue
            yield kv_pb2.SubscribeKvEventsResponse(batch=self._kv_batch(raw, sequence))

    def _kv_batch(self, raw: dict[str, Any], sequence: int) -> kv_pb2.KvEventBatch:
        rank = raw.get("attention_dp_rank", 0)
        group_idx = raw.get("layer_group_id", 0)
        block_size = self._kv_capacity().get("tokensPerBlock")
        hash_algo = raw.get("hash_algo")
        data = raw.get("data", raw)
        event_type = data.get("type")
        events: list[kv_pb2.KvEvent] = []
        if event_type == "stored":
            if block_size is None or block_size <= 0:
                raise ValueError("Cannot bridge KV events without a configured tokens_per_block")
            parent = data.get("parent_hash")
            blocks = data.get("blocks", [])
            complete_blocks = []
            for block in blocks:
                block_hash = block.get("block_hash")
                tokens = block.get("tokens", [])
                if len(tokens) > block_size:
                    raise ValueError("TRT-LLM KV event block exceeds tokens_per_block")
                if len(tokens) < block_size:
                    self._partial_block_hashes.setdefault(rank, set()).add(block_hash)
                    break
                if block.get("cache_salt") not in (None, ""):
                    return self._kv_fail_closed_batch(
                        rank,
                        sequence,
                        "Salted TRT-LLM KV event cannot be represented by OpenEngine",
                    )
                if any(token.get("token_extra_id", 0) for token in tokens):
                    return self._kv_fail_closed_batch(
                        rank,
                        sequence,
                        "Eagle TRT-LLM KV event cannot be represented by OpenEngine",
                    )
                token_ids = [token.get("token_id") for token in tokens]
                if any(
                    not isinstance(token_id, int) or not 0 <= token_id < (1 << 32)
                    for token_id in token_ids
                ):
                    return self._kv_fail_closed_batch(
                        rank,
                        sequence,
                        "Non-text TRT-LLM KV event contains non-integer tokens",
                    )
                complete_blocks.append(block)
            if complete_blocks:
                lora_ids = {
                    int(block["lora_id"])
                    for block in complete_blocks
                    if block.get("lora_id") is not None
                }
                lora_names = {
                    str(block.get("lora_name") or data.get("lora_name"))
                    for block in complete_blocks
                    if block.get("lora_name") or data.get("lora_name")
                }
                if len(lora_ids) > 1 or len(lora_names) > 1:
                    return self._kv_fail_closed_batch(
                        rank,
                        sequence,
                        "TRT-LLM KV event mixes LoRA identities",
                    )
                lora_id = next(iter(lora_ids), None)
                lora_name = next(
                    iter(lora_names),
                    self._lora_names_by_id.get(lora_id, "") if lora_id is not None else "",
                )
                mm_keys = []
                for block_index, block in enumerate(complete_blocks):
                    for key in block.get("mm_keys", []):
                        mm_hash = key.get("hash")
                        if (
                            key.get("type") != "mm_key"
                            or not isinstance(mm_hash, str)
                            or not mm_hash
                        ):
                            return self._kv_fail_closed_batch(
                                rank,
                                sequence,
                                "TRT-LLM KV event contains malformed multimodal metadata",
                            )
                        try:
                            mm_hash_prefix = str(int(mm_hash[:16], 16))
                        except ValueError:
                            return self._kv_fail_closed_batch(
                                rank,
                                sequence,
                                "TRT-LLM KV event contains a non-hex multimodal hash",
                            )
                        mm_keys.append(
                            kv_pb2.OpaqueKeyTuple(
                                values=[
                                    "trt_mm_v1",
                                    str(block_index),
                                    mm_hash_prefix,
                                    str(key.get("start_offset", 0)),
                                ]
                            )
                        )
                stored = kv_pb2.BlockStored(
                    block_hashes=[
                        _block_hash(block.get("block_hash"), hash_algo) for block in complete_blocks
                    ],
                    token_ids=[
                        token["token_id"]
                        for block in complete_blocks
                        for token in block.get("tokens", [])
                    ],
                    block_size=block_size,
                    lora_id=0 if lora_id is None else int(lora_id),
                    lora_name=lora_name,
                    medium=kv_pb2.STORAGE_MEDIUM_GPU,
                    extra_keys=mm_keys,
                    group_idx=group_idx,
                )
                if parent is not None:
                    stored.parent_block_hash.CopyFrom(_block_hash(parent, hash_algo))
                events.append(kv_pb2.KvEvent(block_stored=stored))
            else:
                stored = kv_pb2.BlockStored(
                    block_size=block_size,
                    medium=kv_pb2.STORAGE_MEDIUM_GPU,
                    group_idx=group_idx,
                )
                if parent is not None:
                    stored.parent_block_hash.CopyFrom(_block_hash(parent, hash_algo))
                events.append(kv_pb2.KvEvent(block_stored=stored))
        elif event_type == "removed":
            partial_hashes = self._partial_block_hashes.setdefault(rank, set())
            removed_hashes = []
            for value in data.get("block_hashes", []):
                if value in partial_hashes:
                    partial_hashes.discard(value)
                else:
                    removed_hashes.append(value)
            events.append(
                kv_pb2.KvEvent(
                    block_removed=kv_pb2.BlockRemoved(
                        block_hashes=[_block_hash(value, hash_algo) for value in removed_hashes],
                        medium=kv_pb2.STORAGE_MEDIUM_GPU,
                        group_idx=group_idx,
                    )
                )
            )
        elif event_type == "all_cleared":
            self._partial_block_hashes.pop(rank, None)
            events.append(kv_pb2.KvEvent(all_blocks_cleared=kv_pb2.AllBlocksCleared()))
        return kv_pb2.KvEventBatch(
            sequence_number=sequence,
            timestamp_unix_nanos=time.time_ns(),
            data_parallel_rank=rank,
            events=events,
        )

    @staticmethod
    def _kv_fail_closed_batch(rank: int, sequence: int, message: str) -> kv_pb2.KvEventBatch:
        logger.error("%s; clearing the advertised KV index", message)
        return kv_pb2.KvEventBatch(
            sequence_number=sequence,
            timestamp_unix_nanos=time.time_ns(),
            data_parallel_rank=rank,
            events=[kv_pb2.KvEvent(all_blocks_cleared=kv_pb2.AllBlocksCleared())],
        )
