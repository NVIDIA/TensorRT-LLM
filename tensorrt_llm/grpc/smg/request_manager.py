# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Request manager for the TensorRT-LLM SMG gRPC adapter.

Manages request lifecycle for gRPC requests, converting between protobuf
and TensorRT-LLM types. Designed for high-performance communication with
external routers (e.g., sgl-router) using pre-tokenized input.

Key optimization: Sets detokenize=False in SamplingParams to skip
detokenization and return token IDs only.

The manager also keeps the engine's load record for the router (see ``load_tracker`` for the
rules): what it forwarded and what came back, and, while the engine emits iteration statistics
(``enable_iter_perf_stats``), the engine's own counts and KV usage.
"""

import asyncio
import time
import traceback
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.request import LoRARequest, PromptAdapterRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.llmapi.llm_utils import KvCacheRetentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

from .bindings import trtllm_service_pb2 as pb2
from .load_tracker import (
    STATS_FRESH,
    EngineStats,
    EngineStatsFold,
    LoadRecord,
    LoadTracker,
    compose_record,
    prefill_sample,
)

#: How long one blocking read of the engine's iteration statistics waits for a record.
_STATS_READ_TIMEOUT_S = 0.5


class GrpcRequestManager:
    """Manages gRPC request lifecycle for TensorRT-LLM.

    Responsibilities:
    - Convert protobuf requests to TensorRT-LLM types
    - Set detokenize=False in SamplingParams (key optimization!)
    - Submit requests to LLM.generate_async()
    - Stream token IDs (not text) back to gRPC clients
    - Handle abort/cancel operations

    This is modeled after vLLM's GrpcRequestManager but adapted for TensorRT-LLM's
    GenerationResult async iterator pattern.
    """

    def __init__(self, llm: Any):
        """Initialize the request manager.

        Args:
            llm: The TensorRT-LLM LLM instance (tensorrt_llm.LLM or tensorrt_llm._tensorrt_engine.LLM)
        """
        self.llm = llm
        # Track active requests: request_id -> GenerationResult
        self._rid_to_result: Dict[str, GenerationResult] = {}
        # The load record's sources: the requests forwarded and the engine's iteration statistics.
        self._loads = LoadTracker()
        self._engine_stats = EngineStatsFold()
        self._stats_task: Optional[asyncio.Task] = None

        logger.info("GrpcRequestManager initialized")

    async def generate(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        streaming: bool = True,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
        multi_modal_data: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[GenerationResult, None]:
        """Submit a generation request and stream outputs.

        Args:
            request_id: Unique request identifier (for tracking/abort)
            prompt_token_ids: Pre-tokenized input from Rust router
            sampling_params: Sampling parameters (with detokenize=False!)
            streaming: Whether to stream results
            lora_request: Optional LoRA adapter request
            prompt_adapter_request: Deprecated; prompt adapters are unsupported. Omit or pass None.
            kv_cache_retention_config: KV cache retention config
            disaggregated_params: Disaggregated inference params
            multi_modal_data: Multimodal data dict (e.g. {"image": [PIL images]})

        Yields:
            GenerationResult objects containing token IDs (text will be empty
            because detokenize=False)
        """
        generation_only = (
            disaggregated_params is not None
            and getattr(disaggregated_params, "request_type", None) == "generation_only"
        )
        started = False
        sampled = False
        streamed = 0
        try:
            # Submit to LLM.generate_async which returns a GenerationResult
            # that is an async iterator
            inputs = {"prompt_token_ids": prompt_token_ids}
            if multi_modal_data:
                inputs["multi_modal_data"] = multi_modal_data

            self._loads.forwarded(request_id, len(prompt_token_ids))
            gen_result = self.llm.generate_async(
                inputs,
                sampling_params,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                streaming=streaming,
                kv_cache_retention_config=kv_cache_retention_config,
                disaggregated_params=disaggregated_params,
            )

            # Track the result for potential abort
            self._rid_to_result[request_id] = gen_result

            # Iterate over the async generator
            # GenerationResult implements __aiter__ and __anext__
            async for result in gen_result:
                if not started:
                    started = True
                    self._loads.started(request_id, _cached_tokens(result))
                total = _generated_tokens(result)
                if total > streamed:
                    self._loads.generated(total - streamed)
                    streamed = total
                if not sampled:
                    sample = _prefill_sample_of(result, len(prompt_token_ids), generation_only)
                    if sample is not None:
                        sampled = True
                        self._loads.prefill_computed(*sample)
                yield result

                if result.finished:
                    break

        except asyncio.CancelledError:
            logger.info(f"Request {request_id} cancelled by client")
            await self.abort(request_id)
            raise
        except Exception as e:
            logger.error(f"Error in generate for {request_id}: {e}")
            raise
        finally:
            # Cleanup tracking
            self._rid_to_result.pop(request_id, None)
            self._loads.finished(request_id)

    async def abort(self, request_id: str) -> bool:
        """Abort a running request.

        Args:
            request_id: The request ID to abort

        Returns:
            True if request was found and aborted, False otherwise
        """
        gen_result = self._rid_to_result.get(request_id)

        if gen_result is None:
            logger.debug(f"Abort: request {request_id} not found (may have already completed)")
            return False

        try:
            # GenerationResult has an abort() method
            gen_result.abort()
            self._rid_to_result.pop(request_id, None)
            logger.info(f"Request {request_id} aborted")
            return True
        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            self._rid_to_result.pop(request_id, None)
            return False

    async def health_check(self) -> Tuple[bool, str]:
        """Check if the engine is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            if self.llm is None:
                return False, "LLM not initialized"

            # Check executor health (includes error queue, MPI worker
            # liveness, and fatal error state — not just doing_shutdown).
            if hasattr(self.llm, "_executor"):
                if self.llm._executor is None:
                    return False, "Executor is not available"
                if not self.llm._executor.check_health():
                    error_msg = "Executor is unhealthy"
                    if self.llm._executor._fatal_error is not None:
                        exc = self.llm._executor._fatal_error
                        lines = str(exc).splitlines()
                        short = (lines[0] if lines else type(exc).__name__)[:200]
                        error_msg = f"{type(exc).__name__}: {short}"
                        logger.error(f"Health check fatal error: {repr(exc)}")
                    return False, error_msg

            return True, "OK"
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False, f"Error: {e}"

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration information.

        Returns:
            Dictionary with model config details
        """
        config = {
            "model_path": "",
            "is_generation": True,
            "max_context_length": 0,
            "max_seq_len": 0,
            "vocab_size": 0,
            "supports_vision": False,
        }

        try:
            # Try to get model path
            if hasattr(self.llm, "args"):
                if hasattr(self.llm.args, "model"):
                    config["model_path"] = str(self.llm.args.model)

            # Try to get tokenizer info
            if hasattr(self.llm, "tokenizer") and self.llm.tokenizer is not None:
                # TransformersTokenizer wraps the HF tokenizer without
                # overriding vocab_size, so the wrapper attribute resolves to
                # PreTrainedTokenizerBase's abstract property (raises
                # NotImplementedError). Read from the inner tokenizer.
                tokenizer = self.llm.tokenizer
                inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
                try:
                    vocab_size = inner_tokenizer.vocab_size
                except (AttributeError, NotImplementedError):
                    vocab_size = None
                if vocab_size:
                    config["vocab_size"] = int(vocab_size)

            # Try to get max context length from various sources
            if hasattr(self.llm, "args") and self.llm.args is not None:
                args = self.llm.args
                # Try max_input_len first (input context)
                if hasattr(args, "max_input_len") and args.max_input_len:
                    config["max_context_length"] = args.max_input_len
                # Try max_seq_len (total sequence including output)
                if hasattr(args, "max_seq_len") and args.max_seq_len:
                    config["max_seq_len"] = args.max_seq_len

            # Check for multimodal support
            if hasattr(self.llm, "input_processor"):
                processor_name = type(self.llm.input_processor).__name__
                config["supports_vision"] = processor_name != "DefaultInputProcessor"

        except Exception as e:
            logger.warning(
                f"Error getting model config: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            )

        return config

    def get_num_unfinished_requests(self) -> int:
        """Get the number of currently running requests.

        Returns:
            Number of unfinished requests
        """
        return len(self._rid_to_result)

    # ========== The load record ==========

    def load_record(self) -> LoadRecord:
        """The engine's load as the router reads it now (see ``load_tracker.compose_record``)."""
        now = time.monotonic()
        engine = self._engine_stats.current()
        fresh = engine is not None and now - engine.at <= STATS_FRESH
        estimate = self._loads.estimate(engine.waiting if fresh else None)
        return compose_record(
            estimate,
            engine,
            self._engine_stats.kv(),
            now=now,
            max_running_fallback=self._max_batch_size(),
        )

    def start_load_feed(self) -> None:
        """Start reading the engine's iteration statistics for the load record.

        Only when the engine emits them (``enable_iter_perf_stats``); without them the record
        carries this manager's own counts and no KV usage. Called on the serving event loop.
        """
        if self._stats_task is not None:
            return
        if not self._engine_emits_stats():
            logger.info(
                "Load record: the engine emits no iteration statistics "
                "(enable_iter_perf_stats is off); reporting the forwarded requests only"
            )
            return
        self._stats_task = asyncio.get_running_loop().create_task(self._collect_engine_stats())
        logger.info("Load record: reading the engine's iteration statistics")

    def stop_load_feed(self) -> None:
        """Stop reading the engine's iteration statistics."""
        if self._stats_task is not None:
            self._stats_task.cancel()
            self._stats_task = None

    def _engine_emits_stats(self) -> bool:
        args = getattr(self.llm, "args", None)
        return bool(getattr(args, "enable_iter_perf_stats", False)) and callable(
            getattr(self.llm, "get_stats", None)
        )

    def _max_batch_size(self) -> int:
        args = getattr(self.llm, "args", None)
        window = getattr(args, "max_batch_size", 0)
        return window if isinstance(window, int) and window > 0 else 0

    async def _collect_engine_stats(self) -> None:
        """Drain the engine's iteration-statistics queue on a worker thread.

        This task is the queue's one consumer; a read blocks up to ``_STATS_READ_TIMEOUT_S`` while
        the engine is idle.
        """
        failures = 0
        while True:
            try:
                records = await asyncio.to_thread(self.llm.get_stats, _STATS_READ_TIMEOUT_S)
                now = time.monotonic()
                for record in records:
                    stats = EngineStats.from_iteration(record, now)
                    if stats is not None:
                        self._engine_stats.observe(stats)
                failures = 0
                if not records:
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                failures += 1
                if failures == 1:
                    logger.warning(f"Load record: reading iteration statistics failed: {e}")
                if failures >= 3:
                    logger.warning(
                        "Load record: iteration statistics unavailable; "
                        "reporting the forwarded requests only"
                    )
                    return
                await asyncio.sleep(_STATS_READ_TIMEOUT_S)


def _cached_tokens(result: Any) -> int:
    cached = getattr(result, "cached_tokens", 0)
    return cached if isinstance(cached, int) else 0


def _generated_tokens(result: Any) -> int:
    """The tokens generated so far across the result's sequences (cumulative)."""
    total = 0
    for completion in getattr(result, "outputs", None) or ():
        total += len(getattr(completion, "token_ids", None) or ())
    return total


def _steady_seconds(value: Any) -> Optional[float]:
    """A ``RequestPerfMetrics`` time point as seconds.

    A ``timedelta`` on the engine's steady clock, or a number; ``None`` for anything else.
    """
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return float(total_seconds())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _prefill_sample_of(
    result: Any, prompt_tokens: int, generation_only: bool
) -> Optional[tuple[int, float, float]]:
    """The capacity sample a result contributes (``load_tracker.prefill_sample``).

    Read from the per-request performance metrics the engine attaches when
    ``return_perf_metrics`` is set; ``None`` when the result carries none yet or must not
    calibrate.
    """
    for completion in getattr(result, "outputs", None) or ():
        metrics = getattr(completion, "request_perf_metrics", None)
        timing = getattr(metrics, "timing_metrics", None)
        if timing is None:
            continue
        aborted = getattr(result, "aborted", None)
        cancelled = (
            getattr(completion, "finish_reason", None) == "cancelled"
            or bool(getattr(result, "error_msg", None))
            or (callable(aborted) and bool(aborted()))
        )
        return prefill_sample(
            prompt_tokens,
            _cached_tokens(result),
            _steady_seconds(getattr(timing, "first_scheduled_time", None)),
            _steady_seconds(getattr(timing, "first_token_time", None)),
            cancelled=cancelled,
            generation_only=generation_only,
        )
    return None


def create_sampling_params_from_proto(
    proto_config: pb2.SamplingConfig,
    output_config: pb2.OutputConfig,
    max_tokens: int,
    stop: Optional[List[str]] = None,
    stop_token_ids: Optional[List[int]] = None,
    ignore_eos: bool = False,
    bad: Optional[List[str]] = None,
    bad_token_ids: Optional[List[int]] = None,
    guided_decoding: Optional[pb2.GuidedDecodingParams] = None,
    embedding_bias: Optional[Sequence[float]] = None,
    vocab_size: Optional[int] = None,
    include_stop_token_in_output: bool = False,
) -> SamplingParams:
    """Convert protobuf configuration to TensorRT-LLM SamplingParams.

    Args:
        proto_config: Protobuf SamplingConfig message
        output_config: Protobuf OutputConfig message
        max_tokens: Maximum tokens to generate
        stop: Stop strings (tokenized by TRT-LLM's _setup())
        stop_token_ids: Stop token IDs
        ignore_eos: Whether to ignore end-of-sequence token
        bad: Bad word strings (tokenized by TRT-LLM's _setup())
        bad_token_ids: Bad word token IDs
        guided_decoding: Guided decoding parameters
        embedding_bias: Embedding bias tensor
        vocab_size: Vocabulary size used to validate embedding bias length

    Returns:
        TensorRT-LLM SamplingParams with detokenize=False
    """
    # Build kwargs for SamplingParams
    # KEY OPTIMIZATION: detokenize=False skips Python detokenization!
    kwargs = {
        "max_tokens": max_tokens,
        "detokenize": False,
    }

    # Beam search / sampling
    if proto_config.beam_width > 1:
        kwargs["use_beam_search"] = True
        kwargs["best_of"] = proto_config.beam_width
    if proto_config.num_return_sequences > 0:
        kwargs["n"] = proto_config.num_return_sequences

    # Temperature and sampling parameters (with sensible defaults as safety guard)
    kwargs["temperature"] = (
        proto_config.temperature if proto_config.HasField("temperature") else 1.0
    )
    kwargs["top_p"] = proto_config.top_p if proto_config.HasField("top_p") else 1.0
    if proto_config.HasField("top_k"):
        kwargs["top_k"] = proto_config.top_k
    if proto_config.HasField("min_p"):
        kwargs["min_p"] = proto_config.min_p

    # Top-P decay parameters
    if proto_config.HasField("top_p_min"):
        kwargs["top_p_min"] = proto_config.top_p_min
    if proto_config.HasField("top_p_reset_ids"):
        kwargs["top_p_reset_ids"] = proto_config.top_p_reset_ids
    if proto_config.HasField("top_p_decay"):
        kwargs["top_p_decay"] = proto_config.top_p_decay

    # Seed for reproducibility
    if proto_config.HasField("seed"):
        kwargs["seed"] = proto_config.seed

    # Min/max tokens
    if proto_config.HasField("min_tokens"):
        kwargs["min_tokens"] = proto_config.min_tokens

    # Penalties (repetition_penalty defaults to 1.0 = no penalty)
    kwargs["repetition_penalty"] = (
        proto_config.repetition_penalty if proto_config.HasField("repetition_penalty") else 1.0
    )
    if proto_config.HasField("presence_penalty"):
        kwargs["presence_penalty"] = proto_config.presence_penalty
    if proto_config.HasField("frequency_penalty"):
        kwargs["frequency_penalty"] = proto_config.frequency_penalty

    # Beam search parameters
    if proto_config.HasField("beam_search_diversity_rate"):
        kwargs["beam_search_diversity_rate"] = proto_config.beam_search_diversity_rate
    if proto_config.HasField("length_penalty"):
        kwargs["length_penalty"] = proto_config.length_penalty
    if proto_config.HasField("early_stopping"):
        kwargs["early_stopping"] = proto_config.early_stopping

    # N-gram blocking
    if proto_config.HasField("no_repeat_ngram_size"):
        kwargs["no_repeat_ngram_size"] = proto_config.no_repeat_ngram_size

    # Stop sequences and ignore_eos (TRT-LLM's _setup() tokenizes stop strings)
    if stop:
        kwargs["stop"] = stop
    if stop_token_ids:
        kwargs["stop_token_ids"] = stop_token_ids
    if ignore_eos:
        kwargs["ignore_eos"] = True
    if include_stop_token_in_output:
        kwargs["include_stop_str_in_output"] = True

    # Bad words (TRT-LLM's _setup() tokenizes bad word strings)
    if bad:
        kwargs["bad"] = bad
    if bad_token_ids:
        kwargs["bad_token_ids"] = bad_token_ids

    # Output configuration - logprobs
    if output_config.HasField("logprobs"):
        kwargs["logprobs"] = output_config.logprobs
    if output_config.HasField("prompt_logprobs"):
        kwargs["prompt_logprobs"] = output_config.prompt_logprobs
    if output_config.return_context_logits:
        kwargs["return_context_logits"] = True
    if output_config.return_generation_logits:
        kwargs["return_generation_logits"] = True
    if output_config.exclude_input_from_output:
        kwargs["exclude_input_from_output"] = True
    if output_config.return_perf_metrics:
        kwargs["return_perf_metrics"] = True

    # Embedding bias
    if embedding_bias:
        if vocab_size and len(embedding_bias) != vocab_size:
            raise ValueError(
                f"embedding_bias length ({len(embedding_bias)}) must match vocab_size ({vocab_size})"
            )
        kwargs["embedding_bias"] = list(embedding_bias)

    # Guided decoding
    if guided_decoding and guided_decoding.guide:
        guide_type = guided_decoding.guide_type
        guide_content = guided_decoding.guide

        if guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_JSON:
            # json_object=True for JSON validation without schema constraint
            kwargs["guided_decoding"] = GuidedDecodingParams(json_object=True)
        elif guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_JSON_SCHEMA:
            kwargs["guided_decoding"] = GuidedDecodingParams(json=guide_content)
        elif guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_REGEX:
            kwargs["guided_decoding"] = GuidedDecodingParams(regex=guide_content)
        elif guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_EBNF_GRAMMAR:
            kwargs["guided_decoding"] = GuidedDecodingParams(grammar=guide_content)
        elif guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_STRUCTURAL_TAG:
            kwargs["guided_decoding"] = GuidedDecodingParams(structural_tag=guide_content)

    params = SamplingParams(**kwargs)

    return params


def create_lora_request_from_proto(
    proto_config: Optional[pb2.LoraConfig],
) -> Optional[LoRARequest]:
    """Convert protobuf LoraConfig to TensorRT-LLM LoRARequest.

    Args:
        proto_config: Protobuf LoraConfig message

    Returns:
        LoRARequest or None
    """
    if proto_config is None or proto_config.task_id == 0:
        return None

    return LoRARequest(
        lora_name=f"lora_{proto_config.task_id}",
        lora_int_id=proto_config.task_id,
    )


def create_disaggregated_params_from_proto(
    proto_config: Optional[pb2.DisaggregatedParams],
) -> Optional[DisaggregatedParams]:
    """Convert protobuf DisaggregatedParams to TensorRT-LLM DisaggregatedParams.

    Args:
        proto_config: Protobuf DisaggregatedParams message

    Returns:
        DisaggregatedParams or None
    """
    if proto_config is None:
        return None

    request_type_map = {
        pb2.DisaggregatedParams.REQUEST_TYPE_CONTEXT_AND_GENERATION: "context_and_generation",
        pb2.DisaggregatedParams.REQUEST_TYPE_CONTEXT_ONLY: "context_only",
        pb2.DisaggregatedParams.REQUEST_TYPE_GENERATION_ONLY: "generation_only",
    }

    request_type = request_type_map.get(proto_config.request_type, "context_and_generation")

    params = DisaggregatedParams(request_type=request_type)

    if proto_config.ctx_request_id:
        params.ctx_request_id = proto_config.ctx_request_id

    if proto_config.HasField("context_phase_params"):
        ctx_params = proto_config.context_phase_params
        if ctx_params.first_gen_token_id:
            params.first_gen_tokens = [ctx_params.first_gen_token_id]

    return params
