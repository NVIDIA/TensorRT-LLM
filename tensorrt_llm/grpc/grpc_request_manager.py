# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""gRPC Request Manager for TensorRT-LLM.

Manages request lifecycle for gRPC requests, converting between protobuf
and TensorRT-LLM types. Designed for high-performance communication with
external routers (e.g., sgl-router) using pre-tokenized input.

Key optimization: Sets detokenize=False in SamplingParams to skip
detokenization and return token IDs only.
"""

import asyncio
import traceback
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.request import LoRARequest, PromptAdapterRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.llmapi.llm_utils import KvCacheRetentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

from . import trtllm_service_pb2 as pb2


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
    ) -> AsyncGenerator[GenerationResult, None]:
        """Submit a generation request and stream outputs.

        Args:
            request_id: Unique request identifier (for tracking/abort)
            prompt_token_ids: Pre-tokenized input from Rust router
            sampling_params: Sampling parameters (with detokenize=False!)
            streaming: Whether to stream results
            lora_request: Optional LoRA adapter request
            prompt_adapter_request: Optional prompt adapter request
            kv_cache_retention_config: KV cache retention config
            disaggregated_params: Disaggregated inference params

        Yields:
            GenerationResult objects containing token IDs (text will be empty
            because detokenize=False)
        """
        try:
            # Submit to LLM.generate_async which returns a GenerationResult
            # that is an async iterator
            gen_result = self.llm.generate_async(
                {"prompt_token_ids": prompt_token_ids},
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

            # Check if executor is available and not shutdown
            if hasattr(self.llm, "_executor"):
                if self.llm._executor is None or self.llm._executor.is_shutdown():
                    return False, "Executor is shutdown"

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
                if hasattr(self.llm.tokenizer, "vocab_size"):
                    config["vocab_size"] = self.llm.tokenizer.vocab_size

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


def create_sampling_params_from_proto(
    proto_config: pb2.SamplingConfig,
    output_config: pb2.OutputConfig,
    max_tokens: int,
    end_id: Optional[int] = None,
    pad_id: Optional[int] = None,
    bad_words: Optional[List[pb2.TokenSequence]] = None,
    stop_words: Optional[List[pb2.TokenSequence]] = None,
    guided_decoding: Optional[pb2.GuidedDecodingParams] = None,
    embedding_bias: Optional[List[float]] = None,
) -> SamplingParams:
    """Convert protobuf configuration to TensorRT-LLM SamplingParams.

    Args:
        proto_config: Protobuf SamplingConfig message
        output_config: Protobuf OutputConfig message
        max_tokens: Maximum tokens to generate
        end_id: End-of-sequence token ID
        pad_id: Padding token ID
        bad_words: Bad word token sequences
        stop_words: Stop word token sequences
        guided_decoding: Guided decoding parameters
        embedding_bias: Embedding bias tensor

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
        kwargs["beam_width"] = proto_config.beam_width
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
        kwargs["random_seed"] = proto_config.seed

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

    # End/pad tokens
    if end_id is not None:
        kwargs["end_id"] = end_id
        if end_id == -1:
            kwargs["ignore_eos"] = True
    if pad_id is not None:
        kwargs["pad_id"] = pad_id

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

    # Stop sequences (as token ID lists)
    if stop_words:
        kwargs["stop_words"] = [list(seq.token_ids) for seq in stop_words]
    if bad_words:
        kwargs["bad_words"] = [list(seq.token_ids) for seq in bad_words]

    # Embedding bias
    if embedding_bias:
        kwargs["embedding_bias"] = list(embedding_bias)

    # Guided decoding
    if guided_decoding and guided_decoding.guide:
        guide_type = guided_decoding.guide_type
        guide_content = guided_decoding.guide

        if guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_JSON:
            # json_object=True for JSON validation without schema constraint
            kwargs["guided_decoding_params"] = GuidedDecodingParams(json_object=True)
        elif guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_JSON_SCHEMA:
            kwargs["guided_decoding_params"] = GuidedDecodingParams(json_schema=guide_content)
        elif guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_REGEX:
            kwargs["guided_decoding_params"] = GuidedDecodingParams(regex=guide_content)
        elif guide_type == pb2.GuidedDecodingParams.GUIDE_TYPE_EBNF_GRAMMAR:
            kwargs["guided_decoding_params"] = GuidedDecodingParams(grammar=guide_content)

    return SamplingParams(**kwargs)


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
        params.first_gen_token_id = ctx_params.first_gen_token_id
        if ctx_params.kv_cache_blocks:
            params.kv_cache_blocks = ctx_params.kv_cache_blocks

    return params
