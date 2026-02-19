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

"""gRPC Servicer for TensorRT-LLM.

Implements the TrtllmService gRPC service for high-performance communication
with external routers (e.g., sgl-router) using pre-tokenized input.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import List, Union

import grpc

from tensorrt_llm.executor.result import Logprob, TokenLogprobs
from tensorrt_llm.logger import logger

from . import trtllm_service_pb2, trtllm_service_pb2_grpc
from .grpc_request_manager import (
    GrpcRequestManager,
    create_disaggregated_params_from_proto,
    create_lora_request_from_proto,
    create_sampling_params_from_proto,
)


class TrtllmServiceServicer(trtllm_service_pb2_grpc.TrtllmServiceServicer):
    """gRPC servicer implementing the TrtLlmEngine service.

    Handles RPCs:
    - Generate: Streaming text generation
    - Embed: Embeddings (for embedding models)
    - HealthCheck: Health probe
    - Abort: Cancel a request
    - GetModelInfo: Model metadata
    - GetServerInfo: Server state
    """

    def __init__(self, request_manager: GrpcRequestManager, model_path: str = ""):
        """Initialize the servicer.

        Args:
            request_manager: The GrpcRequestManager instance
            model_path: Path to the model (for metadata)
        """
        self.request_manager = request_manager
        self.model_path = model_path
        self._start_time = time.time()
        logger.info("TrtllmServiceServicer initialized")

    async def Generate(
        self,
        request: trtllm_service_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[trtllm_service_pb2.GenerateResponse, None]:
        """Handle streaming generation requests.

        Args:
            request: The GenerateRequest protobuf
            context: gRPC context

        Yields:
            GenerateResponse protobuf messages (streaming)
        """
        request_id = request.request_id
        logger.info(f"Generate request {request_id} received")

        try:
            # Extract tokenized input (required)
            if not request.HasField("tokenized"):
                yield self._error_response(
                    request_id,
                    "Missing tokenized input",
                    "INVALID_REQUEST",
                    400,
                )
                return

            prompt_token_ids = list(request.tokenized.input_token_ids)

            # Build sampling params with detokenize=False (key optimization!)
            sampling_params = create_sampling_params_from_proto(
                proto_config=request.sampling_config,
                output_config=request.output_config,
                max_tokens=request.max_tokens,
                end_id=request.end_id if request.HasField("end_id") else None,
                pad_id=request.pad_id if request.HasField("pad_id") else None,
                bad_words=list(request.bad_words) if request.bad_words else None,
                stop_words=list(request.stop_words) if request.stop_words else None,
                guided_decoding=request.guided_decoding
                if request.HasField("guided_decoding")
                else None,
                embedding_bias=list(request.embedding_bias) if request.embedding_bias else None,
            )

            # Build LoRA request if present
            lora_request = create_lora_request_from_proto(
                request.lora_config if request.HasField("lora_config") else None
            )

            # Build disaggregated params if present
            disaggregated_params = create_disaggregated_params_from_proto(
                request.disaggregated_params if request.HasField("disaggregated_params") else None
            )

            # Track tokens sent per sequence index to avoid duplicates
            # TRT-LLM's token_ids_diff doesn't clear between iterations for n>1
            sent_token_counts: dict[int, int] = {}

            # Submit to request manager and stream outputs
            # The request manager now yields GenerationResult objects
            async for gen_result in self.request_manager.generate(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                streaming=request.streaming,
                lora_request=lora_request,
                disaggregated_params=disaggregated_params,
            ):
                # Check if client disconnected
                if context.cancelled():
                    logger.info(f"Client disconnected for {request_id}")
                    await self.request_manager.abort(request_id)
                    return

                # Convert GenerationResult to protobuf response
                if request.streaming:
                    for chunk_response in self._chunk_responses(
                        request_id, gen_result, prompt_token_ids, sent_token_counts
                    ):
                        yield chunk_response

                # Send complete responses when finished (one per sequence for n>1)
                if gen_result.finished:
                    for complete_response in self._complete_responses(
                        request_id, gen_result, prompt_token_ids
                    ):
                        yield complete_response

        except asyncio.CancelledError:
            logger.info(f"Request {request_id} cancelled")
            await self.request_manager.abort(request_id)
            raise
        except Exception as e:
            logger.error(f"Error in Generate for {request_id}: {e}")
            yield self._error_response(
                request_id,
                str(e),
                "INTERNAL_ERROR",
                500,
            )

    async def Embed(
        self,
        request: trtllm_service_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_service_pb2.EmbedResponse:
        """Handle embedding requests.

        Args:
            request: The EmbedRequest protobuf
            context: gRPC context

        Returns:
            EmbedResponse protobuf
        """
        logger.warning("Embed RPC not yet implemented")
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Embed RPC not yet implemented")
        return trtllm_service_pb2.EmbedResponse(
            request_id=request.request_id,
            embedding=[],
            prompt_tokens=0,
        )

    async def HealthCheck(
        self,
        request: trtllm_service_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_service_pb2.HealthCheckResponse:
        """Handle health check requests.

        Args:
            request: The HealthCheckRequest protobuf
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf
        """
        is_healthy, message = await self.request_manager.health_check()
        logger.info(f"HealthCheck: healthy={is_healthy}, message={message}")

        return trtllm_service_pb2.HealthCheckResponse(
            status=message,
        )

    async def Abort(
        self,
        request: trtllm_service_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_service_pb2.AbortResponse:
        """Handle abort requests.

        Args:
            request: The AbortRequest protobuf
            context: gRPC context

        Returns:
            AbortResponse protobuf
        """
        request_id = request.request_id
        logger.info(f"Abort request for {request_id}")

        success = await self.request_manager.abort(request_id)

        return trtllm_service_pb2.AbortResponse(
            success=success,
            message=f"Request {request_id} {'aborted' if success else 'not found'}",
        )

    async def GetModelInfo(
        self,
        request: trtllm_service_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_service_pb2.GetModelInfoResponse:
        """Handle model info requests.

        Args:
            request: The GetModelInfoRequest protobuf
            context: gRPC context

        Returns:
            GetModelInfoResponse protobuf
        """
        model_config = self.request_manager.get_model_config()

        return trtllm_service_pb2.GetModelInfoResponse(
            model_id=self.model_path or model_config.get("model_path", ""),
            max_input_len=model_config.get("max_context_length", 0),
            max_seq_len=model_config.get("max_seq_len", 0)
            or model_config.get("max_context_length", 0),
            vocab_size=model_config.get("vocab_size", 0),
        )

    async def GetServerInfo(
        self,
        request: trtllm_service_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_service_pb2.GetServerInfoResponse:
        """Handle server info requests.

        Args:
            request: The GetServerInfoRequest protobuf
            context: gRPC context

        Returns:
            GetServerInfoResponse protobuf
        """
        try:
            import tensorrt_llm

            version = getattr(tensorrt_llm, "__version__", "unknown")
        except Exception:
            version = "unknown"

        # Try to get parallelism info from LLM args
        tp_size = 1
        pp_size = 1
        world_size = 1

        try:
            llm = self.request_manager.llm
            if hasattr(llm, "args") and llm.args is not None:
                args = llm.args
                if hasattr(args, "tensor_parallel_size") and args.tensor_parallel_size:
                    tp_size = args.tensor_parallel_size
                if hasattr(args, "pipeline_parallel_size") and args.pipeline_parallel_size:
                    pp_size = args.pipeline_parallel_size
                world_size = tp_size * pp_size
        except Exception as e:
            logger.debug(f"Could not get parallelism info: {e}")

        return trtllm_service_pb2.GetServerInfoResponse(
            version=version,
            backend="tensorrt-llm",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            context_parallel_size=1,  # Context parallelism is separate from TP/PP
            world_size=world_size,
        )

    # ========== Helper methods ==========

    def _convert_logprobs_to_proto(
        self,
        token_ids: List[int],
        logprobs: Union[TokenLogprobs, List[float], None],
    ) -> List[trtllm_service_pb2.TokenLogprob]:
        """Convert TRT-LLM logprobs to protobuf TokenLogprob messages.

        Handles both formats:
        - List[float]: Simple logprobs (one per token)
        - TokenLogprobs (list[dict[int, Logprob]]): Top-k logprobs per position

        Args:
            token_ids: The sampled token IDs
            logprobs: Logprobs from TRT-LLM (can be List[float] or TokenLogprobs)

        Returns:
            List of TokenLogprob proto messages
        """
        if not logprobs or not token_ids:
            return []

        result = []
        for i, token_id in enumerate(token_ids):
            if i >= len(logprobs):
                break

            lp = logprobs[i]

            if isinstance(lp, dict):
                # TokenLogprobs format: dict[int, Logprob]
                # Each entry maps token_id -> Logprob(logprob, rank)
                token_logprob = trtllm_service_pb2.TokenLogprob(
                    token_id=token_id,
                    logprob=lp[token_id].logprob if token_id in lp else 0.0,
                )
                # Add top logprobs (all entries in the dict)
                for tid, logprob_obj in lp.items():
                    if isinstance(logprob_obj, Logprob):
                        token_logprob.top_logprobs.append(
                            trtllm_service_pb2.TopLogprob(
                                token_id=tid,
                                logprob=logprob_obj.logprob,
                            )
                        )
                result.append(token_logprob)
            elif isinstance(lp, (int, float)):
                # Simple float logprob
                token_logprob = trtllm_service_pb2.TokenLogprob(
                    token_id=token_id,
                    logprob=float(lp),
                )
                result.append(token_logprob)

        return result

    def _chunk_responses(
        self,
        request_id: str,
        gen_result,
        prompt_token_ids: list,
        sent_token_counts: dict[int, int],
    ) -> List[trtllm_service_pb2.GenerateResponse]:
        """Build streaming chunk responses from GenerationResult.

        Uses cumulative token_ids and tracks sent position to compute true deltas.
        TRT-LLM's token_ids_diff doesn't clear between iterations for n>1, so we
        compute deltas ourselves.

        Args:
            request_id: The request ID
            gen_result: TensorRT-LLM GenerationResult
            prompt_token_ids: Original prompt tokens
            sent_token_counts: Dict tracking tokens already sent per sequence index

        Returns:
            List of GenerateResponse with chunk field set (one per output)
        """
        responses = []
        cached_tokens = gen_result.cached_tokens if hasattr(gen_result, "cached_tokens") else 0

        if not gen_result.outputs:
            # No outputs yet, return empty chunk
            responses.append(
                trtllm_service_pb2.GenerateResponse(
                    request_id=request_id,
                    chunk=trtllm_service_pb2.GenerateStreamChunk(
                        token_ids=[],
                        prompt_tokens=len(prompt_token_ids),
                        completion_tokens=0,
                        cached_tokens=cached_tokens,
                    ),
                )
            )
            return responses

        # Process all outputs (for n>1 support)
        for completion in gen_result.outputs:
            index = completion.index
            # Use cumulative token_ids and compute delta ourselves
            # because token_ids_diff doesn't clear between iterations for n>1
            all_tokens = list(completion.token_ids) if completion.token_ids else []
            sent_count = sent_token_counts.get(index, 0)
            delta_tokens = all_tokens[sent_count:]

            # Skip if no new tokens for this sequence
            if not delta_tokens:
                continue

            # Update sent count
            sent_token_counts[index] = len(all_tokens)

            chunk = trtllm_service_pb2.GenerateStreamChunk(
                token_ids=delta_tokens,
                sequence_index=completion.index,
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(completion.token_ids) if completion.token_ids else 0,
                cached_tokens=cached_tokens,
            )

            # Add logprobs if available
            # Note: We compute delta logprobs ourselves since logprobs_diff has same issue as token_ids_diff
            if completion.logprobs:
                all_logprobs = completion.logprobs
                delta_logprobs = all_logprobs[sent_count:] if sent_count < len(all_logprobs) else []
                proto_logprobs = self._convert_logprobs_to_proto(delta_tokens, delta_logprobs)
                chunk.logprobs.extend(proto_logprobs)

            responses.append(
                trtllm_service_pb2.GenerateResponse(
                    request_id=request_id,
                    chunk=chunk,
                )
            )

        return responses

    def _complete_responses(
        self,
        request_id: str,
        gen_result,
        prompt_token_ids: list,
    ) -> List[trtllm_service_pb2.GenerateResponse]:
        """Build final completion responses from GenerationResult.

        For n>1, returns one response per output sequence.

        Args:
            request_id: The request ID
            gen_result: TensorRT-LLM GenerationResult (finished=True)
            prompt_token_ids: Original prompt tokens

        Returns:
            List of GenerateResponse with complete field set (one per output)
        """
        responses = []
        cached_tokens = gen_result.cached_tokens if hasattr(gen_result, "cached_tokens") else 0

        if not gen_result.outputs:
            # No outputs, return error response
            responses.append(
                trtllm_service_pb2.GenerateResponse(
                    request_id=request_id,
                    complete=trtllm_service_pb2.GenerateComplete(
                        output_token_ids=[],
                        finish_reason="error",
                        prompt_tokens=len(prompt_token_ids),
                        completion_tokens=0,
                        cached_tokens=0,
                    ),
                )
            )
            return responses

        # Process all outputs (for n>1 support)
        for completion in gen_result.outputs:
            output_tokens = list(completion.token_ids) if completion.token_ids else []

            complete = trtllm_service_pb2.GenerateComplete(
                output_token_ids=output_tokens,
                sequence_index=completion.index,
                finish_reason=completion.finish_reason or "stop",
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(output_tokens),
                cached_tokens=cached_tokens,
            )

            # Add stop reason if available
            if hasattr(completion, "stop_reason") and completion.stop_reason:
                complete.stop_reason = str(completion.stop_reason)

            # Add generation logprobs if available
            if completion.logprobs:
                proto_logprobs = self._convert_logprobs_to_proto(output_tokens, completion.logprobs)
                complete.logprobs.extend(proto_logprobs)

            # Add prompt logprobs if available
            if hasattr(completion, "prompt_logprobs") and completion.prompt_logprobs:
                # For prompt logprobs, we use the prompt_token_ids
                proto_prompt_logprobs = self._convert_logprobs_to_proto(
                    prompt_token_ids, completion.prompt_logprobs
                )
                complete.prompt_logprobs.extend(proto_prompt_logprobs)

            responses.append(
                trtllm_service_pb2.GenerateResponse(
                    request_id=request_id,
                    complete=complete,
                )
            )

        return responses

    def _error_response(
        self,
        request_id: str,
        message: str,
        error_type: str,
        code: int,
    ) -> trtllm_service_pb2.GenerateResponse:
        """Build an error response.

        Args:
            request_id: The request ID
            message: Error message
            error_type: Error type string
            code: Error code

        Returns:
            GenerateResponse with error field set
        """
        return trtllm_service_pb2.GenerateResponse(
            request_id=request_id,
            error=trtllm_service_pb2.GenerateError(
                message=message,
                type=error_type,
                code=code,
            ),
        )
