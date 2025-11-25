#!/usr/bin/env python3
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

"""
TensorRT-LLM gRPC Server

Starts a gRPC server for TensorRT-LLM using the TrtLlmEngine protocol.
Designed for high-performance communication with external routers (e.g., sgl-router)
using pre-tokenized input and raw token ID output.

This server inherits all arguments from the standard TensorRT-LLM serve command,
ensuring consistent behavior with the OpenAI-compatible HTTP server.

Key Features:
- Pre-tokenized input (no Python tokenization overhead)
- Raw token ID output (no Python detokenization overhead)
- Streaming support with delta tokens
- Full sampling parameter support
- Guided decoding (JSON schema, regex, grammar)
- LoRA and prompt tuning support
- Disaggregated inference support

Usage:
    python -m tensorrt_llm.entrypoints.grpc_server <model_path> [OPTIONS]

Example:
    python -m tensorrt_llm.entrypoints.grpc_server \
        /raid/models/meta-llama/Llama-3.1-8B-Instruct \
        --backend pytorch \
        --host 0.0.0.0 \
        --port 50051 \
        --tp_size 1 \
        --trust_remote_code
"""

import asyncio
import signal
import time
from collections.abc import AsyncGenerator
from typing import Optional

import click
import grpc
import yaml

try:
    from grpc_reflection.v1alpha import reflection
    REFLECTION_AVAILABLE = True
except ImportError:
    REFLECTION_AVAILABLE = False

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.commands.serve import get_llm_args
from tensorrt_llm.grpc import trtllm_engine_pb2, trtllm_engine_pb2_grpc
from tensorrt_llm.grpc.grpc_request_manager import (
    GrpcRequestManager,
    create_sampling_params_from_proto,
    create_lora_request_from_proto,
    create_disaggregated_params_from_proto,
)
from tensorrt_llm.llmapi import BuildConfig
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.logger import logger, severity_map


class TrtLlmEngineServicer(trtllm_engine_pb2_grpc.TrtLlmEngineServicer):
    """
    gRPC servicer implementing the TrtLlmEngine service.

    Handles RPCs:
    - Generate: Streaming text generation
    - Embed: Embeddings (for embedding models)
    - HealthCheck: Health probe
    - Abort: Cancel a request
    - GetModelInfo: Model metadata
    - GetServerInfo: Server state
    """

    def __init__(self, request_manager: GrpcRequestManager, model_path: str = ""):
        """
        Initialize the servicer.

        Args:
            request_manager: The GrpcRequestManager instance
            model_path: Path to the model (for metadata)
        """
        self.request_manager = request_manager
        self.model_path = model_path
        self._start_time = time.time()
        logger.info("TrtLlmEngineServicer initialized")

    async def Generate(
        self,
        request: trtllm_engine_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[trtllm_engine_pb2.GenerateResponse, None]:
        """
        Handle streaming generation requests.

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
            query_token_ids = list(request.tokenized.query_token_ids) if request.tokenized.query_token_ids else None

            # Build sampling params with detokenize=False (key optimization!)
            sampling_params = create_sampling_params_from_proto(
                proto_config=request.sampling_config,
                output_config=request.output_config,
                max_tokens=request.max_tokens,
                streaming=request.streaming,
                end_id=request.end_id if request.HasField("end_id") else None,
                pad_id=request.pad_id if request.HasField("pad_id") else None,
                bad_words=list(request.bad_words) if request.bad_words else None,
                stop_words=list(request.stop_words) if request.stop_words else None,
                guided_decoding=request.guided_decoding if request.HasField("guided_decoding") else None,
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

            # Submit to request manager and stream outputs
            # The request manager now yields GenerationResult objects
            async for gen_result in self.request_manager.generate(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                streaming=request.streaming,
                query_token_ids=query_token_ids,
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
                    yield self._chunk_response(request_id, gen_result, prompt_token_ids)

                # Send complete response when finished
                if gen_result.finished:
                    yield self._complete_response(request_id, gen_result, prompt_token_ids)

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
        request: trtllm_engine_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_engine_pb2.EmbedResponse:
        """
        Handle embedding requests.

        Args:
            request: The EmbedRequest protobuf
            context: gRPC context

        Returns:
            EmbedResponse protobuf
        """
        logger.warning("Embed RPC not yet implemented")
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Embed RPC not yet implemented")
        return trtllm_engine_pb2.EmbedResponse(
            request_id=request.request_id,
            embedding=[],
            prompt_tokens=0,
        )

    async def HealthCheck(
        self,
        request: trtllm_engine_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_engine_pb2.HealthCheckResponse:
        """
        Handle health check requests.

        Args:
            request: The HealthCheckRequest protobuf
            context: gRPC context

        Returns:
            HealthCheckResponse protobuf
        """
        is_healthy, message = await self.request_manager.health_check()
        logger.debug(f"HealthCheck: healthy={is_healthy}, message={message}")

        return trtllm_engine_pb2.HealthCheckResponse(
            status=message,
        )

    async def Abort(
        self,
        request: trtllm_engine_pb2.AbortRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_engine_pb2.AbortResponse:
        """
        Handle abort requests.

        Args:
            request: The AbortRequest protobuf
            context: gRPC context

        Returns:
            AbortResponse protobuf
        """
        request_id = request.request_id
        logger.info(f"Abort request for {request_id}")

        success = await self.request_manager.abort(request_id)

        return trtllm_engine_pb2.AbortResponse(
            success=success,
            message=f"Request {request_id} {'aborted' if success else 'not found'}",
        )

    async def GetModelInfo(
        self,
        request: trtllm_engine_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_engine_pb2.GetModelInfoResponse:
        """
        Handle model info requests.

        Args:
            request: The GetModelInfoRequest protobuf
            context: gRPC context

        Returns:
            GetModelInfoResponse protobuf
        """
        model_config = self.request_manager.get_model_config()

        return trtllm_engine_pb2.GetModelInfoResponse(
            model_id=self.model_path or model_config.get("model_path", ""),
            max_input_len=model_config.get("max_context_length", 0),
            max_seq_len=model_config.get("max_context_length", 0),
            vocab_size=model_config.get("vocab_size", 0),
        )

    async def GetServerInfo(
        self,
        request: trtllm_engine_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> trtllm_engine_pb2.GetServerInfoResponse:
        """
        Handle server info requests.

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

        return trtllm_engine_pb2.GetServerInfoResponse(
            version=version,
            backend="tensorrt-llm",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            world_size=1,
        )

    # ========== Helper methods ==========

    def _chunk_response(
        self,
        request_id: str,
        gen_result,
        prompt_token_ids: list,
    ) -> trtllm_engine_pb2.GenerateResponse:
        """
        Build a streaming chunk response from GenerationResult.
        Uses token_ids_diff to get delta tokens.

        Args:
            request_id: The request ID
            gen_result: TensorRT-LLM GenerationResult
            prompt_token_ids: Original prompt tokens

        Returns:
            GenerateResponse with chunk field set
        """
        # Get first output (for n=1)
        completion = gen_result.outputs[0] if gen_result.outputs else None

        if completion is None:
            return trtllm_engine_pb2.GenerateResponse(
                request_id=request_id,
                chunk=trtllm_engine_pb2.GenerateStreamChunk(
                    token_ids=[],
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=0,
                    cached_tokens=gen_result.cached_tokens if hasattr(gen_result, 'cached_tokens') else 0,
                ),
            )

        # token_ids_diff contains only new tokens since last iteration
        delta_tokens = list(completion.token_ids_diff) if completion.token_ids_diff else []

        chunk = trtllm_engine_pb2.GenerateStreamChunk(
            token_ids=delta_tokens,
            sequence_index=completion.index,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(completion.token_ids) if completion.token_ids else 0,
            cached_tokens=gen_result.cached_tokens if hasattr(gen_result, 'cached_tokens') else 0,
        )

        # Add logprobs if available
        if completion.logprobs_diff:
            for lp in completion.logprobs_diff:
                if isinstance(lp, dict):
                    token_logprob = trtllm_engine_pb2.TokenLogprob(
                        token_id=lp.get("token_id", 0),
                        logprob=lp.get("logprob", 0.0),
                    )
                    chunk.logprobs.append(token_logprob)

        return trtllm_engine_pb2.GenerateResponse(
            request_id=request_id,
            chunk=chunk,
        )

    def _complete_response(
        self,
        request_id: str,
        gen_result,
        prompt_token_ids: list,
    ) -> trtllm_engine_pb2.GenerateResponse:
        """
        Build a final completion response from GenerationResult.

        Args:
            request_id: The request ID
            gen_result: TensorRT-LLM GenerationResult (finished=True)
            prompt_token_ids: Original prompt tokens

        Returns:
            GenerateResponse with complete field set
        """
        # Get first output (for n=1)
        completion = gen_result.outputs[0] if gen_result.outputs else None

        if completion is None:
            return trtllm_engine_pb2.GenerateResponse(
                request_id=request_id,
                complete=trtllm_engine_pb2.GenerateComplete(
                    output_token_ids=[],
                    finish_reason="error",
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=0,
                    cached_tokens=0,
                ),
            )

        # Get all output tokens
        output_tokens = list(completion.token_ids) if completion.token_ids else []

        complete = trtllm_engine_pb2.GenerateComplete(
            output_token_ids=output_tokens,
            sequence_index=completion.index,
            finish_reason=completion.finish_reason or "stop",
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(output_tokens),
            cached_tokens=gen_result.cached_tokens if hasattr(gen_result, 'cached_tokens') else 0,
        )

        # Add stop reason if available
        if hasattr(completion, 'stop_reason') and completion.stop_reason:
            complete.stop_reason = str(completion.stop_reason)

        # Add logprobs if available
        if completion.logprobs:
            for lp in completion.logprobs:
                if isinstance(lp, dict):
                    token_logprob = trtllm_engine_pb2.TokenLogprob(
                        token_id=lp.get("token_id", 0),
                        logprob=lp.get("logprob", 0.0),
                    )
                    complete.logprobs.append(token_logprob)

        # Add prompt logprobs if available
        if hasattr(completion, 'prompt_logprobs') and completion.prompt_logprobs:
            for lp in completion.prompt_logprobs:
                if isinstance(lp, dict):
                    token_logprob = trtllm_engine_pb2.TokenLogprob(
                        token_id=lp.get("token_id", 0),
                        logprob=lp.get("logprob", 0.0),
                    )
                    complete.prompt_logprobs.append(token_logprob)

        return trtllm_engine_pb2.GenerateResponse(
            request_id=request_id,
            complete=complete,
        )

    def _error_response(
        self,
        request_id: str,
        message: str,
        error_type: str,
        code: int,
    ) -> trtllm_engine_pb2.GenerateResponse:
        """
        Build an error response.

        Args:
            request_id: The request ID
            message: Error message
            error_type: Error type string
            code: Error code

        Returns:
            GenerateResponse with error field set
        """
        return trtllm_engine_pb2.GenerateResponse(
            request_id=request_id,
            error=trtllm_engine_pb2.GenerateError(
                message=message,
                type=error_type,
                code=code,
            ),
        )


async def serve_grpc(
    host: str,
    port: int,
    llm_args: dict,
    model_path: str,
):
    """
    Main serving function.

    Args:
        host: Host to bind to
        port: Port to bind to
        llm_args: Arguments for LLM initialization (from get_llm_args)
        model_path: Path to the model
    """
    logger.info("Initializing TensorRT-LLM gRPC server...")

    # Create LLM based on backend
    backend = llm_args.get("backend")
    if backend == "pytorch":
        llm = PyTorchLLM(**llm_args)
    else:
        llm = LLM(**llm_args)

    logger.info("Model loaded successfully")

    # Create request manager
    request_manager = GrpcRequestManager(llm)

    # Create servicer
    servicer = TrtLlmEngineServicer(request_manager, model_path=model_path)

    # Create gRPC server
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),  # Unlimited
            ("grpc.max_receive_message_length", -1),  # Unlimited
            ("grpc.keepalive_time_ms", 30000),  # 30s keepalive
            ("grpc.keepalive_timeout_ms", 10000),  # 10s timeout
        ],
    )

    # Add servicer to server
    trtllm_engine_pb2_grpc.add_TrtLlmEngineServicer_to_server(servicer, server)

    # Enable reflection for grpcurl and other tools
    if REFLECTION_AVAILABLE:
        service_names = (
            trtllm_engine_pb2.DESCRIPTOR.services_by_name["TrtLlmEngine"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
        logger.info("gRPC reflection enabled")

    # Bind to address
    address = f"{host}:{port}"
    server.add_insecure_port(address)

    # Start server
    await server.start()
    logger.info(f"TensorRT-LLM gRPC server started on {address}")
    logger.info("Server is ready to accept requests")

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Serve until shutdown signal
    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Shutting down TensorRT-LLM gRPC server...")

        # Stop gRPC server
        await server.stop(grace=5.0)
        logger.info("gRPC server stopped")

        # Shutdown LLM
        if hasattr(llm, "shutdown"):
            llm.shutdown()
        logger.info("LLM engine stopped")

        logger.info("Shutdown complete")


# Use Click to define CLI with same options as serve command
@click.command("grpc_server")
@click.argument("model", type=str)
@click.option("--tokenizer",
              type=str,
              default=None,
              help="Path | Name of the tokenizer. "
              "Specify this value only if using TensorRT engine as model.")
@click.option("--host",
              type=str,
              default="0.0.0.0",
              help="Hostname of the gRPC server.")
@click.option("--port", type=int, default=50051, help="Port of the gRPC server.")
@click.option("--backend",
              type=click.Choice(["pytorch", "trt"]),
              default="pytorch",
              help="Set to 'pytorch' for pytorch path. Default is cpp path.")
@click.option('--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
@click.option("--max_beam_width",
              type=int,
              default=BuildConfig.max_beam_width,
              help="Maximum number of beams for beam search decoding.")
@click.option("--max_batch_size",
              type=int,
              default=BuildConfig.max_batch_size,
              help="Maximum number of requests that the engine can schedule.")
@click.option(
    "--max_num_tokens",
    type=int,
    default=BuildConfig.max_num_tokens,
    help="Maximum number of batched input tokens after padding is removed in each batch."
)
@click.option(
    "--max_seq_len",
    type=int,
    default=BuildConfig.max_seq_len,
    help="Maximum total length of one request, including prompt and outputs. "
    "If unspecified, the value is deduced from the model config.")
@click.option("--tp_size", type=int, default=1, help='Tensor parallelism size.')
@click.option("--pp_size",
              type=int,
              default=1,
              help='Pipeline parallelism size.')
@click.option("--ep_size",
              type=int,
              default=None,
              help="Expert parallelism size")
@click.option("--cluster_size",
              type=int,
              default=None,
              help="Expert cluster parallelism size")
@click.option("--gpus_per_node",
              type=int,
              default=None,
              help="Number of GPUs per node. Default to None, and it will be "
              "detected automatically.")
@click.option("--kv_cache_free_gpu_memory_fraction",
              type=float,
              default=0.9,
              help="Free GPU memory fraction reserved for KV Cache, "
              "after allocating model weights and buffers.")
@click.option("--trust_remote_code",
              is_flag=True,
              default=False,
              help="Flag for HF transformers.")
@click.option(
    "--extra_llm_api_options",
    type=str,
    default=None,
    help="Path to a YAML file that overwrites the parameters specified by trtllm-serve."
)
@click.option(
    "--reasoning_parser",
    type=click.Choice(list(ReasoningParserFactory.parsers.keys()) + [None]),
    default=None,
    help="[Experimental] Specify the parser for reasoning models.",
)
@click.option(
    "--fail_fast_on_attention_window_too_large",
    is_flag=True,
    default=False,
    help="Exit with runtime error when attention window is too large to fit even a single sequence in the KV cache."
)
def main(
    model: str,
    tokenizer: Optional[str],
    host: str,
    port: int,
    log_level: str,
    backend: str,
    max_beam_width: int,
    max_batch_size: int,
    max_num_tokens: int,
    max_seq_len: int,
    tp_size: int,
    pp_size: int,
    ep_size: Optional[int],
    cluster_size: Optional[int],
    gpus_per_node: Optional[int],
    kv_cache_free_gpu_memory_fraction: float,
    trust_remote_code: bool,
    extra_llm_api_options: Optional[str],
    reasoning_parser: Optional[str],
    fail_fast_on_attention_window_too_large: bool,
):
    """
    TensorRT-LLM gRPC Server

    MODEL: model name | HF checkpoint path | TensorRT engine path

    This server provides a gRPC interface for TensorRT-LLM, designed for
    high-performance communication with external routers (e.g., sgl-router)
    using pre-tokenized input and raw token ID output.

    All options are inherited from the standard TensorRT-LLM serve command.
    """
    logger.set_level(log_level)

    # Use get_llm_args from serve.py to build LLM arguments
    # This ensures consistency with the HTTP server
    llm_args, _ = get_llm_args(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        max_beam_width=max_beam_width,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        moe_expert_parallel_size=ep_size,
        moe_cluster_parallel_size=cluster_size,
        gpus_per_node=gpus_per_node,
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        num_postprocess_workers=0,  # gRPC doesn't need postprocess workers
        trust_remote_code=trust_remote_code,
        reasoning_parser=reasoning_parser,
        fail_fast_on_attention_window_too_large=fail_fast_on_attention_window_too_large,
    )

    # Load extra options from YAML if provided
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_extra_dict = yaml.safe_load(f)
        llm_args = update_llm_args_with_extra_dict(llm_args, llm_args_extra_dict)

    # Run gRPC server
    asyncio.run(serve_grpc(host, port, llm_args, model))


if __name__ == "__main__":
    main()
