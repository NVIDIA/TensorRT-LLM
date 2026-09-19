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

"""Server lifecycle for the TensorRT-LLM SMG gRPC adapter."""

import asyncio
import signal
from typing import Any

import click
import grpc
import uvloop

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm.logger import logger

from .bindings import trtllm_service_pb2, trtllm_service_pb2_grpc
from .request_manager import GrpcRequestManager
from .servicer import TrtllmServiceServicer

_GRPC_MAX_MESSAGE_LENGTH_BYTES = 32 * 1024 * 1024


def launch_smg_server(
    host: str,
    port: int,
    llm_args: dict[str, Any],
    served_model_name: str | None = None,
) -> None:
    """Launch the SMG gRPC server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        llm_args: Arguments for LLM initialization.
        served_model_name: Model name returned by discovery RPCs. Defaults to
            the model path.
    """
    try:
        from grpc_reflection.v1alpha import reflection
    except ModuleNotFoundError as e:
        if e.name != "grpc_reflection":
            raise
        reflection = None

    async def serve_grpc_async() -> None:
        logger.info("Initializing TensorRT-LLM SMG gRPC server...")

        backend = llm_args.get("backend")
        model_path = served_model_name or llm_args.get("model", "")

        if backend == "pytorch":
            llm_args.pop("build_config", None)
            llm = PyTorchLLM(**llm_args)
        elif backend == "_autodeploy":
            from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM

            llm_args.pop("build_config", None)
            llm = AutoDeployLLM(**llm_args)
        else:
            raise click.BadParameter(
                f"{backend} is not a known backend, check help for available options.",
                param_hint="backend",
            )

        logger.info("Model loaded successfully")

        server = None
        try:
            request_manager = GrpcRequestManager(llm)
            servicer = TrtllmServiceServicer(request_manager, model_path=model_path)

            server = grpc.aio.server(
                options=[
                    ("grpc.max_send_message_length", _GRPC_MAX_MESSAGE_LENGTH_BYTES),
                    ("grpc.max_receive_message_length", _GRPC_MAX_MESSAGE_LENGTH_BYTES),
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
                ]
            )
            trtllm_service_pb2_grpc.add_TrtllmServiceServicer_to_server(servicer, server)

            if reflection is not None:
                service_names = (
                    trtllm_service_pb2.DESCRIPTOR.services_by_name["TrtllmService"].full_name,
                    reflection.SERVICE_NAME,
                )
                reflection.enable_server_reflection(service_names, server)
                logger.info("gRPC reflection enabled")

            address = f"{host}:{port}"
            server.add_insecure_port(address)
            await server.start()
            logger.info(f"TensorRT-LLM SMG gRPC server started on {address}")
            logger.info("Server is ready to accept requests")

            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()

            def signal_handler() -> None:
                logger.info("Received shutdown signal")
                stop_event.set()

            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, signal_handler)

            await stop_event.wait()
        finally:
            logger.info("Shutting down TensorRT-LLM SMG gRPC server...")
            try:
                if server is not None:
                    await server.stop(grace=5.0)
                    logger.info("gRPC server stopped")
            finally:
                if hasattr(llm, "shutdown"):
                    llm.shutdown()
                logger.info("LLM engine stopped")
                logger.info("Shutdown complete")

    uvloop.run(serve_grpc_async())


__all__ = ["launch_smg_server"]
