# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMG gRPC server lifecycle for TensorRT-LLM."""

import asyncio
import signal

import click
import grpc
import uvloop

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm.logger import logger

from . import PROTOS_AVAILABLE, trtllm_service_pb2, trtllm_service_pb2_grpc

__all__ = ["launch_server"]

_MAX_MESSAGE_LENGTH_BYTES = 32 * 1024 * 1024


def launch_server(
    host: str,
    port: int,
    llm_args: dict,
    served_model_name: str | None = None,
) -> None:
    """Launch the SMG gRPC server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        llm_args: Arguments used to initialize the LLM.
        served_model_name: Model name exposed by the server. Defaults to the model path.
    """
    if not PROTOS_AVAILABLE:
        raise click.ClickException(
            "SMG gRPC support requires smg-grpc-proto. Install it with "
            "`python -m pip install smg-grpc-proto`."
        )

    from .request_manager import GrpcRequestManager
    from .servicer import TrtllmServiceServicer

    try:
        from grpc_reflection.v1alpha import reflection

        reflection_available = True
    except ImportError:
        reflection_available = False

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

        request_manager = GrpcRequestManager(llm)
        servicer = TrtllmServiceServicer(request_manager, model_path=model_path)
        server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", _MAX_MESSAGE_LENGTH_BYTES),
                ("grpc.max_receive_message_length", _MAX_MESSAGE_LENGTH_BYTES),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ]
        )
        trtllm_service_pb2_grpc.add_TrtllmServiceServicer_to_server(servicer, server)

        if reflection_available:
            service_names = (
                trtllm_service_pb2.DESCRIPTOR.services_by_name["TrtllmService"].full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(service_names, server)
            logger.info("gRPC reflection enabled")

        address = f"{host}:{port}"
        bound_port = server.add_insecure_port(address)
        if bound_port == 0:
            try:
                await server.stop(grace=0)
            finally:
                if hasattr(llm, "shutdown"):
                    llm.shutdown()
            raise RuntimeError(f"Failed to bind SMG gRPC server to {address}")
        await server.start()
        logger.info(f"TensorRT-LLM SMG gRPC server started on {host}:{bound_port}")
        logger.info("Server is ready to accept requests")

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def signal_handler() -> None:
            logger.info("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            logger.info("Shutting down TensorRT-LLM SMG gRPC server...")
            await server.stop(grace=5.0)
            logger.info("gRPC server stopped")

            if hasattr(llm, "shutdown"):
                llm.shutdown()
            logger.info("LLM engine stopped")
            logger.info("Shutdown complete")

    uvloop.run(serve_grpc_async())
