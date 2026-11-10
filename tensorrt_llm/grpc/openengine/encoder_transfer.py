# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM peer service for native encoder-output transfers."""

import asyncio
from typing import Any

import grpc

from .bindings import encoder_transfer_pb2, encoder_transfer_pb2_grpc
from .errors import ENCODER_TRANSFER_REJECTED_METADATA, EncoderTransferRejected


class EncoderTransferServicer(encoder_transfer_pb2_grpc.EncoderTransferServicer):
    def __init__(self, inference: Any) -> None:
        self._inference = inference

    async def Transfer(self, request, context):
        cache = getattr(self._inference, "_encoder_cache", None)
        transfer = getattr(cache, "transfer", None)
        if not transfer or getattr(self._inference, "encoder_role", "") != "encode":
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "encoder transfer is not enabled")
            return
        # Cancellation cannot release source buffers while a native WRITE may run.
        task = asyncio.create_task(asyncio.to_thread(transfer, request.cache, request.destination))
        self._inference._encoder_cleanup_tasks.add(task)

        def completed(done):
            self._inference._encoder_cleanup_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)
        try:
            await asyncio.shield(task)
        except EncoderTransferRejected as error:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                str(error),
                trailing_metadata=(ENCODER_TRANSFER_REJECTED_METADATA,),
            )
            return
        except (RuntimeError, ValueError) as error:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
            return
        return encoder_transfer_pb2.TransferEncoderCacheResponse()
