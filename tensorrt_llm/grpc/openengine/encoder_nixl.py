# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private gRPC transport adapter for native encoder-output transfer."""

import grpc

from tensorrt_llm._torch.disaggregation.encoder_nixl import NixlEncoderCache
from tensorrt_llm._torch.disaggregation.encoder_output import (
    EncodedMediaRef,
    EncoderTransferRejected,
)

from .bindings import encoder_transfer_pb2, encoder_transfer_pb2_grpc
from .errors import ENCODER_TRANSFER_REJECTED_METADATA

__all__ = ["NixlEncoderCache", "request_transfer"]


def request_transfer(
    endpoint: str, reference: EncodedMediaRef, destination: bytes, timeout: float
) -> None:
    """Keep serving transport and wire errors outside the native output manager."""
    try:
        with grpc.insecure_channel(endpoint) as channel:
            encoder_transfer_pb2_grpc.EncoderTransferStub(channel).Transfer(
                encoder_transfer_pb2.TransferEncoderCacheRequest(
                    reference=reference.opaque_state, destination=destination
                ),
                timeout=timeout,
            )
    except grpc.RpcError as error:
        if error.code() == grpc.StatusCode.INVALID_ARGUMENT and (
            ENCODER_TRANSFER_REJECTED_METADATA in (error.trailing_metadata() or ())
        ):
            raise EncoderTransferRejected(error.details()) from error
        raise
