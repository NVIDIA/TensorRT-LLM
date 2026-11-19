# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map public encoder references and inputs to the native TRT-LLM interface."""

from typing import Any

from tensorrt_llm._torch.disaggregation.encoder_output import (
    EncodedMediaRef,
    EncoderCache,
    EncoderInput,
    EncoderMedia,
    PrefillResultExports,
    encoder_role,
    release_encoder_result,
)

from .bindings import encoded_media_pb2, generation_pb2
from .errors import UnsupportedFeatureError

__all__ = [
    "EncoderCache",
    "PrefillResultExports",
    "encoder_input",
    "encoder_role",
    "native_reference",
    "portable_reference",
    "release_encoder_result",
    "validate_encoder",
]


def encoder_input(request: Any) -> EncoderInput:
    """Preserve input identity without exposing protobuf objects to native transfer."""
    kind = request.WhichOneof("input")
    return EncoderInput(
        prompt=request.prompt if kind == "prompt" else None,
        token_ids=tuple(request.token_ids.ids) if kind == "token_ids" else None,
        media=tuple(
            EncoderMedia(
                modality=generation_pb2.Modality.Name(item.modality)
                .removeprefix("MODALITY_")
                .lower(),
                source_type=item.WhichOneof("source") or "",
                source=getattr(item, item.WhichOneof("source"))
                if item.WhichOneof("source")
                else b"",
                mime_type=item.mime_type,
                uuid=item.uuid,
            )
            for item in request.media
        ),
    )


def native_reference(reference: Any) -> EncodedMediaRef:
    """Forward the framework-owned payload without interpreting it."""
    return EncodedMediaRef(reference.reference_format, reference.opaque_state)


def portable_reference(reference: EncodedMediaRef) -> encoded_media_pb2.EncodedMediaRef:
    """Wrap a native reference for the portable transport."""
    return encoded_media_pb2.EncodedMediaRef(
        reference_format=reference.reference_format, opaque_state=reference.opaque_state
    )


def validate_encoder(request: Any, role: str) -> str:
    """Validate the Generate-side encoder consumption boundary."""
    if role == "encode":
        raise UnsupportedFeatureError("encoder-only engines require the Encode RPC")
    if not request.HasField("encoded_media"):
        if (
            role == "consume"
            and request.media
            and not (request.HasField("kv") and request.kv.HasField("session"))
        ):
            raise UnsupportedFeatureError("encoder consumer requires encoded_media for images")
        return ""
    if role != "consume":
        raise UnsupportedFeatureError("loaded engine does not accept encoded media")
    if request.HasField("kv") and request.kv.HasField("session"):
        raise UnsupportedFeatureError("KV decode requests must not consume encoded media")
    if not request.media or any(
        item.modality != generation_pb2.MODALITY_IMAGE for item in request.media
    ):
        raise UnsupportedFeatureError("encoder handoffs require image inputs")
    return "consume"
