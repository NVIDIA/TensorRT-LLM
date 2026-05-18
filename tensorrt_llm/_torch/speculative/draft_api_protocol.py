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
"""Draft API fixed-size wire protocol for LPU speculative transport.

Byte-compatible port of the izzy reference implementation
(``doca_rdma/standalone/protocol.py`` and
``tensorrt_llm/_torch/speculative/draft_api_protocol.py``).

The wire format must stay frozen — interop with izzy LPU servers depends
on the exact 96-byte layout below.  Subsequent phases consume the encoder
through the channel/endpoint layers; never reformat the bytes here.
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum

_UINT32_MAX = (1 << 32) - 1
_UINT8_MAX = (1 << 8) - 1
_VERSION_V1 = 1
_MESSAGE_BYTES = 96
_MAX_TOKENS = 20
_RESERVED_OFFSET = 11
_RESERVED_BYTES = 5
# Layout: little-endian, 2 uint32 (round_seq, position), 3 uint8
# (version, message_type, num_tokens), 5 reserved bytes, 20 uint32 tokens.
_WIRE_STRUCT = struct.Struct("<IIBBB5s20I")


class DraftApiProtocol:
    """Strict encoder/decoder for the Draft API v1 wire format."""

    kVersionV1 = _VERSION_V1
    kMessageBytes = _MESSAGE_BYTES
    kMaxTokens = _MAX_TOKENS
    kReservedOffset = _RESERVED_OFFSET
    kReservedBytes = _RESERVED_BYTES

    class MessageType(IntEnum):
        kDraftToTarget = 0
        kTargetToDraft = 1
        # PEARL extensions (parallel speculative decoding with adaptive draft length).
        # The 96-byte envelope is unchanged; only validation rules differ.
        # tokens[0] carries gamma_next; tokens[1:1+num_to_verify] carries accepted tokens.
        kPearlVerifyContinue = 2
        # Single token emitted after draft-step #1 so target can begin pre-verify.
        kPearlPreVerifyToken = 3
        # All gamma draft tokens emitted after draft-step #gamma.
        kPearlDraftBatch = 4
        # Rollback signal: position carries rollback_count, tokens[0] carries correct_token.
        kPearlRollback = 5
        # Bidirectional probe used by adaptive-gamma profiling.
        kPearlProbe = 6

    class Status(IntEnum):
        kOk = 0
        kInvalidBuffer = 1
        kInvalidSize = 2
        kUnsupportedVersion = 3
        kInvalidMessageType = 4
        kInvalidNumTokens = 5
        kInvalidPosition = 6
        kReservedFieldNonZero = 7

    @dataclass
    class Message:
        version: int = _VERSION_V1
        message_type: "DraftApiProtocol.MessageType | int" = 0
        round_seq_num: int = 0
        position: int = 0
        num_tokens: int = 0
        tokens: list = field(default_factory=lambda: [0] * _MAX_TOKENS)

    @dataclass
    class ValidationOptions:
        max_position: int = _UINT32_MAX
        enforce_reserved_zeros: bool = True

    @staticmethod
    def _status(status):
        if isinstance(status, DraftApiProtocol.Status):
            return status
        try:
            return DraftApiProtocol.Status(int(status))
        except (TypeError, ValueError):
            return DraftApiProtocol.Status.kInvalidBuffer

    @staticmethod
    def _to_message_type(value):
        if isinstance(value, DraftApiProtocol.MessageType):
            return DraftApiProtocol.Status.kOk, value
        try:
            return DraftApiProtocol.Status.kOk, DraftApiProtocol.MessageType(int(value))
        except (TypeError, ValueError):
            return DraftApiProtocol.Status.kInvalidMessageType, None

    @staticmethod
    def _validate_uint8(value):
        return isinstance(value, int) and 0 <= value <= _UINT8_MAX

    @staticmethod
    def _validate_uint32(value):
        return isinstance(value, int) and 0 <= value <= _UINT32_MAX

    @staticmethod
    def validate(message, options=None):
        options = options or DraftApiProtocol.ValidationOptions()

        if not DraftApiProtocol._validate_uint8(message.version) or message.version != _VERSION_V1:
            return DraftApiProtocol.Status.kUnsupportedVersion

        status, message_type = DraftApiProtocol._to_message_type(message.message_type)
        if status != DraftApiProtocol.Status.kOk:
            return status
        assert message_type is not None

        if (
            not DraftApiProtocol._validate_uint32(message.position)
            or message.position > options.max_position
        ):
            return DraftApiProtocol.Status.kInvalidPosition

        if not isinstance(message.num_tokens, int):
            return DraftApiProtocol.Status.kInvalidNumTokens
        if message_type == DraftApiProtocol.MessageType.kTargetToDraft:
            if message.num_tokens != 1:
                return DraftApiProtocol.Status.kInvalidNumTokens
        elif message_type == DraftApiProtocol.MessageType.kDraftToTarget:
            if message.num_tokens == 1:
                # A single-token DraftToTarget reply is only legal when the
                # draft bumped position to max (i.e. exhausted budget).
                if message.position != options.max_position:
                    return DraftApiProtocol.Status.kInvalidNumTokens
            elif message.num_tokens < 2 or message.num_tokens > _MAX_TOKENS:
                return DraftApiProtocol.Status.kInvalidNumTokens
        elif message_type in (
            DraftApiProtocol.MessageType.kPearlPreVerifyToken,
            DraftApiProtocol.MessageType.kPearlRollback,
        ):
            if message.num_tokens != 1:
                return DraftApiProtocol.Status.kInvalidNumTokens
        elif message_type in (
            DraftApiProtocol.MessageType.kPearlVerifyContinue,
            DraftApiProtocol.MessageType.kPearlDraftBatch,
        ):
            if message.num_tokens < 1 or message.num_tokens > _MAX_TOKENS:
                return DraftApiProtocol.Status.kInvalidNumTokens
        elif message_type == DraftApiProtocol.MessageType.kPearlProbe:
            if message.num_tokens > _MAX_TOKENS:
                return DraftApiProtocol.Status.kInvalidNumTokens
        else:
            return DraftApiProtocol.Status.kInvalidMessageType

        if len(message.tokens) != _MAX_TOKENS:
            return DraftApiProtocol.Status.kInvalidNumTokens
        if not all(DraftApiProtocol._validate_uint32(token) for token in message.tokens):
            return DraftApiProtocol.Status.kInvalidBuffer
        if not DraftApiProtocol._validate_uint32(message.round_seq_num):
            return DraftApiProtocol.Status.kInvalidBuffer

        return DraftApiProtocol.Status.kOk

    @staticmethod
    def encode(message, options=None):
        """Encode a protocol message into a 96-byte wire frame."""
        options = options or DraftApiProtocol.ValidationOptions()
        status = DraftApiProtocol.validate(message, options)
        if status != DraftApiProtocol.Status.kOk:
            return status, b""

        _, message_type = DraftApiProtocol._to_message_type(message.message_type)
        assert message_type is not None

        try:
            payload = _WIRE_STRUCT.pack(
                int(message.round_seq_num),
                int(message.position),
                int(message.version),
                int(message_type),
                int(message.num_tokens),
                bytes(_RESERVED_BYTES),
                *[int(token) for token in message.tokens],
            )
        except (struct.error, TypeError, ValueError):
            return DraftApiProtocol.Status.kInvalidBuffer, b""

        return DraftApiProtocol.Status.kOk, payload

    @staticmethod
    def decode(data, options=None):
        """Decode a 96-byte wire frame into a protocol message."""
        options = options or DraftApiProtocol.ValidationOptions()

        if data is None:
            return DraftApiProtocol.Status.kInvalidBuffer, None
        try:
            raw = bytes(data)
        except TypeError:
            return DraftApiProtocol.Status.kInvalidBuffer, None
        if len(raw) != _MESSAGE_BYTES:
            return DraftApiProtocol.Status.kInvalidSize, None

        try:
            unpacked = _WIRE_STRUCT.unpack(raw)
        except struct.error:
            return DraftApiProtocol.Status.kInvalidBuffer, None

        round_seq_num, position, version, message_type, num_tokens, reserved, *tokens = unpacked
        if options.enforce_reserved_zeros and any(value != 0 for value in reserved):
            return DraftApiProtocol.Status.kReservedFieldNonZero, None

        message = DraftApiProtocol.Message(
            version=version,
            message_type=message_type,
            round_seq_num=round_seq_num,
            position=position,
            num_tokens=num_tokens,
            tokens=list(tokens),
        )
        status = DraftApiProtocol.validate(message, options)
        if status != DraftApiProtocol.Status.kOk:
            return status, None
        return DraftApiProtocol.Status.kOk, message

    @staticmethod
    def to_string(status):
        status = DraftApiProtocol._status(status)
        if status == DraftApiProtocol.Status.kOk:
            return "kOk"
        if status == DraftApiProtocol.Status.kInvalidBuffer:
            return "kInvalidBuffer"
        if status == DraftApiProtocol.Status.kInvalidSize:
            return "kInvalidSize"
        if status == DraftApiProtocol.Status.kUnsupportedVersion:
            return "kUnsupportedVersion"
        if status == DraftApiProtocol.Status.kInvalidMessageType:
            return "kInvalidMessageType"
        if status == DraftApiProtocol.Status.kInvalidNumTokens:
            return "kInvalidNumTokens"
        if status == DraftApiProtocol.Status.kInvalidPosition:
            return "kInvalidPosition"
        if status == DraftApiProtocol.Status.kReservedFieldNonZero:
            return "kReservedFieldNonZero"
        return "kUnknown"
