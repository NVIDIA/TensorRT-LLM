# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EncodedMediaRef(_message.Message):
    __slots__ = ("reference_format", "opaque_state")
    REFERENCE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    OPAQUE_STATE_FIELD_NUMBER: _ClassVar[int]
    reference_format: str
    opaque_state: bytes
    def __init__(self, reference_format: _Optional[str] = ..., opaque_state: _Optional[bytes] = ...) -> None: ...
