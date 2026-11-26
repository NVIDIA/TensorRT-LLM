# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EncodedMediaRef(_message.Message):
    __slots__ = ("opaque_state",)
    OPAQUE_STATE_FIELD_NUMBER: _ClassVar[int]
    opaque_state: bytes
    def __init__(self, opaque_state: _Optional[bytes] = ...) -> None: ...
