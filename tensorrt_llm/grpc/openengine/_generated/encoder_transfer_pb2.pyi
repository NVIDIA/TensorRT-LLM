# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TransferEncoderCacheRequest(_message.Message):
    __slots__ = ("reference", "destination")
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_FIELD_NUMBER: _ClassVar[int]
    reference: bytes
    destination: bytes
    def __init__(self, reference: _Optional[bytes] = ..., destination: _Optional[bytes] = ...) -> None: ...

class TransferEncoderCacheResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
