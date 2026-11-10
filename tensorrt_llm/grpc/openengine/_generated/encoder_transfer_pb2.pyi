# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from . import generation_pb2 as _generation_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TransferEncoderCacheRequest(_message.Message):
    __slots__ = ("cache", "destination")
    CACHE_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_FIELD_NUMBER: _ClassVar[int]
    cache: _generation_pb2.EncoderHandoffRef
    destination: bytes
    def __init__(self, cache: _Optional[_Union[_generation_pb2.EncoderHandoffRef, _Mapping]] = ..., destination: _Optional[bytes] = ...) -> None: ...

class TransferEncoderCacheResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
