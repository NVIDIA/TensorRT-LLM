# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from . import encoded_media_pb2 as _encoded_media_pb2
from . import generation_pb2 as _generation_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EncodeRequest(_message.Message):
    __slots__ = ("request_id", "model", "prompt", "token_ids", "media")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    MEDIA_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    model: str
    prompt: str
    token_ids: _generation_pb2.TokenIds
    media: _containers.RepeatedCompositeFieldContainer[_generation_pb2.MediaItem]
    def __init__(self, request_id: _Optional[str] = ..., model: _Optional[str] = ..., prompt: _Optional[str] = ..., token_ids: _Optional[_Union[_generation_pb2.TokenIds, _Mapping]] = ..., media: _Optional[_Iterable[_Union[_generation_pb2.MediaItem, _Mapping]]] = ...) -> None: ...

class EncodeResponse(_message.Message):
    __slots__ = ("request_id", "encoded_media")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ENCODED_MEDIA_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    encoded_media: _encoded_media_pb2.EncodedMediaRef
    def __init__(self, request_id: _Optional[str] = ..., encoded_media: _Optional[_Union[_encoded_media_pb2.EncodedMediaRef, _Mapping]] = ...) -> None: ...

class ReleaseEncodedMediaRequest(_message.Message):
    __slots__ = ("encoded_media",)
    ENCODED_MEDIA_FIELD_NUMBER: _ClassVar[int]
    encoded_media: _encoded_media_pb2.EncodedMediaRef
    def __init__(self, encoded_media: _Optional[_Union[_encoded_media_pb2.EncodedMediaRef, _Mapping]] = ...) -> None: ...

class ReleaseEncodedMediaResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
