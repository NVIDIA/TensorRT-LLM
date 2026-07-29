from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LoraAdapter(_message.Message):
    __slots__ = ("lora_id", "lora_name", "source_path")
    LORA_ID_FIELD_NUMBER: _ClassVar[int]
    LORA_NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PATH_FIELD_NUMBER: _ClassVar[int]
    lora_id: int
    lora_name: str
    source_path: str
    def __init__(self, lora_id: _Optional[int] = ..., lora_name: _Optional[str] = ..., source_path: _Optional[str] = ...) -> None: ...

class LoadLoraRequest(_message.Message):
    __slots__ = ("adapter",)
    ADAPTER_FIELD_NUMBER: _ClassVar[int]
    adapter: LoraAdapter
    def __init__(self, adapter: _Optional[_Union[LoraAdapter, _Mapping]] = ...) -> None: ...

class LoadLoraResponse(_message.Message):
    __slots__ = ("adapter", "already_loaded")
    ADAPTER_FIELD_NUMBER: _ClassVar[int]
    ALREADY_LOADED_FIELD_NUMBER: _ClassVar[int]
    adapter: LoraAdapter
    already_loaded: bool
    def __init__(self, adapter: _Optional[_Union[LoraAdapter, _Mapping]] = ..., already_loaded: bool = ...) -> None: ...

class UnloadLoraRequest(_message.Message):
    __slots__ = ("lora_name",)
    LORA_NAME_FIELD_NUMBER: _ClassVar[int]
    lora_name: str
    def __init__(self, lora_name: _Optional[str] = ...) -> None: ...

class UnloadLoraResponse(_message.Message):
    __slots__ = ("adapter",)
    ADAPTER_FIELD_NUMBER: _ClassVar[int]
    adapter: LoraAdapter
    def __init__(self, adapter: _Optional[_Union[LoraAdapter, _Mapping]] = ...) -> None: ...

class ListLorasRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListLorasResponse(_message.Message):
    __slots__ = ("adapters",)
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    adapters: _containers.RepeatedCompositeFieldContainer[LoraAdapter]
    def __init__(self, adapters: _Optional[_Iterable[_Union[LoraAdapter, _Mapping]]] = ...) -> None: ...
