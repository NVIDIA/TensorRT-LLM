from google.protobuf import struct_pb2 as _struct_pb2
from . import error_pb2 as _error_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StorageMedium(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STORAGE_MEDIUM_UNSPECIFIED: _ClassVar[StorageMedium]
    STORAGE_MEDIUM_GPU: _ClassVar[StorageMedium]
    STORAGE_MEDIUM_CPU_PINNED: _ClassVar[StorageMedium]
    STORAGE_MEDIUM_DISK: _ClassVar[StorageMedium]
    STORAGE_MEDIUM_EXTERNAL: _ClassVar[StorageMedium]
STORAGE_MEDIUM_UNSPECIFIED: StorageMedium
STORAGE_MEDIUM_GPU: StorageMedium
STORAGE_MEDIUM_CPU_PINNED: StorageMedium
STORAGE_MEDIUM_DISK: StorageMedium
STORAGE_MEDIUM_EXTERNAL: StorageMedium

class KvSessionRef(_message.Message):
    __slots__ = ("session_id", "transfer_backend", "endpoints", "dp_rank", "attributes_struct")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_BACKEND_FIELD_NUMBER: _ClassVar[int]
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    DP_RANK_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_STRUCT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    transfer_backend: str
    endpoints: _containers.RepeatedCompositeFieldContainer[KvEndpoint]
    dp_rank: int
    attributes_struct: _struct_pb2.Struct
    def __init__(self, session_id: _Optional[str] = ..., transfer_backend: _Optional[str] = ..., endpoints: _Optional[_Iterable[_Union[KvEndpoint, _Mapping]]] = ..., dp_rank: _Optional[int] = ..., attributes_struct: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class KvEndpoint(_message.Message):
    __slots__ = ("host", "port", "protocol")
    HOST_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    host: str
    port: int
    protocol: str
    def __init__(self, host: _Optional[str] = ..., port: _Optional[int] = ..., protocol: _Optional[str] = ...) -> None: ...

class KvConnectorInfo(_message.Message):
    __slots__ = ("enabled", "transfer_backend", "local_endpoints", "supported_protocols", "supports_remote_prefill", "supports_decode_pull", "supports_abort_cleanup", "schema_version")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_BACKEND_FIELD_NUMBER: _ClassVar[int]
    LOCAL_ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_PROTOCOLS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_REMOTE_PREFILL_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_DECODE_PULL_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_ABORT_CLEANUP_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    transfer_backend: str
    local_endpoints: _containers.RepeatedCompositeFieldContainer[KvEndpoint]
    supported_protocols: _containers.RepeatedScalarFieldContainer[str]
    supports_remote_prefill: bool
    supports_decode_pull: bool
    supports_abort_cleanup: bool
    schema_version: int
    def __init__(self, enabled: bool = ..., transfer_backend: _Optional[str] = ..., local_endpoints: _Optional[_Iterable[_Union[KvEndpoint, _Mapping]]] = ..., supported_protocols: _Optional[_Iterable[str]] = ..., supports_remote_prefill: bool = ..., supports_decode_pull: bool = ..., supports_abort_cleanup: bool = ..., schema_version: _Optional[int] = ...) -> None: ...

class GetKvEventSourcesRequest(_message.Message):
    __slots__ = ("data_parallel_ranks",)
    DATA_PARALLEL_RANKS_FIELD_NUMBER: _ClassVar[int]
    data_parallel_ranks: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data_parallel_ranks: _Optional[_Iterable[int]] = ...) -> None: ...

class GetKvEventSourcesResponse(_message.Message):
    __slots__ = ("sources",)
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    sources: _containers.RepeatedCompositeFieldContainer[KvEventSource]
    def __init__(self, sources: _Optional[_Iterable[_Union[KvEventSource, _Mapping]]] = ...) -> None: ...

class KvEventSource(_message.Message):
    __slots__ = ("transport", "endpoint_addr", "topic", "replay_endpoint", "data_parallel_rank", "encoding", "schema_version", "buffer_steps", "hwm", "max_queue_size")
    TRANSPORT_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_ADDR_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    REPLAY_ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_RANK_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    BUFFER_STEPS_FIELD_NUMBER: _ClassVar[int]
    HWM_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    transport: str
    endpoint_addr: KvEndpoint
    topic: str
    replay_endpoint: str
    data_parallel_rank: int
    encoding: str
    schema_version: int
    buffer_steps: int
    hwm: int
    max_queue_size: int
    def __init__(self, transport: _Optional[str] = ..., endpoint_addr: _Optional[_Union[KvEndpoint, _Mapping]] = ..., topic: _Optional[str] = ..., replay_endpoint: _Optional[str] = ..., data_parallel_rank: _Optional[int] = ..., encoding: _Optional[str] = ..., schema_version: _Optional[int] = ..., buffer_steps: _Optional[int] = ..., hwm: _Optional[int] = ..., max_queue_size: _Optional[int] = ...) -> None: ...

class SubscribeKvEventsRequest(_message.Message):
    __slots__ = ("data_parallel_ranks", "include_snapshot", "start_sequence_number")
    DATA_PARALLEL_RANKS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    START_SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    data_parallel_ranks: _containers.RepeatedScalarFieldContainer[int]
    include_snapshot: bool
    start_sequence_number: int
    def __init__(self, data_parallel_ranks: _Optional[_Iterable[int]] = ..., include_snapshot: bool = ..., start_sequence_number: _Optional[int] = ...) -> None: ...

class SubscribeKvEventsResponse(_message.Message):
    __slots__ = ("batch", "error")
    BATCH_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    batch: KvEventBatch
    error: _error_pb2.EngineError
    def __init__(self, batch: _Optional[_Union[KvEventBatch, _Mapping]] = ..., error: _Optional[_Union[_error_pb2.EngineError, _Mapping]] = ...) -> None: ...

class KvEventBatch(_message.Message):
    __slots__ = ("sequence_number", "timestamp_unix_nanos", "data_parallel_rank", "events")
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_UNIX_NANOS_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_RANK_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    sequence_number: int
    timestamp_unix_nanos: int
    data_parallel_rank: int
    events: _containers.RepeatedCompositeFieldContainer[KvEvent]
    def __init__(self, sequence_number: _Optional[int] = ..., timestamp_unix_nanos: _Optional[int] = ..., data_parallel_rank: _Optional[int] = ..., events: _Optional[_Iterable[_Union[KvEvent, _Mapping]]] = ...) -> None: ...

class KvEvent(_message.Message):
    __slots__ = ("request_id", "kv_session", "block_stored", "block_removed", "all_blocks_cleared")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    KV_SESSION_FIELD_NUMBER: _ClassVar[int]
    BLOCK_STORED_FIELD_NUMBER: _ClassVar[int]
    BLOCK_REMOVED_FIELD_NUMBER: _ClassVar[int]
    ALL_BLOCKS_CLEARED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    kv_session: KvSessionRef
    block_stored: BlockStored
    block_removed: BlockRemoved
    all_blocks_cleared: AllBlocksCleared
    def __init__(self, request_id: _Optional[str] = ..., kv_session: _Optional[_Union[KvSessionRef, _Mapping]] = ..., block_stored: _Optional[_Union[BlockStored, _Mapping]] = ..., block_removed: _Optional[_Union[BlockRemoved, _Mapping]] = ..., all_blocks_cleared: _Optional[_Union[AllBlocksCleared, _Mapping]] = ...) -> None: ...

class BlockStored(_message.Message):
    __slots__ = ("block_hashes", "parent_block_hash", "token_ids", "block_size", "lora_id", "lora_name", "medium", "extra_keys", "group_idx", "kv_cache_spec_kind", "kv_cache_spec_sliding_window")
    BLOCK_HASHES_FIELD_NUMBER: _ClassVar[int]
    PARENT_BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    LORA_ID_FIELD_NUMBER: _ClassVar[int]
    LORA_NAME_FIELD_NUMBER: _ClassVar[int]
    MEDIUM_FIELD_NUMBER: _ClassVar[int]
    EXTRA_KEYS_FIELD_NUMBER: _ClassVar[int]
    GROUP_IDX_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_SPEC_KIND_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_SPEC_SLIDING_WINDOW_FIELD_NUMBER: _ClassVar[int]
    block_hashes: _containers.RepeatedCompositeFieldContainer[KvBlockHash]
    parent_block_hash: KvBlockHash
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    block_size: int
    lora_id: int
    lora_name: str
    medium: StorageMedium
    extra_keys: _containers.RepeatedCompositeFieldContainer[OpaqueKeyTuple]
    group_idx: int
    kv_cache_spec_kind: str
    kv_cache_spec_sliding_window: int
    def __init__(self, block_hashes: _Optional[_Iterable[_Union[KvBlockHash, _Mapping]]] = ..., parent_block_hash: _Optional[_Union[KvBlockHash, _Mapping]] = ..., token_ids: _Optional[_Iterable[int]] = ..., block_size: _Optional[int] = ..., lora_id: _Optional[int] = ..., lora_name: _Optional[str] = ..., medium: _Optional[_Union[StorageMedium, str]] = ..., extra_keys: _Optional[_Iterable[_Union[OpaqueKeyTuple, _Mapping]]] = ..., group_idx: _Optional[int] = ..., kv_cache_spec_kind: _Optional[str] = ..., kv_cache_spec_sliding_window: _Optional[int] = ...) -> None: ...

class BlockRemoved(_message.Message):
    __slots__ = ("block_hashes", "medium", "group_idx")
    BLOCK_HASHES_FIELD_NUMBER: _ClassVar[int]
    MEDIUM_FIELD_NUMBER: _ClassVar[int]
    GROUP_IDX_FIELD_NUMBER: _ClassVar[int]
    block_hashes: _containers.RepeatedCompositeFieldContainer[KvBlockHash]
    medium: StorageMedium
    group_idx: int
    def __init__(self, block_hashes: _Optional[_Iterable[_Union[KvBlockHash, _Mapping]]] = ..., medium: _Optional[_Union[StorageMedium, str]] = ..., group_idx: _Optional[int] = ...) -> None: ...

class AllBlocksCleared(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class KvBlockHash(_message.Message):
    __slots__ = ("value", "encoding")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    value: bytes
    encoding: str
    def __init__(self, value: _Optional[bytes] = ..., encoding: _Optional[str] = ...) -> None: ...

class OpaqueKeyTuple(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...
