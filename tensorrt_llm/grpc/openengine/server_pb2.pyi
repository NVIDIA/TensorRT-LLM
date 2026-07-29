from google.protobuf import struct_pb2 as _struct_pb2
from . import kv_pb2 as _kv_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EngineRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENGINE_ROLE_UNSPECIFIED: _ClassVar[EngineRole]
    ENGINE_ROLE_AGGREGATED: _ClassVar[EngineRole]
    ENGINE_ROLE_PREFILL: _ClassVar[EngineRole]
    ENGINE_ROLE_DECODE: _ClassVar[EngineRole]
ENGINE_ROLE_UNSPECIFIED: EngineRole
ENGINE_ROLE_AGGREGATED: EngineRole
ENGINE_ROLE_PREFILL: EngineRole
ENGINE_ROLE_DECODE: EngineRole

class GetServerInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("engine_name", "engine_version", "engine_role", "instance_id", "supported_models", "parallelism", "kv_connector", "schema_revision", "minimum_client_revision", "schema_release", "capacity", "extra")
    ENGINE_NAME_FIELD_NUMBER: _ClassVar[int]
    ENGINE_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENGINE_ROLE_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_MODELS_FIELD_NUMBER: _ClassVar[int]
    PARALLELISM_FIELD_NUMBER: _ClassVar[int]
    KV_CONNECTOR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_REVISION_FIELD_NUMBER: _ClassVar[int]
    MINIMUM_CLIENT_REVISION_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_RELEASE_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    engine_name: str
    engine_version: str
    engine_role: EngineRole
    instance_id: str
    supported_models: _containers.RepeatedScalarFieldContainer[str]
    parallelism: ParallelismInfo
    kv_connector: _kv_pb2.KvConnectorInfo
    schema_revision: int
    minimum_client_revision: int
    schema_release: str
    capacity: DeploymentCapacity
    extra: _struct_pb2.Struct
    def __init__(self, engine_name: _Optional[str] = ..., engine_version: _Optional[str] = ..., engine_role: _Optional[_Union[EngineRole, str]] = ..., instance_id: _Optional[str] = ..., supported_models: _Optional[_Iterable[str]] = ..., parallelism: _Optional[_Union[ParallelismInfo, _Mapping]] = ..., kv_connector: _Optional[_Union[_kv_pb2.KvConnectorInfo, _Mapping]] = ..., schema_revision: _Optional[int] = ..., minimum_client_revision: _Optional[int] = ..., schema_release: _Optional[str] = ..., capacity: _Optional[_Union[DeploymentCapacity, _Mapping]] = ..., extra: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class DeploymentCapacity(_message.Message):
    __slots__ = ("kv_block_size", "total_kv_blocks", "max_running_requests", "max_batched_tokens", "max_loras")
    KV_BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_KV_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    MAX_RUNNING_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_LORAS_FIELD_NUMBER: _ClassVar[int]
    kv_block_size: int
    total_kv_blocks: int
    max_running_requests: int
    max_batched_tokens: int
    max_loras: int
    def __init__(self, kv_block_size: _Optional[int] = ..., total_kv_blocks: _Optional[int] = ..., max_running_requests: _Optional[int] = ..., max_batched_tokens: _Optional[int] = ..., max_loras: _Optional[int] = ...) -> None: ...

class ParallelismInfo(_message.Message):
    __slots__ = ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "data_parallel_rank", "data_parallel_start_rank", "decode_context_parallel_size")
    TENSOR_PARALLEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PARALLEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_RANK_FIELD_NUMBER: _ClassVar[int]
    DATA_PARALLEL_START_RANK_FIELD_NUMBER: _ClassVar[int]
    DECODE_CONTEXT_PARALLEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    data_parallel_rank: int
    data_parallel_start_rank: int
    decode_context_parallel_size: int
    def __init__(self, tensor_parallel_size: _Optional[int] = ..., pipeline_parallel_size: _Optional[int] = ..., data_parallel_size: _Optional[int] = ..., data_parallel_rank: _Optional[int] = ..., data_parallel_start_rank: _Optional[int] = ..., decode_context_parallel_size: _Optional[int] = ...) -> None: ...

class GetLoadRequest(_message.Message):
    __slots__ = ("include_per_rank",)
    INCLUDE_PER_RANK_FIELD_NUMBER: _ClassVar[int]
    include_per_rank: bool
    def __init__(self, include_per_rank: bool = ...) -> None: ...

class LoadInfo(_message.Message):
    __slots__ = ("instance_id", "timestamp_unix_nanos", "running_requests", "queued_requests", "active_kv_sessions", "used_kv_blocks", "total_kv_blocks", "running_tokens", "waiting_tokens", "prefill_batch_size", "decode_batch_size", "ranks", "attributes")
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_UNIX_NANOS_FIELD_NUMBER: _ClassVar[int]
    RUNNING_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    QUEUED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_KV_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    USED_KV_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_KV_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    RUNNING_TOKENS_FIELD_NUMBER: _ClassVar[int]
    WAITING_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PREFILL_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    DECODE_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    RANKS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    instance_id: str
    timestamp_unix_nanos: int
    running_requests: int
    queued_requests: int
    active_kv_sessions: int
    used_kv_blocks: int
    total_kv_blocks: int
    running_tokens: int
    waiting_tokens: int
    prefill_batch_size: int
    decode_batch_size: int
    ranks: _containers.RepeatedCompositeFieldContainer[RankLoadInfo]
    attributes: _struct_pb2.Struct
    def __init__(self, instance_id: _Optional[str] = ..., timestamp_unix_nanos: _Optional[int] = ..., running_requests: _Optional[int] = ..., queued_requests: _Optional[int] = ..., active_kv_sessions: _Optional[int] = ..., used_kv_blocks: _Optional[int] = ..., total_kv_blocks: _Optional[int] = ..., running_tokens: _Optional[int] = ..., waiting_tokens: _Optional[int] = ..., prefill_batch_size: _Optional[int] = ..., decode_batch_size: _Optional[int] = ..., ranks: _Optional[_Iterable[_Union[RankLoadInfo, _Mapping]]] = ..., attributes: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class RankLoadInfo(_message.Message):
    __slots__ = ("data_parallel_rank", "running_requests", "queued_requests", "used_kv_blocks", "total_kv_blocks", "prefill_batch_size", "decode_batch_size")
    DATA_PARALLEL_RANK_FIELD_NUMBER: _ClassVar[int]
    RUNNING_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    QUEUED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    USED_KV_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_KV_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    PREFILL_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    DECODE_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    data_parallel_rank: int
    running_requests: int
    queued_requests: int
    used_kv_blocks: int
    total_kv_blocks: int
    prefill_batch_size: int
    decode_batch_size: int
    def __init__(self, data_parallel_rank: _Optional[int] = ..., running_requests: _Optional[int] = ..., queued_requests: _Optional[int] = ..., used_kv_blocks: _Optional[int] = ..., total_kv_blocks: _Optional[int] = ..., prefill_batch_size: _Optional[int] = ..., decode_batch_size: _Optional[int] = ...) -> None: ...
