from dataclasses import dataclass, field
from enum import auto
from typing import Dict, List, Optional, Union

from strenum import LowercaseStrEnum

from tensorrt_llm._utils import BaseEnumMeta, DictConversion

from .cluster_info import ClusterInfo, cluster_infos


class CostModel(LowercaseStrEnum, metaclass=BaseEnumMeta):
    ALPHA_BETA = auto()
    PROFILE = auto()
    S_CURVE = auto()
    # Zero cost model is for test purpose.
    # Use zero cost model for communication will make solver prefer sharding
    # Use zero cost model for computation will make solver prefer replication
    ZERO = auto()


@dataclass
class AutoParallelConfig(DictConversion):
    # cluster configuration
    world_size: int = 1
    gpus_per_node: int = 8
    cluster_key: str = None
    cluster_info: Optional[ClusterInfo] = None

    # cost model configuration
    sharding_cost_model: str = CostModel.ALPHA_BETA
    comm_cost_model: str = CostModel.ALPHA_BETA

    # strategy configuration
    enable_pipeline_parallelism: bool = False
    enable_shard_unbalanced_shape: bool = False
    enable_shard_dynamic_shape: bool = False
    enable_reduce_scatter: bool = True

    # parallelization configuration
    builder_flags: Optional[int] = None
    debug_mode: bool = False
    infer_shape: bool = True
    validation_mode: bool = False
    same_buffer_io: Dict[str, str] = field(default_factory=dict)
    same_spec_io: Dict[str, str] = field(default_factory=dict)
    sharded_io_allowlist: List[str] = field(default_factory=list)
    fill_weights: bool = False

    # debug configuration
    parallel_config_cache: Optional[str] = None
    profile_cache: Optional[str] = None
    dump_path: Optional[str] = None
    debug_outputs: Union[List[str], str] = field(default_factory=list)

    def get_cluster_info(self) -> ClusterInfo:
        return self.cluster_info or cluster_infos[self.cluster_key]

    @property
    def enabled(self) -> bool:
        return self.world_size > 1
