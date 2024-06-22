from .auto_parallel import auto_parallel
from .cluster_info import infer_cluster_config
from .config import AutoParallelConfig

__all__ = [
    'auto_parallel',
    'AutoParallelConfig',
    'infer_cluster_config',
]
