import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tensorrt_llm.bindings.executor import ExecutorConfig

from ...builder import BuildConfig
from ...logger import logger
from ...mapping import Mapping
from .resource_manager import BaseResourceManager


@dataclass
class PyTorchConfig:
    """
    Extra arguments for the pytorch backend.
    """

    # Extra resource managers to use in addition to the KV cache manager.
    # Each manager's prepare_resources method is called before the forward pass,
    # and update_resources() is called after the pass finishes. free_resources()
    # is called when a request finishes.
    # The KV cache manager is guaranteed to be invoked after all of these extra
    # managers in all stages.
    extra_resource_managers: Dict[str, BaseResourceManager] = field(
        default_factory=dict)

    # If true, use CUDA graphs for decoding. CUDA graphs are only created
    # for the batch sizes in cuda_graph_batch_sizes, and are enabled for
    # batches that consist of decoding requests *only* (the reason is that
    # it's hard to capture a single graph with prefill requests since the
    # input shapes are a function of the sequence lengths).
    # Note that each CUDA graph can use up to 200 MB of extra memory.
    use_cuda_graph: bool = False
    cuda_graph_batch_sizes: Optional[List[int]] = None
    cuda_graph_max_batch_size: int = 0
    # If true, batches are rounded up to the nearest cuda_graph_batch_size.
    # This is usually a net win for performance.
    cuda_graph_padding_enabled: bool = False
    enable_overlap_scheduler: bool = False

    attn_backend: str = 'TRTLLM'
    kv_cache_dtype: str = "auto"
    print_iter_log: bool = True

    torch_compile_enabled: bool = False
    torch_compile_fullgraph: bool = False
    torch_compile_inductor_enabled: bool = False

    def __post_init__(self) -> None:
        if self.cuda_graph_batch_sizes is not None:
            assert self.cuda_graph_max_batch_size == 0, (
                "Please don't set both cuda_graph_batch_sizes "
                "and cuda_graph_max_batch_size.")
            return sorted(self.cuda_graph_batch_sizes)
        self.cuda_graph_max_batch_size = self.cuda_graph_max_batch_size or 128
        if self.cuda_graph_padding_enabled:
            self.cuda_graph_batch_sizes = [1, 2, 4
                                           ] + [i * 8 for i in range(1, 17)]
        else:
            self.cuda_graph_batch_sizes = list(range(1, 32)) + [64, 128]
        self.cuda_graph_batch_sizes += [
            2**i for i in range(
                8, math.floor(math.log(self.cuda_graph_max_batch_size, 2)))
        ]
        self.cuda_graph_batch_sizes = [
            size for size in self.cuda_graph_batch_sizes
            if size <= self.cuda_graph_max_batch_size
        ]
        if self.cuda_graph_max_batch_size != self.cuda_graph_batch_sizes[-1]:
            self.cuda_graph_batch_sizes.append(self.cuda_graph_max_batch_size)


EXETENDED_EXECUTOR_CONFIG_FIELDS = [
    'backend',
    'pytorch_backend_config',
    'max_seq_len',
    'tokens_per_block',
    'mapping',
    'hf_model_dir',
    'trt_engine_dir',
]


def update_executor_config(
        executor_config: ExecutorConfig,
        backend: Optional[str] = None,
        pytorch_backend_config: Optional[PyTorchConfig] = None,
        mapping: Optional[Mapping] = None,
        build_config: Optional[BuildConfig] = None,
        hf_model_dir: str = None,
        trt_engine_dir: str = None):
    if backend is None:
        return

    for field_name in EXETENDED_EXECUTOR_CONFIG_FIELDS:
        if hasattr(executor_config, field_name):
            raise AttributeError(
                f"{field_name} should be dynamically assigned.")
        setattr(executor_config, field_name, None)

    executor_config.backend = backend
    executor_config.pytorch_backend_config = pytorch_backend_config
    executor_config.mapping = mapping

    logger.info(f"{executor_config.pytorch_backend_config}")

    if build_config is not None:
        executor_config.max_seq_len = build_config.max_seq_len
        executor_config.tokens_per_block = build_config.plugin_config.tokens_per_block

    executor_config.hf_model_dir = hf_model_dir
    executor_config.trt_engine_dir = trt_engine_dir
