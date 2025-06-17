from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from tensorrt_llm.bindings.executor import ExecutorConfig

from ...builder import BuildConfig
from ...llmapi.llm_args import LoadFormat
from ...logger import logger
from ...mapping import Mapping
from ..model_config import MoeLoadBalancerConfig
from ..speculative import SpecConfig
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
    disable_overlap_scheduler: bool = False
    # If set, at most moe_max_num_tokens tokens will be sent to torch.ops.trtllm.fused_moe at the same time.
    # If the number of tokens exceeds moe_max_num_tokens, the input tensors will be split into chunks and a for loop will be used.
    moe_max_num_tokens: Optional[int] = None
    moe_load_balancer: Optional[Union[MoeLoadBalancerConfig, dict, str]] = None

    attn_backend: str = 'TRTLLM'
    moe_backend: str = 'CUTLASS'

    mixed_sampler: bool = False
    """
    If true, will iterate over sampling_params of each request and use the
    corresponding sampling strategy, e.g. top-k, top-p, etc.
    """
    enable_trtllm_sampler: bool = False
    """
    If true, will use the TRTLLM sampler instead of the PyTorch sampler.
    The TRTLLM sampler has a wide coverage of sampling strategies.
    """

    kv_cache_dtype: str = "auto"
    enable_iter_perf_stats: bool = False
    # If true, enables per request stats per iteration
    # Must also set enable_iter_perf_stats to true to get request stats
    enable_iter_req_stats: bool = False
    print_iter_log: bool = False

    torch_compile_enabled: bool = False
    torch_compile_fullgraph: bool = True
    torch_compile_inductor_enabled: bool = False
    torch_compile_piecewise_cuda_graph: bool = False
    # When torch compile is enabled, userbuffers is enabled by default
    torch_compile_enable_userbuffers: bool = True

    # Enable autotuner only when torch compile is enabled
    # TODO: after it can be work stable in warmup stage
    autotuner_enabled: bool = True

    # If true, enable layerwise nvtx marker
    enable_layerwise_nvtx_marker: bool = False
    # How to load the model weights. By default, detect the weight type
    # from the model checkpoint.
    load_format: Union[str, LoadFormat] = 'auto'

    # If true, enable min-latency mode. Currently only used for Llama4.
    enable_min_latency: bool = False
    allreduce_strategy: str = "AUTO"

    # The iteration interval to create responses under the streaming mode.
    # TODO: make this a per-request parameter
    stream_interval: int = 1


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
        speculative_config: Optional[SpecConfig] = None,
        hf_model_dir: Optional[str] = None,
        trt_engine_dir: Optional[str] = None,
        max_input_len: Optional[int] = None,
        max_seq_len: Optional[int] = None):
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
    executor_config.speculative_config = speculative_config

    logger.info(f"{executor_config.pytorch_backend_config}")

    build_config = build_config or BuildConfig()
    # TODO: move to pure-Python KvCacheConfig, and remove dependency on build_config.
    executor_config.tokens_per_block = executor_config.tokens_per_block or build_config.plugin_config.tokens_per_block

    executor_config.hf_model_dir = hf_model_dir
    executor_config.trt_engine_dir = trt_engine_dir

    if max_input_len is not None:
        executor_config.max_input_len = max_input_len

    if max_seq_len is not None:
        executor_config.max_seq_len = max_seq_len
