from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm.bindings.executor import ExecutorConfig

from ...llmapi.llm_args import LoadFormat, SamplerType
from ...logger import logger
from ...mapping import Mapping
from ..model_config import MoeLoadBalancerConfig
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
    use_cuda_graph: bool = True
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

    attention_dp_enable_balance: bool = False
    attention_dp_time_out_iters: int = 50
    attention_dp_batching_wait_iters: int = 10

    max_num_tokens: int = 8192

    batch_wait_timeout_ms: float = 0
    # Iterations to wait before scheduling context even if token budget not reached (0 disables).
    batch_wait_timeout_iters: int = 0
    # Threshold ratio of max_num_tokens for token accumulation before scheduling context.
    # Value range: [0, 1] (0 disables).
    batch_wait_max_tokens_ratio: float = 0.0

    attn_backend: str = 'TRTLLM'
    moe_backend: str = 'CUTLASS'

    moe_disable_finalize_fusion: bool = False
    use_low_precision_moe_combine: bool = False

    sampler_type: SamplerType = SamplerType.auto
    """
    The type of sampler to use. Options are TRTLLMSampler, TorchSampler or auto.
    Defaults to auto, which will use TorchSampler unless BeamSearch is requested.
    """

    kv_cache_dtype: str = "auto"
    mamba_ssm_cache_dtype: str = "auto"

    enable_iter_perf_stats: bool = False
    # If true, enables per request stats per iteration
    # Must also set enable_iter_perf_stats to true to get request stats
    enable_iter_req_stats: bool = False
    print_iter_log: bool = False

    torch_compile_enabled: bool = False
    torch_compile_fullgraph: bool = True
    torch_compile_inductor_enabled: bool = False
    torch_compile_piecewise_cuda_graph: bool = False
    torch_compile_piecewise_cuda_graph_num_tokens: Optional[List[int]] = None
    # When torch compile is enabled, userbuffers is enabled by default
    torch_compile_enable_userbuffers: bool = True
    torch_compile_max_num_streams: int = 1

    # Enable autotuner only when torch compile is enabled
    # TODO: after it can be work stable in warmup stage
    enable_autotuner: bool = True

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

    force_dynamic_quantization: bool = False

    # If true, ONLY the vision encoder part of the full model is loaded/executed.
    mm_encoder_only: bool = False

    # If true, adjust PyTorch CUDA memory fraction to correspond to the
    # total GPU memory minus the statically allocated engine memory.
    # If false, set the PyTorch CUDA memory fraction to 1.0.
    _limit_torch_cuda_mem_fraction: bool = True


EXETENDED_EXECUTOR_CONFIG_FIELDS = [
    'backend',
    'pytorch_backend_config',
    'max_seq_len',
    'mapping',
    'hf_model_dir',
    'mm_encoder_only',
]


def update_executor_config(
        executor_config: ExecutorConfig,
        backend: Optional[str] = None,
        pytorch_backend_config: Optional[PyTorchConfig] = None,
        mapping: Optional[Mapping] = None,
        speculative_config: Optional["DecodingBaseConfig"] = None,
        hf_model_dir: Optional[str] = None,
        max_input_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        checkpoint_format: Optional[str] = None,
        checkpoint_loader: Optional[BaseCheckpointLoader] = None,
        mm_encoder_only: bool = False):
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
    executor_config.mm_encoder_only = mm_encoder_only

    logger.info(f"{executor_config.pytorch_backend_config}")

    executor_config.hf_model_dir = hf_model_dir

    if max_input_len is not None:
        executor_config.max_input_len = max_input_len

    if max_seq_len is not None:
        executor_config.max_seq_len = max_seq_len

    executor_config.checkpoint_loader = _construct_checkpoint_loader(
        backend, checkpoint_loader, checkpoint_format)


def _construct_checkpoint_loader(
        backend: str, checkpoint_loader: Optional[BaseCheckpointLoader],
        checkpoint_format: Optional[str]) -> Optional[BaseCheckpointLoader]:
    if backend == "_autodeploy":
        return None

    from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
        BaseCheckpointLoader
    from tensorrt_llm._torch.models.modeling_utils import (
        get_checkpoint_weight_loader, get_config_loader)

    if checkpoint_loader is None:
        checkpoint_weight_loader = get_checkpoint_weight_loader(
            checkpoint_format)()
        config_loader = get_config_loader(checkpoint_format)()

        checkpoint_loader = BaseCheckpointLoader.get(
            checkpoint_format=checkpoint_format,
            weight_loader=checkpoint_weight_loader,
            weight_mapper=None,
            config_loader=config_loader)

    return checkpoint_loader
