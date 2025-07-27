import bisect
import contextlib
import functools
import gc
import inspect
import math
import os
import traceback
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.speculative import (
    get_num_extra_kv_tokens, update_spec_config_from_model_config)
from tensorrt_llm._torch.speculative.mtp import SampleStateTensorsMTP
from tensorrt_llm._utils import (is_trace_enabled, nvtx_range, release_gc,
                                 torch_dtype_to_str, trace_func)
from tensorrt_llm.inputs.multimodal import (MultimodalParams,
                                            MultimodalRuntimeData)
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraConfig, LoraModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.quantization.utils.fp4_utils import float4_e2m1x2

from ..attention_backend.interface import (AttentionMetadata,
                                           AttentionRuntimeFeatures)
from ..attention_backend.trtllm import TrtllmAttentionMetadata
from ..attention_backend.utils import get_attention_backend
from ..attention_backend.vanilla import VanillaAttentionMetadata
from ..autotuner import AutoTuner, autotune
from ..compilation.backend import Backend
from ..compilation.utils import set_enable_piecewise_cuda_graph_capture_flag
from ..distributed import MPIDist
from ..distributed.communicator import init_pp_comm
from ..expert_statistic import ExpertStatistic
from ..metadata import KVCacheParams
from ..model_config import ModelConfig, MoeLoadBalancerConfig
from ..models import AutoModelForCausalLM
from ..models.modeling_utils import (DecoderModelForCausalLM, MetaInitMode,
                                     timing)
from ..modules.fused_moe.moe_load_balancer import (
    MoeLoadBalancer, MoeLoadBalancerIterContext, maybe_create_moe_load_balancer)
from ..speculative import SpecMetadata, get_spec_metadata
from ..utils import (get_model_extra_attrs, set_torch_compiling,
                     with_model_extra_attrs)
from .config import LoadFormat, PyTorchConfig
from .config_utils import is_mla
from .cuda_graph_runner import DecodingCUDAGraphRunner
from .layerwise_nvtx_marker import LayerwiseNvtxMarker
from .resource_manager import (BaseResourceManager, KVCacheManager,
                               ResourceManager, ResourceManagerType)
from .scheduler import ScheduledRequests

MAX_UINT64 = (1 << 64) - 1


class ModelEngine(ABC):

    @abstractmethod
    def get_max_num_sequences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tensors_device: Optional[SampleStateTensors],
        gather_context_logits: bool = False,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    def warmup(self, resource_manager: ResourceManager) -> None:
        """
        This method is called after the KV cache manager is initialized
        inside the given resource manager. Override to perform any
        warmup actions: instantiating CUDA graphs, running torch.compile, etc.
        """
        return


_KV_CACHE_MAP = {
    "fp8": QuantAlgo.FP8.value,
    "nvfp4": QuantAlgo.NVFP4.value,
    "auto": "auto"
}
_VALID_KV_CACHE_DTYPES = ("fp8", "auto")


def validate_and_set_kv_cache_quant(model_config: ModelConfig,
                                    pyt_kv_cache_dtype: str) -> QuantAlgo:
    logger.info(
        f'Validating KV Cache config against kv_cache_dtype="{pyt_kv_cache_dtype}"'
    )
    # Quantization from hf_quant_config.json
    kv_cache_quant = model_config.quant_config.kv_cache_quant_algo
    # PyTorch configuration quantization
    valid_pyt_quant = bool(pyt_kv_cache_dtype in _VALID_KV_CACHE_DTYPES)
    mapped_pyt_quant = _KV_CACHE_MAP.get(pyt_kv_cache_dtype, None)

    # If we're letting the checkpoint dictate the quant with auto, simply
    # return and do not modify the checkpoint.
    if pyt_kv_cache_dtype == "auto":
        logger.info(
            f'KV cache quantization set to "{pyt_kv_cache_dtype}". Using '
            "checkpoint KV quantization.")
        return

    # If we have an invalid quantization, simply raise an exception.
    if not valid_pyt_quant:
        raise ValueError(
            "Overriding KV cache quantization with an invalid type "
            f'"PyTorchConfig.kv_cache_dtype="{pyt_kv_cache_dtype}" '
            f'Accepted types are "{_VALID_KV_CACHE_DTYPES}".')

    # If we get to this point we have a valid quantization setting, but if
    # we have an existing setting and it doesn't match we shouldn't proceed.
    if kv_cache_quant is not None and mapped_pyt_quant != kv_cache_quant:
        raise RuntimeError(
            "Attempting to override KV cache quantization "
            f'"{kv_cache_quant}" with PyTorchConfig.kv_cache_dtype='
            f'"{pyt_kv_cache_dtype}". You cannot override a checkpoint with a '
            "pre-quantized KV cache that doesn't match.")

    # We have an open ended KV cache in the checkpoint
    # and we have a specified override.
    model_config.quant_config.kv_cache_quant_algo = mapped_pyt_quant


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 0,
) -> None:
    """
    This is similar to this function in SGLang with a few changes:
    https://github.com/sgl-project/sglang/blob/e074e76b31d4fff13e87a455dbc3acdaa92c537a/python/sglang/srt/model_loader/weight_utils.py#L577

    This method is used to initialize weights with dummy values for testing
    models without checkpoints. Unquantized (FP16/BF16/etc) values are generated
    from a uniform distribution over the interval (low, high).

    For some quantized types (FP8/NVFP4), torch has no built-in way to generate random values.
    We simply generate values uniformly across an interval that has been empirically verified
    to not generate NaNs/inf for these.
    """

    def _get_random_min_max(dtype: torch.dtype) -> Tuple[int, int]:
        # These values are not necessarily the largest possible min/max,
        # they need to be small enough to avoid NaNs.
        if dtype in (torch.float8_e4m3fn, torch.int8):
            return (-3.0, 3.0)

        elif dtype == float4_e2m1x2:
            # These correspond to bits of 2 packed FP4 values.
            # Because we only go up to 64, the high 4 bits will
            # always be 0. But this is fine - we just need values
            # that won't generate NaNs.
            return (0, 64)

        else:
            raise NotImplementedError(f"Unknown quantized type: {dtype}.")

    for param in model.state_dict().values():
        generator = torch.Generator(device=param.data.device)
        generator.manual_seed(seed)
        dtype = param.data.dtype

        if param.data.element_size() < 2:
            # We need to do a cast/round since torch doesn't have uniform_
            # support for these dtypes.
            tmp_param = torch.empty(param.data.shape,
                                    dtype=torch.float16,
                                    device=param.data.device)

            quant_min, quant_max = _get_random_min_max(dtype)
            tmp_param = tmp_param.uniform_(quant_min,
                                           quant_max,
                                           generator=generator)

            param.data.copy_(tmp_param.to(dtype))

        # Note: no need to to mess with int32 params, these are probably
        # constants and not weights.
        elif torch.is_floating_point(param):
            param.uniform_(low, high, generator=generator)


def get_rank_model_storage(model):
    total_bytes = 0
    for _, param in model.named_parameters():
        if param.device.type == 'cuda' and param.device.index == torch.cuda.current_device(
        ):
            total_bytes += param.element_size() * param.nelement()
    for _, buf in model.named_buffers():
        if buf.device.type == 'cuda' and buf.device.index == torch.cuda.current_device(
        ):
            total_bytes += buf.element_size() * buf.nelement()
    return total_bytes


def _filter_cuda_graph_batch_sizes(cuda_graph_batch_sizes: list[int],
                                   max_batch_size: int, max_num_tokens: int,
                                   max_draft_len: int,
                                   enable_padding: bool) -> list[int]:
    # This is the largest possible batch size for a pure decoding batch.
    max_cuda_graph_bs = min(max_batch_size,
                            int(max_num_tokens / (1 + max_draft_len)))

    result = []
    # This function assumes cuda_graph_batch_sizes is sorted
    for i, bs in enumerate(cuda_graph_batch_sizes):
        if bs <= max_cuda_graph_bs:
            result.append(bs)
        else:
            # One extra special case for padding. The user gave us at least
            # one batch size to pad to which is larger than the executor's max
            # batch size. In this case, padding to max_cuda_graph_bs is acceptable. The logic
            # is that if the user is OK padding to a batch size B, they should also
            # be OK with padding to some size B' < B since the performance will generally
            # just be better in the smaller case.
            if enable_padding and (i == 0
                                   or result[i - 1] != max_cuda_graph_bs):
                logger.warning(
                    "CUDA graph padding is enabled, but one of the given CUDA graph "
                    f"batch sizes ({bs}) is larger than the executor's max batch size "
                    f"({max_cuda_graph_bs}). We will pad batches to {max_cuda_graph_bs}."
                )
                result.append(max_cuda_graph_bs)
            break

    return result


class PyTorchModelEngine(ModelEngine):

    def __init__(
        self,
        *,
        model_path: str,
        pytorch_backend_config: PyTorchConfig,
        checkpoint_loader: BaseCheckpointLoader,
        batch_size: int = 8,
        max_beam_width: int = 1,
        max_num_tokens: int = 8192,
        max_seq_len: Optional[int] = None,
        mapping: Optional[Mapping] = None,
        attn_runtime_features: Optional[AttentionRuntimeFeatures] = None,
        dist: Optional[MPIDist] = None,
        spec_config: Optional["DecodingBaseConfig"] = None,
        lora_config: Optional[LoraConfig] = None,
        is_draft_model: bool = False,
    ):
        self.ub_buffers = None
        self.batch_size = batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.max_beam_width = max_beam_width

        self.mapping = mapping
        if mapping.has_pp():
            init_pp_comm(mapping)
        self.dist = dist
        if dist is not None:
            ExpertStatistic.create(self.dist.rank)
        self.pytorch_backend_config = pytorch_backend_config
        self.spec_config = spec_config
        self.is_spec_decode = spec_config is not None
        self.is_draft_model = is_draft_model

        self.in_warmup = False

        self.attn_runtime_features = attn_runtime_features or AttentionRuntimeFeatures(
        )

        attn_backend = pytorch_backend_config.attn_backend
        self.model = self._load_model(
            model_path,
            mapping=self.mapping,
            checkpoint_loader=checkpoint_loader,
            attn_backend=attn_backend,
            moe_backend=pytorch_backend_config.moe_backend,
            load_format=pytorch_backend_config.load_format,
            max_num_tokens=max_num_tokens,
            moe_max_num_tokens=pytorch_backend_config.moe_max_num_tokens,
            moe_load_balancer=pytorch_backend_config.moe_load_balancer,
            lora_config=lora_config)
        # In case that some tests use stub models and override `_load_model`.
        if not hasattr(self.model, 'extra_attrs'):
            self.model.extra_attrs = {}
        if self.pytorch_backend_config.enable_layerwise_nvtx_marker:
            layerwise_nvtx_marker = LayerwiseNvtxMarker()
            module_prefix = 'Model'
            if self.model.model_config and self.model.model_config.pretrained_config and self.model.model_config.pretrained_config.architectures:
                module_prefix = '|'.join(
                    self.model.model_config.pretrained_config.architectures)
            layerwise_nvtx_marker.register_hooks(self.model, module_prefix)

        self.enable_attention_dp = self.model.model_config.mapping.enable_attention_dp
        self._disable_overlap_scheduler = self.pytorch_backend_config.disable_overlap_scheduler
        self._torch_compile_backend = None
        self.dtype = self.model.config.torch_dtype
        self._init_model_capacity()

        self._torch_compile_backend = None

        try:
            if pytorch_backend_config.torch_compile_enabled:
                set_torch_compiling(True)
                use_ub = pytorch_backend_config.torch_compile_enable_userbuffers and self._init_userbuffers(
                    self.model.config.hidden_size)
                self._torch_compile_backend = Backend(
                    pytorch_backend_config.torch_compile_inductor_enabled,
                    enable_userbuffers=use_ub,
                    enable_piecewise_cuda_graph=pytorch_backend_config.
                    torch_compile_piecewise_cuda_graph,
                    cuda_graph_batch_sizes=pytorch_backend_config.
                    cuda_graph_batch_sizes,
                    max_num_streams=pytorch_backend_config.
                    torch_compile_max_num_streams)
                if isinstance(self.model, DecoderModelForCausalLM):
                    self.model.model = torch.compile(
                        self.model.model,
                        backend=self._torch_compile_backend,
                        fullgraph=pytorch_backend_config.torch_compile_fullgraph
                    )
                else:
                    self.model = torch.compile(
                        self.model,
                        backend=self._torch_compile_backend,
                        fullgraph=pytorch_backend_config.torch_compile_fullgraph
                    )
                torch._dynamo.config.cache_size_limit = 16
            else:
                set_torch_compiling(False)
        except Exception as e:
            import traceback
            traceback.print_exception(Exception, e, e.__traceback__)
            raise e
        self._torch_compile_enabled = pytorch_backend_config.torch_compile_enabled
        self._torch_compile_piecewise_cuda_graph = pytorch_backend_config.torch_compile_piecewise_cuda_graph

        self.attn_backend = get_attention_backend(attn_backend)

        if self.is_spec_decode:
            self.spec_metadata = None
            update_spec_config_from_model_config(self.spec_config,
                                                 self.model.config)
            max_num_draft_tokens = self.spec_config.max_draft_len * batch_size
            self.draft_tokens_cuda = torch.empty((max_num_draft_tokens, ),
                                                 dtype=torch.int,
                                                 device='cuda')
            self.gather_ids_cuda = torch.empty((self.max_num_tokens, ),
                                               dtype=torch.int,
                                               device='cuda')
            self.previous_pos_indices_cuda = torch.empty(
                (self.max_num_tokens, ), dtype=torch.int, device='cuda')
            self.previous_pos_id_offsets_cuda = torch.zeros(
                (self.max_num_tokens, ), dtype=torch.int, device='cuda')
            self.previous_kv_lens_offsets_cuda = torch.zeros((batch_size, ),
                                                             dtype=torch.int,
                                                             device='cuda')
            self.without_logits = self.spec_config.spec_dec_mode.without_logits(
            )
            self.max_draft_len = spec_config.max_draft_len
        else:
            self.without_logits = False
            self.max_draft_len = 0

        # This field is initialized lazily on the first forward pass.
        # This is convenient because:
        # 1) The attention metadata depends on the KV cache manager.
        # 2) The KV cache manager depends on the model configuration.
        # 3) The model configuration is not loaded until the model engine
        # is initialized.
        #
        # NOTE: This can simplified by decoupling the model config loading and
        # the model engine.
        self.attn_metadata = None
        self.iter_states = {}
        self._cuda_graphs = {}
        self._cuda_graph_mem_pool = self._torch_compile_backend._graph_pool_handle if self._torch_compile_enabled else None
        self._run_cuda_graphs = pytorch_backend_config.use_cuda_graph

        self._cuda_graph_padding_enabled = pytorch_backend_config.cuda_graph_padding_enabled

        self._cuda_graph_batch_sizes = _filter_cuda_graph_batch_sizes(
            pytorch_backend_config.cuda_graph_batch_sizes, self.batch_size,
            self.max_num_tokens, self.max_draft_len,
            self._cuda_graph_padding_enabled
        ) if pytorch_backend_config.cuda_graph_batch_sizes else []

        self._max_cuda_graph_batch_size = (self._cuda_graph_batch_sizes[-1] if
                                           self._cuda_graph_batch_sizes else 0)

        self.previous_batch_indices_cuda = torch.empty((self.max_num_tokens, ),
                                                       dtype=torch.int,
                                                       device='cuda')
        self.input_ids_cuda = torch.empty((self.max_num_tokens, ),
                                          dtype=torch.int,
                                          device='cuda')
        self.position_ids_cuda = torch.empty((self.max_num_tokens, ),
                                             dtype=torch.int,
                                             device='cuda')
        self.iter_counter = 0

        # We look up this key in resource_manager during forward to find the
        # kv cache manager. Can be changed to support multiple model engines
        # with different KV cache managers.
        self.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        self.lora_model_config: Optional[LoraModelConfig] = None
        self.cuda_graph_dummy_request = None

        # Setup the local cache indirection buffer only once and reuse it.
        # This way it can also be used for CUDA graphs.
        if self.use_beam_search:
            self.cache_indirection_attention = torch.zeros(
                (self.batch_size, self.max_beam_width, self.max_seq_len +
                 (0 if self._disable_overlap_scheduler else 1)),
                device="cuda",
                dtype=torch.int32)
        else:
            self.cache_indirection_attention = None

    def set_lora_model_config(self, lora_target_modules: list[str],
                              trtllm_modules_to_hf_modules: dict[str, str]):
        self.lora_model_config = LoraModelConfig(
            lora_target_modules=lora_target_modules,
            trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
            hidden_size=self.model.config.hidden_size,
            dtype=torch_dtype_to_str(self.model.config.torch_dtype))

    @property
    def use_mrope(self):
        use_mrope = False
        try:
            use_mrope = self.model.model_config.pretrained_config.rope_scaling[
                'type'] == 'mrope'
        except Exception:
            pass
        logger.info(f"Detected use_mrope: {use_mrope}")
        return use_mrope

    @property
    def use_beam_search(self):
        return self.max_beam_width > 1

    @contextmanager
    def set_warmup_flag(self):
        self.in_warmup = True
        try:
            yield
        finally:
            self.in_warmup = False

    @staticmethod
    def with_warmup_flag(method):

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            with self.set_warmup_flag():
                return method(self, *args, **kwargs)

        return wrapper

    @contextlib.contextmanager
    def no_cuda_graph(self):
        _run_cuda_graphs = self._run_cuda_graphs
        self._run_cuda_graphs = False
        try:
            yield
        finally:
            self._run_cuda_graphs = _run_cuda_graphs

    @with_warmup_flag
    def warmup(self, resource_manager: ResourceManager) -> None:
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if kv_cache_manager is None:
            logger.info("Skipping warm up as no KV Cache manager allocated.")
            return
        use_mrope = self.use_mrope

        # The lifetime of model engine and kv cache manager can be different.
        # Reset the global cuda graph dummy request to None in warmup.
        self.cuda_graph_dummy_request = None

        def get_cuda_graph_warmup_request(batch_size):
            # Divide by max_beam_width to get an approximation of the number of requests that can be run in parallel.
            available_blocks = kv_cache_manager.get_num_free_blocks(
            ) // self.max_beam_width
            if available_blocks >= batch_size:
                result = ScheduledRequests()
                result.context_requests = []
                # Add (batch_size - 1) dummy requests with seq_len=1.
                # Should only need one more page per request.
                requests = kv_cache_manager.add_dummy_requests(
                    list(range(batch_size - 1)),
                    is_gen=True,
                    max_num_draft_tokens=self.max_draft_len,
                    use_mrope=use_mrope,
                    max_beam_width=self.max_beam_width)
                # Divide by max_beam_width to get an approximation of the number of tokens that can be added to the final request.
                available_tokens = kv_cache_manager.get_num_available_tokens(
                    self.max_draft_len) // self.max_beam_width

                # Add one dummy request with the maximum possible sequence length.
                # The sequence length is limited by both the max_seq_len and the number of available blocks.
                token_num = max(1, min(available_tokens, self.max_seq_len - 1))
                max_seq_len_request = kv_cache_manager.add_dummy_requests(
                    request_ids=[batch_size - 1],
                    token_nums=[token_num],
                    is_gen=True,
                    max_num_draft_tokens=self.max_draft_len,
                    use_mrope=use_mrope,
                    max_beam_width=self.max_beam_width)[0]
                # Add the longest request before all other seq_len=1 request to simulate the padding CUDA graph case.
                # This batch contains both the longest request and the shortest requests,
                # it also contains the maximum number of requests and the maximum token number,
                # which simulates the extreme case for the padding CUDA graph.
                # Thus we can replay this CUDA graph in all other cases.
                requests.insert(0, max_seq_len_request)
                result.generation_requests = requests
                if spec_resource_manager is not None:
                    spec_resource_manager.add_dummy_requests(
                        request_ids=list(range(batch_size)))
            else:
                result = None
            return result

        def get_torch_compile_warmup_request(batch_size,
                                             num_tokens_per_request):
            available_blocks = kv_cache_manager.get_num_free_blocks()
            if available_blocks >= batch_size * math.ceil(
                    num_tokens_per_request / kv_cache_manager.tokens_per_block):
                # Should only need (at most) one more page per request.
                is_gen = num_tokens_per_request == 1

                requests = kv_cache_manager.add_dummy_requests(
                    list(range(batch_size)), [num_tokens_per_request] *
                    batch_size if not is_gen else None,
                    is_gen=is_gen,
                    max_num_draft_tokens=self.max_draft_len)

                if spec_resource_manager is not None:
                    spec_resource_manager.add_dummy_requests(
                        request_ids=list(range(batch_size)))

                result = ScheduledRequests()
                result.context_requests = []
                result.generation_requests = []
                if is_gen:
                    result.generation_requests = requests
                else:
                    result.context_requests = requests
            else:
                result = None
            return result

        def get_autotune_warmup_request():
            available_tokens = kv_cache_manager.get_num_available_tokens(
                self.max_draft_len)
            num_tokens_per_request = min(
                min(available_tokens, self.max_seq_len - 1),
                self.max_num_tokens)
            # Number of tokens required per request must be rounded up to whole number of blocks
            num_tokens_required_per_request = (
                (num_tokens_per_request + kv_cache_manager.tokens_per_block - 1)
                // kv_cache_manager.tokens_per_block
            ) * kv_cache_manager.tokens_per_block

            available_blocks = kv_cache_manager.get_num_free_blocks()

            maximum_tunable_num_tokens = min(
                self.batch_size * num_tokens_per_request, self.max_num_tokens,
                available_blocks * kv_cache_manager.tokens_per_block)

            # Calculate number of full-length requests and remaining tokens
            # Each request has num_tokens_per_request tokens, except possibly the last one
            # Calculations are also limited by how many KV cache blocks are available
            full_len_request_num = min(
                maximum_tunable_num_tokens // num_tokens_per_request,
                max(1, available_tokens // num_tokens_required_per_request))
            remaining_tokens = min(
                maximum_tunable_num_tokens % num_tokens_per_request,
                max(
                    0, available_tokens -
                    full_len_request_num * num_tokens_required_per_request))

            request_num = full_len_request_num if remaining_tokens == 0 else full_len_request_num + 1

            requests = kv_cache_manager.add_dummy_requests(
                request_ids=list(range(full_len_request_num)),
                token_nums=[num_tokens_per_request] * full_len_request_num,
                is_gen=False,
                max_num_draft_tokens=self.max_draft_len)

            if remaining_tokens > 0:
                final_request = kv_cache_manager.add_dummy_requests(
                    request_ids=[full_len_request_num],
                    token_nums=[remaining_tokens],
                    is_gen=False,
                    max_num_draft_tokens=self.max_draft_len)

                requests += final_request

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(
                    request_ids=list(range(request_num)))

            result = ScheduledRequests()
            result.context_requests = requests
            result.generation_requests = []

            return result

        @contextlib.contextmanager
        def release_batch(result: ScheduledRequests | None):
            try:
                yield result
            finally:
                if result is not None:
                    for req in result.all_requests():
                        kv_cache_manager.free_resources(req)
                        if spec_resource_manager is not None:
                            spec_resource_manager.free_resources(req)

        # TODO: current warmup_request is not suitable for star attention
        cp_type = self.mapping.cp_config.get('cp_type', None)
        if cp_type == 'star_attention':
            return

        with contextlib.ExitStack() as stack:
            if self._torch_compile_enabled:

                def disable_optimization(backend: Backend):
                    # Disable torch.compile optimization and fallback to eager execution
                    backend.bypass_optimization()
                    # Disable piecewise CUDA graph capture since the capture run will produce wrong results
                    set_enable_piecewise_cuda_graph_capture_flag(False)

                stack.callback(disable_optimization,
                               self._torch_compile_backend)

                self._torch_compile_backend.enable_optimization()
                set_enable_piecewise_cuda_graph_capture_flag(True)

                # Disable cuda graph capture here so that we can properly capture it later
                with self.no_cuda_graph():
                    available_tokens = kv_cache_manager.get_num_available_tokens(
                        self.max_draft_len)
                    warmup_batch_size = [1, self.batch_size // 2]
                    if self.batch_size < 2:
                        warmup_batch_size = [1]
                    for bs in warmup_batch_size:
                        for num_tokens_per_request in [
                                1,
                                min(self.max_num_tokens // max(bs, 1),
                                    min(available_tokens, self.max_seq_len - 1))
                        ]:
                            with release_batch(
                                    get_torch_compile_warmup_request(
                                        bs, num_tokens_per_request)) as batch:
                                if batch is None:
                                    # No KV cache space!
                                    continue
                                logger.info(
                                    f"Run warmup for batch size={bs}, pure {'context' if num_tokens_per_request > 1 else 'generation'} phase"
                                )
                                self.forward(batch,
                                             new_tensors_device=None,
                                             resource_manager=resource_manager)
                                torch.cuda.synchronize()

            if self.pytorch_backend_config.enable_autotuner:
                with self.no_cuda_graph(), autotune():
                    result = get_autotune_warmup_request()
                    with release_batch(result) as batch:
                        if batch is None:
                            # No KV cache space!
                            pass
                        else:
                            logger.info(
                                f"Run autotuning warmup for batch size={1}")
                            self.forward(batch,
                                         new_tensors_device=None,
                                         resource_manager=resource_manager)
                            torch.cuda.synchronize()

                    logger.info(f"Autotuner Cache size after warmup " +
                                str(len(AutoTuner.get().profiling_cache)))

            if not (self._run_cuda_graphs
                    or self._torch_compile_piecewise_cuda_graph):
                return

            logger.info(
                f"Creating CUDA graph instances for {len(self._cuda_graph_batch_sizes)} batch sizes."
            )
            # Reverse the order of the cuda graph batch sizes to make smaller batch size graph could reuse larger batch size graph memory
            cuda_graph_batch_sizes = sorted(self._cuda_graph_batch_sizes,
                                            reverse=True)
            for bs in cuda_graph_batch_sizes:
                if bs > self.batch_size:
                    # skip batch size larger than self.batch_size
                    continue
                with release_batch(get_cuda_graph_warmup_request(bs)) as batch:
                    if batch is None:
                        # No KV cache space!
                        return
                    logger.info(
                        f"Run generation only CUDA graph warmup for batch size={bs}"
                    )
                    self.forward(batch,
                                 new_tensors_device=None,
                                 resource_manager=resource_manager)
                    torch.cuda.synchronize()

                if self._torch_compile_piecewise_cuda_graph and self._torch_compile_enabled:
                    with self.no_cuda_graph():
                        with release_batch(
                                get_torch_compile_warmup_request(1,
                                                                 bs)) as batch:
                            logger.info(
                                f"Run piecewise CUDA graph warmup for batch size={bs}"
                            )

                            for _ in range(3):
                                self.forward(batch,
                                             new_tensors_device=None,
                                             resource_manager=resource_manager)
                            self.forward(batch,
                                         new_tensors_device=None,
                                         resource_manager=resource_manager)
                            torch.cuda.synchronize()
                            gc.collect()
                            torch.cuda.empty_cache()

    def _set_up_attn_metadata(self, kv_cache_manager: KVCacheManager):
        enable_paged_context_mla = is_mla(
            self.model.model_config.pretrained_config) and (
                self.attn_runtime_features.cache_reuse
                or self.attn_runtime_features.chunked_prefill)
        cache_indirection = self.cache_indirection_attention if self.attn_backend.Metadata is TrtllmAttentionMetadata else None
        if kv_cache_manager is None:
            return self.attn_backend.Metadata(
                max_num_requests=self.batch_size,
                max_num_tokens=self.max_num_tokens,
                max_num_sequences=self.batch_size * self.max_beam_width,
                kv_cache_manager=None,
                mapping=self.mapping,
                runtime_features=self.attn_runtime_features,
                enable_flash_mla=self.model.model_config.enable_flash_mla,
                enable_paged_context_mla=enable_paged_context_mla,
                cache_indirection=cache_indirection)

        if self.attn_metadata is not None:
            # This assertion can be relaxed if needed: just create a new metadata
            # object if it changes.
            assert self.attn_metadata.kv_cache_manager is kv_cache_manager
            return self.attn_metadata

        self.attn_metadata = self.attn_backend.Metadata(
            max_num_requests=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_num_sequences=self.batch_size * self.max_beam_width,
            kv_cache_manager=kv_cache_manager,
            mapping=self.mapping,
            runtime_features=self.attn_runtime_features,
            enable_flash_mla=self.model.model_config.enable_flash_mla,
            enable_paged_context_mla=enable_paged_context_mla,
            cache_indirection=cache_indirection)

        return self.attn_metadata

    def _set_up_spec_metadata(
            self,
            spec_resource_manager: Optional[BaseResourceManager],
            no_cache=False):
        if no_cache:
            return get_spec_metadata(
                self.spec_config,
                self.model.config,
                self.batch_size,
                max_num_tokens=self.max_num_tokens,
                spec_resource_manager=spec_resource_manager,
                is_draft_model=self.is_draft_model)

        if self.spec_metadata is not None:
            return self.spec_metadata
        self.spec_metadata = get_spec_metadata(
            self.spec_config,
            self.model.config,
            self.batch_size,
            max_num_tokens=self.max_num_tokens,
            spec_resource_manager=spec_resource_manager,
            is_draft_model=self.is_draft_model)
        return self.spec_metadata

    def _get_padded_batch(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager,
            spec_resource_manager: Optional[BaseResourceManager] = None) -> int:
        can_run_cuda_graph = scheduled_requests.can_run_cuda_graph
        batch_size = scheduled_requests.batch_size
        # The number of sequences in the batch is the number of prompts times the beam width.
        new_batch_size = batch_size * self.max_beam_width
        if self._run_cuda_graphs and self.enable_attention_dp and self.mapping.tp_size > 1:
            graph_batch_size = self.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            all_can_graph = all(graph_batch[0]
                                for graph_batch in graph_batch_size)
            if all_can_graph:
                new_batch_size = max(gen_only_batch[1]
                                     for gen_only_batch in graph_batch_size)

        if (not self._run_cuda_graphs or not self._cuda_graph_padding_enabled
                or not can_run_cuda_graph
                or new_batch_size > self._max_cuda_graph_batch_size):
            return 0

        padded_batch_size = self._round_up_batch_size(new_batch_size)
        if batch_size == padded_batch_size:
            return 0

        padding_size = padded_batch_size - batch_size
        if padding_size + scheduled_requests.batch_size > self.batch_size:
            return 0

        # No padding if it would create too many concurrent requests.
        # This is not strictly required, but we should probably
        # respect the requirement just in case that changes in the future.
        if self.cuda_graph_dummy_request is None:
            available_blocks = kv_cache_manager.get_num_free_blocks()
            # No padding if not enough KV cache space
            if available_blocks < 1:
                return 0

            cuda_graph_dummy_request_ids = [MAX_UINT64 - 1]
            self.cuda_graph_dummy_request = kv_cache_manager.add_dummy_requests(
                cuda_graph_dummy_request_ids,
                is_gen=True,
                max_num_draft_tokens=self.max_draft_len,
                use_mrope=self.use_mrope,
                max_beam_width=self.max_beam_width)[0]
            self.cuda_graph_dummy_request.is_cuda_graph_dummy = True
            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(
                    request_ids=cuda_graph_dummy_request_ids)

        scheduled_requests.generation_requests.extend(
            [self.cuda_graph_dummy_request] * padding_size)

        return padding_size

    @contextlib.contextmanager
    def _maybe_pad_batch(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager,
            spec_resource_manager: Optional[BaseResourceManager] = None):
        """
        CUDA graphs can only be used for specific batch sizes.

        If using CUDA graphs, this method will add dummy requests to the given
        batch so we can always use a CUDA graph. It is a context manager
        because the padded requests will be removed from scheduled requests.
        """
        padding_size = self._get_padded_batch(scheduled_requests,
                                              kv_cache_manager,
                                              spec_resource_manager)
        try:
            yield scheduled_requests
        finally:
            if padding_size > 0:
                scheduled_requests.generation_requests = scheduled_requests.generation_requests[:
                                                                                                -padding_size]

    def _round_up_batch_size(self, batch_size: int) -> int:
        """
        Round up the given batch size to the nearest batch size that is
        associated with a CUDA graph.
        """
        idx = bisect.bisect_left(self._cuda_graph_batch_sizes, batch_size)
        return self._cuda_graph_batch_sizes[idx]

    def _maybe_get_cuda_graph(
        self,
        batch: ScheduledRequests,
        spec_config: Optional["DecodingBaseConfig"] = None
    ) -> Optional[DecodingCUDAGraphRunner]:
        """
        Get a CUDA graph runner or return None (e.g. if CUDA graphs are disabled
        or if the batch size is too big).
        """
        # disable when doing statistic
        if ExpertStatistic.set_iter(self.iter_counter):
            return None

        spec_max_draft_tokens = spec_config.max_draft_len if self.is_spec_decode else 0
        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = len(batch.generation_requests)
        if self._run_cuda_graphs and self.enable_attention_dp and self.mapping.tp_size > 1:
            all_can_graph_batch = self.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            is_all_gen_only = all(all_can_graph[0]
                                  for all_can_graph in all_can_graph_batch)
            all_batch_size_equal = all(
                all_gen_only[1] == all_can_graph_batch[0][1]
                for all_gen_only in all_can_graph_batch)

            if not is_all_gen_only or not all_batch_size_equal:
                return None

        if not self._run_cuda_graphs or not can_run_cuda_graph:
            return None

        if batch_size in self._cuda_graphs:
            return self._cuda_graphs[batch_size]

        if batch_size not in self._cuda_graph_batch_sizes:
            return None

        num_sequences_in_batch = batch_size * self.max_beam_width
        attn_metadata = self.attn_metadata.create_cuda_graph_metadata(
            num_sequences_in_batch, False, spec_max_draft_tokens)
        assert attn_metadata.is_cuda_graph

        if self.is_spec_decode:
            spec_metadata = self.spec_metadata.create_cuda_graph_metadata(
                num_sequences_in_batch)
            spec_metadata.draft_tokens = self.draft_tokens_cuda
        else:
            spec_metadata = None

        self._cuda_graphs[batch_size] = DecodingCUDAGraphRunner(
            num_sequences_in_batch, "cuda", attn_metadata, spec_metadata,
            self.use_mrope)
        return self._cuda_graphs[batch_size]

    def __del__(self) -> None:
        if getattr(self, 'ub_buffers', None):
            for u in self.ub_buffers:
                ub.ub_deallocate(u.addr)
        # Release model weights.
        release_gc()

    def _load_model(self,
                    checkpoint_dir: str,
                    checkpoint_loader: BaseCheckpointLoader,
                    load_format: LoadFormat,
                    max_num_tokens: int,
                    moe_max_num_tokens: Optional[int] = None,
                    moe_load_balancer: Optional[MoeLoadBalancerConfig] = None,
                    lora_config: Optional[LoraConfig] = None,
                    **kwargs):

        config = checkpoint_loader.load_config(
            checkpoint_dir,
            trust_remote_code=True,
            enable_min_latency=self.pytorch_backend_config.enable_min_latency,
            use_cuda_graph=self.pytorch_backend_config.use_cuda_graph,
            force_dynamic_quantization=self.pytorch_backend_config.
            force_dynamic_quantization,
            spec_config=self.spec_config,
            max_num_tokens=max_num_tokens,
            moe_max_num_tokens=moe_max_num_tokens,
            moe_load_balancer=moe_load_balancer,
            lora_config=lora_config,
            allreduce_strategy=self.pytorch_backend_config.allreduce_strategy,
            **kwargs)

        validate_and_set_kv_cache_quant(
            config, self.pytorch_backend_config.kv_cache_dtype)
        num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
        if num_layers > 0:
            config.pretrained_config.num_hidden_layers = num_layers
            for sub_config in ["text_config", "vision_config"]:
                if hasattr(config.pretrained_config, sub_config):
                    getattr(config.pretrained_config,
                            sub_config).num_hidden_layers = num_layers

        with timing("Model init total"), maybe_create_moe_load_balancer(
                config, self.mapping) as moe_load_balancer:
            try:
                with MetaInitMode():
                    model = AutoModelForCausalLM.from_config(config)

                memo = dict()

                def init_meta_tensor(t: torch.Tensor):
                    if t.device != torch.device('meta'):
                        return t
                    if t not in memo:
                        memo[t] = torch.empty_like(t, device='cuda')
                    return memo[t]

                model._apply(init_meta_tensor)

            except Exception:
                logger.info(
                    f"Fallback to regular model init: {traceback.format_exc(limit=1)}\n"
                )
                model = AutoModelForCausalLM.from_config(config)

            model.to("cuda")
            rank_model_storage = get_rank_model_storage(model)
            logger.info(
                f"Use {rank_model_storage / (1024**3):.2f} GB for model weights."
            )
            if load_format == LoadFormat.AUTO:
                if hasattr(model, 'llm_checkpoint_dir'):
                    weights = checkpoint_loader.load_weights(
                        model.llm_checkpoint_dir)
                else:
                    weights = checkpoint_loader.load_weights(checkpoint_dir)

                weight_mapper = checkpoint_loader.get_initilized_weight_mapper(
                    model, config)
                self._call_load_weights(model.load_weights, weights,
                                        weight_mapper)

                if self.spec_config is not None and self.spec_config.spec_dec_mode.need_load_draft_weights(
                ):
                    weights = checkpoint_loader.load_weights(
                        self.spec_config.speculative_model_dir)
                    self._call_load_weights(model.load_draft_weights, weights,
                                            weight_mapper)

            elif load_format == LoadFormat.DUMMY:
                initialize_dummy_weights(model)

            else:
                raise NotImplementedError(
                    f"No load support for load format: {load_format}")

            if isinstance(moe_load_balancer, MoeLoadBalancer):
                setattr(self, "moe_load_balancer", moe_load_balancer)
                moe_load_balancer.register_weight_slots_after_to_cuda()
                logger.info("moe_load_balancer finalizing model...")
                moe_load_balancer.finalize_model()
                logger.info("moe_load_balancer finalize model done")

            torch.cuda.current_stream().synchronize()
        return model

    def _call_load_weights(self, load_method, weights, weight_mapper):
        # TODO smor- this is a temporary solution to load weights.
        # Once checkpoint format is unified, this method will be removed.
        from inspect import getfullargspec
        args = getfullargspec(load_method).args
        if "weight_mapper" in args:
            load_method(weights, weight_mapper=weight_mapper)
        else:
            load_method(weights)

    def _init_max_seq_len(self):
        inferred_max_seq_len = self.model.infer_max_seq_len()
        if self.max_seq_len is None:
            logger.info(
                f"max_seq_len is not specified, using inferred value {inferred_max_seq_len}"
            )
            self.max_seq_len = inferred_max_seq_len

        elif inferred_max_seq_len < self.max_seq_len:
            # NOTE: py_executor_creator makes sure that the executor uses this
            # smaller value as its max_seq_len too.
            logger.warning(
                f"Specified {self.max_seq_len=} is larger than what the model can support "
                f"({inferred_max_seq_len}). Setting max_seq_len to {inferred_max_seq_len}. "
            )
            self.max_seq_len = inferred_max_seq_len

    def _init_max_num_tokens(self):
        # Modified from tensorrt_llm/_common.py check_max_num_tokens
        if self.max_num_tokens is None:
            self.max_num_tokens = self.max_seq_len * self.batch_size
        if self.max_num_tokens > self.max_seq_len * self.batch_size:
            logger.warning(
                f"max_num_tokens ({self.max_num_tokens}) shouldn't be greater than "
                f"max_seq_len * max_batch_size ({self.max_seq_len * self.batch_size}), "
                f"specifying to max_seq_len * max_batch_size ({self.max_seq_len * self.batch_size})."
            )
            self.max_num_tokens = self.max_seq_len * self.batch_size

    def _init_model_capacity(self):
        self._init_max_seq_len()
        self._init_max_num_tokens()

    def _release_cuda_graphs(self):
        for _, graph in self._cuda_graphs.items():
            del graph
        self._cuda_graphs.clear()
        torch.cuda.empty_cache()
        del self._cuda_graph_mem_pool
        self._cuda_graph_mem_pool = None

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports. PyExecutor need this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size
        return num_batches * self.batch_size

    def _preprocess_inputs(self, inputs: Dict[str, Any]):
        """
        Make some changes to the device inputs and avoid block the async data transfer
        """
        if self.is_spec_decode and not self._disable_overlap_scheduler:
            # When enabling overlap scheduler, the kv cache for draft tokens will
            # be prepared in advance by using the max_draft_len. But we need to use
            # new_tokens_lens_device to get the real past kv lengths and the
            # correct position ids. And to avoid blocking the async data transfer,
            # we need to preprocess the inputs in forward to update the position_ids and
            # kv cache length.
            if inputs['attn_metadata'].kv_cache_manager is not None:
                num_seqs = inputs['attn_metadata'].num_seqs
                num_ctx_requests = inputs['attn_metadata'].num_contexts
                num_gen_requests = inputs['attn_metadata'].num_generations
                num_ctx_tokens = inputs['attn_metadata'].num_ctx_tokens
                previous_batch_tokens = inputs['input_ids'].shape[
                    0] - num_ctx_tokens
                inputs['position_ids'][0, num_ctx_tokens:] += (
                    self.previous_pos_id_offsets_cuda[:previous_batch_tokens])
                inputs['attn_metadata'].kv_lens_cuda[
                    num_ctx_requests:num_seqs] += (
                        self.previous_kv_lens_offsets_cuda[:num_gen_requests])
        return inputs

    def _prepare_tp_inputs(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[SampleStateTensors] = None,
            cache_indirection_buffer: Optional[torch.Tensor] = None):
        """
        Prepare inputs for Pytorch Model.
        """

        # if new_tensors_device exist, input_ids will only contain new context tokens
        input_ids = []  # per sequence
        sequence_lengths = []  # per sequence
        prompt_lengths = []  # per sequence
        request_ids = []  # per request
        gather_ids = []
        position_ids = []  # per sequence
        num_cached_tokens_per_seq = []  # per sequence
        draft_tokens = []
        draft_lens = []
        multimodal_params_list = []
        gen_request_seq_slots = []  # per generation request

        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            all_prompt_tokens = request.get_tokens(0)
            draft_lens.append(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            prompt_tokens = all_prompt_tokens[begin_compute:end_compute]
            position_ids.extend(
                range(begin_compute, begin_compute + len(prompt_tokens)))
            input_ids.extend(prompt_tokens)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(prompt_tokens))
            prompt_lengths.append(len(prompt_tokens))
            past_seen_token_num = begin_compute
            num_cached_tokens_per_seq.append(past_seen_token_num)

            # Multimodal
            # TODO: enable chunk prefill for multimodal (maybe need to pass prompt_tokens to MultimodalRuntimeData)
            py_multimodal_runtime = MultimodalRuntimeData(
                mm_token_lengths=request.multimodal_lengths,
                mm_token_positions=request.multimodal_positions,
                num_cached_tokens=past_seen_token_num
            ) if request.multimodal_hashes is not None else None

            multimodal_params = MultimodalParams(
                multimodal_data=request.py_multimodal_data,
                multimodal_runtime=py_multimodal_runtime)
            multimodal_params.to_device("multimodal_data",
                                        "cuda",
                                        pin_memory=True)

            if multimodal_params.has_content():
                multimodal_params_list.append(multimodal_params)

            request.py_batch_idx = request.py_seq_slot

        num_ctx_requests = len(scheduled_requests.context_requests)
        num_ctx_tokens = len(input_ids)
        new_tokens_device, new_tokens_lens_device, next_draft_tokens_device = None, None, None
        if new_tensors_device is not None:
            # speculative decoding cases: [batch, 1 + draft_len], others: [batch]
            new_tokens_device = new_tensors_device.new_tokens
            if self.without_logits:
                assert isinstance(new_tensors_device, SampleStateTensorsMTP)
                new_tokens_lens_device = new_tensors_device.new_tokens_lens  # [batch]
                next_draft_tokens_device = new_tensors_device.next_draft_tokens  # [batch, draft_len]

        # Requests with draft tokens are treated like extend requests. Dummy extend requests should be
        # at the end of extend_requests.
        extend_requests = []
        extend_dummy_requests = []
        generation_requests = []
        for request in scheduled_requests.generation_requests:
            if len(request.py_draft_tokens
                   ) > 0 or next_draft_tokens_device is not None:
                if request.is_dummy:
                    extend_dummy_requests.append(request)
                else:
                    extend_requests.append(request)
            else:
                generation_requests.append(request)
            # Multimodal
            multimodal_params = MultimodalParams(
                multimodal_data=request.py_multimodal_data)
            multimodal_params.strip_for_generation()
            multimodal_params.to_device("multimodal_data",
                                        "cuda",
                                        pin_memory=True)
            if multimodal_params.has_content():
                multimodal_params_list.append(multimodal_params)
        extend_requests += extend_dummy_requests

        if not self._disable_overlap_scheduler and self.is_spec_decode:
            assert self.spec_config.spec_dec_mode.support_overlap_scheduler(
            ), f"{self.spec_config.decoding_type} does not support overlap scheduler"

        # will contain previous batch indices of generation requests
        previous_batch_indices = []
        previous_pos_indices = []
        for request in extend_requests:
            # the request has no previous tensor:
            # (1) next_draft_tokens_device is None, which means overlap scheduler is disabled; or
            # (2) a dummy request; or
            # (3) the first step in the generation server of disaggregated serving
            if next_draft_tokens_device is None or request.is_dummy or request.py_batch_idx is None:
                # get token ids, including input token ids and draft token ids. For these dummy requests,
                # no need to copy the token ids.
                if not (request.is_attention_dp_dummy
                        or request.is_cuda_graph_dummy):
                    input_ids.append(request.get_last_tokens(0))
                    input_ids.extend(request.py_draft_tokens)
                    draft_tokens.extend(request.py_draft_tokens)
                # get other ids and lengths
                num_draft_tokens = len(request.py_draft_tokens)
                past_seen_token_num = request.max_beam_num_tokens - 1
                draft_lens.append(num_draft_tokens)

                if self.is_spec_decode and self.spec_config.spec_dec_mode.extend_ctx(
                        self.attn_backend):
                    # We're treating the prompt lengths as context requests here, so
                    # the the prompt lens should not include the cached tokens.
                    prompt_lengths.append(1 + num_draft_tokens)
                else:
                    prompt_lengths.append(request.py_prompt_len)

                sequence_lengths.append(1 + num_draft_tokens)
                gather_ids.extend(
                    list(
                        range(len(position_ids),
                              len(position_ids) + 1 + self.max_draft_len)))
                position_ids.extend(
                    list(
                        range(past_seen_token_num,
                              past_seen_token_num + 1 + num_draft_tokens)))
                num_cached_tokens_per_seq.append(past_seen_token_num)
                request_ids.append(request.py_request_id)
                # update batch index
                request.py_batch_idx = request.py_seq_slot
            else:
                # update batch index
                previous_batch_idx = request.py_batch_idx
                request.py_batch_idx = request.py_seq_slot
                # inputs
                # overlap scheduler can only support the speculative decoding
                # methods with a fixed number of draft tokens
                sequence_lengths.append(1 + self.max_draft_len)
                past_seen_token_num = request.max_beam_num_tokens - 1
                draft_lens.append(self.max_draft_len)
                gather_ids.extend(
                    list(
                        range(len(position_ids),
                              len(position_ids) + 1 + self.max_draft_len)))
                position_ids.extend(
                    list(
                        range(past_seen_token_num,
                              past_seen_token_num + 1 + self.max_draft_len)))
                # previous tensor
                previous_batch_indices.append(previous_batch_idx)
                previous_pos_indices.extend([previous_batch_idx] *
                                            (1 + self.max_draft_len))
                num_cached_tokens_per_seq.append(past_seen_token_num +
                                                 self.max_draft_len + 1)
                prompt_lengths.append(request.py_prompt_len)
                request_ids.append(request.py_request_id)

        for request in generation_requests:
            beam_width = request.sampling_config.beam_width
            for beam in range(beam_width):
                # the request has no previous tensor:
                # (1) new_tokens_device is None, which means overlap scheduler is disabled; or
                # (2) a dummy request; or
                # (3) the first step in the generation server of disaggregated serving
                if new_tokens_device is None or request.is_dummy or request.py_batch_idx is None:
                    # skip adding input_ids of CUDA graph dummy requests so that new_tokens_device
                    # can be aligned to the correct positions.
                    if not request.is_cuda_graph_dummy:
                        input_ids.append(request.get_last_tokens(beam))
                    past_seen_token_num = request.max_beam_num_tokens - 1
                else:
                    # the request has previous tensor
                    # previous_batch_indices is used per request, not per beam
                    # Only append it once for the first beam of each request
                    first_beam = 0
                    if beam == first_beam:
                        previous_batch_indices.append(request.py_batch_idx)
                    past_seen_token_num = request.max_beam_num_tokens

                position_ids.append(past_seen_token_num)
                num_cached_tokens_per_seq.append(past_seen_token_num)
                prompt_lengths.append(request.py_prompt_len)
                draft_lens.append(0)
                sequence_lengths.append(1)
                gather_ids.append(len(position_ids) - 1)

            request_ids.append(request.py_request_id)
            gen_request_seq_slots.append(request.py_seq_slot)
            request.py_batch_idx = request.py_seq_slot

        previous_batch_len = len(previous_batch_indices)

        def previous_seq_slots_device():
            previous_batch_indices_host = torch.tensor(previous_batch_indices,
                                                       dtype=torch.int,
                                                       pin_memory=True)
            previous_slots = self.previous_batch_indices_cuda[:
                                                              previous_batch_len]
            previous_slots.copy_(previous_batch_indices_host, non_blocking=True)
            return previous_slots

        num_tokens = len(input_ids)
        num_draft_tokens = len(draft_tokens)
        total_num_tokens = len(position_ids)
        assert total_num_tokens <= self.max_num_tokens, (
            "total_num_tokens should be less than or equal to max_num_tokens")
        # if exist requests that do not have previous batch, copy input_ids and draft_tokens
        if num_tokens > 0:
            input_ids = torch.tensor(input_ids,
                                     dtype=torch.int,
                                     pin_memory=True)
            self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)
        if num_draft_tokens > 0:
            draft_tokens = torch.tensor(draft_tokens,
                                        dtype=torch.int,
                                        pin_memory=True)
            self.draft_tokens_cuda[:len(draft_tokens)].copy_(draft_tokens,
                                                             non_blocking=True)
        if next_draft_tokens_device is not None:
            # Initialize these two values to zeros
            self.previous_pos_id_offsets_cuda *= 0
            self.previous_kv_lens_offsets_cuda *= 0

            if previous_batch_len > 0:
                previous_slots = previous_seq_slots_device()
                # previous input ids
                previous_batch_tokens = previous_batch_len * (
                    1 + self.max_draft_len)
                new_tokens = new_tokens_device.transpose(
                    0, 1)[previous_slots, :].flatten()
                self.input_ids_cuda[num_tokens:num_tokens +
                                    previous_batch_tokens].copy_(
                                        new_tokens, non_blocking=True)
                # previous draft tokens
                previous_batch_draft_tokens = previous_batch_len * self.max_draft_len
                self.draft_tokens_cuda[num_draft_tokens:num_draft_tokens +
                                       previous_batch_draft_tokens].copy_(
                                           next_draft_tokens_device[
                                               previous_slots, :].flatten(),
                                           non_blocking=True)
                # prepare data for the preprocess inputs
                kv_len_offsets_device = new_tokens_lens_device - self.max_draft_len - 1
                previous_pos_indices_host = torch.tensor(previous_pos_indices,
                                                         dtype=torch.int,
                                                         pin_memory=True)
                self.previous_pos_indices_cuda[0:previous_batch_tokens].copy_(
                    previous_pos_indices_host, non_blocking=True)

                # The order of requests in a batch: [context requests, generation requests]
                # generation requests: ['requests that do not have previous batch', 'requests that already have previous batch', 'dummy requests']
                #   1) 'requests that do not have previous batch': disable overlap scheduler or the first step in the generation server of disaggregated serving.
                #   2) 'requests that already have previous batch': previous iteration's requests.
                #   3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                # Therefore, both of self.previous_pos_id_offsets_cuda and self.previous_kv_lens_offsets_cuda are also 3 segments.
                #   For 1) 'requests that do not have previous batch': disable overlap scheduler or the first step in the generation server of disaggregated serving.
                #       Set these requests' previous_pos_id_offsets and previous_kv_lens_offsets to '0' to skip the value changes in _preprocess_inputs.
                #       Already set to '0' during initialization.
                #   For 2) 'requests that already have previous batch': enable overlap scheduler.
                #       Set their previous_pos_id_offsets and previous_kv_lens_offsets according to new_tokens_lens_device and kv_len_offsets_device.
                #   For 3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                #       Already set to '0' during initialization.

                num_extend_reqeust_wo_dummy = len(extend_requests) - len(
                    extend_dummy_requests)
                self.previous_pos_id_offsets_cuda[
                    (num_extend_reqeust_wo_dummy - previous_batch_len) *
                    (1 + self.max_draft_len):num_extend_reqeust_wo_dummy *
                    (1 + self.max_draft_len)].copy_(
                        new_tokens_lens_device[self.previous_pos_indices_cuda[
                            0:previous_batch_tokens]],
                        non_blocking=True)

                self.previous_kv_lens_offsets_cuda[
                    num_extend_reqeust_wo_dummy -
                    previous_batch_len:num_extend_reqeust_wo_dummy].copy_(
                        kv_len_offsets_device[previous_slots],
                        non_blocking=True)

        elif new_tokens_device is not None:
            seq_slots_device = previous_seq_slots_device()
            max_draft_len = max(draft_lens)
            new_tokens = new_tokens_device[:max_draft_len + 1,
                                           seq_slots_device, :self.
                                           max_beam_width]
            self.input_ids_cuda[num_tokens:num_tokens +
                                previous_batch_len * self.max_beam_width].copy_(
                                    new_tokens.flatten(), non_blocking=True)

        if (not self._disable_overlap_scheduler
                and next_draft_tokens_device is None
                and len(extend_requests) > 0):
            # During warmup, for those generation requests, we don't have previous tensors,
            # so we need to set the previous_pos_id_offsets and previous_kv_lens_offsets to zeros
            # to skip the value changes in _preprocess_inputs. Otherwise, there will be illegal memory access
            # when writing key/values to the KV cache.
            self.previous_pos_id_offsets_cuda *= 0
            self.previous_kv_lens_offsets_cuda *= 0

        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=True)
        self.position_ids_cuda[:total_num_tokens].copy_(position_ids,
                                                        non_blocking=True)
        if self.is_spec_decode:
            self.gather_ids_cuda[:len(gather_ids)].copy_(torch.tensor(
                gather_ids, dtype=torch.int, pin_memory=True),
                                                         non_blocking=True)

        if not attn_metadata.is_cuda_graph:
            # Assumes seq lens do not change between CUDA graph invocations. This applies
            # to draft sequences too. This means that all draft sequences must be padded.
            attn_metadata.seq_lens = torch.tensor(
                sequence_lengths,
                dtype=torch.int,
                pin_memory=True,
            )

        num_generation_requests = len(scheduled_requests.generation_requests)
        # Cache indirection is only used for beam search on generation requests
        if self.use_beam_search and num_generation_requests > 0:
            # CUDA Graph needs to set beam width during warmup (where the graph is captured), to ensure that cache indirection buffer is correctly picked up by the CUDA graph
            is_cuda_graph_during_warmup = self.in_warmup and attn_metadata.is_cuda_graph
            if cache_indirection_buffer is not None:
                #Copy cache indirection to local buffer with offsets changing:  seq_slots[i] -> i
                self.cache_indirection_attention[:num_generation_requests].copy_(
                    cache_indirection_buffer[gen_request_seq_slots])
            if cache_indirection_buffer is not None or is_cuda_graph_during_warmup:
                attn_metadata.beam_width = self.max_beam_width
        else:
            attn_metadata.beam_width = 1

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = len(scheduled_requests.context_requests)
        if self.is_spec_decode and self.spec_config.spec_dec_mode.extend_ctx(
                self.attn_backend):
            attn_metadata.num_contexts += len(extend_requests)

        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=get_num_extra_kv_tokens(self.spec_config))
        attn_metadata.kv_cache_manager = kv_cache_manager

        attn_metadata.prepare()

        lora_params = self._get_lora_params_from_requests(
            scheduled_requests, attn_metadata)

        # Prepare inputs
        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:total_num_tokens],
            'position_ids':
            self.position_ids_cuda[:total_num_tokens].unsqueeze(0),
            'inputs_embeds': None,
            "multimodal_params": multimodal_params_list,
        }

        # Directly input mrope_position_deltas as a Tensor for cuda graph, because dictionary could not be captured.
        if attn_metadata.is_cuda_graph and len(multimodal_params_list) > 0:
            if 'mrope_position_deltas' in multimodal_params_list[
                    0].multimodal_data.get('mrope_config', {}):
                mrope_position_deltas_list = [
                    multimodal_params.multimodal_data['mrope_config']
                    ['mrope_position_deltas']
                    for multimodal_params in multimodal_params_list
                ]
                inputs['mrope_position_deltas'] = torch.cat(
                    mrope_position_deltas_list, dim=0)

        if bool(lora_params):
            inputs['lora_params'] = lora_params

        if spec_metadata is not None:
            total_draft_lens = sum(draft_lens)
            spec_metadata.draft_tokens = self.draft_tokens_cuda[:
                                                                total_draft_lens]
            spec_metadata.request_ids = request_ids
            spec_metadata.gather_ids = self.gather_ids_cuda[:len(gather_ids)]
            spec_metadata.num_generations = len(
                scheduled_requests.generation_requests)
            spec_metadata.num_tokens = total_num_tokens
            spec_metadata.seq_lens = sequence_lengths
            spec_metadata.prepare()
            inputs['spec_metadata'] = spec_metadata

        # support attention dp
        if self.enable_attention_dp:
            if spec_metadata is not None:
                all_rank_num_tokens = self.dist.tp_allgather([
                    attn_metadata.num_tokens, spec_metadata.num_tokens,
                    len(sequence_lengths)
                ])
                attn_all_rank_num_tokens = [
                    item[0] for item in all_rank_num_tokens
                ]
                spec_all_rank_num_tokens = [
                    item[1] for item in all_rank_num_tokens
                ]
                all_rank_num_seqs = [item[2] for item in all_rank_num_tokens]
                attn_metadata.all_rank_num_tokens = attn_all_rank_num_tokens
                spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
                spec_metadata.all_rank_num_seqs = all_rank_num_seqs
            else:
                all_rank_num_tokens = self.dist.tp_allgather(
                    attn_metadata.num_tokens)
                attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        num_generation_tokens = len(generation_requests) + len(
            extend_requests) + sum(draft_lens)
        self.iter_states['num_ctx_requests'] = num_ctx_requests
        self.iter_states['num_ctx_tokens'] = num_ctx_tokens
        self.iter_states['num_generation_tokens'] = num_generation_tokens
        return inputs, self.gather_ids_cuda[:len(
            gather_ids)] if self.is_spec_decode else None

    def _prepare_tp_inputs_no_cache(
            self,
            scheduled_requests: ScheduledRequests,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None):
        """
        Prepare inputs for Pytorch Model.
        """
        sequence_lengths = []
        input_ids = []
        gather_ids = []
        position_ids = []
        multi_modal_data = []
        draft_lens = []
        request_ids = []

        for request in scheduled_requests.context_requests:
            prompt_tokens = request.get_tokens(0)
            input_ids.extend(prompt_tokens)
            request_ids.append(request.py_request_id)
            if request.position_ids is None:
                position_ids.extend(range(len(prompt_tokens)))
            else:
                position_ids.extend(request.position_ids)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(prompt_tokens))
            draft_lens.append(0)
            multimodal_embedding = request.multimodal_embedding
            if multimodal_embedding is not None:
                multi_modal_data.append(multimodal_embedding)

        num_tokens = len(input_ids)
        assert num_tokens <= self.max_num_tokens, (
            "num_tokens should be less than or equal to max_num_tokens")
        input_ids = torch.tensor(input_ids, dtype=torch.int, pin_memory=True)
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=True)
        self.position_ids_cuda[:num_tokens].copy_(position_ids,
                                                  non_blocking=True)
        if self.is_spec_decode:
            self.gather_ids_cuda[:len(gather_ids)].copy_(torch.tensor(
                gather_ids, dtype=torch.int, pin_memory=True),
                                                         non_blocking=True)

        if not attn_metadata.is_cuda_graph:
            # No need to overwrite seq lens when using CUDA graphs -
            # CUDA graphs are only used for pure decoding batches
            # and have static batch size, so the seqlens never change.
            # Note that it's important to not free the seq_lens_cuda
            # buffer once the graph has been captured also - this will invalidate
            # the graph and force an expensive recapture.
            attn_metadata.seq_lens = torch.tensor(
                sequence_lengths,
                dtype=torch.int,
                pin_memory=True,
            )

        attn_metadata.num_contexts = len(scheduled_requests.context_requests)
        if self.enable_attention_dp:
            all_rank_num_tokens = self.dist.allgather(attn_metadata.num_tokens)
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens
        # this is for no cache attention, not for dummy attention
        if attn_metadata.kv_cache_manager is None:
            assert isinstance(
                attn_metadata,
                (VanillaAttentionMetadata, TrtllmAttentionMetadata)
            ), "Only vanilla and trtllm attention metadata are supported for no cache attention for now"
            attn_metadata.max_seq_len = self.max_seq_len
            attn_metadata.request_ids = request_ids
            attn_metadata.prepare()

        lora_params = self._get_lora_params_from_requests(
            scheduled_requests, attn_metadata)

        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:num_tokens],
            'position_ids': self.position_ids_cuda[:num_tokens].unsqueeze(0),
            'inputs_embeds': None,
            'multi_modal_data': multi_modal_data
        }

        if bool(lora_params):
            inputs['lora_params'] = lora_params

        if spec_metadata is not None:
            total_draft_lens = sum(draft_lens)
            spec_metadata.draft_tokens = self.draft_tokens_cuda[:
                                                                total_draft_lens]
            spec_metadata.request_ids = request_ids
            spec_metadata.gather_ids = self.gather_ids_cuda[:len(gather_ids)]
            spec_metadata.num_generations = len(
                scheduled_requests.generation_requests)
            spec_metadata.num_tokens = num_tokens
            spec_metadata.seq_lens = sequence_lengths
            spec_metadata.prepare()
            inputs['spec_metadata'] = spec_metadata

        # support attention dp
        if self.enable_attention_dp:
            if spec_metadata is not None:
                all_rank_num_tokens = self.dist.tp_allgather([
                    attn_metadata.num_tokens, spec_metadata.num_tokens,
                    len(sequence_lengths)
                ])
                attn_all_rank_num_tokens = [
                    item[0] for item in all_rank_num_tokens
                ]
                spec_all_rank_num_tokens = [
                    item[1] for item in all_rank_num_tokens
                ]
                all_rank_num_seqs = [item[2] for item in all_rank_num_tokens]
                attn_metadata.all_rank_num_tokens = attn_all_rank_num_tokens
                spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
                spec_metadata.all_rank_num_seqs = all_rank_num_seqs
            else:
                all_rank_num_tokens = self.dist.tp_allgather(
                    attn_metadata.num_tokens)
                attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        return inputs, None

    def _prepare_star_attention_inputs(self,
                                       scheduled_requests: ScheduledRequests,
                                       kv_cache_manager,
                                       attn_metadata: AttentionMetadata):
        """
        Prepare inputs for Pytorch Model.
        """
        sequence_lengths = []
        input_ids = []
        prompt_lengths = []
        request_ids = []
        gather_ids = []
        position_ids = []
        # for star attention, we need customized block ids
        block_ids_per_seq = []
        num_cached_tokens_per_seq = []
        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            prompt_lengths.append(request.py_prompt_len)

            ctx_iter = request.ctx_iters
            ctx_blocks = request.ctx_blocks
            ctx_position_blocks = request.ctx_position_blocks
            all_cache_indices = kv_cache_manager.get_cache_indices(request)
            ### for the first iteration, we need to construct input as C[0]  + C[1]
            if ctx_iter == 0:
                input_id = ctx_blocks[0] + ctx_blocks[1]
                num_kv_blocks = kv_cache_manager.get_num_kv_blocks(
                    len(input_id))
                position_id = ctx_position_blocks[0] + ctx_position_blocks[1]
                past_seen_token_num = 0
                all_cache_indices = all_cache_indices[:num_kv_blocks]
            else:
                input_id = ctx_blocks[ctx_iter + 1]
                position_id = ctx_position_blocks[ctx_iter + 1]
                ## compute C[0] and ctx_blocks
                if ctx_iter < len(ctx_blocks) - 2:
                    if self.mapping.cp_rank == 0:
                        anchor_block = ctx_blocks[
                            0][:self.mapping.cp_config['cp_anchor_size']]
                    else:
                        anchor_block = ctx_blocks[0]

                    num_anchor_cache_blocks = kv_cache_manager.get_num_kv_blocks(
                        len(anchor_block))
                    ### we need to construct input as C[0] + C[x+i]
                    #C0 has been computed, can be shared across all blocks
                    anchor_indices = all_cache_indices[:num_anchor_cache_blocks]

                    # C1~C[ctx_iter] should be skipped in the computation
                    token_start_idx = sum(
                        len(block) for block in ctx_blocks[:(ctx_iter + 1)])
                    token_end_idx = sum(
                        len(block) for block in ctx_blocks[:(ctx_iter + 2)])
                    block_start_idx = kv_cache_manager.get_num_kv_blocks(
                        token_start_idx)
                    block_end_idx = kv_cache_manager.get_num_kv_blocks(
                        token_end_idx)
                    block_indices = all_cache_indices[
                        block_start_idx:block_end_idx]

                    all_cache_indices = anchor_indices + block_indices
                    past_seen_token_num = len(
                        anchor_block)  ### C[0] can be reused
                else:
                    continue
            input_ids.extend(input_id)
            position_ids.extend(position_id)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(input_id))
            block_ids_per_seq.extend([all_cache_indices])
            num_cached_tokens_per_seq.append(past_seen_token_num)
        num_contexts = len(sequence_lengths)
        for request in scheduled_requests.context_requests:
            ctx_iter = request.ctx_iters
            ctx_blocks = request.ctx_blocks
            ctx_position_blocks = request.ctx_position_blocks
            num_kvblocks_per_ctx_block = kv_cache_manager.get_num_kv_blocks(
                len(ctx_blocks[0]))
            all_cache_indices = kv_cache_manager.get_cache_indices(request)
            ### for query phase
            ## compute C[0~blocks] with query for the first rank
            ## compute C[1~blocks] with query for the other rank
            if ctx_iter == len(ctx_blocks) - 2:
                input_id = ctx_blocks[ctx_iter + 1]
                position_id = ctx_position_blocks[ctx_iter + 1]
                if self.mapping.cp_rank == 0:
                    past_seen_token_num = sum(
                        len(block) for block in ctx_blocks[:ctx_iter + 1])
                else:
                    # drop C0, free KV cache
                    all_cache_indices = all_cache_indices[
                        num_kvblocks_per_ctx_block:]
                    past_seen_token_num = sum(
                        len(block) for block in ctx_blocks[1:ctx_iter + 1])
                if self.mapping.cp_rank == self.mapping.cp_size - 1:
                    num_kv_tokens = past_seen_token_num + len(input_id)
                else:
                    num_kv_tokens = past_seen_token_num  # don't need to append/compute query's kv cache
                num_kv_blocks = kv_cache_manager.get_num_kv_blocks(
                    num_kv_tokens)
                all_cache_indices = all_cache_indices[:num_kv_blocks]
            else:
                continue

            input_ids.extend(input_id)
            position_ids.extend(position_id)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(input_id))
            block_ids_per_seq.extend([all_cache_indices])
            num_cached_tokens_per_seq.append(past_seen_token_num)
        num_queries = len(sequence_lengths) - num_contexts

        # Requests with draft tokens are treated like extend requests.
        extend_requests = [
            request for request in scheduled_requests.generation_requests
            if request.py_draft_tokens
        ]
        generation_requests = [
            request for request in scheduled_requests.generation_requests
            if not request.py_draft_tokens
        ]
        is_spec_decode = len(extend_requests) > 0
        assert not is_spec_decode, 'star attention does not support draft tokens now.'

        for request in generation_requests:
            request_ids.append(request.py_request_id)
            prompt_lengths.append(request.py_prompt_len)

            input_token_id = request.get_token(0, request.get_num_tokens(0) - 1)
            input_ids.append(input_token_id)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(1)
            past_seen_token_num = request.max_beam_num_tokens - 1

            # for sp, we only increase the generated KV cache for the last rank
            ctx_blocks = request.ctx_blocks
            total_anchor_ctx_query_len = sum(
                [len(block) for block in ctx_blocks])
            query_len = len(ctx_blocks[-1])
            anchor_len = len(ctx_blocks[0])

            if self.mapping.cp_size == 1:
                past_seen_token_num = total_anchor_ctx_query_len + request.gen_iters
                num_kv_tokens = past_seen_token_num + 1
            else:
                if self.mapping.cp_rank == self.mapping.cp_size - 1:
                    past_seen_token_num = total_anchor_ctx_query_len + request.gen_iters - anchor_len
                    num_kv_tokens = past_seen_token_num + 1
                else:
                    if self.mapping.cp_rank != 0:
                        past_seen_token_num = total_anchor_ctx_query_len - anchor_len - query_len
                    else:
                        past_seen_token_num = total_anchor_ctx_query_len - query_len
                    num_kv_tokens = past_seen_token_num  # don't need to append kv cache

            num_kv_blocks = kv_cache_manager.get_num_kv_blocks(num_kv_tokens)
            all_cache_indices = kv_cache_manager.get_cache_indices(request)
            if self.mapping.cp_rank != 0:
                num_kvblocks_per_ctx_block = kv_cache_manager.get_num_kv_blocks(
                    anchor_len)
                all_cache_indices = all_cache_indices[
                    num_kvblocks_per_ctx_block:]
            cache_indices = all_cache_indices[:num_kv_blocks]
            last_query_pos_id = request.ctx_position_blocks[-1][-1]
            position_ids.append(last_query_pos_id + request.gen_iters + 1)
            block_ids_per_seq.extend([all_cache_indices])
            num_cached_tokens_per_seq.append(past_seen_token_num)

        num_tokens = len(input_ids)
        assert num_tokens <= self.max_num_tokens, (
            "num_tokens should be less than or equal to max_num_tokens")
        input_ids = torch.tensor(input_ids, dtype=torch.int, pin_memory=True)
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=True)
        self.position_ids_cuda[:num_tokens].copy_(position_ids,
                                                  non_blocking=True)

        if not attn_metadata.is_cuda_graph:
            # No need to overwrite seq lens when using CUDA graphs -
            # CUDA graphs are only used for pure decoding batches
            # and have static batch size, so the seqlens never change.
            # Note that it's important to not free the seq_lens_cuda
            # buffer once the graph has been captured also - this will invalidate
            # the graph and force an expensive recapture.
            attn_metadata.seq_lens = torch.tensor(
                sequence_lengths,
                dtype=torch.int,
                pin_memory=True,
            )

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = num_contexts
        attn_metadata.num_queries = num_queries

        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            block_ids_per_seq=block_ids_per_seq,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq)

        attn_metadata.kv_cache_manager = kv_cache_manager

        attn_metadata.prepare()
        if self.enable_attention_dp:
            all_rank_num_tokens = self.dist.tp_allgather(
                attn_metadata.num_tokens)
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        return {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:num_tokens],
            'position_ids': self.position_ids_cuda[:num_tokens].unsqueeze(0),
            'inputs_embeds': None
        }, gather_ids if is_spec_decode else None

    def _get_lora_params_from_requests(self,
                                       scheduled_requests: ScheduledRequests,
                                       attn_metadata: AttentionMetadata):
        '''
        lora_params: dict
        {
            layer_id: dict
            {
                module_id: dict
                {
                    adapter_size: torch tensor: int
                    is_dora: torch tensor: bool
                    weight_pointers: torch tensor: int64
                }
            }
        }
        '''
        lora_params = {}
        tmp_lora_params = {}

        request_list = scheduled_requests.context_requests + scheduled_requests.generation_requests

        # trace all requests to get the union set of the lora params
        for request in request_list:
            if request.py_lora_task_layer_module_configs is None:
                continue

            for module in request.py_lora_task_layer_module_configs:
                module_id = module.module_id
                layer_id = module.layer_id
                adapter_size = module.adapter_size
                is_dora = module.scaling_vec_pointer == 0
                weights_in_pointer = module.weights_in_pointer
                weights_out_pointer = module.weights_out_pointer
                scaling_vec_pointer = module.scaling_vec_pointer
                if weights_in_pointer is None:
                    weights_in_pointer = 0
                if weights_out_pointer is None:
                    weights_out_pointer = 0
                if scaling_vec_pointer is None:
                    scaling_vec_pointer = 0

                if layer_id not in lora_params:
                    lora_params[layer_id] = {}
                if module_id not in lora_params[layer_id]:
                    lora_params[layer_id][module_id] = {}

                if 'adapter_size' not in lora_params[layer_id][module_id]:
                    lora_params[layer_id][module_id]['adapter_size'] = []
                if 'is_dora' not in lora_params[layer_id][module_id]:
                    lora_params[layer_id][module_id]['is_dora'] = []
                if 'weight_pointers' not in lora_params[layer_id][module_id]:
                    lora_params[layer_id][module_id]['weight_pointers'] = []

                tmp_lora_params[
                    f'{request.py_request_id}_{layer_id}_{module_id}_adapter_size'] = [
                        adapter_size
                    ]
                tmp_lora_params[
                    f'{request.py_request_id}_{layer_id}_{module_id}_is_dora'] = [
                        is_dora
                    ]
                tmp_lora_params[
                    f'{request.py_request_id}_{layer_id}_{module_id}_weights_pointer'] = [
                        weights_in_pointer, weights_out_pointer,
                        scaling_vec_pointer
                    ]

        for request in request_list:
            # Need to set default values for this case
            if request.py_lora_task_layer_module_configs is None:
                for layer_id in lora_params:
                    for module_id in lora_params[layer_id]:
                        lora_params[layer_id][module_id]['adapter_size'].append(
                            0)
                        lora_params[layer_id][module_id]['is_dora'].append(
                            False)
                        lora_params[layer_id][module_id]['weight_pointers'] += [
                            0, 0, 0
                        ]

            else:
                for layer_id in lora_params:
                    for module_id in lora_params[layer_id]:
                        if f'{request.py_request_id}_{layer_id}_{module_id}_adapter_size' not in tmp_lora_params:
                            lora_params[layer_id][module_id][
                                'adapter_size'].append(0)
                            lora_params[layer_id][module_id]['is_dora'].append(
                                False)
                            lora_params[layer_id][module_id][
                                'weight_pointers'] += [0, 0, 0]
                        else:
                            lora_params[layer_id][module_id][
                                'adapter_size'] += tmp_lora_params[
                                    f'{request.py_request_id}_{layer_id}_{module_id}_adapter_size']
                            lora_params[layer_id][module_id][
                                'is_dora'] += tmp_lora_params[
                                    f'{request.py_request_id}_{layer_id}_{module_id}_is_dora']
                            lora_params[layer_id][module_id][
                                'weight_pointers'] += tmp_lora_params[
                                    f'{request.py_request_id}_{layer_id}_{module_id}_weights_pointer']

        for layer_id in lora_params:
            for module_id in lora_params[layer_id]:
                lora_params[layer_id][module_id][
                    'adapter_size'] = torch.IntTensor(
                        lora_params[layer_id][module_id]['adapter_size'])
                lora_params[layer_id][module_id][
                    'weight_pointers'] = torch.LongTensor(
                        lora_params[layer_id][module_id]['weight_pointers'])

        if bool(lora_params):
            lora_params['host_request_types'] = attn_metadata.host_request_types
            lora_params['prompt_lens_cpu'] = attn_metadata.prompt_lens_cpu
            lora_params['num_seqs'] = attn_metadata.num_seqs

        return lora_params

    @nvtx_range("_prepare_inputs")
    def _prepare_inputs(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[SampleStateTensors] = None,
            cache_indirection_buffer: Optional[torch.Tensor] = None):
        if self.mapping is not None and 'cp_type' in self.mapping.cp_config:
            cp_type = self.mapping.cp_config['cp_type']
            if 'star_attention' == cp_type:
                return self._prepare_star_attention_inputs(
                    scheduled_requests, kv_cache_manager, attn_metadata)
            else:
                assert False, f'Unsupport cp_type {cp_type}'
        else:
            return self._prepare_tp_inputs(scheduled_requests, kv_cache_manager,
                                           attn_metadata, spec_metadata,
                                           new_tensors_device,
                                           cache_indirection_buffer)

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self.model.extra_attrs)
    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tensors_device: Optional[SampleStateTensors] = None,
        gather_context_logits: bool = False,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
    ):

        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)

        attn_metadata = self._set_up_attn_metadata(kv_cache_manager)
        if self.is_spec_decode:
            spec_resource_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            spec_metadata = self._set_up_spec_metadata(spec_resource_manager,
                                                       no_cache=kv_cache_manager
                                                       is None)
            # attn_metadata now depends on spec_metadata since it determines the shape/content of spec_dec parameter Tensors
            attn_metadata.update_spec_dec_param(
                spec_metadata.spec_dec_mode.attention_need_spec_dec_mode(),
                spec_metadata.is_spec_dec_tree,
                spec_metadata.is_spec_dec_dynamic_tree,
                spec_metadata.max_draft_len)
        else:
            spec_resource_manager = None
            spec_metadata = None

        moe_load_balancer = None
        if hasattr(self, 'moe_load_balancer'):
            moe_load_balancer = getattr(self, 'moe_load_balancer')
            if not self.in_warmup:
                moe_enable_statistic = True
                moe_enable_update = True
                moe_load_balancer.set_next_iter_info(moe_enable_statistic,
                                                     moe_enable_update)

        if kv_cache_manager is None:
            inputs, gather_ids = self._prepare_tp_inputs_no_cache(
                scheduled_requests, attn_metadata, spec_metadata)

            with MoeLoadBalancerIterContext(moe_load_balancer):
                return self._forward_step(inputs, gather_ids,
                                          gather_context_logits)
        with self._maybe_pad_batch(scheduled_requests, kv_cache_manager,
                                   spec_resource_manager) as scheduled_requests:
            maybe_graph = self._maybe_get_cuda_graph(
                scheduled_requests, spec_config=self.spec_config)
            if maybe_graph is not None:
                attn_metadata = maybe_graph.attn_metadata
                if self.is_spec_decode:
                    spec_metadata = maybe_graph.spec_metadata
            else:
                attn_metadata = self.attn_metadata
                if self.is_spec_decode:
                    spec_metadata = self.spec_metadata

            inputs, gather_ids = self._prepare_inputs(
                scheduled_requests, kv_cache_manager, attn_metadata,
                spec_metadata, new_tensors_device, cache_indirection_buffer)

            self.iter_counter += 1

            if maybe_graph is None:
                with MoeLoadBalancerIterContext(moe_load_balancer):
                    outputs = self._forward_step(inputs, gather_ids,
                                                 gather_context_logits)
            else:
                if maybe_graph.needs_capture():

                    def capture_forward_fn(inputs: Dict[str, Any]):
                        with MoeLoadBalancerIterContext(moe_load_balancer):
                            return self._forward_step(
                                inputs,
                                gather_ids=gather_ids,
                                gather_context_logits=gather_context_logits)

                    pool = maybe_graph.capture(
                        capture_forward_fn,
                        self._cuda_graph_mem_pool,
                    )
                    self._cuda_graph_mem_pool = pool

                    # here we don't need to use context since cuda graph capture didn't run kernel.
                    # maybe we need a cleaner way to do this.
                    outputs = maybe_graph.run(inputs)
                else:
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        outputs = maybe_graph.run(inputs)

            self._execute_logit_post_processors(scheduled_requests, outputs)

            return outputs

    def model_forward(self, **kwargs):
        attrs = get_model_extra_attrs()
        assert attrs is not None, "Model extra attrs is not set"
        attrs["attention_metadata"] = weakref.ref(kwargs['attn_metadata'])
        attrs.update(self.model.model_config.extra_attrs)

        if self._torch_compile_backend is not None:
            # Register aux streams and events to model extra attrs.
            # The streams and events are list which could be updated during compilation.
            attrs["aux_streams"] = weakref.ref(
                self._torch_compile_backend.aux_streams)
            attrs["events"] = weakref.ref(self._torch_compile_backend.events)
            attrs["global_stream"] = torch.cuda.current_stream()

        if is_trace_enabled("TLLM_TRACE_MODEL_FORWARD"):
            return trace_func(self.model.forward)(**kwargs)
        else:
            return self.model.forward(**kwargs)

    @nvtx_range("_forward_step")
    def _forward_step(self,
                      inputs: Dict[str, Any],
                      gather_ids: Optional[torch.Tensor],
                      gather_context_logits: bool = False) -> Dict[str, Any]:
        inputs = self._preprocess_inputs(inputs)
        if inputs.get('spec_metadata', None):
            gather_ids = inputs['spec_metadata'].gather_ids
        if self.without_logits:
            outputs = self.model_forward(**inputs)
            return outputs

        # For simplicity, just return all the the logits if we have special gather_ids
        # from speculative decoding.
        logits = self.model_forward(
            **inputs,
            return_context_logits=gather_ids is not None
            or gather_context_logits,
        )
        if gather_ids is not None:
            return {'logits': logits[gather_ids]}
        else:
            return {'logits': logits}

    def _init_userbuffers(self, hidden_size):
        if self.mapping.tp_size <= 1:
            return False

        # Disable UB for unsupported platforms
        if not ub.ub_supported():
            return False
        ub.initialize_userbuffers_manager(self.mapping.tp_size,
                                          self.mapping.pp_size,
                                          self.mapping.cp_size,
                                          self.mapping.rank,
                                          self.mapping.gpus_per_node,
                                          hidden_size * self.max_num_tokens * 2)
        return True

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        """
        When doing spec decode, sometimes draft models need to share certain weights
        with their target models. Here, we set up such weights by invoking
        self.model.load_weights_from_target_model if such a method exists.
        """
        loader = getattr(self.model, "load_weights_from_target_model", None)
        if callable(loader):
            loader(target_model)

    def _execute_logit_post_processors(self,
                                       scheduled_requests: ScheduledRequests,
                                       outputs: dict):
        """Apply logit post processors (in-place modify outputs Tensors) if any."""

        if not (self.mapping.is_last_pp_rank()):
            return

        if not isinstance(outputs, dict) or "logits" not in outputs:
            # TODO: support models that don't return outputs as dict
            return

        num_ctx_req = len(scheduled_requests.context_requests)
        logits_tensor = outputs["logits"]

        for idx, request in enumerate(scheduled_requests.all_requests()):
            logits_processors = getattr(request, "py_logits_post_processors",
                                        None)
            if not logits_processors:
                continue

            token_ids = request.get_tokens(0)
            if idx < num_ctx_req and request.py_orig_prompt_len < len(
                    token_ids):
                # Skip as we only need to apply logit processor on the last context request
                continue

            logits_row = logits_tensor[idx]
            # Reshape to align w/ the shape used in the TRT backend,
            # so the same logit processors can be used across both backends.
            logits_row = logits_row.view(1, 1, -1)
            token_ids = [token_ids]
            for lp in logits_processors:
                lp_params = inspect.signature(lp).parameters

                assert 4 <= len(lp_params) <= 5, (
                    "Logit post processor signature must match the `LogitsProcessor` interface "
                    "defined in `tensorrtllm.sampling_params`.")
                lp(request.py_request_id, logits_row, token_ids, None, None)

            logits_tensor[idx] = logits_row.view(-1)
