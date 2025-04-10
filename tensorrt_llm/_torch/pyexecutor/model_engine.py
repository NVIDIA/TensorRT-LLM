import bisect
import contextlib
import glob
import math
import os
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import safetensors
import torch

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._utils import nvtx_range, release_gc, trace_func
from tensorrt_llm.bindings.executor import GuidedDecodingConfig
from tensorrt_llm.logger import logger
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
from ..metadata import KVCacheParams
from ..model_config import ModelConfig
from ..models import AutoModelForCausalLM
from ..models.modeling_utils import MetaInitMode, timing
from ..pipeline_interface import PipelineInterface
from ..speculative import SpecConfig, SpecMetadata, get_spec_metadata
from ..utils import set_torch_compiling
from .config import LoadFormat, PyTorchConfig
from .cuda_graph_runner import DecodingCUDAGraphRunner
from .distributed import MPIDist
from .guided_decoder import GuidedDecoder
from .layerwise_nvtx_marker import LayerwiseNvtxMarker
from .resource_manager import (BaseResourceManager, KVCacheManager,
                               ResourceManager)
from .scheduler import ScheduledRequests

MAX_UINT64 = (1 << 64) - 1


class ModelEngine(ABC):

    @abstractmethod
    def get_max_num_sequences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tensors_device: Optional[Dict[str, torch.Tensor]],
                extra_model_inputs: Optional[Dict[str, Any]]):
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


def load_weights(checkpoint_dir: str):
    weights = {}
    weight_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
    if weight_files:
        for file in weight_files:
            logger.info(f"Loading {file}")
            part_weights = safetensors.torch.load_file(file)
            weights.update(part_weights)
        return weights

    weight_files = glob.glob(f"{checkpoint_dir}/*.bin")
    if not weight_files:
        weight_files = glob.glob(f"{checkpoint_dir}/*.pth")

    if weight_files:
        for file in weight_files:
            # try mmap first, if failed, turn off mmap
            try:
                part_weights = torch.load(file,
                                          weights_only=True,
                                          map_location='cpu',
                                          mmap=True)
            except Exception:
                logger.warning(
                    f"Failed to load {file} with mmap=True, fallback to mmap=False"
                )
                part_weights = torch.load(file,
                                          weights_only=True,
                                          map_location='cpu',
                                          mmap=False)
            weights.update(part_weights)
        return weights

    raise RuntimeError(f"No weight files found in {checkpoint_dir}.")


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


KV_CACHE_MANAGER_KEY = 'kv_cache_manager'
DRAFT_KV_CACHE_MANAGER_KEY = 'draft_kv_cache_manager'


class PyTorchModelEngine(ModelEngine):

    def __init__(
        self,
        model_path: str,
        pytorch_backend_config: PyTorchConfig,
        batch_size: int = 8,
        max_num_tokens: int = 8192,
        max_seq_len: Optional[int] = None,
        mapping: Optional[Mapping] = None,
        attn_runtime_features: Optional[AttentionRuntimeFeatures] = None,
        dist: Optional[MPIDist] = None,
        spec_config: Optional[SpecConfig] = None,
        guided_decoding_config: Optional[GuidedDecodingConfig] = None,
    ):
        self.ub_buffers = None
        self.batch_size = batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len

        self.mapping = mapping
        if mapping.has_pp():
            PipelineInterface.init_pp_comm(mapping)
        self.dist = dist
        self.pytorch_backend_config = pytorch_backend_config
        self.spec_config = spec_config
        # We keep a reference to the last used spec metadata to
        # accommodate certain target/draft model use cases. See
        # py_executor.py for how this is used.
        self.last_spec_metadata = None

        self.attn_runtime_features = attn_runtime_features or AttentionRuntimeFeatures(
        )

        attn_backend = pytorch_backend_config.attn_backend
        # _convert_load_format should already be called by
        # __post_init__, but call it again just in case.
        # The config object is not a frozen data class, so it's
        # possible the user changed it after initialization.
        pytorch_backend_config._convert_load_format()
        self.model = self._load_model(
            model_path,
            mapping=self.mapping,
            attn_backend=attn_backend,
            load_format=pytorch_backend_config.load_format,
            max_num_tokens=max_num_tokens,
            moe_max_num_tokens=pytorch_backend_config.moe_max_num_tokens,
        )
        if self.pytorch_backend_config.enable_layerwise_nvtx_marker:
            layerwise_nvtx_marker = LayerwiseNvtxMarker()
            module_prefix = 'Model'
            if self.model.model_config and self.model.model_config.pretrained_config and self.model.model_config.pretrained_config.architectures:
                module_prefix = '|'.join(
                    self.model.model_config.pretrained_config.architectures)
            layerwise_nvtx_marker.register_hooks(self.model, module_prefix)

        self.enable_attention_dp = self.model.model_config.mapping.enable_attention_dp
        self._enable_overlap_scheduler = self.pytorch_backend_config.enable_overlap_scheduler
        self.dtype = self.model.config.torch_dtype
        self._init_model_capacity()

        self.guided_decoder: Optional[GuidedDecoder] = None
        if self.mapping.is_last_pp_rank(
        ) and guided_decoding_config is not None:
            self.guided_decoder = GuidedDecoder(guided_decoding_config,
                                                self.batch_size,
                                                self.model.vocab_size_padded)

        try:
            if pytorch_backend_config.torch_compile_enabled:
                set_torch_compiling(True)
                use_ub = pytorch_backend_config.torch_compile_enable_userbuffers and self._init_userbuffers(
                    self.model.config.hidden_size,
                    self.model.model_config.get_quant_config(), self.dtype)
                self.model = torch.compile(
                    self.model,
                    backend=Backend(
                        pytorch_backend_config.torch_compile_inductor_enabled,
                        enable_userbuffers=use_ub),
                    fullgraph=pytorch_backend_config.torch_compile_fullgraph)
            else:
                set_torch_compiling(False)
        except Exception as e:
            import traceback
            traceback.print_exception(Exception, e, e.__traceback__)
            raise e
        self._torch_compile_enabled = pytorch_backend_config.torch_compile_enabled

        self.attn_backend = get_attention_backend(attn_backend)

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
        self._cuda_graph_mem_pool = None
        self._run_cuda_graphs = pytorch_backend_config.use_cuda_graph

        self._cuda_graph_padding_enabled = pytorch_backend_config.cuda_graph_padding_enabled
        self._cuda_graph_batch_sizes = [
            bs for bs in pytorch_backend_config.cuda_graph_batch_sizes
            if bs <= self.max_num_tokens and bs <= self.batch_size
        ]
        self._max_cuda_graph_batch_size = self._cuda_graph_batch_sizes[-1]

        self.previous_batch_indices_cuda = torch.empty((self.max_num_tokens, ),
                                                       dtype=torch.int,
                                                       device='cuda')
        self.input_ids_cuda = torch.empty((self.max_num_tokens, ),
                                          dtype=torch.int,
                                          device='cuda')
        self.position_ids_cuda = torch.empty((self.max_num_tokens, ),
                                             dtype=torch.int,
                                             device='cuda')
        if self.spec_config is not None:
            self.spec_metadata = None
            self.spec_config.update_from_model_config(self.model.config)
            max_num_draft_tokens = self.spec_config.max_draft_tokens * batch_size
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
            self.max_draft_len = spec_config.max_draft_tokens
        else:
            self.without_logits = False
            self.max_draft_len = 0
        self.iter_counter = 0

        # We look up this key in resource_manager during forward to find the
        # kv cache manager. Can be changed to support multiple model engines
        # with different KV cache managers.
        self.kv_cache_manager_key = KV_CACHE_MANAGER_KEY

    def warmup(self, resource_manager: ResourceManager) -> None:
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        spec_resource_manager = resource_manager.get_resource_manager(
            'spec_resource_manager')
        if kv_cache_manager is None:
            logger.info("Skipping warm up as no KV Cache manager allocated.")
            return

        def get_cuda_graph_warmup_request(batch_size):
            available_blocks = kv_cache_manager.get_num_free_blocks()

            max_num_draft_tokens = self.spec_config.max_draft_tokens if self.spec_config is not None else 0

            if available_blocks >= batch_size:
                result = ScheduledRequests()
                result.context_requests = []
                # Add (batch_size - 1) dummy requests with seq_len=1.
                # Should only need one more page per request.
                requests = kv_cache_manager.add_dummy_requests(
                    list(range(batch_size - 1)),
                    is_gen=True,
                    max_num_draft_tokens=max_num_draft_tokens,
                )
                available_blocks -= batch_size - 1
                available_tokens = available_blocks * kv_cache_manager.tokens_per_block
                # When we generate last token for the max_seq_len case,
                # we only need to store (max_seq_len - 1 - max_num_draft_tokens) tokens in the KV cache.
                # For the max_seq_len, some speculative decoding methods need extra kv tokens in kv cache
                # manager to support different kv lengths for the draft/target layers. So, we also
                # need to remove those extra tokens from the max_seq_len.
                token_num = max(
                    1,
                    min(
                        available_tokens, kv_cache_manager.max_seq_len -
                        kv_cache_manager.num_extra_kv_tokens - 1 -
                        max_num_draft_tokens),
                )
                # Add one dummy request with the maximum possible sequence length.
                # The sequence length is limited by both the max_seq_len and the number of available blocks.
                max_seq_len_request = kv_cache_manager.add_dummy_requests(
                    request_ids=[batch_size - 1],
                    token_nums=[token_num],
                    is_gen=True,
                    max_num_draft_tokens=max_num_draft_tokens,
                )[0]
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

        def get_torch_compile_warmup_request(batch_size, num_tokens):
            available_blocks = kv_cache_manager.get_num_free_blocks()
            if available_blocks >= batch_size * math.ceil(
                    num_tokens / kv_cache_manager.tokens_per_block):
                # Should only need (at most) one more page per request.
                is_gen = num_tokens == 1
                max_num_draft_tokens = self.spec_config.max_draft_tokens if self.spec_config is not None and is_gen else 0

                requests = kv_cache_manager.add_dummy_requests(
                    list(range(batch_size)), [num_tokens] * batch_size,
                    is_gen=is_gen,
                    max_num_draft_tokens=max_num_draft_tokens)

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

        @contextlib.contextmanager
        def release_batch(result):
            try:
                yield result
            finally:
                if result is not None:
                    for req in result.generation_requests:
                        kv_cache_manager.free_resources(req)
                        if spec_resource_manager is not None:
                            spec_resource_manager.free_resources(req)
                    for req in result.context_requests:
                        kv_cache_manager.free_resources(req)
                        if spec_resource_manager is not None:
                            spec_resource_manager.free_resources(req)

        @contextlib.contextmanager
        def no_cuda_graph():
            _run_cuda_graphs = self._run_cuda_graphs
            self._run_cuda_graphs = False
            try:
                yield
            finally:
                self._run_cuda_graphs = _run_cuda_graphs

        # TODO: current warmup_request is not suitable for star attention
        cp_type = self.mapping.cp_config.get('cp_type', None)
        if cp_type == 'star_attention':
            return

        # TODO: CUDA graph support with eagle.
        if self.spec_config is not None and self.spec_config.spec_dec_mode.is_eagle3(
        ):
            return

        if self._torch_compile_enabled:
            # Disable cuda graph capture here so that we can properly capture it later
            with no_cuda_graph():
                warmup_batch_size = [1, self.batch_size // 2]
                if self.batch_size < 2:
                    warmup_batch_size = [1]
                for bs in warmup_batch_size:
                    for num_tokens in [
                            1,
                            min(self.max_num_tokens // max(bs, 1),
                                kv_cache_manager.max_seq_len - 1)
                    ]:
                        with release_batch(
                                get_torch_compile_warmup_request(
                                    bs, num_tokens)) as batch:
                            if batch is None:
                                # No KV cache space!
                                continue
                            logger.info(
                                f"Run warmup for batch size={bs}, pure {'context' if num_tokens is not None else 'generation'} phase"
                            )
                            self.forward(batch,
                                         new_tensors_device=None,
                                         resource_manager=resource_manager)
                            torch.cuda.synchronize()

        if self.pytorch_backend_config.autotuner_enabled:
            with no_cuda_graph(), autotune():
                num_tokens = min(self.max_num_tokens,
                                 kv_cache_manager.max_seq_len - 1)
                with release_batch(
                        get_torch_compile_warmup_request(1,
                                                         num_tokens)) as batch:
                    if batch is None:
                        # No KV cache space!
                        pass
                    else:
                        logger.info(f"Run autotuning warmup for batch size={1}")
                        self.forward(batch,
                                     new_tensors_device=None,
                                     resource_manager=resource_manager)
                        torch.cuda.synchronize()

                logger.info(f"Autotuner Cache size after warmup " +
                            str(len(AutoTuner.get().profiling_cache)))

        if not self._run_cuda_graphs:
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
                logger.info(f"Run warmup for batch size={bs}")
                self.forward(batch,
                             new_tensors_device=None,
                             resource_manager=resource_manager)
                torch.cuda.synchronize()

    def _set_up_attn_metadata(self,
                              kv_cache_manager: KVCacheManager,
                              is_dummy_forward: bool = False):
        # is_dummy_forward is used to indicate whether the forward is
        # a dummy forward for memory estimation OR
        # a real forward w.o. kv cache
        if kv_cache_manager is None:
            return self.attn_backend.Metadata(
                max_num_requests=self.batch_size,
                max_num_tokens=self.max_num_tokens,
                kv_cache_manager=None,
                mapping=self.mapping,
                runtime_features=self.attn_runtime_features,
                enable_flash_mla=self.model.model_config.enable_flash_mla,
                is_dummy_attention=is_dummy_forward)

        if self.attn_metadata is not None:
            # This assertion can be relaxed if needed: just create a new metadata
            # object if it changes.
            assert self.attn_metadata.kv_cache_manager is kv_cache_manager
            return self.attn_metadata

        self.attn_metadata = self.attn_backend.Metadata(
            max_num_requests=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            kv_cache_manager=kv_cache_manager,
            mapping=self.mapping,
            runtime_features=self.attn_runtime_features,
            enable_flash_mla=self.model.model_config.enable_flash_mla)
        return self.attn_metadata

    def _set_up_spec_metadata(
            self,
            spec_resource_manager: Optional[BaseResourceManager],
            no_cache=False):
        if no_cache:
            return get_spec_metadata(
                self.spec_config,
                self.batch_size,
                spec_resource_manager=spec_resource_manager)

        if self.spec_metadata is not None:
            return self.spec_metadata
        self.spec_metadata = get_spec_metadata(
            self.spec_config,
            self.batch_size,
            spec_resource_manager=spec_resource_manager)
        return self.spec_metadata

    def _get_padded_batch(self, scheduled_requests: ScheduledRequests,
                          kv_cache_manager):
        can_run_cuda_graph = scheduled_requests.can_run_cuda_graph
        batch_size = scheduled_requests.batch_size
        new_batch_size = batch_size
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
            return None

        padded_batch_size = self._round_up_batch_size(new_batch_size)
        if batch_size == padded_batch_size:
            return None

        padding_size = padded_batch_size - batch_size

        available_blocks = kv_cache_manager.get_num_free_blocks()

        # No padding if:
        # 1) Not enough KV cache space.
        # 2) It would create too many concurrent requests.
        # 2 is not strictly required, but we should probably
        # respect the requirement just in case that changes in the future.
        if available_blocks < padding_size or padding_size + scheduled_requests.batch_size > self.batch_size:
            return None

        # Set the dummy request ids starting at (uint64 max value - padding_size - 1) to avoid conflict with
        # active request IDs
        max_req_id = MAX_UINT64 - padding_size - 1
        max_num_draft_tokens = self.spec_config.max_draft_tokens if self.spec_config is not None else 0
        generation_requests = kv_cache_manager.add_dummy_requests(
            [max_req_id + i + 1 for i in range(padding_size)],
            is_gen=True,
            max_num_draft_tokens=max_num_draft_tokens)
        scheduled_requests.generation_requests.extend(generation_requests)
        return generation_requests

    @contextlib.contextmanager
    def _maybe_pad_batch(self, scheduled_requests: ScheduledRequests,
                         kv_cache_manager):
        """
        CUDA graphs can only be used for specific batch sizes.

        If using CUDA graphs, this method will add dummy requests to the given
        batch so we can always use a CUDA graph. It is a context manager
        because the padded requests allocate KV pages that should be freed
        when you're done with them.
        """
        padding_requests = self._get_padded_batch(scheduled_requests,
                                                  kv_cache_manager)
        try:
            yield scheduled_requests
        finally:
            if padding_requests is not None:
                padding_len = len(padding_requests)
                scheduled_requests.generation_requests = scheduled_requests.generation_requests[:
                                                                                                -padding_len]
                for req in padding_requests:
                    kv_cache_manager.free_resources(req)

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
        spec_config: Optional[SpecConfig] = None
    ) -> Optional[DecodingCUDAGraphRunner]:
        """
        Get a CUDA graph runner or return None (e.g. if CUDA graphs are disabled
        or if the batch size is too big).
        """
        spec_max_draft_tokens = spec_config.max_draft_tokens if spec_config is not None else 0
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

        attn_metadata = self.attn_metadata.create_cuda_graph_metadata(
            batch_size, False, spec_max_draft_tokens)
        assert attn_metadata.is_cuda_graph

        if self.spec_config is not None:
            spec_metadata = self.spec_metadata.create_cuda_graph_metadata(
                batch_size)
            spec_metadata.draft_tokens = self.draft_tokens_cuda
        else:
            spec_metadata = None

        pipeline_interface = None
        if self.mapping.pp_rank > 0:
            pipeline_interface = self.model.create_pipeline_interface(
                batch_size)
        self._cuda_graphs[batch_size] = DecodingCUDAGraphRunner(
            batch_size, "cuda", attn_metadata, spec_metadata,
            pipeline_interface, self.mapping.has_pp())
        return self._cuda_graphs[batch_size]

    def __del__(self) -> None:
        if self.ub_buffers:
            for u in self.ub_buffers:
                ub.ub_deallocate(u.addr)
        # Release model weights.
        release_gc()

    def _load_model(self, checkpoint_dir: str, load_format: LoadFormat,
                    max_num_tokens: int, moe_max_num_tokens: int, **kwargs):
        config = ModelConfig.from_pretrained(checkpoint_dir,
                                             trust_remote_code=True,
                                             **kwargs)
        config.spec_config = self.spec_config
        config.max_num_tokens = max_num_tokens
        config.moe_max_num_tokens = moe_max_num_tokens

        validate_and_set_kv_cache_quant(
            config, self.pytorch_backend_config.kv_cache_dtype)
        num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
        if num_layers > 0:
            config.pretrained_config.num_hidden_layers = num_layers

        with timing("Model init total"):
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

            if load_format == LoadFormat.AUTO:
                if hasattr(model, 'llm_checkpoint_dir'):
                    weights = load_weights(model.llm_checkpoint_dir)
                else:
                    weights = load_weights(checkpoint_dir)

                model.load_weights(weights)

            elif load_format == LoadFormat.DUMMY:
                initialize_dummy_weights(model)

            else:
                raise NotImplementedError(
                    f"No load support for load format: {load_format}")

            torch.cuda.current_stream().synchronize()
        return model

    def _init_max_seq_len(self):
        if self.max_seq_len is None:
            inferred_max_seq_len = self.model.infer_max_seq_len()
            logger.info(
                f"max_seq_len is not specified, using inferred value {inferred_max_seq_len}"
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

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports. PyExecutor need this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size if self.mapping.has_pp() else 1
        return num_batches * self.batch_size

    def _preprocess_inputs(self, inputs: Dict[str, Any]):
        """
        Make some changes to the device inputs and avoid block the async data transfer
        """
        if self.spec_config is not None and self._enable_overlap_scheduler:
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
            new_tensors_device: Optional[Dict[str, torch.Tensor]] = None):
        """
        Prepare inputs for Pytorch Model.
        """

        # if new_tensors_device exist, input_ids will only contain new context tokens
        input_ids = []
        sequence_lengths = []
        prompt_lengths = []
        request_ids = []
        gather_ids = []
        position_ids = []
        num_cached_tokens_per_seq = []
        multi_modal_data = []
        draft_tokens = []
        draft_lens = []
        mrope_config = defaultdict(list)

        batch_idx = 0

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
            past_seen_token_num = request.context_current_position
            num_cached_tokens_per_seq.append(past_seen_token_num)
            prompt_embedding_table = request.prompt_embedding_table()
            if prompt_embedding_table is not None:
                multi_modal_data.append(prompt_embedding_table)

            mrope_rotary_cos_sin = request.get_mrope_rotary_cos_sin()
            if mrope_rotary_cos_sin is not None:
                mrope_config['mrope_rotary_cos_sin'].append(
                    mrope_rotary_cos_sin)
            request.py_batch_idx = batch_idx
            batch_idx += 1

        num_ctx_requests = batch_idx
        num_ctx_tokens = len(input_ids)
        new_tokens_device, new_tokens_lens_device, next_draft_tokens_device = None, None, None
        if new_tensors_device is not None:
            # speculative decoding cases: [batch, 1 + draft_len], others: [batch]
            new_tokens_device = new_tensors_device["new_tokens_device"]
            if self.without_logits:
                new_tokens_lens_device = new_tensors_device[
                    "new_tokens_lens_device"]  # [batch]
                next_draft_tokens_device = new_tensors_device[
                    "next_draft_tokens_device"]  # [batch, draft_len]

        # Requests with draft tokens are treated like extend requests.
        extend_requests = []
        generation_requests = []
        for request in scheduled_requests.generation_requests:
            if request.py_draft_tokens is not None or next_draft_tokens_device is not None:
                extend_requests.append(request)
            else:
                generation_requests.append(request)

            mrope_position_deltas = request.mrope_position_deltas
            if mrope_position_deltas is not None:
                mrope_config['mrope_position_deltas'].append(
                    torch.tensor([mrope_position_deltas],
                                 dtype=torch.int32).to('cuda',
                                                       non_blocking=True))
        is_spec_decode = len(extend_requests) > 0
        if self._enable_overlap_scheduler and is_spec_decode:
            spec_dec_mode = self.spec_config.spec_dec_mode
            assert spec_dec_mode.support_overlap_scheduler(
            ), f"{self.spec_config.spec_dec_name} does not support overlap scheduler"

        # will contain previous batch incices of generation requests
        previous_batch_indices = []
        previous_pos_indices = []
        for request in extend_requests:
            if next_draft_tokens_device is None or request.py_batch_idx is None:
                num_draft_tokens = len(request.py_draft_tokens)
                input_ids.append(request.get_last_tokens(0))
                gather_ids.append(len(input_ids) - 1)
                sequence_lengths.append(1 + num_draft_tokens)
                past_seen_token_num = request.max_beam_num_tokens - 1
                position_ids.append(past_seen_token_num)
                draft_lens.append(num_draft_tokens)
                prompt_lengths.append(num_draft_tokens + 1)
                # draft tokens
                input_ids.extend(request.py_draft_tokens)
                gather_ids.extend(
                    list(
                        range(
                            len(input_ids) - num_draft_tokens, len(input_ids))))
                position_ids.extend(
                    list(
                        range(past_seen_token_num + 1,
                              past_seen_token_num + 1 + num_draft_tokens)))
                draft_tokens.extend(request.py_draft_tokens)
                num_cached_tokens_per_seq.append(past_seen_token_num)
            else:
                # batch index
                previous_batch_idx = request.py_batch_idx
                request.py_batch_idx = batch_idx
                batch_idx += 1
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

        sequence_lengths.extend([1] * len(generation_requests))
        gather_ids.extend(
            list(
                range(len(position_ids),
                      len(position_ids) + len(generation_requests))))
        for request in generation_requests:
            if new_tokens_device is None or request.py_batch_idx is None:
                input_ids.append(request.get_last_tokens(0))
                past_seen_token_num = request.max_beam_num_tokens - 1
            else:
                past_seen_token_num = request.max_beam_num_tokens
            request_ids.append(request.py_request_id)
            position_ids.append(past_seen_token_num)
            num_cached_tokens_per_seq.append(past_seen_token_num)
            prompt_lengths.append(request.py_prompt_len)
            draft_lens.append(0)

            # skip dummy generation requests created in CUDA graph mode
            if request.py_batch_idx is not None and new_tokens_device is not None:
                previous_batch_indices.append(request.py_batch_idx)

            request.py_batch_idx = batch_idx
            batch_idx += 1

        num_tokens = len(input_ids)
        previous_batchs = len(previous_batch_indices)
        if num_tokens > 0:
            input_ids = torch.tensor(input_ids,
                                     dtype=torch.int,
                                     pin_memory=True)
            if len(scheduled_requests.context_requests) == 0:
                self.input_ids_cuda[previous_batchs:num_tokens +
                                    previous_batchs].copy_(input_ids,
                                                           non_blocking=True)
            else:
                self.input_ids_cuda[:num_tokens].copy_(input_ids,
                                                       non_blocking=True)
        if next_draft_tokens_device is not None:
            if len(previous_batch_indices) > 0:
                previous_batch_indices = torch.tensor(previous_batch_indices,
                                                      dtype=torch.int,
                                                      pin_memory=True)
                self.previous_batch_indices_cuda[:previous_batchs].copy_(
                    previous_batch_indices, non_blocking=True)
                # previous input ids
                previous_batch_tokens = previous_batchs * (1 +
                                                           self.max_draft_len)
                self.input_ids_cuda[
                    num_tokens:num_tokens +
                    previous_batch_tokens].copy_(new_tokens_device[
                        self.previous_batch_indices_cuda[:previous_batchs], :].
                                                 flatten(),
                                                 non_blocking=True)
                # previous draft tokens
                previous_batch_draft_tokens = previous_batchs * self.max_draft_len
                self.draft_tokens_cuda[:previous_batch_draft_tokens].copy_(
                    next_draft_tokens_device[
                        self.previous_batch_indices_cuda[:previous_batchs], :].
                    flatten(),
                    non_blocking=True)
                # prepare data for the preprocess inputs
                kv_len_offsets_device = new_tokens_lens_device - self.max_draft_len - 1
                previous_pos_indices = torch.tensor(previous_pos_indices,
                                                    dtype=torch.int,
                                                    pin_memory=True)
                self.previous_pos_indices_cuda[:previous_batch_tokens].copy_(
                    previous_pos_indices, non_blocking=True)
                self.previous_pos_id_offsets_cuda[:previous_batch_tokens].copy_(
                    new_tokens_lens_device[
                        self.previous_pos_indices_cuda[:previous_batch_tokens]],
                    non_blocking=True)
                self.previous_kv_lens_offsets_cuda[:previous_batchs].copy_(
                    kv_len_offsets_device[
                        self.previous_batch_indices_cuda[:previous_batchs]],
                    non_blocking=True)
            else:
                # change the data to zeros to skip the value changes in _preprocess_inputs
                self.previous_pos_id_offsets_cuda *= 0
                self.previous_kv_lens_offsets_cuda *= 0
        elif new_tokens_device is not None:
            previous_batch_tokens = len(previous_batch_indices)
            previous_batch_indices = torch.tensor(previous_batch_indices,
                                                  dtype=torch.int,
                                                  pin_memory=True)
            self.previous_batch_indices_cuda[:previous_batch_tokens].copy_(
                previous_batch_indices, non_blocking=True)
            if len(scheduled_requests.context_requests) == 0:
                self.input_ids_cuda[:previous_batchs].copy_(new_tokens_device[
                    self.previous_batch_indices_cuda[:previous_batchs]],
                                                            non_blocking=True)
            else:
                self.input_ids_cuda[
                    num_tokens:num_tokens + previous_batchs].copy_(
                        new_tokens_device[
                            self.previous_batch_indices_cuda[:previous_batchs]],
                        non_blocking=True)

        total_num_tokens = len(position_ids)
        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=True)
        self.position_ids_cuda[:total_num_tokens].copy_(position_ids,
                                                        non_blocking=True)
        if self.spec_config is not None:
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

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = len(scheduled_requests.context_requests)
        if self.spec_config is not None and self.spec_config.spec_dec_mode.extend_ctx(
        ):
            attn_metadata.num_contexts += len(extend_requests)

        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=0 if self.spec_config is None else
            self.spec_config.num_extra_kv_tokens)
        attn_metadata.kv_cache_manager = kv_cache_manager

        attn_metadata.prepare()

        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:total_num_tokens],
            'position_ids':
            self.position_ids_cuda[:total_num_tokens].unsqueeze(0),
            'inputs_embeds': None,
            'multi_modal_data': multi_modal_data,
            'mrope_config': mrope_config
        }

        if spec_metadata is not None:
            total_draft_lens = sum(draft_lens)
            if len(draft_tokens) > 0:
                draft_tokens = torch.tensor(draft_tokens,
                                            dtype=torch.int,
                                            pin_memory=True)
                self.draft_tokens_cuda[:len(draft_tokens)].copy_(
                    draft_tokens, non_blocking=True)
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

        if self.mapping.has_pp():
            pipeline_interface = None
            if self.mapping.pp_rank > 0:
                pipeline_interface = self.model.create_pipeline_interface(
                    inputs['input_ids'].shape[0])
                pipeline_interface.recv()
            inputs['pipeline_interface'] = pipeline_interface

        num_generation_tokens = len(generation_requests) + len(
            extend_requests) + sum(draft_lens)
        self.iter_states['num_ctx_requests'] = num_ctx_requests
        self.iter_states['num_ctx_tokens'] = num_ctx_tokens
        self.iter_states['num_generation_tokens'] = num_generation_tokens
        return inputs, self.gather_ids_cuda[:len(gather_ids
                                                 )] if is_spec_decode else None

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
            prompt_embedding_table = request.prompt_embedding_table()
            if prompt_embedding_table is not None:
                multi_modal_data.append(prompt_embedding_table)

        num_tokens = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.int, pin_memory=True)
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=True)
        self.position_ids_cuda[:num_tokens].copy_(position_ids,
                                                  non_blocking=True)
        if self.spec_config is not None:
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
        if not attn_metadata.is_dummy_attention and attn_metadata.kv_cache_manager is None:
            assert isinstance(
                attn_metadata,
                (VanillaAttentionMetadata, TrtllmAttentionMetadata)
            ), "Only vanilla and trtllm attention metadata are supported for no cache attention for now"
            attn_metadata.max_seq_len = self.max_seq_len
            attn_metadata.request_ids = request_ids
            attn_metadata.prepare()

        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:num_tokens],
            'position_ids': self.position_ids_cuda[:num_tokens].unsqueeze(0),
            'inputs_embeds': None,
            'multi_modal_data': multi_modal_data
        }

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

        if self.mapping.has_pp():
            pipeline_interface = None
            if self.mapping.pp_rank > 0:
                pipeline_interface = self.model.create_pipeline_interface(
                    inputs['input_ids'].shape[0])
                pipeline_interface.recv()
            inputs['pipeline_interface'] = pipeline_interface

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
        output_token_idx = 0
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
            request.output_token_idx = output_token_idx
            output_token_idx += 1
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
            request.output_token_idx = output_token_idx
            output_token_idx += 1
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
            request.output_token_idx = output_token_idx
            output_token_idx += 1

        num_tokens = len(input_ids)
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

    @nvtx_range("_prepare_inputs")
    def _prepare_inputs(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[Dict[str, torch.Tensor]] = None):
        if self.mapping is not None and 'cp_type' in self.mapping.cp_config:
            cp_type = self.mapping.cp_config['cp_type']
            if 'star_attention' == cp_type:
                assert not self.mapping.has_pp(
                ), "Star attention does not support pipeline parallel yet"
                return self._prepare_star_attention_inputs(
                    scheduled_requests, kv_cache_manager, attn_metadata)
            else:
                assert False, f'Unsupport cp_type {cp_type}'
        else:
            return self._prepare_tp_inputs(scheduled_requests, kv_cache_manager,
                                           attn_metadata, spec_metadata,
                                           new_tensors_device)

    @torch.inference_mode()
    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tensors_device: Optional[Dict[str, torch.Tensor]] = None,
                extra_model_inputs: Optional[Dict[str, Any]] = None,
                is_dummy_forward: bool = False):

        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)

        attn_metadata = self._set_up_attn_metadata(kv_cache_manager,
                                                   is_dummy_forward)
        if self.spec_config is not None:
            spec_resource_manager = resource_manager.get_resource_manager(
                'spec_resource_manager')
            spec_metadata = self._set_up_spec_metadata(spec_resource_manager,
                                                       no_cache=kv_cache_manager
                                                       is None)
        else:
            spec_metadata = None

        if kv_cache_manager is None:
            inputs, gather_ids = self._prepare_tp_inputs_no_cache(
                scheduled_requests, attn_metadata, spec_metadata)
            if extra_model_inputs is not None:
                inputs.update(extra_model_inputs)
            self.last_spec_metadata = spec_metadata

            if self.mapping.has_pp() and not self.mapping.is_last_pp_rank():
                pp_interface = self._forward_step_intermediate(inputs)
                pp_interface.send()
                return self._post_forward_intermediate(inputs, pp_interface,
                                                       gather_ids)
            else:
                return self._forward_step(inputs, gather_ids)

        with self._maybe_pad_batch(scheduled_requests,
                                   kv_cache_manager) as scheduled_requests:
            maybe_graph = self._maybe_get_cuda_graph(
                scheduled_requests, spec_config=self.spec_config)
            if maybe_graph is not None:
                attn_metadata = maybe_graph.attn_metadata
                if self.spec_config is not None:
                    spec_metadata = maybe_graph.spec_metadata
            else:
                attn_metadata = self.attn_metadata
                if self.spec_config is not None:
                    spec_metadata = self.spec_metadata

            inputs, gather_ids = self._prepare_inputs(scheduled_requests,
                                                      kv_cache_manager,
                                                      attn_metadata,
                                                      spec_metadata,
                                                      new_tensors_device)
            if extra_model_inputs is not None:
                inputs.update(extra_model_inputs)
            self.last_spec_metadata = spec_metadata

            self.iter_counter += 1

            if maybe_graph is None:
                if self.mapping.has_pp() and not self.mapping.is_last_pp_rank():
                    pp_interface = self._forward_step_intermediate(inputs)
                    pp_interface.send()
                    outputs = self._post_forward_intermediate(
                        inputs, pp_interface, gather_ids)
                else:
                    outputs = self._forward_step(inputs, gather_ids)
            else:
                if maybe_graph.needs_capture():
                    if not self.mapping.is_last_pp_rank():
                        capture_fn = lambda inputs: self._forward_step_intermediate(
                            inputs)
                    else:
                        capture_fn = lambda inputs: self._forward_step(
                            inputs, gather_ids=None)

                    pool = maybe_graph.capture(capture_fn,
                                               self._cuda_graph_mem_pool)
                    self._cuda_graph_mem_pool = pool

                outputs = maybe_graph.run(inputs)
                if not self.mapping.is_last_pp_rank():
                    pp_interface = PipelineInterface(*outputs)
                    pp_interface.send()
                    outputs = self._post_forward_intermediate(inputs,
                                                              pp_interface,
                                                              gather_ids=None)

            # Note: To overlap the CPU and GPU computation as much as possible,
            # guided_decoder.build should be called immediately after the launch of the single step;
            # while guided_decoder.execute should be called right before the samplings.
            # We can insert other CPU computation between them in the future.
            if self.mapping.is_last_pp_rank(
            ) and self.guided_decoder is not None:
                guided_decoder_resource_manager = resource_manager.get_resource_manager(
                    "guided_decoder_resource_manager")
                self.guided_decoder.build(scheduled_requests,
                                          guided_decoder_resource_manager)
                self.guided_decoder.execute(scheduled_requests,
                                            outputs['logits'],
                                            guided_decoder_resource_manager)

            return outputs

    def model_forward(self, **kwargs):
        if self.mapping.rank == 0 and int(
                os.environ.get("TLLM_TRACE_MODEL_FORWARD", "0")) == 1:
            return trace_func(self.model.forward)(**kwargs)
        else:
            return self.model.forward(**kwargs)

    @nvtx_range("_forward_step")
    def _forward_step(self, inputs: Dict[str, Any],
                      gather_ids: Optional[torch.Tensor]) -> torch.Tensor:
        inputs = self._preprocess_inputs(inputs)
        if self.without_logits:
            outputs = self.model_forward(**inputs)
            return outputs

        # For simplicity, just return all the the logits if we have special gather_ids
        # from speculative decoding.
        logits = self.model_forward(**inputs,
                                    return_context_logits=gather_ids
                                    is not None)
        if gather_ids is not None:
            return {'logits': logits[gather_ids]}
        else:
            return {'logits': logits}

    @nvtx_range("_forward_step_intermediate")
    def _forward_step_intermediate(self, inputs: Dict[str, Any]):
        pipeline_interface = self.model_forward(**inputs)
        return pipeline_interface

    @nvtx_range("_post_forward_intermediate")
    def _post_forward_intermediate(self, inputs: Dict[str, Any],
                                   pipeline_interface: PipelineInterface,
                                   gather_ids: Optional[torch.Tensor]):
        """
        Instead of returning the logits for intermediate pipeline stages, we return the hidden states at last tokens.
        This is useful to allocate the new tokens to recv from previous pipeline stage.
        """
        attn_metadata = inputs['attn_metadata']
        hidden_states = pipeline_interface['hidden_states']
        if len(hidden_states.shape) == 1:  # During run without KV cache
            hidden_states = hidden_states.unsqueeze(0)

        if gather_ids is None:
            if attn_metadata is not None:
                last_tokens = torch.cumsum(
                    attn_metadata.seq_lens_cuda,
                    dim=0,
                    dtype=torch.long,
                ) - 1
                hidden_states = hidden_states[last_tokens]
            else:
                hidden_states = hidden_states[-1]

        if gather_ids is not None:
            return {'hidden_states': hidden_states[gather_ids]}
        else:
            return {'hidden_states': hidden_states}

    def _init_userbuffers(self, hidden_size, quant_config, dtype):
        # No quant, do not allow UB
        if self.mapping.tp_size <= 1:
            return False

        if quant_config is None:
            return False

        # UB currently only support FP8 quant
        if not quant_config.layer_quant_mode.has_fp8_qdq():
            return False

        if dtype != torch.float16 and dtype != torch.bfloat16:
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
