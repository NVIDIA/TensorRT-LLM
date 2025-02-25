import bisect
import contextlib
import glob
import math
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import safetensors
import torch

import tensorrt_llm._torch
import tensorrt_llm.bindings
import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.attention_backend import *
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionMetadata, AttentionRuntimeFeatures)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.metadata import *
from tensorrt_llm._torch.models import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode, timing
from tensorrt_llm._torch.pyexecutor.distributed import MPIDist
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..._utils import nvtx_range
from ...models.modeling_utils import QuantAlgo
from .config import PyTorchConfig
from .cuda_graph_runner import DecodingCUDAGraphRunner
from .resource_manager import ResourceManager
from .scheduler import ScheduledRequests


class ModelEngine(ABC):

    @abstractmethod
    def get_max_num_sequences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, scheduled_requests: ScheduledRequests,
                new_tokens_device: Optional[torch.Tensor],
                resource_manager: ResourceManager):
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


def validate_and_set_kv_cache_quant(
        model_config: tensorrt_llm._torch.model_config.ModelConfig,
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
            part_weights = torch.load(file,
                                      weights_only=True,
                                      map_location='cpu',
                                      mmap=True)
            weights.update(part_weights)
        return weights

    raise RuntimeError(f"No weight files found in {checkpoint_dir}.")


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
    ):
        self.batch_size = batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len

        self.mapping = mapping
        self.dist = dist
        self.pytorch_backend_config = pytorch_backend_config

        self.attn_runtime_features = attn_runtime_features or AttentionRuntimeFeatures(
        )

        attn_backend = pytorch_backend_config.attn_backend
        self.model = self._load_model(
            model_path,
            mapping=self.mapping,
            attn_backend=attn_backend,
        )
        self.enable_attention_dp = self.model.model_config.mapping.enable_attention_dp
        self.dtype = self.model.config.torch_dtype
        self._init_model_capacity()
        self.ub_buffers = None

        try:
            if pytorch_backend_config.torch_compile_enabled:
                use_ub = pytorch_backend_config.torch_compile_enable_userbuffers and self._init_userbuffers(
                    self.model.config.hidden_size,
                    self.model.model_config.get_quant_config(), self.dtype)
                self.model = torch.compile(
                    self.model,
                    backend=Backend(
                        pytorch_backend_config.torch_compile_inductor_enabled,
                        enable_userbuffers=use_ub),
                    fullgraph=pytorch_backend_config.torch_compile_fullgraph)
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
        self.iter_counter = 0

    def warmup(self, resource_manager: ResourceManager) -> None:
        kv_cache_manager = resource_manager.get_resource_manager(
            'kv_cache_manager')

        def get_cuda_graph_warmup_request(batch_size):
            available_blocks = kv_cache_manager.get_num_free_blocks()
            if available_blocks >= batch_size:
                result = ScheduledRequests()
                result.context_requests = []
                # Add (batch_size - 1) dummy requests with seq_len=1.
                # Should only need one more page per request.
                requests = kv_cache_manager.add_dummy_requests(
                    list(range(batch_size - 1)),
                    is_gen=True,
                )
                available_blocks -= batch_size - 1
                available_tokens = available_blocks * kv_cache_manager.tokens_per_block
                # When we generate last token for the max_seq_len case,
                # we only need to store (max_seq_len - 1) tokens in the KV cache.
                token_num = max(
                    1,
                    min(available_tokens, kv_cache_manager.max_seq_len - 1),
                )
                # Add one dummy request with the maximum possible sequence length.
                # The sequence length is limited by both the max_seq_len and the number of available blocks.
                max_seq_len_request = kv_cache_manager.add_dummy_requests(
                    request_ids=[batch_size - 1],
                    token_nums=[token_num],
                    is_gen=True,
                )[0]
                # Add the longest request before all other seq_len=1 request to simulate the padding CUDA graph case.
                # This batch contains both the longest request and the shortest requests,
                # it also contains the maximum number of requests and the maximum token number,
                # which simulates the extreme case for the padding CUDA graph.
                # Thus we can replay this CUDA graph in all other cases.
                requests.insert(0, max_seq_len_request)
                result.generation_requests = requests
            else:
                result = None
            return result

        def get_torch_compile_warmup_request(batch_size, num_tokens):
            available_blocks = kv_cache_manager.get_num_free_blocks()
            if available_blocks >= batch_size * math.ceil(
                    num_tokens / kv_cache_manager.tokens_per_block):
                # Should only need (at most) one more page per request.
                is_gen = num_tokens == 1
                requests = kv_cache_manager.add_dummy_requests(list(
                    range(batch_size)), [num_tokens] * batch_size,
                                                               is_gen=is_gen)
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
                    for req in result.context_requests:
                        kv_cache_manager.free_resources(req)

        @contextlib.contextmanager
        def no_cuda_graph():
            _run_cuda_graphs = self._run_cuda_graphs
            self._run_cuda_graphs = False
            try:
                yield
            finally:
                self._run_cuda_graphs = _run_cuda_graphs

        if self._torch_compile_enabled:
            # Disable cuda graph capture here so that we can properly capture it later
            with no_cuda_graph():
                for bs in [1, self.batch_size // 2]:
                    for num_tokens in [1, self.max_num_tokens // bs]:
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
                                         new_tokens_device=None,
                                         resource_manager=resource_manager)
                            torch.cuda.synchronize()

        if not self._run_cuda_graphs:
            return

        logger.info(
            f"Creating CUDA graph instances for {len(self._cuda_graph_batch_sizes)} batch sizes."
        )
        for bs in self._cuda_graph_batch_sizes:
            with release_batch(get_cuda_graph_warmup_request(bs)) as batch:
                if batch is None:
                    # No KV cache space!
                    return
                logger.info(f"Run warmup for batch size={bs}")
                self.forward(batch,
                             new_tokens_device=None,
                             resource_manager=resource_manager)
                torch.cuda.synchronize()

    def _set_up_attn_metadata(self, kv_cache_manager: KVCacheManager):
        if kv_cache_manager is None:
            return self.attn_backend.Metadata(
                max_num_requests=self.batch_size,
                max_num_tokens=self.max_num_tokens,
                kv_cache_manager=None,
                mapping=self.mapping,
                runtime_features=self.attn_runtime_features)

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
            runtime_features=self.attn_runtime_features)
        return self.attn_metadata

    def _get_padded_batch(self, scheduled_requests: ScheduledRequests,
                          kv_cache_manager):
        if (not self._run_cuda_graphs or not self._cuda_graph_padding_enabled
                or not scheduled_requests.is_generation_only
                or scheduled_requests.batch_size
                > self._max_cuda_graph_batch_size):
            return None

        batch_size = scheduled_requests.batch_size
        padded_batch_size = self._round_up_batch_size(batch_size)
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

        # Every request ID in the batch needs to be unique for the C++
        # KV cache manager.
        max_req_id = max(req.py_request_id
                         for req in scheduled_requests.generation_requests)
        generation_requests = kv_cache_manager.add_dummy_requests(
            [max_req_id + i + 1 for i in range(padding_size)], is_gen=True)

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
            batch: ScheduledRequests) -> Optional[DecodingCUDAGraphRunner]:
        """
        Get a CUDA graph runner or return None (e.g. if CUDA graphs are disabled
        or if the batch size is too big).
        """

        if not self._run_cuda_graphs or not batch.is_generation_only:
            return None

        batch_size = len(batch.generation_requests)

        if batch_size in self._cuda_graphs:
            return self._cuda_graphs[batch_size]

        if batch_size not in self._cuda_graph_batch_sizes:
            return None

        attn_metadata = self.attn_metadata.create_cuda_graph_metadata(
            batch_size)
        assert attn_metadata.is_cuda_graph

        self._cuda_graphs[batch_size] = DecodingCUDAGraphRunner(
            batch_size, "cuda", attn_metadata)
        return self._cuda_graphs[batch_size]

    def __del__(self) -> None:
        if self.ub_buffers:
            for u in self.ub_buffers:
                ub.ub_deallocate(u)
        torch.cuda.empty_cache()

    def _load_model(self, checkpoint_dir: str, **kwargs):
        config = tensorrt_llm._torch.model_config.ModelConfig.from_pretrained(
            checkpoint_dir, trust_remote_code=True, **kwargs)

        validate_and_set_kv_cache_quant(
            config, self.pytorch_backend_config.kv_cache_dtype)

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

            if hasattr(model, 'llm_checkpoint_dir'):
                weights = load_weights(model.llm_checkpoint_dir)
            else:
                weights = load_weights(checkpoint_dir)
            model.load_weights(weights)
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
        return self.batch_size

    def _prepare_tp_inputs(self,
                           scheduled_requests: ScheduledRequests,
                           kv_cache_manager: KVCacheManager,
                           attn_metadata: AttentionMetadata,
                           new_tokens_device: Optional[torch.Tensor] = None):
        """
        Prepare inputs for Pytorch Model.
        """

        # if new_tokens_device exist, input_ids will only contain new context tokens
        input_ids = []
        sequence_lengths = []
        prompt_lengths = []
        request_ids = []
        gather_ids = []
        position_ids = []
        num_cached_tokens_per_seq = []
        multi_modal_data = []

        batch_idx = 0

        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            all_prompt_tokens = request.get_tokens(0)

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

            request.py_batch_idx = batch_idx
            batch_idx += 1

        num_ctx_requests = batch_idx
        num_ctx_tokens = len(input_ids)
        # Requests with draft tokens are treated like extend requests.
        extend_requests = []
        generation_requests = []
        for request in scheduled_requests.generation_requests:
            if request.has_draft_tokens():
                extend_requests.append(request)
            else:
                generation_requests.append(request)
        is_spec_decode = len(extend_requests) > 0
        if self.pytorch_backend_config.enable_overlap_scheduler:
            assert not is_spec_decode, "speculative decoding does not support overlap scheduler"

        for request in extend_requests:
            request_ids.append(request.py_request_id)
            input_token_id = request.get_last_tokens(0)
            input_ids.append(input_token_id)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(1 + len(request.draft_tokens))
            prompt_lengths.append(request.py_prompt_len)

            past_seen_token_num = request.max_beam_num_tokens - 1
            position_ids.append(past_seen_token_num)

            for i, draft_token in enumerate(request.draft_tokens):
                input_ids.append(draft_token)
                gather_ids.append(len(input_ids) - 1)
                position_ids.append(past_seen_token_num + i + 1)

            num_cached_tokens_per_seq.append(past_seen_token_num)

        # will contain previous batch incices of generation requests
        previous_batch_indices = []
        num_generation_tokens = len(generation_requests)
        sequence_lengths.extend([1] * num_generation_tokens)
        gather_ids.extend(
            list(
                range(
                    len(input_ids) - 1,
                    len(input_ids) + num_generation_tokens)))
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

            # skip dummy generation requests created in CUDA graph mode
            if request.py_batch_idx is not None:
                previous_batch_indices.append(request.py_batch_idx)
                request.py_batch_idx = batch_idx
                batch_idx += 1

        num_tokens = len(input_ids)
        if num_tokens > 0:
            input_ids = torch.tensor(input_ids, dtype=torch.int)
            self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)
        if new_tokens_device is not None:
            previous_batch_tokens = len(previous_batch_indices)
            previous_batch_indices = torch.tensor(previous_batch_indices,
                                                  dtype=torch.int)
            self.previous_batch_indices_cuda[:previous_batch_tokens].copy_(
                previous_batch_indices, non_blocking=True)
            self.input_ids_cuda[
                num_tokens:num_tokens +
                previous_batch_tokens].copy_(new_tokens_device[
                    self.previous_batch_indices_cuda[:previous_batch_tokens]],
                                             non_blocking=True)

        total_num_tokens = len(position_ids)
        position_ids = torch.tensor(position_ids, dtype=torch.int)
        self.position_ids_cuda[:total_num_tokens].copy_(position_ids,
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
            )

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = len(
            scheduled_requests.context_requests) + len(extend_requests)

        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=num_cached_tokens_per_seq)
        attn_metadata.kv_cache_manager = kv_cache_manager

        attn_metadata.prepare()
        self.iter_states['num_ctx_requests'] = num_ctx_requests
        self.iter_states['num_ctx_tokens'] = num_ctx_tokens
        self.iter_states['num_generation_tokens'] = num_generation_tokens
        return {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:total_num_tokens],
            'position_ids':
            self.position_ids_cuda[:total_num_tokens].unsqueeze(0),
            'inputs_embeds': None,
            'multi_modal_data': multi_modal_data
        }, gather_ids if is_spec_decode else None

    def _prepare_tp_inputs_no_cache(self, scheduled_requests: ScheduledRequests,
                                    attn_metadata: AttentionMetadata):
        """
        Prepare inputs for Pytorch Model.
        """
        sequence_lengths = []
        input_ids = []
        gather_ids = []
        position_ids = []
        multi_modal_data = []

        for request in scheduled_requests.context_requests:
            prompt_tokens = request.get_tokens(0)
            input_ids.extend(prompt_tokens)
            if request.position_ids is None:
                position_ids.extend(range(len(prompt_tokens)))
            else:
                position_ids.extend(request.position_ids)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(prompt_tokens))
            prompt_embedding_table = request.prompt_embedding_table()
            if prompt_embedding_table is not None:
                multi_modal_data.append(prompt_embedding_table)

        num_tokens = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = torch.tensor(position_ids, dtype=torch.int)
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
            )

        attn_metadata.num_contexts = len(scheduled_requests.context_requests)

        return {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:num_tokens],
            'position_ids': self.position_ids_cuda[:num_tokens].unsqueeze(0),
            'inputs_embeds': None,
            'multi_modal_data': multi_modal_data
        }, None

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
            if request.draft_tokens
        ]
        generation_requests = [
            request for request in scheduled_requests.generation_requests
            if not request.draft_tokens
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
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = torch.tensor(position_ids, dtype=torch.int)
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

        return {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:num_tokens],
            'position_ids': self.position_ids_cuda[:num_tokens].unsqueeze(0),
            'inputs_embeds': None
        }, gather_ids if is_spec_decode else None

    @nvtx_range("_prepare_inputs")
    def _prepare_inputs(self,
                        scheduled_requests: ScheduledRequests,
                        kv_cache_manager: KVCacheManager,
                        attn_metadata: AttentionMetadata,
                        new_tokens_device: Optional[torch.Tensor] = None):
        if self.mapping is not None and 'cp_type' in self.mapping.cp_config:
            cp_type = self.mapping.cp_config['cp_type']
            if 'star_attention' == cp_type:
                return self._prepare_star_attention_inputs(
                    scheduled_requests, kv_cache_manager, attn_metadata)
            else:
                assert False, f'Unsupport cp_type {cp_type}'
        else:
            return self._prepare_tp_inputs(scheduled_requests, kv_cache_manager,
                                           attn_metadata, new_tokens_device)

    @torch.inference_mode()
    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tokens_device: Optional[torch.Tensor] = None):

        kv_cache_manager = resource_manager.get_resource_manager(
            'kv_cache_manager')

        attn_metadata = self._set_up_attn_metadata(kv_cache_manager)

        if kv_cache_manager is None:
            inputs, gather_ids = self._prepare_tp_inputs_no_cache(
                scheduled_requests, attn_metadata)
            if self.enable_attention_dp:
                all_rank_num_tokens = self.dist.allgather(
                    attn_metadata.num_tokens)
                attn_metadata.all_rank_num_tokens = all_rank_num_tokens
            return self._forward_step(inputs, gather_ids)

        with self._maybe_pad_batch(scheduled_requests,
                                   kv_cache_manager) as scheduled_requests:
            maybe_graph = self._maybe_get_cuda_graph(scheduled_requests)
            if maybe_graph is not None:
                attn_metadata = maybe_graph.attn_metadata
            else:
                attn_metadata = self.attn_metadata

            inputs, gather_ids = self._prepare_inputs(scheduled_requests,
                                                      kv_cache_manager,
                                                      attn_metadata,
                                                      new_tokens_device)
            if self.enable_attention_dp:
                all_rank_num_tokens = self.dist.allgather(
                    attn_metadata.num_tokens)
                attn_metadata.all_rank_num_tokens = all_rank_num_tokens

            self.iter_counter += 1
            if maybe_graph is None:
                return self._forward_step(inputs, gather_ids)
            else:
                assert gather_ids is None, "Cannot use speculative decoding with CUDA graphs."

                if maybe_graph.needs_capture():
                    pool = maybe_graph.capture(
                        lambda inputs: self._forward_step(inputs,
                                                          gather_ids=None),
                        self._cuda_graph_mem_pool,
                    )
                    self._cuda_graph_mem_pool = pool

                return maybe_graph.run(inputs)

    @nvtx_range("_forward_step")
    def _forward_step(self, inputs: Dict[str, Any],
                      gather_ids: Optional[torch.Tensor]) -> torch.Tensor:
        # For simplicity, just return all the the logits if we have special gather_ids
        # from speculative decoding.
        logits = self.model.forward(**inputs,
                                    return_context_logits=gather_ids
                                    is not None)
        if gather_ids is not None:
            return {'logits': logits[gather_ids]}
        else:
            return {'logits': logits}

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
        ub.ub_initialize(self.mapping.tp_size)
        if not ub.ub_is_initialized():
            return False
        self.ub_buffers = [
            ub.ub_allocate(0, hidden_size * self.max_num_tokens * 2),
            ub.ub_allocate(1, hidden_size * self.max_num_tokens * 2),
        ]
        return True
