from abc import ABC, abstractmethod

import torch

import tensorrt_llm
import tensorrt_llm.bindings

from ..._utils import str_dtype_to_torch, trt_dtype_to_torch
from ...bindings import KVCacheType
from ...logger import logger
from ...mapping import Mapping
from ...plugin.plugin import CustomAllReduceHelper
from .llm_request import LlmRequestState
from .resource_manager import ResourceManager
from .scheduler import ScheduledRequests

ModelConfig = tensorrt_llm.bindings.ModelConfig
LayerType = tensorrt_llm.bindings.LayerType


class RuntimeBuffer(ABC):

    def __init__(self, runtime, model_config: ModelConfig, mapping: Mapping,
                 meta_config: dict):
        self.runtime = runtime
        self._model_config = model_config
        self.mapping = mapping
        self.device = torch.device(
            f'cuda:{self.mapping.rank % self.mapping.gpus_per_node}')
        self.meta_config = meta_config
        self.input_buffers = {}
        self.output_buffers = {}

        # create static buffers and pre-allocate buffers
        self.create_static_buffers()
        self.preallocate_buffers()

        # expected tensors
        self.input_tensor_names = []
        self.input_preallocated_tensor_names = []
        self.output_tensor_names = []
        self.output_preallocated_tensor_names = []

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.runtime.engine.get_tensor_dtype(name))
        return dtype

    @abstractmethod
    def create_static_buffers(self):
        pass

    @abstractmethod
    def preallocate_buffers(self):
        pass

    @abstractmethod
    def prepare_batch_inputs(self, scheduled_requests: ScheduledRequests,
                             resource_manager: ResourceManager):
        pass

    @abstractmethod
    def prepare_batch_outputs(self, scheduled_requests: ScheduledRequests,
                              resource_manager: ResourceManager):
        pass


class TRTLLMBuffer(RuntimeBuffer):

    def __init__(self, runtime, model_config: ModelConfig, mapping: Mapping,
                 meta_config: dict):
        self.runtime = runtime
        self._model_config = model_config
        self.mapping = mapping
        self.device = torch.device(
            f'cuda:{self.mapping.rank % self.mapping.gpus_per_node}')
        self.meta_config = meta_config
        self.input_buffers = {}
        self.output_buffers = {}
        self.internal_buffers = {}

        # params
        self.vocab_size_padded = self._model_config.vocab_size_padded(
            self.mapping.world_size)
        self.total_input_size = 0
        self.num_tokens = 0

        # create static buffers and pre-allocate buffers
        self.create_static_buffers()
        self.preallocate_buffers()

        # expected tensors
        self.input_tensor_names = []
        self.input_preallocated_tensor_names = []
        self.output_tensor_names = []
        self.output_preallocated_tensor_names = []
        if self.mapping.is_first_pp_rank():
            self.input_tensor_names += ['input_ids']
        else:
            self.input_tensor_names += ['hidden_states_input']

        if self.mapping.is_last_pp_rank():
            self.output_tensor_names += ['logits']
            if not self._model_config.compute_context_logits:
                self.input_tensor_names += ['last_token_ids']
        else:
            self.output_preallocated_tensor_names += ['hidden_states_output']
        self.input_tensor_names += [
            'sequence_length',
            'context_lengths',
            'host_request_types',
            'host_context_lengths',
        ]
        if self.mapping.tp_size > 1:
            self.input_tensor_names += ['all_reduce_workspace']

    @property
    def max_batch_size(self):
        return self._model_config.max_batch_size

    @property
    def max_input_len(self):
        return self._model_config.max_input_len

    @property
    def max_beam_width(self):
        return self._model_config.max_beam_width

    @property
    def max_num_tokens(self):
        return self._model_config.max_num_tokens

    @property
    def vocab_size(self):
        return self._model_config.vocab_size

    @property
    def hidden_size(self):
        return self._model_config.hidden_size

    @property
    def gather_context_logits(self):
        return self._model_config.compute_context_logits

    @property
    def gather_generation_logits(self):
        return self._model_config.compute_generation_logits

    @property
    def dtype(self):
        return str_dtype_to_torch(self._model_config.data_type)

    def create_static_buffers(self):
        if self.mapping.tp_size > 1:
            self.ipc_buffers, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.mapping.tp_size))
            self.input_buffers.update(
                {'all_reduce_workspace': self.all_reduce_workspace})

    def preallocate_buffers(self):
        # get max buffer size
        max_num_ctx_tokens = self.max_batch_size * self.max_input_len
        max_num_gen_tokens = self.max_batch_size * self.max_beam_width
        if self.max_num_tokens is not None:
            max_num_tokens = self.max_num_tokens
        else:
            max_num_tokens = max(max_num_ctx_tokens, max_num_gen_tokens)

        # pre-allocate buffers
        if self.mapping.has_pp():
            hidden_states = torch.empty(
                (max_num_tokens, self.hidden_size * self.mapping.tp_size),
                dtype=self._tensor_dtype('hidden_states_input'),
                device=self.device)
            self.input_buffers['hidden_states_input'] = hidden_states
            self.output_buffers['hidden_states_output'] = hidden_states

    def prepare_batch_inputs(self, scheduled_requests: ScheduledRequests,
                             resource_manager: ResourceManager):
        context_batch = scheduled_requests.context_requests
        generation_batch = scheduled_requests.generation_requests

        # get buffer size
        seq_slots = []
        context_lengths = []
        seq_lengths = []

        num_ctx_reqs = len(context_batch)
        max_ctx_len = 0
        num_ctx_tokens = []
        for req in context_batch:
            seq_slots.append(req.seq_slot)
            prompt_len = req.prompt_len
            max_ctx_len = max(max_ctx_len, prompt_len)
            num_ctx_tokens.append(prompt_len)

        num_gen_seqs = 0
        num_gen_tokens = []
        for req in generation_batch:
            seq_slots.append(req.seq_slot)
            prompt_len = req.prompt_len
            req_beam_width = 1  # req.sampling_config.beam_width
            num_gen_seqs += req_beam_width
            num_gen_tokens.extend([1] * req_beam_width)
            context_lengths.extend([prompt_len] * req_beam_width)
            seq_lengths.extend([req.max_beam_num_tokens] * req_beam_width)

        self.total_input_size = sum(num_ctx_tokens + num_gen_tokens)
        self.num_tokens = num_ctx_tokens + num_gen_tokens
        context_lengths = num_ctx_tokens + context_lengths
        seq_lengths = num_ctx_tokens + seq_lengths

        # prepare common buffers
        input_ids_list = []
        for req in context_batch:
            input_ids_list.extend(req.get_tokens(0))

        for req in generation_batch:
            for beam in range(1):  # req.sampling_config.beam_width
                input_ids_list.append(
                    req.get_token(beam,
                                  req.get_num_tokens(beam) - 1))

        if self.mapping.is_first_pp_rank():
            input_ids = torch.tensor(input_ids_list,
                                     dtype=torch.int32,
                                     device=self.device)
            self.input_buffers.update({'input_ids': input_ids})
        else:
            shape = (self.total_input_size,
                     self.hidden_size * self.mapping.tp_size)
            self.input_buffers['hidden_states_input'].resize_(*shape)

        if not self.gather_context_logits and self.mapping.is_last_pp_rank():
            num_tokens = torch.tensor(self.num_tokens,
                                      dtype=torch.int32,
                                      device=self.device)
            last_token_ids = torch.cumsum(num_tokens, dim=0).int()
            self.input_buffers.update({'last_token_ids': last_token_ids})

        host_request_types = torch.tensor([0] * num_ctx_reqs +
                                          [1] * num_gen_seqs,
                                          dtype=torch.int32,
                                          device='cpu')
        self.input_buffers.update({'host_request_types': host_request_types})

        context_lengths = torch.tensor(context_lengths,
                                       dtype=torch.int32,
                                       device=self.device)
        host_context_lengths = context_lengths.cpu()
        self.input_buffers.update({'context_lengths': context_lengths})
        self.input_buffers.update(
            {'host_context_lengths': host_context_lengths})

        seq_lengths = torch.tensor(seq_lengths,
                                   dtype=torch.int32,
                                   device=self.device)
        self.input_buffers.update({'sequence_length': seq_lengths})
        return

    def prepare_batch_outputs(self, scheduled_requests: ScheduledRequests,
                              resource_manager: ResourceManager):
        context_batch = scheduled_requests.context_requests
        generation_batch = scheduled_requests.generation_requests

        num_ctx_reqs = len(context_batch)
        num_gen_tokens = 0
        for req in generation_batch:
            num_gen_tokens += 1  # req.sampling_config.beam_width

        num_logits = num_ctx_reqs + num_gen_tokens

        if self.mapping.is_last_pp_rank():
            # TODO: add support for gather_context_logits and gather_generation_logits
            logits = torch.empty((num_logits, self.vocab_size_padded),
                                 dtype=self._tensor_dtype('logits'),
                                 device=self.device)
            self.output_buffers.update({'logits': logits})
        else:
            shape = (self.total_input_size,
                     self.hidden_size * self.mapping.tp_size)
            self.output_buffers['hidden_states_output'].resize_(*shape)


class TRTTransformerBuffer(RuntimeBuffer):

    def __init__(self, runtime, model_config: ModelConfig, mapping: Mapping,
                 meta_config: dict):
        self.runtime = runtime
        self._model_config = model_config
        self.mapping = mapping
        self.device = torch.device(
            f'cuda:{self.mapping.rank % self.mapping.gpus_per_node}')
        self.meta_config = meta_config
        self.input_buffers = {}
        self.output_buffers = {}

        # create static buffers and pre-allocate buffers
        self.create_static_buffers()
        self.preallocate_buffers()

        # expected tensors
        self.input_tensor_names = []
        self.input_preallocated_tensor_names = []
        self.output_tensor_names = []
        self.output_preallocated_tensor_names = []
        if self.mapping.is_first_pp_rank():
            self.input_tensor_names += ['position_ids']

        if self.use_kv_cache:
            self.input_preallocated_tensor_names += ['cache_indirection']
            self.input_tensor_names += ['kv_cache_block_offsets']
            self.input_tensor_names += ['host_kv_cache_block_offsets']
            self.input_tensor_names += ['host_kv_cache_pool_pointers']
            self.input_tensor_names += ['host_kv_cache_pool_mapping']
            self.input_tensor_names += ['host_past_key_value_lengths']
        self.input_tensor_names += [
            'host_max_attention_window_sizes',
            'host_sink_token_length',
            'host_runtime_perf_knobs',
            'host_context_progress',
        ]

    @property
    def max_batch_size(self):
        return self._model_config.max_batch_size

    @property
    def max_beam_width(self):
        return self._model_config.max_beam_width

    @property
    def max_seq_len(self):
        assert self._model_config.max_seq_len > 0, "please use the trtllm-build to rebuild model and set a correct max_seq_len."
        return self._model_config.max_seq_len

    @property
    def layer_types(self):
        return self._model_config.layer_types

    @property
    def num_layers(self):
        num_layers = self._model_config.num_attention_layers(
        ) + self._model_config.num_rnn_layers()
        assert (num_layers) % self.mapping.pp_size == 0, \
            f"num_layers {num_layers} must be a multiple of pipeline parallelism size {self.mapping.pp_size}"
        return num_layers // self.mapping.pp_size

    @property
    def num_attention_layers(self):
        return self._model_config.num_attention_layers(self.mapping.pp_size)

    @property
    def num_rnn_layers(self):
        return self._model_config.num_rnn_layers(self.mapping.pp_size)

    @property
    def first_layer(self):
        return self.num_layers * self.mapping.pp_rank

    @property
    def last_layer(self):
        return self.first_layer + self.num_layers

    @property
    def kv_cache_type(self):
        return self._model_config.kv_cache_type

    @property
    def use_kv_cache(self):
        return self._model_config.kv_cache_type != KVCacheType.DISABLED

    @property
    def max_attention_window_size(self):
        if 'max_attention_window_size' in self.meta_config:
            return self.meta_config['max_attention_window_size']
        else:
            return None

    @property
    def sink_token_length(self):
        if 'sink_token_length' in self.meta_config:
            return self.meta_config['sink_token_length']
        else:
            return 0

    @property
    def multi_block_mode(self):
        if 'multi_block_mode' in self.meta_config:
            return self.meta_config['multi_block_mode']
        else:
            return False

    @property
    def enable_context_fmha_fp32_acc(self):
        if 'enable_context_fmha_fp32_acc' in self.meta_config:
            return self.meta_config['enable_context_fmha_fp32_acc']
        else:
            return False

    def create_static_buffers(self):
        # host_max_attention_window_sizes
        max_attention_window_size = self.max_attention_window_size
        warning = False
        if max_attention_window_size is None:
            max_attention_window_size = [self.max_seq_len
                                         ] * self.num_attention_layers
        elif isinstance(max_attention_window_size, int):
            warning = max_attention_window_size > self.max_seq_len
            max_attention_window_size = min(max_attention_window_size,
                                            self.max_seq_len)
            max_attention_window_size = [max_attention_window_size
                                         ] * self.num_attention_layers
        elif isinstance(max_attention_window_size, list[int]):
            warning = False
            for i in range(len(max_attention_window_size)):
                warning = (max_attention_window_size[i]
                           > self.max_seq_len) or warning
                max_attention_window_size[i] = min(max_attention_window_size[i],
                                                   self.max_seq_len)
        else:
            assert False, "invalid max_attention_window_size!"
        if warning:
            logger.warning(
                "The value of max_attention_window_size should ideally not exceed max_seq_length. "
                "Therefore, it has been adjusted to match the value of max_seq_length."
            )
        host_max_attention_window_sizes = []
        attn_win_size_len = len(max_attention_window_size)
        for i in range(self.num_attention_layers):
            host_max_attention_window_sizes.append(max_attention_window_size[
                (self.layer_types[0:self.first_layer].count(LayerType.ATTENTION)
                 + i) % attn_win_size_len])
        host_max_attention_window_sizes = torch.tensor(
            host_max_attention_window_sizes, dtype=torch.int32, device='cpu')
        self.input_buffers.update({
            'host_max_attention_window_sizes':
            host_max_attention_window_sizes
        })

        # host_sink_token_length
        host_sink_token_length = torch.tensor([self.sink_token_length],
                                              dtype=torch.int32,
                                              device='cpu')
        self.input_buffers.update(
            {'host_sink_token_length': host_sink_token_length})

        # host_runtime_perf_knobs
        host_runtime_perf_knobs = -1 * torch.ones(
            16, dtype=torch.int32, device='cpu')
        host_runtime_perf_knobs[0] = 1 if self.multi_block_mode else 0
        host_runtime_perf_knobs[
            1] = 1 if self.enable_context_fmha_fp32_acc else 0
        self.input_buffers.update(
            {'host_runtime_perf_knobs': host_runtime_perf_knobs})

        # host_context_progress
        host_context_progress = torch.tensor([0], dtype=torch.int64)
        self.input_buffers.update(
            {'host_context_progress': host_context_progress})

    def preallocate_buffers(self):
        # cache_indirection
        if self.max_attention_window_size is not None:
            max_atten_win = max(self.max_attention_window_size)
        else:
            max_atten_win = self.max_seq_len
        cache_indirection = torch.full(
            [self.max_batch_size, self.max_beam_width, max_atten_win],
            0,
            dtype=torch.int32,
            device=self.device)
        self.input_buffers.update({'cache_indirection': cache_indirection})

    def prepare_batch_inputs(self, scheduled_requests: ScheduledRequests,
                             resource_manager: ResourceManager):
        context_batch = scheduled_requests.context_requests
        generation_batch = scheduled_requests.generation_requests

        max_ctx_len = len(context_batch)
        num_gen_seqs = 0
        max_beam_width = 1
        position_ids = []
        past_key_value_len = []

        for req in context_batch:
            prompt_len = req.prompt_len
            position_ids.extend([i for i in range(prompt_len)])
            past_key_value_len.append(prompt_len)

        for req in generation_batch:
            req_beam_width = 1  # req.sampling_config.beam_width
            num_gen_seqs += req_beam_width
            max_beam_width = max(max_beam_width, req_beam_width)
            position_ids.extend(
                [req.get_num_tokens(i) - 1 for i in range(req_beam_width)])
            past_key_value_len.extend([req.max_beam_num_tokens] *
                                      req_beam_width)

        num_reqs = len(context_batch) + len(generation_batch)
        num_seqs = max_ctx_len + num_gen_seqs

        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int32,
                                    device=self.device)
        self.input_buffers.update({'position_ids': position_ids})

        host_past_key_value_lengths = torch.tensor(past_key_value_len,
                                                   dtype=torch.int32,
                                                   device='cpu')
        self.input_buffers.update(
            {'host_past_key_value_lengths': host_past_key_value_lengths})

        # kv block offsets
        host_kv_cache_block_offsets = torch.zeros([
            resource_manager('kv_cache_manager').num_pools, num_seqs, 2,
            resource_manager('kv_cache_manager').max_blocks_per_seq
        ],
                                                  dtype=torch.int32,
                                                  device='cpu')
        num_seqs = 0
        for req_batch in [context_batch, generation_batch]:
            for req in req_batch:
                is_context_req = req.state == LlmRequestState.CONTEXT_INIT
                beam_width = 1 if is_context_req else 1  # req.sampling_config.beam_width
                _ = resource_manager(
                    'kv_cache_manager').impl.copy_block_offsets(
                        host_kv_cache_block_offsets, num_seqs, req.request_id)
                num_seqs += beam_width
        # requests' block offsets collected as [num_pools, num_seqs, 2, max_blocks_per_seq], copy to device
        kv_cache_block_offsets = host_kv_cache_block_offsets.to(self.device)
        self.input_buffers.update({
            'host_kv_cache_block_offsets':
            host_kv_cache_block_offsets,
            'kv_cache_block_offsets':
            kv_cache_block_offsets
        })

        # kv pool pointers
        self.input_buffers.update({
            'host_kv_cache_pool_pointers':
            resource_manager('kv_cache_manager').kv_cache_pool_pointers
        })

        # kv pool mapping
        self.input_buffers.update({
            'host_kv_cache_pool_mapping':
            resource_manager('kv_cache_manager').kv_cache_pool_mapping
        })

        # cache indirection
        shape = (num_reqs, max_beam_width,
                 self.input_buffers['cache_indirection'].shape[-1])
        self.input_buffers['cache_indirection'].resize_(*shape)

        # host_context_progress
        host_context_progress = torch.tensor([0], dtype=torch.int64)
        self.input_buffers.update(
            {'host_context_progress': host_context_progress})

        # reset cache indirection

        # copy cache indirection

        return

    def prepare_batch_outputs(self, scheduled_requests: ScheduledRequests,
                              resource_manager: ResourceManager):
        pass
