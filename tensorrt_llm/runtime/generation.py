# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart

from .._utils import pad_vocab_size, trt_dtype_to_torch
from ..logger import logger
from ..mapping import Mapping
from ..quantization import QuantMode
from .kv_cache_manager import GenerationSequence, KVCacheManager
from .session import _scoped_stream


def to_word_list_format(word_dict: List[List[str]], tokenizer=None):
    '''
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    '''
    assert tokenizer != None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(word)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def _prepare_input_ids(tensors: Sequence[torch.Tensor]):
    tensors = [torch.flatten(t) for t in tensors]
    data = torch.unsqueeze(torch.concat(tensors), 0)
    row_lengths = [t.size(0) for t in tensors]
    row_lengths = torch.tensor(row_lengths,
                               dtype=torch.int32,
                               device=data.device)
    return (data, row_lengths)


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


def _update_cuda_graph_instance(instance, graph):
    err = cudart.cudaGraphExecUpdate(instance, graph)
    if err != cudart.cudaError_t.cudaSuccess:
        # When updating cuda graph failed, destroy and instantiate one.
        CUASSERT(cudart.cudaGraphExecDestroy(instance))
        instance = CUASSERT(cudart.cudaGraphInstantiate(graph, 0))[0]
    return instance


def _prepare_attention_mask(
    input_ids: torch.Tensor,
    pad_id: Optional[int] = None,
):
    if pad_id is not None:
        return input_ids.ne(pad_id).int()
    else:
        return torch.ones(input_ids.shape,
                          dtype=torch.int32,
                          device=input_ids.device)


def _tile_beam_width(tensor: torch.Tensor, num_beams: int):
    new_shape = np.array(tensor.shape)
    new_shape[0] = new_shape[0] * num_beams

    tile_size = np.ones(new_shape.shape, dtype=np.int32)
    tile_size = np.insert(tile_size, 1, num_beams)

    new_tensor = torch.unsqueeze(tensor, 1)
    new_tensor = new_tensor.tile(tile_size.tolist())
    new_tensor = new_tensor.reshape(new_shape.tolist())
    return new_tensor


class _Runtime(object):
    runtime_rank: int
    runtime: trt.Runtime
    engine: trt.ICudaEngine
    ctx_context: trt.IExecutionContext
    context_0: trt.IExecutionContext
    context_1: trt.IExecutionContext
    cuda_graph_instances: List[cudart.cudaGraphExec_t]

    def __init__(self, engine_buffer, mapping: Mapping):
        self.__prepare(mapping, engine_buffer)

    def __create_and_setup_context(self, address, profile_idx,
                                   stream) -> trt.IExecutionContext:
        context = self.engine.create_execution_context_without_device_memory()
        assert context is not None
        context.device_memory = address
        context.set_optimization_profile_async(profile_idx, stream)
        return context

    def __prepare(self, mapping: Mapping, engine_buffer):
        self.runtime_rank = mapping.rank
        local_rank = self.runtime_rank % mapping.gpus_per_node
        torch.cuda.set_device(local_rank)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.runtime = trt.Runtime(logger.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        assert self.engine is not None
        # The device_memory_size stores the memory required by the largest profile
        address = CUASSERT(cudart.cudaMalloc(self.engine.device_memory_size))[0]

        # cuda graph ping-pong instances
        self.cuda_graph_instances = [None for _ in range(2)]

        with _scoped_stream() as stream:
            if self.engine.num_optimization_profiles == 1:
                # At step = 0, context_1 is active
                # At step = 1, context_0 is active
                # At step = 2, context_1 is active
                self.context_0 = self.__create_and_setup_context(
                    address, 0, stream)
                self.context_1 = self.__create_and_setup_context(
                    address, 0, stream)
                self.ctx_context = self.context_1
            elif self.engine.num_optimization_profiles == 2:
                # At step = 0, ctx_context is active
                # At step = 1, context_0 is active
                # At step = 2, context_1 is active
                self.ctx_context = self.__create_and_setup_context(
                    address, 0, stream)
                self.context_0 = self.__create_and_setup_context(
                    address, 1, stream)
                self.context_1 = self.__create_and_setup_context(
                    address, 1, stream)
            else:
                assert False, "Maximum of up to two optimization profiles only"

    def _set_shape(self, context: trt.IExecutionContext,
                   shape_dict: Dict[str, List[int]]):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                ok = context.set_input_shape(name, shape_dict[name])
                logger.debug(
                    f"setting input tensor {name} with shape {shape_dict[name]}"
                )
                if not ok:
                    raise ValueError(
                        f"Couldn't assign {name} with shape {shape_dict[name]}, "
                        f"engine supports [min, opt, max] = {self.engine.get_profile_shape(context.active_optimization_profile, name)}"
                    )

    def _set_buffer(self, context: trt.IExecutionContext,
                    buffer_dict: Dict[str, torch.Tensor]):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in buffer_dict.keys():
                dtype = self.engine.get_tensor_dtype(name)
                shape = context.get_tensor_shape(name)
                buffer_dict[name] = torch.zeros(tuple(shape),
                                                dtype=trt_dtype_to_torch(dtype),
                                                device='cuda')
            assert buffer_dict[name].is_contiguous(
            ), f"{name} is not contiguous()"
            context.set_tensor_address(name, buffer_dict[name].data_ptr())

    def _run(self, context: trt.IExecutionContext, stream=None) -> bool:
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        ok = context.execute_async_v3(stream)
        return ok


@dataclass
class ModelConfig:
    vocab_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    gpt_attention_plugin: bool
    remove_input_padding: bool = False
    model_name: str = ""
    paged_kv_cache: bool = False
    tokens_per_block: int = 64
    use_prompt_tuning: bool = False
    quant_mode: QuantMode = QuantMode(0)


@dataclass
class SamplingConfig:
    end_id: int
    pad_id: int

    num_beams: int = field(default=1)
    temperature: Union[float, torch.Tensor] = field(default=1.0)
    top_k: Union[int, torch.Tensor] = field(default=1)
    top_p: Union[float, torch.Tensor] = field(default=0.0)
    length_penalty: Union[float, torch.Tensor] = field(default=1.0)
    repetition_penalty: Union[float, torch.Tensor] = field(default=1.0)
    min_length: Union[int, torch.Tensor] = field(default=1)
    presence_penalty: Union[float, torch.Tensor] = field(default=0.0)
    use_beam_hyps: bool = field(default=True)

    ## None here means user didn't set it, and dynamicDecodeOp.cpp take optional value
    ## The real default value is set in dynamicDecodeOp.cpp when it's None
    beam_search_diversity_rate: Union[float, torch.Tensor] = field(init=False,
                                                                   default=None)
    random_seed: Union[int, torch.Tensor] = field(init=False, default=None)
    output_cum_log_probs: bool = field(init=False, default=False)
    output_log_probs: bool = field(init=False, default=False)


class GenerationSession(object):

    _model_config: ModelConfig
    mapping: Mapping
    runtime: _Runtime
    device: torch.device
    batch_size: int
    buffer_allocated: bool
    debug_mode: bool
    quant_mode: QuantMode
    cuda_graph_mode: bool

    def __init__(self,
                 model_config: ModelConfig,
                 engine_buffer,
                 mapping: Mapping,
                 debug_mode=False,
                 cuda_graph_mode=False):
        assert isinstance(model_config, ModelConfig)
        self._model_config = model_config
        self.mapping = mapping
        self.runtime = _Runtime(engine_buffer, mapping)
        self.device = torch.device(
            f'cuda:{self.runtime.runtime_rank % mapping.gpus_per_node}')
        torch.cuda.set_device(self.device)
        self.debug_mode = debug_mode

        self.cuda_graph_mode = cuda_graph_mode
        # Optional inputs for dynamic decoder
        self.top_p_decay = None
        self.top_p_min = None
        self.top_p_reset_ids = None
        #TODO: in tensorrt_llm/cpp/tensorrt_llm/thop/dynamicDecodeOp.cpp it's T, can be float or half?
        self.embedding_bias_opt = None

        self.buffer = None
        self.buffer_allocated = False

        pp_size = 1
        self.vocab_size_padded = pad_vocab_size(self.vocab_size,
                                                self.mapping.tp_size)
        self.decoder_logits_dtype = self._tensor_dtype('logits')
        if self.decoder_logits_dtype not in [torch.float16, torch.float32]:
            logger.warning(
                "Logits dtype not supported by decoder. Falling back to float32. You may want to change the logits dtype to float16 in your model definition."
            )
            self.decoder_logits_dtype = torch.float32
        self.dynamic_decoder = torch.classes.FasterTransformer.DynamicDecodeOp(
            self.vocab_size, self.vocab_size_padded, self.mapping.tp_size,
            pp_size, self.decoder_logits_dtype)

        self.gather_tree = torch.ops.tensorrt_llm.gather_tree

        expected_tensor_names = ['input_ids'] \
            + [f'past_key_value_{i}' for i in range(model_config.num_layers)] \
            + ['logits'] \
            + [f'present_key_value_{i}' for i in range(model_config.num_layers)] \
            + ['position_ids'] \
            + ['last_token_ids']

        expected_tensor_names += ['cache_indirection']

        if self.paged_kv_cache:
            expected_tensor_names += [
                f'kv_cache_block_pointers_{i}' for i in range(self.num_layers)
            ]

        if model_config.gpt_attention_plugin:
            expected_tensor_names += [
                'sequence_length',
                'context_lengths',
                'host_request_types',
                'host_past_key_value_lengths',
            ]
            if model_config.remove_input_padding:
                expected_tensor_names.append('host_context_lengths')
        else:
            expected_tensor_names += [
                'attention_mask',
            ]

        if model_config.use_prompt_tuning:
            expected_tensor_names += [
                'prompt_embedding_table', 'tasks', 'prompt_vocab_size'
            ]

        found_tensor_names = [
            self.runtime.engine.get_tensor_name(i)
            for i in range(self.runtime.engine.num_io_tensors)
        ]
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected, to use this GenerationSession, " \
                    "you need to use GPTLMHeadModel.prepare_inputs to create TRT Network inputs."
            )

    @property
    def vocab_size(self):
        return self._model_config.vocab_size

    @property
    def num_layers(self):
        return self._model_config.num_layers

    @property
    def num_heads(self):
        return self._model_config.num_heads

    @property
    def hidden_size(self):
        return self._model_config.hidden_size

    @property
    def use_gpt_attention_plugin(self):
        return self._model_config.gpt_attention_plugin

    @property
    def paged_kv_cache(self):
        return self._model_config.paged_kv_cache

    @property
    def tokens_per_block(self):
        return self._model_config.tokens_per_block

    @property
    def remove_input_padding(self):
        return self._model_config.remove_input_padding

    @property
    def num_heads_kv(self):
        return self._model_config.num_kv_heads

    @property
    def head_size(self):
        return self.hidden_size // self.num_heads

    @property
    def quant_mode(self):
        return self._model_config.quant_mode

    def __setup_decoder(self, input_ids: torch.Tensor,
                        sampling_config: SamplingConfig,
                        host_context_lengths: torch.Tensor):
        '''Allocate buffers and setup the post-processing decoder kernel
        '''
        batch_size = host_context_lengths.shape[0]
        scfg = sampling_config  # just to make a shorter name, no other meaning
        if isinstance(scfg.top_k, torch.Tensor):
            assert scfg.top_k.dtype == torch.int32, f"scfg.top_k.dtype ({scfg.top_k.dtype}) must be torch.int32"
            assert scfg.top_k.shape[
                0] == batch_size, f"scfg.top_k.shape[0] ({scfg.top_k.shape[0]}) must equal to batch_size ({batch_size})"
            self.top_k = scfg.top_k
        else:
            self.top_k = torch.full([batch_size], scfg.top_k, dtype=torch.int32)

        if isinstance(scfg.top_p, torch.Tensor):
            assert scfg.top_p.dtype == torch.float32, f"scfg.top_p.dtype ({scfg.top_p.dtype}) must be torch.float32"
            assert scfg.top_p.shape[
                0] == batch_size, f"scfg.top_p.shape[0] ({top_p.shape[0]}) must equal to batch_size ({batch_size})"
            self.top_p = scfg.top_p
        else:
            self.top_p = torch.full([batch_size],
                                    scfg.top_p,
                                    dtype=torch.float32)

        if isinstance(scfg.temperature, torch.Tensor):
            assert scfg.temperature.dtype == torch.float32, f"scfg.temperature.dtype ({scfg.temperature.dtype}) must be torch.float32"
            assert scfg.temperature.shape[
                0] == batch_size, f"scfg.temperature.shape[0] ({scfg.temperature.shape[0]}) must equal to batch_size ({batch_size})"
            self.temperature = scfg.temperature
        else:
            self.temperature = torch.full([batch_size],
                                          scfg.temperature,
                                          dtype=torch.float32)

        if isinstance(scfg.repetition_penalty, torch.Tensor):
            assert scfg.repetition_penalty.dtype == torch.float32, f"scfg.repetition_penalty.dtype ({scfg.repetition_penalty.dtype}) must be torch.float32"
            assert scfg.repetition_penalty.shape[
                0] == batch_size, f"scfg.repetition_penalty.shape[0] ({scfg.repetition_penalty.shape[0]}) must equal to batch_size ({batch_size})"
            self.repetition_penalty = scfg.repetition_penalty
        elif scfg.repetition_penalty == 1.0:
            self.repetition_penalty = None
        else:
            self.repetition_penalty = torch.full([batch_size],
                                                 scfg.repetition_penalty,
                                                 dtype=torch.float32)

        self.length_penalty = torch.FloatTensor([scfg.length_penalty
                                                 ])  # only support scalar now

        if isinstance(scfg.presence_penalty, torch.Tensor):
            assert scfg.presence_penalty.dtype == torch.float32, f"scfg.presence_penalty.dtype ({scfg.presence_penalty.dtype}) must be torch.float32"
            assert scfg.presence_penalty.shape[
                0] == batch_size, f"scfg.presence_penalty.shape[0] ({scfg.presence_penalty.shape[0]}) must equal to batch_size ({batch_size})"
            self.presence_penalty = scfg.presence_penalty
        elif scfg.presence_penalty == 0.0:
            self.presence_penalty = None
        else:
            self.presence_penalty = torch.full([batch_size],
                                               scfg.presence_penalty,
                                               dtype=torch.float32)

        assert (
            scfg.presence_penalty == 0.0 or scfg.repetition_penalty == 0.0
        ), f"presence_penalty({scfg.presence_penalty}) and repetition_penalty({scfg.repetition_penalty}) cannot be larger than 0.0 at the same time."

        if isinstance(scfg.min_length, torch.Tensor):
            assert scfg.min_length.dtype == torch.int32, f"scfg.min_length.dtype ({scfg.min_length.dtype}) must be torch.int32"
            assert scfg.min_length.shape[
                0] == batch_size, f"scfg.min_length.shape[0] ({scfg.min_length.shape[0]}) must equal to batch_size ({batch_size})"
            self.min_length = scfg.min_length
        else:
            self.min_length = torch.full([batch_size],
                                         scfg.min_length,
                                         dtype=torch.int32)

        if isinstance(scfg.beam_search_diversity_rate, torch.Tensor):
            assert scfg.beam_search_diversity_rate.dtype == torch.float32, f"scfg.beam_search_diversity_rate.dtype ({scfg.beam_search_diversity_rate.dtype}) must be torch.float32"
            assert scfg.beam_search_diversity_rate.shape[
                0] == batch_size, f"scfg.beam_search_diversity_rate.shape[0] ({scfg.beam_search_diversity_rate.shape[0]}) must equal to batch_size ({batch_size})"
            self.beam_search_diversity_rate = scfg.beam_search_diversity_rate
        elif scfg.beam_search_diversity_rate is not None:
            self.beam_search_diversity_rate = torch.full(
                [batch_size],
                scfg.beam_search_diversity_rate,
                dtype=torch.float32)
        else:
            self.beam_search_diversity_rate = None

        if isinstance(scfg.random_seed, torch.Tensor):
            assert scfg.random_seed.dtype == torch.int64, f"scfg.random_seed.dtype ({scfg.random_seed.dtype}) must be torch.int64"
            assert scfg.random_seed.shape[
                0] == batch_size, f"scfg.random_seed.shape[0] ({scfg.random_seed.shape[0]}) must equal to batch_size ({batch_size})"
            self.random_seed = scfg.random_seed
        elif scfg.random_seed is not None:
            self.random_seed = torch.full([batch_size],
                                          scfg.random_seed,
                                          dtype=torch.int64)
        else:
            self.random_seed = None

        self.dynamic_decoder.setup(
            batch_size, scfg.num_beams, self.top_k, self.top_p,
            self.temperature, self.repetition_penalty, self.presence_penalty,
            self.min_length, self.length_penalty,
            self.beam_search_diversity_rate, self.random_seed, self.top_p_decay,
            self.top_p_min, self.top_p_reset_ids)

        assert scfg.end_id is not None, "end_id cannot be none"
        assert scfg.pad_id is not None, 'pad_id cannot be none'
        self.end_ids = torch.full((batch_size * scfg.num_beams, ),
                                  scfg.end_id,
                                  dtype=torch.int32,
                                  device=self.device)
        max_context_length = host_context_lengths.max()

        if input_ids.shape[0] != host_context_lengths.shape[0]:
            # dim 0 of input_ids is not batch size, which means remove_padding is enabled
            split_ids_list = list(
                torch.split(input_ids,
                            host_context_lengths.numpy().tolist(),
                            dim=1))
            padded_input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(split_ids_list,
                                           dtype=torch.int32,
                                           device='cuda'),
                scfg.pad_id).reshape(batch_size, max_context_length)
        else:
            padded_input_ids = input_ids
        if scfg.num_beams > 1:
            tiled_input_ids = _tile_beam_width(padded_input_ids, scfg.num_beams)
            tiled_input_ids = tiled_input_ids.reshape(batch_size,
                                                      scfg.num_beams,
                                                      max_context_length)
            tiled_input_ids.permute(2, 0, 1)
            self.output_ids = torch.cat(
                (tiled_input_ids,
                 torch.full((batch_size, scfg.num_beams,
                             self.max_seq_length - max_context_length),
                            scfg.end_id,
                            dtype=padded_input_ids.dtype,
                            device=padded_input_ids.device)),
                axis=-1)
        else:
            self.output_ids = torch.cat(
                (padded_input_ids,
                 torch.full(
                     (batch_size, self.max_seq_length - max_context_length),
                     scfg.end_id,
                     dtype=padded_input_ids.dtype,
                     device=padded_input_ids.device)),
                axis=-1)

        self.parent_ids = torch.zeros(
            (batch_size, scfg.num_beams, self.max_seq_length),
            dtype=torch.int32,
            device=self.device)

        if scfg.num_beams > 1:
            self.new_tokens = torch.zeros([batch_size, scfg.num_beams, 1],
                                          dtype=torch.int32,
                                          device=self.device)
        else:
            self.new_tokens = torch.zeros([batch_size, 1],
                                          dtype=torch.int32,
                                          device=self.device)

        if scfg.num_beams > 1 or scfg.output_cum_log_probs:
            self.cum_log_probs = torch.full((batch_size, scfg.num_beams),
                                            -1e20,
                                            dtype=torch.float32,
                                            device=self.device)
            self.cum_log_probs[:, 0] = 0.0
        else:
            self.cum_log_probs = None

        if scfg.output_log_probs:
            self.log_probs = torch.zeros(
                (self.max_new_tokens, batch_size, scfg.num_beams),
                dtype=torch.float32,
                device=self.device)
        else:
            self.log_probs = None

        self.finished = torch.zeros((batch_size, scfg.num_beams),
                                    dtype=torch.bool,
                                    device=self.device)

        if scfg.use_beam_hyps:
            self.beam_hyps_output_ids_tgt = torch.full(
                size=[batch_size, scfg.num_beams * 2, self.max_seq_length],
                fill_value=scfg.end_id,
                dtype=torch.int32,
                device=self.device)
            self.beam_hyps_sequence_lengths_tgt = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.int32,
                device=self.device)
            self.beam_hyps_cum_log_probs = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_normed_scores = torch.zeros(
                [batch_size, scfg.num_beams * 2],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_log_probs = torch.zeros(
                [batch_size, scfg.num_beams * 2, self.max_seq_length],
                dtype=torch.float,
                device=self.device)
            self.beam_hyps_min_normed_scores = torch.zeros([batch_size],
                                                           dtype=torch.float,
                                                           device=self.device)
            self.beam_hyps_num_beams = torch.zeros([batch_size],
                                                   dtype=torch.int32,
                                                   device=self.device)
            self.beam_hyps_is_done = torch.zeros([batch_size],
                                                 dtype=torch.bool,
                                                 device=self.device)
        else:
            self.beam_hyps_output_ids_tgt = None
            self.beam_hyps_sequence_lengths_tgt = None
            self.beam_hyps_cum_log_probs = None
            self.beam_hyps_normed_scores = None
            self.beam_hyps_log_probs = None
            self.beam_hyps_min_normed_scores = None
            self.beam_hyps_num_beams = None
            self.beam_hyps_is_done = None

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.runtime.engine.get_tensor_dtype(name))
        return dtype

    def setup(self,
              batch_size: int,
              max_context_length: int,
              max_new_tokens: int,
              beam_width: int = 1):
        # Store these params related to buffer size to check against
        # the input shape with the params given in decode()
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_context_length + max_new_tokens
        self.beam_width = beam_width

        self.buffer = {
            'logits':
            torch.empty((batch_size, self.vocab_size_padded),
                        dtype=self._tensor_dtype('logits'),
                        device=self.device)
        }

        if self.paged_kv_cache:
            blocks = batch_size * beam_width * math.ceil(
                self.max_seq_length / self.tokens_per_block)
            cache_shape = (
                blocks,
                2,
                self.num_heads_kv,
                self.tokens_per_block,
                self.head_size,
            )
        else:
            cache_shape = (
                batch_size,
                2,
                self.num_heads_kv,
                self.max_seq_length,
                self.head_size,
            )
        for i in range(self.num_layers):
            if self.quant_mode.has_fp8_kv_cache():
                # Since torch does not support fp8 now, using int8 here.
                kv_cache_type = torch.int8
            else:
                kv_cache_type = self._tensor_dtype(f'present_key_value_{i}')
            self.buffer[f'present_key_value_{i}'] = torch.empty(
                cache_shape, dtype=kv_cache_type, device=self.device)
        if self.use_gpt_attention_plugin:
            self.sequence_length_buffer = torch.ones((batch_size, ),
                                                     dtype=torch.int32,
                                                     device=self.device)
        else:
            # We need two set of kv cache buffers,
            # one for inputs, and the other for outputs.
            # They will take turns to act as input and output buffers.
            for i in range(self.num_layers):
                self.buffer[f'1_present_key_value_{i}'] = torch.empty(
                    cache_shape,
                    dtype=self._tensor_dtype(f'present_key_value_{i}'),
                    device=self.device)

        self.buffer_allocated = True

    def _get_context_shape_buffer(self,
                                  input_ids: torch.Tensor,
                                  context_lengths: torch.Tensor,
                                  host_context_lengths: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  last_token_ids: torch.Tensor,
                                  attention_mask: torch.Tensor,
                                  cache_indirection: torch.Tensor,
                                  kv_cache_block_pointers: List[torch.Tensor],
                                  prompt_embedding_table: torch.Tensor = None,
                                  tasks: torch.Tensor = None,
                                  prompt_vocab_size: torch.Tensor = None):
        ctx_shape = {
            'input_ids': input_ids.shape,
            'context_lengths': context_lengths.shape,
            'position_ids': position_ids.shape,
            'last_token_ids': last_token_ids.shape,
            'cache_indirection': cache_indirection.shape,
        }
        ctx_buffer = {
            'input_ids': input_ids.contiguous(),
            'logits': self.buffer['logits'],
            'context_lengths': context_lengths.contiguous(),
            'position_ids': position_ids.contiguous(),
            'last_token_ids': last_token_ids.contiguous(),
            'cache_indirection': cache_indirection.contiguous(),
        }
        # import pdb; pdb.set_trace()
        if prompt_embedding_table is not None:
            ctx_buffer[
                'prompt_embedding_table'] = prompt_embedding_table.contiguous()
            ctx_shape['prompt_embedding_table'] = prompt_embedding_table.shape

            ctx_buffer['tasks'] = tasks.contiguous()
            ctx_shape['tasks'] = tasks.shape

            ctx_buffer['prompt_vocab_size'] = prompt_vocab_size.contiguous()
            ctx_shape['prompt_vocab_size'] = prompt_vocab_size.shape

        if self.paged_kv_cache:
            for idx in range(self.num_layers):
                ctx_buffer[
                    f'kv_cache_block_pointers_{idx}'] = kv_cache_block_pointers[
                        idx].contiguous()
                shape = kv_cache_block_pointers[idx].shape
                shape = [shape[0] * shape[1], *shape[2:]]
                ctx_shape[f'kv_cache_block_pointers_{idx}'] = shape

        batch_size = context_lengths.shape[0]
        for idx in range(self.num_layers):
            if not self.use_gpt_attention_plugin:
                kv_cache_shape = (batch_size, 2, self.num_heads_kv, 0,
                                  self.head_size)
                # for empty tensor, TRT does not really use the tensor data, so any dtype is fine
                kv_cache_buffer = torch.zeros((1, ),
                                              dtype=torch.float32,
                                              device=self.device)
                ctx_shape.update({
                    f'past_key_value_{idx}': kv_cache_shape,
                })
                ctx_buffer.update({
                    f'past_key_value_{idx}':
                    kv_cache_buffer,
                    f'present_key_value_{idx}':
                    self.buffer[f'present_key_value_{idx}'],
                })
            else:
                key_value_cache = self.buffer[f'present_key_value_{idx}']
                cache_shape = key_value_cache.shape
                ctx_shape.update({
                    f'past_key_value_{idx}': cache_shape,
                })
                ctx_buffer.update({
                    f'past_key_value_{idx}': key_value_cache,
                    f'present_key_value_{idx}': key_value_cache
                })
        if self.use_gpt_attention_plugin:
            host_request_types = torch.zeros_like(context_lengths,
                                                  device='cpu').int()
            ctx_shape.update({
                'sequence_length': (batch_size, ),
                'host_past_key_value_lengths': (batch_size, ),
                'host_request_types': host_request_types.shape,
            })
            ctx_buffer.update({
                'sequence_length':
                self.sequence_length_buffer,
                'host_past_key_value_lengths':
                torch.tensor([0] * batch_size, dtype=torch.int32),
                'host_request_types':
                host_request_types.contiguous(),
            })
            if self.remove_input_padding:
                ctx_buffer[
                    'host_context_lengths'] = host_context_lengths.contiguous()
                ctx_shape['host_context_lengths'] = host_context_lengths.shape
        else:
            ctx_shape.update({'attention_mask': attention_mask.shape})
            ctx_buffer.update({'attention_mask': attention_mask.contiguous()})
        return ctx_shape, ctx_buffer

    def _get_next_step_shape_buffer(self,
                                    batch_size: int,
                                    beam_width: int,
                                    max_context_length: int,
                                    step: int,
                                    context_lengths: torch.Tensor,
                                    host_context_lengths: torch.Tensor,
                                    position_ids: torch.Tensor,
                                    last_token_ids: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    cache_indirection: torch.Tensor,
                                    kv_cache_block_pointers: List[torch.Tensor],
                                    prompt_embedding_table: torch.Tensor = None,
                                    tasks: torch.Tensor = None,
                                    prompt_vocab_size: torch.Tensor = None):
        next_step_shape = {
            'input_ids':
            (1, batch_size * beam_width) if self.remove_input_padding else
            (batch_size * beam_width, 1),
            'context_lengths':
            context_lengths.shape,
            'position_ids':
            position_ids.shape,
            'last_token_ids':
            last_token_ids.shape,
            'cache_indirection':
            cache_indirection.shape,
        }
        next_step_buffer = {
            'input_ids': self.new_tokens,
            'logits': self.buffer['logits'],
            'context_lengths': context_lengths.contiguous(),
            'position_ids': position_ids.contiguous(),
            'last_token_ids': last_token_ids.contiguous(),
            'cache_indirection': cache_indirection.contiguous(),
        }
        if self.remove_input_padding:
            next_step_shape['host_context_lengths'] = host_context_lengths.shape
            next_step_buffer[
                'host_context_lengths'] = host_context_lengths.contiguous()

        if self.paged_kv_cache:
            for idx in range(self.num_layers):
                next_step_buffer[
                    f'kv_cache_block_pointers_{idx}'] = kv_cache_block_pointers[
                        idx].contiguous()
                shape = kv_cache_block_pointers[idx].shape
                shape = [shape[0] * shape[1], *shape[2:]]
                next_step_shape[f'kv_cache_block_pointers_{idx}'] = shape

        if prompt_embedding_table is not None:
            next_step_buffer[
                'prompt_embedding_table'] = prompt_embedding_table.contiguous()
            next_step_shape[
                'prompt_embedding_table'] = prompt_embedding_table.shape

            next_step_buffer['tasks'] = tasks.contiguous()
            next_step_shape['tasks'] = tasks.shape

            next_step_buffer[
                'prompt_vocab_size'] = prompt_vocab_size.contiguous()
            next_step_shape['prompt_vocab_size'] = prompt_vocab_size.shape

        for idx in range(self.num_layers):
            if not self.use_gpt_attention_plugin:
                if step % 2:
                    next_step_buffer.update({
                        f'past_key_value_{idx}':
                        self.buffer[f'1_present_key_value_{idx}'],
                        f'present_key_value_{idx}':
                        self.buffer[f'present_key_value_{idx}'],
                    })
                else:
                    next_step_buffer.update({
                        f'past_key_value_{idx}':
                        self.buffer[f'present_key_value_{idx}'],
                        f'present_key_value_{idx}':
                        self.buffer[f'1_present_key_value_{idx}'],
                    })
                next_shape = (batch_size * beam_width, 2, self.num_heads_kv,
                              max_context_length + step, self.head_size)
                next_step_shape[f'past_key_value_{idx}'] = next_shape
            else:
                key_value_cache = self.buffer[f'present_key_value_{idx}']
                cache_shape = key_value_cache.shape
                next_step_buffer.update({
                    f'past_key_value_{idx}':
                    key_value_cache,
                    f'present_key_value_{idx}':
                    key_value_cache,
                })
                next_step_shape[f'past_key_value_{idx}'] = cache_shape
        if self.use_gpt_attention_plugin:
            host_request_types = torch.ones_like(context_lengths,
                                                 device='cpu').int()
            next_step_shape.update({
                'sequence_length': (batch_size * beam_width, ),
                'host_past_key_value_lengths': (batch_size * beam_width, ),
                'host_request_types':
                host_request_types.shape
            })
            next_step_buffer.update({
                # Sequence lengths are not used in the context phase actually.
                'sequence_length':
                self.sequence_length_buffer,
                'host_past_key_value_lengths':
                torch.tensor([max_context_length + step] *
                             (batch_size * beam_width),
                             dtype=torch.int32),
                'host_request_types':
                host_request_types,
            })
            if self.remove_input_padding:
                next_step_buffer[
                    'host_context_lengths'] = host_context_lengths.contiguous()
                next_step_shape[
                    'host_context_lengths'] = host_context_lengths.shape
        else:
            next_step_shape.update({'attention_mask': attention_mask.shape})
            next_step_buffer.update({
                'attention_mask':
                attention_mask.contiguous(),
            })
        return next_step_shape, next_step_buffer

    def _prepare_context_inputs(self, batch_size, context_lengths,
                                host_context_lengths, use_gpt_attention_plugin,
                                remove_input_padding, **kwargs):

        last_token_ids = context_lengths.detach().clone()
        if use_gpt_attention_plugin:
            max_context_length = kwargs.pop('max_context_length')
            if remove_input_padding:
                position_ids = torch.unsqueeze(
                    torch.concat([
                        torch.arange(0,
                                     host_context_lengths[i],
                                     dtype=torch.int32,
                                     device='cuda') for i in range(batch_size)
                    ]), 0)
                last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
            else:
                position_ids = torch.tensor(range(max_context_length),
                                            dtype=torch.int32,
                                            device='cuda').reshape(
                                                [1,
                                                 -1]).expand([batch_size, -1])
            return {
                'position_ids': position_ids,
                'last_token_ids': last_token_ids
            }
        else:
            input_ids = kwargs.pop('input_ids')
            pad_id = kwargs.pop('pad_id', None)
            attention_mask = _prepare_attention_mask(input_ids, pad_id)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.int()

            return {
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'last_token_ids': last_token_ids
            }

    def _prepare_generation_inputs(self, batch_size, context_lengths,
                                   use_gpt_attention_plugin,
                                   remove_input_padding, **kwargs):
        last_token_ids = torch.ones_like(context_lengths)
        if use_gpt_attention_plugin:
            step = kwargs.pop('step')
            position_ids = context_lengths + step
            if remove_input_padding:
                position_ids = torch.unsqueeze(position_ids, 0)
                last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
            else:
                position_ids = torch.unsqueeze(position_ids, 1)

            return {
                'position_ids': position_ids,
                'last_token_ids': last_token_ids
            }
        else:
            attention_mask = kwargs.pop('attention_mask')
            num_beams = kwargs.pop('num_beams')
            attention_mask = torch.cat((attention_mask,
                                        attention_mask.new_ones(
                                            (batch_size * num_beams, 1))),
                                       dim=-1).contiguous()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids.int()

            return {
                'position_ids': position_ids,
                'last_token_ids': last_token_ids,
                'attention_mask': attention_mask,
            }

    def decode_batch(self, input_ids: Sequence[torch.Tensor],
                     sampling_config: SamplingConfig):
        input_ids, context_lengths = _prepare_input_ids(input_ids)
        return self.decode(input_ids, context_lengths, sampling_config)

    def decode(self,
               input_ids: torch.Tensor,
               context_lengths: torch.Tensor,
               sampling_config: SamplingConfig,
               prompt_embedding_table: torch.Tensor = None,
               tasks: torch.Tensor = None,
               prompt_vocab_size: torch.Tensor = None,
               do_return_sequence_length: bool = False,
               stop_words_list=None,
               bad_words_list=None,
               no_repeat_ngram_size=None):
        scfg = sampling_config
        batch_size = context_lengths.size(0)
        beam_width = scfg.num_beams
        max_context_length = torch.max(context_lengths).item()
        host_context_lengths = context_lengths.cpu()
        assert batch_size == self.batch_size, \
            "Given batch size is different from the one used in setup()," \
            "rerun the setup function with the new batch size to avoid buffer overflow."
        assert max_context_length == self.max_context_length, \
            "Given input length is large then the one used in setup()," \
            "rerun the setup function with the new max_context_length to avoid buffer overflow."
        assert beam_width == self.beam_width, \
            "Given beam width is different from the one used in setup()," \
            "rerun the setup function with the new beam width to avoid buffer overflow."
        ite = 0  # index of local batches, will always be 0 if pp_size = 1

        self.__setup_decoder(input_ids, scfg, host_context_lengths)
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        sequence_limit_lengths = torch.full((batch_size, 1),
                                            self.max_seq_length,
                                            dtype=torch.int32,
                                            device=self.device)

        # Sequence_lengths for the dynamic decoder still has the input paddings.
        sequence_lengths = torch.full((batch_size * beam_width, 1),
                                      max_context_length,
                                      dtype=torch.int32,
                                      device=self.device)

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                self.max_seq_length,
            ),
                       0,
                       dtype=torch.int32,
                       device=self.device),
            torch.full((
                batch_size,
                beam_width,
                self.max_seq_length,
            ),
                       0,
                       dtype=torch.int32,
                       device=self.device)
        ]  # ping-pong buffers

        # Init KV cache block manager
        if self.paged_kv_cache:
            max_blocks_per_seq = math.ceil(self.max_seq_length /
                                           self.tokens_per_block)
            blocks = batch_size * beam_width * max_blocks_per_seq
            memory_pools = [
                self.buffer[f'present_key_value_{i}']
                for i in range(self.num_layers)
            ]
            self.kv_cache_manager = KVCacheManager(memory_pools, blocks,
                                                   self.tokens_per_block,
                                                   max_blocks_per_seq,
                                                   beam_width)

            # Add sequences to the manager
            for bi in range(batch_size):
                generation_sequence = GenerationSequence(seq_idx=bi,
                                                         batch_idx=bi)
                self.kv_cache_manager.add_sequence(generation_sequence,
                                                   max_context_length)

        kv_cache_block_pointers = []
        # start context phase
        for step in range(0, self.max_new_tokens):
            if step % 2:
                context = self.runtime.context_0
                this_src_cache_indirection = cache_indirections[1]
                this_tgt_cache_indirection = cache_indirections[0]
                next_src_cache_indirection = cache_indirections[0]
            else:
                context = self.runtime.context_1
                this_src_cache_indirection = cache_indirections[0]
                this_tgt_cache_indirection = cache_indirections[1]
                next_src_cache_indirection = cache_indirections[1]

            if step == 0:
                model_inputs = self._prepare_context_inputs(
                    batch_size=batch_size,
                    context_lengths=context_lengths,
                    host_context_lengths=host_context_lengths,
                    use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                    remove_input_padding=self.remove_input_padding,
                    max_context_length=max_context_length,
                    input_ids=input_ids,
                    pad_id=scfg.pad_id)

                position_ids = model_inputs.get('position_ids')
                last_token_ids = model_inputs.get('last_token_ids')
                attention_mask = model_inputs.get('attention_mask', None)

                if self.paged_kv_cache:
                    kv_cache_block_pointers = self.kv_cache_manager.get_pointer_arrays(
                        1)

                ctx_shape, ctx_buffer = self._get_context_shape_buffer(
                    input_ids, context_lengths, host_context_lengths,
                    position_ids, last_token_ids, attention_mask,
                    this_src_cache_indirection, kv_cache_block_pointers,
                    prompt_embedding_table, tasks, prompt_vocab_size)
                context = self.runtime.ctx_context
                self.runtime._set_shape(context, ctx_shape)
                self.runtime._set_buffer(context, ctx_buffer)
                if self.debug_mode:
                    self.debug_buffer = ctx_buffer
                if self.cuda_graph_mode:
                    # context mode, clean cuda graph instances
                    self.cuda_graph_instances = [None for _ in range(2)]

            # dynamic_decoder currently use torch's current stream, so must let TRT enqueue use same stream here
            stream = torch.cuda.current_stream().cuda_stream
            instance_idx = step % 2
            if self.cuda_graph_mode and self.runtime.cuda_graph_instances[
                    instance_idx] is not None:
                # launch cuda graph
                CUASSERT(
                    cudart.cudaGraphLaunch(
                        self.runtime.cuda_graph_instances[instance_idx],
                        stream))
                ok = True
            else:
                ok = self.runtime._run(context, stream)

            if not ok:
                raise RuntimeError('Executing TRT engine failed!')
            if self.debug_mode:
                torch.cuda.synchronize()

            if step == 0 and beam_width > 1:

                if not self.use_gpt_attention_plugin:
                    attention_mask = _tile_beam_width(attention_mask,
                                                      beam_width)
                context_lengths = _tile_beam_width(context_lengths, beam_width)
                host_context_lengths = _tile_beam_width(host_context_lengths,
                                                        beam_width)
                if tasks is not None:
                    tasks = _tile_beam_width(tasks, beam_width)

                # Move tiling before logit computing of context
                for key in self.buffer.keys():
                    if "present_key_value" in key:
                        self.buffer[key] = _tile_beam_width(
                            self.buffer[key], beam_width)
                self.buffer['logits'] = _tile_beam_width(
                    self.buffer['logits'], beam_width)

            # Initialize sequence_lengths (no paddings) for the generation phase.
            if step == 0:
                self.sequence_length_buffer = context_lengths.detach().clone()

            if not step == self.max_new_tokens - 1:
                # Set shape and address for the next step
                model_inputs = self._prepare_generation_inputs(
                    batch_size=batch_size,
                    context_lengths=context_lengths,
                    use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                    remove_input_padding=self.remove_input_padding,
                    step=step,
                    num_beams=beam_width,
                    attention_mask=attention_mask,
                )

                position_ids = model_inputs.get('position_ids')
                last_token_ids = model_inputs.get('last_token_ids')
                attention_mask = model_inputs.get('attention_mask', None)

                if self.paged_kv_cache:
                    kv_cache_block_pointers = self.kv_cache_manager.get_pointer_arrays(
                        beam_width)

                next_context = self.runtime.context_1 if step % 2 else self.runtime.context_0
                next_step_shape, next_step_buffer = self._get_next_step_shape_buffer(
                    batch_size, beam_width, max_context_length, step,
                    context_lengths, host_context_lengths, position_ids,
                    last_token_ids, attention_mask, next_src_cache_indirection,
                    kv_cache_block_pointers, prompt_embedding_table, tasks,
                    prompt_vocab_size)
                self.runtime._set_shape(next_context, next_step_shape)
                self.runtime._set_buffer(next_context, next_step_buffer)
                if self.debug_mode:
                    self.debug_buffer = next_step_buffer
                if self.cuda_graph_mode:
                    # capture cuda graph
                    CUASSERT(
                        cudart.cudaStreamBeginCapture(
                            stream, cudart.cudaStreamCaptureMode.
                            cudaStreamCaptureModeGlobal))
                    next_context.execute_async_v3(stream)
                    next_graph = CUASSERT(
                        cudart.cudaStreamEndCapture(stream))[0]

                    instance_idx = (step + 1) % 2

                    if self.runtime.cuda_graph_instances[
                            instance_idx] is not None:
                        self.runtime.cuda_graph_instances[
                            instance_idx] = _update_cuda_graph_instance(
                                self.runtime.cuda_graph_instances[instance_idx],
                                next_graph)
                    else:
                        self.runtime.cuda_graph_instances[
                            instance_idx] = CUASSERT(
                                cudart.cudaGraphInstantiate(next_graph, 0))[0]

                    # Pre-upload cuda graph to stream
                    CUASSERT(
                        cudart.cudaGraphUpload(
                            self.runtime.cuda_graph_instances[instance_idx],
                            stream))

            logits = self.buffer['logits']
            if logits is not None:
                # [batch_size x beam_width, vocab_size_padded] -> [batch_size, beam_width, vocab_size_padded]
                next_token_logits = logits.reshape(
                    (batch_size, beam_width, -1)).to(self.decoder_logits_dtype)
                decode_step = step + max_context_length
                should_stop = self.dynamic_decoder.forward(
                    next_token_logits, decode_step, max_context_length, ite,
                    batch_size, self.end_ids, self.embedding_bias_opt,
                    context_lengths, sequence_limit_lengths, stop_words_list,
                    bad_words_list, no_repeat_ngram_size,
                    this_src_cache_indirection, self.output_ids,
                    self.new_tokens, self.finished, self.sequence_length_buffer,
                    self.cum_log_probs, self.log_probs, self.parent_ids,
                    this_tgt_cache_indirection, self.beam_hyps_output_ids_tgt,
                    self.beam_hyps_sequence_lengths_tgt,
                    self.beam_hyps_cum_log_probs, self.beam_hyps_normed_scores,
                    self.beam_hyps_log_probs, self.beam_hyps_min_normed_scores,
                    self.beam_hyps_num_beams, self.beam_hyps_is_done,
                    scfg.use_beam_hyps)

                if should_stop.item():
                    if self.paged_kv_cache:
                        # Free all blocks in all sequences.
                        # With in-flight batching and while loop we'll free some sequences, when they are done
                        self.kv_cache_manager.step([True] * batch_size)

                    # output shape of self.gather_tree: [batch_size, beam_width, output_len]
                    final_output_ids = self.gather_tree(
                        self.sequence_length_buffer, self.output_ids,
                        self.parent_ids, self.end_ids, context_lengths,
                        self.cum_log_probs, self.beam_hyps_output_ids_tgt,
                        self.beam_hyps_sequence_lengths_tgt,
                        self.beam_hyps_cum_log_probs,
                        self.beam_hyps_normed_scores, self.beam_hyps_log_probs,
                        self.beam_hyps_min_normed_scores,
                        self.beam_hyps_num_beams, self.beam_hyps_is_done,
                        self.finished, self.length_penalty, batch_size,
                        beam_width, self.max_seq_length, scfg.use_beam_hyps)
                    if do_return_sequence_length:
                        return final_output_ids, self.sequence_length_buffer.reshape(
                            [batch_size, beam_width])
                    else:
                        return final_output_ids

            if self.paged_kv_cache and step < self.max_new_tokens - 1:
                # Iterate to the next step in KV cache manager.
                # Increase number of tokens for all unfinished sequences.
                # And allocate new blocks if needed.
                # We set this to False for all sequences, since we use only length criterion to stop now
                self.kv_cache_manager.step([False] * batch_size)

        if self.paged_kv_cache:
            # Free all blocks in all sequences.
            # With in-flight batching and while loop we'll free some sequences, when they are done
            self.kv_cache_manager.step([True] * batch_size)

        # output shape of self.gather_tree: [batch_size, beam_width, output_len]
        final_output_ids = self.gather_tree(
            self.sequence_length_buffer, self.output_ids, self.parent_ids,
            self.end_ids, context_lengths, self.cum_log_probs,
            self.beam_hyps_output_ids_tgt, self.beam_hyps_sequence_lengths_tgt,
            self.beam_hyps_cum_log_probs, self.beam_hyps_normed_scores,
            self.beam_hyps_log_probs, self.beam_hyps_min_normed_scores,
            self.beam_hyps_num_beams, self.beam_hyps_is_done, self.finished,
            self.length_penalty, batch_size, beam_width, self.max_seq_length,
            scfg.use_beam_hyps)

        if do_return_sequence_length:
            return final_output_ids, self.sequence_length_buffer.reshape(
                [batch_size, beam_width])
        else:
            return final_output_ids


class ChatGLM6BHeadModelGenerationSession(GenerationSession):

    def _prepare_context_inputs(self, batch_size, context_lengths,
                                use_gpt_attention_plugin, remove_input_padding,
                                **kwargs):

        assert use_gpt_attention_plugin
        assert not remove_input_padding
        last_token_ids = context_lengths.detach().clone()
        max_context_length = kwargs.pop('max_context_length')
        position_ids = torch.zeros([batch_size, 2, max_context_length],
                                   dtype=torch.int32)
        position_ids[:, 0, :] = torch.arange(max_context_length)
        for i in range(batch_size):
            position_ids[i, 0, max_context_length - 1] = max_context_length - 2
            position_ids[i, 1, max_context_length - 1] = 1
            position_ids[i, :, context_lengths[i]:] = 0
        position_ids = position_ids.cuda()
        return {'position_ids': position_ids, 'last_token_ids': last_token_ids}

    def _prepare_generation_inputs(self, batch_size, context_lengths,
                                   use_gpt_attention_plugin,
                                   remove_input_padding, **kwargs):
        assert use_gpt_attention_plugin
        assert not remove_input_padding
        last_token_ids = torch.ones_like(context_lengths)

        step = kwargs.pop('step')
        num_beams = kwargs.pop('num_beams')

        data = []
        for i in range(batch_size):
            data.append([[context_lengths[i * num_beams] - 2], [step + 2]])
        position_ids = torch.tensor(data, dtype=torch.int32, device='cuda')

        return {'position_ids': position_ids, 'last_token_ids': last_token_ids}
