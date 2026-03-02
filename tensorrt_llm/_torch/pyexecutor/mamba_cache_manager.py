# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Dict, List, Optional, Union

import torch
from functools import reduce

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager, ModelConfigCpp, get_pp_layers)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import LayerType
from tensorrt_llm.bindings.internal.batch_manager import KvCacheConnectorManager, LinearAttentionMetadata, LinearCacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


class MambaCacheManager(BaseResourceManager):

    def __init__(
        self,
        d_state: int,
        d_conv: int,
        num_heads: int,
        n_groups: int,
        head_dim: int,
        num_layers: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: torch.dtype,
        ssm_cache_dtype: torch.dtype,
        layer_mask: Optional[List[bool]] = None,
    ) -> None:

        self.mamba_ssm_cache_dtype = ssm_cache_dtype

        # get tp size
        tp_size = mapping.tp_size

        # derive mamba parameters for conv and ssm states
        d_inner = head_dim * num_heads
        conv_dim = d_inner + 2 * n_groups * d_state
        nheads = num_heads

        # check that can be partitioned
        assert nheads % tp_size == 0, "nheads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"

        # partition conv_dim and nheads
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size

        # conv and ssm states device
        device = torch.device("cuda")

        pp_layers, num_layers = get_pp_layers(
            num_layers,
            mapping,
            layer_mask=layer_mask,
        )
        num_local_layers = len(pp_layers)
        self.mamba_layer_offsets = {
            idx: offset
            for offset, idx in enumerate(pp_layers)
        }

        # mamba conv states
        self.conv_states = torch.empty(
            size=[
                num_local_layers,
                max_batch_size,
                conv_dim,
                d_conv - 1,
            ],
            dtype=dtype,
            device=device,
        )

        # mamba ssm states
        self.ssm_states = torch.empty(
            size=[
                num_local_layers,
                max_batch_size,
                nheads,
                head_dim,
                d_state,
            ],
            dtype=self.mamba_ssm_cache_dtype,
            device=device,
        )

        # mamba cache available blocks
        self.mamba_cache_free_blocks = [i for i in range(max_batch_size)]

        # mamba cache index, maps request_id -> state indices
        self.mamba_cache_index: Dict[int, int] = {}

        # mamba cache state indices
        self.state_indices: torch.Tensor = torch.arange(max_batch_size,
                                                        device=device,
                                                        dtype=torch.int32)

    def _prepare_mamba_cache_blocks(self, request_ids: List[int]):
        state_indices = []
        for r in request_ids:
            # cache hit
            if r in self.mamba_cache_index:
                state_indices.append(self.mamba_cache_index[r])
            # cache miss
            else:
                if len(self.mamba_cache_free_blocks) == 0:
                    raise Exception("run out of mamba cache blocks")
                block = self.mamba_cache_free_blocks.pop()
                self.mamba_cache_index[r] = block
                state_indices.append(block)
        self.state_indices[:len(state_indices)] = torch.as_tensor(
            state_indices, dtype=torch.int32, device=self.ssm_states.device)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_ids = [
            i.py_request_id for i in scheduled_batch.context_requests
        ]
        generation_ids = [
            i.py_request_id for i in scheduled_batch.generation_requests
        ]
        request_ids = context_ids + generation_ids
        self._prepare_mamba_cache_blocks(request_ids)

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        if request_id in self.mamba_cache_index:
            block = self.mamba_cache_index.pop(request_id)
            self.mamba_cache_free_blocks.append(block)

    def get_state_indices(self) -> torch.Tensor:
        # print(f"get_state_indices: {self.state_indices}")
        return self.state_indices

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.conv_states[layer_offset]

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.ssm_states[layer_offset]

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.mamba_ssm_cache_dtype

    def shutdown(self):
        # release tensor memory, keeping python references as tensors
        self.conv_states = torch.tensor([])
        self.ssm_states = torch.tensor([])
        self.state_indices = torch.tensor([])
        torch.cuda.empty_cache()


class MambaHybridCacheManager(KVCacheManager, MambaCacheManager):

    def __init__(
        self,
        # mamba cache parameters
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_num_heads: int,
        mamba_n_groups: int,
        mamba_head_dim: int,
        mamba_num_layers: int,
        mamba_layer_mask: List[bool],
        mamba_cache_dtype: torch.dtype,
        mamba_ssm_cache_dtype: torch.dtype,

        # kv cache parameters
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        layer_mask: List[bool],
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        is_estimating_kv_cache: bool = False,
        model_config: Optional[ModelConfigCpp] = None,
    ) -> None:

        # mamba hybrid cache requires block reuse to be disabled in KV cache config
        assert not kv_cache_config.enable_block_reuse, "mamba hybrid cache requires block reuse to be disabled in KV cache config"

        # initialize mamba cache manager
        MambaCacheManager.__init__(
            self,
            mamba_d_state,
            mamba_d_conv,
            mamba_num_heads,
            mamba_n_groups,
            mamba_head_dim,
            mamba_num_layers,
            max_batch_size,
            mapping,
            mamba_cache_dtype,
            mamba_ssm_cache_dtype,
            mamba_layer_mask,
        )

        # initialize kv cache manager
        KVCacheManager.__init__(
            self,
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            is_estimating_kv_cache=is_estimating_kv_cache,
        )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        MambaCacheManager.prepare_resources(self, scheduled_batch)
        KVCacheManager.prepare_resources(self, scheduled_batch)

    def free_resources(self, request: LlmRequest):
        MambaCacheManager.free_resources(self, request)
        KVCacheManager.free_resources(self, request)

    def shutdown(self):
        MambaCacheManager.shutdown(self)
        KVCacheManager.shutdown(self)


class LinearHybridCacheManager(KVCacheManager):
    def __init__(self,
                 # mamba cache parameters
                 mamba_d_state: int,
                 mamba_d_conv: int,
                 mamba_num_heads: int,
                 mamba_n_groups: int,
                 mamba_head_dim: int,
                 mamba_num_layers: int,
                 mamba_layer_mask: List[bool],
                 mamba_cache_dtype: torch.dtype,
                 mamba_ssm_cache_dtype: torch.dtype,
                 kv_cache_config: KvCacheConfig,
                 kv_cache_type: CacheTypeCpp,
                 *,
                 num_layers: int,
                 num_kv_heads: Union[int, List[Optional[int]]],
                 head_dim: int,
                 tokens_per_block: int,
                 # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
                 # It's derived from the model's BuildConfig for consistency with the C++ backend.
                 max_seq_len: int,
                 max_batch_size: int,
                 mapping: Mapping,
                 dtype: DataType = DataType.HALF,
                 spec_config: Optional["DecodingBaseConfig"] = None,
                 layer_mask: Optional[List[bool]] = None,
                 max_num_tokens: int = 8192,
                 model_config: Optional[ModelConfigCpp] = None,
                 max_beam_width: int = 1,
                 is_draft: bool = False,
                 kv_connector_manager: Optional[KvCacheConnectorManager] = None,
                 enable_indexer_k_cache: bool = False,
                 indexer_k_cache_quant_block_size: int = 128,
                 indexer_k_cache_index_head_dim: int = 0,
                 is_estimating_kv_cache: bool = False,
                 snapshot_interval: int = 128,
                 **kwargs,
                 ) -> None:
        # Derive ssm_state_shape and conv_state_shape from mamba params (same as MambaCacheManager)
        tp_size = mapping.tp_size
        d_inner = mamba_head_dim * mamba_num_heads
        conv_dim = d_inner + 2 * mamba_n_groups * mamba_d_state
        nheads = mamba_num_heads
        assert nheads % tp_size == 0, "mamba_num_heads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size
        self.conv_state_shape = [conv_dim, mamba_d_conv - 1]
        self.ssm_state_shape = [nheads, mamba_head_dim, mamba_d_state]
        self.ssm_state_dtype = mamba_ssm_cache_dtype
        self.conv_state_dtype = mamba_cache_dtype
        self.ssm_count = reduce(lambda x, y: x * y, self.ssm_state_shape)
        self.conv_count = reduce(lambda x, y: x * y, self.conv_state_shape)
        self.ssm_bytes = self.ssm_count * self.ssm_state_dtype.itemsize
        self.conv_bytes = self.conv_count * self.conv_state_dtype.itemsize
        self.linear_attention_metadata = LinearAttentionMetadata()
        # TODO(xiweny): is this needed?
        # self.linear_attention_metadata.linear_layer_indices = [0, 1]
        self.linear_attention_metadata.cache_type = LinearCacheType.RECURRENT_STATES.value
        self.linear_attention_metadata.all_recurrent_states_bytes = self.ssm_bytes + self.conv_bytes
        self.linear_attention_metadata.input_features_bytes_per_token = 0
        self.linear_attention_metadata.states_snapshot_interval = snapshot_interval

        if kv_cache_config.enable_partial_reuse:
            logger.warning(
                "Partial reuse is not supported for linear hybrid cache, disabling partial reuse")
            kv_cache_config.enable_partial_reuse = False
        kv_cache_config.max_attention_window = []
        for i in range(mamba_num_layers + num_layers):
            kv_cache_config.max_attention_window.append(
                LinearCacheType.RECURRENT_STATES.value if mamba_layer_mask[i] else max_seq_len)
        # pass remaining arguments to super class
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=mamba_num_layers + num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            # layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            is_draft=is_draft,
            kv_connector_manager=kv_connector_manager,
            enable_indexer_k_cache=enable_indexer_k_cache,
            indexer_k_cache_quant_block_size=indexer_k_cache_quant_block_size,
            indexer_k_cache_index_head_dim=indexer_k_cache_index_head_dim,
            is_estimating_kv_cache=is_estimating_kv_cache,
            linear_attention_metadata=self.linear_attention_metadata,
        )
        self.linear_pp_layers, _ = get_pp_layers(
            mamba_num_layers,
            mapping,
            layer_mask=mamba_layer_mask,
        )
        idx = 0
        self.linear_layer_offsets = {}
        for layer_id in self.linear_pp_layers:
            self.linear_layer_offsets[layer_id] = idx
            idx += 1
        self.num_linear_layers = mamba_num_layers
        self.host_block_offsets = torch.zeros([
            self.impl.num_pools, self.max_batch_size, 2,
            self.max_blocks_per_seq
        ], dtype=torch.int32, device="cpu")
        self.requests = []
        self.recurrent_states_pool_index = self.kv_cache_pool_mapping[self.linear_pp_layers[0]][0]
        for layer_id in self.linear_pp_layers:
            assert self.kv_cache_pool_mapping[layer_id][0] == self.recurrent_states_pool_index, f"All linear layers should be in the same pool, but layer_id: {layer_id} is in pool {self.kv_cache_pool_mapping[layer_id][0]} while the recurrent states pool is {self.recurrent_states_pool_index}"
        self._cuda_state_indices = torch.zeros([self.max_batch_size], dtype=torch.int32, device="cuda")

    def add_dummy_requests(
        self,
        request_ids: List[int],
        # Note that token_nums should be past_kv_len + input_len (without
        # spec decoding). The draft tokens will be added in this function,
        # so we don't need to take care of it in the caller. When preparing
        # token_nums, we should not take the draft tokens into account, so
        # don't use the kv_cache_manager.max_seq_len, which includes both
        # extra tokens and draft tokens.
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        # For capturable drafting loops. During normal inference, the draft model always
        # has enough KV cache space to fit all of our draft tokens. During warmup, however,
        # we need to make the KV cache manager aware that multiple autoregressive steps will
        # occur.
        num_extra_decoding_steps: int = 0,
    ) -> List[LlmRequest]:
        # print(f"add_dummy_requests for request_ids {request_ids}")
        requests = super().add_dummy_requests(request_ids, token_nums, is_gen, prepare_resource,
                                              max_num_draft_tokens, use_mrope, max_beam_width, num_extra_decoding_steps)
        self.requests.extend(requests)
        return requests

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        # print(
        #     f"prepare_resources with {len(scheduled_batch.context_requests)} context requests and {len(scheduled_batch.generation_requests)} generation requests")
        self.requests = scheduled_batch.context_requests + \
            scheduled_batch.generation_requests
        super().prepare_resources(scheduled_batch)
        self._setup_state_indices()

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        # print(f"free_resources for request {request.py_request_id}")
        if request in self.requests:
            self.requests.remove(request)
        super().free_resources(request, pin_on_release)

    # TODO: this should be called only once per iteration (not per layer)
    def _setup_state_indices(self) -> torch.Tensor:
        # return torch.tensor([req.py_request_id for req in self.requests], dtype=torch.int32, device="cuda")
        block_indices = []
        for req in self.requests:
            next_step = req.get_num_tokens(0) if req.is_context_finished else (req.context_current_position -
                                                                               1 + req.context_chunk_size)
            # print(f"next_step for request {req.py_request_id}: {next_step}")
            block_indices.append(next_step // self.tokens_per_block)
            block_ids = self.get_cache_indices(
                req, LinearCacheType.RECURRENT_STATES.value)
            # print(f"block_ids for request {req.py_request_id}: {block_ids}")
        self.impl.copy_batch_block_offsets(
            self.host_block_offsets, [req.py_request_id for req in self.requests], 1, 0)
        host_linear_block_offsets = torch.zeros([len(self.requests)], dtype=torch.int32, device="cpu")
        for i in range(len(self.requests)):
            value = self.host_block_offsets[self.recurrent_states_pool_index, i, 0, block_indices[i]]
            assert value % self.num_linear_layers == 0 and value >= 0 and value < self.blocks_per_window[LinearCacheType.RECURRENT_STATES.value][0] * self.num_linear_layers, \
                f"value: {value} at index {i}is not in the range of [0, {self.blocks_per_window[LinearCacheType.RECURRENT_STATES.value][0] * self.num_linear_layers}).\nself.host_linear_block_offsets[self.recurrent_states_pool_index, :, 0, 0]: {self.host_block_offsets[self.recurrent_states_pool_index, :, 0, 0]}"
            host_linear_block_offsets[i] = value // self.num_linear_layers
        # print(f"block_indices: {block_indices}")
        # print(f"self.host_linear_block_offsets: {self.host_linear_block_offsets[0, :len(block_indices), 0, :12]}")
        # print(f"host_linear_block_offsets: {host_linear_block_offsets}")
        self._cuda_state_indices[:len(self.requests)] = host_linear_block_offsets.cuda()

    def get_state_indices(self) -> torch.Tensor:
        return self._cuda_state_indices

    # [total_block_num, *ssm_state_shape] (one block for one layer)
    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        # return self.temp_ssm_states[layer_idx]
        # [total_block_num, 1, ssm_bytes + conv_bytes]
        pool = self.impl.get_recurrent_states_pool().view([-1, self.ssm_bytes + self.conv_bytes])
        # print(f"layer_idx: {layer_idx}, pool: {hex(pool.data_ptr())}, shape: {pool.shape}, dtype: {pool.dtype}")
        layer_idx = self.linear_layer_offsets[layer_idx]
        # print(f"shape of pool: {pool.shape}, dtype: {pool.dtype}")
        offset = (self.ssm_bytes + self.conv_bytes) // self.ssm_state_dtype.itemsize * layer_idx

        flat = pool.view(self.ssm_state_dtype)
        assert flat.data_ptr() == pool.data_ptr()
        target_shape = [pool.shape[0] // self.num_linear_layers, *self.ssm_state_shape]
        target_strides = [
            flat.stride(0) * self.num_linear_layers,
            self.ssm_state_shape[1] * self.ssm_state_shape[2],
            self.ssm_state_shape[2],
            1,
        ]
        my_ssm_states = torch.as_strided(
            flat, target_shape, target_strides,
            storage_offset=offset)
        # print(
        #     f"my_ssm_states: {hex(my_ssm_states.data_ptr())}, {my_ssm_states.shape}, is_view: {my_ssm_states._is_view()}")
        # print(f"layer_idx: {layer_idx}, linear_layer_offsets[layer_idx]: {self.linear_layer_offsets[layer_idx]}")
        # assert not my_ssm_states.is_contiguous()
        return my_ssm_states

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        # return self.temp_conv_states[layer_idx]

        # [total_block_num, num_linear_layers, ssm_bytes + conv_bytes] -> [total_block_num * num_linear_layers, ssm_bytes + conv_bytes]
        pool = self.impl.get_recurrent_states_pool().view([-1, self.ssm_bytes + self.conv_bytes])
        # print(f"layer_idx: {layer_idx}, pool: {hex(pool.data_ptr())}, shape: {pool.shape}, dtype: {pool.dtype}")
        layer_idx = self.linear_layer_offsets[layer_idx]
        # print(f"shape of pool: {pool.shape}, dtype: {pool.dtype}")
        offset = self.ssm_bytes // self.conv_state_dtype.itemsize + \
            (self.ssm_bytes + self.conv_bytes) // self.conv_state_dtype.itemsize * layer_idx
        flat = pool.view(self.conv_state_dtype)
        # flat should be a view of pool
        assert flat.data_ptr() == pool.data_ptr()
        target_shape = [pool.shape[0] // self.num_linear_layers, *self.conv_state_shape]
        target_strides = [flat.stride(0) * self.num_linear_layers , self.conv_state_shape[-1], 1]
        my_conv_states = torch.as_strided(
            flat, target_shape, target_strides,
            storage_offset=offset)
        # print(f"layer_idx: {layer_idx}, linear_layer_offsets[layer_idx]: {self.linear_layer_offsets[layer_idx]}")
        # print(
        #     f"my_conv_states: {hex(my_conv_states.data_ptr())}, {my_conv_states.shape}, {my_conv_states.stride()}")
        # assert not my_conv_states.is_contiguous()
        return my_conv_states

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.ssm_state_dtype
