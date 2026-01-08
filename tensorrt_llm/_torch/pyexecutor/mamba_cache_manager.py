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

from typing import List, Optional, Union

import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

RnnStateManagerCpp = tensorrt_llm.bindings.internal.batch_manager.RnnStateManager
WorldConfig = tensorrt_llm.bindings.WorldConfig


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
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:

        self.mamba_ssm_cache_dtype = ssm_cache_dtype

        world_config = WorldConfig(
            tensor_parallelism=mapping.tp_size,
            pipeline_parallelism=mapping.pp_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
        )

        dtype_binding = torch_dtype_to_binding(dtype)
        ssm_cache_dtype_binding = torch_dtype_to_binding(ssm_cache_dtype)

        self._stream = stream if stream is not None else torch.cuda.current_stream(
        )

        self.mamba_impl = RnnStateManagerCpp(
            d_state=d_state,
            d_conv=d_conv,
            num_heads=num_heads,
            n_groups=n_groups,
            head_dim=head_dim,
            num_layers=num_layers,
            max_batch_size=max_batch_size,
            world_config=world_config,
            stream=self._stream.cuda_stream,
            dtype=dtype_binding,
            ssm_cache_dtype=ssm_cache_dtype_binding,
            layer_mask=layer_mask,
        )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_ids = [
            i.py_request_id for i in scheduled_batch.context_requests
        ]
        generation_ids = [
            i.py_request_id for i in scheduled_batch.generation_requests
        ]
        request_ids = context_ids + generation_ids
        self.mamba_impl.allocate_cache_blocks(request_ids)

    def free_resources(self, request: LlmRequest):
        self.mamba_impl.free_cache_block(request.py_request_id)

    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        self.mamba_impl.allocate_cache_blocks(request_ids)

    def get_cache_index(self, request_id: int) -> int:
        return self.mamba_impl.get_cache_index(request_id)

    def get_state_indices(self, request_ids: List[int],
                          is_padding: List[bool]) -> List[int]:
        return self.mamba_impl.get_state_indices(request_ids, is_padding)

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        return self.mamba_impl.get_conv_states(layer_idx)

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        return self.mamba_impl.get_ssm_states(layer_idx)

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.mamba_ssm_cache_dtype

    def shutdown(self):
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
        execution_stream: Optional[torch.cuda.Stream] = None,
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
            execution_stream,
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
            execution_stream=execution_stream,
        )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        MambaCacheManager.prepare_resources(self, scheduled_batch)
        KVCacheManager.prepare_resources(self, scheduled_batch)

    def free_resources(self, request: LlmRequest):
        MambaCacheManager.free_resources(self, request)
        KVCacheManager.free_resources(self, request)

    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        MambaCacheManager.add_dummy_requests(self, request_ids)
        return KVCacheManager.add_dummy_requests(self, request_ids, **kwargs)

    def shutdown(self):
        MambaCacheManager.shutdown(self)
        KVCacheManager.shutdown(self)
