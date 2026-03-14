# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import atexit
import os
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

import tensorrt_llm.bindings

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager, ModelConfigCpp,
    get_pp_layers)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import prefer_pinned, torch_dtype_to_binding, mpi_rank
from tensorrt_llm.bindings.internal.batch_manager import (
    KvCacheConnectorManager, LinearAttentionMetadata, LinearCacheType)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

RnnStateManagerCpp = tensorrt_llm.bindings.internal.batch_manager.RnnStateManager
WorldConfig = tensorrt_llm.bindings.WorldConfig

GB = 1 << 30


def get_tensor_size_bytes(tensor):
    """Calculate tensor size in bytes."""
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement()
    elif isinstance(tensor, list):
        return sum(get_tensor_size_bytes(t) for t in tensor)
    return 0


def use_cpp_mamba_cache_manager() -> bool:
    """Check if C++ MambaCacheManager should be used.

    Returns True if TRTLLM_USE_CPP_MAMBA='1' is set, False otherwise.
    By default, PythonMambaCacheManager is used.
    """
    return os.environ.get('TRTLLM_USE_CPP_MAMBA', '0') == '1'


class CppMambaCacheManager(BaseResourceManager):
    """C++ backed Mamba cache manager using RnnStateManager bindings."""

    def __init__(
        self,
        d_state: int,
        d_conv: int,
        num_heads: int,
        n_groups: int,
        head_dim: int,
        num_layers: int,
        max_num_sequences: int,
        mapping: Mapping,
        dtype: torch.dtype,
        ssm_cache_dtype: torch.dtype,
        layer_mask: Optional[List[bool]] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.mamba_ssm_cache_dtype = ssm_cache_dtype

        # get tp size
        tp_size = mapping.tp_size if not mapping.enable_attention_dp else 1
        world_config = WorldConfig(
            tensor_parallelism=tp_size,
            pipeline_parallelism=mapping.pp_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
        )

        dtype_binding = torch_dtype_to_binding(dtype)
        ssm_cache_dtype_binding = torch_dtype_to_binding(
            ssm_cache_dtype if ssm_cache_dtype is not None else dtype)

        self._stream = stream if stream is not None else torch.cuda.current_stream(
        )

        pp_layers, _ = get_pp_layers(num_layers, mapping, layer_mask=layer_mask)

        self.mamba_impl = RnnStateManagerCpp(
            d_state=d_state,
            d_conv=d_conv,
            num_heads=num_heads,
            n_groups=n_groups,
            head_dim=head_dim,
            max_batch_size=max_num_sequences,
            world_config=world_config,
            stream=self._stream.cuda_stream,
            dtype=dtype_binding,
            ssm_cache_dtype=ssm_cache_dtype_binding,
            pp_layers=pp_layers,
            num_layers=num_layers,
        )
        self._max_num_sequences = max_num_sequences

    def get_max_resource_count(self) -> int:
        # Return the maximum number of sequences that can be cached.
        return self._max_num_sequences

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        # For Mamba cache manager, we always need one slot per request.
        return 1

    def is_speculative(self) -> bool:
        # C++ MambaCacheManager does not support speculative decoding
        return False

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
        # For CUDA graph dummy requests, the blocks will be allocated
        # when get_state_indices is called.
        from .cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
        request_ids = [
            rid for rid in request_ids if rid != CUDA_GRAPH_DUMMY_REQUEST_ID
        ]
        if request_ids:
            self.mamba_impl.allocate_cache_blocks(request_ids)

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


class PythonMambaCacheManager(BaseResourceManager):

    @dataclass(frozen=True, kw_only=True)
    class State:
        """Base state container for Mamba cache."""
        conv: torch.Tensor
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            for k, v in vars(self).items():
                kwargs[k] = v[layer]
            return type(self)(**kwargs)

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        """Speculative state with intermediate states for draft tokens."""
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: torch.Tensor

    def __init__(
        self,
        d_state: int,
        d_conv: int,
        num_heads: int,
        n_groups: int,
        head_dim: int,
        num_layers: int,
        max_batch_size: int,
        spec_state_size: int,
        mapping: Mapping,
        dtype: torch.dtype,
        ssm_cache_dtype: torch.dtype,
        layer_mask: Optional[List[bool]] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ) -> None:

        self.mamba_ssm_cache_dtype = ssm_cache_dtype
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.spec_state_size = spec_state_size

        # get tp size
        tp_size = mapping.tp_size if not mapping.enable_attention_dp else 1

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

        conv_state_shape = (conv_dim, d_conv - 1)
        ssm_state_shape = (nheads, head_dim, d_state)

        # create mamba conv and ssm states
        conv_states = torch.empty(
            size=(num_local_layers, max_batch_size) + conv_state_shape,
            dtype=dtype,
            device=device,
        )

        ssm_states = torch.empty(
            size=(num_local_layers, max_batch_size) + ssm_state_shape,
            dtype=self.mamba_ssm_cache_dtype,
            device=device,
        )

        # create state container
        if speculative_num_draft_tokens is not None:

            # Cache intermediate SSM states per draft token(include new sampled token) during target model verification phase
            intermediate_ssm_states = torch.zeros(
                size=(num_local_layers, self.spec_state_size,
                      speculative_num_draft_tokens + 1) + ssm_state_shape,
                dtype=self.mamba_ssm_cache_dtype,
                device=device,
            )

            # Cache intermediate conv windows per draft token(include new sampled token) during target model verification phase
            intermediate_conv_window_cache = torch.zeros(
                size=(num_local_layers, self.spec_state_size,
                      speculative_num_draft_tokens + 1) + conv_state_shape,
                dtype=dtype,
                device=device,
            )

            self.mamba_cache = self.SpeculativeState(
                conv=conv_states,
                temporal=ssm_states,
                intermediate_ssm=intermediate_ssm_states,
                intermediate_conv_window=intermediate_conv_window_cache,
            )

            logger.info(
                f"Mamba Cache is allocated. "
                f"max_mamba_cache_size: {max_batch_size}, "
                f"conv_state size: {get_tensor_size_bytes(conv_states) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(ssm_states) / GB:.2f}GB, "
                f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_states) / GB:.2f}GB, "
                f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB"
            )
        else:
            self.mamba_cache = self.State(
                conv=conv_states,
                temporal=ssm_states,
            )

            logger.info(
                f"Mamba Cache is allocated. "
                f"max_mamba_cache_size: {max_batch_size}, "
                f"conv_state size: {get_tensor_size_bytes(conv_states) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(ssm_states) / GB:.2f}GB"
            )

        # mamba cache available blocks
        self.mamba_cache_free_blocks = [i for i in range(max_batch_size)]

        # mamba cache index, maps request_id -> state indices
        self.mamba_cache_index: Dict[int, int] = {}

        # mamba cache state indices
        self.state_indices: torch.Tensor = torch.arange(max_batch_size,
                                                        device=device,
                                                        dtype=torch.int32)
        # save mamba state indices for requests
        self.state_indices_list: List[int] = []
        # save intermediate state indices for requests
        self.intermediate_state_indices = torch.arange(max_batch_size,
                                                       dtype=torch.int32,
                                                       device=device)

        # Store max_batch_size for resource management
        self._max_batch_size = max_batch_size

    def get_max_resource_count(self) -> int:
        """Return the maximum number of sequences that can be cached."""
        return self._max_batch_size

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        """For Mamba cache manager, we always need one slot per request."""
        return 1

    @torch.inference_mode()
    def _prepare_mamba_cache_blocks(self, request_ids: List[int]):
        self.state_indices_list.clear()
        for r in request_ids:
            # cache hit
            if r in self.mamba_cache_index:
                self.state_indices_list.append(self.mamba_cache_index[r])
            # cache miss
            else:
                if len(self.mamba_cache_free_blocks) == 0:
                    raise Exception("run out of mamba cache blocks")
                block = self.mamba_cache_free_blocks.pop()
                self.mamba_cache_index[r] = block
                self.state_indices_list.append(block)
        self.state_indices[:len(self.state_indices_list)].copy_(
            torch.tensor(self.state_indices_list,
                         dtype=torch.int32,
                         pin_memory=prefer_pinned()),
            non_blocking=True)

    # When there exists padded requests, the state indices should not be repeated.
    def reorder_state_indices_when_padding_requests(self, request_size,
                                                    padding_size):
        if padding_size == 0:
            return

        assert request_size + padding_size <= self.state_indices.numel(
        ), "Padding requests run out of available mamba cache blocks"
        # we can use mamba_cache_free_blocks for padding_requests
        if padding_size <= len(self.mamba_cache_free_blocks):
            self.state_indices[request_size:request_size +
                               padding_size] = torch.tensor(
                                   self.mamba_cache_free_blocks[:padding_size],
                                   dtype=self.state_indices.dtype,
                                   pin_memory=prefer_pinned()).to(
                                       self.state_indices.device,
                                       non_blocking=True)
        # But just finished requests won't free their used resources immediately
        # In explicit, the running order is self.scheduler.schedule_request, self._forward_step() and self._process_previous_batch() in the PyExecutor.
        # In this way, the current forward step will remove finished requests but will not remove mamba_cache immediately.
        else:
            all_mamba_cache_indices = set(range(self.state_indices.numel()))
            allocated_indices = set(self.state_indices_list)
            free_indices = list(all_mamba_cache_indices - allocated_indices)
            self.state_indices[request_size:request_size +
                               padding_size] = torch.tensor(
                                   free_indices[:padding_size],
                                   dtype=self.state_indices.dtype,
                                   pin_memory=prefer_pinned()).to(
                                       self.state_indices.device,
                                       non_blocking=True)

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

    def get_state_indices(self,
                          request_ids: List[int] = None,
                          is_padding: List[bool] = None) -> torch.Tensor:
        return self.state_indices

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).conv

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).temporal

    def get_intermediate_ssm_states(self,
                                    layer_idx: int) -> Optional[torch.Tensor]:
        if not isinstance(self.mamba_cache, self.SpeculativeState):
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).intermediate_ssm

    def get_intermediate_conv_states(self,
                                     layer_idx: int) -> Optional[torch.Tensor]:
        """Get intermediate conv states for speculative decoding."""
        if not isinstance(self.mamba_cache, self.SpeculativeState):
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(
            layer_offset).intermediate_conv_window

    def is_speculative(self) -> bool:
        return isinstance(self.mamba_cache, self.SpeculativeState)

    def mamba_layer_cache(self,
                          layer_idx: int) -> Union[State, SpeculativeState]:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset)

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.mamba_ssm_cache_dtype

    def shutdown(self):
        """Release tensor memory."""
        # Clear state indices
        self.state_indices = torch.tensor([])

        # Clear mamba cache states
        if isinstance(self.mamba_cache, self.SpeculativeState):
            self.mamba_cache = self.SpeculativeState(
                conv=torch.tensor([]),
                temporal=torch.tensor([]),
                intermediate_ssm=torch.tensor([]),
                intermediate_conv_window=torch.tensor([]),
            )
        else:
            self.mamba_cache = self.State(
                conv=torch.tensor([]),
                temporal=torch.tensor([]),
            )

        torch.cuda.empty_cache()

    @torch.compile(options={"max-autotune": True})
    def update_mamba_states(self, attn_metadata: "AttentionMetadata",
                            num_accepted_tokens: torch.Tensor):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = num_accepted_tokens[
            num_contexts:num_contexts + num_gens] - 1
        state_indices_d = self.state_indices[num_contexts:num_contexts +
                                             num_gens]

        conv_states = self.mamba_cache.conv
        ssm_states = self.mamba_cache.temporal

        intermediate_state_cache = self.mamba_cache.intermediate_ssm
        intermediate_conv_window_cache = self.mamba_cache.intermediate_conv_window

        src_state_indices = self.intermediate_state_indices[:num_gens]

        accepted_ssm_state = intermediate_state_cache[:, src_state_indices,
                                                      num_accepted_draft_tokens]
        ssm_states[:, state_indices_d, :] = accepted_ssm_state

        accepted_conv_state = intermediate_conv_window_cache[:,
                                                             src_state_indices,
                                                             num_accepted_draft_tokens]
        conv_states[:, state_indices_d, :] = accepted_conv_state


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
        spec_state_size: int,
        mapping: Mapping,
        dtype: torch.dtype,
        ssm_cache_dtype: torch.dtype,
        layer_mask: Optional[List[bool]] = None,
        stream: Optional[torch.cuda.Stream] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ) -> None:
        max_num_sequences = max_batch_size * mapping.pp_size
        self._use_cpp = use_cpp_mamba_cache_manager()

        if self._use_cpp:
            assert speculative_num_draft_tokens is None, \
                "speculative_num_draft_tokens is not supported in CppMambaCacheManager"
            self._impl = CppMambaCacheManager(
                d_state=d_state,
                d_conv=d_conv,
                num_heads=num_heads,
                n_groups=n_groups,
                head_dim=head_dim,
                num_layers=num_layers,
                max_num_sequences=max_num_sequences,
                mapping=mapping,
                dtype=dtype,
                ssm_cache_dtype=ssm_cache_dtype,
                layer_mask=layer_mask,
                stream=stream,
            )
        else:
            self._impl = PythonMambaCacheManager(
                d_state=d_state,
                d_conv=d_conv,
                num_heads=num_heads,
                n_groups=n_groups,
                head_dim=head_dim,
                num_layers=num_layers,
                max_batch_size=max_num_sequences,
                spec_state_size=spec_state_size,
                mapping=mapping,
                dtype=dtype,
                ssm_cache_dtype=ssm_cache_dtype,
                layer_mask=layer_mask,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            )

    def get_max_resource_count(self) -> int:
        return self._impl.get_max_resource_count()

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return self._impl.get_needed_resource_to_completion(request)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        self._impl.prepare_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        self._impl.free_resources(request)

    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        if self._use_cpp:
            self._impl.add_dummy_requests(request_ids, **kwargs)

    def get_state_indices(
        self,
        request_ids: Optional[List[int]] = None,
        is_padding: Optional[List[bool]] = None
    ) -> Union[torch.Tensor, List[int]]:
        return self._impl.get_state_indices(request_ids, is_padding)

    def reorder_state_indices_when_padding_requests(self, request_size: int,
                                                    padding_size: int):
        assert not self._use_cpp, "reorder_state_indices_when_padding_requests is not supported in CppMambaCacheManager"
        self._impl.reorder_state_indices_when_padding_requests(
            request_size, padding_size)

    @property
    def mamba_cache_free_blocks(self) -> List[int]:
        assert not self._use_cpp, "mamba_cache_free_blocks is not supported in CppMambaCacheManager"
        return self._impl.mamba_cache_free_blocks

    @property
    def mamba_cache_index(self) -> Dict[int, int]:
        assert not self._use_cpp, "mamba_cache_index is not supported in CppMambaCacheManager"
        return self._impl.mamba_cache_index

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        return self._impl.get_conv_states(layer_idx)

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        return self._impl.get_ssm_states(layer_idx)

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self._impl.get_mamba_ssm_cache_dtype()

    def get_intermediate_ssm_states(self,
                                    layer_idx: int) -> Optional[torch.Tensor]:
        assert not self._use_cpp, "get_intermediate_ssm_states is not supported in CppMambaCacheManager"
        return self._impl.get_intermediate_ssm_states(layer_idx)

    def get_intermediate_conv_states(self,
                                     layer_idx: int) -> Optional[torch.Tensor]:
        assert not self._use_cpp, "get_intermediate_conv_states is not supported in CppMambaCacheManager"
        return self._impl.get_intermediate_conv_states(layer_idx)

    def is_speculative(self) -> bool:
        return self._impl.is_speculative()

    def mamba_layer_cache(
        self, layer_idx: int
    ) -> Union[PythonMambaCacheManager.State,
               PythonMambaCacheManager.SpeculativeState, None]:
        assert not self._use_cpp, "mamba_layer_cache is not supported in CppMambaCacheManager"
        return self._impl.mamba_layer_cache(layer_idx)

    def shutdown(self):
        self._impl.shutdown()

    def update_mamba_states(self, attn_metadata: "AttentionMetadata",
                            num_accepted_tokens: torch.Tensor):
        assert not self._use_cpp, "update_mamba_states is not supported in CppMambaCacheManager"
        self._impl.update_mamba_states(attn_metadata, num_accepted_tokens)


class MambaHybridCacheManagerV1(KVCacheManager, MambaCacheManager):

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
            max_batch_size,
            mapping,
            mamba_cache_dtype,
            mamba_ssm_cache_dtype,
            mamba_layer_mask,
            execution_stream,
            speculative_num_draft_tokens=spec_config.max_draft_len
            if spec_config is not None else None,
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

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        KVCacheManager.update_resources(self, scheduled_batch, attn_metadata,
                                        kv_cache_dtype_byte_size)

    def update_mamba_states(self, attn_metadata: "AttentionMetadata",
                            num_accepted_tokens: torch.Tensor):
        MambaCacheManager.update_mamba_states(self, attn_metadata,
                                              num_accepted_tokens)


def calc_context_stop_positions(prompt_len: int, tokens_per_block: int, mamba_prefix_cache_step: int, save_last_snapshot: bool = False) -> list[int]:
    stop_positions = range(0, prompt_len, mamba_prefix_cache_step)
    stop_positions = list(stop_positions)
    last_ckpt = prompt_len // tokens_per_block * tokens_per_block
    if save_last_snapshot and (last_ckpt not in stop_positions):
        stop_positions.append(last_ckpt)
    if prompt_len not in stop_positions:
        stop_positions.append(prompt_len)
    return stop_positions

    

class LinearHybridCacheManager(KVCacheManager):

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

        self.use_fake_pool = os.getenv("USE_FAKE_POOL", "0") == "1"

        print(f"conv_state_shape: {self.conv_state_shape}, ssm_state_shape: {self.ssm_state_shape}, conv_bytes: {self.conv_bytes}, ssm_bytes: {self.ssm_bytes}")
        self.linear_attention_metadata = LinearAttentionMetadata()
        # TODO(xiweny): confirm if this is needed
        # self.linear_attention_metadata.linear_layer_indices = [0, 1]
        self.linear_attention_metadata.cache_type = LinearCacheType.RECURRENT_STATES.value
        self.linear_attention_metadata.all_recurrent_states_bytes = 1 if self.use_fake_pool else (self.ssm_bytes + self.conv_bytes)
        self.linear_attention_metadata.input_features_bytes_per_token = 0
        self.linear_attention_metadata.states_snapshot_interval = kv_cache_config.mamba_prefix_cache_step
        # self.linear_attention_metadata.save_last_snapshot = True

        if kv_cache_config.enable_partial_reuse:
            logger.warning(
                "Partial reuse is not supported for linear hybrid cache, disabling partial reuse"
            )
            kv_cache_config.enable_partial_reuse = False
        kv_cache_config.max_attention_window = []
        for i in range(mamba_num_layers + num_layers):
            kv_cache_config.max_attention_window.append(
                LinearCacheType.RECURRENT_STATES.
                value if mamba_layer_mask[i] else max_seq_len)
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
            self.impl.num_pools, self.max_batch_size, 2, self.max_blocks_per_seq
        ],
                                              dtype=torch.int32,
                                              device="cpu")
        self.requests = []
        self.recurrent_states_pool_index = self.kv_cache_pool_mapping[
            self.linear_pp_layers[0]][0]
        for layer_id in self.linear_pp_layers:
            assert self.kv_cache_pool_mapping[layer_id][
                0] == self.recurrent_states_pool_index, f"All linear layers should be in the same pool, but layer_id: {layer_id} is in pool {self.kv_cache_pool_mapping[layer_id][0]} while the recurrent states pool is {self.recurrent_states_pool_index}"
        self._cuda_state_indices = torch.zeros([self.max_batch_size],
                                               dtype=torch.int32,
                                               device="cuda")
        self.kv_cache_config = kv_cache_config
        if self.use_fake_pool:
            self.fake_state_indices = torch.arange(self.max_batch_size, dtype=torch.int32, device="cuda")
            block_num = 128
            self.fake_ssm_states = torch.empty([self.num_linear_layers, block_num, *self.ssm_state_shape], dtype=self.ssm_state_dtype, device="cuda")
            self.fake_conv_states = torch.empty([self.num_linear_layers, block_num, *self.conv_state_shape], dtype=self.conv_state_dtype, device="cuda")

        pool = self.impl.get_recurrent_states_pool()
        print(f"address range of linear pool: {hex(pool.data_ptr())} to {hex(pool.data_ptr() + pool.numel() * pool.itemsize)}")

        self._request_block_ids = {}
        self._previous_ssm_states = {}
        # req_id -> (reason, prev_block_ids, block_ids, current_position); only first error per request.
        self._block_id_check_failures: Dict[int, tuple[str, List[int], List[int], int]] = {}
        atexit.register(self._report_block_id_check_failures)

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
        draft_kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> List[LlmRequest]:
        # print(f"add_dummy_requests for request_ids {request_ids}")
        requests = super().add_dummy_requests(request_ids, token_nums, is_gen,
                                              prepare_resource,
                                              max_num_draft_tokens, use_mrope,
                                              max_beam_width,
                                              num_extra_decoding_steps,
                                              draft_kv_cache_manager)
        self.requests.extend(requests)
        if self.use_fake_pool:
            self._setup_fake_states()
        else:
            self._setup_state_indices()
        return requests


    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        # print(
        #     f"prepare_resources with {len(scheduled_batch.context_requests)} context requests and {len(scheduled_batch.generation_requests)} generation requests")
        self.requests = scheduled_batch.context_requests + \
            scheduled_batch.generation_requests
        super().prepare_resources(scheduled_batch)
        if self.kv_cache_config.enable_block_reuse:
            for req in scheduled_batch.context_requests:
                req.context_chunk_size = self.calc_next_context_chunk_size(req)
        for req in self.requests:
            # if req.is_context_finished:
            #     print(f"request {req.py_request_id}: num_tokens={self.get_num_tokens(req)}, prompt_len={req.prompt_len}")
            # else:
            #     print(f"request {req.py_request_id}: num_tokens={self.get_num_tokens(req)}, prompt_len={req.prompt_len}, context_current_position={req.context_current_position}, context_chunk_size={req.context_chunk_size}")
            self.impl.copy_linear_attention_block(req)

                
            # self._check_block_ids(req)
        self.impl.refresh_blocks()
        # ssm_states = self.get_ssm_states(0)
        # for ctxreq in scheduled_batch.context_requests:
        #     block_ids = self.get_cache_indices(ctxreq, LinearCacheType.RECURRENT_STATES.value)
        #     curr_pos = ctxreq.context_current_position - 1
        #     if curr_pos < 0:
        #         print(f"new context request {ctxreq.py_request_id}, prompt_len={ctxreq.prompt_len}, block_ids={block_ids}")
        #         continue
        #     next_pos = curr_pos + ctxreq.context_chunk_size
        #     curr_block_id = block_ids[curr_pos // self.tokens_per_block]
        #     next_block_id = block_ids[next_pos // self.tokens_per_block]
        #     curr_ssm_states = ssm_states[curr_block_id].clone()
        #     next_ssm_states = ssm_states[next_block_id].clone()
        #     if not torch.equal(curr_ssm_states, next_ssm_states):
        #         print(f"fail to copy states for request {ctxreq.py_request_id}, should have copied from {curr_block_id} to {next_block_id}. curr_pos={curr_pos}, next_pos={next_pos}, block_ids={block_ids}")
            
        if self.use_fake_pool:
            self._setup_fake_states()
        else:
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
            if req.is_context_finished:
                next_step = self.get_num_tokens(req) - 1
            elif self.kv_cache_config.enable_block_reuse:
                next_step = (req.context_current_position - 1 + req.context_chunk_size)
            else:
                next_step = req.prompt_len - 1
            block_indices.append(next_step // self.tokens_per_block)
            # print(f"request {req.py_request_id}, next_step={next_step}, block_index={next_step // self.tokens_per_block} block_ids: {self.get_cache_indices(req, LinearCacheType.RECURRENT_STATES.value)}")
        self.impl.copy_batch_block_offsets(
            self.host_block_offsets,
            [req.py_request_id for req in self.requests], 1, 0)
        host_linear_block_offsets = torch.zeros([len(self.requests)],
                                                dtype=torch.int32,
                                                device="cpu")
        for i in range(len(self.requests)):
            value = self.host_block_offsets[self.recurrent_states_pool_index, i,
                                            0, block_indices[i]]
            assert value % self.num_linear_layers == 0 and value >= 0 and value < self.blocks_per_window[LinearCacheType.RECURRENT_STATES.value][0] * self.num_linear_layers, \
                f"value: {value} at index {i}is not in the range of [0, {self.blocks_per_window[LinearCacheType.RECURRENT_STATES.value][0] * self.num_linear_layers}).\nself.host_linear_block_offsets[self.recurrent_states_pool_index, :, 0, 0]: {self.host_block_offsets[self.recurrent_states_pool_index, :, 0, 0]}"
            host_linear_block_offsets[i] = value // self.num_linear_layers
        # print(f"block_indices: {block_indices}")
        # print(f"self.host_block_offsets: {self.host_block_offsets[self.recurrent_states_pool_index, :len(block_indices), 0, :20]}")
        # print(f"host_linear_block_offsets: {host_linear_block_offsets}")

        # torch.fill_(self._cuda_state_indices, 0)
        self._cuda_state_indices[:len(self.requests
                                      )] = host_linear_block_offsets.cuda()
        self._host_state_indices = host_linear_block_offsets.clone()


    def _setup_fake_states(self):
        block_indices = []
        self.next_block_id = []
        for req in self.requests:
            if req.is_context_finished:
                next_step = self.get_num_tokens(req) - 1
                current_step = next_step - 1
            elif self.kv_cache_config.enable_block_reuse:
                next_step = (req.context_current_position - 1 + req.context_chunk_size)
                current_step = req.context_current_position - 1
            else:
                next_step = req.prompt_len
                current_step = req.context_current_position - 1
            block_ids = self.get_cache_indices(req, LinearCacheType.RECURRENT_STATES.value)
            current_block_id = block_ids[current_step // self.tokens_per_block]
            next_block_id = block_ids[next_step // self.tokens_per_block]
            self.next_block_id.append(next_block_id)
            print(f"current_block_id: {current_block_id}, next_block_id: {next_block_id}")
            if current_block_id != next_block_id and not req.is_context_finished:
                print(f"fake copy states: {current_block_id} to {next_block_id}")
                ssm_states, conv_states = self._get_fake_states(current_block_id)
                next_ssm_states, next_conv_states = self._get_fake_states(next_block_id)
                next_ssm_states.copy_(ssm_states)
                next_conv_states.copy_(conv_states)

        self.fake_state_indices[:len(self.requests)] = torch.tensor(self.next_block_id, dtype=torch.int32, device="cuda")


    def get_state_indices(self) -> torch.Tensor:
        if self.use_fake_pool:
            return self.fake_state_indices
        return self._cuda_state_indices

    def calc_next_context_chunk_size(self, request: LlmRequest) -> int:
        """Compute the next prefill chunk size for a context request when block reuse is enabled.

        When kv_cache_config.enable_block_reuse is True, context prefill must stop exactly at
        the positions returned by calc_context_stop_positions (mamba_prefix_cache_step boundaries
        and block boundaries). This returns the chunk_size to use for the next prefill step so
        that the next stop position is not exceeded.

        Args:
            request: Context request with prompt_len and context_current_position set.

        Returns:
            Number of tokens to prefill in the next step (0 if context is already complete).
        """
        prompt_len = request.prompt_len
        current = request.context_current_position
        if current >= prompt_len:
            return 0
        step = self.linear_attention_metadata.states_snapshot_interval
        stop_positions = calc_context_stop_positions(
            prompt_len, self.tokens_per_block, step
        )
        stop_positions = sorted(set(stop_positions))
        for pos in stop_positions:
            if pos > current:
                return pos - current
        return prompt_len - current

    # [total_block_num, *ssm_state_shape] (one block for one layer)
    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        if self.use_fake_pool:
            return self.fake_ssm_states[self.linear_layer_offsets[layer_idx]]
        # return self.temp_ssm_states[layer_idx]
        # [total_block_num, 1, ssm_bytes + conv_bytes]
        pool = self.impl.get_recurrent_states_pool().view(torch.uint8).view(
            [-1, self.ssm_bytes + self.conv_bytes])
        # print(f"layer_idx: {layer_idx}, pool: {hex(pool.data_ptr())}, shape: {pool.shape}, dtype: {pool.dtype}")
        layer_idx = self.linear_layer_offsets[layer_idx]
        # print(f"shape of pool: {pool.shape}, dtype: {pool.dtype}")
        offset = (self.ssm_bytes +
                  self.conv_bytes) // self.ssm_state_dtype.itemsize * layer_idx

        flat = pool.view(self.ssm_state_dtype)
        assert flat.data_ptr() == pool.data_ptr()
        target_shape = [
            pool.shape[0] // self.num_linear_layers, *self.ssm_state_shape
        ]
        target_strides = [
            flat.stride(0) * self.num_linear_layers,
            self.ssm_state_shape[1] * self.ssm_state_shape[2],
            self.ssm_state_shape[2],
            1,
        ]
        my_ssm_states = torch.as_strided(flat,
                                         target_shape,
                                         target_strides,
                                         storage_offset=offset)
        # print(
        #     f"my_ssm_states: {hex(my_ssm_states.data_ptr())}, {my_ssm_states.shape}, is_view: {my_ssm_states._is_view()}")
        # print(f"layer_idx: {layer_idx}, linear_layer_offsets[layer_idx]: {self.linear_layer_offsets[layer_idx]}")
        # assert not my_ssm_states.is_contiguous()
        return my_ssm_states

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        if self.use_fake_pool:
            return self.fake_conv_states[self.linear_layer_offsets[layer_idx]]
        # return self.temp_conv_states[layer_idx]

        # [total_block_num, num_linear_layers, ssm_bytes + conv_bytes] -> [total_block_num * num_linear_layers, ssm_bytes + conv_bytes]
        pool = self.impl.get_recurrent_states_pool().view(torch.uint8).view(
            [-1, self.ssm_bytes + self.conv_bytes])
        # print(f"layer_idx: {layer_idx}, pool: {hex(pool.data_ptr())}, shape: {pool.shape}, dtype: {pool.dtype}")
        layer_idx = self.linear_layer_offsets[layer_idx]
        # print(f"shape of pool: {pool.shape}, dtype: {pool.dtype}")
        offset = self.ssm_bytes // self.conv_state_dtype.itemsize + \
            (self.ssm_bytes + self.conv_bytes) // self.conv_state_dtype.itemsize * layer_idx
        flat = pool.view(self.conv_state_dtype)
        # flat should be a view of pool
        assert flat.data_ptr() == pool.data_ptr()
        target_shape = [
            pool.shape[0] // self.num_linear_layers, *self.conv_state_shape
        ]
        target_strides = [
            flat.stride(0) * self.num_linear_layers, self.conv_state_shape[-1],
            1
        ]
        my_conv_states = torch.as_strided(flat,
                                          target_shape,
                                          target_strides,
                                          storage_offset=offset)
        # print(f"layer_idx: {layer_idx}, linear_layer_offsets[layer_idx]: {self.linear_layer_offsets[layer_idx]}")
        # print(
        #     f"my_conv_states: {hex(my_conv_states.data_ptr())}, {my_conv_states.shape}, {my_conv_states.stride()}")
        # assert not my_conv_states.is_contiguous()
        return my_conv_states

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.ssm_state_dtype

    def _get_fake_states(self, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fake_ssm_states[:, block_id], self.fake_conv_states[:, block_id]

    def _report_block_id_check_failures(self) -> None:
        """Print all collected block_id check failures at process exit."""
        if not self._block_id_check_failures:
            return
        if mpi_rank() != 0:
            return
        logger.error(
            f"MambaCacheManager block_id check reported {len(self._block_id_check_failures)} failure(s):"
        )
        for req_id in sorted(self._block_id_check_failures):
            reason, prev_block_ids, block_ids, current_position = self._block_id_check_failures[
                req_id
            ]
            logger.error(f"  request {req_id}: {reason}")
            logger.error(f"    current_position={current_position}")
            logger.error(f"    prev_block_ids={prev_block_ids}")
            logger.error(f"    block_ids={block_ids}")

    def _check_block_ids(self, request: LlmRequest):
        id = request.py_request_id
        block_ids = self.get_cache_indices(request, LinearCacheType.RECURRENT_STATES.value)
        prev_block_ids = self._request_block_ids.get(id)

        def fail(reason: str) -> None:
            if id in self._block_id_check_failures:
                return
            current_position = (
                request.context_current_position
                if not request.is_context_finished
                else self.get_num_tokens(request)
            )
            logger.warning(f"block_id check failed for request {id}: {reason}")
            self._block_id_check_failures[id] = (
                reason,
                list(prev_block_ids) if prev_block_ids is not None else [],
                list(block_ids),
                current_position,
            )
            if len(self._block_id_check_failures) >= 2:
                logger.error("Too many block_id check failures, exiting...")
                self._report_block_id_check_failures()
                import sys
                sys.exit(1)
        # If request is new (context current position is 0), but request_id present in _request_block_ids, it's likely due to warmup dummy requests. Just ignore the existing one.
        if prev_block_ids is None or request.context_current_position == 0:
            self._request_block_ids[id] = list(block_ids)
            return

        # The block id must meet following requirements:
        # 1. In context phase, block ids must never change
        # 2. In generation phase, block id only grows when self.get_num_tokens(req) is a multiple of tokens_per_block.
        #    When growing, the previous last block is shifted to the next slot, and a placeholder block (negative id) is inserted before.
        # For example: [0, -2, 1, -3, 2] -> [0, -2, 1, -3, -4, 2] when self.get_num_tokens(req) is 3 * tokens_per_block.
        if not request.is_context_finished:
            # Context phase: block ids must never change.
            if block_ids != prev_block_ids:
                fail(
                    f"in context phase block_ids must not change, "
                    f"got prev={prev_block_ids} current={block_ids}"
                )
                return
        else:
            # Generation phase: block id only grows when (num_tokens - 1) % tokens_per_block == 0.
            num_tokens = self.get_num_tokens(request)
            num_tokens_minus_one = self.get_num_tokens(request) - 1
            if num_tokens_minus_one % self.tokens_per_block == 0:
                # Allowed to grow: prev[:-1] + [placeholder] + [prev[-1]].
                if len(block_ids) != len(prev_block_ids) + 1:
                    fail(
                        f"on growth step (num_tokens={num_tokens}) block_ids length must be prev+1, "
                        f"got len(prev)={len(prev_block_ids)} len(current)={len(block_ids)}"
                    )
                    return
                if block_ids[-1] != prev_block_ids[-1] and (num_tokens_minus_one > request.prompt_len and self.linear_attention_metadata.save_last_snapshot):  # corner case
                    fail(
                        f"last block id must be unchanged when growing, prompt_len={request.prompt_len}, (num_tokens={num_tokens}), "
                        f"got prev[-1]={prev_block_ids[-1]} current[-1]={block_ids[-1]}"
                    )
                    return
                if block_ids[-2] >= 0:
                    fail(
                        f"new slot before last must be placeholder (negative id), "
                        f"got {block_ids[-2]}"
                    )
                    return
                if block_ids[:-2] != prev_block_ids[:-1]:
                    fail(
                        f"prefix before new placeholder must match prev[:-1], "
                        f"got prev[:-1]={prev_block_ids[:-1]} current[:-2]={block_ids[:-2]}"
                    )
                    return
            else:
                # No growth: block_ids must be unchanged.
                if block_ids != prev_block_ids:
                    fail(
                        f"in generation phase when not on block boundary "
                        f"block_ids must not change, num_tokens = {num_tokens}, "
                        f"got prev={prev_block_ids} current={block_ids}"
                    )
                    return
        self._request_block_ids[id] = list(block_ids)


MambaHybridCacheManager = LinearHybridCacheManager
