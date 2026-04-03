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

import os
from abc import ABC, abstractmethod
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
from tensorrt_llm._utils import (nvtx_range, prefer_pinned,
                                 torch_dtype_to_binding)
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


class BaseMambaCacheManager(ABC):
    """Abstract interface for accessing mamba/recurrent state caches.

    Implemented by MambaCacheManager (standalone mamba-only models) and
    LinearHybridCacheManager (hybrid attention+mamba models). Use
    ``isinstance(mgr, BaseMambaCacheManager)`` to check for mamba capability.
    """

    @abstractmethod
    def get_state_indices(self, *args, **kwargs) -> torch.Tensor:
        ...

    @abstractmethod
    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        ...

    @abstractmethod
    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        ...

    @abstractmethod
    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        ...

    @abstractmethod
    def is_speculative(self) -> bool:
        ...

    @abstractmethod
    def mamba_layer_cache(self, layer_idx: int):
        ...

    def reorder_state_indices_when_padding_requests(self, request_size: int,
                                                    padding_size: int):
        """Ensure padding slots use distinct state indices. No-op by default;
        overridden by PythonMambaCacheManager which manages its own index pool."""


class CppMambaCacheManager(BaseResourceManager):
    """Mamba state manager backed by the C++ RnnStateManager bindings.

    Manages only mamba states (conv + SSM). Used when TRTLLM_USE_CPP_MAMBA=1,
    which is required for disaggregated serving deployments.
    Does not support speculative decoding.
    """

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
    """Pure-Python mamba state manager with speculative decoding support.

    Manages only mamba states (conv + SSM) using PyTorch tensors on GPU.
    Supports caching intermediate states for speculative decoding verification.
    """

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
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size

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


class MambaCacheManager(BaseResourceManager, BaseMambaCacheManager):
    """Facade for standalone mamba state management (no KV cache).

    Delegates to CppMambaCacheManager (when TRTLLM_USE_CPP_MAMBA=1, required
    for disaggregated serving) or PythonMambaCacheManager (default, supports
    speculative decoding).
    """

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
    """Hybrid cache manager combining separate KVCacheManager and MambaCacheManager.

    Manages KV cache and mamba states in independent pools. Used for
    speculative decoding or disaggregated serving (via CppMambaCacheManager).
    Does not support block reuse / prefix caching for mamba states.
    """

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


def calc_context_stop_positions(prompt_len: int,
                                tokens_per_block: int,
                                mamba_prefix_cache_step: int,
                                save_last_snapshot: bool = False) -> list[int]:
    stop_positions = range(0, prompt_len, mamba_prefix_cache_step)
    stop_positions = list(stop_positions)
    last_ckpt = prompt_len // tokens_per_block * tokens_per_block
    if save_last_snapshot and (last_ckpt not in stop_positions):
        stop_positions.append(last_ckpt)
    if prompt_len not in stop_positions:
        stop_positions.append(prompt_len)
    return stop_positions


class LinearHybridCacheManager(KVCacheManager, BaseMambaCacheManager):
    """Hybrid cache manager storing mamba states inside the KVCacheManager pool.

    Both KV cache blocks and recurrent state blocks are managed by the unified
    C++ KVCacheManager, enabling block reuse / prefix caching across attention
    and mamba layers. This is the default hybrid manager.

    Disaggregated serving and speculative decoding are not supported yet.
    """

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
        # round conv_bytes to 1KB
        self.conv_bytes = ((self.conv_bytes + 1023) // 1024) * 1024

        self.linear_attention_metadata = LinearAttentionMetadata()
        self.linear_attention_metadata.cache_type = LinearCacheType.RECURRENT_STATES.value
        self.linear_attention_metadata.all_recurrent_states_bytes = self.ssm_bytes + self.conv_bytes
        self.linear_attention_metadata.states_snapshot_interval = kv_cache_config.mamba_prefix_cache_step

        if kv_cache_config.enable_partial_reuse:
            logger.warning(
                "Partial reuse is not supported for linear hybrid cache, disabling partial reuse"
            )
            kv_cache_config.enable_partial_reuse = False

        full_attention_layer_mask = layer_mask.copy()

        kv_cache_config.max_attention_window = []
        layer_mask = [
            mamba_layer_mask[i] or full_attention_layer_mask[i]
            for i, _ in enumerate(mamba_layer_mask)
        ]
        for i in range(len(layer_mask)):
            if layer_mask[i]:
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
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
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
            self.layer_offsets[self.linear_pp_layers[0]]][0]
        self._cuda_state_indices = torch.zeros([self.max_batch_size],
                                               dtype=torch.int32,
                                               device="cuda")
        self.kv_cache_config = kv_cache_config

        self.ssm_states_mapping = {}
        self.conv_states_mapping = {}
        for layer_id in self.linear_pp_layers:
            ssm_states = self._get_ssm_states(layer_id)
            conv_states = self._get_conv_states(layer_id)
            self.ssm_states_mapping[layer_id] = ssm_states
            self.conv_states_mapping[layer_id] = conv_states

        self._request_block_ids = {}
        self.iter = 0
        self.is_estimating_kv_cache = is_estimating_kv_cache

    def shutdown(self):
        # Release tensor views into the pool before the pool memory is freed,
        # so their deleters don't see stale pointers.
        self.ssm_states_mapping = None
        self.conv_states_mapping = None
        super().shutdown()

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
        requests = super().add_dummy_requests(request_ids, token_nums, is_gen,
                                              prepare_resource,
                                              max_num_draft_tokens, use_mrope,
                                              max_beam_width,
                                              num_extra_decoding_steps,
                                              draft_kv_cache_manager)
        self.requests.extend(requests)
        self._setup_state_indices()
        return requests

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        super().update_resources(scheduled_batch, attn_metadata,
                                 kv_cache_dtype_byte_size)

    @nvtx_range("hybrid_prepare_resources")
    def _prepare_resources(self, scheduled_batch: ScheduledRequests):
        self.iter += 1
        self.requests = scheduled_batch.context_requests + \
            scheduled_batch.generation_requests
        for req in self.requests:
            self.impl.copy_linear_attention_block(req)
        # self.impl.sync_transfer_manager_with_buffer_manager()
        self.impl.refresh_blocks()
        self._setup_state_indices()

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        super().prepare_resources(scheduled_batch)
        self._prepare_resources(scheduled_batch)

    def is_speculative(self) -> bool:
        # C++ MambaCacheManager does not support speculative decoding
        return False

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        return self.ssm_states_mapping[layer_idx]

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        return self.conv_states_mapping[layer_idx]

    def mamba_layer_cache(
            self, layer_idx: int) -> Union[PythonMambaCacheManager.State, None]:
        ret = PythonMambaCacheManager.State(
            conv=self.conv_states_mapping[layer_idx],
            temporal=self.ssm_states_mapping[layer_idx])
        return ret

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        if request in self.requests:
            self.requests.remove(request)
        super().free_resources(request, pin_on_release)

    # TODO: this should be called only once per iteration (not per layer)
    def _setup_state_indices(self) -> torch.Tensor:
        block_indices = []
        for req in self.requests:
            if req.is_context_finished:
                next_step = self.get_num_tokens(req) - 1
            elif self.kv_cache_config.enable_block_reuse:
                next_step = (req.context_current_position - 1 +
                             req.context_chunk_size)
            else:
                next_step = req.prompt_len - 1
            block_indices.append(next_step // self.tokens_per_block)
        self.impl.copy_batch_block_offsets(
            self.host_block_offsets,
            [req.py_request_id for req in self.requests], 1, 0)
        host_linear_block_offsets = torch.zeros([len(self.requests)],
                                                dtype=torch.int32,
                                                device="cpu")
        for i in range(len(self.requests)):
            # With layer-first pool layout, setOffsets produces the block index directly
            # (no longer multiplied by num_linear_layers)
            value = self.host_block_offsets[self.recurrent_states_pool_index, i,
                                            0, block_indices[i]]
            assert value >= 0 and value < self.blocks_per_window[LinearCacheType.RECURRENT_STATES.value][0], \
                f"value: {value} at index {i} is not in the range of [0, {self.blocks_per_window[LinearCacheType.RECURRENT_STATES.value][0]}).\nself.host_block_offsets[self.recurrent_states_pool_index, :, 0, 0]: {self.host_block_offsets[self.recurrent_states_pool_index, :, 0, 0]}"
            host_linear_block_offsets[i] = value

        torch.fill_(self._cuda_state_indices, 0)
        self._cuda_state_indices[:len(self.requests
                                      )] = host_linear_block_offsets.cuda()
        self._host_state_indices = host_linear_block_offsets.clone()

    def get_state_indices(self) -> torch.Tensor:
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
        if not self.kv_cache_config.enable_block_reuse:
            assert current == 0, f"Expected context_current_position to be 0 when block reuse is disabled, but got {current}"
            return prompt_len - current
        step = self.linear_attention_metadata.states_snapshot_interval
        stop_positions = calc_context_stop_positions(prompt_len,
                                                     self.tokens_per_block,
                                                     step)
        stop_positions = sorted(set(stop_positions))
        for pos in stop_positions:
            if pos > current:
                return pos - current
        return prompt_len - current

    # [total_block_num, *ssm_state_shape] (one block for one layer)
    def _get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        # Pool layout: {numLayers, numBlocks, ssm_bytes + conv_bytes} (as uint8)
        pool: torch.Tensor = self.impl.get_recurrent_states_pool().view(
            torch.uint8).reshape(self.num_linear_layers, -1,
                                 self.ssm_bytes + self.conv_bytes)
        layer_idx = self.linear_layer_offsets[layer_idx]
        # layer_pool: {numBlocks, ssm_bytes + conv_bytes}, contiguous
        layer_pool = pool[layer_idx]
        flat = layer_pool.view(self.ssm_state_dtype)
        assert flat.data_ptr() == layer_pool.data_ptr()
        total_elems_per_block = (
            self.ssm_bytes + self.conv_bytes) // self.ssm_state_dtype.itemsize
        target_shape = [flat.shape[0], *self.ssm_state_shape]
        target_strides = [
            total_elems_per_block,
            self.ssm_state_shape[1] * self.ssm_state_shape[2],
            self.ssm_state_shape[2],
            1,
        ]
        my_ssm_states = torch.as_strided(flat,
                                         target_shape,
                                         target_strides,
                                         storage_offset=flat.storage_offset())
        return my_ssm_states

    def _get_conv_states(self, layer_idx: int) -> torch.Tensor:
        # Pool layout: {numLayers, numBlocks, ssm_bytes + conv_bytes} (as uint8)
        pool: torch.Tensor = self.impl.get_recurrent_states_pool().view(
            torch.uint8).reshape(self.num_linear_layers, -1,
                                 self.ssm_bytes + self.conv_bytes)
        layer_idx = self.linear_layer_offsets[layer_idx]
        # layer_pool: {numBlocks, ssm_bytes + conv_bytes}, contiguous
        layer_pool = pool[layer_idx]
        flat = layer_pool.view(self.conv_state_dtype)
        assert flat.data_ptr() == layer_pool.data_ptr()
        total_elems_per_block = (
            self.ssm_bytes + self.conv_bytes) // self.conv_state_dtype.itemsize
        offset = self.ssm_bytes // self.conv_state_dtype.itemsize
        target_shape = [flat.shape[0], *self.conv_state_shape]
        target_strides = [total_elems_per_block, self.conv_state_shape[-1], 1]
        my_conv_states = torch.as_strided(flat,
                                          target_shape,
                                          target_strides,
                                          storage_offset=offset +
                                          flat.storage_offset())
        return my_conv_states

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.ssm_state_dtype


class _MambaHybridCacheManagerMeta(type):
    """Metaclass that enables isinstance/issubclass checks against
    MambaHybridCacheManager for both V1 and Linear implementations."""

    def __instancecheck__(cls, instance):
        if cls is MambaHybridCacheManager:
            return isinstance(
                instance, (MambaHybridCacheManagerV1, LinearHybridCacheManager))
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        if cls is MambaHybridCacheManager:
            return issubclass(
                subclass, (MambaHybridCacheManagerV1, LinearHybridCacheManager))
        return super().__subclasscheck__(subclass)

    def __getattr__(cls, name):
        """Forward class-level attribute access (e.g. static methods) to
        the KVCacheManager."""
        return getattr(KVCacheManager, name)


class MambaHybridCacheManager(metaclass=_MambaHybridCacheManagerMeta):
    """Factory that selects the appropriate hybrid cache manager.

    Selection logic:
    - Speculative decoding or TRTLLM_USE_CPP_MAMBA=1 -> MambaHybridCacheManagerV1
    - Otherwise (default) -> LinearHybridCacheManager
    """

    def __new__(
        cls,
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
        **kwargs,
    ):
        positional_args = (
            mamba_d_state,
            mamba_d_conv,
            mamba_num_heads,
            mamba_n_groups,
            mamba_head_dim,
            mamba_num_layers,
            mamba_layer_mask,
            mamba_cache_dtype,
            mamba_ssm_cache_dtype,
            kv_cache_config,
            kv_cache_type,
        )

        spec_config = kwargs.get('spec_config', None)
        use_v1 = (use_cpp_mamba_cache_manager() or spec_config is not None)

        if use_v1:
            logger.info(
                "Using MambaHybridCacheManagerV1 for hybrid cache management")
            return MambaHybridCacheManagerV1(*positional_args, **kwargs)
        else:
            logger.info(
                "Using LinearHybridCacheManager for hybrid cache management")
            return LinearHybridCacheManager(*positional_args, **kwargs)
