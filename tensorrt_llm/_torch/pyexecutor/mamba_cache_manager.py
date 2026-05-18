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

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

import tensorrt_llm.bindings

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager, get_pp_layers)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range, torch_dtype_to_binding
from tensorrt_llm.bindings.internal.batch_manager import (
    LinearAttentionMetadata, LinearCacheType)
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
    """Abstract interface for accessing mamba/recurrent state caches."""

    @abstractmethod
    def get_state_indices(self, *args, **kwargs) -> torch.Tensor:
        """Return slot indices of each request.

        Shape: [max_batch_size]
        """
        ...

    @abstractmethod
    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        """Return conv states for specific layer.

        Shape: [slot_size, conv_dim, d_conv - 1]
        """
        ...

    @abstractmethod
    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        """Return SSM states for specific layer.

        Shape: [slot_size, num_heads, head_dim, d_state]
        """
        ...

    @abstractmethod
    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        ...

    @abstractmethod
    def is_speculative(self) -> bool:
        ...

    @abstractmethod
    def mamba_layer_cache(
        self, layer_idx: int
    ) -> Union['PythonMambaCacheManager.State',
               'PythonMambaCacheManager.SpeculativeState', None]:
        ...


class CppMambaCacheManager(BaseResourceManager):
    """Mamba state manager backed by the C++ RnnStateManager bindings.

    Manages only mamba states (conv + SSM). Used when TRTLLM_USE_CPP_MAMBA=1.
    Supports disaggregated serving.
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
        # Allocate a permanent slot for every id, including CUDA-graph
        # padding sentinels (matches PythonMambaCacheManager). Padding
        # entries in get_state_indices then resolve via mCacheIndex to
        # the sentinel's reserved slot and never alias a live request.
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
    Supports speculative decoding and disaggregated serving.
    """

    @dataclass(frozen=True, kw_only=True)
    class State:
        """Base state container for Mamba cache."""
        conv: torch.Tensor
        # Represents last state, or "two back" if prev_num_accepted_tokens is > 0
        temporal: torch.Tensor

        # Subclasses override to list fields shared across layers (not indexed by layer).
        _SHARED_FIELDS = frozenset()

        def at_layer_idx(self, layer: int):
            kwargs = {}
            for k, v in vars(self).items():
                if v is None:
                    kwargs[k] = None
                elif k in self._SHARED_FIELDS:
                    kwargs[k] = v
                else:
                    kwargs[k] = v[layer]
            return type(self)(**kwargs)

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        """Speculative state with intermediate states for draft tokens.

        Supports two SSM update paths (only one set of tensors is allocated):
        - Legacy: caches full intermediate SSM states (intermediate_ssm)
        - Replay: compact double-buffered cache (old_x, old_B, old_dt, old_dA_cumsum)
        """
        _SHARED_FIELDS = frozenset(
            {"prev_num_accepted_tokens", "cache_buf_idx"})

        intermediate_conv_window: torch.Tensor  # always allocated

        # Legacy path: full intermediate SSM states at each step
        intermediate_ssm: torch.Tensor | None = None

        # Replay path: compact double-buffered cache
        # prev_num_accepted_tokens: # accepted tokens (always >= 1 if drafting).
        # 0 means temporal saved state is actually the last state, not two back.
        prev_num_accepted_tokens: torch.Tensor | None = None  # (cache,) int — shared across layers
        cache_buf_idx: torch.Tensor | None = None  # (cache,) int32 — shared across layers
        old_x: torch.Tensor | None = None  # (layers, cache, T, nheads, dim)
        old_B: torch.Tensor | None = None  # (layers, cache, 2, T, ngroups, dstate)
        # Processed dt: softplus(raw_dt + dt_bias), clamped to dt_limit.
        old_dt: torch.Tensor | None = None  # (layers, cache, 2, nheads, T) fp32
        old_dA_cumsum: torch.Tensor | None = None  # (layers, cache, 2, nheads, T) fp32

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
        model_type: str = "nemotron_hybrid",
        use_replay_state_update: bool = False,
    ) -> None:

        self.mamba_ssm_cache_dtype = ssm_cache_dtype
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.spec_state_size = spec_state_size
        self._use_replay_state_update = use_replay_state_update

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
        d_inner_local = d_inner // tp_size
        ng_ds_local = n_groups * d_state // tp_size
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size
        d_inner = d_inner // tp_size

        # Per-section dims for conv_state.
        # Qwen3-Next: [Q | K | V] = [ng*ds, ng*ds, d_inner]
        # Nemotron_hybrid: [x | B | C] = [d_inner, ng*ds, ng*ds]
        if model_type == "qwen3_next":
            self.conv_section_dims = [ng_ds_local, ng_ds_local, d_inner_local]
        elif model_type == "nemotron_hybrid":
            self.conv_section_dims = [d_inner_local, ng_ds_local, ng_ds_local]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

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
            T = speculative_num_draft_tokens + 1

            # Conv intermediate cache — same for both paths
            intermediate_conv_window_cache = torch.zeros(
                size=(num_local_layers, self.spec_state_size, T) +
                conv_state_shape,
                dtype=dtype,
                device=device,
            )

            # SSM speculative cache — path-specific tensors
            spec_kwargs = {}
            if self._use_replay_state_update:
                assert n_groups % tp_size == 0, \
                    "replay state update requires n_groups divisible by tp_size"
                n_groups_per_rank = n_groups // tp_size

                # Compact replay cache.
                # old_x is single-buffered (written by main kernel after replay).
                # old_B, old_dt, old_dA_cumsum are double-buffered (written by
                # precompute kernel concurrently with main kernel via PDL).
                spec_kwargs['prev_num_accepted_tokens'] = torch.zeros(
                    max_batch_size, dtype=int, device=device)
                spec_kwargs['cache_buf_idx'] = torch.zeros(max_batch_size,
                                                           dtype=torch.int32,
                                                           device=device)
                spec_kwargs['old_x'] = torch.zeros(num_local_layers,
                                                   max_batch_size,
                                                   T,
                                                   nheads,
                                                   head_dim,
                                                   dtype=dtype,
                                                   device=device)
                spec_kwargs['old_B'] = torch.zeros(num_local_layers,
                                                   max_batch_size,
                                                   2,
                                                   T,
                                                   n_groups_per_rank,
                                                   d_state,
                                                   dtype=dtype,
                                                   device=device)
                spec_kwargs['old_dt'] = torch.zeros(num_local_layers,
                                                    max_batch_size,
                                                    2,
                                                    nheads,
                                                    T,
                                                    dtype=torch.float32,
                                                    device=device)
                spec_kwargs['old_dA_cumsum'] = torch.zeros(num_local_layers,
                                                           max_batch_size,
                                                           2,
                                                           nheads,
                                                           T,
                                                           dtype=torch.float32,
                                                           device=device)
                ssm_spec_cache = [
                    spec_kwargs['old_x'], spec_kwargs['old_B'],
                    spec_kwargs['old_dt'], spec_kwargs['old_dA_cumsum']
                ]
                spec_path_label = "replay"
            else:
                # Legacy: full intermediate SSM states at each step
                spec_kwargs['intermediate_ssm'] = torch.zeros(
                    size=(num_local_layers, self.spec_state_size, T) +
                    ssm_state_shape,
                    dtype=self.mamba_ssm_cache_dtype,
                    device=device)
                ssm_spec_cache = [spec_kwargs['intermediate_ssm']]
                spec_path_label = "legacy"

            self.mamba_cache = self.SpeculativeState(
                conv=conv_states,
                temporal=ssm_states,
                intermediate_conv_window=intermediate_conv_window_cache,
                **spec_kwargs,
            )

            logger.info(
                f"Mamba Cache ({spec_path_label}) is allocated. "
                f"max_mamba_cache_size: {max_batch_size}, "
                f"conv_state size: {get_tensor_size_bytes(conv_states) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(ssm_states) / GB:.2f}GB, "
                f"ssm_spec_cache size: {get_tensor_size_bytes(ssm_spec_cache) / GB:.2f}GB, "
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

        # Permanent slot shared by every CUDA-graph padding sentinel id
        # (CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len, one per
        # draft length). Pool sizing must include +1 headroom for this;
        # see MixedMambaHybridCacheManager.
        self._padding_slot: int = self.mamba_cache_free_blocks.pop()

        # save intermediate state indices for requests
        self.intermediate_state_indices = torch.arange(max_batch_size,
                                                       dtype=torch.int32,
                                                       device=device)

        # Store max_batch_size for resource management
        self._max_batch_size = max_batch_size

    def get_max_resource_count(self) -> int:
        """Return the maximum number of sequences that can be cached."""
        return self._max_batch_size

    def filter_ctx_requests_by_capacity(self, context_requests: list) -> list:
        """Return the prefix of *context_requests* that fits in the
        available Mamba cache blocks.  Requests that already have a
        cached block do not consume a free block."""
        free = len(self.mamba_cache_free_blocks)
        result = []
        for r in context_requests:
            if r.py_request_id in self.mamba_cache_index:
                result.append(r)
            elif free > 0:
                result.append(r)
                free -= 1
            else:
                break
        return result

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        """For Mamba cache manager, we always need one slot per request."""
        return 1

    @torch.inference_mode()
    def _prepare_mamba_cache_blocks(self, request_ids: List[int]):
        for r in request_ids:
            if r in self.mamba_cache_index:
                continue
            if len(self.mamba_cache_free_blocks) == 0:
                raise RuntimeError("run out of mamba cache blocks")
            block = self.mamba_cache_free_blocks.pop()
            self.mamba_cache_index[r] = block
            if (isinstance(self.mamba_cache, self.SpeculativeState)
                    and self._use_replay_state_update):
                self.mamba_cache.prev_num_accepted_tokens[block] = 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_ids = [
            i.py_request_id for i in scheduled_batch.context_requests
        ]
        generation_ids = [
            i.py_request_id for i in scheduled_batch.generation_requests
        ]
        request_ids = context_ids + generation_ids
        self._prepare_mamba_cache_blocks(request_ids)

    def _is_padding_sentinel(self, request_id: int) -> bool:
        # cuda_graph_runner caches one dummy per runtime_draft_len value
        # (see _get_padded_batch), so any id in the range of dummy request IDs
        # may be live concurrently.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
            CUDA_GRAPH_DUMMY_REQUEST_ID
        max_dl = self.speculative_num_draft_tokens or 0
        return (CUDA_GRAPH_DUMMY_REQUEST_ID - max_dl <= request_id <=
                CUDA_GRAPH_DUMMY_REQUEST_ID)

    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        # Sentinels alias to the shared _padding_slot; non-sentinel
        # dummies (warmup, attention-DP idle padding) get their own
        # slot and are freed individually.
        if not request_ids:
            return
        for r in request_ids:
            if r in self.mamba_cache_index:
                continue
            if self._is_padding_sentinel(r):
                self.mamba_cache_index[r] = self._padding_slot
            else:
                if len(self.mamba_cache_free_blocks) == 0:
                    raise RuntimeError("run out of mamba cache blocks")
                block = self.mamba_cache_free_blocks.pop()
                self.mamba_cache_index[r] = block

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        if request_id not in self.mamba_cache_index:
            return
        block = self.mamba_cache_index.pop(request_id)
        # _padding_slot stays reserved; only non-sentinel blocks return.
        if not self._is_padding_sentinel(request_id):
            self.mamba_cache_free_blocks.append(block)

    def get_state_indices(self, request_ids: List[int],
                          is_padding: List[bool]) -> List[int]:
        return [self.mamba_cache_index[rid] for rid in request_ids]

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

    @property
    def use_replay_state_update(self) -> bool:
        return self._use_replay_state_update

    def shutdown(self):
        """Release tensor memory."""
        # Clear mamba cache states
        empty = torch.tensor([])

        def _drop(tensor):
            return empty if tensor is not None else None

        if isinstance(self.mamba_cache, self.SpeculativeState):
            self.mamba_cache = self.SpeculativeState(
                conv=empty,
                temporal=empty,
                intermediate_conv_window=empty,
                intermediate_ssm=_drop(self.mamba_cache.intermediate_ssm),
                prev_num_accepted_tokens=_drop(
                    self.mamba_cache.prev_num_accepted_tokens),
                cache_buf_idx=_drop(self.mamba_cache.cache_buf_idx),
                old_x=_drop(self.mamba_cache.old_x),
                old_B=_drop(self.mamba_cache.old_B),
                old_dt=_drop(self.mamba_cache.old_dt),
                old_dA_cumsum=_drop(self.mamba_cache.old_dA_cumsum),
            )
        else:
            self.mamba_cache = self.State(conv=empty, temporal=empty)

        torch.cuda.empty_cache()

    @torch.compile(options={"max-autotune": True})
    def update_mamba_states(self, attn_metadata: "AttentionMetadata",
                            num_accepted_tokens: torch.Tensor,
                            state_indices: torch.Tensor):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = num_accepted_tokens[
            num_contexts:num_contexts + num_gens] - 1
        state_indices_d = state_indices[num_contexts:num_contexts + num_gens]
        src_state_indices = self.intermediate_state_indices[:num_gens]

        if self._use_replay_state_update:
            # SSM state is handled incrementally by the kernel.  Update the
            # number of accepted tokens and flip the double-buffer index so the
            # next step's replay reads from the buffer that was just written by
            # the precompute kernel.
            self.mamba_cache.prev_num_accepted_tokens[state_indices_d] = \
                num_accepted_tokens[num_contexts:num_contexts + num_gens]
            self.mamba_cache.cache_buf_idx[state_indices_d] = \
                1 - self.mamba_cache.cache_buf_idx[state_indices_d]
        else:
            # Legacy: copy accepted SSM state from intermediate cache.
            ssm_states = self.mamba_cache.temporal
            intermediate_ssm_cache = self.mamba_cache.intermediate_ssm
            accepted_ssm_state = intermediate_ssm_cache[:, src_state_indices,
                                                        num_accepted_draft_tokens]
            ssm_states[:, state_indices_d, :] = accepted_ssm_state

        # Conv: both paths save all intermediate conv windows, carry over the accepted one.
        conv_states = self.mamba_cache.conv
        intermediate_conv_window_cache = self.mamba_cache.intermediate_conv_window
        accepted_conv_state = intermediate_conv_window_cache[:,
                                                             src_state_indices,
                                                             num_accepted_draft_tokens]
        conv_states[:, state_indices_d, :] = accepted_conv_state


class MambaCacheManager(BaseResourceManager, BaseMambaCacheManager):
    """Facade for standalone mamba state management (no KV cache).

    Delegates to CppMambaCacheManager (when TRTLLM_USE_CPP_MAMBA=1) or PythonMambaCacheManager.
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
        model_type: str = "nemotron_hybrid",
        use_replay_state_update: bool = False,
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
                model_type=model_type,
                use_replay_state_update=use_replay_state_update,
            )

    def get_max_resource_count(self) -> int:
        return self._impl.get_max_resource_count()

    def filter_ctx_requests_by_capacity(self, context_requests: list) -> list:
        if self._use_cpp:
            return context_requests
        return self._impl.filter_ctx_requests_by_capacity(context_requests)

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return self._impl.get_needed_resource_to_completion(request)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        self._impl.prepare_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        self._impl.free_resources(request)

    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        self._impl.add_dummy_requests(request_ids, **kwargs)

    def get_state_indices(
        self,
        request_ids: Optional[List[int]] = None,
        is_padding: Optional[List[bool]] = None
    ) -> Union[torch.Tensor, List[int]]:
        return self._impl.get_state_indices(request_ids, is_padding)

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

    @property
    def use_replay_state_update(self) -> bool:
        return getattr(self._impl, 'use_replay_state_update', False)

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
                            num_accepted_tokens: torch.Tensor,
                            state_indices: torch.Tensor):
        # Non-speculative configs don't allocate intermediate state; the
        # promotion is a clean no-op.
        if not self._impl.is_speculative():
            return
        # Belt-and-suspenders: C++ is non-speculative today so this is
        # unreachable. Fires if C++ ever grows speculative support
        # without also implementing the scatter there.
        assert not self._use_cpp, "update_mamba_states is not supported in CppMambaCacheManager"
        self._impl.update_mamba_states(attn_metadata, num_accepted_tokens,
                                       state_indices)


class MambaHybridCacheManager(BaseResourceManager, BaseMambaCacheManager):
    """Marker base class for hybrid mamba cache manager implementations.

    Used purely for ``isinstance`` / type-hint purposes so callers can refer
    to the family without caring about the concrete implementation. Concrete
    selection (Mixed vs Cpp) lives in ``_util.py:_get_model_kv_cache_manager_cls``.
    """


class MixedMambaHybridCacheManager(KVCacheManager, MambaCacheManager,
                                   MambaHybridCacheManager):
    """Hybrid cache manager combining separate KVCacheManager and MambaCacheManager.

    Manages KV cache and mamba states in independent pools, with support of
    speculative decoding and disaggregated serving.
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
        model_type: str = "nemotron_hybrid",
        is_draft: bool = False,
        use_replay_state_update: bool = False,
    ) -> None:

        # mamba hybrid cache requires block reuse to be disabled in KV cache config
        assert not kv_cache_config.enable_block_reuse, "mamba hybrid cache requires block reuse to be disabled in KV cache config"

        # +1 headroom for PythonMambaCacheManager._padding_slot, which
        # is shared by every CUDA-graph padding sentinel regardless of
        # max_draft_len.
        pool_size = max_batch_size + 1

        MambaCacheManager.__init__(
            self,
            mamba_d_state,
            mamba_d_conv,
            mamba_num_heads,
            mamba_n_groups,
            mamba_head_dim,
            mamba_num_layers,
            pool_size,
            max_batch_size,
            mapping,
            mamba_cache_dtype,
            mamba_ssm_cache_dtype,
            mamba_layer_mask,
            execution_stream,
            speculative_num_draft_tokens=(spec_config.tokens_per_gen_step - 1
                                          if spec_config is not None else None),
            model_type=model_type,
            use_replay_state_update=use_replay_state_update,
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
            is_draft=is_draft,
        )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        MambaCacheManager.prepare_resources(self, scheduled_batch)
        KVCacheManager.prepare_resources(self, scheduled_batch)

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        MambaCacheManager.free_resources(self, request)
        KVCacheManager.free_resources(self, request, pin_on_release)

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


def calc_context_stop_positions(prompt_len: int,
                                tokens_per_block: int,
                                mamba_state_cache_interval: int,
                                save_last_snapshot: bool = False) -> list[int]:
    """Compute token positions at which mamba state snapshots should be saved.

    Returns positions spaced by ``mamba_state_cache_interval`` plus the final
    prompt length (and optionally the last block-aligned position).
    """
    stop_positions = list(
        range(mamba_state_cache_interval, prompt_len,
              mamba_state_cache_interval))
    last_ckpt = prompt_len // tokens_per_block * tokens_per_block
    if save_last_snapshot and (last_ckpt not in stop_positions):
        stop_positions.append(last_ckpt)
    if prompt_len not in stop_positions:
        stop_positions.append(prompt_len)
    return stop_positions


class CppMambaHybridCacheManager(KVCacheManager, MambaHybridCacheManager):
    """Hybrid cache manager storing mamba states inside the KVCacheManager pool.

    Both KV cache blocks and recurrent state blocks are managed by the unified
    C++ KVCacheManager, enabling block reuse / prefix caching across attention
    and mamba layers. This is the default hybrid manager.

    Disaggregated serving is not supported yet.
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
        layer_mask: Optional[
            List[bool]] = None,  # this is the full attention layer mask
        is_estimating_kv_cache: bool = False,
        use_replay_state_update: bool = False,
        **kwargs,
    ) -> None:
        # 3 kinds of layers:
        # 1) Mamba layers (mamba_layer_mask is True)
        # 2) Full attention layers (full_attention_layer_mask is True)
        # 3) Not managed layers (both masks are False)
        total_layers = len(mamba_layer_mask)
        if layer_mask is None:
            full_attention_layer_mask = [False] * total_layers
        elif len(layer_mask) != total_layers:
            raise ValueError(
                f"layer_mask length ({len(layer_mask)}) must match "
                f"mamba_layer_mask length ({total_layers})")
        else:
            full_attention_layer_mask = list(layer_mask)
        layer_mask = [
            mamba_layer_mask[i] or full_attention_layer_mask[i]
            for i in range(total_layers)
        ]
        # PP sharding is done across all layers.
        # This is called again in the super().__init__, but we want it to run first
        # to set up mtp states before the C++ backend is initialized.
        self.pp_layers, _ = get_pp_layers(
            mamba_num_layers + num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.mamba_pp_layers = [
            layer_idx for layer_idx in self.pp_layers
            if mamba_layer_mask[layer_idx]
        ]
        self.local_num_mamba_layers = len(self.mamba_pp_layers)
        self.requests = []
        # Seed externally visible mamba fields before any early return so that
        # accessors (get_mamba_ssm_cache_dtype, use_replay_state_update) work
        # on ranks with no local mamba layers.
        self._use_replay_state_update = use_replay_state_update
        self.ssm_state_dtype = mamba_ssm_cache_dtype

        if self.local_num_mamba_layers == 0:
            logger.info(
                "No local mamba layers for this rank, skipping mamba cache initialization"
            )
            super().__init__(
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
                layer_mask=full_attention_layer_mask,
                is_estimating_kv_cache=is_estimating_kv_cache,
            )
            return

        # Derive ssm_state_shape and conv_state_shape from mamba params (same as MambaCacheManager)
        tp_size = mapping.tp_size if not mapping.enable_attention_dp else 1
        d_inner = mamba_head_dim * mamba_num_heads
        conv_dim = d_inner + 2 * mamba_n_groups * mamba_d_state
        nheads = mamba_num_heads
        assert nheads % tp_size == 0, "mamba_num_heads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"
        if use_replay_state_update:
            assert mamba_n_groups % tp_size == 0, \
                "replay state update requires mamba_n_groups divisible by tp_size"
        self._n_groups_per_rank = mamba_n_groups // tp_size
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size
        self.conv_state_shape = [conv_dim, mamba_d_conv - 1]
        self.ssm_state_shape = [nheads, mamba_head_dim, mamba_d_state]
        self.conv_state_dtype = mamba_cache_dtype
        self.ssm_count = math.prod(self.ssm_state_shape)
        self.conv_count = math.prod(self.conv_state_shape)
        self.ssm_bytes = self.ssm_count * self.ssm_state_dtype.itemsize
        self.conv_bytes = self.conv_count * self.conv_state_dtype.itemsize

        total_bytes = self.ssm_bytes + self.conv_bytes
        if total_bytes % self.ssm_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"Total state bytes ({total_bytes}) not divisible by "
                f"ssm_state_dtype size ({self.ssm_state_dtype.itemsize})")
        if total_bytes % self.conv_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"Total state bytes ({total_bytes}) not divisible by "
                f"conv_state_dtype size ({self.conv_state_dtype.itemsize})")
        if self.ssm_bytes % self.conv_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"SSM state bytes ({self.ssm_bytes}) not divisible by "
                f"conv_state_dtype size ({self.conv_state_dtype.itemsize})")
        self.linear_attention_metadata = LinearAttentionMetadata()
        self.linear_attention_metadata.cache_type = LinearCacheType.RECURRENT_STATES.value
        self.linear_attention_metadata.all_recurrent_states_bytes = self.ssm_bytes + self.conv_bytes
        self.linear_attention_metadata.states_snapshot_interval = kv_cache_config.mamba_state_cache_interval if kv_cache_config.enable_block_reuse else 0
        kv_cache_config = kv_cache_config.model_copy(deep=True)
        if kv_cache_config.enable_partial_reuse:
            logger.warning(
                "Partial reuse is not supported for mamba hybrid models, disabling partial reuse"
            )
            kv_cache_config.enable_partial_reuse = False

        kv_cache_config.max_attention_window = []
        for i in range(len(layer_mask)):
            if layer_mask[i]:
                kv_cache_config.max_attention_window.append(
                    LinearCacheType.RECURRENT_STATES.
                    value if mamba_layer_mask[i] else max_seq_len)

        # Normalize num_kv_heads to a per-layer list and zero out mamba
        # layer positions: those layers carry SSM/conv state instead of KV
        # heads, so the parent KV cache should not allocate KV head storage
        # for them.
        if isinstance(num_kv_heads, int):
            per_layer_kv_heads = [num_kv_heads] * total_layers
        else:
            if len(num_kv_heads) != total_layers:
                raise ValueError(
                    f"num_kv_heads list length ({len(num_kv_heads)}) does not "
                    f"match total layers ({total_layers})")
            per_layer_kv_heads = list(num_kv_heads)
        for i, is_mamba in enumerate(mamba_layer_mask):
            if is_mamba:
                per_layer_kv_heads[i] = 0

        self._setup_mtp_intermediate_states(spec_config, max_batch_size)

        # pass remaining arguments to super class
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=mamba_num_layers + num_layers,
            num_kv_heads=per_layer_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            is_estimating_kv_cache=is_estimating_kv_cache,
            linear_attention_metadata=self.linear_attention_metadata,
        )

        assert self.local_num_mamba_layers > 0, "At least one mamba layer is required"
        self.mamba_layer_offsets = {}
        for idx, layer_id in enumerate(self.mamba_pp_layers):
            self.mamba_layer_offsets[layer_id] = idx

        self.host_block_offsets = torch.zeros([
            self.impl.num_pools, self.max_batch_size, 2, self.max_blocks_per_seq
        ],
                                              dtype=torch.int32,
                                              device="cpu")
        self.recurrent_states_pool_index = self.kv_cache_pool_mapping[
            self.layer_offsets[self.mamba_pp_layers[0]]][0]

        self.cuda_state_indices = torch.zeros([self.max_batch_size],
                                              dtype=torch.int32,
                                              device="cuda")
        self.kv_cache_config = kv_cache_config
        self.is_estimating_kv_cache = is_estimating_kv_cache

        self._setup_states()
        self._setup_replay_buffers(spec_config)

    @staticmethod
    def get_cache_size_per_token(
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        """Affine memory model for the unified hybrid KV pool.

        Returns ``(slope_bytes_per_token, intercept_bytes)``:

        * ``slope`` = attention KV bytes per token (parent's formula) plus
          the amortized regular-snapshot bytes per token from mamba layers.
        * ``intercept`` = ``max_batch_size * num_mamba_layers_per_rank *
          state_bytes_per_layer * STATIC_SLOTS_PER_REQUEST``.

        Memory budget -> max tokens then becomes
        ``T = (budget - intercept) // slope`` instead of plain
        ``T = budget // bytes_per_token``.
        """
        # Lazy import to avoid pulling config_utils into module import order.
        from tensorrt_llm._torch.pyexecutor.config_utils import \
            extract_mamba_kv_cache_params

        # Attention slope from the parent's existing formula.
        attention_slope = KVCacheManager.get_cache_size_per_token(
            model_config, mapping, num_layers=num_layers, **kwargs)

        params = extract_mamba_kv_cache_params(
            model_config.pretrained_config,
            quant_config=model_config.quant_config,
        )

        state_bytes_per_layer = params.get_states_bytes_per_layer(mapping)

        # This not precise since pp layers are sharded by their order in model, not by their types.
        # e.g. the upper half are all mamba layers while the lower half are all attention layers.
        # But that's close enough for real world models where mamba and attention layers are interleaved
        # and we don't have access with layer_masks at this point.
        num_mamba_layers_per_rank = len(
            mapping.pp_layers(params.num_mamba_layers))
        state_bytes_per_rank = num_mamba_layers_per_rank * state_bytes_per_layer

        # Per-request fixed cost. STATIC_SLOTS_PER_REQUEST = 1 today (the
        # live mamba state); fixed-position snapshots are not yet
        # implemented and would simply increment this constant.
        STATIC_SLOTS_PER_REQUEST = 1
        intercept = (max_batch_size * state_bytes_per_rank *
                     STATIC_SLOTS_PER_REQUEST)

        # Regular-snapshot bytes per token. None / non-positive intervals
        # mean "no regular snapshots", so the mamba contribution is zero.
        interval = kv_cache_config.mamba_state_cache_interval if kv_cache_config.enable_block_reuse else 0
        if interval is None or interval <= 0:
            mamba_slope = 0
        else:
            mamba_slope = state_bytes_per_rank // interval
        # heuristic: When block reuse is enabled, we assume the mamba snapshots are dominant instead of active states,
        # otherwise we may run out of kv cache blocks prior to mamba blocks due to the large number of max_batch_size.
        # So we ignore intercept and only calculate max_tokens based on slope
        # This can be improved by a more accurate max_batch_size and ISL/OSL estimation in the future.
        if mamba_slope > 0:
            intercept = 0
        return attention_slope + mamba_slope, intercept

    def shutdown(self):
        # Release tensor views into the pool before the pool memory is freed,
        # so their deleters don't see stale pointers.
        self.all_ssm_states = None
        self.all_conv_states = None
        self.intermediate_ssm_states = None
        self.intermediate_conv_states = None
        self.intermediate_state_indices = None
        self.prev_num_accepted_tokens = None
        self.cache_buf_idx = None
        self.old_x = None
        self.old_B = None
        self.old_dt = None
        self.old_dA_cumsum = None
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
        if requests:
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
        self.requests = scheduled_batch.context_requests + \
            scheduled_batch.generation_requests
        for req in self.requests:
            self.impl.copy_linear_attention_block(req)
        self.impl.refresh_blocks()
        self._setup_state_indices()
        # Reset replay double-buffer state for fresh context blocks. A reused
        # block (prefix-cache hit or block recycled across requests) may carry
        # stale prev_num_accepted_tokens / cache_buf_idx values from a prior
        # owner; the replay kernel reads these on the first decode step.
        if self._use_replay_state_update and self.prev_num_accepted_tokens is not None:
            num_contexts = len(scheduled_batch.context_requests)
            if num_contexts > 0:
                ctx_slots = self.cuda_state_indices[:num_contexts].long()
                self.prev_num_accepted_tokens[ctx_slots] = 0
                # don't care which half of doulbe-buffer is using
                # self.cache_buf_idx[ctx_slots] = 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        super().prepare_resources(scheduled_batch)
        if self.local_num_mamba_layers == 0:
            return
        self._prepare_resources(scheduled_batch)

    def is_speculative(self) -> bool:
        return self.spec_config is not None

    def update_mamba_states(self,
                            attn_metadata: "AttentionMetadata",
                            num_accepted_tokens: torch.Tensor,
                            state_indices: Optional[torch.Tensor] = None):
        if self.local_num_mamba_layers == 0:
            return
        # Note: cannot use @torch.compile here because all_ssm_states and
        # all_conv_states are dtype-reinterpreted views of the C++ pool
        # (uint8 -> typed), and aot_autograd does not support mutations on
        # views with different dtypes.
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = num_accepted_tokens[
            num_contexts:num_contexts + num_gens] - 1
        # Match the API of MambaCacheManager.update_mamba_states: callers
        # may pass per-request state slot indices explicitly (e.g. MTP via
        # attn_metadata.mamba_metadata.state_indices). Fall back to this
        # manager's own slot mapping when not provided.
        if state_indices is None:
            state_indices = self.get_state_indices()
        state_indices_d = state_indices[num_contexts:num_contexts + num_gens]

        src_state_indices = self.intermediate_state_indices[:num_gens]

        if self._use_replay_state_update:
            # SSM state is handled incrementally by the replay kernel.  Update
            # the per-slot accepted-token counter and flip the double-buffer
            # index so the next step reads from the buffer that was just
            # written by the precompute kernel.
            accepted = num_accepted_tokens[num_contexts:num_contexts + num_gens]
            self.prev_num_accepted_tokens[state_indices_d] = accepted.to(
                self.prev_num_accepted_tokens.dtype)
            self.cache_buf_idx[state_indices_d] = \
                1 - self.cache_buf_idx[state_indices_d]
        else:
            # Legacy: copy accepted SSM states from intermediate buffer back to pool
            accepted_ssm = self.intermediate_ssm_states[:, src_state_indices,
                                                        num_accepted_draft_tokens]
            self.all_ssm_states[:, state_indices_d, :] = accepted_ssm

        # Conv: both paths save all intermediate conv windows, carry over the accepted one.
        accepted_conv = self.intermediate_conv_states[:, src_state_indices,
                                                      num_accepted_draft_tokens]
        self.all_conv_states[:, state_indices_d, :] = accepted_conv

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        return self.all_ssm_states[self.mamba_layer_offsets[layer_idx]]

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        return self.all_conv_states[self.mamba_layer_offsets[layer_idx]]

    def get_intermediate_ssm_states(self,
                                    layer_idx: int) -> Optional[torch.Tensor]:
        if self.intermediate_ssm_states is None:
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.intermediate_ssm_states[layer_offset]

    def get_intermediate_conv_states(self,
                                     layer_idx: int) -> Optional[torch.Tensor]:
        if self.intermediate_conv_states is None:
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.intermediate_conv_states[layer_offset]

    def mamba_layer_cache(
        self, layer_idx: int
    ) -> Union[PythonMambaCacheManager.State,
               PythonMambaCacheManager.SpeculativeState, None]:
        conv = self.get_conv_states(layer_idx)
        ssm = self.get_ssm_states(layer_idx)
        if self.spec_config is not None:
            layer_offset = self.mamba_layer_offsets[layer_idx]
            spec_kwargs = {}
            if self._use_replay_state_update:
                # Per-layer slices for the replay kernel; shared 1D tensors
                # (cache_buf_idx, prev_num_accepted_tokens) are passed
                # untouched via the SpeculativeState._SHARED_FIELDS contract.
                spec_kwargs['old_x'] = self.old_x[layer_offset]
                spec_kwargs['old_B'] = self.old_B[layer_offset]
                spec_kwargs['old_dt'] = self.old_dt[layer_offset]
                spec_kwargs['old_dA_cumsum'] = self.old_dA_cumsum[layer_offset]
                spec_kwargs['cache_buf_idx'] = self.cache_buf_idx
                spec_kwargs['prev_num_accepted_tokens'] = (
                    self.prev_num_accepted_tokens)
            else:
                spec_kwargs['intermediate_ssm'] = self.intermediate_ssm_states[
                    layer_offset]
            return PythonMambaCacheManager.SpeculativeState(
                conv=conv,
                temporal=ssm,
                intermediate_conv_window=self.
                intermediate_conv_states[layer_offset],
                **spec_kwargs,
            )
        return PythonMambaCacheManager.State(conv=conv, temporal=ssm)

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        if request in self.requests:
            self.requests.remove(request)
        super().free_resources(request, pin_on_release)

    def _setup_state_indices(self) -> None:
        if self.local_num_mamba_layers == 0:
            return
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
        host_block_offsets = torch.zeros([len(self.requests)],
                                         dtype=torch.int32,
                                         device="cpu")
        for i in range(len(self.requests)):
            # With layer-first pool layout, setOffsets produces the block index directly
            # (no longer multiplied by num_local_mamba_layers)
            value = self.host_block_offsets[self.recurrent_states_pool_index, i,
                                            0, block_indices[i]]
            max_blocks = self.blocks_per_window[
                LinearCacheType.RECURRENT_STATES.value][0]
            if value < 0 or value >= max_blocks:
                req = self.requests[i]
                raise RuntimeError(
                    f"Invalid recurrent state block index {value} "
                    f"(expected 0 <= index < {max_blocks}) for request {i}, "
                    f"prompt_len={req.prompt_len}, "
                    f"is_context_finished={req.is_context_finished}, "
                    f"context_current_position={req.context_current_position}, "
                    f"prepopulated_token_num={req.prepopulated_prompt_len}, "
                    f"context_chunk_size={req.context_chunk_size if not req.is_context_finished else 'N/A'}, "
                    f"block_index for next step is {block_indices[i]}, "
                    f"\nblock_ids={self.impl.get_cache_block_ids(req.py_request_id, LinearCacheType.RECURRENT_STATES.value)}"
                )
            host_block_offsets[i] = value

        torch.fill_(self.cuda_state_indices, 0)
        self.cuda_state_indices[:len(self.requests)] = host_block_offsets.cuda()
        self._host_state_indices = host_block_offsets.clone()

    def get_state_indices(
            self,
            request_ids: Optional[List[int]] = None,
            is_padding: Optional[List[bool]] = None) -> torch.Tensor:
        return self.cuda_state_indices

    def calc_next_context_chunk_size(self, request: LlmRequest) -> int:
        """Compute the next prefill chunk size for a context request when block reuse is enabled.

        When kv_cache_config.enable_block_reuse is True, context prefill must stop exactly at
        the positions returned by calc_context_stop_positions (mamba_state_cache_interval boundaries
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

    def _setup_states(self) -> None:
        # Pool layout: {numLocalLayers, numBlocks, ssm_bytes + conv_bytes} (as uint8)
        pool: torch.Tensor = self.impl.get_recurrent_states_pool().view(
            torch.uint8).reshape(self.local_num_mamba_layers, -1,
                                 self.ssm_bytes + self.conv_bytes)
        num_blocks_in_pool = pool.shape[1]
        self.all_ssm_states = pool[:, :, :self.ssm_bytes].view(
            self.ssm_state_dtype).view(
                [self.local_num_mamba_layers, num_blocks_in_pool] +
                self.ssm_state_shape)
        self.all_conv_states = pool[:, :, self.ssm_bytes:self.ssm_bytes +
                                    self.conv_bytes].view(
                                        self.conv_state_dtype).view([
                                            self.local_num_mamba_layers,
                                            num_blocks_in_pool
                                        ] + self.conv_state_shape)

    def _setup_mtp_intermediate_states(self, spec_config,
                                       max_batch_size) -> None:
        self.spec_config = spec_config
        self.intermediate_ssm_states = None
        self.intermediate_conv_states = None
        self.intermediate_state_indices = None
        if self.spec_config is not None:
            # DFlash/PARD use 2K query tokens per gen, so size by tokens_per_gen_step.
            speculative_num_draft_tokens = self.spec_config.tokens_per_gen_step - 1
            num_local_mamba_layers = len(self.mamba_pp_layers)

            # Legacy SSM intermediate buffer is only needed when replay is
            # disabled; replay reads from the per-block double-buffered cache
            # set up in _setup_replay_buffers instead.
            if not self._use_replay_state_update:
                self.intermediate_ssm_states = torch.zeros(
                    size=[
                        num_local_mamba_layers, max_batch_size,
                        speculative_num_draft_tokens + 1
                    ] + self.ssm_state_shape,
                    dtype=self.ssm_state_dtype,
                    device="cuda",
                )

            self.intermediate_conv_states = torch.zeros(
                size=[
                    num_local_mamba_layers, max_batch_size,
                    speculative_num_draft_tokens + 1
                ] + self.conv_state_shape,
                dtype=self.conv_state_dtype,
                device="cuda",
            )

            self.intermediate_state_indices = torch.arange(max_batch_size,
                                                           dtype=torch.int32,
                                                           device="cuda")

    def _setup_replay_buffers(self, spec_config) -> None:
        """Allocate per-pool-block replay buffers used by replay_selective_state_update.

        Unlike the Mixed cache manager (where slots are 0..max_batch_size-1),
        the unified C++ KV pool assigns recurrent-state block indices up to
        ``num_blocks_in_pool``. The replay kernel indexes ``cache_buf_idx`` and
        ``prev_num_accepted_tokens`` by these block indices, so the buffers
        must match the pool extent rather than ``max_batch_size``.
        """
        if spec_config is None or not self._use_replay_state_update:
            # Replay tensors are sized by the recurrent-state pool block count and
            # are allocated in _setup_replay_buffers after _setup_states().
            self.prev_num_accepted_tokens = None
            self.cache_buf_idx = None
            self.old_x = None
            self.old_B = None
            self.old_dt = None
            self.old_dA_cumsum = None
            return

        T = spec_config.max_draft_len + 1
        num_local_mamba_layers = self.local_num_mamba_layers
        # all_ssm_states: [num_local_mamba_layers, num_blocks_in_pool, ...]
        cache_size = self.all_ssm_states.shape[1]
        nheads, head_dim, d_state = self.ssm_state_shape
        n_groups_per_rank = self._n_groups_per_rank
        device = self.all_ssm_states.device

        # Shared across layers (consumed by the replay kernel via slot index).
        self.prev_num_accepted_tokens = torch.zeros(cache_size,
                                                    dtype=torch.int32,
                                                    device=device)
        self.cache_buf_idx = torch.zeros(cache_size,
                                         dtype=torch.int32,
                                         device=device)
        # x is not double-buffered
        self.old_x = torch.zeros(num_local_mamba_layers,
                                 cache_size,
                                 T,
                                 nheads,
                                 head_dim,
                                 dtype=self.conv_state_dtype,
                                 device=device)
        # Per-layer double-buffered caches.
        self.old_B = torch.zeros(num_local_mamba_layers,
                                 cache_size,
                                 2,
                                 T,
                                 n_groups_per_rank,
                                 d_state,
                                 dtype=self.conv_state_dtype,
                                 device=device)
        self.old_dt = torch.zeros(num_local_mamba_layers,
                                  cache_size,
                                  2,
                                  nheads,
                                  T,
                                  dtype=torch.float32,
                                  device=device)
        self.old_dA_cumsum = torch.zeros(num_local_mamba_layers,
                                         cache_size,
                                         2,
                                         nheads,
                                         T,
                                         dtype=torch.float32,
                                         device=device)

    @property
    def use_replay_state_update(self) -> bool:
        return self._use_replay_state_update

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.ssm_state_dtype
