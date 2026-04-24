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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

import tensorrt_llm.bindings

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager, get_pp_layers)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import prefer_pinned, torch_dtype_to_binding
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
        _SHARED_FIELDS = frozenset({"prev_num_accepted_tokens", "cache_buf_idx"})

        intermediate_conv_window: torch.Tensor  # always allocated

        # Legacy path: full intermediate SSM states at each step
        intermediate_ssm: torch.Tensor | None = None

        # Replay path: compact double-buffered cache
        # prev_num_accepted_tokens: # accepted tokens (always >= 1 if drafting).
        # 0 means temporal saved state is actually the last state, not two back.
        prev_num_accepted_tokens: torch.Tensor | None = None  # (cache,) int — shared across layers
        cache_buf_idx: torch.Tensor | None = None              # (cache,) int32 — shared across layers
        old_x: torch.Tensor | None = None          # (layers, cache, T, nheads, dim)
        old_B: torch.Tensor | None = None          # (layers, cache, 2, T, ngroups, dstate)
        # Processed dt: softplus(raw_dt + dt_bias), clamped to dt_limit.
        old_dt: torch.Tensor | None = None         # (layers, cache, 2, nheads, T) fp32
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
        assert n_groups % tp_size == 0, "n_groups must be divisible by tp_size"

        # partition conv_dim and nheads
        d_inner_local = d_inner // tp_size
        ng_ds_local = n_groups * d_state // tp_size
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size
        n_groups = n_groups // tp_size
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
                size=(num_local_layers, self.spec_state_size,
                      T) + conv_state_shape,
                dtype=dtype,
                device=device,
            )

            # SSM speculative cache — path-specific tensors
            spec_kwargs = {}
            if self._use_replay_state_update:
                # Compact replay cache.
                # old_x is single-buffered (written by main kernel after replay).
                # old_B, old_dt, old_dA_cumsum are double-buffered (written by
                # precompute kernel concurrently with main kernel via PDL).
                spec_kwargs['prev_num_accepted_tokens'] = torch.zeros(
                    max_batch_size, dtype=int, device=device)
                spec_kwargs['cache_buf_idx'] = torch.zeros(
                    max_batch_size, dtype=torch.int32, device=device)
                spec_kwargs['old_x'] = torch.zeros(
                    num_local_layers, max_batch_size, T, nheads, head_dim,
                    dtype=dtype, device=device)
                spec_kwargs['old_B'] = torch.zeros(
                    num_local_layers, max_batch_size, 2, T, n_groups, d_state,
                    dtype=dtype, device=device)
                spec_kwargs['old_dt'] = torch.zeros(
                    num_local_layers, max_batch_size, 2, nheads, T,
                    dtype=torch.float32, device=device)
                spec_kwargs['old_dA_cumsum'] = torch.zeros(
                    num_local_layers, max_batch_size, 2, nheads, T,
                    dtype=torch.float32, device=device)
                ssm_spec_cache = [spec_kwargs['old_x'], spec_kwargs['old_B'],
                                  spec_kwargs['old_dt'], spec_kwargs['old_dA_cumsum']]
                spec_path_label = "replay"
            else:
                # Legacy: full intermediate SSM states at each step
                spec_kwargs['intermediate_ssm'] = torch.zeros(
                    size=(num_local_layers, self.spec_state_size,
                          T) + ssm_state_shape,
                    dtype=self.mamba_ssm_cache_dtype, device=device)
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
                    raise RuntimeError("run out of mamba cache blocks")
                block = self.mamba_cache_free_blocks.pop()
                self.mamba_cache_index[r] = block
                self.state_indices_list.append(block)
                if (isinstance(self.mamba_cache, self.SpeculativeState)
                        and self._use_replay_state_update):
                    self.mamba_cache.prev_num_accepted_tokens[block] = 0
        self.state_indices[:len(self.state_indices_list)].copy_(
            torch.tensor(self.state_indices_list,
                         dtype=torch.int32,
                         pin_memory=prefer_pinned()),
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

    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        from .cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
        request_ids = [
            rid for rid in request_ids if rid != CUDA_GRAPH_DUMMY_REQUEST_ID
        ]
        if request_ids:
            for r in request_ids:
                if r not in self.mamba_cache_index:
                    if len(self.mamba_cache_free_blocks) == 0:
                        raise RuntimeError("run out of mamba cache blocks")
                    block = self.mamba_cache_free_blocks.pop()
                    self.mamba_cache_index[r] = block

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        if request_id in self.mamba_cache_index:
            block = self.mamba_cache_index.pop(request_id)
            self.mamba_cache_free_blocks.append(block)

    def get_state_indices(self, request_ids: List[int],
                          is_padding: List[bool]) -> List[int]:
        assert len(request_ids) == len(is_padding), (
            "request_ids and is_padding must have the same size")

        used_slots = {
            self.mamba_cache_index[req_id]
            for req_id, pad in zip(request_ids, is_padding) if not pad
        }
        available_slots = iter(
            sorted(set(range(self.state_indices.numel())) - used_slots))

        def slot_for(req_id: int, pad: bool):
            if pad:
                try:
                    return next(available_slots)
                except StopIteration:
                    raise RuntimeError(
                        "Run out of available slots for padding") from None
            return self.mamba_cache_index[req_id]

        result = [
            slot_for(rid, pad) for rid, pad in zip(request_ids, is_padding)
        ]
        return result

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
        # Clear state indices
        self.state_indices = torch.tensor([])

        # Clear mamba cache states
        empty = torch.tensor([])
        def _drop(tensor):
            return empty if tensor is not None else None

        if isinstance(self.mamba_cache, self.SpeculativeState):
            self.mamba_cache = self.SpeculativeState(
                conv=empty, temporal=empty,
                intermediate_conv_window=empty,
                intermediate_ssm=_drop(self.mamba_cache.intermediate_ssm),
                prev_num_accepted_tokens=_drop(self.mamba_cache.prev_num_accepted_tokens),
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
                            num_accepted_tokens: torch.Tensor):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = num_accepted_tokens[
            num_contexts:num_contexts + num_gens] - 1
        state_indices_d = self.state_indices[num_contexts:num_contexts +
                                             num_gens]
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
            accepted_ssm_state = intermediate_ssm_cache[:,
                                                        src_state_indices,
                                                        num_accepted_draft_tokens]
            ssm_states[:, state_indices_d, :] = accepted_ssm_state

        # Conv: both paths save all intermediate conv windows, carry over the accepted one.
        conv_states = self.mamba_cache.conv
        intermediate_conv_window_cache = self.mamba_cache.intermediate_conv_window
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
                            num_accepted_tokens: torch.Tensor):
        assert not self._use_cpp, "update_mamba_states is not supported in CppMambaCacheManager"
        self._impl.update_mamba_states(attn_metadata, num_accepted_tokens)


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
        model_type: str = "nemotron_hybrid",
        use_replay_state_update: bool = False,
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
            speculative_num_draft_tokens=(spec_config.max_draft_len
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
