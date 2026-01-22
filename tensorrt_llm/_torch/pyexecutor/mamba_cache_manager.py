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

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager, get_pp_layers)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata
    from tensorrt_llm._torch.pyexecutor.sampler import SampleState

GB = 1 << 30
def get_tensor_size_bytes(tensor):
    """Calculate tensor size in bytes."""
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement()
    elif isinstance(tensor, list):
        return sum(get_tensor_size_bytes(t) for t in tensor)
    return 0


class MambaCacheManager(BaseResourceManager):

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

        def mem_usage_bytes(self):
            """Calculate total memory usage in bytes."""
            total = 0
            for f in dataclass_fields(self):
                tensor = getattr(self, f.name)
                if isinstance(tensor, torch.Tensor):
                    total += tensor.element_size() * tensor.nelement()
            return total

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

            # Cache intermediate SSM states per draft token(include new sampled token) during target verify
            intermediate_ssm_states = torch.zeros(
                size=(num_local_layers, self.spec_state_size,
                      speculative_num_draft_tokens + 1) + ssm_state_shape,
                dtype=self.mamba_ssm_cache_dtype,
                device=device,
            )

            # Cache intermediate conv windows per draft token(include new sampled token) during target verify
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
                         pin_memory=True),
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
                                   pin_memory=True).to(
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
                                   pin_memory=True).to(
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

    def get_state_indices(self) -> torch.Tensor:
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

    def _update_speculative_mamba_states(
            self, num_accepted_draft_tokens: torch.Tensor,
            state_indices_tensor: torch.Tensor):
        """Update Mamba states after MTP verification

        Copy the states of accepted speculative steps from the intermediate cache to the main cache.

        Args:
            num_accepted_draft_tokens: Number of accepted steps for each request, shape [request_number]
                          e.g.: [2, -1, 3, 1] means request 0 accepts 2 steps, request 1 rejects (-1),
                                request 2 accepts 3 steps, request 3 accepts 1 step
            state_indices_tensor: Main cache index for each request, shape [request_number]
                                e.g.: [5, 12, 3, 7] means 4 requests use main cache slots 5, 12, 3, 7

        """
        request_number = num_accepted_draft_tokens.shape[0]
        intermediate_state_indices = torch.arange(
            request_number,
            dtype=torch.int32,
            device=state_indices_tensor.device)

        conv_states = self.mamba_cache.conv
        ssm_states = self.mamba_cache.temporal

        intermediate_state_cache = self.mamba_cache.intermediate_ssm
        intermediate_conv_window_cache = self.mamba_cache.intermediate_conv_window

        valid_mask = num_accepted_draft_tokens >= 0

        # if no valid accepted tokens (all -1), return
        if not valid_mask.any():
            return

        dst_state_indices = state_indices_tensor[valid_mask].to(torch.int64)
        src_state_indices = intermediate_state_indices[valid_mask].to(
            torch.int64)
        last_steps = num_accepted_draft_tokens[valid_mask].to(torch.int64)

        accepted_ssm_state = intermediate_state_cache[:, src_state_indices,
                                                      last_steps]
        ssm_states[:, dst_state_indices, :] = accepted_ssm_state.to(
            ssm_states.dtype, copy=False)

        accepted_conv_state = intermediate_conv_window_cache[:,
                                                             src_state_indices,
                                                             last_steps]
        conv_states[:, dst_state_indices, :] = accepted_conv_state.to(
            conv_states.dtype, copy=False)

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None,
                         sample_state: Optional["SampleState"] = None):
        """Update Mamba cache resources

        After the verification phase of speculative decoding, update the intermediate cache state to the main cache.

        Args:
            scheduled_batch: Scheduled batch requests
            attn_metadata: Attention metadata
            kv_cache_dtype_byte_size: KV cache data type byte size
            sample_state: Sample state from previous iteration, contains new_tokens_lens for MTP
        """

        if not self.is_speculative():
            return
        num_decodes = len(scheduled_batch.generation_requests)

        if num_decodes == 0:
            return

        num_accepted_draft_tokens = []
        state_indices_updated = []
        if sample_state is not None:
            # for overlap scheduler, use sample_state to get num_accepted_draft_tokens for previous iteration because sampler update is not called when this function is called
            valid_slots = []
            valid_cache_indices = []

            for req in scheduled_batch.generation_requests:
                if req.state != LlmRequestState.GENERATION_COMPLETE:
                    valid_slots.append(req.py_seq_slot)
                    valid_cache_indices.append(
                        self.mamba_cache_index[req.py_request_id])
                else:
                    valid_slots.append(-1)
                    valid_cache_indices.append(-1)

            if len(valid_cache_indices) == 0:
                return

            # Perform all operations on GPU to avoid synchronization overhead
            slots_tensor = torch.tensor(valid_slots,
                                        dtype=torch.int32,
                                        device='cuda')
            state_indices_updated = torch.tensor(valid_cache_indices,
                                                 dtype=torch.int32,
                                                 device='cuda')

            # Use device version of new_tokens_lens and perform indexing on GPU
            valid_mask = slots_tensor >= 0
            num_accepted_draft_tokens = torch.full((len(valid_slots), ),
                                                   -1,
                                                   dtype=torch.int32,
                                                   device='cuda')
            if valid_mask.any():
                num_accepted_draft_tokens[
                    valid_mask] = sample_state.device.new_tokens_lens[
                        slots_tensor[valid_mask]] - 1
        else:
            for req in scheduled_batch.generation_requests:
                if req.state != LlmRequestState.GENERATION_COMPLETE:

                    num_accepted_draft_tokens.append(
                        req.py_num_accepted_draft_tokens)
                    state_indices_updated.append(
                        self.mamba_cache_index[req.py_request_id])
                else:
                    num_accepted_draft_tokens.append(-1)
                    state_indices_updated.append(-1)

            if len(state_indices_updated) == 0:
                return

            num_accepted_draft_tokens = torch.tensor(num_accepted_draft_tokens,
                                                     dtype=torch.int32,
                                                     device='cuda')
            state_indices_updated = torch.tensor(state_indices_updated,
                                                 dtype=torch.int32,
                                                 device='cuda')
        self._update_speculative_mamba_states(num_accepted_draft_tokens,
                                              state_indices_updated)


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
            max_batch_size,
            mapping,
            mamba_cache_dtype,
            mamba_ssm_cache_dtype,
            mamba_layer_mask,
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

    def shutdown(self):
        MambaCacheManager.shutdown(self)
        KVCacheManager.shutdown(self)

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None,
                         update_mamba_cache_manager: bool = True):
        # for non-overlap scheduler, update mamba cache manager and kv cache manager
        # for overlap scheduler, only updtae kv cache manager, and will call update_resources_for_mamba_cache_manager to update mamba cache manager
        if update_mamba_cache_manager:
            MambaCacheManager.update_resources(
                self,
                scheduled_batch,
                attn_metadata,
                kv_cache_dtype_byte_size,
            )
        KVCacheManager.update_resources(self, scheduled_batch, attn_metadata,
                                        kv_cache_dtype_byte_size)

    def update_resources_for_mamba_cache_manager(
            self, scheduled_batch: ScheduledRequests,
            sample_state: "SampleState"):
        MambaCacheManager.update_resources(self,
                                           scheduled_batch,
                                           sample_state=sample_state)
