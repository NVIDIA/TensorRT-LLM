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
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._utils import (
    get_size_in_bytes,
    mpi_comm,
    mpi_disabled,
    mpi_rank,
    prefer_pinned,
    torch_comm,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime import ModelConfig as ModelConfigPython
from tensorrt_llm.sampling_params import SamplingParams

from ....logger import logger
from ....mapping import CpType, Mapping
from ..connectors.kv_cache_connector import KvCacheConnectorManager
from ..llm_request import LlmRequest, LlmRequestState, SamplingConfig, get_draft_token_length
from ..scheduler import ScheduledRequests
from .base import BaseResourceManager, request_context
from .kv_cache_spec_ops import _update_kv_cache_draft_token_location, get_pp_layers
from .vswa import (
    calculate_max_num_blocks_for_vswa,
    get_window_size_to_layers,
    validate_and_adjust_attention_windows,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

BufferManagerCpp = tensorrt_llm.bindings.internal.runtime.BufferManager
KVCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
ModelConfigCpp = tensorrt_llm.bindings.ModelConfig
DataType = tensorrt_llm.bindings.DataType
KVCacheEventManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheEventManager

TempAttentionWindowInputs = tensorrt_llm.bindings.internal.batch_manager.TempAttentionWindowInputs
BlocksPerWindow = Dict[int, Tuple[int, int]]


class KVCacheManager(BaseResourceManager):
    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
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
        execution_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.kv_cache_type = kv_cache_type
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.is_draft = is_draft
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {idx: offset for offset, idx in enumerate(self.pp_layers)}

        self.kv_connector_manager = kv_connector_manager

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size for _ in range(self.num_local_layers)
            ]
            self.total_num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size for _ in range(self.num_layers)
            ]
        else:
            assert len(num_kv_heads) == self.num_layers

            def append_to_kv_heads_per_layer(
                num_kv_heads_per_layer: List[int], kv_head: Optional[int]
            ):
                if kv_head is not None:
                    num_kv_heads_per_layer.append((kv_head + tp_size - 1) // tp_size)
                else:
                    num_kv_heads_per_layer.append(0)

            self.num_kv_heads_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    kv_head = num_kv_heads[i]
                    append_to_kv_heads_per_layer(self.num_kv_heads_per_layer, kv_head)

            self.total_num_kv_heads_per_layer = []
            for i in range(self.num_layers):
                kv_head = num_kv_heads[i]
                append_to_kv_heads_per_layer(self.total_num_kv_heads_per_layer, kv_head)

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        from ...speculative import get_num_extra_kv_tokens

        self.num_extra_kv_tokens = get_num_extra_kv_tokens(spec_config)
        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size
        self.attention_dp_events_gather_period_ms = (
            kv_cache_config.attention_dp_events_gather_period_ms
        )
        self.max_num_tokens = max_num_tokens
        self.max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self.max_total_draft_tokens = (
            (spec_config.tokens_per_gen_step - 1) if spec_config is not None else 0
        )

        # Determine max_attention_window_vec
        if kv_cache_config.max_attention_window is None:
            self.max_attention_window_vec = [max_seq_len]
        else:
            self.max_attention_window_vec = kv_cache_config.max_attention_window.copy()
            self.max_attention_window_vec = [
                min(max_seq_len, w) for w in self.max_attention_window_vec
            ]

        sink_token_length = (
            kv_cache_config.sink_token_length
            if kv_cache_config.sink_token_length is not None
            else 0
        )

        # Determine if this is VSWA (Variable Sliding Window Attention)
        self.is_vswa = len(set(self.max_attention_window_vec)) > 1

        # Calculate kv cache blocks for each window size
        if is_estimating_kv_cache:
            self.blocks_in_primary_pool = int(kv_cache_config.max_tokens // tokens_per_block)

            host_cache_size = (
                kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
            )
            max_tokens_secondary = host_cache_size // self.get_cache_bytes_per_token()
            self.blocks_in_secondary_pool = int(max_tokens_secondary // tokens_per_block)

            blocks_per_window = {
                window_size: (self.blocks_in_primary_pool, self.blocks_in_secondary_pool)
                for window_size in set(self.max_attention_window_vec)
            }
            logger.info(
                f"[kv cache manager] Primary/secondary blocks for window sizes "
                f"set to {blocks_per_window} for estimation dry run"
            )
        else:
            if self.is_vswa:
                if model_config is None:
                    raise ValueError(
                        "model_config is required for VSWA (Variable Sliding Window Attention)"
                    )
                assert isinstance(kv_cache_config, KvCacheConfig), (
                    "calculate_max_num_blocks_for_vswa only accepts KvCacheConfig"
                )
                (
                    blocks_per_window,
                    self.max_attention_window_vec,
                    self._primary_pool_memory_bytes,
                    self._secondary_pool_memory_bytes,
                ) = calculate_max_num_blocks_for_vswa(
                    kv_cache_config=kv_cache_config,
                    max_attention_window_vec=self.max_attention_window_vec,
                    num_kv_heads_per_layer=self.num_kv_heads_per_layer,
                    num_local_layers=self.num_local_layers,
                    kv_factor=self.kv_factor,
                    dtype=self.dtype,
                    tokens_per_block=tokens_per_block,
                    is_vswa=self.is_vswa,
                    mapping=self.mapping,
                    model_config=model_config,
                    calculate_scaling_factor_size_bytes_fn=KVCacheManager.calculate_scaling_factor_size_bytes,
                )
                if mapping.world_size > 1:
                    if mpi_disabled():
                        for window_size, (
                            primary_blocks,
                            secondary_blocks,
                        ) in blocks_per_window.items():
                            reduced_primary_blocks = torch_comm().allreduce(
                                primary_blocks, op=torch.distributed.ReduceOp.MIN
                            )
                            reduced_secondary_blocks = torch_comm().allreduce(
                                secondary_blocks, op=torch.distributed.ReduceOp.MIN
                            )
                            blocks_per_window[window_size] = (
                                reduced_primary_blocks,
                                reduced_secondary_blocks,
                            )
                    else:
                        for window_size, (
                            primary_blocks,
                            secondary_blocks,
                        ) in blocks_per_window.items():
                            reduced_primary_blocks = mpi_comm().allreduce(
                                primary_blocks, op=MPI.MIN
                            )
                            reduced_secondary_blocks = mpi_comm().allreduce(
                                secondary_blocks, op=MPI.MIN
                            )
                            blocks_per_window[window_size] = (
                                reduced_primary_blocks,
                                reduced_secondary_blocks,
                            )
                    logger.info(
                        f"[MPI rank={mapping.rank}] Original blocks_per_window: {blocks_per_window}"
                    )
                    logger.info(
                        f"[MPI rank={mapping.rank}] Reduced blocks_per_window: {blocks_per_window}"
                    )
            else:
                self.blocks_in_primary_pool, self.blocks_in_secondary_pool = (
                    self.calculate_max_num_blocks(
                        kv_cache_config=kv_cache_config,
                        head_dim=head_dim,
                        tokens_per_block=tokens_per_block,
                        mapping=mapping,
                        dtype=dtype,
                        kv_factor=self.kv_factor,
                    )
                )
                blocks_per_window = {
                    self.max_attention_window_vec[0]: (
                        self.blocks_in_primary_pool,
                        self.blocks_in_secondary_pool,
                    )
                }

        # Validate and adjust attention windows against their upper bounds
        blocks_per_window, self.max_seq_len, self.max_attention_window_vec = (
            validate_and_adjust_attention_windows(
                max_attention_window_vec=self.max_attention_window_vec,
                blocks_per_window=blocks_per_window,
                tokens_per_block=tokens_per_block,
                sink_token_length=sink_token_length,
                max_seq_len=self.max_seq_len,
                max_beam_width=max_beam_width,
                get_max_atten_window_upper_bound_fn=self.get_max_atten_window_upper_bound,
            )
        )

        if kv_cache_type != CacheTypeCpp.SELF:
            assert len(blocks_per_window) == 1, (
                "Only one window size is supported for non-self KV cache"
            )
            memory_pools = blocks_per_window[self.max_attention_window_vec[0]]
            blocks_per_window = {self.max_seq_len: memory_pools}
            logger.info(
                f"Adjusted attention window size to {self.max_seq_len} in blocks_per_window"
            )

        temp_attention_window_inputs = self._set_temp_attention_window_inputs()

        self._stream = execution_stream if execution_stream is not None else torch.cuda.Stream()
        logger.info(f"[KVCacheManager] execution_stream: {self._stream}")
        kwargs = {
            "num_kv_heads_per_layer": self.num_kv_heads_per_layer,
            "size_per_head": head_dim,
            "tokens_per_block": tokens_per_block,
            "blocks_per_window": blocks_per_window,
            "max_num_sequences": max_batch_size,
            "max_beam_width": max_beam_width,
            "max_attention_window_vec": self.max_attention_window_vec,
            "temp_attention_window_inputs": temp_attention_window_inputs,
            "dtype": dtype,
            "sink_token_length": sink_token_length,
            "stream": self._stream.cuda_stream,
            "max_sequence_length": max_seq_len,
            "enable_block_reuse": kv_cache_config.enable_block_reuse,
            "cache_type": kv_cache_type,
            "enable_partial_reuse": kv_cache_config.enable_partial_reuse,
            "copy_on_partial_reuse": kv_cache_config.copy_on_partial_reuse,
            "kv_connector_manager": self.kv_connector_manager,
            "enable_indexer_k_cache": enable_indexer_k_cache,
            "indexer_k_cache_quant_block_size": indexer_k_cache_quant_block_size,
            "indexer_k_cache_index_head_dim": indexer_k_cache_index_head_dim,
        }

        if self.event_buffer_max_size > 0:
            if mapping.enable_attention_dp:
                kwargs["event_manager"] = KVCacheEventManagerCpp(
                    max_kv_event_entries=self.event_buffer_max_size,
                    attention_dp_rank=mapping.rank,
                    attention_dp_size=mapping.world_size,
                    attention_dp_events_gather_period_ms=self.attention_dp_events_gather_period_ms,
                )
            elif mpi_rank() == 0:
                kwargs["event_manager"] = KVCacheEventManagerCpp(
                    max_kv_event_entries=self.event_buffer_max_size
                )

        self.impl = KVCacheManagerCpp(**kwargs)
        self._warmup_reused_blocks = 0
        self._warmup_missed_blocks = 0

        self.impl.allocate_pools(False)
        self.kv_cache_pool_pointers = self.impl.get_block_pool_pointers()
        kv_cache_block_scale_pool_pointers = self.impl.get_block_scale_pool_pointers()
        if kv_cache_block_scale_pool_pointers.numel() > 0:
            self.kv_cache_pool_pointers = torch.stack(
                [self.kv_cache_pool_pointers, kv_cache_block_scale_pool_pointers], dim=-1
            )

        self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()
        self.num_pools = self.impl.num_pools
        self.max_blocks_per_seq = self.impl.max_blocks_per_seq
        self.enable_block_reuse = kv_cache_config.enable_block_reuse
        self.enable_partial_reuse = kv_cache_config.enable_partial_reuse
        self.host_kv_cache_block_offsets = torch.empty(
            self.num_pools,
            max_batch_size * max_beam_width,
            2,
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )

    def probe_prefix_match_length(self, input_tokens, lora_task_id=None):
        """Probe the KV cache radix tree for prefix match length."""
        if not self.enable_block_reuse:
            return 0
        if getattr(self.impl, "is_variable_window", False):
            return 0
        if not input_tokens:
            return 0
        from tensorrt_llm.bindings import SamplingConfig
        from tensorrt_llm.bindings.internal.batch_manager import BlockKey
        from tensorrt_llm.bindings.internal.batch_manager import LlmRequest as CppLlmRequest

        block_key = BlockKey(tokens=input_tokens, lora_task_id=lora_task_id)
        unique_tokens = block_key.unique_tokens
        dummy_req = CppLlmRequest(
            request_id=0,
            max_new_tokens=0,
            input_tokens=input_tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
            lora_task_id=lora_task_id,
        )
        num_blocks = self.impl.count_reusable_blocks(unique_tokens, dummy_req, False)
        return num_blocks * self.tokens_per_block

    def shutdown(self):
        self.impl.release_pools()

    def get_max_resource_count(self) -> int:
        return self.impl.max_num_blocks

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        context_token_count = request.orig_prompt_len
        num_context_blocks = context_token_count // self.tokens_per_block
        remaining_tokens = (
            context_token_count
            + request.max_new_tokens
            - num_context_blocks * self.tokens_per_block
        )
        need_blocks = num_context_blocks + math.ceil(remaining_tokens / self.tokens_per_block)
        return need_blocks

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        with request_context(self.is_draft, scheduled_batch):
            self.impl.sync_transfer_manager_with_buffer_manager()

            for req in scheduled_batch.context_requests:
                req_beam_width = req.sampling_config.beam_width
                if (
                    "cp_type" in self.mapping.cp_config
                    and CpType.STAR == self.mapping.cp_config["cp_type"]
                ):
                    if req.ctx_iters == 0:
                        seq_len = sum(len(ctx_block) for ctx_block in req.ctx_blocks)
                        self.impl.add_sequence(
                            req.py_request_id,
                            seq_len
                            + (
                                len(req.query_id)
                                if self.mapping.cp_rank == self.mapping.cp_size - 1
                                else 0
                            ),
                            req_beam_width,
                            req,
                        )
                else:
                    if req.is_first_context_chunk and self._kv_connector_should_add_sequence(req):
                        self.impl.add_sequence(
                            req.py_request_id, req.prompt_len, req_beam_width, req
                        )
                        for _ in range(self.num_extra_kv_tokens):
                            self.impl.add_token(req.py_request_id)
                        for _ in range(get_draft_token_length(req)):
                            self.impl.add_token(req.py_request_id)

                        if self.kv_connector_manager is not None:
                            block_ids = self.get_cache_indices(req)
                            self.kv_connector_manager.update_state_after_alloc(req, block_ids)

            scheduled_batch.reset_context_requests()

            for req in scheduled_batch.generation_requests:
                if self.mapping.has_cp_helix():
                    decode_block_id = (req.py_decoding_iter - 1) // self.tokens_per_block
                    if decode_block_id % self.mapping.cp_size == self.mapping.cp_rank:
                        req.py_helix_is_inactive_rank = False
                        req.seqlen_this_rank_cp += 1
                    else:
                        req.py_helix_is_inactive_rank = True
                        continue
                self.impl.add_token(req.py_request_id)
                for _ in range(get_draft_token_length(req)):
                    self.impl.add_token(req.py_request_id)

            self.impl.refresh_blocks()

        if self.kv_connector_manager is not None:
            self.kv_connector_manager.build_scheduler_output(scheduled_batch, self)

    def _kv_connector_should_add_sequence(self, request: LlmRequest) -> bool:
        return self.kv_connector_manager is None or self.kv_connector_manager.should_add_sequence(
            request
        )

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        num_extra_decoding_steps: int = 0,
        draft_kv_cache_manager: Optional[BaseResourceManager] = None,
    ):
        available_blocks = self.get_num_free_blocks()
        if available_blocks < 1:
            return None

        beam_width = max_beam_width
        requests = []
        for i, req_id in enumerate(request_ids):
            sampling_params = SamplingParams(
                n=beam_width, best_of=beam_width, use_beam_search=beam_width > 1
            )
            token_num = token_nums[i] if token_nums is not None else 1 + max_num_draft_tokens
            if self.mapping.has_cp_helix():
                token_num = max(token_num, 2)
            encoder_input_tokens = [1] * token_num if self.impl.cross_kv else None
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=1,
                input_tokens=[1] * token_num,
                sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
                is_streaming=False,
                encoder_input_tokens=encoder_input_tokens,
            )
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            if prepare_resource:
                self.impl.add_sequence(req_id, token_num, beam_width, req)
                for _ in range(self.num_extra_kv_tokens):
                    self.impl.add_token(req_id)

                for _ in range(num_extra_decoding_steps):
                    self.impl.add_token(req_id)

                if draft_kv_cache_manager is not None:
                    draft_kv_cache_manager.impl.add_sequence(req_id, token_num, beam_width, req)
                    for _ in range(self.num_extra_kv_tokens):
                        draft_kv_cache_manager.impl.add_token(req_id)

            if is_gen:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.prompt_len = token_num - 1
                req.py_prompt_len = req.prompt_len
                if self.mapping.has_cp_helix():
                    if self.mapping.cp_size - 1 == self.mapping.cp_rank:
                        req.py_helix_is_inactive_rank = False
                        req.prompt_len = token_num - 1
                        req.py_prompt_len = req.prompt_len
                        req.seqlen_this_rank_cp = req.prompt_len
                        req.total_input_len_cp = token_num * self.mapping.cp_size - 1
                        req.py_decoding_iter = 1
                    else:
                        req.py_helix_is_inactive_rank = True
                        req.prompt_len = token_num
                        req.py_prompt_len = req.prompt_len
                        req.seqlen_this_rank_cp = req.prompt_len
                        req.total_input_len_cp = token_num * self.mapping.cp_size - 1
                        req.py_decoding_iter = 1
                req.py_draft_tokens = [1] * max_num_draft_tokens
                if prepare_resource:
                    for _ in range(max_num_draft_tokens):
                        self.impl.add_token(req_id)

                    if draft_kv_cache_manager is not None:
                        for _ in range(max_num_draft_tokens):
                            draft_kv_cache_manager.impl.add_token(req_id)

            if use_mrope:
                dummy_mrope_position_ids = (
                    torch.arange(0, token_num, dtype=torch.int32).expand(3, 1, -1).clone()
                )
                req.py_multimodal_data = {
                    "mrope_config": {"mrope_position_ids": dummy_mrope_position_ids}
                }
                if is_gen:
                    dummy_mrope_position_deltas = torch.zeros(1, dtype=torch.int32).unsqueeze(0)
                    req.py_multimodal_data["mrope_config"]["mrope_position_deltas"] = (
                        dummy_mrope_position_deltas
                    )
            requests.append(req)
        return requests

    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: "AttentionMetadata" = None,
        kv_cache_dtype_byte_size: float = None,
    ):
        if not self.is_draft:
            _update_kv_cache_draft_token_location(
                self, scheduled_batch, attn_metadata, kv_cache_dtype_byte_size
            )

        # Rewind KV cache for requests with rejected draft tokens.
        # Skip:
        # - GENERATION_COMPLETE: finished requests
        # - CONTEXT_INIT: requests whose state was reset after being paused with KV cache freed.
        #   With overlap scheduler, the scheduler pauses a request and frees KV cache at iteration N,
        #   while the previous batch (N-1) is still trying to update the KV cache after forward pass.
        for request in scheduled_batch.generation_requests:
            if request.state in (LlmRequestState.GENERATION_COMPLETE, LlmRequestState.CONTEXT_INIT):
                continue
            if request.py_rewind_len > 0:
                self.rewind_kv_cache(request, request.py_rewind_len)

        # For context requests, store completed context blocks for KV cache reuse.
        # We wait until context_remaining_length == 0 (all chunks processed) before
        # storing, so that SWA windows are safe to store — blocks won't go out-of-window
        # and be evicted while the context is still in-flight.
        for request in scheduled_batch.context_requests:
            self.impl.store_context_blocks(request)

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        return self.impl.remove_sequence(request.py_request_id, request, pin_on_release)

    def store_blocks_for_reuse(self, request: LlmRequest, pin_blocks: bool = False):
        return self.impl.store_blocks_for_reuse(request.py_request_id, request, pin_blocks)

    @staticmethod
    def calculate_scaling_factor_size_bytes(
        cache_size: int, quant_vector_size: int, scaling_factor_dtype: DataType
    ) -> int:
        assert cache_size % quant_vector_size == 0, (
            "NVFP4 cache size must be divisible by quant vector size"
        )
        return get_size_in_bytes(cache_size // quant_vector_size, scaling_factor_dtype)

    @staticmethod
    def _resolve_num_attention_layers(
        model_config: ModelConfigPython,
        mapping: Mapping,
        num_layers: Optional[int] = None,
    ) -> int:
        if num_layers is not None:
            return max(num_layers, 1)
        return max(len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)

    @staticmethod
    def get_cache_size_per_token(
        model_config: ModelConfigPython,
        mapping: Mapping,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        config = model_config.pretrained_config
        num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        if isinstance(num_key_value_heads, Iterable):
            num_key_value_heads = sum(num_key_value_heads) / len(num_key_value_heads)

        mla = hasattr(config, "kv_lora_rank")
        if mla:
            head_dim = config.kv_lora_rank + config.qk_rope_head_dim
            kv_factor = 1
        else:
            tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
            head_dim = getattr(config, "head_dim", None)
            if not isinstance(head_dim, int):
                head_dim = config.hidden_size // config.num_attention_heads
            head_dim = head_dim * num_key_value_heads // tp_size
            kv_factor = 2

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers
        )
        mem_per_token = kv_factor * num_attention_layers * head_dim
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
            mem_per_token *= 1
        elif quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache():
            mem_per_token = math.ceil(mem_per_token / 2) + math.ceil(mem_per_token / 16)
        else:
            assert quant_config is None or (not quant_config.quant_mode.has_kv_cache_quant()), (
                "Quantized kv cache is not expected"
            )
            mem_per_token *= 2
        return mem_per_token

    def get_cache_bytes_per_token(self):
        cache_size_per_token = self.kv_factor * sum(self.num_kv_heads_per_layer) * self.head_dim

        if self.dtype not in (
            DataType.FP8,
            DataType.HALF,
            DataType.BF16,
            DataType.FLOAT,
            DataType.NVFP4,
        ):
            raise ValueError(f"Cannot support {self.dtype} KV cache.")

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token, self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token, quant_vector_size=16, scaling_factor_dtype=DataType.FP8
            )
        return cache_size_bytes_per_token

    def _get_window_size_to_layers(self) -> Dict[int, List[int]]:
        return get_window_size_to_layers(self.max_attention_window_vec, self.num_local_layers)

    def calculate_max_num_blocks(
        self,
        kv_cache_config: KvCacheConfig,
        head_dim: int,
        tokens_per_block: int,
        mapping: Mapping,
        dtype: DataType,
        kv_factor: int = 2,
    ):
        free_mem_fraction = (
            kv_cache_config.free_gpu_memory_fraction
            if kv_cache_config.free_gpu_memory_fraction is not None
            else 0.9
        )

        cache_size_bytes_per_token = self.get_cache_bytes_per_token()

        free_mem, total_mem = torch.cuda.mem_get_info()

        assert free_mem_fraction < 1.0, (
            f"Invalid freeMemFraction, freeMemFraction {free_mem_fraction} must be smaller than 1.0"
        )
        max_tokens = free_mem_fraction * free_mem / cache_size_bytes_per_token

        if kv_cache_config.max_tokens is not None:
            if kv_cache_config.free_gpu_memory_fraction is not None:
                max_tokens = min(kv_cache_config.max_tokens, max_tokens)
                logger.warning(
                    f"Both free_gpu_memory_fraction and max_tokens are set "
                    f"(to {free_mem_fraction} and {max_tokens} with free "
                    f"memory {free_mem / (1 << 30)}GiB of total memory "
                    f"{total_mem / (1 << 30)}GiB, respectively). "
                    f"The smaller value will be used."
                )
            else:
                max_tokens = kv_cache_config.max_tokens
                logger.info(f"max_tokens is set by kv_cache_config.max_tokens: {max_tokens}")

        if mapping.world_size > 1:
            dist = Distributed.get(mapping)
            max_tokens = dist.allreduce(
                max_tokens,
                op=ReduceOp.MIN,
            )

        blocks_in_primary_pool = int(max_tokens // tokens_per_block)

        host_cache_size = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
        max_tokens_secondary = host_cache_size // self.get_cache_bytes_per_token()
        blocks_in_secondary_pool = int(max_tokens_secondary // tokens_per_block)

        return blocks_in_primary_pool, blocks_in_secondary_pool

    def get_max_atten_window_upper_bound(
        self,
        blocks_in_primary_pool,
        tokens_per_block,
        max_beam_width,
        sink_token_len,
        max_seq_len: Optional[int],
    ):
        token_capacity = blocks_in_primary_pool * tokens_per_block
        max_blocks_per_seq = math.floor(token_capacity / (max_beam_width * tokens_per_block))
        assert max_blocks_per_seq > 0, "Impossible to fit in any sequence in kvCache"

        max_token_num = max_blocks_per_seq * tokens_per_block
        sink_tokens_in_last_block = sink_token_len % tokens_per_block
        sink_bubble_len = (
            0 if sink_tokens_in_last_block == 0 else tokens_per_block - sink_tokens_in_last_block
        )
        max_atten_window_upper_bound = max_token_num - sink_bubble_len
        if (
            max_seq_len is not None
            and max_seq_len > max_atten_window_upper_bound
            and max_beam_width > 1
        ):
            max_atten_window_upper_bound -= tokens_per_block
        assert max_atten_window_upper_bound > 0, "Impossible to fit in any sequence in kvCache"
        return max_atten_window_upper_bound

    def get_cache_indices(
        self, request: LlmRequest, window_size: Optional[int] = None
    ) -> List[int]:
        if window_size is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("window_size must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]

        result = self.impl.get_cache_block_ids(request.py_request_id, window_size)
        assert len(result) == 1
        return result[0]

    def unpin_blocks_by_id(self, kv_cache_block_id: int):
        self.impl.unpin_blocks_by_id(kv_cache_block_id)

    def get_last_block_id(self, request_id: int) -> int:
        return self.impl.get_last_block_id(request_id)

    def get_priority_by_block_id(self, block_id: int, window_size: Optional[int] = None) -> int:
        if window_size is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("window_size must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]
        return self.impl.get_priority_by_block_id(block_id, window_size)

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
    ) -> List[List[int]]:
        if layer_idx is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("layer_idx must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]
        else:
            layer_offset = self.layer_offsets[layer_idx]
            window_size = self.max_attention_window_vec[
                layer_offset % len(self.max_attention_window_vec)
            ]

        result = self.impl.get_batch_cache_block_ids(request_ids, window_size)
        for i in range(len(result)):
            assert (len(result[i])) == 1
            result[i] = result[i][0]
        return result

    def get_num_free_blocks(self) -> int:
        if self.is_vswa:
            stats = self.impl.get_kv_cache_stats()
            logger.info(
                f"For VSWA case, we return the minimum of the number of "
                f"free blocks for each window size: "
                f"{stats.num_free_blocks_per_window_size}"
            )
            return min(self.impl.get_kv_cache_stats().num_free_blocks_per_window_size.values())
        else:
            return self.impl.get_kv_cache_stats().free_num_blocks

    def get_num_kv_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block

    def get_num_available_tokens(
        self, token_num_upper_bound: int, max_num_draft_tokens: int = 0, **kwargs
    ) -> int:
        return min(
            token_num_upper_bound,
            self.get_num_free_blocks() * self.tokens_per_block
            - self.num_extra_kv_tokens
            - max_num_draft_tokens,
        )

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        layer_offset = self.layer_offsets[layer_idx]
        result = self.impl.get_primary_pool_data(layer_offset)

        assert kv_layout in ["NHD", "HND"], f"Unsupported kv_layout: {kv_layout}"
        if kv_layout == "NHD":
            return result.reshape(
                result.shape[0],
                self.kv_factor,
                self.tokens_per_block,
                self.num_kv_heads_per_layer[layer_offset],
                self.head_dim,
            )
        else:
            return result.reshape(
                result.shape[0],
                self.kv_factor,
                self.num_kv_heads_per_layer[layer_offset],
                self.tokens_per_block,
                self.head_dim,
            )

    def get_indexer_k_cache_pool_data(self, layer_idx: int) -> torch.Tensor:
        result = self.impl.get_indexer_k_cache_pool_data(layer_idx)
        return result.view(result.shape[0], -1)

    def check_invalid_values_in_kv_cache(self, fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor(
            [False], dtype=torch.bool, device=torch.cuda.current_device()
        )
        for layer_idx, layer_offset in self.layer_offsets.items():
            buffer = self.impl.get_primary_pool_data(layer_offset)
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i : i + 256]
                try:
                    has_invalid_values.logical_or_(torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for "
                "current kv cache dtype, related checks are skipped"
            )
        return bool(has_invalid_values)

    def get_unique_primary_pool(self) -> torch.Tensor:
        return self.impl.get_unique_primary_pool()

    def get_block_ids_per_seq(self, request_ids: List[int]) -> torch.Tensor:
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        block_ids_per_seq_tensors = [
            torch.tensor(sublist, dtype=torch.int) for sublist in block_ids_per_seq
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            block_ids_per_seq_tensors, batch_first=True, padding_value=0
        )
        return padded_tensor

    def flush_iteration_events(self):
        self.impl.flush_iteration_events()

    def get_latest_events(self, timeout_ms: Optional[float] = 0):
        return self.impl.get_latest_events(timeout_ms)

    def get_kv_cache_stats(self):
        stats = self.impl.get_kv_cache_stats()
        stats.reused_blocks -= self._warmup_reused_blocks
        stats.missed_blocks -= self._warmup_missed_blocks
        total = stats.reused_blocks + stats.missed_blocks
        stats.cache_hit_rate = (stats.reused_blocks / total) if total > 0 else 0.0
        return stats

    def snapshot_warmup_baseline(self):
        raw = self.impl.get_kv_cache_stats()
        self._warmup_reused_blocks = raw.reused_blocks
        self._warmup_missed_blocks = raw.missed_blocks

    def get_iteration_stats(self):
        """Get per-iteration KV cache stats keyed by window size. Resets deltas on each call."""
        return self.impl.get_iteration_stats()

    def rewind_kv_cache(self, request: LlmRequest, rewind_len: int):
        self.impl.rewind_kv_cache(request.py_request_id, rewind_len)

    def pin_blocks(self, request_id: int):
        self.impl.pin_blocks(request_id)

    def _set_temp_attention_window_inputs(self) -> Optional[TempAttentionWindowInputs]:
        is_sliding_window = min(self.max_attention_window_vec) < self.max_seq_len
        if is_sliding_window:
            temp_attention_window_inputs = TempAttentionWindowInputs()
            temp_attention_window_inputs.paged_context_fmha = True
            temp_attention_window_inputs.max_input_len = self.max_seq_len - 1
            temp_attention_window_inputs.max_num_tokens = self.max_num_tokens
            return temp_attention_window_inputs
        else:
            return None

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_context: int,
        num_seqs: int,
    ):
        self.impl.copy_batch_block_offsets(
            self.host_kv_cache_block_offsets, request_ids[:num_context], 1, 0
        )
        self.impl.copy_batch_block_offsets(
            self.host_kv_cache_block_offsets, request_ids[num_context:], beam_width, num_context
        )

        for pool_idx in range(self.host_kv_cache_block_offsets.shape[0]):
            dst_tensor[pool_idx, :num_seqs].copy_(
                self.host_kv_cache_block_offsets[pool_idx, :num_seqs], non_blocking=True
            )

    def reset_reuse_state(self):
        """Reset the reuse state of the KV cache manager."""
        self.impl.reset_reuse_state()
