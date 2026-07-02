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
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple, Union

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    prefer_pinned,
)
from tensorrt_llm.bindings.internal.batch_manager import KvCacheStats
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    IndexMapper,
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime import ModelConfig as ModelConfigPython

# isort: off
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    DEFAULT_BEAM_INDEX,
    AttentionLayerConfig,
    BufferConfig,
    CacheTierConfig,
    GpuCacheTierConfig,
    HostCacheTierConfig,
)

# isort: on
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager as KVCacheManagerPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId, TokenIdExt, _KVCache
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX, GPU_LEVEL
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import exact_div, typed_range
from tensorrt_llm.sampling_params import SamplingParams

from ...._utils import nvtx_range
from ....logger import logger
from ....mapping import CpType, Mapping
from ..connectors.kv_cache_connector import KvCacheConnectorManager
from ..llm_request import LlmRequest, LlmRequestState, SamplingConfig, get_draft_token_length
from ..scheduler import ScheduledRequests
from .base import BaseResourceManager, Role, request_context
from .kv_cache_spec_ops import _update_kv_cache_draft_token_location, get_pp_layers

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata

CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
ModelConfigCpp = tensorrt_llm.bindings.ModelConfig
DataType = tensorrt_llm.bindings.DataType


class KVCacheManagerV2(BaseResourceManager):
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
        spec_config=None,
        layer_mask: Optional[List[bool]] = None,
        vocab_size: int = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfigCpp] = None,
        max_beam_width: int = 1,
        is_draft: bool = False,
        kv_connector_manager: Optional[KvCacheConnectorManager] = None,
        execution_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype

        assert kv_connector_manager is None, (
            "kv_connector_manager is not supported for KVCacheManagerV2"
        )
        assert max_beam_width == 1, "max_beam_width must be 1 for KVCacheManagerV2"
        assert not (mapping.cp_config.get("cp_type") == CpType.STAR), (
            "Star attention is not supported for KVCacheManagerV2"
        )

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
        self.max_beam_width = max_beam_width

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        from ...speculative import get_num_extra_kv_tokens

        self.num_extra_kv_tokens = get_num_extra_kv_tokens(spec_config)
        self.max_total_draft_tokens = (
            spec_config.max_total_draft_tokens if spec_config is not None else 0
        )

        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size

        assert self.event_buffer_max_size == 0, "event_buffer_max_size must be 0"

        self._stream = (
            execution_stream if execution_stream is not None else torch.cuda.current_stream()
        )
        logger.info(f"[KVCacheManager] execution_stream: {self._stream}")

        # Determine max_attention_window_vec
        if kv_cache_config.max_attention_window is not None:
            self.max_attention_window_vec = kv_cache_config.max_attention_window.copy()
            self.max_attention_window_vec = [
                min(max_seq_len, w) for w in self.max_attention_window_vec
            ]

            self.max_attention_window_vec = [
                None if w == max_seq_len else w for w in self.max_attention_window_vec
            ]

        else:
            self.max_attention_window_vec = [None]

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

        self.is_vswa = len(set(self.max_attention_window_vec)) > 1

        quota = float("inf")
        if (
            kv_cache_config.max_gpu_total_bytes is not None
            and kv_cache_config.max_gpu_total_bytes > 0
        ):
            quota = int(kv_cache_config.max_gpu_total_bytes)
            logger.info(f"max_gpu_total_bytes is provided. New quota is {quota / (1 << 30)}GiB")
        if kv_cache_config.max_tokens is not None:
            quota_from_max_tokens = int(
                math.ceil(
                    self._get_quota_from_max_tokens(kv_cache_config.max_tokens)
                    / kv_cache_config.max_util_for_resume
                )
            )
            quota = min(quota, quota_from_max_tokens)
            logger.info(
                f"max_tokens {kv_cache_config.max_tokens} is provided. "
                f"Allowed quota from max_tokens is "
                f"{quota_from_max_tokens / (1 << 30)}GiB. "
                f"New quota is {quota / (1 << 30)}GiB"
            )

        assert quota != float("inf"), (
            "Quota not set. Check kv_cache_config.max_tokens or kv_cache_config.max_gpu_total_bytes"
        )
        logger.info(f"KV cache manager v2 device quota set to {quota / (1 << 30)}GiB")

        cache_tiers: List[CacheTierConfig] = [GpuCacheTierConfig(quota=quota)]
        if kv_cache_config.host_cache_size is not None and kv_cache_config.host_cache_size > 0:
            cache_tiers.append(HostCacheTierConfig(quota=kv_cache_config.host_cache_size))
            logger.info(
                f"KV cache manager v2 host cache quota set to {kv_cache_config.host_cache_size / (1 << 30)}GiB"
            )

        config = self._build_cache_config(
            kv_cache_config,
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
        )

        self.kv_cache_manager_py_config = config

        self.impl = KVCacheManagerPy(config)

        self.num_pools = len(self.impl.layer_grouping)

        num_layers = len(config.layers)
        self.layer_to_pool_mapping_dict: dict[int, int] = {
            layer_id: self.impl.get_layer_group_id(layer_id)
            for layer_id in typed_range(LayerId(num_layers))
        }

        (self.kv_cache_pool_pointers, self.kv_cache_pool_mapping) = (
            self._build_pool_mapping_tensors()
        )

        self.max_blocks_per_seq = (max_seq_len + tokens_per_block - 1) // tokens_per_block
        if self.max_blocks_per_seq % 4 != 0:
            self.max_blocks_per_seq = ((self.max_blocks_per_seq + 3) // 4) * 4

        self.kv_cache_map: dict[int, _KVCache] = {}

        self._gpu_max_tokens = kv_cache_config.max_tokens

        max_num_tokens = self.get_num_available_tokens(token_num_upper_bound=max_seq_len)

        if max_seq_len > max_num_tokens:
            logger.warning(
                f"max_seq_len {max_seq_len} is greater than max_num_tokens "
                f"{max_num_tokens} that can be allocated in kv cache manager, "
                f"setting max_seq_len to {max_num_tokens}"
            )
            self.max_seq_len = max_num_tokens

        self.enable_block_reuse = kv_cache_config.enable_block_reuse
        self.enable_partial_reuse = kv_cache_config.enable_partial_reuse

        max_num_sequences = max_batch_size * mapping.pp_size
        self.index_mapper = IndexMapper(max_num_sequences + 1, max_beam_width)
        self.index_scales = torch.empty(
            self.num_pools, dtype=torch.int32, pin_memory=prefer_pinned(), device="cpu"
        )
        self.kv_offset = torch.empty(
            self.num_pools, dtype=torch.int32, pin_memory=prefer_pinned(), device="cpu"
        )
        for pool_id in range(self.num_pools):
            layer_id = self.impl.layer_grouping[pool_id][0]
            self.index_scales[pool_id] = self.impl.get_page_index_scale(layer_id, Role.KEY)
            if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
                self.kv_offset[pool_id] = exact_div(
                    self.impl.get_mem_pool_base_address(layer_id, Role.VALUE)
                    - self.impl.get_mem_pool_base_address(layer_id, Role.KEY),
                    self.impl.get_page_stride(layer_id, Role.KEY),
                )
            else:
                self.kv_offset[pool_id] = 0

        self.host_kv_cache_block_offsets = torch.empty(
            self.num_pools,
            (max_num_sequences + 1) * max_beam_width,
            2,  # key and value
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )

    def _get_quota_from_max_tokens(self, max_tokens: int) -> int:
        return int(max_tokens * self.get_cache_bytes_per_token())

    def _build_pool_mapping_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_cache_pool_pointers = torch.tensor(
            [
                [
                    self.impl.get_mem_pool_base_address(
                        self.impl.layer_grouping[pool_id][0], Role.KEY
                    ),
                    0,
                ]
                for pool_id in range(self.num_pools)
            ],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

        if self.dtype == DataType.NVFP4:
            kv_cache_pool_pointers = torch.stack(
                [
                    kv_cache_pool_pointers,
                    torch.tensor(
                        [
                            [
                                self.impl.get_mem_pool_base_address(
                                    self.impl.layer_grouping[pool_id][0], Role.KEY_BLOCK_SCALE
                                ),
                                0,
                            ]
                            for pool_id in range(self.num_pools)
                        ],
                        dtype=torch.int64,
                        device="cpu",
                        pin_memory=prefer_pinned(),
                    ),
                ],
                dim=-1,
            )

        kv_cache_pool_mapping_list = []
        for layer_id in typed_range(LayerId(self.num_local_layers)):
            layer_group_id = self.impl.get_layer_group_id(layer_id)
            if self.dtype != DataType.NVFP4:
                addr_offset = self.impl.get_mem_pool_base_address(layer_id, Role.KEY) - int(
                    kv_cache_pool_pointers[layer_group_id][0]
                )
            else:
                addr_offset = self.impl.get_mem_pool_base_address(layer_id, Role.KEY) - int(
                    kv_cache_pool_pointers[layer_group_id][0][0]
                )
                block_scale_addr_offset = self.impl.get_mem_pool_base_address(
                    layer_id, Role.KEY_BLOCK_SCALE
                ) - int(kv_cache_pool_pointers[layer_group_id][0][1])
                block_scale_offset = exact_div(
                    block_scale_addr_offset,
                    self.get_layer_bytes_per_token(layer_id, Role.KEY_BLOCK_SCALE)
                    * self.kv_factor
                    * self.tokens_per_block,
                )
            offset = exact_div(
                addr_offset,
                self.get_layer_bytes_per_token(layer_id, Role.KEY)
                * self.kv_factor
                * self.tokens_per_block,
            )

            if self.dtype == DataType.NVFP4:
                assert block_scale_offset == offset, (
                    "Block scale offset and offset should be the same"
                )

            kv_cache_pool_mapping_list.append([layer_group_id, offset])

        kv_cache_pool_mapping = torch.tensor(
            kv_cache_pool_mapping_list, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        return kv_cache_pool_pointers, kv_cache_pool_mapping

    def _build_cache_config(
        self,
        kv_cache_config: KvCacheConfig,
        *,
        tokens_per_block: int,
        vocab_size: Optional[int],
        cache_tiers: List[CacheTierConfig],
    ) -> KVCacheManagerConfigPy:
        buffer_type = [Role.KEY]
        if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
            buffer_type.append(Role.VALUE)
        if kv_cache_config.dtype == "nvfp4":
            assert self.head_dim % 2 == 0, "head_dim must be divisible by 2 for nvfp4 kv cache"
            buffer_type.append(Role.KEY_BLOCK_SCALE)
            if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
                buffer_type.append(Role.VALUE_BLOCK_SCALE)

        return KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            layers=[
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(
                            role=role,
                            size=self.get_layer_bytes_per_token(
                                local_layer_idx=layer_id, data_role=role
                            )
                            * tokens_per_block,
                        )
                        for role in buffer_type
                    ],
                    sliding_window_size=self.max_attention_window_vec[
                        layer_id % len(self.max_attention_window_vec)
                    ],
                    num_sink_tokens=None,
                )
                for layer_id in typed_range(LayerId(self.num_local_layers))
            ],
        )

    @property
    def blocks_in_primary_pool(self) -> int:
        return self.impl.get_page_index_upper_bound(0, Role.KEY)

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
        if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
            addr_value = self.impl.get_mem_pool_base_address(layer_offset, Role.VALUE)
            page_size_key = self.impl.get_page_stride(layer_offset, Role.KEY)
            page_size_value = self.impl.get_page_stride(layer_offset, Role.VALUE)

            assert addr_key + page_size_value == addr_value and page_size_key == page_size_value

        assert kv_layout in ["NHD", "HND"], f"Unsupported kv_layout: {kv_layout}"

        element_per_container = 1
        dtype = self.dtype
        if dtype == DataType.NVFP4:
            element_per_container = 2
            dtype = torch.int8

        if kv_layout == "NHD":
            shape = [
                self.impl.get_page_index_upper_bound(layer_offset, Role.KEY) // self.kv_factor,
                self.kv_factor,
                self.tokens_per_block,
                self.num_kv_heads_per_layer[layer_offset],
                self.head_dim // element_per_container,
            ]
        else:
            shape = [
                self.impl.get_page_index_upper_bound(layer_offset, Role.KEY) // self.kv_factor,
                self.kv_factor,
                self.num_kv_heads_per_layer[layer_offset],
                self.tokens_per_block,
                self.head_dim // element_per_container,
            ]

        return convert_to_torch_tensor(
            TensorWrapper(
                addr_key,
                dtype,
                shape,
            )
        )

    def get_num_available_tokens(
        self, *, token_num_upper_bound: int, batch_size: int = 1, max_num_draft_tokens: int = 0
    ) -> int:
        extra_tokens = self.num_extra_kv_tokens + max_num_draft_tokens
        clamped = (
            self.impl.clamp_max_seq_len_for_mem(batch_size, token_num_upper_bound + extra_tokens)
            - extra_tokens
        )
        if self._gpu_max_tokens is not None:
            clamped = min(clamped, self._gpu_max_tokens)
        return clamped

    def get_num_free_blocks(self) -> int:
        assert len(self.kv_cache_map) == 0, (
            "get_num_free_blocks is only used when the kv cache manager is empty"
        )
        max_num_pages = max(
            [
                self.impl.get_page_index_upper_bound(layer_id, Role.KEY)
                for layer_id in typed_range(LayerId(self.num_local_layers))
            ]
        )
        return max_num_pages // self.kv_factor

    # ---- Scheduling API (called by KVCacheV2Scheduler) ----

    def is_request_active(self, request_id: int) -> bool:
        kv_cache = self.kv_cache_map.get(request_id)
        return kv_cache is not None and kv_cache.is_active

    def _required_gen_capacity(self, req: LlmRequest, current_capacity: int) -> int:
        draft_len = get_draft_token_length(req)
        return current_capacity + 1 + draft_len

    def try_allocate_generation(self, req: LlmRequest) -> bool:
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return False

        if not kv_cache.is_active:
            if not kv_cache.resume(self._stream.cuda_stream):
                return False
            self._restore_page_index_bufs(req.py_request_id, kv_cache)

        return kv_cache.resize(self._required_gen_capacity(req, kv_cache.capacity))

    def _restore_page_index_bufs(self, request_id: int, kv_cache) -> None:
        index = self.index_mapper.get_index(request_id)
        for i in range(self.max_beam_width):
            for pool_idx in range(self.num_pools):
                buffer: torch.Tensor = self.host_kv_cache_block_offsets[
                    pool_idx, index * self.max_beam_width + i, 0
                ]
                kv_cache.set_base_page_index_buf(i, pool_idx, memoryview(buffer.numpy()))

    def _resume_and_restore(self, req_id: int, kv_cache) -> bool:
        if kv_cache.is_active:
            return True
        if not kv_cache.resume(self._stream.cuda_stream):
            return False
        self._restore_page_index_bufs(req_id, kv_cache)
        return True

    def prepare_context(self, req: LlmRequest) -> bool:
        if req.is_first_context_chunk:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None:
                kv_cache = self._create_kv_cache(
                    req.py_request_id,
                    req.lora_task_id,
                    req.get_tokens(DEFAULT_BEAM_INDEX)[:-1] if self.enable_block_reuse else None,
                )
                kv_cache.cuda_stream = self._stream.cuda_stream

            if not self.enable_block_reuse:
                kv_cache.stop_committing()
            else:
                req.context_current_position = kv_cache.num_committed_tokens
                req.set_prepopulated_prompt_len(
                    kv_cache.num_committed_tokens, self.tokens_per_block
                )

            return self._resume_and_restore(req.py_request_id, kv_cache)
        else:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            assert kv_cache is not None, (
                f"KV cache missing for non-first context chunk, request {req.py_request_id}"
            )
            return self._resume_and_restore(req.py_request_id, kv_cache)

    def resize_context(self, req: LlmRequest, num_tokens: int) -> bool:
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return False

        target = req.context_current_position + num_tokens + self.num_extra_kv_tokens
        capacity = max(kv_cache.capacity, target)

        if not kv_cache.resize(capacity):
            if req.is_first_context_chunk:
                kv_cache.suspend()
            return False

        return True

    def suspend_request(self, req: LlmRequest) -> None:
        kv_cache = self.kv_cache_map.get(req.py_request_id)
        if kv_cache is not None and kv_cache.is_active:
            kv_cache.suspend()

    # ---- prepare_resources ----

    @nvtx_range("prepare_resources_kv_cache_manager_v2")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        if self.is_draft:
            self._prepare_draft_resources(scheduled_batch)
            return

    def _prepare_draft_resources(self, scheduled_batch: ScheduledRequests):
        with request_context(True, scheduled_batch):
            for req in scheduled_batch.context_requests:
                kv_cache = self.kv_cache_map.get(req.py_request_id)
                if kv_cache is None:
                    kv_cache = self._create_kv_cache(req.py_request_id, req.lora_task_id, None)
                    kv_cache.stop_committing()
                if not self._resume_and_restore(req.py_request_id, kv_cache):
                    raise RuntimeError(
                        f"Failed to resume draft KV cache for request {req.py_request_id}"
                    )
                draft_len = get_draft_token_length(req)
                capacity = (
                    req.context_current_position
                    + req.context_chunk_size
                    + draft_len
                    + self.num_extra_kv_tokens
                )
                if not kv_cache.resize(capacity):
                    raise RuntimeError(
                        f"Draft KV cache context resize failed for request "
                        f"{req.py_request_id}: could not resize to {capacity} tokens"
                    )

            for req in scheduled_batch.generation_requests:
                kv_cache = self.kv_cache_map.get(req.py_request_id)
                if kv_cache is None:
                    raise RuntimeError(
                        f"Missing draft KV cache for generation request {req.py_request_id}"
                    )
                if not self._resume_and_restore(req.py_request_id, kv_cache):
                    raise RuntimeError(
                        f"Failed to resume draft KV cache for request {req.py_request_id}"
                    )
                new_cap = self._required_gen_capacity(req, kv_cache.capacity)
                if not kv_cache.resize(new_cap):
                    raise RuntimeError(
                        f"Draft KV cache generation resize failed for request "
                        f"{req.py_request_id}: could not resize to {new_cap} tokens"
                    )

    def get_kv_cache_stats(self):
        kv_cache_stats = KvCacheStats()
        kv_cache_stats.allocated_bytes = self.impl.get_quota(GPU_LEVEL)

        return kv_cache_stats

    def get_iteration_stats(self):
        """V2 does not support per-iteration stats yet."""
        return None

    def get_block_ids_per_seq(self, request_ids: List[int]) -> torch.Tensor:
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        block_ids_per_seq_tensors = [
            torch.tensor(
                [i // self.num_local_layers if i != BAD_PAGE_INDEX else 0 for i in sublist],
                dtype=torch.int,
            )
            for sublist in block_ids_per_seq
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            block_ids_per_seq_tensors, batch_first=True, padding_value=0
        )
        return padded_tensor

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
        draft_kv_cache_manager: Optional["BaseResourceManager"] = None,
    ):
        beam_width = max_beam_width
        requests = []

        def release_resources(
            current_request: LlmRequest, free_draft_resources: bool = False
        ) -> None:
            for req in requests:
                self.free_resources(req)
            self.free_resources(current_request)
            if draft_kv_cache_manager is not None:
                for req in requests:
                    draft_kv_cache_manager.free_resources(req)
                if free_draft_resources:
                    draft_kv_cache_manager.free_resources(current_request)

        for i, req_id in enumerate(request_ids):
            sampling_params = SamplingParams(
                n=beam_width, best_of=beam_width, use_beam_search=beam_width > 1
            )
            token_num = token_nums[i] if token_nums is not None else 1 + max_num_draft_tokens
            encoder_input_tokens = None
            input_tokens = [1 for _ in range(token_num)]
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=1,
                input_tokens=input_tokens,
                sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
                is_streaming=False,
                encoder_input_tokens=encoder_input_tokens,
            )
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            if prepare_resource:
                kv_cache = self._create_kv_cache(req.py_request_id, req.lora_task_id, input_tokens)
                assert kv_cache.num_committed_tokens == 0
                success = kv_cache.resume(self._stream.cuda_stream)
                if not success:
                    release_resources(req)
                    return None
                kv_cache.stop_committing()
                dummy_capacity = token_num + self.num_extra_kv_tokens + num_extra_decoding_steps
                success = kv_cache.resize(dummy_capacity)
                if not success:
                    release_resources(req)
                    return None
                draft_kv_cache = None
                if draft_kv_cache_manager is not None:
                    draft_kv_cache = draft_kv_cache_manager._create_kv_cache(
                        req.py_request_id, req.lora_task_id, input_tokens
                    )
                    success = draft_kv_cache.resume(torch.cuda.current_stream().cuda_stream)
                    if not success:
                        release_resources(req, free_draft_resources=True)
                        return None
                    draft_kv_cache.stop_committing()
                    success = draft_kv_cache.resize(dummy_capacity)
                    if not success:
                        release_resources(req, free_draft_resources=True)
                        return None

            if is_gen:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.prompt_len = token_num - 1
                req.py_prompt_len = req.prompt_len
                req.py_draft_tokens = [1] * max_num_draft_tokens
                if prepare_resource:
                    new_capacity = kv_cache.capacity + max_num_draft_tokens + 1
                    success = kv_cache.resize(new_capacity)
                    if not success:
                        release_resources(req)
                        return None
                    if draft_kv_cache is not None:
                        success = draft_kv_cache.resize(new_capacity)
                        if not success:
                            release_resources(req, free_draft_resources=True)
                            return None

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

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        kv_cache = self.kv_cache_map.pop(request.py_request_id, None)
        if kv_cache is None:
            return
        if (
            self.enable_block_reuse
            and not self.is_draft
            and not request.is_dummy_request
            and request.context_current_position > kv_cache.num_committed_tokens
        ):
            kv_cache.commit(
                request.get_tokens(0)[
                    kv_cache.num_committed_tokens : request.context_current_position
                ]
            )
            kv_cache.stop_committing()
        kv_cache.close()
        self.index_mapper.remove_sequence(request.py_request_id)

    def get_batch_cache_indices(self, request_ids: List[int], layer_id: int = 0) -> List[List[int]]:
        return self._get_batch_cache_indices_by_pool_id(
            request_ids,
            pool_id=self.layer_to_pool_mapping_dict[self.layer_offsets[layer_id]],
            is_kv_aggregate=True,
        )

    def _get_batch_cache_indices_by_pool_id(
        self, request_ids: List[int], *, pool_id: int = 0, is_kv_aggregate: bool = True
    ) -> List[List[int]]:
        if is_kv_aggregate:
            div_factor = self.kv_factor
        else:
            div_factor = 1

        res = []

        for req_id in request_ids:
            idx_tensor = torch.as_tensor(self.kv_cache_map[req_id].get_base_page_indices(pool_id))
            res.append(
                (
                    torch.where(
                        idx_tensor != BAD_PAGE_INDEX,
                        idx_tensor * self.index_scales[pool_id] // div_factor,
                        BAD_PAGE_INDEX,
                    )
                ).tolist()
            )

        return res

    def get_cache_bytes_per_token(self) -> int:
        data_roles = [Role.KEY]
        if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
            data_roles.append(Role.VALUE)
        if self.dtype == DataType.NVFP4:
            data_roles.append(Role.KEY_BLOCK_SCALE)
            if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
                data_roles.append(Role.VALUE_BLOCK_SCALE)

        return sum(
            self.get_layer_bytes_per_token(local_layer_idx=local_layer_idx, data_role=data_role)
            for local_layer_idx in range(self.num_local_layers)
            for data_role in data_roles
        )

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: Role):
        if self.dtype not in (
            DataType.FP8,
            DataType.HALF,
            DataType.BF16,
            DataType.FLOAT,
            DataType.NVFP4,
        ):
            raise ValueError(f"Cannot support {self.dtype} KV cache.")

        if data_role == Role.ALL:
            kv_factor = self.kv_factor
        elif data_role in [Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE]:
            if data_role in [Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE]:
                assert self.dtype == DataType.NVFP4, (
                    "NVFP4 is the only supported dtype for block quant data roles"
                )
            if data_role == Role.VALUE:
                assert self.kv_cache_type != CacheTypeCpp.SELFKONLY, (
                    "VALUE data role is not supported for SELFKONLY cache type"
                )
            kv_factor = 1
        else:
            raise ValueError(f"Invalid data role: {data_role}")

        cache_size_per_token = (
            kv_factor * self.num_kv_heads_per_layer[local_layer_idx] * self.head_dim
        )

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token, self.dtype)

        if data_role in [Role.KEY, Role.VALUE]:
            return cache_size_bytes_per_token

        quant_size_per_token = 0

        if self.dtype == DataType.NVFP4:
            quant_size_per_token = self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8,
            )

        if data_role in [Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE]:
            return quant_size_per_token

        # Role.ALL combines both
        return cache_size_bytes_per_token + quant_size_per_token

    @staticmethod
    def calculate_scaling_factor_size_bytes(
        cache_size: int, quant_vector_size: int, scaling_factor_dtype: DataType
    ) -> int:
        assert cache_size % quant_vector_size == 0, (
            "NVFP4 cache size must be divisible by quant vector size"
        )
        return get_size_in_bytes(cache_size // quant_vector_size, scaling_factor_dtype)

    def check_invalid_values_in_kv_cache(self, fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor(
            [False], dtype=torch.bool, device=torch.cuda.current_device()
        )
        pool_handled = set()

        for layer_id, layer_offset in self.layer_offsets.items():
            pool_id = self.layer_to_pool_mapping_dict[layer_offset]
            if pool_id in pool_handled:
                continue
            buffer = self.get_buffers(layer_id)
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i : i + 256]
                try:
                    has_invalid_values.logical_or_(torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
            pool_handled.add(pool_id)
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for "
                "current kv cache dtype, related checks are skipped"
            )
        return bool(has_invalid_values)

    def shutdown(self):
        for kv_cache in self.kv_cache_map.values():
            kv_cache.close()
        self.kv_cache_map.clear()
        self.impl.shutdown()

    def get_max_resource_count(self) -> int:
        return 1

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    @staticmethod
    def get_cache_size_per_token(
        model_config: ModelConfigPython,
        mapping: Mapping,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        from .kv_cache_manager import KVCacheManager

        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
            mem_per_token = 1

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
        mem_per_token *= num_attention_layers * head_dim

        mem_per_token *= kv_factor
        return mem_per_token

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
        for req in scheduled_batch.context_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            if not kv_cache.is_active:
                continue
            if self.enable_block_reuse and not self.is_draft and not req.is_dummy_request:
                if req.context_current_position > kv_cache.num_committed_tokens:
                    kv_cache.commit(
                        req.get_tokens(DEFAULT_BEAM_INDEX)[
                            kv_cache.num_committed_tokens : req.context_current_position
                        ]
                    )
                if req.context_remaining_length == 0:
                    kv_cache.stop_committing()
            else:
                new_capacity = None
                if self.is_draft and kv_cache.capacity < req.context_current_position:
                    new_capacity = req.context_current_position + self.num_extra_kv_tokens
                success = kv_cache.resize(new_capacity, req.context_current_position)
                if not success:
                    raise ValueError(
                        f"Failed to resize history length of KV cache for "
                        f"request {req.py_request_id} to "
                        f"{req.context_current_position} tokens at "
                        f"context update"
                    )

        for req in scheduled_batch.generation_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            if not kv_cache.is_active:
                continue
            new_capacity = (
                None
                if req.state in (LlmRequestState.GENERATION_COMPLETE, LlmRequestState.CONTEXT_INIT)
                else kv_cache.capacity - req.py_rewind_len
            )
            success = kv_cache.resize(new_capacity, req.max_beam_num_tokens - 1)
            if not success:
                raise ValueError(
                    f"Failed to resize KV cache for request "
                    f"{req.py_request_id} to capacity {new_capacity} "
                    f"and history length "
                    f"{req.max_beam_num_tokens - 1} tokens at "
                    f"generation update"
                )

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ):
        assert beam_width == 1, "beam_width must be 1 for KVCacheManagerV2"

        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        assert copy_idx.shape[0] == num_seqs

        copy_batch_block_offsets_to_device(
            self.host_kv_cache_block_offsets,
            dst_tensor,
            copy_idx,
            self.index_scales,
            self.kv_offset,
            self._stream.cuda_stream,
        )

    def _create_kv_cache(
        self, request_id: int, lora_task_id: int | None, input_tokens: Sequence[TokenIdExt] | None
    ):
        assert request_id not in self.kv_cache_map, (
            f"KV cache for request {request_id} already exists"
        )
        kv_cache = self.impl.create_kv_cache(lora_task_id, input_tokens)
        self.kv_cache_map[request_id] = kv_cache
        index = self.index_mapper.add_new_sequence(request_id)
        for i in range(self.max_beam_width):
            for pool_idx in range(self.num_pools):
                buffer: torch.Tensor = self.host_kv_cache_block_offsets[
                    pool_idx, index * self.max_beam_width + i, 0
                ]
                kv_cache.set_base_page_index_buf(i, pool_idx, memoryview(buffer.numpy()))
        return kv_cache

    def reset_reuse_state(self):
        self.impl.clear_reusable_blocks()
