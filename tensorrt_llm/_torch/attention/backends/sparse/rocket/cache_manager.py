# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

import torch
from triton import next_power_of_2

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import BlockManager, KVCacheManager
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole, PageIndexMode

from .params import RocketKVParams

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, SparseAttentionConfig
    from tensorrt_llm.sampling_params import SamplingParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

ROCKET_KT_ROLE = DataRole("rocket_kt")
"""KVCacheManagerV2 buffer role of the per-layer RocketKV KT (key landmark) cache."""

# The attention op addresses a layer's K/V pages at a fixed stride inside the
# K/V pool, so KT pages must not share that pool even when their page size
# matches the K/V page size.
_KT_COALESCING_GROUP = 1


def _get_rocket_params(
    sparse_attention_config: Optional["SparseAttentionConfig"], pretrained_config
) -> RocketKVParams:
    if sparse_attention_config is None:
        raise ValueError("sparse_attention_config is required for RocketKV cache")
    sparse_params = sparse_attention_config.to_sparse_params(pretrained_config=pretrained_config)
    if not isinstance(sparse_params, RocketKVParams):
        raise ValueError("RocketKV cache requires RocketKV sparse parameters")
    return sparse_params


def _get_kt_cache_dtype(sparse_params: RocketKVParams) -> torch.dtype:
    return torch.bfloat16 if sparse_params.kt_cache_dtype == "bfloat16" else torch.float8_e5m2


def _get_local_kv_hidden_size(model_config: ModelConfig, mapping: Mapping) -> float:
    """Return ``num_kv_heads * head_dim`` of one rank-local layer (averaged over layers)."""
    config = model_config.pretrained_config
    num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    if isinstance(num_key_value_heads, Iterable):
        num_key_value_heads = sum(num_key_value_heads) / len(num_key_value_heads)

    tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
    head_dim = getattr(config, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = config.hidden_size // config.num_attention_heads
    return head_dim * num_key_value_heads // tp_size


class RocketKVCacheManager(KVCacheManager):
    def __init__(
        self,
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
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        pretrained_config=None,
        **kwargs,
    ) -> None:
        sparse_params = _get_rocket_params(sparse_attention_config, pretrained_config)
        assert not kv_cache_config.enable_block_reuse, (
            "RocketKV cache requires block reuse to be disabled in KV cache config"
        )
        self.kt_tokens_per_block = next_power_of_2(
            math.ceil(tokens_per_block / sparse_params.page_size)
        )
        self.kt_cache_dtype = _get_kt_cache_dtype(sparse_params)

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
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            **kwargs,
        )
        self.page_size = sparse_params.page_size
        self.prompt_budget = sparse_params.prompt_budget
        self.max_batch_size = max_batch_size

        # Per layer KT cache pool
        # Use the same number of blocks as the paged kv cache. In this way, the scheduler can use the same number of
        # blocks to schedule requests.
        # Use kt_tokens_per_block to make sure the KT cache is large enough to hold the kt tokens,
        # since kt_tokens_per_block * num_blocks * page_size >= tokens_per_block * num_blocks.
        self.num_blocks = self.blocks_in_primary_pool
        self.kt_cache_pool_per_layer = [
            torch.empty(
                (self.num_blocks, self.kt_tokens_per_block, local_kv_heads, head_dim * 2),
                device="cuda",
                dtype=self.kt_cache_dtype,
            )
            for local_kv_heads in self.num_kv_heads_per_layer
        ]
        self.max_kt_blocks_per_seq = self.num_blocks

        # Block manager to manage the KT cache blocks for each request. Different layers share the
        # same block ids.
        self.kt_cache_manager = BlockManager(self.num_blocks, self.kt_tokens_per_block)

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        max_num_draft_tokens: int = 0,
        kv_reserve_draft_tokens: Optional[int] = None,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        encoder_output_lens: Optional[List[int]] = None,
        draft_kv_cache_manager=None,
        capture_sampling_params: Optional["SamplingParams"] = None,
    ):
        requests = super().add_dummy_requests(
            request_ids=request_ids,
            token_nums=token_nums,
            is_gen=is_gen,
            max_num_draft_tokens=max_num_draft_tokens,
            kv_reserve_draft_tokens=kv_reserve_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
            encoder_output_lens=encoder_output_lens,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
        )
        if requests is None:
            return None
        for req in requests:
            request_id = req.py_request_id
            kt_token_num = math.ceil(req.max_beam_num_tokens / self.page_size)
            self.kt_cache_manager.add_tokens(request_id, kt_token_num)
        return requests

    def get_kt_buffers(self, layer_idx: int):
        return self.kt_cache_pool_per_layer[layer_idx]

    def copy_kt_block_offsets(
        self, request_ids: List[int], block_offsets: torch.Tensor
    ) -> torch.Tensor:
        self.kt_cache_manager.copy_block_offsets(request_ids, block_offsets)

    def prepare_resources(self, scheduled_batch):
        super().prepare_resources(scheduled_batch)
        for req in scheduled_batch.context_requests:
            request_id = req.py_request_id
            num_tokens = req.prompt_len
            kt_token_num = math.ceil(num_tokens / self.page_size)
            self.kt_cache_manager.add_tokens(request_id, kt_token_num)

        for req in scheduled_batch.generation_requests:
            request_id = req.py_request_id
            num_tokens = req.max_beam_num_tokens + 1
            if num_tokens % self.page_size == 1:
                self.kt_cache_manager.add_tokens(request_id, 1)

    def update_resources(
        self,
        scheduled_batch,
        attn_metadata: AttentionMetadata = None,
        kv_cache_dtype_byte_size: float = None,
    ):
        for request in scheduled_batch.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                seq_len = request.get_num_tokens(0)
                rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
                self.rewind_kv_cache(request, rewind_len)
                # get the rewind length for kt cache
                num_tokens = request.max_beam_num_tokens
                updated_kt_token_num = num_tokens - rewind_len
                rewind_len = math.ceil(num_tokens / self.page_size) - math.ceil(
                    updated_kt_token_num / self.page_size
                )
                self.kt_cache_manager.rewind_cache(request, rewind_len)

    def free_resources(self, request):
        super().free_resources(request)
        self.kt_cache_manager.free_resources(request)

    @staticmethod
    def get_cache_size_per_token(
        model_config: ModelConfig, mapping: Mapping, num_layers: Optional[int] = None, **kwargs
    ):
        # main KV cache dtype bytes (one each for K and V)
        kv_dtype_bytes = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
            kv_dtype_bytes = 1

        head_dim = _get_local_kv_hidden_size(model_config, mapping)

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers
        )

        # K and V: two tensors stored at the main KV cache dtype.
        mem_per_token = kv_dtype_bytes * 2 * num_attention_layers * head_dim

        tokens_per_block = kwargs["tokens_per_block"]
        sparse_params = _get_rocket_params(
            model_config.sparse_attention_config, model_config.pretrained_config
        )
        kt_tokens_per_block = next_power_of_2(math.ceil(tokens_per_block / sparse_params.page_size))
        # KT (key-landmark) cache: a separate pool stored at kt_cache_dtype,
        # NOT the main KV dtype. Per real token it holds
        # kt_tokens_per_block / tokens_per_block landmark slots, each storing
        # the concatenated (min, max) landmark -- a factor of 2 over head_dim.
        # Size it at its own dtype's byte width (1 for float8_e5m2, else 2) so
        # the estimate is independent of the main KV dtype and matches both
        # get_cache_bytes_per_token and the physical pool allocated in
        # RocketKVCacheManager.__init__.
        kt_dtype_bytes = 1 if sparse_params.kt_cache_dtype == "float8_e5m2" else 2
        mem_per_token += (
            kt_dtype_bytes
            * 2
            * kt_tokens_per_block
            / tokens_per_block
            * num_attention_layers
            * head_dim
        )
        return mem_per_token

    def get_cache_bytes_per_token(self):
        # Main K/V cache: stored at the main KV cache dtype (self.dtype).
        cache_size_per_token = math.ceil(
            self.kv_factor * sum(self.num_kv_heads_per_layer) * self.head_dim
        )

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

        # KT (key-landmark) cache: a separate pool physically allocated at
        # kt_cache_dtype, NOT the main KV dtype -- see __init__:
        #   torch.empty((num_blocks, kt_tokens_per_block, num_kv_heads,
        #                head_dim * 2), dtype=kt_cache_dtype).
        # The trailing factor of 2 is the concatenated (min, max) landmark per
        # head_dim. Per real token the pool holds
        # kt_tokens_per_block / tokens_per_block landmark slots. Size it at its
        # own dtype's byte width (1 for float8_e5m2, else 2) so the estimate
        # matches the allocation regardless of self.dtype.
        kt_dtype_bytes = 1 if self.kt_cache_dtype == torch.float8_e5m2 else 2
        kt_elements_per_token = math.ceil(
            2
            * self.kt_tokens_per_block
            / self.tokens_per_block
            * sum(self.num_kv_heads_per_layer)
            * self.head_dim
        )
        cache_size_bytes_per_token += kt_elements_per_token * kt_dtype_bytes

        return cache_size_bytes_per_token


class RocketKVCacheManagerV2(KVCacheManagerV2):
    """KVCacheManagerV2 that stores the RocketKV KT cache in the KV pages.

    Every layer registers a KT buffer beside its K/V buffers. A page holds the
    K/V rows of ``tokens_per_block`` tokens and the ``tokens_per_block //
    page_size`` KT rows (the per-page min/max key landmarks) that summarize
    them, so KT storage is allocated, freed, suspended, and resumed with the
    K/V pages it describes, and the KV cache byte budget covers it. A request's
    KT block table is its V2 base page indices multiplied by the KT buffer's
    page-index scale; each layer's KT view starts at that layer's shared base
    address.

    RocketKV keeps ``min(prompt_len, prompt_budget)`` KV entries of a prompt.
    When a prompt completes, the cache is shrunk to that retained length, and
    its history watermark follows the retained entries rather than the
    request's logical token count.
    """

    # Set once construction completes; see kv_compression_manages_history.
    _constructed: bool = False

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: Union[int, List[int]],
        tokens_per_block: int,
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        pretrained_config=None,
        **kwargs,
    ) -> None:
        sparse_params = _get_rocket_params(sparse_attention_config, pretrained_config)
        if kv_cache_config.enable_block_reuse:
            raise ValueError(
                "RocketKV cache requires block reuse to be disabled in KV cache config"
            )
        if tokens_per_block % sparse_params.page_size != 0:
            raise ValueError(
                f"RocketKV with KVCacheManagerV2 requires tokens_per_block ({tokens_per_block}) "
                f"to be a multiple of the RocketKV page_size ({sparse_params.page_size}), "
                "because each KV page stores the KT rows of its own tokens."
            )
        self.page_size = sparse_params.page_size
        self.prompt_budget = sparse_params.prompt_budget
        self.kt_tokens_per_block = tokens_per_block // sparse_params.page_size
        self.kt_cache_dtype = _get_kt_cache_dtype(sparse_params)
        self.kt_cache_page_scale = 1
        self.kt_cache_pool_per_layer: List[torch.Tensor] = []

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
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            pretrained_config=pretrained_config,
            **kwargs,
        )
        self.max_kt_blocks_per_seq = self.max_blocks_per_seq
        self._init_kt_cache_views()
        self._constructed = True

    @property
    def kv_compression_manages_history(self) -> bool:
        """Always true: generation updates must leave the history watermark alone.

        Once a prompt is pruned, a request's retained KV length trails its
        logical token count, which generation updates would otherwise write as
        the history length.
        """
        return True

    @kv_compression_manages_history.setter
    def kv_compression_manages_history(self, value: bool) -> None:
        # The base initializer assigns the flag; only a KV-cache compression
        # manager assigns it after construction.
        if self._constructed:
            raise NotImplementedError(
                "RocketKV sparse attention does not support KV-cache compression"
            )

    def _kt_bytes_per_block(self, local_layer_idx: int, tokens_per_block: int) -> int:
        return (
            (tokens_per_block // self.page_size)
            * self.num_kv_heads_per_layer[local_layer_idx]
            * 2
            * self.head_dim_per_layer[local_layer_idx]
            * self.kt_cache_dtype.itemsize
        )

    def _extra_buffers_per_layer(self, *, tokens_per_block: int) -> dict[int, List[BufferConfig]]:
        return {
            local_layer_idx: [
                BufferConfig(
                    role=ROCKET_KT_ROLE,
                    size=self._kt_bytes_per_block(local_layer_idx, tokens_per_block),
                    coalescing_group=_KT_COALESCING_GROUP,
                )
            ]
            for local_layer_idx in range(self.num_local_layers)
        }

    def _init_kt_cache_views(self) -> None:
        converters = [
            self.impl.get_page_index_converter(local_layer_idx, ROCKET_KT_ROLE)
            for local_layer_idx in range(self.num_local_layers)
        ]
        if any(int(converter.expansion) != 1 for converter in converters):
            raise ValueError("RocketKV KT cache does not support page expansion")
        scales = {int(converter.scale) for converter in converters}
        if len(scales) > 1:
            # The KT block table is shared by every layer.
            raise ValueError(
                "RocketKV requires a uniform KT cache page-index scale across layers, "
                f"got {sorted(scales)}"
            )
        self.kt_cache_page_scale = scales.pop() if scales else 1
        self.kt_cache_pool_per_layer = [
            self._get_kt_cache_pool_view(local_layer_idx)
            for local_layer_idx in range(self.num_local_layers)
        ]

    def _get_kt_cache_pool_view(self, local_layer_idx: int) -> torch.Tensor:
        """Return one layer's KT cache view.

        The view is ``[num_pages, kt_tokens_per_block, num_kv_heads, 2 * head_dim]``
        and is indexed by the KT block table.
        """
        address = self.impl.get_mem_pool_base_address(
            local_layer_idx, ROCKET_KT_ROLE, PageIndexMode.SHARED
        )
        num_pages = self.impl.get_page_index_upper_bound(local_layer_idx, ROCKET_KT_ROLE)
        page_bytes = self._kt_bytes_per_block(local_layer_idx, self.tokens_per_block)
        pool = convert_to_torch_tensor(
            TensorWrapper(int(address), torch.uint8, [num_pages, page_bytes])
        )
        return pool.view(self.kt_cache_dtype).view(
            num_pages,
            self.kt_tokens_per_block,
            self.num_kv_heads_per_layer[local_layer_idx],
            2 * self.head_dim_per_layer[local_layer_idx],
        )

    def get_kt_buffers(self, layer_idx: int) -> torch.Tensor:
        """Return the page-indexed KT cache view of a global layer."""
        return self.kt_cache_pool_per_layer[self.layer_offsets[layer_idx]]

    def copy_kt_block_offsets(self, request_ids: List[int], block_offsets: torch.Tensor) -> None:
        """Write the KT block tables of ``request_ids`` into ``block_offsets``.

        Args:
            request_ids: Requests in batch order.
            block_offsets: Host int32 tensor ``[>= len(request_ids), max_kt_blocks_per_seq]``.
                Row ``i`` receives the KT page indices of ``request_ids[i]``.
        """
        num_seqs = len(request_ids)
        if num_seqs == 0:
            return
        copy_idx = self.index_mapper.get_copy_index(list(request_ids), num_seqs, 1)
        copy_idx = copy_idx.to(device="cpu", dtype=torch.long)
        base_page_indices = self.host_kv_cache_block_offsets[0, copy_idx, 0, :]
        block_offsets[:num_seqs, : base_page_indices.shape[1]].copy_(
            base_page_indices * self.kt_cache_page_scale
        )

    def _resize_context_history(self, req: "LlmRequest", kv_cache, history_length: int) -> None:
        if self.kv_cache_type == CacheTypeCpp.SELF and req.context_remaining_length == 0:
            # The prompt forward stored only the retained entries; release the
            # pages past them and mark them as the cache history.
            retained_length = min(req.prompt_len, self.prompt_budget)
            if not kv_cache.resize(retained_length, retained_length):
                raise ValueError(
                    f"Failed to shrink the KV cache of request {req.py_request_id} to "
                    f"its {retained_length} retained RocketKV prompt tokens"
                )
            self._after_shrink(req.py_request_id, kv_cache)
            return
        super()._resize_context_history(req, kv_cache, history_length)

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        if data_role == ROCKET_KT_ROLE:
            return math.ceil(
                self._kt_bytes_per_block(local_layer_idx, self.tokens_per_block)
                / self.tokens_per_block
            )
        cache_bytes = super().get_layer_bytes_per_token(local_layer_idx, data_role)
        if data_role == Role.ALL:
            cache_bytes += self.get_layer_bytes_per_token(local_layer_idx, ROCKET_KT_ROLE)
        return cache_bytes

    def get_cache_bytes_per_token(self) -> int:
        return sum(
            self.get_layer_bytes_per_token(local_layer_idx, Role.ALL)
            for local_layer_idx in range(self.num_local_layers)
        )

    @classmethod
    def get_cache_size_per_token(
        cls,
        model_config: ModelConfig,
        mapping: Mapping,
        num_layers: Optional[int] = None,
        *,
        tokens_per_block: int,
        **kwargs,
    ):
        cache_size_per_token, fixed_cost = super().get_cache_size_per_token(
            model_config,
            mapping,
            num_layers,
            tokens_per_block=tokens_per_block,
            **kwargs,
        )
        sparse_params = _get_rocket_params(
            model_config.sparse_attention_config, model_config.pretrained_config
        )
        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers
        )
        # Per token, a page holds 1 / page_size KT rows of (min, max) landmarks.
        kt_bytes_per_token = (
            2
            * _get_local_kv_hidden_size(model_config, mapping)
            * _get_kt_cache_dtype(sparse_params).itemsize
            / sparse_params.page_size
        )
        return (
            cache_size_per_token + math.ceil(kt_bytes_per_token * num_attention_layers),
            fixed_cost,
        )

    def shutdown(self) -> None:
        self.kt_cache_pool_per_layer = []
        super().shutdown()
