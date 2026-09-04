# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from tensorrt_llm._torch.attention_backend.block_sparse import BlockSparseForwardInputs
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.prims_ts._block_sparse.config import (
    _validate_block_sparse_static_profile,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams
from .prims_ts import PrimsTSFmha
from .utils import get_kv_page_offset

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BlockSparsePagedTSWrapper,
        BlockSparseTSWrapper,
    )
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


@dataclass(frozen=True, slots=True)
class _BlockSparsePlanKey:
    """Static wrapper profile shared by compatible attention layers."""

    device: torch.device
    batch_size: int
    seq_len_q: int
    kv_capacity: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int | None
    q_block_size: int
    kv_block_size: int
    max_blocks_per_row: int
    mask_type: Literal["dense", "causal"]
    dtype: torch.dtype
    use_kv_valid_bits: bool
    sparse_format: Literal["bsr", "bitmask"]
    use_proxy_routes: bool


_prims_ts_block_sparse_import_error: Exception | None = None
try:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BlockSparsePagedTSWrapper as _BlockSparsePagedTSWrapper,
    )
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BlockSparseTSWrapper as _BlockSparseTSWrapper,
    )
except (ImportError, OSError) as error:
    _BlockSparseTSWrapper = _BlockSparsePagedTSWrapper = None
    _prims_ts_block_sparse_import_error = error


class PrimsTSBlockSparseFmha(PrimsTSFmha):
    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        self.bind_plan_cache({})

    def bind_plan_cache(self, cache_state: dict[str, object]) -> None:
        """Bind wrapper plans to an explicitly owned cache."""

        self._contiguous_wrappers = cast(
            dict[_BlockSparsePlanKey, "BlockSparseTSWrapper"],
            cache_state.setdefault("contiguous_wrappers", {}),
        )
        self._paged_wrappers = cast(
            dict[_BlockSparsePlanKey, "BlockSparsePagedTSWrapper"],
            cache_state.setdefault("paged_wrappers", {}),
        )

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        return (
            _BlockSparseTSWrapper is not None
            and _BlockSparsePagedTSWrapper is not None
            and super().is_available(attn)
        )

    def is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(
            q,
            k,
            v,
            metadata,
            forward_args,
            phase=phase,
        )
        if not supported:
            logger.debug(f"PrimTS block-sparse FMHA does not support request: {reason}")
        return supported

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> tuple[bool, str]:
        if getattr(metadata, "kv_cache_manager", None) is None:
            reason = (
                "contiguous block-sparse attention only supports the context phase"
                if phase not in (None, FmhaPhase.CONTEXT)
                else self._contiguous_unsupported_reason(q, k, v, metadata, forward_args)
            )
        else:
            reason = (
                "paged block-sparse attention only supports the generation phase"
                if phase not in (None, FmhaPhase.GENERATION)
                else self._paged_unsupported_reason(q, metadata, forward_args)
            )
        return reason is None, reason or ""

    def _make_plan_key(
        self,
        q: torch.Tensor,
        inputs: BlockSparseForwardInputs,
        *,
        batch_size: int,
        seq_len_q: int,
        kv_capacity: int,
        page_size: int | None,
        mask_type: Literal["dense", "causal"],
    ) -> _BlockSparsePlanKey:
        max_blocks_per_row = inputs.max_blocks_per_row
        if max_blocks_per_row is None:
            max_blocks_per_row = math.ceil(kv_capacity / inputs.kv_block_size)
        return _BlockSparsePlanKey(
            device=q.device,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            kv_capacity=kv_capacity,
            num_heads=self.attn.num_heads,
            num_kv_heads=self.attn.num_kv_heads,
            head_dim=self.attn.head_dim,
            page_size=page_size,
            q_block_size=inputs.q_block_size,
            kv_block_size=inputs.kv_block_size,
            max_blocks_per_row=max_blocks_per_row,
            mask_type=mask_type,
            dtype=q.dtype,
            use_kv_valid_bits=inputs.kv_valid_bits is not None,
            sparse_format=inputs.sparse_format,
            use_proxy_routes=inputs.use_proxy_routes,
        )

    def _get_or_plan_wrapper(
        self,
        key: _BlockSparsePlanKey,
        *,
        paged: bool,
    ) -> "BlockSparseTSWrapper | BlockSparsePagedTSWrapper":
        cache = self._paged_wrappers if paged else self._contiguous_wrappers
        wrapper = cache.get(key)
        if wrapper is not None:
            return wrapper
        wrapper_type = _BlockSparsePagedTSWrapper if paged else _BlockSparseTSWrapper
        if wrapper_type is None:
            raise RuntimeError(
                "PrimTS block-sparse attention is unavailable: "
                f"{_prims_ts_block_sparse_import_error}"
            )
        if paged and key.page_size is None:
            raise RuntimeError("Paged block-sparse plans require a page size")
        wrapper = wrapper_type()
        plan_args = (
            key.batch_size,
            key.seq_len_q,
            key.kv_capacity,
            key.num_heads,
            key.num_kv_heads,
            key.head_dim,
            key.q_block_size,
            key.kv_block_size,
        )
        if paged:
            plan_args += (key.page_size,)
        plan_kwargs = {
            "device": key.device,
            "max_blocks_per_row": key.max_blocks_per_row,
            "use_kv_valid_bits": key.use_kv_valid_bits,
            "mask_type": key.mask_type,
            "q_data_type": key.dtype,
            "kv_data_type": key.dtype,
            "o_data_type": key.dtype,
        }
        if not paged:
            plan_kwargs.update(
                sparse_format=key.sparse_format,
                use_proxy_routes=key.use_proxy_routes,
            )
        wrapper.plan(*plan_args, **plan_kwargs)
        cache[key] = wrapper
        return wrapper

    def _paged_storage_unsupported_reason(
        self,
        metadata: "TrtllmAttentionMetadata",
    ) -> str | None:
        if metadata.kv_layout != "HND":
            return "only HND KV-cache layout is supported"
        if metadata.host_kv_cache_pool_pointers is None:
            return "KV-cache pool pointers are required"
        pool_mapping = metadata.host_kv_cache_pool_mapping
        if pool_mapping is None:
            return "KV-cache pool mapping is required"

        manager = metadata.kv_cache_manager
        if isinstance(manager, KVCacheManagerV2):
            if manager.enable_swa_scratch_reuse:
                return "KVCacheManagerV2 SWA scratch reuse is not supported"
        elif isinstance(manager, KVCacheManager):
            if manager.num_pools != 1:
                return "KVCacheManagerV1 with multiple memory pools is not supported"
            local_layer_idx = self.attn.local_layer_idx
            if (
                pool_mapping.ndim != 2
                or pool_mapping.shape[1] < 2
                or local_layer_idx is None
                or not 0 <= local_layer_idx < pool_mapping.shape[0]
            ):
                return "KVCacheManagerV1 has an invalid layer-to-pool mapping"
            pool_index = int(pool_mapping[local_layer_idx, 0])
            layer_index_in_pool = int(pool_mapping[local_layer_idx, 1])
            if pool_index != 0 or not 0 <= layer_index_in_pool < manager.num_local_layers:
                return "KVCacheManagerV1 has an invalid layer-to-pool mapping"
        else:
            return f"unsupported KV cache manager {type(manager).__name__}"
        if metadata.tokens_per_block not in self.SUPPORTED_PAGE_SIZES:
            return f"page size {metadata.tokens_per_block} is unsupported"
        try:
            if (
                get_kv_page_offset(
                    self.attn,
                    metadata,
                    0,
                    cache=self._kv_page_offset_cache,
                )
                is None
            ):
                return "the K-to-V page displacement could not be resolved"
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
            return f"invalid KV-cache storage metadata: {error}"
        return None

    @staticmethod
    def _get_block_sparse_inputs(
        forward_args: AttentionForwardArgs,
    ) -> BlockSparseForwardInputs | None:
        sparse_runtime_params = forward_args.sparse_runtime_params
        return (
            sparse_runtime_params.block_sparse_inputs if sparse_runtime_params is not None else None
        )

    @staticmethod
    def _has_legacy_sparse_runtime(forward_args: AttentionForwardArgs) -> bool:
        prediction = forward_args.sparse_runtime_params
        if prediction is None:
            return False
        return any(
            (
                prediction.sparse_kv_indices is not None,
                prediction.sparse_kv_offsets is not None,
                prediction.sparse_attn_indices is not None,
                prediction.sparse_attn_offsets is not None,
                prediction.sparse_attn_kv_lens is not None,
                bool(prediction.sparse_attn_indices_block_size),
                prediction.aux_kv_cache_pool_ptr is not None,
                bool(prediction.threshold_scale_factor_prefill),
                bool(prediction.threshold_scale_factor_decode),
            )
        )

    @staticmethod
    def _block_offsets_unsupported_reason(
        q: torch.Tensor,
        block_offsets: object,
        batch_size: int,
    ) -> str | None:
        if not isinstance(block_offsets, torch.Tensor):
            return "paged KV-cache block offsets are required"
        if block_offsets.ndim != 4:
            return "KV-cache block offsets must be rank-4 [pool, sequence, 2, page]"
        if block_offsets.dtype != torch.int32 or block_offsets.device != q.device:
            return "KV-cache block offsets must be int32 on the query device"
        if not block_offsets.is_contiguous():
            return "KV-cache block offsets must be contiguous"
        if (
            block_offsets.shape[0] <= 0
            or block_offsets.shape[1] < batch_size
            or block_offsets.shape[2] != 2
            or block_offsets.shape[3] <= 0
        ):
            return "KV-cache block offsets do not cover the active batch and page capacity"
        return None

    def _common_unsupported_reason(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> str | None:
        inputs = self._get_block_sparse_inputs(forward_args)
        if not isinstance(inputs, BlockSparseForwardInputs):
            return "live block-sparse forward inputs are required"
        if getattr(metadata, "is_cross", False):
            return "cross attention is not supported"
        output = forward_args.output
        if not isinstance(output, torch.Tensor):
            return "a caller-owned output tensor is required"
        if not isinstance(q, torch.Tensor) or q.ndim != 2 or not q.is_contiguous():
            return "attention input must be a contiguous rank-2 tensor"
        if output.ndim != 2 or not output.is_contiguous():
            return "output must be a contiguous rank-2 tensor"
        if output.device != q.device or output.dtype != q.dtype:
            return "Q and output must share device and dtype"
        expected_output_shape = (int(q.shape[0]), self.attn.num_heads * self.attn.head_dim)
        if tuple(output.shape) != expected_output_shape:
            return f"output must have shape {expected_output_shape}"
        if q.dtype not in self.SUPPORTED_DTYPES:
            return f"query dtype {q.dtype} is unsupported"
        if not math.isfinite(self.attn.q_scaling) or self.attn.q_scaling <= 0:
            return "q_scaling must be finite and positive"
        if self.attn.is_mla_enable:
            return "MLA is not supported"
        if getattr(metadata, "helix_position_offsets", None) is not None:
            return "Helix parallelism is not supported"
        if int(getattr(metadata, "num_sparse_topk", 0)) > 0:
            return "legacy sparse attention metadata is not supported"
        if self._has_legacy_sparse_runtime(forward_args):
            return "legacy sparse prediction cannot be combined with block-sparse inputs"
        if forward_args.enable_dsv4_epilogue_fusion:
            return "DSv4 epilogue fusion is not supported"
        if forward_args.sage_attn_qk_int8 or any(
            getattr(forward_args, name) > 0
            for name in (
                "sage_attn_num_elts_per_blk_q",
                "sage_attn_num_elts_per_blk_k",
                "sage_attn_num_elts_per_blk_v",
            )
        ):
            return "SageAttention is not supported"
        if forward_args.softmax_stats_tensor is not None:
            return "softmax statistics output is not supported"
        if (
            forward_args.output_sf is not None
            or forward_args.out_scale is not None
            or forward_args.out_scale_sf is not None
        ):
            return "quantized output is not supported"
        if (
            forward_args.attention_mask_data is not None
            or forward_args.relative_attention_bias is not None
            or forward_args.attention_sinks is not None
        ):
            return "custom attention masks, bias, and sinks are not supported"
        if forward_args.attention_mask not in (
            PredefinedAttentionMask.FULL,
            PredefinedAttentionMask.CAUSAL,
        ):
            return "only full and causal masks are supported"
        if q.device.type != "cuda":
            return "CUDA tensors are required"
        return None

    def _static_profile_unsupported_reason(
        self,
        q: torch.Tensor,
        inputs: BlockSparseForwardInputs,
        *,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv: int,
        page_size: int | None,
        mask_type: Literal["dense", "causal"],
    ) -> str | None:
        max_blocks_per_row = inputs.max_blocks_per_row
        if max_blocks_per_row is None:
            max_blocks_per_row = math.ceil(seq_len_kv / inputs.kv_block_size)
        try:
            _validate_block_sparse_static_profile(
                batch_size=batch_size,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                num_qo_heads=self.attn.num_heads,
                num_kv_heads=self.attn.num_kv_heads,
                head_dim=self.attn.head_dim,
                q_block_size=inputs.q_block_size,
                kv_block_size=inputs.kv_block_size,
                use_kv_valid_bits=inputs.kv_valid_bits is not None,
                mask_type=mask_type,
                q_dtype=q.dtype,
                kv_dtype=q.dtype,
                output_dtype=q.dtype,
                max_blocks_per_row=max_blocks_per_row,
                page_size=page_size,
            )
        except (ValueError, NotImplementedError, OverflowError) as error:
            return str(error)
        return None

    def _paged_unsupported_reason(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> str | None:
        inputs = self._get_block_sparse_inputs(forward_args)
        if not isinstance(inputs, BlockSparseForwardInputs):
            return "live block-sparse forward inputs are required"
        if inputs.sparse_format != "bsr" or inputs.use_proxy_routes:
            return "paged block-sparse attention only supports BSR exact routes"
        common_reason = self._common_unsupported_reason(q, metadata, forward_args)
        if common_reason is not None:
            return common_reason
        if not all(
            callable(getattr(thop, name, None))
            for name in (
                "get_trtllm_gen_generation_workspace_layout",
                "trtllm_gen_generation_preprocess",
            )
        ):
            return "TRT-LLM generation preprocessing ops are unavailable"
        if not forward_args.is_fused_qkv:
            return "paged block-sparse attention requires fused QKV input"
        if forward_args.attention_input_type != AttentionInputType.generation_only:
            return "only generation-only paged requests are supported"
        if int(getattr(metadata, "num_contexts", 0)) != 0:
            return "paged block-sparse attention does not support context requests"
        batch_size = int(getattr(metadata, "num_generations", 0))
        if batch_size <= 0 or q.shape[0] % batch_size:
            return "query tokens must be uniformly divisible across generation requests"
        seq_len_q = int(q.shape[0]) // batch_size
        if seq_len_q <= 0:
            return "each generation request must contain at least one query token"
        query_lengths = getattr(metadata, "seq_lens", None)
        if (
            not isinstance(query_lengths, torch.Tensor)
            or query_lengths.device.type != "cpu"
            or query_lengths.dtype != torch.int32
            or not query_lengths.is_contiguous()
            or query_lengths.numel() < batch_size
        ):
            return "host int32 query lengths are required to prove a uniform fixed query shape"
        active_query_lengths = query_lengths[:batch_size]
        if not bool(active_query_lengths.eq(seq_len_q).all()):
            return "query lengths must be batch-uniform and match the fixed query shape"
        expected_width = (self.attn.num_heads + 2 * self.attn.num_kv_heads) * self.attn.head_dim
        if int(q.shape[1]) != expected_width:
            return f"fused QKV width must be {expected_width}"
        paged_kv_reason = self._paged_storage_unsupported_reason(metadata)
        if paged_kv_reason is not None:
            return paged_kv_reason
        block_tables = metadata.kv_cache_block_offsets
        block_offsets_reason = self._block_offsets_unsupported_reason(q, block_tables, batch_size)
        if block_offsets_reason is not None:
            return block_offsets_reason
        block_tables = cast(torch.Tensor, block_tables)
        if int(getattr(metadata, "beam_width", 1)) != 1:
            return "beam search is not supported"
        if any(
            bool(getattr(metadata, name, False))
            for name in (
                "is_spec_decoding_enabled",
                "use_spec_decoding",
                "is_spec_dec_tree",
                "is_spec_dec_dynamic_tree",
            )
        ):
            return "speculative decoding is not supported"
        if self.attn.attention_chunk_size:
            return "chunked attention is not supported"
        if self.attn.position_embedding_type in (4, 5, 6, 7, 10):
            return f"position embedding type {self.attn.position_embedding_type} is not supported"
        page_size = int(getattr(metadata, "tokens_per_block", 0))
        if page_size <= 0:
            return "page size must be positive"
        max_seq_len_kv = int(block_tables.shape[-1]) * page_size
        try:
            logical_max_seq_len = int(metadata.max_seq_len)
        except (AttributeError, TypeError, ValueError):
            return "a positive logical maximum sequence length is required"
        if logical_max_seq_len <= 0 or logical_max_seq_len > max_seq_len_kv:
            return "logical maximum sequence length must fit the page-table capacity"
        attention_window_size = forward_args.attention_window_size
        if (
            isinstance(attention_window_size, bool)
            or not isinstance(attention_window_size, int)
            or attention_window_size < logical_max_seq_len
            or attention_window_size > max_seq_len_kv
        ):
            return "attention window must fit the non-cyclic page-table capacity"
        try:
            quant_mode = QuantMode(self.attn.quant_mode)
        except (TypeError, ValueError):
            return "invalid quantization mode"
        if quant_mode.has_kv_cache_quant():
            return "quantized KV cache is not supported"
        cache_dtype = metadata.kv_cache_manager.dtype
        if not isinstance(cache_dtype, torch.dtype):
            cache_dtype = binding_to_torch_dtype(cache_dtype)
        if cache_dtype != q.dtype:
            return "query, paged KV cache, and output dtypes must match"
        host_seq_lens = getattr(metadata, "kv_lens_runtime", None)
        if (
            not isinstance(host_seq_lens, torch.Tensor)
            or host_seq_lens.dtype != torch.int32
            or host_seq_lens.device.type != "cpu"
            or host_seq_lens.numel() < batch_size
            or not host_seq_lens.is_contiguous()
        ):
            return "live host int32 KV lengths are required for safe policy selection"
        active_host_seq_lens = host_seq_lens[:batch_size]
        min_seq_len_kv = int(active_host_seq_lens.min())
        max_past_kv_length = int(active_host_seq_lens.max())
        if min_seq_len_kv <= 0:
            return "every active request must contain at least one KV token"
        mask_type = self._get_prims_mask_type(forward_args)
        if mask_type == "causal" and min_seq_len_kv < seq_len_q:
            return "causal KV lengths must be at least the fixed query length"
        if max_past_kv_length > logical_max_seq_len:
            return "an active KV length exceeds the logical maximum sequence length"
        return self._static_profile_unsupported_reason(
            q,
            inputs,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=max_seq_len_kv,
            page_size=page_size,
            mask_type=mask_type,
        )

    def _ensure_preprocess_workspace(
        self,
        q: torch.Tensor,
        workspace: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        layout = self._get_generation_workspace_layout(
            q.dtype,
            batch_size,
            int(q.shape[0]),
        )
        required_bytes = int(layout["total_size"])
        available_bytes = workspace.numel() * workspace.element_size()
        if available_bytes < required_bytes:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "TRT-LLM QKV preprocessing workspace must be sized before CUDA Graph capture"
                )
            workspace.resize_((math.ceil(required_bytes / workspace.element_size()),))

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        del k, v, forward_args
        with torch.cuda.device(q.device):
            self._ensure_preprocess_workspace(
                q,
                workspace,
                batch_size=int(metadata.num_generations),
            )
            if self._multi_processor_count is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("GPU properties must be prepared before CUDA Graph capture")
                self._multi_processor_count = torch.cuda.get_device_properties(
                    q.device
                ).multi_processor_count

    def run_generation(self, params: FmhaParams) -> None:
        q = params.qkv_input
        output_buffer = params.context_buf
        sequence_lengths = params.sequence_lengths
        assert q is not None and output_buffer is not None and sequence_lengths is not None
        metadata = params.meta
        forward_args = params.fwd
        inputs = self._get_block_sparse_inputs(forward_args)
        assert isinstance(inputs, BlockSparseForwardInputs)
        batch_size = params.num_requests
        seq_len_q = params.input_seq_length
        page_size = params.tokens_per_block
        block_offsets = metadata.kv_cache_block_offsets
        assert block_offsets is not None
        max_seq_len_kv = int(block_offsets.shape[-1]) * page_size
        mask_type = self._get_prims_mask_type(forward_args)
        assert self._multi_processor_count is not None
        (
            q_processed,
            kv_pool,
            block_tables,
            _kv_scale_pool,
            _bmm1_scale,
            _bmm2_scale,
            fmha_workspace,
            _cu_seqlens,
            _max_q_len,
            _max_kv_len,
            _window_left,
            _is_multi_token_gen,
        ) = self._run_generation_preprocess(params)
        if fmha_workspace.numel() != 0:
            raise RuntimeError("PrimTS block-sparse preprocessing returned an FMHA workspace.")
        if _is_multi_token_gen:
            raise RuntimeError(
                "TRT-LLM preprocessing reported an unexpected speculative or "
                "variable-query generation profile"
            )
        if q_processed is None or kv_pool is None or block_tables is None:
            raise RuntimeError("TRT-LLM preprocessing did not return paged PrimTS metadata")
        kv_page_offset = get_kv_page_offset(
            params.attn,
            metadata,
            params.seq_offset,
            cache=self._kv_page_offset_cache,
        )
        if kv_page_offset is None:
            raise RuntimeError("PrimTS could not resolve the K-to-V page displacement")
        k_cache, v_cache = self._standard_kv_views(kv_pool, kv_page_offset)
        fixed_block_tables = self._get_fixed_block_tables(
            block_tables,
            batch_size,
        )
        seq_lens = self._get_sequence_lengths(sequence_lengths, batch_size)
        query = q_processed.view(
            batch_size,
            seq_len_q,
            self.attn.num_heads,
            self.attn.head_dim,
        )
        output = output_buffer.view_as(query)
        key = self._make_plan_key(
            query,
            inputs,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            kv_capacity=max_seq_len_kv,
            page_size=page_size,
            mask_type=mask_type,
        )
        wrapper = cast(
            "BlockSparsePagedTSWrapper",
            self._get_or_plan_wrapper(key, paged=True),
        )
        wrapper.run(
            query,
            (k_cache, v_cache),
            block_tables=fixed_block_tables,
            seq_lens_kv=seq_lens,
            block_indptr=inputs.block_indptr,
            block_indices=inputs.block_indices,
            kv_valid_bits=inputs.kv_valid_bits,
            sm_scale=self._get_bmm1_scale(self.attn),
            out=output,
        )

    def _contiguous_views(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_tensor: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(route_tensor.shape[0])
        if batch_size <= 0 or q.shape[0] % batch_size or k.shape[0] % batch_size:
            raise ValueError("flat Q and K token counts must be divisible by route batch size")
        seq_len_q = int(q.shape[0]) // batch_size
        seq_len_kv = int(k.shape[0]) // batch_size
        return (
            q.view(batch_size, seq_len_q, self.attn.num_heads, self.attn.head_dim),
            k.view(batch_size, seq_len_kv, self.attn.num_kv_heads, self.attn.head_dim),
            v.view(batch_size, seq_len_kv, self.attn.num_kv_heads, self.attn.head_dim),
            output.view(batch_size, seq_len_q, self.attn.num_heads, self.attn.head_dim),
        )

    def _contiguous_unsupported_reason(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> str | None:
        common_reason = self._common_unsupported_reason(q, metadata, forward_args)
        if common_reason is not None:
            return common_reason
        if self.attn.position_embedding_type != 0 or forward_args.mrope_position_deltas is not None:
            return "contiguous Q/K/V must have position embedding applied before attention"
        if forward_args.is_fused_qkv or k is None or v is None:
            return "contiguous block-sparse attention requires separate Q, K, and V"
        if forward_args.cu_q_seqlens is not None or forward_args.cu_kv_seqlens is not None:
            return "packed variable-length Q/KV inputs are not supported"
        if not all(tensor.ndim == 2 and tensor.is_contiguous() for tensor in (k, v)):
            return "K and V must be contiguous rank-2 tensors"
        if k.shape != v.shape:
            return "K and V must have identical shapes"
        if any(tensor.device != q.device or tensor.dtype != q.dtype for tensor in (k, v)):
            return "Q, K, and V must share device and dtype"
        if int(k.shape[1]) != self.attn.num_kv_heads * self.attn.head_dim:
            return "K and V hidden dimensions do not match the attention configuration"
        inputs = self._get_block_sparse_inputs(forward_args)
        assert isinstance(inputs, BlockSparseForwardInputs)
        if inputs.use_proxy_routes and self._get_prims_mask_type(forward_args) != "dense":
            return "block-sparse proxy routes require mask_type='dense'"
        route_tensor = (
            inputs.block_indptr if inputs.sparse_format == "bsr" else inputs.exact_block_bits
        )
        if not isinstance(route_tensor, torch.Tensor):
            return "the selected route representation must be a torch.Tensor"
        expected_rank = 3 if inputs.sparse_format == "bsr" else 4
        if route_tensor.ndim != expected_rank:
            return f"{inputs.sparse_format} routes must be rank-{expected_rank}"
        try:
            q_view, k_view, _, _ = self._contiguous_views(
                q,
                k,
                v,
                route_tensor,
                forward_args.output,
            )
        except (RuntimeError, ValueError) as error:
            return str(error)
        batch_size, seq_len_q = map(int, q_view.shape[:2])
        query_lengths = getattr(metadata, "seq_lens", None)
        if (
            not isinstance(query_lengths, torch.Tensor)
            or query_lengths.device.type != "cpu"
            or query_lengths.dtype != torch.int32
            or not query_lengths.is_contiguous()
            or query_lengths.numel() < batch_size
        ):
            return "host int32 query lengths are required to prove a uniform fixed query shape"
        if not bool(query_lengths[:batch_size].eq(seq_len_q).all()):
            return "query lengths must be batch-uniform and match the fixed query shape"
        return self._static_profile_unsupported_reason(
            q,
            inputs,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=int(k_view.shape[1]),
            page_size=None,
            mask_type=self._get_prims_mask_type(forward_args),
        )

    def _forward_contiguous(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_args: AttentionForwardArgs,
    ) -> None:
        inputs = self._get_block_sparse_inputs(forward_args)
        assert isinstance(inputs, BlockSparseForwardInputs)
        assert forward_args.output is not None
        q_view, k_view, v_view, out_view = self._contiguous_views(
            q,
            k,
            v,
            inputs.block_indptr if inputs.sparse_format == "bsr" else inputs.exact_block_bits,
            forward_args.output,
        )
        mask_type = self._get_prims_mask_type(forward_args)
        key = self._make_plan_key(
            q_view,
            inputs,
            batch_size=int(q_view.shape[0]),
            seq_len_q=int(q_view.shape[1]),
            kv_capacity=int(k_view.shape[1]),
            page_size=None,
            mask_type=mask_type,
        )
        wrapper = cast(
            "BlockSparseTSWrapper",
            self._get_or_plan_wrapper(key, paged=False),
        )
        wrapper.run(
            q_view,
            k_view,
            v_view,
            block_indptr=inputs.block_indptr,
            block_indices=inputs.block_indices,
            exact_block_bits=inputs.exact_block_bits,
            k_summary=inputs.k_summary,
            v_summary=inputs.v_summary,
            kv_valid_bits=inputs.kv_valid_bits,
            sm_scale=self._get_bmm1_scale(self.attn),
            out=out_view,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        if getattr(metadata, "kv_cache_manager", None) is None:
            reason = self._contiguous_unsupported_reason(q, k, v, metadata, forward_args)
            if reason is not None:
                raise RuntimeError(f"unsupported contiguous block-sparse request: {reason}")
            assert k is not None and v is not None
            self._forward_contiguous(q, k, v, forward_args)
            return

        if k is not None or v is not None:
            raise RuntimeError("paged block-sparse attention requires fused QKV input")
        reason = self._paged_unsupported_reason(q, metadata, forward_args)
        if reason is not None:
            raise RuntimeError(f"unsupported paged block-sparse request: {reason}")
        super().forward(q, k, v, metadata, forward_args)
