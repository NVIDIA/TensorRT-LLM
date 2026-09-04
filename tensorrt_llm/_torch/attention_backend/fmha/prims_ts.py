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

"""TRT-LLM FMHA adapter for the vendored PrimTS Blackwell kernels."""

from __future__ import annotations

import math
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Optional

import torch
from packaging.version import InvalidVersion, Version

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import binding_to_torch_dtype, get_sm_version
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div, pad_up
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha
from .utils import get_kv_page_offset

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.prims_ts.context import BatchPrefillPagedTSWrapper
    from tensorrt_llm._torch.attention_backend.prims_ts.decode import BatchDecodePagedTSWrapper
    from tensorrt_llm._torch.attention_backend.prims_ts.mla_decode import (
        BatchMLADecodePagedTSWrapper,
    )
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


_MIN_CUTLASS_DSL_VERSION = Version("4.7.0")
_MIN_CUTLASS_COMPILER_VERSION = "13.3"
_WORKSPACE_ALIGNMENT = 32


class PrimsTSFmha(PhasedFmha):
    """Blackwell task-scheduled paged context and decode FMHA library."""

    SUPPORTED_PAGE_SIZES = {16, 32, 64, 128}
    SUPPORTED_CONTEXT_HEAD_DIMS = {128, 256}
    SUPPORTED_DECODE_HEAD_DIMS = {64, 128, 256}
    SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
    MAX_DECODE_GQA_RATIO = 32

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        # Cache each manager/pool's K-to-V page displacement without retaining
        # the manager itself.
        self._kv_page_offset_cache: dict[tuple[int, int], int] = {}
        self._multi_processor_count: Optional[int] = None
        # Every other plan attribute is fixed by this layer/model instance.
        # Batch size is the only execution profile that needs its own wrapper.
        self._context_wrappers: dict[int, "BatchPrefillPagedTSWrapper"] = {}
        self._decode_wrappers: dict[int, "BatchDecodePagedTSWrapper"] = {}
        self._mla_decode_wrappers: dict[int, "BatchMLADecodePagedTSWrapper"] = {}
        # Decode plans retain views into the shared workspace and are invalidated
        # whenever its underlying allocation changes.
        self._workspace_allocation: Optional[tuple[object, ...]] = None
        # Byte range reserved for PrimTS decode scratch within the caller-owned
        # shared FMHA workspace.
        self._decode_workspace_offset_bytes: Optional[int] = None
        self._decode_workspace_required_bytes = 0

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        sm = get_sm_version()
        if sm not in (100, 103):
            logger.debug(f"PrimTS FMHA is unavailable: requires SM100 or SM103, got SM{sm}.")
            return False
        try:
            installed_version = Version(version("nvidia-cutlass-dsl"))
        except (PackageNotFoundError, InvalidVersion):
            logger.debug("PrimTS FMHA is unavailable: nvidia-cutlass-dsl>=4.7 is required.")
            return False
        if installed_version < _MIN_CUTLASS_DSL_VERSION:
            logger.debug(
                "PrimTS FMHA is unavailable: "
                f"nvidia-cutlass-dsl>={_MIN_CUTLASS_DSL_VERSION} is required, "
                f"got {installed_version}."
            )
            return False
        try:
            cutlass = import_module("cutlass")
            compiler_version_supported = cutlass.target_version(
                min_version=_MIN_CUTLASS_COMPILER_VERSION
            )
        except Exception as error:  # noqa: BLE001 - availability probes must fail closed
            logger.debug(
                "PrimTS FMHA is unavailable: could not query the active CUTLASS compiler "
                f"version: {error}"
            )
            return False
        if not compiler_version_supported:
            logger.debug(
                "PrimTS FMHA is unavailable: the active CUTLASS compiler must target "
                f"CUDA>={_MIN_CUTLASS_COMPILER_VERSION}."
            )
            return False
        try:
            import_module("cutlass.experimental.task_scheduling")
        except ImportError:
            logger.debug("PrimTS FMHA is unavailable: CUTLASS task scheduling is missing.")
            return False
        missing_ops = cls._missing_fused_nanobind_ops()
        if missing_ops:
            logger.debug(f"PrimTS FMHA is unavailable: missing nanobind ops {missing_ops}.")
            return False
        return True

    @staticmethod
    def _missing_fused_nanobind_ops() -> list[str]:
        required_ops = (
            "get_trtllm_gen_context_workspace_layout",
            "get_trtllm_gen_generation_workspace_layout",
            "trtllm_gen_context_preprocess",
            "trtllm_gen_context_postprocess",
            "trtllm_gen_generation_preprocess",
            "build_trtllm_gen_kv_cache_metadata",
        )
        return [name for name in required_ops if not hasattr(thop, name)]

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(
            q,
            k,
            v,
            self.attn,
            metadata,
            forward_args,
            phase=phase,
        )
        if not supported:
            logger.debug(f"PrimTS FMHA does not support request: {reason}")
        return supported

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> tuple[bool, str]:
        """Return a conservative, side-effect-free whole-request support decision."""
        # PrimTS prepares workspace for every active request phase before
        # dispatch. Accept the phased dispatcher keyword, but do not narrow
        # support until that preparation is phase-aware too.
        del phase
        if q.device.type != "cuda":
            return False, "CUDA tensors are required."
        if not q.is_contiguous():
            return False, "the fused attention input must be contiguous."
        if k is not None or v is not None:
            return False, "only fused QKV input is supported."
        if not fwd.is_fused_qkv:
            return False, "only fused QKV input is supported."
        if meta.is_cross:
            return False, "cross attention is not supported."
        if meta.kv_cache_manager is None:
            return False, "a KV cache manager is required."
        if meta.kv_cache_block_offsets is None:
            return False, "paged KV-cache block offsets are required."
        if meta.host_kv_cache_pool_pointers is None:
            return False, "KV-cache pool pointers are required."
        if meta.host_kv_cache_pool_mapping is None:
            return False, "KV-cache pool mapping is required."
        if meta.kv_layout != "HND":
            return False, "only HND KV-cache layout is supported."
        kv_cache_manager = meta.kv_cache_manager
        if isinstance(kv_cache_manager, KVCacheManagerV2):
            if kv_cache_manager.enable_swa_scratch_reuse:
                return False, "KVCacheManagerV2 SWA scratch reuse is not supported."
        elif isinstance(kv_cache_manager, KVCacheManager):
            if kv_cache_manager.num_pools != 1:
                return False, "KVCacheManagerV1 with multiple memory pools is not supported."
            pool_mapping = meta.host_kv_cache_pool_mapping
            local_layer_idx = attn.local_layer_idx
            num_local_layers = kv_cache_manager.num_local_layers
            if (
                pool_mapping.ndim != 2
                or pool_mapping.shape[1] < 2
                or local_layer_idx is None
                or local_layer_idx < 0
                or local_layer_idx >= pool_mapping.shape[0]
            ):
                return False, "KVCacheManagerV1 has an invalid layer-to-pool mapping."
            pool_index = int(pool_mapping[local_layer_idx, 0])
            layer_idx_in_pool = int(pool_mapping[local_layer_idx, 1])
            if pool_index != 0 or not 0 <= layer_idx_in_pool < num_local_layers:
                return False, "KVCacheManagerV1 has an invalid layer-to-pool mapping."
        else:
            return False, f"unsupported KV cache manager {type(kv_cache_manager).__name__}."

        output = fwd.output
        if output is None:
            return False, "an output tensor is required."
        if output.device != q.device or not output.is_contiguous():
            return False, "output must be a contiguous tensor on the query device."
        if fwd.output_sf is not None or fwd.out_scale is not None:
            return False, "quantized attention output is not supported."

        if attn.sparse_params is not None:
            return False, "sparse attention is not supported."
        if (
            fwd.sparse_runtime_params.sparse_kv_indices is not None
            or fwd.sparse_runtime_params.sparse_attn_indices is not None
        ):
            return False, "sparse attention metadata is not supported."
        if meta.num_sparse_topk > 0:
            return False, "sparse attention metadata is not supported."
        if meta.helix_position_offsets is not None:
            return False, "Helix parallelism is not supported."
        if fwd.relative_attention_bias is not None:
            return False, "relative attention bias is not supported."
        if fwd.attention_sinks is not None:
            return False, "attention sinks are not supported."
        if fwd.attention_mask_data is not None:
            return False, "custom attention masks are not supported."
        if fwd.enable_dsv4_epilogue_fusion:
            return False, "DSv4 epilogue fusion is not supported."
        if (
            fwd.sage_attn_num_elts_per_blk_q > 0
            or fwd.sage_attn_num_elts_per_blk_k > 0
            or fwd.sage_attn_num_elts_per_blk_v > 0
        ):
            return False, "SageAttention is not supported."

        if meta.beam_width != 1:
            return False, "beam search is not supported."
        if (
            meta.is_spec_decoding_enabled
            or meta.use_spec_decoding
            or meta.is_spec_dec_tree
            or meta.is_spec_dec_dynamic_tree
        ):
            return False, "speculative decoding is not supported by the initial adapter."

        try:
            mask_type = AttentionMaskType(fwd.mask_type)
        except (AttributeError, TypeError, ValueError):
            return False, "the attention mask is not causal or dense."
        if mask_type not in (AttentionMaskType.causal, AttentionMaskType.padding):
            return False, f"attention mask type {mask_type} is not supported."

        position_embedding_type = int(attn.position_embedding_type)
        if position_embedding_type in (4, 5, 6, 7, 10):
            return False, f"position embedding type {position_embedding_type} is not supported."

        try:
            quant_mode = QuantMode(attn.quant_mode)
        except (TypeError, ValueError):
            return False, "invalid KV-cache quantization mode."
        if quant_mode.has_kv_cache_quant():
            return False, "quantized KV cache is not supported by the initial adapter."

        input_type = fwd.attention_input_type
        if input_type not in (
            AttentionInputType.context_only,
            AttentionInputType.generation_only,
            AttentionInputType.mixed,
        ):
            return False, f"invalid attention input type {input_type}."
        num_contexts = int(meta.num_contexts)
        num_generations = int(meta.num_generations)
        has_context = num_contexts > 0 and input_type != AttentionInputType.generation_only
        has_generation = num_generations > 0 and input_type != AttentionInputType.context_only
        if not has_context and not has_generation:
            return False, "the request contains no active attention phase."
        if has_context and meta.is_cuda_graph:
            return False, "context planning is not CUDA-graph capturable."
        if has_context and (attn.attention_chunk_size or 0) != 0:
            return False, "chunked context attention is not supported."

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block not in self.SUPPORTED_PAGE_SIZES:
            return False, (
                f"page size {tokens_per_block} is unsupported; "
                f"supported sizes are {sorted(self.SUPPORTED_PAGE_SIZES)}."
            )
        if attn.num_heads <= 0 or attn.num_kv_heads <= 0:
            return False, "query and KV head counts must be positive."
        is_mla = attn.is_mla_enable
        if is_mla:
            if attn.num_kv_heads != 1:
                return False, "MLA decode requires one logical KV head."
            if attn.num_heads > 128:
                return False, "MLA decode supports at most 128 local query heads."
        else:
            if attn.num_heads % attn.num_kv_heads != 0:
                return False, "the query head count must be divisible by the KV head count."
            if has_generation and attn.num_heads // attn.num_kv_heads > self.MAX_DECODE_GQA_RATIO:
                return False, f"decode GQA ratio exceeds {self.MAX_DECODE_GQA_RATIO}."

        if q.dtype not in self.SUPPORTED_DTYPES:
            return False, f"query dtype {q.dtype} is unsupported."
        cache_dtype = binding_to_torch_dtype(meta.kv_cache_manager.dtype)
        if cache_dtype != q.dtype:
            return False, f"query and KV-cache dtypes must match, got {q.dtype} and {cache_dtype}."
        if output.dtype != q.dtype:
            return False, f"output dtype must match query dtype, got {output.dtype} and {q.dtype}."

        if q.ndim != 2:
            return False, f"fused attention input must be rank 2, got rank {q.ndim}."
        if is_mla:
            if has_context or input_type != AttentionInputType.generation_only:
                return False, "MLA is supported only for generation-only requests."
            if q.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
                return False, "MLA decode requires BF16 query, cache, and output."
            if attn.kv_lora_rank != 512 or attn.qk_rope_head_dim != 64:
                return False, (
                    "MLA decode requires kv_lora_rank=512 and qk_rope_head_dim=64, got "
                    f"{attn.kv_lora_rank} and {attn.qk_rope_head_dim}."
                )
            if attn.head_dim != attn.kv_lora_rank + attn.qk_rope_head_dim:
                return False, "MLA head dimension must equal the latent plus RoPE dimensions."
            if attn.qk_nope_head_dim is None or attn.qk_nope_head_dim <= 0:
                return False, "MLA decode requires a positive qk_nope_head_dim."
            expected_width = attn.num_heads * (attn.kv_lora_rank + attn.qk_rope_head_dim)
            if q.shape[1] != expected_width:
                return False, f"MLA query width must be {expected_width}, got {q.shape[1]}."
            if output.numel() != q.shape[0] * attn.num_heads * attn.kv_lora_rank:
                return False, "MLA output has an incompatible extent."
        else:
            expected_width = (attn.num_heads + 2 * attn.num_kv_heads) * attn.head_dim
            if q.shape[1] != expected_width:
                return False, f"fused QKV width must be {expected_width}, got {q.shape[1]}."
            if output.numel() != q.shape[0] * attn.num_heads * attn.head_dim:
                return False, "attention output has an incompatible extent."
            if has_context and attn.head_dim not in self.SUPPORTED_CONTEXT_HEAD_DIMS:
                return False, f"context head dimension {attn.head_dim} is unsupported."
            if has_generation and attn.head_dim not in self.SUPPORTED_DECODE_HEAD_DIMS:
                return False, f"decode head dimension {attn.head_dim} is unsupported."

        num_ctx_tokens = int(meta.num_ctx_tokens)
        num_gen_tokens = (
            q.shape[0]
            if input_type == AttentionInputType.generation_only
            else q.shape[0] - num_ctx_tokens
        )
        if has_generation and (
            num_gen_tokens <= 0 or num_generations <= 0 or num_gen_tokens % num_generations != 0
        ):
            return False, "generation tokens must be uniformly divisible across requests."
        if has_generation and num_gen_tokens != num_generations:
            return False, "only single-token generation is supported by the initial adapter."

        host_kv_lens = meta.kv_lens_runtime
        if host_kv_lens is None or host_kv_lens.numel() < num_contexts + num_generations:
            return False, "host KV lengths are required for safe policy selection."
        active_kv_lens = host_kv_lens[: num_contexts + num_generations]
        if active_kv_lens.numel() == 0 or int(active_kv_lens.min()) <= 0:
            return False, "every active request must contain at least one KV token."
        max_kv_length = int(active_kv_lens.max())
        max_seq_len = int(meta.max_seq_len)
        if max_kv_length > max_seq_len:
            return False, "an active KV length exceeds the configured maximum sequence length."
        attention_window_size = fwd.attention_window_size
        if not isinstance(attention_window_size, int) or attention_window_size <= 0:
            return False, "attention_window_size must be a positive integer."
        if attention_window_size < max_seq_len:
            return False, (
                "sliding-window attention uses cyclic TRT-LLM page tables, which are not "
                "compatible with the PrimTS fixed row-strided page-table ABI."
            )

        if (
            not is_mla
            and get_kv_page_offset(
                attn,
                meta,
                0,
                cache=self._kv_page_offset_cache,
            )
            is None
        ):
            return False, "the K-to-V page displacement could not be resolved."
        return True, ""

    @staticmethod
    def _get_fixed_block_tables(
        block_tables: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return the live TRT-LLM K-page table as a zero-copy row-strided view."""
        if block_tables.ndim != 3 or block_tables.shape[1] != 2:
            raise RuntimeError(
                "PrimTS expects block tables with shape [batch, 2, max_blocks], got "
                f"{tuple(block_tables.shape)}."
            )
        if block_tables.dtype != torch.int32:
            raise RuntimeError(f"PrimTS expects int32 block tables, got {block_tables.dtype}.")
        if batch_size <= 0 or batch_size > block_tables.shape[0]:
            raise RuntimeError(
                f"Invalid PrimTS block-table batch size {batch_size} for "
                f"{block_tables.shape[0]} rows."
            )
        return block_tables[:batch_size, 0, :]

    @staticmethod
    def _get_sequence_lengths(
        sequence_lengths: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return the live active sequence-length view required by PrimTS."""
        if sequence_lengths.dtype != torch.int32:
            raise RuntimeError(
                f"PrimTS expects int32 sequence lengths, got {sequence_lengths.dtype}."
            )
        if batch_size <= 0 or batch_size > sequence_lengths.numel():
            raise RuntimeError(
                f"Invalid PrimTS sequence-length batch size {batch_size} for "
                f"{sequence_lengths.numel()} entries."
            )
        return sequence_lengths[:batch_size]

    def _update_workspace_allocation(self, workspace: torch.Tensor) -> None:
        """Invalidate plans that retain views into a reallocated workspace."""

        storage = workspace.untyped_storage()
        allocation = (
            workspace.device,
            storage.data_ptr(),
            storage.nbytes(),
            workspace.storage_offset(),
            workspace.numel(),
            workspace.element_size(),
        )
        if allocation == self._workspace_allocation:
            return
        self._workspace_allocation = allocation
        self._decode_wrappers.clear()
        self._mla_decode_wrappers.clear()

    def _get_or_plan_context_wrapper(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        batch_size: int,
        max_seq_len_q: int,
        max_kv_len: int,
        page_size: int,
        mask_type: str,
        window_left: int,
        sm_scale: float,
        output_dtype: torch.dtype,
    ) -> "BatchPrefillPagedTSWrapper":
        wrapper = self._context_wrappers.get(batch_size)
        if wrapper is not None:
            return wrapper
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PrimTS context must be planned before CUDA graph capture.")
        from tensorrt_llm._torch.attention_backend.prims_ts.context import (
            BatchPrefillPagedTSWrapper,
        )

        wrapper = BatchPrefillPagedTSWrapper(kv_layout="HND")
        wrapper.plan(
            device=q.device,
            batch_size=batch_size,
            max_seq_len_q=max_seq_len_q,
            max_kv_len=max_kv_len,
            num_qo_heads=int(q.shape[-2]),
            num_kv_heads=int(k_cache.shape[1]),
            head_dim=int(q.shape[-1]),
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            out_dtype=output_dtype,
            page_size=page_size,
            mask_type=mask_type,
            window_left=window_left,
            sm_scale=sm_scale,
            output_scale=1.0,
        )
        self._context_wrappers[batch_size] = wrapper
        return wrapper

    def _get_or_plan_decode_wrapper(
        self,
        workspace_buffer: torch.Tensor,
        *,
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        seq_len_q: int,
        max_kv_len: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        output_dtype: torch.dtype,
        mask_type: str,
        window_left: int,
    ) -> "BatchDecodePagedTSWrapper":
        if batch_size <= 0:
            raise RuntimeError("PrimTS decode requires a positive batch size.")
        wrapper = self._decode_wrappers.get(batch_size)
        if wrapper is not None:
            return wrapper
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PrimTS decode must be planned before CUDA graph capture.")
        from tensorrt_llm._torch.attention_backend.prims_ts.decode import BatchDecodePagedTSWrapper

        wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
        wrapper.plan(
            workspace_buffer.device,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_kv_len,
            max_seq_len_q=seq_len_q,
            packed_query=False,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=output_dtype,
            mask_type=mask_type,
            window_left=window_left,
            workspace_buffer=workspace_buffer,
        )
        self._decode_wrappers[batch_size] = wrapper
        return wrapper

    def _get_or_plan_mla_decode_wrapper(
        self,
        workspace_buffer: torch.Tensor,
        *,
        batch_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        max_seq_len_q: int,
        max_kv_len: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        output_dtype: torch.dtype,
        mask_type: str,
    ) -> "BatchMLADecodePagedTSWrapper":
        if batch_size <= 0:
            raise RuntimeError("PrimTS MLA decode requires a positive batch size.")
        wrapper = self._mla_decode_wrappers.get(batch_size)
        if wrapper is not None:
            return wrapper
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PrimTS MLA decode must be planned before CUDA graph capture.")
        from tensorrt_llm._torch.attention_backend.prims_ts.mla_decode import (
            BatchMLADecodePagedTSWrapper,
        )

        wrapper = BatchMLADecodePagedTSWrapper()
        wrapper.plan(
            workspace_buffer.device,
            batch_size,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            max_kv_len,
            max_seq_len_q=max_seq_len_q,
            packed_query=False,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=output_dtype,
            mask_type=mask_type,
            workspace_buffer=workspace_buffer,
        )
        self._mla_decode_wrappers[batch_size] = wrapper
        return wrapper

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        del k, v
        block_offsets = metadata.kv_cache_block_offsets
        if block_offsets is None:
            raise RuntimeError("PrimTS requires paged KV-cache block offsets.")
        column_capacity = int(block_offsets.shape[-1])
        input_type = forward_args.attention_input_type
        has_context = metadata.num_contexts > 0 and input_type != AttentionInputType.generation_only
        has_generation = (
            metadata.num_generations > 0 and input_type != AttentionInputType.context_only
        )
        if self._multi_processor_count is None:
            self._multi_processor_count = torch.cuda.get_device_properties(
                q.device
            ).multi_processor_count

        required_preprocess_bytes = 0
        if has_context and not self.attn.is_mla_enable:
            context_layout = thop.get_trtllm_gen_context_workspace_layout(
                q.dtype,
                int(metadata.num_contexts),
                int(metadata.num_ctx_tokens),
                self.attn.num_heads,
                self.attn.head_dim,
                self.attn.rope_dim,
                True,
                False,
                skip_fmha_workspace=True,
            )
            required_preprocess_bytes = max(
                required_preprocess_bytes, int(context_layout["total_size"])
            )
        if has_generation and not self.attn.is_mla_enable:
            num_gen_tokens_for_layout = (
                q.shape[0]
                if input_type == AttentionInputType.generation_only
                else q.shape[0] - int(metadata.num_ctx_tokens)
            )
            generation_layout = thop.get_trtllm_gen_generation_workspace_layout(
                q.dtype,
                int(metadata.num_generations),
                num_gen_tokens_for_layout,
                self.attn.num_heads,
                self.attn.head_dim,
                self.attn.rope_dim,
                self.attn.num_kv_heads,
                0,
                False,
                skip_fmha_workspace=True,
            )
            required_preprocess_bytes = max(
                required_preprocess_bytes, int(generation_layout["total_size"])
            )

        if not has_generation:
            current_workspace_bytes = workspace.numel() * workspace.element_size()
            if current_workspace_bytes < required_preprocess_bytes:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "TRT-LLM QKV preprocessing workspace must be sized before "
                        "CUDA graph capture."
                    )
                required_numel = ceil_div(required_preprocess_bytes, workspace.element_size())
                workspace.resize_((required_numel,))
            self._update_workspace_allocation(workspace)
            return

        batch_size = int(metadata.num_generations)
        num_gen_tokens = (
            q.shape[0]
            if input_type == AttentionInputType.generation_only
            else q.shape[0] - int(metadata.num_ctx_tokens)
        )
        seq_len_q = num_gen_tokens // batch_size
        max_seq_len = column_capacity * int(metadata.tokens_per_block)
        mask_type = self._get_prims_mask_type(forward_args)
        # is_supported() rejects cyclic sliding-window page tables. Keep the
        # sizing key aligned with the non-windowed plan selected at runtime.
        window_left = -1

        if self.attn.is_mla_enable:
            from tensorrt_llm._torch.attention_backend.prims_ts import (
                get_prims_ts_batch_decode_mla_workspace_size,
            )

            required_bytes = get_prims_ts_batch_decode_mla_workspace_size(
                batch_size,
                self.attn.num_heads,
                int(self.attn.kv_lora_rank),
                int(self.attn.qk_rope_head_dim),
                int(metadata.tokens_per_block),
                max_seq_len,
                max_seq_len_q=seq_len_q,
                q_dtype=q.dtype,
                kv_dtype=q.dtype,
                out_dtype=forward_args.output.dtype,
                mask_type=mask_type,
                device=q.device,
            )
        else:
            from tensorrt_llm._torch.attention_backend.prims_ts import (
                get_prims_ts_batch_decode_workspace_size,
            )

            required_bytes = get_prims_ts_batch_decode_workspace_size(
                batch_size,
                self.attn.num_heads,
                self.attn.num_kv_heads,
                self.attn.head_dim,
                int(metadata.tokens_per_block),
                max_seq_len,
                seq_len_q=seq_len_q,
                q_dtype=q.dtype,
                kv_dtype=q.dtype,
                out_dtype=forward_args.output.dtype,
                mask_type=mask_type,
                window_left=window_left,
                device=q.device,
            )

        decode_workspace_min_offset_bytes: Optional[int] = None
        if self.attn.is_mla_enable:
            required_workspace_bytes = required_bytes
        else:
            self._decode_workspace_required_bytes = required_bytes
            # QKV preprocessing leaves its query and sequence metadata live
            # while PrimTS runs. Keep PrimTS scratch in a separate aligned tail,
            # anchored to the final root allocation so cached batch profiles
            # retain a stable address across mixed-context layout changes.
            # FlashInfer requires the caller workspace base to be 32-byte aligned.
            decode_workspace_min_offset_bytes = pad_up(
                required_preprocess_bytes, _WORKSPACE_ALIGNMENT
            )
            required_workspace_bytes = decode_workspace_min_offset_bytes + required_bytes
        current_workspace_bytes = workspace.numel() * workspace.element_size()
        if current_workspace_bytes < required_workspace_bytes:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "PrimTS caller workspace must be sized before CUDA graph capture."
                )
            required_numel = ceil_div(required_workspace_bytes, workspace.element_size())
            workspace.resize_((required_numel,))
        if decode_workspace_min_offset_bytes is not None:
            current_workspace_bytes = workspace.numel() * workspace.element_size()
            available_tail_bytes = current_workspace_bytes - required_bytes
            decode_workspace_offset_bytes = (
                available_tail_bytes // _WORKSPACE_ALIGNMENT * _WORKSPACE_ALIGNMENT
            )
            if decode_workspace_offset_bytes < decode_workspace_min_offset_bytes:
                raise RuntimeError("PrimTS decode workspace tail does not fit its root allocation.")
            self._decode_workspace_offset_bytes = decode_workspace_offset_bytes
        self._update_workspace_allocation(workspace)

    @staticmethod
    def _get_prims_mask_type(forward_args: AttentionForwardArgs) -> str:
        mask_type = AttentionMaskType(forward_args.mask_type)
        return "causal" if mask_type == AttentionMaskType.causal else "dense"

    @staticmethod
    def _get_bmm1_scale(attn: "TrtllmAttention") -> float:
        return 1.0 / (math.sqrt(attn.head_dim) * attn.q_scaling)

    @staticmethod
    def _standard_kv_views(
        kv_pool: torch.Tensor,
        kv_page_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_pool.ndim != 4:
            raise RuntimeError(
                f"PrimTS expects a flat rank-4 KV page pool, got {tuple(kv_pool.shape)}."
            )
        usable_pages = kv_pool.shape[0] - kv_page_offset
        if kv_page_offset <= 0 or usable_pages <= 0:
            raise RuntimeError(
                f"Invalid PrimTS K-to-V page displacement {kv_page_offset} "
                f"for a {kv_pool.shape[0]}-page pool."
            )
        return (
            kv_pool.narrow(0, 0, usable_pages),
            kv_pool.narrow(0, kv_page_offset, usable_pages),
        )

    def run_context(self, params: FmhaParams) -> None:
        if params.qkv_input is None or params.context_buf is None:
            raise RuntimeError("PrimTS context requires QKV input and an output buffer.")
        if params.sequence_lengths is None or params.context_lengths is None:
            raise RuntimeError("PrimTS context requires sequence and context lengths.")
        if self._multi_processor_count is None:
            raise RuntimeError("PrimTS context workspace was not prepared.")

        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        attention_chunk_size = attn.attention_chunk_size or 0
        (
            q_processed,
            kv_pool,
            block_tables,
            _kv_scale_pool,
            _bmm1_scale,
            _bmm2_scale,
            fmha_workspace,
            cu_q_seqlens,
            _cu_kv_seqlens,
            _max_q_len,
            _max_kv_len,
            window_left,
        ) = thop.trtllm_gen_context_preprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.context_lengths,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_inv_freq,
            attn.rotary_cos_sin,
            fwd.mrope_rotary_cos_sin,
            attn.local_layer_idx,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            fwd.mask_type,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            params.batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            self._get_bmm1_scale(attn),
            1.0,
            attention_chunk_size,
            False,
            True,
            False,
            self._multi_processor_count,
            params.total_num_blocks,
            params.kv_factor,
            True,
            fwd.cross_kv,
            False,
            skip_fmha_workspace=True,
        )
        if fmha_workspace.numel() != 0:
            raise RuntimeError("PrimTS context preprocessing returned an FMHA workspace.")
        # The returned pool and block table share the THOP flat-page index ABI.
        if kv_pool is None or block_tables is None:
            raise RuntimeError("TRT-LLM preprocessing did not return PrimTS KV metadata.")
        kv_page_offset = get_kv_page_offset(
            attn,
            meta,
            params.seq_offset,
            cache=self._kv_page_offset_cache,
        )
        if kv_page_offset is None:
            raise RuntimeError("PrimTS could not resolve the K-to-V page displacement.")
        k_cache, v_cache = self._standard_kv_views(kv_pool, kv_page_offset)
        max_seq_len_q = int(meta.max_context_length)
        max_kv_len = int(meta.max_seq_len)
        fixed_block_tables = self._get_fixed_block_tables(block_tables, params.batch_size)
        seq_lens_kv = self._get_sequence_lengths(
            params.sequence_lengths,
            params.batch_size,
        )
        mask_type = self._get_prims_mask_type(fwd)
        wrapper = self._get_or_plan_context_wrapper(
            q_processed,
            k_cache,
            v_cache,
            batch_size=params.batch_size,
            max_seq_len_q=max_seq_len_q,
            max_kv_len=max_kv_len,
            page_size=params.tokens_per_block,
            mask_type=mask_type,
            window_left=window_left,
            sm_scale=self._get_bmm1_scale(attn),
            output_dtype=params.context_buf.dtype,
        )
        wrapper.run(
            q_processed,
            k_cache,
            v_cache,
            cu_q_seqlens,
            block_tables=fixed_block_tables,
            seq_lens_kv=seq_lens_kv,
            out=params.context_buf,
            validate=False,
        )

        thop.trtllm_gen_context_postprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.context_lengths,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_cos_sin,
            fwd.mrope_rotary_cos_sin,
            attn.local_layer_idx,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            fwd.mask_type,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            params.batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            self._get_bmm1_scale(attn),
            False,
            True,
            False,
            attention_chunk_size,
            self._multi_processor_count,
            skip_fmha_workspace=True,
        )

    def run_generation(self, params: FmhaParams) -> None:
        if params.qkv_input is None or params.context_buf is None:
            raise RuntimeError("PrimTS decode requires QKV input and an output buffer.")
        if params.sequence_lengths is None:
            raise RuntimeError("PrimTS decode requires sequence lengths.")
        if self._multi_processor_count is None:
            raise RuntimeError("PrimTS decode workspace was not prepared.")

        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        batch_size = params.batch_size
        attention_chunk_size = attn.attention_chunk_size or 0
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
            window_left,
            is_multi_token_gen,
        ) = thop.trtllm_gen_generation_preprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.spec_decoding_generation_lengths,
            params.spec_decoding_position_offsets,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_inv_freq,
            attn.rotary_cos_sin,
            fwd.mrope_position_deltas,
            attn.local_layer_idx,
            params.seq_offset,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            self._get_bmm1_scale(attn),
            1.0,
            False,
            attn.predicted_tokens_per_seq,
            attention_chunk_size,
            self._multi_processor_count,
            params.total_num_blocks,
            params.kv_factor,
            True,
            False,
            skip_fmha_workspace=True,
        )
        if fmha_workspace.numel() != 0:
            raise RuntimeError("PrimTS generation preprocessing returned an FMHA workspace.")
        if is_multi_token_gen:
            raise RuntimeError("PrimTS was selected for unsupported speculative decoding.")
        # The returned pool and block table share the THOP flat-page index ABI.
        if kv_pool is None or block_tables is None:
            raise RuntimeError("TRT-LLM preprocessing did not return PrimTS KV metadata.")
        kv_page_offset = get_kv_page_offset(
            attn,
            meta,
            params.seq_offset,
            cache=self._kv_page_offset_cache,
        )
        if kv_page_offset is None:
            raise RuntimeError("PrimTS could not resolve the K-to-V page displacement.")
        k_cache, v_cache = self._standard_kv_views(kv_pool, kv_page_offset)
        fixed_block_tables = self._get_fixed_block_tables(block_tables, batch_size)
        max_seq_len = int(block_tables.shape[-1]) * params.tokens_per_block
        seq_lens = self._get_sequence_lengths(params.sequence_lengths, batch_size)
        query = q_processed.view(
            batch_size,
            params.input_seq_length,
            attn.num_heads,
            attn.head_dim,
        )
        output = params.context_buf.view_as(query)
        if params.input_seq_length == 1:
            query = query[:, 0]
            output = output[:, 0]
        mask_type = self._get_prims_mask_type(fwd)
        decode_workspace = self._get_decode_workspace(params.workspace)
        wrapper = self._get_or_plan_decode_wrapper(
            decode_workspace,
            batch_size=batch_size,
            num_qo_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            page_size=params.tokens_per_block,
            seq_len_q=params.input_seq_length,
            max_kv_len=max_seq_len,
            q_dtype=query.dtype,
            kv_dtype=k_cache.dtype,
            output_dtype=output.dtype,
            mask_type=mask_type,
            window_left=window_left,
        )
        plan_state = wrapper._plan_state
        if plan_state is None:
            raise RuntimeError("PrimTS decode wrapper has no compiled plan.")
        # Only the fused global-memory reducer consumes the split-KV counter.
        # Direct, cluster-reduced, and separately reduced plans do not read it.
        policy = dict(plan_state.policy)
        requires_control_reset = bool(policy["use_split_kv"]) and not (
            bool(policy["use_separate_reduction_kernel"])
            or bool(policy["use_cluster_smem_reduction"])
        )
        if requires_control_reset:
            plan_state.workspace.split_kv_counter.zero_()
        wrapper.run(
            query,
            (k_cache, v_cache),
            seq_lens,
            block_tables=fixed_block_tables,
            bmm1_scale=self._get_bmm1_scale(attn),
            bmm2_scale=1.0,
            out=output,
            validate=False,
        )

    def _get_decode_workspace(
        self,
        root_workspace: torch.Tensor,
    ) -> torch.Tensor:
        """Return the caller-owned PrimTS tail after QKV preprocessing storage."""

        byte_offset = self._decode_workspace_offset_bytes
        if byte_offset is None:
            raise RuntimeError("PrimTS decode workspace was not prepared.")
        root_bytes = root_workspace.reshape(-1).view(torch.uint8)
        byte_end = byte_offset + self._decode_workspace_required_bytes
        if byte_end > root_bytes.numel():
            raise RuntimeError("PrimTS decode workspace was not sized before kernel execution.")
        return root_bytes[byte_offset:byte_end]

    def run_mla_generation(self, params: FmhaParams) -> None:
        if params.qkv_input is None or params.context_buf is None:
            raise RuntimeError("PrimTS MLA decode requires query input and an output buffer.")
        if params.sequence_lengths is None:
            raise RuntimeError("PrimTS MLA decode requires sequence lengths.")

        attn = params.attn
        meta = params.meta
        batch_size = params.batch_size
        kv_cache, block_tables, _kv_scale_pool = thop.build_trtllm_gen_kv_cache_metadata(
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            meta.kv_cache_block_offsets,
            attn.local_layer_idx,
            attn.num_kv_heads,
            params.tokens_per_block,
            attn.head_dim,
            params.kv_factor,
            params.total_num_blocks,
            attn.quant_mode,
            params.seq_offset,
            batch_size,
            params.qkv_input.dtype,
        )
        # The returned pool and block table share the THOP flat-page index ABI.
        if kv_cache is None or block_tables is None:
            raise RuntimeError("TRT-LLM did not return PrimTS MLA KV metadata.")
        fixed_block_tables = self._get_fixed_block_tables(block_tables, batch_size)
        seq_len_q = params.input_seq_length
        query = params.qkv_input.view(
            batch_size,
            seq_len_q,
            attn.num_heads,
            int(attn.kv_lora_rank) + int(attn.qk_rope_head_dim),
        )
        output = params.context_buf.view(
            batch_size,
            seq_len_q,
            attn.num_heads,
            int(attn.kv_lora_rank),
        )
        max_seq_len = int(block_tables.shape[-1]) * params.tokens_per_block
        bmm1_scale = 1.0 / (
            attn.q_scaling * math.sqrt(int(attn.qk_nope_head_dim) + int(attn.qk_rope_head_dim))
        )
        mask_type = self._get_prims_mask_type(params.fwd)
        seq_lens = self._get_sequence_lengths(
            params.sequence_lengths,
            batch_size,
        )
        caller_workspace = params.workspace.reshape(-1).view(torch.uint8)
        wrapper = self._get_or_plan_mla_decode_wrapper(
            caller_workspace,
            batch_size=batch_size,
            num_heads=attn.num_heads,
            kv_lora_rank=int(attn.kv_lora_rank),
            qk_rope_head_dim=int(attn.qk_rope_head_dim),
            page_size=params.tokens_per_block,
            max_seq_len_q=seq_len_q,
            max_kv_len=max_seq_len,
            q_dtype=query.dtype,
            kv_dtype=kv_cache.dtype,
            output_dtype=output.dtype,
            mask_type=mask_type,
        )
        wrapper.run(
            query,
            kv_cache,
            block_tables=fixed_block_tables,
            seq_lens=seq_lens,
            out=output,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            validate=False,
        )


__all__ = ["PrimsTSFmha"]
