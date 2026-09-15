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

"""TRT-LLM FMHA adapter for the vendored PrimTS block-sparse kernels."""

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal, cast

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.prims_ts._block_sparse.config import (
    _validate_block_sparse_static_profile,
)
from tensorrt_llm._torch.attention.backends.sparse.params import BlockSparseForwardInputs
from tensorrt_llm.logger import logger

from .interface import FmhaPhase
from .phased import FmhaParams
from .prims_ts import (
    PrimsTSFmha,
    get_attention_feature_unsupported_reason,
    get_paged_kv_policy_unsupported_reason,
    get_paged_kv_storage_unsupported_reason,
)
from .utils import get_kv_page_offset, get_multi_processor_count_for_device

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.prims_ts import (
        BlockSparsePagedTSWrapper,
        BlockSparseTSWrapper,
    )
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

from tensorrt_llm._torch.attention.backends.prims_ts import (
    BlockSparsePagedTSWrapper as _BlockSparsePagedTSWrapper,
)
from tensorrt_llm._torch.attention.backends.prims_ts import (
    BlockSparseTSWrapper as _BlockSparseTSWrapper,
)


@dataclass(frozen=True, slots=True)
class _BlockSparsePlanKey:
    """Static wrapper profile shared by compatible attention layers.

    The key is the single description of a plan: support checks validate it
    against the kernel library and the wrapper cache plans from it. Every field
    is an argument of ``plan()``, so two requests share a planned wrapper
    exactly when the kernel could serve both with one plan. Per-layer constants
    such as the head topology and dtype stay in the key because a bound cache
    may be shared by layers with different geometries.
    """

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

    def unsupported_reason(self) -> str | None:
        try:
            _validate_block_sparse_static_profile(
                batch_size=self.batch_size,
                seq_len_q=self.seq_len_q,
                seq_len_kv=self.kv_capacity,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                q_block_size=self.q_block_size,
                kv_block_size=self.kv_block_size,
                use_kv_valid_bits=self.use_kv_valid_bits,
                mask_type=self.mask_type,
                q_dtype=self.dtype,
                kv_dtype=self.dtype,
                output_dtype=self.dtype,
                max_blocks_per_row=self.max_blocks_per_row,
                page_size=self.page_size,
            )
        except (ValueError, NotImplementedError, OverflowError) as error:
            return str(error)
        return None

    def plan(self) -> "BlockSparseTSWrapper | BlockSparsePagedTSWrapper":
        paged = self.page_size is not None
        wrapper_type = _BlockSparsePagedTSWrapper if paged else _BlockSparseTSWrapper
        assert wrapper_type is not None
        wrapper = wrapper_type()
        plan_args = (
            self.batch_size,
            self.seq_len_q,
            self.kv_capacity,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_block_size,
            self.kv_block_size,
        )
        plan_kwargs = {
            "device": self.device,
            "max_blocks_per_row": self.max_blocks_per_row,
            "use_kv_valid_bits": self.use_kv_valid_bits,
            "mask_type": self.mask_type,
            "q_data_type": self.dtype,
            "kv_data_type": self.dtype,
            "o_data_type": self.dtype,
        }
        if paged:
            plan_args += (self.page_size,)
        else:
            plan_kwargs.update(
                sparse_format=self.sparse_format,
                use_proxy_routes=self.use_proxy_routes,
            )
        wrapper.plan(*plan_args, **plan_kwargs)
        return wrapper


def _get_block_sparse_inputs(
    forward_args: AttentionForwardArgs,
) -> BlockSparseForwardInputs | None:
    return forward_args.sparse_runtime_params.block_sparse_inputs


def _has_other_sparse_runtime(forward_args: AttentionForwardArgs) -> bool:
    """Whether the runtime carrier holds any sparse state besides block-sparse routes."""
    params = forward_args.sparse_runtime_params
    for field in fields(params):
        if field.name == "block_sparse_inputs":
            continue
        value = getattr(params, field.name)
        if isinstance(value, torch.Tensor) or (value is not None and value != 0):
            return True
    return False


def _route_batch_size(inputs: BlockSparseForwardInputs) -> int:
    routes = inputs.block_indptr if inputs.sparse_format == "bsr" else inputs.exact_block_bits
    return int(routes.shape[0])


def _uniform_seq_len_q(
    q: torch.Tensor,
    metadata: "TrtllmAttentionMetadata",
    batch_size: int,
) -> int | None:
    """Return the fixed per-request query length, or ``None`` if the batch is ragged."""
    seq_lens = metadata.seq_lens
    if batch_size <= 0 or q.shape[0] % batch_size:
        return None
    if seq_lens is None or seq_lens.numel() < batch_size:
        return None
    seq_len_q = int(q.shape[0]) // batch_size
    if not bool(seq_lens[:batch_size].eq(seq_len_q).all()):
        return None
    return seq_len_q


class PrimsTSBlockSparseFmha(PrimsTSFmha):
    """Contiguous context and fixed-Q paged generation block-sparse FMHA."""

    supports_block_sparse_inputs = True

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        self._contiguous_wrappers: dict[_BlockSparsePlanKey, "BlockSparseTSWrapper"] = {}
        self._paged_wrappers: dict[_BlockSparsePlanKey, "BlockSparsePagedTSWrapper"] = {}

    def bind_plan_cache(self, cache_state: dict[str, object]) -> None:
        """Share planned wrappers with every adapter bound to ``cache_state``.

        Attention layers that execute serially, such as the blocks of one
        diffusion transformer, see identical static profiles. Binding them to
        one model-scoped container plans each profile once and allocates its
        route workspace once. Call before the first forward.
        """

        self._contiguous_wrappers = cast(
            dict[_BlockSparsePlanKey, "BlockSparseTSWrapper"],
            cache_state.setdefault("contiguous_wrappers", {}),
        )
        self._paged_wrappers = cast(
            dict[_BlockSparsePlanKey, "BlockSparsePagedTSWrapper"],
            cache_state.setdefault("paged_wrappers", {}),
        )

    def _is_supported(
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
        reason = self._common_unsupported_reason(metadata, forward_args)
        if reason is None:
            paged = metadata.kv_cache_manager is not None
            expected_phase = FmhaPhase.GENERATION if paged else FmhaPhase.CONTEXT
            if phase not in (None, expected_phase):
                storage = "paged" if paged else "contiguous"
                reason = (
                    f"{storage} block-sparse attention only supports the "
                    f"{expected_phase.name.lower()} phase"
                )
            elif paged:
                reason = self._paged_unsupported_reason(q, metadata, forward_args)
            else:
                reason = self._contiguous_unsupported_reason(q, k, v, metadata, forward_args)
        return reason is None, reason or ""

    def _common_unsupported_reason(
        self,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> str | None:
        """Gates shared by the contiguous and paged block-sparse paths."""
        if _get_block_sparse_inputs(forward_args) is None:
            return "block-sparse forward inputs are required"
        if metadata.is_cross:
            return "cross attention is not supported"
        if self.attn.is_mla_enable:
            return "MLA is not supported"
        if metadata.num_sparse_topk > 0 or _has_other_sparse_runtime(forward_args):
            return "legacy sparse attention cannot be combined with block-sparse inputs"
        feature_reason = get_attention_feature_unsupported_reason(metadata, forward_args)
        if feature_reason is not None:
            return feature_reason
        if forward_args.softmax_stats_tensor is not None:
            return "softmax statistics output is not supported"
        if (
            forward_args.output_sf is not None
            or forward_args.out_scale is not None
            or forward_args.out_scale_sf is not None
        ):
            return "quantized output is not supported"
        if forward_args.attention_mask not in (
            PredefinedAttentionMask.FULL,
            PredefinedAttentionMask.CAUSAL,
        ):
            return "only full and causal masks are supported"
        return None

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
    ) -> "BlockSparseTSWrapper | BlockSparsePagedTSWrapper":
        cache = self._paged_wrappers if key.page_size is not None else self._contiguous_wrappers
        wrapper = cache.get(key)
        if wrapper is None:
            wrapper = key.plan()
            cache[key] = wrapper
        return wrapper

    def _contiguous_unsupported_reason(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> str | None:
        if forward_args.is_fused_qkv or k is None or v is None:
            return "contiguous block-sparse attention requires separate Q, K, and V"
        if self.attn.position_embedding_type != 0 or forward_args.mrope_position_deltas is not None:
            return "contiguous Q/K/V must have position embedding applied before attention"
        if forward_args.cu_q_seqlens is not None or forward_args.cu_kv_seqlens is not None:
            return "packed variable-length Q/KV inputs are not supported"
        inputs = _get_block_sparse_inputs(forward_args)
        assert inputs is not None
        mask_type = self._get_prims_mask_type(forward_args)
        if inputs.use_proxy_routes and mask_type != "dense":
            return "block-sparse proxy routes require mask_type='dense'"
        batch_size = _route_batch_size(inputs)
        seq_len_q = _uniform_seq_len_q(q, metadata, batch_size)
        if seq_len_q is None or k.shape[0] % batch_size:
            return "query and KV token counts must be batch-uniform over the route batch size"
        key = self._make_plan_key(
            q,
            inputs,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            kv_capacity=int(k.shape[0]) // batch_size,
            page_size=None,
            mask_type=mask_type,
        )
        return key.unsupported_reason()

    def _paged_unsupported_reason(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> str | None:
        inputs = _get_block_sparse_inputs(forward_args)
        assert inputs is not None
        if inputs.sparse_format != "bsr" or inputs.use_proxy_routes:
            return "paged block-sparse attention only supports BSR exact routes"
        if not forward_args.is_fused_qkv:
            return "paged block-sparse attention requires fused QKV input"
        if (
            forward_args.attention_input_type == AttentionInputType.context_only
            or metadata.num_contexts != 0
        ):
            return "paged block-sparse attention requires a generation-only batch"
        reason = get_paged_kv_storage_unsupported_reason(
            self.attn, metadata
        ) or get_paged_kv_policy_unsupported_reason(self.attn, metadata)
        if reason is not None:
            return reason
        if metadata.tokens_per_block not in self.SUPPORTED_PAGE_SIZES:
            return f"page size {metadata.tokens_per_block} is unsupported"
        if self.attn.attention_chunk_size:
            return "chunked attention is not supported"
        if get_kv_page_offset(self.attn, metadata, 0, cache=self._kv_page_offset_cache) is None:
            return "the K-to-V page displacement could not be resolved"

        batch_size = int(metadata.num_generations)
        seq_len_q = _uniform_seq_len_q(q, metadata, batch_size)
        if seq_len_q is None:
            return "query lengths must be batch-uniform and match the fixed query shape"
        block_tables = metadata.kv_cache_block_offsets
        if block_tables.shape[1] < batch_size:
            return "paged KV-cache block offsets must cover the generation batch"
        page_size = int(metadata.tokens_per_block)
        kv_capacity = int(block_tables.shape[-1]) * page_size
        logical_max_seq_len = int(metadata.max_seq_len)
        if logical_max_seq_len > kv_capacity:
            return "logical maximum sequence length must fit the page-table capacity"
        attention_window_size = forward_args.attention_window_size
        if (
            attention_window_size is None
            or attention_window_size < logical_max_seq_len
            or attention_window_size > kv_capacity
        ):
            return "attention window must fit the non-cyclic page-table capacity"
        host_seq_lens = metadata.kv_lens_runtime[:batch_size]
        min_seq_len_kv = int(host_seq_lens.min())
        if min_seq_len_kv <= 0:
            return "every active request must contain at least one KV token"
        mask_type = self._get_prims_mask_type(forward_args)
        if mask_type == "causal" and min_seq_len_kv < seq_len_q:
            return "causal KV lengths must be at least the fixed query length"
        if int(host_seq_lens.max()) > logical_max_seq_len:
            return "an active KV length exceeds the logical maximum sequence length"
        key = self._make_plan_key(
            q,
            inputs,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            kv_capacity=kv_capacity,
            page_size=page_size,
            mask_type=mask_type,
        )
        return key.unsupported_reason()

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
            # Contiguous requests run without a KV cache and never touch the
            # generation preprocessing workspace.
            if metadata.kv_cache_manager is not None:
                layout = self._get_generation_workspace_layout(
                    q.dtype,
                    int(metadata.num_generations),
                    int(q.shape[0]),
                )
                required_bytes = int(layout["total_size"])
                if workspace.numel() * workspace.element_size() < required_bytes:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "TRT-LLM QKV preprocessing workspace must be sized before "
                            "CUDA Graph capture"
                        )
                    workspace.resize_((math.ceil(required_bytes / workspace.element_size()),))
            if self._multi_processor_count is None:
                self._multi_processor_count = get_multi_processor_count_for_device(q.device.index)

    def run_generation(self, params: FmhaParams) -> None:
        q = params.qkv_input
        output_buffer = params.context_buf
        sequence_lengths = params.sequence_lengths
        assert q is not None and output_buffer is not None and sequence_lengths is not None
        metadata = params.meta
        forward_args = params.fwd
        inputs = _get_block_sparse_inputs(forward_args)
        assert inputs is not None
        batch_size = params.num_requests
        seq_len_q = params.input_seq_length
        page_size = params.tokens_per_block
        block_offsets = metadata.kv_cache_block_offsets
        assert block_offsets is not None
        preprocess = self._run_generation_preprocess(params)
        q_processed, kv_pool, block_tables = preprocess[:3]
        fmha_workspace = preprocess[6]
        if fmha_workspace.numel() != 0:
            raise RuntimeError("PrimTS block-sparse preprocessing returned an FMHA workspace.")
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
        query = q_processed.view(
            batch_size,
            seq_len_q,
            self.attn.num_heads,
            self.attn.head_dim,
        )
        key = self._make_plan_key(
            query,
            inputs,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            kv_capacity=int(block_offsets.shape[-1]) * page_size,
            page_size=page_size,
            mask_type=self._get_prims_mask_type(forward_args),
        )
        wrapper = cast("BlockSparsePagedTSWrapper", self._get_or_plan_wrapper(key))
        wrapper.run(
            query,
            (k_cache, v_cache),
            block_tables=self._get_fixed_block_tables(block_tables, batch_size),
            seq_lens_kv=self._get_sequence_lengths(sequence_lengths, batch_size),
            block_indptr=inputs.block_indptr,
            block_indices=inputs.block_indices,
            kv_valid_bits=inputs.kv_valid_bits,
            sm_scale=self._get_bmm1_scale(self.attn),
            out=output_buffer.view_as(query),
        )

    def _forward_contiguous(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_args: AttentionForwardArgs,
    ) -> None:
        inputs = _get_block_sparse_inputs(forward_args)
        assert inputs is not None
        assert forward_args.output is not None
        batch_size = _route_batch_size(inputs)
        query = q.view(batch_size, -1, self.attn.num_heads, self.attn.head_dim)
        key_states = k.view(batch_size, -1, self.attn.num_kv_heads, self.attn.head_dim)
        value_states = v.view_as(key_states)
        key = self._make_plan_key(
            query,
            inputs,
            batch_size=batch_size,
            seq_len_q=int(query.shape[1]),
            kv_capacity=int(key_states.shape[1]),
            page_size=None,
            mask_type=self._get_prims_mask_type(forward_args),
        )
        wrapper = cast("BlockSparseTSWrapper", self._get_or_plan_wrapper(key))
        wrapper.run(
            query,
            key_states,
            value_states,
            block_indptr=inputs.block_indptr,
            block_indices=inputs.block_indices,
            exact_block_bits=inputs.exact_block_bits,
            k_summary=inputs.k_summary,
            v_summary=inputs.v_summary,
            kv_valid_bits=inputs.kv_valid_bits,
            sm_scale=self._get_bmm1_scale(self.attn),
            out=forward_args.output.view_as(query),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        if metadata.kv_cache_manager is None:
            assert k is not None and v is not None
            self._forward_contiguous(q, k, v, forward_args)
            return
        super().forward(q, k, v, metadata, forward_args)
