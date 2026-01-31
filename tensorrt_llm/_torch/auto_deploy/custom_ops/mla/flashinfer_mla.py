"""FlashInfer-based MLA (Multi-head Latent Attention) backend with paged caching.

This module provides:
- FlashInferMLAAttention: attention descriptor using FlashInfer MLA kernels
- flashinfer_mla_with_cache: cached backend op with paged KV cache

FlashInfer MLA uses:
- Prefill: BatchPrefillWithRaggedKVCacheWrapper with expanded K, V tensors
- Decode: BatchMLAPagedAttentionWrapper with paged compressed KV cache

FlashInfer MLA Cache Layout:
    mla_cache: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
    - No num_heads dimension (MLA-specific optimization)
    - ckv portion: mla_cache[:, :, :kv_lora_rank] (zero-copy slice)
    - kpe portion: mla_cache[:, :, kv_lora_rank:] (zero-copy slice)

Reference: https://docs.flashinfer.ai/api/mla.html
"""

import math
from dataclasses import dataclass, fields
from typing import Dict, List, Literal, Optional, Tuple

import flashinfer
import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ...utils.cuda_graph import cuda_graph_state
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    PagedResourceHandler,
    PrepareMetadataCallable,
    PrepareMetadataHostCallable,
    ResourceHandlerDict,
)


@dataclass
class MLADecodePlanParams:
    """Parameters that affect the FlashInfer MLA decode execution plan."""

    num_heads: int
    kv_lora_rank: int  # head_dim_ckv
    qk_rope_head_dim: int  # head_dim_kpe
    qk_nope_head_dim: int
    v_head_dim: int
    num_seq: int
    page_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    sm_scale: Optional[float] = None

    def __hash__(self):
        """Convert all fields to a string representation and concatenate them."""
        return hash("_".join([str(getattr(self, f.name)) for f in fields(self)]))


@dataclass
class MLAPrefillPlanParams:
    """Parameters that affect the FlashInfer MLA prefill execution plan."""

    num_heads: int
    num_kv_heads: int  # For MLA with expanded KV, same as num_heads
    head_dim_qk: int  # qk_nope_head_dim + qk_rope_head_dim
    head_dim_vo: int  # v_head_dim (value/output head dimension)
    num_seq: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    sm_scale: Optional[float] = None
    causal: bool = True

    def __hash__(self):
        """Convert all fields to a string representation and concatenate them."""
        return hash("_".join([str(getattr(self, f.name)) for f in fields(self)]))


class _FlashInferMLAPlanner:
    """A class interface to handle FlashInfer MLA-related planning/wrapping operations.

    For MLA attention:
    - Prefill uses BatchPrefillWithRaggedKVCacheWrapper with expanded K, V tensors
    - Decode uses BatchMLAPagedAttentionWrapper with paged compressed KV cache
    """

    workspace_buffer: Optional[torch.Tensor]
    prefill_wrapper: Optional[flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper]
    decode_wrapper: Optional["flashinfer.mla.BatchMLAPagedAttentionWrapper"]
    cached_cuda_graph_decode_wrappers: Dict[
        MLADecodePlanParams, "flashinfer.mla.BatchMLAPagedAttentionWrapper"
    ]
    plan_params_prefill: Optional[MLAPrefillPlanParams]
    plan_params_decode: Optional[MLADecodePlanParams]
    kv_layout: Literal["NHD", "HND"] = "NHD"

    def __init__(self):
        self.workspace_buffer = None
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.cached_cuda_graph_decode_wrappers = {}
        self.plan_params_prefill = None
        self.plan_params_decode = None

    def _init_decode_wrapper(
        self,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        last_page_len: Optional[torch.Tensor] = None,
    ):
        assert self.workspace_buffer is not None
        if use_cuda_graph:
            return flashinfer.mla.BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                self.kv_layout,
                use_cuda_graph=use_cuda_graph,
                qo_indptr_buf=qo_indptr,
                kv_indptr_buf=kv_indptr,
                kv_indices_buf=kv_indices,
                kv_last_page_len_buf=last_page_len,
            )
        else:
            return flashinfer.mla.BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=use_cuda_graph,
            )

    def reset(self, device: torch.device) -> None:
        self.plan_params_prefill = None
        self.plan_params_decode = None

        if isinstance(self.workspace_buffer, torch.Tensor):
            return

        self.__init__()  # reset all state

        # NOTE: avoid OOM for many cudagraphs
        self.workspace_buffer = torch.empty(320 * 1024 * 1024, device=device, dtype=torch.uint8)

        # Prefill uses BatchPrefillWithRaggedKVCacheWrapper with expanded K, V
        self.prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer,
            self.kv_layout,
        )
        # Decode uses BatchMLAPagedAttentionWrapper with paged compressed KV cache
        self.decode_wrapper = self._init_decode_wrapper()

    def plan_prefill(
        self,
        qo_indptr_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        plan_params: MLAPrefillPlanParams,
    ) -> flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper:
        """Plan prefill using BatchPrefillWithRaggedKVCacheWrapper.

        For MLA prefill, we expand compressed_kv to get full K, V tensors
        and use standard ragged KV cache attention.
        """
        # check for re-planning
        if plan_params != self.plan_params_prefill:
            self.prefill_wrapper.plan(
                qo_indptr_host,
                kv_indptr_host,
                plan_params.num_heads,
                plan_params.num_kv_heads,
                plan_params.head_dim_qk,
                head_dim_vo=plan_params.head_dim_vo,
                use_fp16_qk_reduction=False,
                causal=plan_params.causal,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
                sm_scale=plan_params.sm_scale,
            )
            self.plan_params_prefill = plan_params

        return self.prefill_wrapper

    def plan_decode(
        self,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        plan_params: MLADecodePlanParams,
    ) -> "flashinfer.mla.BatchMLAPagedAttentionWrapper":
        """Plan decode using BatchMLAPagedAttentionWrapper.

        For MLA decode, we use the paged compressed KV cache with
        FlashInfer's optimized MLA kernels.
        """

        # plan decode helper function
        def _plan_decode(
            wrapper: "flashinfer.mla.BatchMLAPagedAttentionWrapper",
        ):
            batch_size = kv_page_indptr.shape[0] - 1
            # for a pure decode batch, qo_indptr is just [0, 1, 2, ..., batch_size]
            qo_indptr = torch.arange(
                batch_size + 1, device=kv_page_indptr.device, dtype=torch.int32
            )
            # Compute actual KV lengths from paging metadata:
            # kv_len = (num_pages - 1) * page_size + last_page_len
            num_pages_per_seq = kv_page_indptr[1:] - kv_page_indptr[:-1]
            kv_len_arr = (num_pages_per_seq - 1) * plan_params.page_size + kv_last_page_len
            wrapper.plan(
                qo_indptr,
                kv_page_indptr,
                kv_page_indices,
                kv_len_arr,
                plan_params.num_heads,
                plan_params.kv_lora_rank,  # head_dim_ckv
                plan_params.qk_rope_head_dim,  # head_dim_kpe
                plan_params.page_size,
                causal=True,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
                sm_scale=plan_params.sm_scale,
            )

        # Plan during warm-up of cuda graph capture
        if (
            cuda_graph_state.in_warm_up()
            and plan_params not in self.cached_cuda_graph_decode_wrappers
        ):
            wrapper = self._init_decode_wrapper(
                use_cuda_graph=True,
                qo_indptr=torch.arange(
                    kv_page_indptr.shape[0], device=kv_page_indptr.device, dtype=torch.int32
                ),
                kv_indptr=kv_page_indptr,
                kv_indices=kv_page_indices,
                last_page_len=kv_last_page_len,
            )
            self.cached_cuda_graph_decode_wrappers[plan_params] = wrapper
            _plan_decode(self.cached_cuda_graph_decode_wrappers[plan_params])

        # Return pre-cached decode wrapper during cuda graph capture
        if torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up():
            wrapper = self.cached_cuda_graph_decode_wrappers[plan_params]
            return wrapper

        # check for re-planning
        if plan_params != self.plan_params_decode:
            _plan_decode(self.decode_wrapper)
            self.plan_params_decode = plan_params

        return self.decode_wrapper


_GlobalFlashInferMLAPlanner = _FlashInferMLAPlanner()


@torch.library.custom_op("auto_deploy::flashinfer_mla_prepare_metadata", mutates_args=())
def prepare_flashinfer_mla_metadata(
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
) -> List[torch.Tensor]:
    """Prepare metadata for FlashInfer MLA attention.

    This prepares batch_indices and positions for cache appends, similar to
    the standard FlashInfer attention preparation.
    """
    # retrieve host-side metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode

    _GlobalFlashInferMLAPlanner.reset(position_ids.device)

    qo_indptr = cu_seqlen[: num_seq + 1]

    # Compute batch_indices and positions for cache appends
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr, seq_len_with_cache[:num_seq], num_tokens
    )

    return batch_indices, positions


@prepare_flashinfer_mla_metadata.register_fake
def prepare_flashinfer_mla_metadata_fake(
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
):
    num_tokens = position_ids.shape[0] * position_ids.shape[1]
    return (
        torch.empty(num_tokens, dtype=torch.int32, device=position_ids.device),  # batch_indices
        torch.empty(num_tokens, dtype=torch.int32, device=position_ids.device),  # positions
    )


def prepare_flashinfer_mla_metadata_host(
    batch_info_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    last_page_len_host: torch.Tensor,
) -> None:
    """Host-side preparation for FlashInfer MLA attention.

    Note: Unlike standard FlashInfer attention which uses flashinfer.decode.fast_decode_plan
    for decode-only batches, FlashInfer MLA does not currently have an equivalent fast
    planning function. All MLA planning happens in the cached op itself.
    """
    # TODO: Add fast planning for MLA when FlashInfer supports it
    # For reference, standard attention uses:
    #   flashinfer.decode.fast_decode_plan(wrapper, cu_num_pages, cache_loc, ...)
    # MLA would need a similar function with MLA-specific parameters
    pass


@torch.library.custom_op("auto_deploy::flashinfer_mla_with_cache", mutates_args=())
def flashinfer_mla_with_cache(
    # 5 tensor args (matching torch_mla source op)
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, S, kv_lora_rank]
    kpe: torch.Tensor,  # [B, S, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    # Standard paged metadata
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    # Extra FlashInfer metadata
    flashinfer_batch_indices: torch.Tensor,
    flashinfer_positions: torch.Tensor,
    # Paged cache (combined layout)
    mla_cache: torch.Tensor,  # [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
    # Constants
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """FlashInfer MLA attention with paged cache.

    Uses FlashInfer's optimized kernels:
    - Prefill: BatchPrefillWithRaggedKVCacheWrapper with expanded K, V tensors
    - Decode: BatchMLAPagedAttentionWrapper with paged compressed KV cache

    FlashInfer MLA Cache Layout:
        mla_cache: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
        - ckv portion: mla_cache[:, :, :kv_lora_rank] (zero-copy slice)
        - kpe portion: mla_cache[:, :, kv_lora_rank:] (zero-copy slice)

    Args:
        q_nope: Query non-positional component [B, S, N, qk_nope_head_dim]
        q_pe: Query positional component [B, S, N, qk_rope_head_dim]
        compressed_kv: Compressed KV latent [B, S, kv_lora_rank]
        kpe: Key positional encoding [B, S, 1, qk_rope_head_dim]
        kv_b_proj_weight: Projection weight [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        (metadata args): Standard paged attention metadata
        mla_cache: Paged MLA cache with combined ckv and kpe
        scale: Softmax scale factor
        kv_lora_rank: Rank of compressed KV

    Returns:
        Attention output [B, S, N, v_head_dim]
    """
    # Get dimensions
    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Infer v_head_dim from kv_b_proj_weight
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    # Get batch info
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    # Set scale
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    page_size = mla_cache.shape[1]

    # Slice the combined cache into ckv and kpe portions (zero-copy views)
    ckv_cache = mla_cache[:, :, :kv_lora_rank]
    kpe_cache = mla_cache[:, :, kv_lora_rank:]

    # Flatten inputs to [total_tokens, ...] format
    bs = b * s
    q_nope_flat = q_nope.contiguous().view(bs, num_heads, qk_nope_head_dim)
    q_pe_flat = q_pe.contiguous().view(bs, num_heads, qk_rope_head_dim)
    compressed_kv_flat = compressed_kv.contiguous().view(bs, kv_lora_rank)
    kpe_flat = kpe.contiguous().view(bs, qk_rope_head_dim)

    # Convert cache dtype if needed
    if mla_cache.dtype == torch.float8_e4m3fn:
        compressed_kv_flat = compressed_kv_flat.to(torch.float8_e4m3fn)
        kpe_flat = kpe_flat.to(torch.float8_e4m3fn)

    # Append to paged cache using FlashInfer's append function
    flashinfer.page.append_paged_mla_kv_cache(
        compressed_kv_flat,
        kpe_flat,
        flashinfer_batch_indices,
        flashinfer_positions,
        ckv_cache,
        kpe_cache,
        cache_loc,
        cu_num_pages[: num_seq + 1],
        last_page_len[:num_seq],
    )

    # Pre-allocate output
    if num_prefill > 0 and num_decode > 0:
        y = torch.empty(bs, num_heads, v_head_dim, dtype=q_nope.dtype, device=q_nope.device)
    else:
        y = None

    # =========================================================================
    # PREFILL phase: Use BatchPrefillWithRaggedKVCacheWrapper with expanded K, V
    # =========================================================================
    if num_prefill > 0:
        q_nope_prefill = q_nope_flat[:num_prefill_tokens]
        q_pe_prefill = q_pe_flat[:num_prefill_tokens]
        compressed_kv_prefill = compressed_kv_flat[:num_prefill_tokens]
        kpe_prefill = kpe_flat[:num_prefill_tokens]

        # Expand compressed_kv using kv_b_proj_weight to get k_nope and v
        # compressed_kv: [num_prefill_tokens, kv_lora_rank]
        # kv_b_proj_weight: [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        # kv_expanded: [num_prefill_tokens, N * (qk_nope_head_dim + v_head_dim)]
        kv_expanded = torch.matmul(compressed_kv_prefill, kv_b_proj_weight.t())
        kv_expanded = kv_expanded.view(num_prefill_tokens, num_heads, qk_nope_head_dim + v_head_dim)

        # Split into k_nope and v
        k_nope_prefill = kv_expanded[:, :, :qk_nope_head_dim]  # [tokens, N, qk_nope_head_dim]
        v_prefill = kv_expanded[:, :, qk_nope_head_dim:].contiguous()  # [tokens, N, v_head_dim]

        # Expand kpe to all heads: [tokens, qk_rope_head_dim] -> [tokens, N, qk_rope_head_dim]
        kpe_expanded = kpe_prefill.unsqueeze(1).expand(-1, num_heads, -1).contiguous()

        # Concatenate to form full Q and K
        # Q: [tokens, N, qk_head_dim]
        q_prefill = torch.cat([q_nope_prefill, q_pe_prefill], dim=-1).contiguous()
        # K: [tokens, N, qk_head_dim]
        k_prefill = torch.cat([k_nope_prefill, kpe_expanded], dim=-1).contiguous()

        pp_prefill = MLAPrefillPlanParams(
            num_heads=num_heads,
            num_kv_heads=num_heads,  # For MLA with expanded KV, same as num_heads
            head_dim_qk=qk_head_dim,
            head_dim_vo=v_head_dim,
            num_seq=num_prefill,
            q_dtype=q_nope.dtype,
            kv_dtype=k_prefill.dtype,
            sm_scale=scale,
        )

        wrapper_prefill = _GlobalFlashInferMLAPlanner.plan_prefill(
            qo_indptr_host=cu_seqlen_host[: num_prefill + 1],
            kv_indptr_host=cu_seqlen_host[: num_prefill + 1],  # Same as qo for self-attention
            plan_params=pp_prefill,
        )

        y_prefill = wrapper_prefill.run(
            q_prefill,
            k_prefill,
            v_prefill,
        )

        if y is not None:
            y[:num_prefill_tokens] = y_prefill
        else:
            y = y_prefill

    # =========================================================================
    # DECODE phase: Use BatchMLAPagedAttentionWrapper with paged compressed KV
    # =========================================================================
    if num_decode > 0:
        q_nope_decode = q_nope_flat[num_prefill_tokens:num_total_tokens]
        q_pe_decode = q_pe_flat[num_prefill_tokens:num_total_tokens]

        # FlashInfer MLA operates in the compressed latent space.
        # We need to:
        # 1. Absorb W_kn (K-nope projection) into q_nope
        # 2. Run attention in compressed space
        # 3. Project output back using W_v

        # Extract W_kn and W_v from kv_b_proj_weight
        # kv_b_proj_weight: [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        # Reshape to [N, qk_nope_head_dim + v_head_dim, kv_lora_rank]
        kv_b_proj_reshaped = kv_b_proj_weight.view(
            num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
        )
        # W_kn: [N, qk_nope_head_dim, kv_lora_rank]
        w_kn = kv_b_proj_reshaped[:, :qk_nope_head_dim, :]
        # W_v: [N, v_head_dim, kv_lora_rank]
        w_v = kv_b_proj_reshaped[:, qk_nope_head_dim:, :]

        # Absorb W_kn into q_nope:
        # q_nope_decode: [num_decode, N, qk_nope_head_dim]
        # w_kn: [N, qk_nope_head_dim, kv_lora_rank]
        # q_nope_absorbed: [num_decode, N, kv_lora_rank]
        q_nope_absorbed = torch.einsum("bnd,ndk->bnk", q_nope_decode, w_kn)

        pp_decode = MLADecodePlanParams(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            num_seq=num_decode,
            page_size=page_size,
            q_dtype=q_nope.dtype,
            kv_dtype=ckv_cache.dtype,
            sm_scale=scale,
        )

        wrapper_decode = _GlobalFlashInferMLAPlanner.plan_decode(
            kv_page_indptr=cu_num_pages[num_prefill : num_seq + 1],
            kv_page_indices=cache_loc,
            kv_last_page_len=last_page_len[num_prefill:num_seq],
            plan_params=pp_decode,
        )

        # Run attention in compressed space
        # y_decode_compressed: [num_decode, N, kv_lora_rank]
        y_decode_compressed = wrapper_decode.run(
            q_nope_absorbed,
            q_pe_decode,
            ckv_cache,
            kpe_cache,
        )

        # Project output back from latent space to v_head_dim
        # y_decode_compressed: [num_decode, N, kv_lora_rank]
        # w_v: [N, v_head_dim, kv_lora_rank]
        # y_decode: [num_decode, N, v_head_dim]
        y_decode = torch.einsum("bnk,nvk->bnv", y_decode_compressed, w_v)

        if y is not None:
            y[num_prefill_tokens:num_total_tokens] = y_decode
        else:
            y = y_decode

    return y.view(b, s, num_heads, v_head_dim)


@flashinfer_mla_with_cache.register_fake
def flashinfer_mla_with_cache_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    flashinfer_batch_indices: torch.Tensor,
    flashinfer_positions: torch.Tensor,
    mla_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
) -> torch.Tensor:
    """Fake implementation for flashinfer_mla_with_cache."""
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[-1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    return q_nope.new_empty(
        q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
    ).contiguous()


@AttentionRegistry.register("flashinfer_mla")
class FlashInferMLAAttention(AttentionDescriptor):
    """Attention descriptor for FlashInfer-based MLA with paged cache.

    This descriptor uses FlashInfer's optimized MLA kernels:
    - Source op: torch_mla (same as torch_mla backend)
    - Cached op: flashinfer_mla_with_cache with paged cache

    FlashInfer MLA Cache Layout:
        mla_cache: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
        - No num_heads dimension (MLA-specific optimization)
        - ckv portion: mla_cache[:, :, :kv_lora_rank] (zero-copy slice)
        - kpe portion: mla_cache[:, :, kv_lora_rank:] (zero-copy slice)

    Reference: https://docs.flashinfer.ai/api/mla.html
    """

    @classmethod
    def _get_planner(cls) -> _FlashInferMLAPlanner:
        return _GlobalFlashInferMLAPlanner

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of tensor arguments expected by the source op."""
        return 5  # q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""
        return torch.ops.auto_deploy.torch_mla

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Get the cached attention op."""
        return torch.ops.auto_deploy.flashinfer_mla_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Get the list of standard metadata arguments for paged attention."""
        return [
            "batch_info_host",
            "cu_seqlen_host",
            "cu_num_pages",
            "cu_num_pages_host",
            "cache_loc",
            "last_page_len",
            "last_page_len_host",
            "seq_len_with_cache_host",
        ]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        """Get the prepare_metadata op for FlashInfer MLA."""
        return (torch.ops.auto_deploy.flashinfer_mla_prepare_metadata.default, 2, [])

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Get cache initializers using FlashInfer MLA paged cache layout.

        Creates a single paged cache with combined layout:
        - mla_cache: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
        - ckv portion: mla_cache[:, :, :kv_lora_rank] (zero-copy slice)
        - kpe portion: mla_cache[:, :, kv_lora_rank:] (zero-copy slice)
        """
        # Extract dimensions from source node args
        # torch_mla signature: q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight, ...
        compressed_kv_fake: FakeTensor = source_attn_node.args[2].meta["val"]
        kpe_fake: FakeTensor = source_attn_node.args[3].meta["val"]

        # Get dimensions
        # compressed_kv: [B, S, kv_lora_rank]
        # kpe: [B, S, 1, qk_rope_head_dim]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]

        cache_dtype = cls.resolve_cache_dtype(cache_config.dtype, compressed_kv_fake.dtype)

        # FlashInfer MLA uses a single paged cache with no num_heads dimension
        # Layout: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
        return {
            "mla_cache": PagedResourceHandler(
                kv_lora_rank + qk_rope_head_dim,
                dtype=cache_dtype,
            ),
        }

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        """Get function for host-side preparation."""
        return prepare_flashinfer_mla_metadata_host

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Get constants to pass to the cached attention op."""
        # Extract kv_lora_rank for cache operations
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kv_lora_rank = compressed_kv_fake.shape[-1]

        # Get scale from kwargs
        scale = source_attn_node.kwargs.get("scale", None)

        return [scale, kv_lora_rank]
