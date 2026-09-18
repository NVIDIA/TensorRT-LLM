# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 context-phase FMHA backed by MSA's fmha_sm100 kernel.

MsaPrefillFmha wraps the fmha_sm100 paged sparse GQA kernel and participates
in the standard TrtllmAttention.forward dispatch loop. The owning MiniMax-M3
MSA attention layer runs an MsaIndexer to select the per-query KV blocks and
publishes them on forward_args.sparse_runtime_params; this class attends over
them.

It serves the context phase alone. fmha_sm100 schedules a generation row like
a context row, and a mixed batch cannot split the two apart (see
_mixed_batch_split in fmha_sm100/api.py), so the generation phase is
MsaDecodeFmha's outright.

Every import of the kernels below is function-local. This module is on the
import path of every attention.backends.trtllm import, and the minimax_m3
package init reaches msa_backend, which subclasses TrtllmAttention, so a
module-scope import here would close a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def run_msa_sparse_gqa(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    kv_block_indexes: Optional[torch.Tensor] = None,
    *,
    kv_indices: torch.Tensor,
    sm_scale: float,
    qo_lens_cpu: Optional[torch.Tensor] = None,
    kv_lens_cpu: Optional[torch.Tensor] = None,
    qo_offset_cpu: Optional[torch.Tensor] = None,
    causal: bool = True,
    head_dim: Optional[int] = None,
    plan: Optional[tuple] = None,
    out: Optional[torch.Tensor] = None,
    use_fp8: bool = False,
) -> None:
    """Run fmha_sm100 paged GQA (plan/run split).

    `kv_block_indexes`: if set, sparse top-k mode (fixed `kv_block_num=topk`);
    if None, dense mode attending all pages in `kv_indices`.
    `plan`: prebuilt execution plan; if None, built inline from the CPU length
    tensors (the prebuilt plan for a step prepare() staged, inline for a test).
    `head_dim`: defaults to the only head dimension the kernel supports.
    `out`: destination buffer the kernel writes in place.
    `use_fp8`: FP8 KV cache. The caller must pass FP8 `q` to match the FP8 paged
    K/V, since the kernel variant shares one dtype across q/k/v. Also selects the
    FP8 AOT kernels for an inline sparse-prefill plan.
    """
    from ..sparse.minimax_m3.kernels.msa_utils import MSA_REQUIRED_HEAD_DIM, require_msa_module

    if head_dim is None:
        head_dim = MSA_REQUIRED_HEAD_DIM
    fmha_sm100 = require_msa_module()

    if q.dim() != 3:
        raise ValueError(
            f"MSA paged GQA expects q [total_q, num_qo_heads, head_dim]; got {tuple(q.shape)}."
        )
    if q.shape[-1] != head_dim:
        raise NotImplementedError(f"MSA paged GQA supports head_dim={head_dim}; got {q.shape[-1]}.")
    if k_paged.dim() != 4 or v_paged.dim() != 4:
        raise ValueError(
            "MSA paged GQA expects paged KV [num_pages, num_kv_heads, page_size, head_dim]; "
            f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
        )
    if k_paged.shape != v_paged.shape:
        raise ValueError(
            f"MSA paged GQA requires k and v to share shape; "
            f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
        )

    if plan is None:
        # kv_block_num is planned only for the sparse (block-indexed) path;
        # dense paged GQA leaves it unset and attends the full page table.
        kv_block_num = int(kv_block_indexes.shape[-1]) if kv_block_indexes is not None else -1
        plan = fmha_sm100.fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            int(q.shape[1]),  # num query heads.
            num_kv_heads=int(k_paged.shape[1]),
            qo_offset=qo_offset_cpu,
            page_size=int(k_paged.shape[2]),
            kv_block_num=kv_block_num,
            causal=causal,
            num_kv_splits=1,
            use_fp8_kvcache=use_fp8,
        )
    fmha_sm100.fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        out=out,
        sm_scale=sm_scale,
        output_maxscore=False,
    )


def _aligned_nvfp4_dequant_scales(
    attn: "TrtllmAttention", kv_scale_quant_orig: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable, separately 16-byte-aligned K/V dequant scales.

    M3 checkpoints expose the Q/K/V scales as one contiguous three-float
    tensor.  Slicing elements 1 and 2 leaves addresses four and eight bytes
    past the allocation base, while CuTe DSL requires every tensor argument
    to start on a 16-byte boundary.  Keep one padded two-row buffer per
    layer-attention object.  It is populated during eager warmup and then
    reused unchanged by CUDA-graph capture and replay.
    """
    if kv_scale_quant_orig.dtype != torch.float32 or kv_scale_quant_orig.numel() < 3:
        raise ValueError("MiniMax-M3 NVFP4 dequantization scales must be FP32 [Q, K, V]")

    cache = getattr(attn, "_msa_nvfp4_dequant_scales", None)
    source_ptr = int(kv_scale_quant_orig.data_ptr())
    if cache is None or getattr(attn, "_msa_nvfp4_dequant_scale_source_ptr", None) != source_ptr:
        if kv_scale_quant_orig.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MiniMax-M3 NVFP4 scale alignment buffer must be initialized during eager warmup"
            )
        cache = torch.empty((2, 4), dtype=torch.float32, device=kv_scale_quant_orig.device)
        cache[:, 0].copy_(kv_scale_quant_orig[1:3])
        attn._msa_nvfp4_dequant_scales = cache
        attn._msa_nvfp4_dequant_scale_source_ptr = source_ptr

    k_global_scale = cache[0, :1]
    v_global_scale = cache[1, :1]
    assert k_global_scale.data_ptr() % 16 == 0
    assert v_global_scale.data_ptr() % 16 == 0
    return k_global_scale, v_global_scale


def run_msa_nvfp4_sparse_gqa(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    scale_buffers: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    metadata: "TrtllmAttentionMetadata",
    *,
    sm_scale: float,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    num_rows: Optional[int] = None,
    out: torch.Tensor,
) -> None:
    """Run sparse attention over M3's packed NVFP4 cache with the MSA CSR kernel.

    This is the prefill-shaped path: it builds a k2q CSR worklist per call and
    tiles 128 query rows at a time. num_rows narrows it to the leading batch
    rows, so a mixed step can hand its generation suffix to the Triton NVFP4
    decode and size this worklist from the context prefix alone.
    """
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        MSA_REQUIRED_TOPK,
        require_msa_module,
    )

    fmha_sm100 = require_msa_module()
    sparse = getattr(fmha_sm100, "sparse", None)
    if sparse is None:
        try:
            from fmha_sm100 import sparse
        except ImportError as exc:
            raise RuntimeError("MiniMax-M3 NVFP4 KV cache requires the Fan MSA sparse API") from exc
    if not hasattr(sparse, "build_k2q_csr") or not hasattr(sparse, "sparse_atten_nvfp4_kv_func"):
        raise RuntimeError(
            "The loaded MSA build lacks build_k2q_csr or "
            "sparse_atten_nvfp4_kv_func; use the NVFP4-capable Fan revision"
        )

    for name, scale in (("K", k_global_scale), ("V", v_global_scale)):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise ValueError(f"MiniMax-M3 NVFP4 {name} dequantization scale must be one FP32 value")
        if scale.data_ptr() % 16 != 0:
            raise ValueError(
                f"MiniMax-M3 NVFP4 {name} dequantization scale must be 16-byte aligned"
            )
    if scale_buffers.shape[:2] != k_paged.shape[:1] + (2,):
        raise ValueError(
            "MiniMax-M3 NVFP4 scale buffers must be [pages, 2, heads, page, D/16]; "
            f"got {tuple(scale_buffers.shape)} for K {tuple(k_paged.shape)}"
        )

    batch = int(getattr(metadata, "_msa_live_batch", 0))
    if batch <= 0:
        raise RuntimeError("MiniMax-M3 NVFP4 sparse attention metadata was not prepared")
    max_q_len = int(metadata._msa_max_q_len)
    max_kv_len = int(metadata._msa_max_kv_len_all)
    total_k = int(metadata._msa_total_k)
    total_k_rows = int(metadata._msa_total_k_rows)
    if num_rows is not None:
        if not 0 < num_rows <= batch:
            raise ValueError(f"MiniMax-M3 NVFP4 row limit ({num_rows}) must lie in (0, {batch}].")
        batch = num_rows
        # Staged by prepare() over the context rows, which are exactly the rows
        # a narrowed call covers; see _stage_step_fields.
        max_q_len, max_kv_len, total_k, total_k_rows = metadata._msa_context_prefix_bounds
        if total_k_rows <= 0:
            raise RuntimeError(
                "MiniMax-M3 NVFP4 attention was narrowed to a context prefix of "
                f"{num_rows} row(s), but prepare() staged no bounds for one."
            )
    cu_q = metadata.msa_cu_q_lens[: batch + 1]
    cu_kv = metadata.msa_cu_kv_lens[: batch + 1]
    q2k = kv_block_indexes.permute(1, 0, 2).contiguous()
    topk = int(q2k.shape[-1])
    page_size = int(k_paged.shape[2])
    if topk != MSA_REQUIRED_TOPK:
        raise ValueError(f"MiniMax-M3 MSA NVFP4 requires topK={MSA_REQUIRED_TOPK}, got {topk}")

    k2q_row_ptr, k2q_q_indices, schedule = sparse.build_k2q_csr(
        q2k,
        cu_q,
        cu_kv,
        page_size,
        total_k=total_k,
        max_seqlen_k=max_kv_len,
        max_seqlen_q=max_q_len,
        total_rows=total_k_rows,
        qhead_per_kv=int(q.shape[1]) // int(k_paged.shape[1]),
        return_schedule=True,
    )
    result = sparse.sparse_atten_nvfp4_kv_func(
        q,
        k_paged.view(torch.uint8),
        v_paged.view(torch.uint8),
        scale_buffers[:, 0].view(torch.uint8),
        scale_buffers[:, 1].view(torch.uint8),
        k_global_scale,
        v_global_scale,
        k2q_row_ptr,
        k2q_q_indices,
        topk,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_kv,
        max_seqlen_q=max_q_len,
        max_seqlen_k=max_kv_len,
        blk_kv=page_size,
        causal=True,
        softmax_scale=sm_scale,
        partial_dtype=torch.bfloat16,
        return_softmax_lse=False,
        page_table=metadata.msa_block_table[:batch],
        seqused_k=metadata.msa_seq_lens_cuda[:batch],
        schedule=schedule,
    )
    out.copy_(result)


def run_msa_prefill_gqa(
    attn: "TrtllmAttention",
    q: torch.Tensor,
    metadata: "TrtllmAttentionMetadata",
    output: torch.Tensor,
    *,
    kv_block_indexes: Optional[torch.Tensor],
    plan: Optional[tuple],
    row_first: int,
    num_rows: int,
    kv_scale_quant_orig: Optional[torch.Tensor] = None,
) -> None:
    """Run paged GQA over one row range into output in place.

    Shared by the sparse layers (kv_block_indexes is the per-query top-k table
    for these rows, with the sparse plan) and the dense layers
    (kv_block_indexes None, with the dense plan, attending the full page
    table).

    `q` and `output` are already the phase's token slice. `row_first` and
    `num_rows` are its batch rows, which narrow the host length tensors an
    inline plan would read.
    """
    from ..sparse.minimax_m3.kernels.msa_utils import msa_paged_kv

    head_dim = attn.head_dim
    num_tokens = int(q.shape[0])
    if num_tokens == 0:
        return
    q_view = q.view(num_tokens, attn.num_heads, head_dim)
    out_view = output.view(num_tokens, attn.num_heads, head_dim)
    k_paged, v_paged = msa_paged_kv(metadata.kv_cache_manager, attn.layer_idx)
    sm_scale = (head_dim**-0.5) / float(attn.q_scaling)

    if getattr(metadata.kv_cache_manager, "is_nvfp4_layer", lambda _: False)(attn.layer_idx):
        if kv_block_indexes is None or kv_scale_quant_orig is None:
            raise RuntimeError(
                "NVFP4 sparse prefill requires selected blocks and dequantization scales"
            )
        if row_first != 0:
            raise ValueError("NVFP4 sparse prefill must cover the context prefix")
        k_scale, v_scale = _aligned_nvfp4_dequant_scales(attn, kv_scale_quant_orig)
        run_msa_nvfp4_sparse_gqa(
            q_view,
            k_paged,
            v_paged,
            metadata.kv_cache_manager.get_block_scale_buffers(attn.layer_idx, "HND"),
            kv_block_indexes,
            metadata,
            sm_scale=sm_scale,
            k_global_scale=k_scale,
            v_global_scale=v_scale,
            num_rows=num_rows,
            out=out_view,
        )
        return

    # The fmha_sm100 variant is chosen from q.dtype and shares one dtype across
    # q/k/v, so q must be FP8 to match an FP8 paged K/V. MiniMax-M3 has no
    # KV-cache scales, so the scale is 1.0 and this is a plain E4M3 cast.
    use_fp8 = k_paged.dtype == torch.float8_e4m3fn
    if use_fp8:
        q_view = q_view.to(torch.float8_e4m3fn)

    def rows_of(lens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Narrow a per-request host length tensor to this phase's rows.

        Slicing keeps the pinned backing, so an inline plan still stages these
        with non-blocking copies.
        """
        if lens is None:
            return None
        return lens[row_first : row_first + num_rows]

    run_msa_sparse_gqa(
        q_view,
        k_paged,
        v_paged,
        kv_block_indexes,
        kv_indices=metadata.msa_kv_indices,
        sm_scale=sm_scale,
        qo_lens_cpu=rows_of(metadata.msa_qo_lens_cpu),
        kv_lens_cpu=rows_of(metadata.msa_kv_lens_cpu),
        qo_offset_cpu=rows_of(metadata.msa_qo_offset_cpu),
        causal=True,
        head_dim=head_dim,
        plan=plan,
        out=out_view,
        use_fp8=use_fp8,
    )


class MsaPrefillFmha(PhasedFmha):
    """SM100 paged GQA FMHA powered by MSA's fmha_sm100 kernel.

    Handles the context phase of every MiniMax-M3 MSA layer. Sparse layers pass
    the indexer's selected KV block indices on
    forward_args.sparse_runtime_params.sparse_attn_indices and attend those
    blocks; dense layers leave the indices None and attend the full page table.
    Requires head_dim 128 and 4-D HND paged K/V.

    The generation phase is MsaDecodeFmha's, so run_generation is left to the
    base class, which refuses it.
    """

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        from ..sparse.minimax_m3.kernels.msa_utils import is_msa_layer

        return is_msa_layer(attn)

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        # A step's context rows are this library's and its generation rows
        # MsaDecodeFmha's, whatever the step looks like, so each library claims
        # its one phase and their partition of the phases is total. The
        # phase-less query asks for the whole step, which neither can serve
        # alone: a mixed step belongs to the two together through CombinedFmha.
        return phase is FmhaPhase.CONTEXT

    def run_context(self, params: FmhaParams) -> None:
        from ..sparse.minimax_m3.kernels.msa_utils import write_msa_phase_kv

        metadata = params.meta
        write_msa_phase_kv(
            params.attn,
            params.key_input,
            params.value_input,
            metadata,
            params.fwd.attention_input_type,
            token_offset=params.token_offset,
        )
        # Sparse layers attend the per-query top-k blocks with the sparse plan;
        # dense layers leave the indices None and attend the full page table
        # with the dense plan.
        kv_block_indexes = params.fwd.sparse_runtime_params.sparse_attn_indices
        is_sparse_layer = kv_block_indexes is not None
        if is_sparse_layer:
            kv_block_indexes = kv_block_indexes[
                params.token_offset : params.token_offset + params.num_tokens
            ]
        run_msa_prefill_gqa(
            params.attn,
            params.attention_input,
            metadata,
            params.context_buf,
            kv_block_indexes=kv_block_indexes,
            plan=(
                metadata.msa_prefill_gqa_plan
                if is_sparse_layer
                else metadata.msa_prefill_dense_plan
            ),
            row_first=params.seq_offset,
            num_rows=metadata.num_contexts,
            kv_scale_quant_orig=params.fwd.kv_scale_quant_orig,
        )


__all__ = ["MsaPrefillFmha", "run_msa_prefill_gqa", "run_msa_sparse_gqa"]
