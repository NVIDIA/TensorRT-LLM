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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ..sparse.minimax_m3_kernels.msa_utils import (
    MSA_REQUIRED_HEAD_DIM,
    is_msa_layer,
    msa_paged_kv,
    require_msa_module,
    write_msa_step_kv,
)
from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention_backend.trtllm import (
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
    head_dim: int = MSA_REQUIRED_HEAD_DIM,
    plan: Optional[tuple] = None,
    out: Optional[torch.Tensor] = None,
    use_fp8: bool = False,
) -> None:
    """Run fmha_sm100 paged GQA (plan/run split).

    `kv_block_indexes`: if set, sparse top-k mode (fixed `kv_block_num=topk`);
    if None, dense mode attending all pages in `kv_indices`.
    `plan`: prebuilt execution plan; if None, built inline from the CPU length
    tensors (the prebuilt plan for a step prepare() staged, inline for a test).
    `out`: destination buffer the kernel writes in place.
    `use_fp8`: FP8 KV cache. The caller must pass FP8 `q` to match the FP8 paged
    K/V, since the kernel variant shares one dtype across q/k/v. Also selects the
    FP8 AOT kernels for an inline sparse-prefill plan.
    """
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
    head_dim = attn.head_dim
    num_tokens = int(q.shape[0])
    if num_tokens == 0:
        return
    q_view = q.view(num_tokens, attn.num_heads, head_dim)
    out_view = output.view(num_tokens, attn.num_heads, head_dim)
    k_paged, v_paged = msa_paged_kv(metadata.kv_cache_manager, attn.layer_idx)
    sm_scale = (head_dim**-0.5) / float(attn.q_scaling)

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
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        return is_msa_layer(attn)

    def is_supported(
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
        # MsaDecodeFmha's, whatever the step looks like, so the two verdicts
        # are opposite constants and their partition of the phases is total.
        return phase is not FmhaPhase.GENERATION

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        workspace: torch.Tensor,
    ) -> None:
        write_msa_step_kv(self.attn, k, v, metadata, forward_args.attention_input_type)

    def run_context(self, params: FmhaParams) -> None:
        metadata = params.meta
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
        )


__all__ = ["MsaPrefillFmha", "run_msa_prefill_gqa", "run_msa_sparse_gqa"]
