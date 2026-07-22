# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-sparse GQA FMHA backed by MSA's fmha_sm100 kernel.

MsaSparseGqaFmha wraps the fmha_sm100 paged sparse GQA kernel and
participates in the standard TrtllmAttention.forward dispatch loop. The
owning MiniMax-M3 MSA attention layer runs an MsaIndexer to select the
per-query KV blocks and publishes them on forward_args.sparse_prediction;
this class attends over them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import is_sm_100f

from .interface import Fmha

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
    head_dim: int = 128,
    plan: Optional[tuple] = None,
    out: Optional[torch.Tensor] = None,
    use_fp8: bool = False,
) -> None:
    """Run fmha_sm100 paged GQA (plan/run split).

    `kv_block_indexes`: if set, sparse top-k mode (fixed `kv_block_num=topk`);
    if None, dense mode attending all pages in `kv_indices`.
    `plan`: prebuilt execution plan; if None, built inline from the CPU length
    tensors (eager prefill/tests vs. CUDA-graph decode).
    `out`: destination buffer the kernel writes in place.
    `use_fp8`: FP8 KV cache. The caller must pass FP8 `q` to match the FP8 paged
    K/V, since the kernel variant shares one dtype across q/k/v. Also selects the
    FP8 AOT kernels for an inline sparse-prefill plan; no-op for the decode planner.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import require_msa_module

    fmha_sm100 = require_msa_module()

    if q.dim() != 3:
        raise ValueError(
            f"MsaSparseGqaFmha expects q [total_q, num_qo_heads, head_dim]; got {tuple(q.shape)}."
        )
    if q.shape[-1] != head_dim:
        raise NotImplementedError(
            f"MsaSparseGqaFmha supports head_dim={head_dim}; got {q.shape[-1]}."
        )
    if k_paged.dim() != 4 or v_paged.dim() != 4:
        raise ValueError(
            "MsaSparseGqaFmha expects paged KV [num_pages, num_kv_heads, page_size, head_dim]; "
            f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
        )
    if k_paged.shape != v_paged.shape:
        raise ValueError(
            f"MsaSparseGqaFmha requires k and v to share shape; "
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


def run_msa_paged_gqa(
    attn: "TrtllmAttention",
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    metadata: "TrtllmAttentionMetadata",
    output: torch.Tensor,
    *,
    kv_block_indexes: Optional[torch.Tensor],
    plan: Optional[tuple],
) -> None:
    """Write the new-token main K/V, then run paged GQA into output in place.

    Shared by the sparse layers (kv_block_indexes is the per-query top-k table,
    with the sparse plan) and the dense layers (kv_block_indexes None, with the
    dense plan, attending the full page table). fmha_sm100 reads the paged cache
    directly, so the new-token K/V must be resident before the run.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
        msa_paged_kv,
        write_msa_main_kv,
    )

    layer_idx = attn.layer_idx
    head_dim = attn.head_dim
    kv_cache_manager = metadata.kv_cache_manager
    num_tokens = int(q.shape[0])
    if k is not None and v is not None:
        write_msa_main_kv(
            kv_cache_manager, layer_idx, metadata.msa_out_cache_loc[:num_tokens], k, v
        )

    # q may be a strided column-view of a fused [q|k|v] buffer (the model skips
    # the split contiguous copy on this path). fmha_sm100 reads q's real strides
    # through the TMA descriptor, so reshape here is a zero-copy view for that
    # layout; it only falls back to a copy for an otherwise non-viewable q.
    q_view = q.reshape(num_tokens, attn.num_heads, head_dim)
    # output is freshly allocated and contiguous; view keeps out_view aliasing it
    # so the kernel's in-place write lands in the caller's buffer.
    out_view = output.view(num_tokens, attn.num_heads, head_dim)
    k_paged, v_paged = msa_paged_kv(kv_cache_manager, layer_idx)
    sm_scale = (head_dim**-0.5) / float(attn.q_scaling)

    # The fmha_sm100 variant is chosen from q.dtype and shares one dtype across
    # q/k/v, so q must be FP8 to match an FP8 paged K/V. MiniMax-M3 has no
    # KV-cache scales, so the scale is 1.0 and this is a plain E4M3 cast. When the
    # model's fused QK-norm+RoPE already emitted FP8 q/k/v (the FP8-KV fast path),
    # this .to() is a no-op; it stays as a safety net for callers that pass bf16 q.
    use_fp8 = k_paged.dtype == torch.float8_e4m3fn
    if use_fp8 and q_view.dtype != torch.float8_e4m3fn:
        q_view = q_view.to(torch.float8_e4m3fn)

    run_msa_sparse_gqa(
        q_view,
        k_paged,
        v_paged,
        kv_block_indexes,
        kv_indices=metadata.msa_kv_indices,
        sm_scale=sm_scale,
        qo_lens_cpu=metadata.msa_qo_lens_cpu,
        kv_lens_cpu=metadata.msa_kv_lens_cpu,
        qo_offset_cpu=metadata.msa_qo_offset_cpu,
        causal=True,
        head_dim=head_dim,
        plan=plan,
        out=out_view,
        use_fp8=use_fp8,
    )


class MsaSparseGqaFmha(Fmha):
    """SM100 paged GQA FMHA powered by MSA's fmha_sm100 kernel.

    Handles every MiniMax-M3 MSA layer. Sparse layers pass the indexer's
    selected KV block indices on forward_args.sparse_prediction.sparse_attn_indices
    and attend those blocks; dense layers leave the indices None and attend the
    full page table.

        Inherits Fmha rather than PhasedFmha: fmha_sm100 takes a single plan and
        the selected block indices span the whole batch, so it handles a mixed
        context and generation batch in one call and there is no
        context/generation split from PhasedFmha to reuse. Requires head_dim 128
        and 4-D HND paged K/V.
    """

    @classmethod
    def is_available(cls, attn: Optional["TrtllmAttention"] = None) -> bool:
        # fmha_sm100 runs only on the SM100 family and ships in the MSA git
        # submodule, so it is unavailable off SM100 or without the package.
        # Imported lazily because the minimax_m3 package init imports the trtllm
        # attention classes, which a module-scope import here would cycle with.
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
            msa_package_available,
        )

        if not is_sm_100f() or not msa_package_available():
            return False
        # Only the MiniMax-M3 MSA layer uses this library. Matching the lowered
        # sparse algorithm lets the base create_fmha_libs add it to that layer
        # alone, so no create_fmha_libs override is needed. Dense layers (e.g.
        # an Eagle3 draft model) have no sparse_params.
        return attn.sparse_params is not None and attn.sparse_params.algorithm == "minimax_m3"

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
    ) -> None:
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires an output buffer.")

        # Sparse layers attend the per-query top-k blocks with the sparse plan;
        # dense layers leave the indices None and attend the full page table
        # with the dense plan.
        kv_block_indexes = forward_args.sparse_prediction.sparse_attn_indices
        if kv_block_indexes is not None:
            plan = metadata.msa_decode_gqa_plan
            if plan is None:
                plan = getattr(metadata, "msa_eager_gqa_plan", None)
        else:
            plan = metadata.msa_decode_dense_plan
            if plan is None:
                plan = getattr(metadata, "msa_eager_dense_plan", None)
        run_msa_paged_gqa(
            self.attn,
            q,
            k,
            v,
            metadata,
            output,
            kv_block_indexes=kv_block_indexes,
            plan=plan,
        )


__all__ = ["MsaSparseGqaFmha"]
