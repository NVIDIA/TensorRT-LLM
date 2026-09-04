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

import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.logger import logger

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


# Convert only the selected NVFP4 pages, then hand the compact FP8 scratch to
# the established preplanned MSA consumer. Keep this opt-in until its serving
# A/B and CUDA-graph coverage are accepted.
_MSA_NVFP4_STANDARD_STAGE_ENABLED = os.environ.get("TRTLLM_M3_NVFP4_STANDARD_STAGE", "0") == "1"
_MSA_NVFP4_STANDARD_STAGE_MAX_DQL = 8


def _nvfp4_standard_stage_capacity(metadata: "TrtllmAttentionMetadata") -> int:
    """Bound selected-page scratch by the largest accepted decode step.

    ``msa_q_batch_row`` is sized from ``max_num_tokens`` because it is shared
    with context execution.  The staged-standard route is pure decode only,
    however, and accepts at most eight query tokens for each request row.
    Allocating from the context-sized buffer can otherwise reserve tens of
    GiB on a server configured for large chunked prefill.
    """
    token_capacity = int(metadata.msa_q_batch_row.shape[0])
    request_capacity = int(metadata.msa_block_table.shape[0])
    return min(token_capacity, request_capacity * _MSA_NVFP4_STANDARD_STAGE_MAX_DQL)


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
    kv_scale_orig_quant: Optional[torch.Tensor] = None,
    kv_scale_quant_orig: Optional[torch.Tensor] = None,
) -> None:
    """Write the new-token main K/V, then run paged GQA into output in place.

    Shared by the sparse layers (kv_block_indexes is the per-query top-k table,
    with the sparse plan) and the dense layers (kv_block_indexes None, with the
    dense plan, attending the full page table). fmha_sm100 reads the paged cache
    directly, so the new-token K/V must be resident before the run.

    This is also where a step splits by request phase. TensorRT-LLM orders a
    batch context-first, so the generation requests are its token suffix: a
    ported decode kernel takes that suffix and fmha_sm100 keeps the context
    prefix, running under the plan prepare() built over those rows alone (see
    _msa_fmha_plan_rows). The prefix is empty on a pure-decode step, so one code
    path covers both. The split lives here rather than in a PhasedFmha subclass
    because MiniMaxM3MsaSparseAttention.forward_prepopulated_kv also calls this
    helper directly, bypassing TrtllmAttention.forward.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
        msa_decode_span_bounds,
        msa_paged_kv,
        msa_ported_decode_active,
        write_msa_main_kv,
    )

    layer_idx = attn.layer_idx
    head_dim = attn.head_dim
    kv_cache_manager = metadata.kv_cache_manager
    num_tokens = int(q.shape[0])
    # The fused per-layer scatter (msa_write_layer_caches) may have written
    # this layer's K/V already; consume the marker so it never goes stale.
    prewritten = getattr(metadata, "_msa_prewritten_layer", None) == layer_idx
    if prewritten:
        metadata._msa_prewritten_layer = None
    if k is not None and v is not None and not prewritten:
        # num_tokens is the padded extent under piecewise CUDA graphs, so the live
        # count is what separates real slots from the sentinel tail. Only the MSA
        # metadata pads its slot array, so a backend without the accessor hands
        # over live rows only.
        live_token_count = getattr(metadata, "msa_live_token_count", None)
        num_live_tokens = live_token_count() if live_token_count is not None else num_tokens
        write_msa_main_kv(
            kv_cache_manager,
            layer_idx,
            metadata.msa_out_cache_loc[:num_tokens],
            k,
            v,
            num_live_tokens=num_live_tokens,
        )

    # q may be a strided column-view of a fused [q|k|v] buffer (the model skips
    # the split contiguous copy on this path). fmha_sm100 reads q's real strides
    # through the TMA descriptor, so reshape here is a zero-copy view for that
    # layout; it only falls back to a copy for an otherwise non-viewable q.
    q_view = q.reshape(num_tokens, attn.num_heads, head_dim)
    # output is freshly allocated and contiguous; view keeps out_view aliasing it
    # so the kernel's in-place write lands in the caller's buffer.
    out_view = output.view(num_tokens, attn.num_heads, head_dim)
    sm_scale = (head_dim**-0.5) / float(attn.q_scaling)

    # Query tokens [gen_tok0, gen_tok1) and batch rows [gen_row0, gen_row1) the
    # ported kernels own: the whole batch on a pure-decode step, the trailing
    # generation slice on a mixed one, and nothing when none is ported.
    gen_tok0, gen_tok1, gen_row0, gen_row1, decode_query_len = msa_decode_span_bounds(
        metadata, num_tokens
    )
    nvfp4_predicate = getattr(kv_cache_manager, "is_nvfp4_layer", None)
    layer_uses_nvfp4 = (
        bool(nvfp4_predicate(layer_idx))
        if nvfp4_predicate is not None
        else bool(getattr(metadata, "_msa_main_kv_is_nvfp4", lambda: False)())
    )
    if layer_uses_nvfp4:
        if kv_scale_orig_quant is None or kv_scale_quant_orig is None:
            raise RuntimeError(
                "MiniMax-M3 NVFP4 attention requires K/V quantization and dequantization scales"
            )
        if kv_block_indexes is None:
            raise RuntimeError(
                f"MiniMax-M3 layer {layer_idx} has NVFP4 sparse storage but no sparse indexes"
            )
        k_paged, v_paged = msa_paged_kv(kv_cache_manager, layer_idx)
        (
            k_global_scale,
            v_global_scale,
            k_global_scale_value,
            v_global_scale_value,
        ) = _aligned_nvfp4_dequant_scales(attn, kv_scale_quant_orig)
        run_msa_nvfp4_sparse_gqa(
            q_view,
            k_paged,
            v_paged,
            kv_cache_manager.get_block_scale_buffers(layer_idx, "HND"),
            kv_block_indexes,
            metadata,
            sm_scale=sm_scale,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
            k_global_scale_value=k_global_scale_value,
            v_global_scale_value=v_global_scale_value,
            plan=plan,
            out=out_view,
        )
        return

    k_paged, v_paged = msa_paged_kv(kv_cache_manager, layer_idx)

    # Leading query tokens fmha_sm100 must still run: the whole batch until a
    # ported kernel takes the generation slice, then the context prefix alone.
    fmha_tokens = num_tokens
    ported = msa_ported_decode_active(metadata)
    # The span ends where q does, being the whole token axis minus the context
    # prefix. A longer q carries a token pad that the dispatch clips off (see
    # _dispatch_attention_over_live_tokens in modeling_minimaxm3), so one here
    # means a token index no longer names a request.
    if ported and gen_tok1 != num_tokens:
        raise RuntimeError(
            f"MiniMax-M3 paged GQA got {num_tokens} query tokens for a span "
            f"ending at token {gen_tok1}. Every token past the span belongs to "
            "no request; see msa_decode_span_bounds."
        )

    if kv_block_indexes is not None and ported:
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.triton_sparse_decode import (
            minimax_m3_sparse_attn_decode,
        )

        # The Triton kernel widens an E4M3 q from the fused producer in-register
        # and runs the QK/PV math at out_view.dtype, so no widened copy is
        # materialized here. E4M3 widens exactly, leaving the same values the
        # fmha_sm100 path would have used.
        minimax_m3_sparse_attn_decode(
            q_view[gen_tok0:],
            k_paged,
            v_paged,
            # [total_q, num_kv_heads, topk] -> head-major, contiguous when the
            # indexer emitted a head-major table and the slice is the whole
            # batch (see msa_ported_decode_active, which both sites read). The
            # kernel takes every stride, so a mixed step's strided suffix is fine.
            kv_block_indexes[gen_tok0:].permute(1, 0, 2),
            metadata.msa_block_table[gen_row0:gen_row1],
            metadata.msa_seq_lens_cuda[gen_row0:gen_row1],
            sm_scale=sm_scale,
            output=out_view[gen_tok0:],
            decode_query_len=decode_query_len,
        )
        fmha_tokens = gen_tok0

    elif kv_block_indexes is None and ported:
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.trtllm_gen_dense_decode import (
            dense_decode_unsupported_reason,
            minimax_m3_trtllm_gen_dense_decode,
        )

        unsupported = dense_decode_unsupported_reason(kv_cache_manager, head_dim)
        if unsupported is not None:
            raise RuntimeError(
                "MiniMax-M3 resolved a generation span for this step's dense "
                f"layers and skipped the fmha_sm100 dense plan, but {unsupported} "
                "The two must agree, and there is no plan left to run the span; "
                "see _resolve_decode_kernels."
            )
        # The sub-page block table prepare() staged, if it could; the kernel
        # expands its own when the factor does not match this layer's.
        staged_subpage_rows = getattr(metadata, "msa_subpage_rows", None)
        staged_table, staged_factor = (
            staged_subpage_rows(gen_row0, gen_row1)
            if staged_subpage_rows is not None
            else (None, 0)
        )
        minimax_m3_trtllm_gen_dense_decode(
            q_view[gen_tok0:],
            kv_cache_manager,
            layer_idx,
            metadata.msa_block_table[gen_row0:gen_row1],
            metadata.msa_seq_lens_cuda[gen_row0:gen_row1],
            sm_scale=sm_scale,
            output=out_view[gen_tok0:],
            decode_query_len=decode_query_len,
            # Bounded by the span's own rows, so a long context request
            # cannot inflate the kernel's scheduling hint.
            max_seq_len=int(metadata.msa_max_kv_len),
            max_num_requests=int(metadata.max_num_requests),
            staged_subpage_table=staged_table,
            staged_subpages_per_slot=staged_factor,
        )
        fmha_tokens = gen_tok0

    if fmha_tokens == 0:
        return

    if fmha_tokens == num_tokens:
        # Reaching fmha_sm100 for the whole batch when a span was resolved means
        # neither branch above took it, after prepare had already skipped this
        # layer's plan and, on a pure-decode step, the flattened page table the
        # call below reads. Fail loudly: running on with a stale msa_kv_indices
        # would silently attend the wrong pages.
        if ported:
            raise RuntimeError(
                "MiniMax-M3 paged GQA reached fmha_sm100 with no plan for a "
                f"{'sparse' if kv_block_indexes is not None else 'dense'} layer. "
                "The step resolved a generation span, which the ported decode "
                "kernels own; see _resolve_decode_kernels."
            )
        fmha_rows = None
    else:
        # The context prefix, matching the rows `plan` was built over. The
        # flattened page table needs no slice: context pages are its prefix and
        # the plan implies how many of them to read.
        fmha_rows = gen_row0

    # The fmha_sm100 variant is chosen from q.dtype and shares one dtype across
    # q/k/v, so q must be FP8 to match an FP8 paged K/V. MiniMax-M3 has no
    # KV-cache scales, so the scale is 1.0 and this is a plain E4M3 cast. When the
    # model's fused QK-norm+RoPE already emitted FP8 q/k/v (the FP8-KV fast path),
    # this .to() is a no-op; it stays as a safety net for callers that pass bf16 q.
    fmha_q = q_view[:fmha_tokens]
    use_fp8 = k_paged.dtype == torch.float8_e4m3fn
    if use_fp8 and fmha_q.dtype != torch.float8_e4m3fn:
        fmha_q = fmha_q.to(torch.float8_e4m3fn)

    def rows_of(lens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Narrow a per-request host length tensor to the rows fmha_sm100 runs.

        Slicing a pinned tensor keeps the pinned backing, so the inline planner
        still stages these with non-blocking copies.
        """
        if lens is None or fmha_rows is None:
            return lens
        return lens[:fmha_rows]

    run_msa_sparse_gqa(
        fmha_q,
        k_paged,
        v_paged,
        None if kv_block_indexes is None else kv_block_indexes[:fmha_tokens],
        kv_indices=metadata.msa_kv_indices,
        sm_scale=sm_scale,
        qo_lens_cpu=rows_of(metadata.msa_qo_lens_cpu),
        kv_lens_cpu=rows_of(metadata.msa_kv_lens_cpu),
        qo_offset_cpu=rows_of(metadata.msa_qo_offset_cpu),
        causal=True,
        head_dim=head_dim,
        plan=plan,
        out=out_view[:fmha_tokens],
        use_fp8=use_fp8,
    )


def _aligned_nvfp4_dequant_scales(
    attn: "TrtllmAttention", kv_scale_quant_orig: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, Optional[float], Optional[float]]:
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
    scale_values = getattr(attn, "_msa_nvfp4_dequant_scale_values", None)
    source_ptr = int(kv_scale_quant_orig.data_ptr())
    if (
        cache is None
        or (_MSA_NVFP4_STANDARD_STAGE_ENABLED and scale_values is None)
        or getattr(attn, "_msa_nvfp4_dequant_scale_source_ptr", None) != source_ptr
    ):
        if kv_scale_quant_orig.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MiniMax-M3 NVFP4 scale alignment buffer must be initialized during eager warmup"
            )
        cache = torch.empty((2, 4), dtype=torch.float32, device=kv_scale_quant_orig.device)
        cache[:, 0].copy_(kv_scale_quant_orig[1:3])
        attn._msa_nvfp4_dequant_scales = cache
        attn._msa_nvfp4_dequant_scale_source_ptr = source_ptr
        if _MSA_NVFP4_STANDARD_STAGE_ENABLED:
            # The standard FMHA launch takes scales as host values. Resolve
            # them during eager warmup so graph capture never executes item().
            attn._msa_nvfp4_dequant_scale_values = (
                float(kv_scale_quant_orig[1].item()),
                float(kv_scale_quant_orig[2].item()),
            )

    k_global_scale = cache[0, :1]
    v_global_scale = cache[1, :1]
    assert k_global_scale.data_ptr() % 16 == 0
    assert v_global_scale.data_ptr() % 16 == 0
    k_value, v_value = getattr(attn, "_msa_nvfp4_dequant_scale_values", (None, None))
    return k_global_scale, v_global_scale, k_value, v_value


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
    k_global_scale_value: Optional[float] = None,
    v_global_scale_value: Optional[float] = None,
    plan: Optional[tuple] = None,
    out: torch.Tensor,
) -> None:
    """Run sparse attention over M3's packed NVFP4 cache.

    The default path is Fan's direct NVFP4 CSR kernel for every phase. An
    opt-in pure-decode path stages only the selected pages to compact FP8 and
    invokes the established preplanned MSA consumer. Neither path may fall
    through to Triton, which would reinterpret packed E2M1 bytes as scalar K/V.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
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
    cu_q = metadata.msa_cu_q_lens[: batch + 1]
    cu_kv = metadata.msa_cu_kv_lens[: batch + 1]
    q2k = kv_block_indexes.permute(1, 0, 2).contiguous()
    topk = int(q2k.shape[-1])
    page_size = int(k_paged.shape[2])
    if topk != MSA_REQUIRED_TOPK:
        raise ValueError(f"MiniMax-M3 MSA NVFP4 requires topK={MSA_REQUIRED_TOPK}, got {topk}")

    pure_decode = (
        int(getattr(metadata, "num_contexts", 0)) == 0
        and int(getattr(metadata, "num_generations", 0)) > 0
    )
    standard_stage = getattr(sparse, "stage_selected_nvfp4_to_fp8", None)
    stage_ready = (
        _MSA_NVFP4_STANDARD_STAGE_ENABLED
        and pure_decode
        and 1 <= int(metadata._msa_max_q_len) <= _MSA_NVFP4_STANDARD_STAGE_MAX_DQL
        and int(k_paged.shape[1]) > 0
        and int(q.shape[1]) == 16 * int(k_paged.shape[1])
        and page_size == 128
        and standard_stage is not None
        and plan is not None
    )
    if _MSA_NVFP4_STANDARD_STAGE_ENABLED and pure_decode:
        logger.info_once(
            "MiniMax-M3 NVFP4 staged-standard pure-decode route "
            f"{'accepted' if stage_ready else 'rejected'}: "
            f"q_dtype={q.dtype}, Hq={int(q.shape[1])}, "
            f"Hkv={int(k_paged.shape[1])}, P={page_size}, "
            f"DQL={int(metadata._msa_max_q_len)}, "
            f"stage_api={standard_stage is not None}, plan={plan is not None}",
            key=(
                "minimax_m3_nvfp4_staged_standard_route_accepted"
                if stage_ready
                else "minimax_m3_nvfp4_staged_standard_route_rejected"
            ),
        )

    if stage_ready:
        if k_global_scale_value is None or v_global_scale_value is None:
            raise RuntimeError(
                "MiniMax-M3 standard staged NVFP4 decode requires host global-scale values"
            )
        total_q = int(q.shape[0])
        kv_heads = int(k_paged.shape[1])
        head_dim = int(q.shape[2])
        capacity = _nvfp4_standard_stage_capacity(metadata)
        if total_q > capacity:
            raise RuntimeError(
                "MiniMax-M3 staged NVFP4 query count exceeds its graph-stable "
                f"capacity: {total_q} > {capacity}"
            )

        scratch_owner = metadata.kv_cache_manager
        scratch_key = (
            q.device,
            torch.float8_e4m3fn,
            kv_heads,
            page_size,
            head_dim,
            topk,
            capacity,
        )
        scratch_cache = getattr(scratch_owner, "_msa_nvfp4_selected_scratch_cache", None)
        if scratch_cache is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MiniMax-M3 staged NVFP4 scratch must be initialized during eager warmup"
                )
            scratch_cache = {}
            scratch_owner._msa_nvfp4_selected_scratch_cache = scratch_cache
        scratch_entry = scratch_cache.get(scratch_key)
        if scratch_entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MiniMax-M3 staged NVFP4 scratch must be initialized during eager warmup"
                )
            scratch_shape = (capacity * topk, kv_heads, page_size, head_dim)
            scratch_entry = (
                torch.empty(scratch_shape, dtype=torch.float8_e4m3fn, device=q.device),
                torch.empty(scratch_shape, dtype=torch.float8_e4m3fn, device=q.device),
                torch.arange(capacity * topk, dtype=torch.int32, device=q.device)
                .view(capacity, 1, topk)
                .expand(capacity, kv_heads, topk)
                .contiguous(),
            )
            scratch_cache[scratch_key] = scratch_entry

        scratch_k_storage, scratch_v_storage, physical_storage = scratch_entry
        compact_pages = total_q * topk
        scratch_k = scratch_k_storage[:compact_pages]
        scratch_v = scratch_v_storage[:compact_pages]
        physical_pages = physical_storage[:total_q]
        direct_q2k = kv_block_indexes.contiguous()
        q_batch_row = metadata.msa_q_batch_row[:total_q]
        q_intra = metadata.msa_q_intra[:total_q]
        standard_stage(
            k_paged.view(torch.uint8),
            scale_buffers[:, 0].view(torch.uint8),
            direct_q2k,
            q_batch_row,
            metadata.msa_block_table[:batch],
            scratch_k,
            is_v=False,
        )
        standard_stage(
            v_paged.view(torch.uint8),
            scale_buffers[:, 1].view(torch.uint8),
            direct_q2k,
            q_batch_row,
            metadata.msa_block_table[:batch],
            scratch_v,
            is_v=True,
        )

        custom_mask = (
            metadata.spec_decoding_packed_mask
            if (
                bool(getattr(metadata, "is_spec_dec_dynamic_tree", False))
                and int(getattr(metadata, "num_generations", 0)) > 0
            )
            else None
        )
        fmha_q = q if q.dtype == torch.float8_e4m3fn else q.to(torch.float8_e4m3fn)
        returned, _ = fmha_sm100.fmha_sm100(
            fmha_q,
            scratch_k,
            scratch_v,
            plan,
            kv_indices=metadata.msa_block_table.flatten(),
            kv_block_indexes=direct_q2k,
            kv_physical_block_indexes=physical_pages,
            sparse_custom_mask=custom_mask,
            sparse_custom_mask_q_indices=q_intra if custom_mask is not None else None,
            sparse_custom_mask_batch_indices=(q_batch_row if custom_mask is not None else None),
            out=out,
            sm_scale=sm_scale,
            k_scale=float(k_global_scale_value),
            v_scale=float(v_global_scale_value),
            output_maxscore=False,
        )
        if returned.data_ptr() != out.data_ptr():
            raise RuntimeError(
                "MiniMax-M3 standard staged NVFP4 decode did not use its output buffer"
            )
        return

    k2q_row_ptr, k2q_q_indices, schedule = sparse.build_k2q_csr(
        q2k,
        cu_q,
        cu_kv,
        int(k_paged.shape[2]),
        total_k=int(metadata._msa_total_k),
        max_seqlen_k=int(metadata._msa_max_kv_len_all),
        max_seqlen_q=int(metadata._msa_max_q_len),
        total_rows=int(metadata._msa_total_k_rows),
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
        max_seqlen_q=int(metadata._msa_max_q_len),
        max_seqlen_k=int(metadata._msa_max_kv_len_all),
        blk_kv=int(k_paged.shape[2]),
        causal=True,
        softmax_scale=sm_scale,
        partial_dtype=torch.bfloat16,
        return_softmax_lse=False,
        page_table=metadata.msa_block_table[:batch],
        seqused_k=metadata.msa_seq_lens_cuda[:batch],
        schedule=schedule,
    )
    out.copy_(result)


class MsaSparseGqaFmha(Fmha):
    """SM100 paged GQA FMHA powered by MSA's fmha_sm100 kernel.

    Handles every MiniMax-M3 MSA layer. Sparse layers pass the indexer's
    selected KV block indices on forward_args.sparse_prediction.sparse_attn_indices
    and attend those blocks; dense layers leave the indices None and attend the
    full page table.

        Inherits Fmha rather than PhasedFmha even though a mixed batch is split
        by phase, because that split has to happen in run_msa_paged_gqa rather
        than in forward: MiniMaxM3MsaSparseAttention.forward_prepopulated_kv
        calls that helper directly, so a split placed in PhasedFmha.forward
        would miss it. PhasedFmha also cannot reach the third ported kernel, the
        indexer scorer, which runs before forward from run_indexer. Requires
        head_dim 128 and 4-D HND paged K/V.
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
            kv_scale_orig_quant=forward_args.kv_scale_orig_quant,
            kv_scale_quant_orig=forward_args.kv_scale_quant_orig,
        )


__all__ = ["MsaSparseGqaFmha"]
