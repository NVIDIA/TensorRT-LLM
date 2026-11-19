# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/common/ops/sparse_attn.py
"""Triton block-sparse GQA decode attention for MiniMax-M3.

Flash-decoding over the blocks the indexer selected: one CTA per (query token,
top-k chunk, KV head) accumulates a partial output plus its log-sum-exp, and a
second kernel merges the chunks by LSE weight. That beats running the
context-schedule FMHA kernel at decode, where a single query token leaves most
of a 128-row Q tile idle.

GB300/SM103 tuning targets GQA group 16, top-k 16 and four-token Eagle3
verification. Split-K, query pairing and launch tables require four KV heads;
the linear-softmax policy also covers one and two KV heads. Other hardware
and shapes retain the default policies, including supported SM100 NVFP4.

Vendored from the vLLM source linked in the file header (v0.26.1rc0-77-g6f91edf96).
Differences from upstream:

* K and V are separate HND paged views ([num_pages, num_kv_heads, page_size,
  head_dim]) with independent strides rather than one cache fused along the
  last dim, because that is how the M3 KV pool is laid out here.
* Only the scalar KV-scale mode is kept; M3 stores unscaled E4M3 K/V.
* A packed NVFP4 cache is read in place, dequantized in-register from its E2M1
  nibbles and E4M3 block scales.
* Partial-output and LSE scratch come from the persistent buffer arena so their
  addresses survive CUDA graph replay.
* A single top-k chunk writes the final output directly, avoiding scratch and
  the otherwise redundant LSE merge.
* The merge kernel zeroes rows whose chunks are all empty instead of letting
  them produce NaN, since CUDA-graph padding rows flow on into the rest of the
  network here.

This is the MiniMax-M3 NVFP4 / sm_107 / tp2 target's copy. On top of the
in-tree decode it routes long-context TP4 steps to the one-launch
``m3_attn_lc`` kernel, short-sequence verify steps of at least
``MQ_MIN_TOKENS`` rows to the multi-query producer, releases PDL dependents
early, and has the merge emit the MXFP8 o_proj hand-off (``m3_prequant.SLOT``).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton.language as tl

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    check_decode_span_shape,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    _NVFP4_TUNED_CAPABILITIES as _NVFP4_TUNED_CAPABILITIES,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    _TARGET_GRID as _TARGET_GRID,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    MergeMxfp8 as MergeMxfp8,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    _check_nvfp4_inputs as _check_nvfp4_inputs,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    _sm100f_nvfp4_launch_options as _sm100f_nvfp4_launch_options,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    _sm100f_nvfp4_merge_launch_options as _sm100f_nvfp4_merge_launch_options,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    _sm100f_nvfp4_num_topk_chunks as _sm100f_nvfp4_num_topk_chunks,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    _sm100f_nvfp4_query_group_size as _sm100f_nvfp4_query_group_size,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    _sm100f_nvfp4_use_linear_softmax as _sm100f_nvfp4_use_linear_softmax,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.sparse_decode_policy import (
    resolve_num_topk_chunks as resolve_num_topk_chunks,
)

from ...kernels.attention.triton_sparse_decode import _dequant_nvfp4_rows as _dequant_nvfp4_rows
from ...kernels.attention.triton_sparse_decode import _e2m1x2_to_f16x2 as _e2m1x2_to_f16x2
from ...kernels.attention.triton_sparse_decode import (
    _gqa_sparse_decode_kernel as _gqa_sparse_decode_kernel,
)
from ...kernels.attention.triton_sparse_decode import (
    _gqa_sparse_decode_pair_kernel as _gqa_sparse_decode_pair_kernel,
)
from ...kernels.attention.triton_sparse_decode import (
    _merge_topk_attn_out_kernel as _merge_topk_attn_out_kernel,
)
from ...kernels.attention.triton_sparse_decode import _rcp_approx_ftz as _rcp_approx_ftz
from ...kernels.attention.triton_sparse_decode import _store_mxfp8_head as _store_mxfp8_head
from . import m3_attn_lc, m3_prequant, m3_report

# One sparse block is exactly one KV page.
SPARSE_BLOCK_SIZE = 128  # capacity: fixed: the MSA block size of the checkpoint

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_TL_DTYPES = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}

# Total CTAs the split-K partitioning aims for before it stops splitting.

# NVFP4 quantizes in groups of 16 cache elements along head_dim, one E4M3 scale
# byte each; see the fp4_quantize call in msa_scatter.fused_write_layer_caches_nvfp4.
NVFP4_SF_VEC_SIZE = 16


# Fewest query rows for which the merge writes the MXFP8 o_proj hand-off: every decode step, so the MXFP8 o_proj
# never runs its own quantize kernel after attention.
MERGE_MXFP8_MIN_TOKENS = 1  # capacity: fixed: a row lower bound
# The sparse decode (LC2, the Triton merge) also writes the MXFP8 o_proj hand-off (bit for bit
# torch.ops.trtllm.mxfp8_quantize of its BF16 rows) and publishes it to m3_prequant.SLOT.
MERGE_MXFP8_HANDOFF = True
# The long-context decode (m3_attn_lc) stages the request's block-table row before its PDL grid wait: off (measured
# +1.6..+1.8 us at 32 rows; flat to -0.2 us below), up to LC_PREWAIT_MAX_ROWS query rows when on.
LC_PREWAIT = False
LC_PREWAIT_MAX_ROWS = 16  # capacity: fixed: measured occupancy crossover
# The long-context decode's shapes (modeling's SPARSE_DECODE_LC keys: "mxfp8" = a step of at most MergeMxfp8.MAX_ROWS
# rows) and the identity-selection producer's (modeling's SPARSE_DECODE_PRODUCER keys), with their report paths.
LC_KEYS = frozenset({"mxfp8"})
LC_PATH = "m3_sparse_decode_lc"
PRODUCER_KEYS = frozenset({"per_row"})
PRODUCER_PATH = "sparse_decode_producer"


MERGE_MXFP8 = MergeMxfp8()


# NVFP4 instructions are supported on the datacenter Blackwell and Rubin targets.
_SM100F_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
# Shape-specific tuning was measured on GB300 only. SM100 uses the defaults.


# Multi-query producer (triton_sparse_decode_mq) for identity-selection verify steps with at least this many query
# tokens; below it the per-token producer is as fast.
MQ_MIN_TOKENS = 17  # capacity: fixed: the multi-query producer's crossover
# Warps per multi-query producer program.
MQ_NUM_WARPS = 8


def _longattn_lc_decode(
    q,
    k_paged,
    v_paged,
    topk_idx,
    block_table,
    seq_lens,
    *,
    sm_scale,
    output,
    decode_query_len,
    k_block_scale,
    v_block_scale,
    k_global_scale,
    v_global_scale,
) -> None:
    """minimax_m3_sparse_attn_decode through m3_attn_lc: same table, output and MXFP8 hand-off contract."""
    use_pdl = get_env_enable_pdl()
    total_q, num_heads, head_dim = q.shape
    prewait = bool(LC_PREWAIT) and total_q <= LC_PREWAIT_MAX_ROWS
    mxfp8 = None
    m3_prequant.SLOT.clear()
    if MERGE_MXFP8_HANDOFF and MERGE_MXFP8_MIN_TOKENS <= total_q <= MergeMxfp8.MAX_ROWS:
        mxfp8 = MERGE_MXFP8.buffers(num_heads * head_dim, output.device)
    # The early trigger lets the o_proj launch and stream its weights while attention runs.
    m3_attn_lc.m3_lc_attn(
        q,
        k_paged,
        v_paged,
        topk_idx,
        block_table,
        seq_lens,
        sm_scale=sm_scale,
        output=output,
        decode_query_len=decode_query_len,
        k_block_scale=k_block_scale,
        v_block_scale=v_block_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        linear_softmax=False,
        mxfp8=mxfp8,
        use_pdl=use_pdl,
        prewait=prewait,
        n1_ids=m3_attn_lc.N1_IDS.take(),
    )
    if prewait:
        m3_report.taken(LC_PREWAIT_PATH, total_q)
    if mxfp8 is not None:
        m3_prequant.SLOT.publish(
            output.view(total_q, num_heads * head_dim), *MergeMxfp8.handoff(mxfp8, total_q)
        )


# Engagement path of the identity-selection producers reading their blocks from the kv lengths (IDENTITY).
DEVICE_IDENTITY_PATH = "sparse_decode_device_identity"
# Engagement path of LC2's pre-wait block-table staging (LC_PREWAIT, up to its row cap).
LC_PREWAIT_PATH = "sparse_decode_lc_prewait"


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_paged: torch.Tensor,  # [num_pages, num_kv_heads, page_size, head_dim]
    v_paged: torch.Tensor,  # same layout as k_paged
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    *,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    kv_scale: Optional[torch.Tensor] = None,
    k_block_scale: Optional[torch.Tensor] = None,
    v_block_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[torch.Tensor] = None,
    v_global_scale: Optional[torch.Tensor] = None,
    num_topk_chunks: Optional[int] = None,
    identity_blocks: bool = False,
) -> None:
    """Block-sparse GQA decode attention, written into output in place.

    q may be FP8, as a fused QK-norm/RoPE producer emits it. The kernel widens
    it to output.dtype in-register rather than making the caller materialize a
    widened copy, which would cost a standalone kernel on every sparse layer.
    The attention math is unchanged either way, since E4M3 widens exactly.

    kv_scale is an optional scalar dequantization factor for an FP8 cache;
    MiniMax-M3 stores unscaled E4M3, so it is normally None.

    Passing the four NVFP4 scale tensors selects the packed-NVFP4 cache instead:
    k_paged/v_paged then hold two E2M1 elements per byte, the block scales are
    paged E4M3 bytes in the layouts msa_scatter writes, and the two per-tensor
    dequant scales are single-element FP32 device tensors.

    num_topk_chunks overrides the split-K factor and exists for tests: the
    merged result must not depend on it.
    """
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = int(k_paged.shape[1])
    check_decode_span_shape(
        "MiniMax-M3 Triton sparse decode", total_q, int(seq_lens.shape[0]), decode_query_len
    )
    if int(k_paged.shape[2]) != SPARSE_BLOCK_SIZE:
        raise ValueError(
            f"MiniMax-M3 sparse decode requires page_size={SPARSE_BLOCK_SIZE}; "
            f"got {int(k_paged.shape[2])}."
        )
    if num_heads % num_kv_heads:
        # The kernel covers heads pid_kh * gqa_group_size + [0, group), so a
        # head count that is not a whole number of groups leaves the tail heads
        # unwritten and the merge kernel reads uninitialized partials.
        raise ValueError(
            f"MiniMax-M3 sparse decode requires num_heads ({num_heads}) to be a "
            f"multiple of num_kv_heads ({num_kv_heads})."
        )
    nvfp4_args = (k_block_scale, v_block_scale, k_global_scale, v_global_scale)
    kv_nvfp4 = any(arg is not None for arg in nvfp4_args)
    scale_cols = head_dim // NVFP4_SF_VEC_SIZE
    if kv_nvfp4:
        capability = torch.cuda.get_device_capability(q.device)
        if capability not in _SM100F_CAPABILITIES:
            raise NotImplementedError(
                "MiniMax-M3 NVFP4 sparse decode requires SM100/SM103/SM107; "
                f"got SM{capability[0] * 10 + capability[1]}."
            )
        _check_nvfp4_inputs(k_paged, v_paged, nvfp4_args, head_dim=head_dim, scale_cols=scale_cols)
        # Two E2M1 elements per byte; Triton must see them unsigned so that the
        # high-nibble shift does not sign-extend.
        k_paged, v_paged = k_paged.view(torch.uint8), v_paged.view(torch.uint8)
        k_block_scale, v_block_scale = (
            k_block_scale.view(torch.uint8),
            v_block_scale.view(torch.uint8),
        )
    max_topk = int(topk_idx.shape[-1])
    gqa_group_size = num_heads // num_kv_heads
    use_scale = k_paged.dtype in _FP8_DTYPES and kv_scale is not None
    # Triton needs a real pointer even for the unused arguments.
    scale_arg = kv_scale if use_scale else output
    k_sc_arg, v_sc_arg = (k_block_scale, v_block_scale) if kv_nvfp4 else (output, output)
    k_gs_arg, v_gs_arg = (k_global_scale, v_global_scale) if kv_nvfp4 else (output, output)

    widen_q = q.dtype in _FP8_DTYPES
    if widen_q and output.dtype not in _TL_DTYPES:
        raise ValueError(
            f"MiniMax-M3 sparse decode cannot widen FP8 q into {output.dtype}; "
            f"supported compute dtypes are {sorted(d.__str__() for d in _TL_DTYPES)}."
        )
    # Packed NVFP4 dequantizes to FP16. The production producer supplies FP8
    # q, which widens exactly to FP16, so keep both dot operands there instead
    # of converting every K/V tile to BF16. Accumulation remains FP32.
    if widen_q:
        compute_dtype = tl.float16 if kv_nvfp4 else _TL_DTYPES[output.dtype]
    else:
        compute_dtype = tl.float32

    # Long-context TP2 steps: attention, split-K merge and MXFP8 hand-off in one launch. A selected table on an NVFP4
    # cache with FP8 q at the TP2 heads is the kernel's (modeling's SPARSE_DECODE_LC); operands it declines are a miss.
    lc_key = "mxfp8" if MERGE_MXFP8_MIN_TOKENS <= total_q <= MergeMxfp8.MAX_ROWS else "plain"
    lc_covered = (
        kv_nvfp4
        and widen_q
        and not identity_blocks
        and num_topk_chunks is None
        and num_heads == 32
        and num_kv_heads == 2
        and lc_key in LC_KEYS
    )
    if lc_covered and not m3_attn_lc.lc_applies(
        q, k_paged, v_paged, topk_idx, k_block_scale, v_block_scale, output
    ):
        m3_report.missed(
            LC_PATH,
            total_q,
            "the step's q / KV pages / scales / top-k table / output are not the layout the kernel is built for",
        )
        lc_covered = False
    if lc_covered:
        _longattn_lc_decode(
            q,
            k_paged,
            v_paged,
            topk_idx,
            block_table,
            seq_lens,
            sm_scale=sm_scale,
            output=output,
            decode_query_len=decode_query_len,
            k_block_scale=k_block_scale,
            v_block_scale=v_block_scale,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
        )
        return

    query_group_size = (
        _sm100f_nvfp4_query_group_size(
            total_q=total_q,
            num_kv_heads=num_kv_heads,
            gqa_group_size=gqa_group_size,
            max_topk=max_topk,
            decode_query_len=decode_query_len,
        )
        if kv_nvfp4
        else 1
    )
    if num_topk_chunks is None:
        if query_group_size == 2:
            measured_chunks = 8
        else:
            measured_chunks = (
                _sm100f_nvfp4_num_topk_chunks(
                    total_q=total_q,
                    num_kv_heads=num_kv_heads,
                    gqa_group_size=gqa_group_size,
                    max_topk=max_topk,
                    decode_query_len=decode_query_len,
                )
                if kv_nvfp4
                else None
            )
        num_topk_chunks = (
            measured_chunks
            if measured_chunks is not None
            else resolve_num_topk_chunks(total_q, num_kv_heads, max_topk)
        )
    elif num_topk_chunks <= 0 or num_topk_chunks & (num_topk_chunks - 1):
        raise ValueError(f"num_topk_chunks must be a positive power of two; got {num_topk_chunks}.")

    # Every query token of a request reads the same blocks under identity selection (short-sequence mode): one
    # program per (request, block) covers all of them and dequantizes each block once.
    use_mq = (
        identity_blocks
        and total_q >= MQ_MIN_TOKENS
        and kv_nvfp4
        and num_kv_heads == 1
        and decode_query_len > 1
        and head_dim == 128
        and gqa_group_size * decode_query_len <= 128
    )
    if identity_blocks and kv_nvfp4 and widen_q:
        # The identity-selection route is modeling's SPARSE_DECODE_PRODUCER ("per_row"; "multi_query" needs one KV
        # head, so it never runs at the TP2 heads); a variant the range does not certify is a miss (raises under
        # require).
        producer_key = "multi_query" if use_mq else "per_row"
        if producer_key not in PRODUCER_KEYS:
            m3_report.missed(
                PRODUCER_PATH,
                total_q,
                f"the identity-selection producer variant {producer_key!r} is not certified at {num_kv_heads} KV heads",
            )
    if use_mq:
        num_topk_chunks = max_topk
    direct_output = num_topk_chunks == 1
    if direct_output:
        # With one chunk, acc_o is already normalized and final. Point the
        # producer at the caller's output and compile away the LSE store.
        # Keeping a real dummy pointer preserves Triton's pointer contract.
        o_arg = output
        lse_arg = output
        o_strides = (0, output.stride(0), output.stride(1), output.stride(2))
        lse_strides = (0, 0, 0)
    else:
        # Persistent arena rather than torch.empty, so the partials keep one
        # address across CUDA graph replays. FP32 keeps the split-K factor from
        # perturbing the merged result.
        reserve = torch.cuda.is_current_stream_capturing()
        o_arg = get_memory_buffers().get_buffer(
            [num_topk_chunks, total_q, num_heads, head_dim],
            torch.float32,
            buffer_name="m3_sparse_decode_o_partial",
            reserve_buffer=reserve,
        )
        lse_arg = get_memory_buffers().get_buffer(
            [num_topk_chunks, total_q, num_heads],
            torch.float32,
            buffer_name="m3_sparse_decode_lse_partial",
            reserve_buffer=reserve,
        )
        o_strides = tuple(o_arg.stride())
        lse_strides = tuple(lse_arg.stride())

    use_pdl = get_env_enable_pdl()
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    use_linear_softmax = kv_nvfp4 and _sm100f_nvfp4_use_linear_softmax(
        total_q=total_q,
        num_kv_heads=num_kv_heads,
        gqa_group_size=gqa_group_size,
        max_topk=max_topk,
        decode_query_len=decode_query_len,
    )
    if query_group_size == 2:
        launch_options = {"num_warps": 4, "num_stages": 1}
    else:
        launch_options = (
            _sm100f_nvfp4_launch_options(
                total_q=total_q,
                num_kv_heads=num_kv_heads,
                gqa_group_size=gqa_group_size,
                max_topk=max_topk,
                decode_query_len=decode_query_len,
            )
            if kv_nvfp4
            else {}
        )

    # The pair path merges the two sorted selector rows on device. Equal blocks
    # share one K/V dequantization; divergent blocks update only their owning
    # query, so the optimization does not rely on runtime selection equality.
    if total_q % query_group_size or decode_query_len % query_group_size:
        raise ValueError(
            f"query group {query_group_size} must divide total_q={total_q} and "
            f"decode_query_len={decode_query_len}."
        )

    producer_kernel = (
        _gqa_sparse_decode_pair_kernel if query_group_size == 2 else _gqa_sparse_decode_kernel
    )
    device_identity = identity_blocks
    if device_identity and not use_mq:
        m3_report.taken(DEVICE_IDENTITY_PATH, total_q)
    single_kernel_args = (
        {
            # No split-K arrival counters: the separate merge kernel combines the chunks.
            "merge": (output, None),
            "tail": (output.stride(0), output.stride(1), output.stride(2), num_kv_heads),
            # An early release pays off when the dependent is the consumer of the
            # final output (it can prefetch), or a merge grid that itself releases
            # its dependent at entry: the o_proj then launches during this kernel and
            # streams its weights while attention runs. Above 8 query rows the
            # concurrent weight stream slows the larger attention grid more than the
            # prefetch saves (GB300 TP4: -1.1% at 48 rows).
            "options": {"FUSED_MERGE": False, "EARLY_TRIGGER": direct_output or total_q <= 8},
        }
        if query_group_size == 1
        else {"merge": (), "tail": (), "options": {}}
    )
    pair_kernel_options = (
        {"QUERY_GROUP_SIZE": query_group_size, "MAX_TOPK": max_topk}
        if query_group_size == 2
        else {}
    )
    if use_mq:
        from ...kernels.attention.triton_sparse_decode_mq import launch_mq_producer

        launch_mq_producer(
            q,
            k_paged,
            v_paged,
            k_block_scale,
            v_block_scale,
            k_global_scale,
            v_global_scale,
            topk_idx,
            block_table,
            seq_lens,
            o_arg,
            lse_arg,
            sm_scale=sm_scale,
            decode_query_len=decode_query_len,
            use_pdl=use_pdl,
            early_trigger=False,
            num_warps=MQ_NUM_WARPS,
        )
    else:
        producer_kernel[
            (total_q // query_group_size) * num_topk_chunks,
            num_kv_heads,
        ](
            q,
            k_paged,
            v_paged,
            scale_arg,
            k_sc_arg,
            v_sc_arg,
            k_gs_arg,
            v_gs_arg,
            topk_idx,
            o_arg,
            lse_arg,
            *single_kernel_args["merge"],
            block_table,
            seq_lens,
            total_q,
            gqa_group_size,
            head_dim,
            max_topk,
            sm_scale,
            decode_query_len,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_paged.stride(0),
            k_paged.stride(1),
            k_paged.stride(2),
            k_paged.stride(3),
            v_paged.stride(0),
            v_paged.stride(1),
            v_paged.stride(2),
            v_paged.stride(3),
            k_sc_arg.stride(0),
            k_sc_arg.stride(1),
            v_sc_arg.stride(0),
            v_sc_arg.stride(1),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            *o_strides,
            *lse_strides,
            block_table.stride(0),
            block_table.shape[1],
            k_paged.shape[0],
            *single_kernel_args["tail"],
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_SCALE=use_scale,
            WIDEN_Q=widen_q,
            COMPUTE_DTYPE=compute_dtype,
            USE_PDL=use_pdl,
            KV_NVFP4=kv_nvfp4,
            USE_LINEAR_SOFTMAX=use_linear_softmax,
            SCALE_COLS=scale_cols,
            DIRECT_OUTPUT=direct_output,
            # Identity selection from the lengths this kernel reads, not the staged table.
            IDENTITY=device_identity,
            **single_kernel_args["options"],
            **pair_kernel_options,
            **pdl_launch,
            **launch_options,
        )
    m3_prequant.SLOT.clear()
    if not direct_output:
        width = num_heads * head_dim
        mxfp8 = (
            MERGE_MXFP8.buffers(width, output.device)
            if MERGE_MXFP8_HANDOFF
            and MERGE_MXFP8_MIN_TOKENS <= total_q <= MergeMxfp8.MAX_ROWS
            and head_dim == 128
            and output.dim() == 3
            and output.is_contiguous()
            and output.dtype == torch.bfloat16
            else None
        )
        mxfp8_args = (
            {
                "q8_ptr": mxfp8[0],
                "sf_ptr": mxfp8[1],
                "num_sf_k_tiles": width // 128,
                "MXFP8_OUT": True,
            }
            if mxfp8 is not None
            else {}
        )
        merge_launch_options = (
            _sm100f_nvfp4_merge_launch_options(
                total_q=total_q,
                num_kv_heads=num_kv_heads,
                gqa_group_size=gqa_group_size,
                max_topk=max_topk,
                decode_query_len=decode_query_len,
            )
            if kv_nvfp4
            else {}
        )
        _merge_topk_attn_out_kernel[(total_q, num_heads)](
            o_arg,
            lse_arg,
            output,
            head_dim,
            *o_strides,
            *lse_strides,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_PDL=use_pdl,
            # Release the o_proj at the merge's entry so it streams its weights during the merge.
            EARLY_TRIGGER=True,
            **mxfp8_args,
            **pdl_launch,
            **merge_launch_options,
        )
        if mxfp8 is not None:
            m3_prequant.SLOT.publish(
                output.view(total_q, width), *MergeMxfp8.handoff(mxfp8, total_q)
            )


__all__ = [
    "MERGE_MXFP8",
    "MERGE_MXFP8_MIN_TOKENS",
    "MergeMxfp8",
    "NVFP4_SF_VEC_SIZE",
    "SPARSE_BLOCK_SIZE",
    "_sm100f_nvfp4_merge_launch_options",
    "_sm100f_nvfp4_num_topk_chunks",
    "_sm100f_nvfp4_query_group_size",
    "_sm100f_nvfp4_use_linear_softmax",
    "minimax_m3_sparse_attn_decode",
    "resolve_num_topk_chunks",
]
