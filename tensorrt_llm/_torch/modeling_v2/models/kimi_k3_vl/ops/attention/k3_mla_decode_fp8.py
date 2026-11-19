# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuTe DSL FP8 MLA decode (``BlackwellMultiHeadLatentAttentionForwardFP8``) on a paged FP8 latent
cache, with its per-request split-KV plan.

Callers: the FP8-KV MLA layers (91 and 92) of the target and the DSpark drafter's attention. One
call is one launch of the decode kernel (its split pass, plus its reduction pass where the
workspace splits the KV); :func:`launch_split_plan` is the once-per-step Triton plan the decode
reads as ``block_split_kvs``.

The tactic is the one the served runner's autotuner can only pick (one tiler, one split
candidate, non-persistent): ``((128, 128), (128, 256), split_kv, False)`` with ``split_kv`` the
default split of the batch's power-of-two bucket (:func:`decode_tactic`), then lowered to what the
workspace fits.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.utils import last_positive_power_of_2
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div, pad_up

from ...kernels.k3r_mla_decode import cute_dsl_mla as _plan

#: The tilers of the only tactic (``mma_qk_tiler_mn``, ``mma_pv_tiler_mn``).
MMA_QK_TILER_MN = (128, 128)
MMA_PV_TILER_MN = (128, 256)
#: Split partials the reduction loads per round before summing them (same order, same bits; the
#: served runner's TRTLLM_K3_DRAFT3_REDUCE_UNROLL=8).
REDUCE_UNROLL = 8
#: CTA cluster of the decode (2-CTA MMA).
CLUSTER_SHAPE_MNK = (2, 1, 1)
#: Query rows per M tile: ``seq_len_q * num_heads`` rows of a request are packed into tiles.
MMA_TILE_M = 128
#: Largest split of any tactic.
MAX_SPLIT_KV = 32
#: CTA waves the default split targets.
SPLIT_KV_WAVES = 8
WORKSPACE_ALIGN = 128
LSE_DTYPE_BYTES = 4
ACC_BITS = 32

#: Compiled kernels by feature combination (process cache).
_KERNELS: dict = {}


@functools.cache
def max_active_blocks() -> int:
    """``max_active_clusters * 2``: the CTAs of one full wave of 2-CTA clusters. Queried once per
    process, before CUDA graph capture."""
    import cutlass

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("k3_mla_decode_fp8: max_active_blocks must be known before capture")
    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(math.prod(CLUSTER_SHAPE_MNK))
    return int(clusters) * CLUSTER_SHAPE_MNK[0]


def num_q_tiles(num_heads: int, seq_len_q: int) -> int:
    """Packed 128-row query tiles per request."""
    return ceil_div(num_heads * seq_len_q, MMA_TILE_M)


def split_plan_geometry(num_heads: int, seq_len_q: int) -> Tuple[int, int]:
    """``(rows_per_request, wave_clusters)`` of :func:`launch_split_plan` for this decode."""
    return num_q_tiles(num_heads, seq_len_q), max(max_active_blocks() // CLUSTER_SHAPE_MNK[0], 1)


def default_split_kv(batch: int, q_tiles: int, active_blocks: int) -> int:
    """About :data:`SPLIT_KV_WAVES` waves of ``(2, batch * q_tiles, split_kv)`` CTAs, a power of
    two, at most :data:`MAX_SPLIT_KV`."""
    target = SPLIT_KV_WAVES * active_blocks / (2 * batch * q_tiles)
    return min(MAX_SPLIT_KV, 1 << max(0, round(math.log2(target))))


def decode_tactic(batch_size: int, num_heads: int, seq_len_q: int):
    """``(mma_qk_tiler_mn, mma_pv_tiler_mn, split_kv, is_persistent)`` of a batch."""
    split = default_split_kv(
        last_positive_power_of_2(batch_size), num_q_tiles(num_heads, seq_len_q), max_active_blocks()
    )
    return MMA_QK_TILER_MN, MMA_PV_TILER_MN, split, False


@functools.cache
def _max_split_kv_rows(heads: int, seq_len_q: int, max_batch_size: int, active_blocks: int) -> int:
    rows = 0
    for s in range(1, seq_len_q + 1):
        tiles = num_q_tiles(heads, s)
        bucket = 1
        while bucket <= max(1, max_batch_size):
            split = default_split_kv(bucket, tiles, active_blocks)
            if split > 1:
                rows = max(rows, bucket * tiles * split)
            bucket *= 2
    return rows


def workspace_layout(
    num_heads: int, seq_len_q: int, kv_lora_rank: int, max_batch_size: int
) -> Tuple[int, int, int, int, int]:
    """``(lse_offset, lse_size, split_offset, split_size, total)`` bytes of the workspace: the
    LSE of ``max_batch_size`` requests, then the split partials of the largest bucket split."""
    lse_size = seq_len_q * max_batch_size * num_heads * LSE_DTYPE_BYTES
    split_offset = pad_up(lse_size, WORKSPACE_ALIGN)
    rows = _max_split_kv_rows(num_heads, seq_len_q, max_batch_size, max_active_blocks())
    split_size = rows * MMA_TILE_M * (kv_lora_rank + 1) * ACC_BITS // 8
    return 0, lse_size, split_offset, split_size, split_offset + pad_up(split_size, WORKSPACE_ALIGN)


def workspace_bytes(num_heads: int, seq_len_q: int, kv_lora_rank: int, max_batch_size: int) -> int:
    """Bytes of the workspace a decode of these dimensions needs (a fixed size per
    ``max_batch_size``, so one buffer serves every graph batch)."""
    return workspace_layout(num_heads, seq_len_q, kv_lora_rank, max_batch_size)[4]


def launch_split_plan(
    cache_seqs: torch.Tensor,
    plan: torch.Tensor,
    *,
    num_requests: int,
    rows_per_request: int,
    wave_clusters: int,
) -> None:
    """The per-request split counts of this step's KV lengths ``cache_seqs`` into ``plan`` (one
    Triton launch, ``_mla_split_kv_plan_kernel``)."""
    _plan.launch_mla_split_kv_plan(
        cache_seqs,
        plan,
        num_requests=num_requests,
        rows_per_request=rows_per_request,
        wave_clusters=wave_clusters,
    )


def _compile(
    key,
    *,
    num_heads: int,
    seq_len_q: int,
    page_size: int,
    is_var_split_kv: bool,
    emit_softmax_stats: bool,
    args,
):
    import cutlass
    import cutlass.cute as cute

    from ...kernels.k3r_mla_decode.mla_decode_fp8 import MAX_SPLITS as MLA_DECODE_MAX_SPLITS
    from ...kernels.k3r_mla_decode.mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8

    logger.info(f"k3_mla_decode_fp8: compiling kernel variant {key}")
    clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(math.prod(CLUSTER_SHAPE_MNK))
    mla = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        MMA_QK_TILER_MN,
        MMA_PV_TILER_MN,
        clusters,
        page_size,
        0.0,
        False,
        True,
        is_var_split_kv,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        reducer_max_splits=MLA_DECODE_MAX_SPLITS if is_var_split_kv else 32,
        emit_softmax_stats=emit_softmax_stats,
        reduce_unroll=REDUCE_UNROLL,
    )
    (q_latent, q_rope, c_latent, c_rope, page_table, o, lse, stats, split_ws, split_kv) = args[:10]
    cache_seqs, kv_bounds, block_split_kvs, softmax_scale, output_scale, stream = args[10:]
    out_dtype = cutlass.BFloat16 if o.dtype == torch.bfloat16 else cutlass.Float16

    def dyn(t, align=16, **kw):
        return cute.runtime.from_dlpack(t, assumed_align=align).mark_layout_dynamic(**kw)

    o_ct = dyn(o, leading_dim=1).mark_compact_shape_dynamic(
        mode=1, stride_order=(3, 2, 0, 1), divisibility=(128 // out_dtype.width)
    )
    compile_args = [
        dyn(q_latent, leading_dim=1),
        dyn(q_rope, leading_dim=1),
        dyn(c_latent, leading_dim=1),
        dyn(c_rope, leading_dim=1),
        dyn(page_table, leading_dim=0),
        o_ct,
        dyn(lse, leading_dim=0),
    ]
    target = mla
    if emit_softmax_stats:
        target = mla.run_with_softmax_stats
        compile_args.append(dyn(stats, leading_dim=3))
    compile_args += [
        cute.runtime.from_dlpack(split_ws, assumed_align=32).mark_layout_dynamic()
        if split_ws is not None
        else None,
        split_kv,
        cute.runtime.from_dlpack(cache_seqs, assumed_align=16).mark_layout_dynamic(),
        cute.runtime.from_dlpack(kv_bounds, assumed_align=4).mark_layout_dynamic()
        if kv_bounds is not None
        else None,
        None,
        cute.runtime.from_dlpack(block_split_kvs, assumed_align=4).mark_layout_dynamic()
        if is_var_split_kv
        else None,
        cute.runtime.from_dlpack(softmax_scale).mark_layout_dynamic(),
        cute.runtime.from_dlpack(output_scale).mark_layout_dynamic(),
        stream,
    ]
    return cute.compile(target, *compile_args, options="--opt-level 2")


class K3MlaDecodeFp8(OpWrapper):
    """Absorbed MLA decode of ``B`` requests x ``S_q`` query tokens x ``H`` heads on an FP8 E4M3
    paged latent cache: ``o`` (BF16, ``[H, 512, S_q, B]`` view) from the FP8 query halves
    ``q_latent`` ``[H, 512, S_q, B]`` / ``q_rope`` ``[H, 64, S_q, B]`` and the cache pages
    ``c_latent`` ``[page_size, 512, num_pages]`` / ``c_rope`` ``[page_size, 64, num_pages]`` (views of
    one ``[num_pages, page_size, 576]`` pool), ``page_table`` ``[max_pages, B]`` int32 (a
    transposed view), ``cache_seqs`` ``[B]`` int32 KV lengths (causal over the last ``S_q``
    positions; ``kv_bounds`` ``[B * S_q]`` int32 per-token rank-local bounds under Helix speculative
    verify), ``softmax_scale`` / ``output_scale`` float32 device scalars (element 0), the optional
    Helix ``softmax_stats`` ``[B * S_q, H, 2]`` (natural-log max, sum) and the optional per-request
    split plan ``block_split_kvs`` ``[B]`` int32 (:func:`launch_split_plan`). ``workspace`` holds
    at least :func:`workspace_bytes` bytes of ``max_batch_size``. Writes ``o``, ``softmax_stats`` and
    the workspace; returns None."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # Spec: `heads` H, `requests` B, `seq_len_q`, `kv_lens` per request, `helix` (softmax stats +
    # per-token bounds), `plan` (block_split_kvs). Latent 512, rope 64, page 128.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix16 ULL MLA layers 91 / 92: one request verifying 8 tokens, Helix stats",
            spec=dict(heads=96, requests=1, seq_len_q=8, kv_lens=[1500], helix=True, plan=True),
        ),
        Cell(
            why="helix16 ULL c4: four requests, skewed KV, the plan splits them differently",
            spec=dict(
                heads=96,
                requests=4,
                seq_len_q=8,
                kv_lens=[90, 4100, 700, 2049],
                helix=True,
                plan=True,
            ),
        ),
        Cell(
            why="DSpark drafter block decode: 64 heads, a block of 7 query tokens per request",
            spec=dict(heads=64, requests=2, seq_len_q=7, kv_lens=[257, 33], helix=True, plan=True),
        ),
    )

    note = """
    The split pass writes FP32 partials and LSE into the workspace where the split is above one and
    the workspace fits it; the reduction pass then merges them in split order. `block_split_kvs`
    applies only with that workspace (otherwise the call runs unsplit and ignores it). The kernel
    is compiled once per (heads, query length, page size, output dtype, workspace use, Helix
    bounds, plan) in a process cache: compile before graph capture. `max_batch_size` fixes the
    workspace layout (and with it the largest split a batch between two buckets keeps), so callers
    pass the engine's maximum request count, not the batch.
    """

    def __call__(
        self,
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        o: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        softmax_scale: torch.Tensor,
        output_scale: torch.Tensor,
        max_batch_size: int,
        softmax_stats: Optional[torch.Tensor],
        kv_bounds: Optional[torch.Tensor],
        block_split_kvs: Optional[torch.Tensor] = None,
    ) -> None:
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda

        if get_sm_version() not in (100, 103, 107):
            raise ValueError(
                f"k3_mla_decode_fp8 requires SM100 / SM103 / SM107, got SM{get_sm_version()}"
            )
        batch_size = cache_seqs.shape[0]
        _, _, split_kv, is_persistent = decode_tactic(batch_size, num_heads, seq_len_q)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        d_latent = q_latent.shape[1]
        stats_kernel = None
        if softmax_stats is not None:
            stats_kernel = softmax_stats.view(batch_size, seq_len_q, num_heads, 2).permute(
                2, 1, 0, 3
            )
        max_batch = max(batch_size, max_batch_size)
        lse_offset, lse_size, split_offset, split_size, _ = workspace_layout(
            num_heads, seq_len_q, d_latent, max_batch
        )
        ws = workspace.view(torch.uint8).reshape(-1)
        lse = (
            ws[lse_offset : lse_offset + lse_size]
            .view(torch.float32)
            .view(max_batch, seq_len_q, num_heads)[:batch_size]
            .permute(2, 1, 0)
        )
        split_ws = ws[split_offset : split_offset + split_size]
        split_rows = split_size // (MMA_TILE_M * (d_latent + 1) * cutlass.Float32.width // 8)
        fitting = split_rows // (batch_size * num_q_tiles(num_heads, seq_len_q))
        use_workspace = split_kv > 1 and fitting >= 1
        split_kv = max(1, min(split_kv, fitting))
        if not use_workspace:
            block_split_kvs = None
        is_var_split_kv = block_split_kvs is not None
        out_dtype = torch.bfloat16 if o.dtype == torch.bfloat16 else torch.float16
        key = (
            num_heads,
            seq_len_q,
            page_size,
            softmax_stats is not None,
            out_dtype,
            use_workspace,
            is_persistent,
            kv_bounds is not None,
            is_var_split_kv,
            REDUCE_UNROLL,
        )
        compiled = _KERNELS.get(key)
        args = (
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            o,
            lse,
            stats_kernel,
            split_ws if use_workspace else None,
            split_kv,
            cache_seqs,
            kv_bounds,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
        )
        if compiled is None:
            compiled = _compile(
                key,
                num_heads=num_heads,
                seq_len_q=seq_len_q,
                page_size=page_size,
                is_var_split_kv=is_var_split_kv,
                emit_softmax_stats=softmax_stats is not None,
                args=args,
            )
            _KERNELS[key] = compiled
        page_table_arg = page_table
        if page_table.shape[0] == 1 and page_table.shape[1] == 1:
            # A (1, 1) table has no extent > 1 to deduce its leading dimension from at call time.
            page_table_arg = cute.runtime.from_dlpack(
                page_table, assumed_align=16
            ).mark_layout_dynamic(leading_dim=0)
        runtime_args = [q_latent, q_rope, c_latent, c_rope, page_table_arg, o, lse]
        if softmax_stats is not None:
            runtime_args.append(stats_kernel)
        runtime_args += [
            split_ws if use_workspace else None,
            split_kv,
            cache_seqs,
            kv_bounds,
            None,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
        ]
        compiled(*runtime_args)

    def reference(
        self,
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        o: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        softmax_scale: torch.Tensor,
        output_scale: torch.Tensor,
        max_batch_size: int,
        softmax_stats: Optional[torch.Tensor],
        kv_bounds: Optional[torch.Tensor],
        block_split_kvs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FP32 attention of every query token over its causal KV prefix: ``(o [H, 512, S_q, B]
        BF16, stats [B * S_q, H, 2] or None)``. ``workspace``, ``max_batch_size`` and
        ``block_split_kvs`` do not change the result."""
        heads, latent, s_q, batch = q_latent.shape
        out = torch.empty((heads, latent, s_q, batch), dtype=o.dtype, device=o.device)
        stats = None
        if softmax_stats is not None:
            stats = torch.empty((batch * s_q, heads, 2), dtype=torch.float32, device=o.device)
        sm = softmax_scale.view(-1)[0].float()
        osc = output_scale.view(-1)[0].float()
        for b in range(batch):
            kv_len = int(cache_seqs[b])
            pages = page_table[:, b][: ceil_div(kv_len, page_size)].long()
            k_lat = c_latent[:, :, pages].permute(2, 0, 1).reshape(-1, latent)[:kv_len].float()
            k_rope = (
                c_rope[:, :, pages].permute(2, 0, 1).reshape(-1, c_rope.shape[1])[:kv_len].float()
            )
            for t in range(s_q):
                bound = kv_len - (s_q - 1) + t
                if kv_bounds is not None:
                    bound = int(kv_bounds[b * s_q + t])
                ql = q_latent[:, :, t, b].float()
                qr = q_rope[:, :, t, b].float()
                logits = (ql @ k_lat[:bound].t() + qr @ k_rope[:bound].t()) * sm
                row_max = logits.amax(-1, keepdim=True)
                p = torch.exp(logits - row_max)
                row_sum = p.sum(-1, keepdim=True)
                out[:, :, t, b] = ((p @ k_lat[:bound]) / row_sum * osc).to(o.dtype)
                if stats is not None:
                    stats[b * s_q + t, :, 0] = row_max.squeeze(-1)
                    stats[b * s_q + t, :, 1] = row_sum.squeeze(-1)
        return out, stats

    def is_valid(
        self,
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        o: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        softmax_scale: torch.Tensor,
        output_scale: torch.Tensor,
        max_batch_size: int,
        softmax_stats: Optional[torch.Tensor],
        kv_bounds: Optional[torch.Tensor],
        block_split_kvs: Optional[torch.Tensor] = None,
    ) -> None:
        batch = cache_seqs.shape[0]
        need = workspace_bytes(num_heads, seq_len_q, q_latent.shape[1], max(batch, max_batch_size))
        got = workspace.numel() * workspace.element_size()
        assert workspace.is_contiguous() and got >= need, f"workspace {got} < {need} bytes"
        if softmax_stats is not None:
            assert softmax_stats.is_contiguous() and softmax_stats.dtype == torch.float32
            assert tuple(softmax_stats.shape) == (batch * seq_len_q, num_heads, 2)
        if kv_bounds is not None:
            assert kv_bounds.dtype == torch.int32 and kv_bounds.is_contiguous()
            assert kv_bounds.numel() == batch * seq_len_q
        if block_split_kvs is not None:
            assert block_split_kvs.dtype == torch.int32 and block_split_kvs.is_contiguous()
            assert block_split_kvs.numel() == batch


k3_mla_decode_fp8 = K3MlaDecodeFp8()

__all__ = [
    "K3MlaDecodeFp8",
    "decode_tactic",
    "k3_mla_decode_fp8",
    "launch_split_plan",
    "max_active_blocks",
    "num_q_tiles",
    "split_plan_geometry",
    "workspace_bytes",
    "workspace_layout",
]
