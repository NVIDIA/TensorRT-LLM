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
"""The MLA DSpark drafter's helix layer, with fewer launches.

The MLA drafter keeps one latent per token, so its context pages can be sharded over a helix group:
every rank attends all heads over its pages and the per-rank partials are merged by a helix
exchange. Switches (each falls back to the current path where it does not apply):

* :func:`ctx_split_mapping` (Kimi K3.1 generation servers on plain TP): that page-sharded attention and
  exchange on a plain TP group, over the replicated context, from ``TRTLLM_DSPARK_CTX_SPLIT_MIN_BATCH``
  (default 8) generation requests.
* :func:`draft_helix_store_enabled` (with the fused tail kernels, drafting Kimi K3.1 engines without attention
  DP): under helix too, one kernel
  rotates the query's rope half, normalizes and rotates the block latents into a scratch row
  buffer, and one masked scatter stores the rows this rank's pages hold
  (``scatter_rows_to_pool(rows_mask=owned)``), instead of the separate RoPE launches, an FP8 cast
  and a gather / where / index_put round trip of the pool; the FP8 query of the decode is cast in
  one launch (``to_draft_fp8_1k``) instead of a clamp and a cast. Same values.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm.mapping import CpType, Mapping

from ..pyexecutor.input_commit import k31_draft_tp_engine, k31_tp_generation_server


def draft_helix_store_enabled() -> bool:
    """Whether the drafter takes the fused block RoPE with an owner-masked pool store and the
    one-launch FP8 query cast (same values)."""
    return k31_draft_tp_engine()


@triton.jit
def _draft_fp8_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    # clamp(-448, 448) keeping NaN (min / max would drop it), then round to nearest even.
    clamped = tl.minimum(tl.maximum(v, -448.0), 448.0)
    v = tl.where(v != v, v, clamped)
    tl.store(dst_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def to_draft_fp8_1k(x: torch.Tensor) -> torch.Tensor:
    """``to_draft_fp8(x)`` (saturated E4M3 at unit scale) of a contiguous tensor in one launch."""
    assert x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    numel = x.numel()
    if numel:
        block = 4096
        _draft_fp8_kernel[(triton.cdiv(numel, block),)](x, out, numel, BLOCK=block, num_warps=4)
    return out


CTX_SPLIT_MIN_BATCH_ENV_VAR = "TRTLLM_DSPARK_CTX_SPLIT_MIN_BATCH"
CTX_SPLIT_A2A_ENV_VAR = "TRTLLM_DSPARK_CTX_SPLIT_A2A"
_CTX_SPLIT_DEFAULT_MIN_BATCH = 8


def ctx_split_mapping(mapping: Mapping) -> Optional[Mapping]:
    """The drafter's context-split mapping on a plain TP group, or None.

    On Kimi K3.1 generation servers under plain tensor parallelism (no attention DP, no helix, one pipeline stage,
    the TP group = the world) the MLA drafter's latent context, replicated on every rank, is read by page instead
    of by head: global page ``g`` of a request is attended by TP rank ``g % tp`` with every head (``q_b`` /
    ``kv_b`` replicated), and the per-rank partials are merged by the helix exchange back to each rank's
    ``64 / tp`` heads for the unchanged row-parallel ``o_proj``. Every rank keeps and writes the whole context
    (the cache layout, the KV manager and the transceiver are unchanged); only the reads are split.

    The returned helix mapping (``tp 1, cp = tp``) puts CP rank ``r`` on TP rank ``r``. Its exchange is the MNNVL
    FIFO all-to-all; ``TRTLLM_DSPARK_CTX_SPLIT_A2A=nccl`` selects the NCCL all-to-all.
    """
    if not k31_tp_generation_server():
        return None
    if (
        mapping.tp_size <= 1
        or mapping.enable_attention_dp
        or mapping.cp_size != 1
        or mapping.pp_size != 1
        or mapping.world_size != mapping.tp_size
    ):
        return None
    use_nccl = os.environ.get(CTX_SPLIT_A2A_ENV_VAR, "native") == "nccl"
    return Mapping(
        world_size=mapping.world_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        cp_size=mapping.tp_size,
        cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": use_nccl},
        tp_size=1,
        pp_size=1,
        moe_tp_size=mapping.moe_tp_size,
        moe_ep_size=mapping.moe_ep_size,
        moe_cluster_size=mapping.moe_cluster_size,
    )


def ctx_split_min_batch() -> int:
    """Smallest generation batch that takes the context split (``TRTLLM_DSPARK_CTX_SPLIT_MIN_BATCH``, default 8);
    below it every rank decodes its own heads over the whole context, as without the split."""
    return int(os.environ.get(CTX_SPLIT_MIN_BATCH_ENV_VAR, _CTX_SPLIT_DEFAULT_MIN_BATCH))


def ctx_split_local_page_table(
    page_tables: torch.Tensor, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """This rank's pages of ``[B, pages]`` block tables under the context split, as a fresh row-major int32
    ``[B, max(2, ceil(pages / cp))]`` table: local page ``k`` is global page ``k * cp + cp_rank``
    (``HelixDraftShard``'s layout). Columns past this rank's last page are 0 and never read (the local lengths
    bound them). A fresh buffer, not a strided view: the decode reads the transposed table, which needs unit
    stride along the batch, and a width-1 slice would keep the parent's strides."""
    batch, pages = page_tables.shape
    local = page_tables[:, cp_rank::cp_size]
    out = torch.zeros(
        (batch, max(2, -(-pages // cp_size))), dtype=torch.int32, device=page_tables.device
    )
    out[:, : local.shape[1]].copy_(local)
    return out


@triton.jit
def _ctx_split_fixup_kernel(
    ctx_len_ptr,
    rows_ptr,
    table_ptr,
    table_stride,
    table_width,
    local_ptr,
    local_width,
    pages_ptr,
    offsets_ptr,
    owned_ptr,
    seq_lens_ptr,
    kv_bounds_ptr,
    page_size,
    cp_size,
    cp_rank,
    BLOCK_Q: tl.constexpr,
    BLOCK_Q_POW2: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    row = tl.load(rows_ptr + b).to(tl.int64)
    ctx = tl.load(ctx_len_ptr + b).to(tl.int64)
    src = table_ptr + row * table_stride
    # This rank's pages: local page k = global page k * cp + rank (0 past the table).
    for k0 in range(0, local_width, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        g = k.to(tl.int64) * cp_size + cp_rank
        val = tl.load(src + g, mask=(k < local_width) & (g < table_width), other=0)
        tl.store(local_ptr + b.to(tl.int64) * local_width + k, val, mask=k < local_width)
    # The block's own positions, in the rank-local layout (HelixDraftShard.local_slots).
    j = tl.arange(0, BLOCK_Q_POW2)
    jm = j < BLOCK_Q
    pos = ctx + j
    page = pos // page_size
    local_page = page // cp_size
    g = local_page * cp_size + cp_rank
    phys = tl.load(src + g, mask=jm & (g < table_width), other=0).to(tl.int64)
    tl.store(pages_ptr + b * BLOCK_Q + j, phys, mask=jm)
    tl.store(offsets_ptr + b * BLOCK_Q + j, pos % page_size, mask=jm)
    tl.store(owned_ptr + b * BLOCK_Q + j, (page % cp_size == cp_rank).to(tl.uint8), mask=jm)
    # Owned positions below ctx + block (HelixDraftShard.local_lengths), per request and per query token.
    seq = ctx + BLOCK_Q
    span = page_size * cp_size
    rem = seq % span - cp_rank * page_size
    local_len = (seq // span) * page_size + tl.minimum(tl.maximum(rem, 0), page_size)
    tl.store(seq_lens_ptr + b, local_len.to(tl.int32))
    tl.store(kv_bounds_ptr + b * BLOCK_Q + j, local_len.to(tl.int32) + 0 * j, mask=jm)


def ctx_split_block_fixup(
    ctx_len: torch.Tensor,
    table: torch.Tensor,
    rows: torch.Tensor,
    block_size: int,
    page_size: int,
    cp_size: int,
    cp_rank: int,
):
    """The context split's per-step decode operands in one launch.

    What ``ctx_split_local_page_table(table.index_select(0, rows), cp, rank)`` followed by
    ``modeling_dspark._build_mla_block_fixup(ctx_len, local.long(), block, page, HelixDraftShard(cp, rank,
    page))`` produce, bit for bit: the int32 local table ``[B, max(2, ceil(pages / cp))]``, the int64 page and
    offset of each block position in it ``[B, block]``, whether this rank owns the position (bool
    ``[B, block, 1]``), and the int32 local lengths (per request and per query token).

    Args:
        ctx_len: ``[B]`` context lengths before the block.
        table: ``[rows, pages]`` int32 block tables of the whole context (unit column stride).
        rows: ``[B]`` the table row of each request.
    """
    batch = ctx_len.shape[0]
    pages_w = table.shape[1]
    local_width = max(2, -(-pages_w // cp_size))
    device = ctx_len.device
    assert table.dtype == torch.int32 and table.stride(1) == 1
    local = torch.empty((batch, local_width), dtype=torch.int32, device=device)
    pages = torch.empty((batch, block_size), dtype=torch.int64, device=device)
    offsets = torch.empty((batch, block_size), dtype=torch.int64, device=device)
    owned = torch.empty((batch, block_size, 1), dtype=torch.bool, device=device)
    seq_lens = torch.empty((batch,), dtype=torch.int32, device=device)
    kv_bounds = torch.empty((batch * block_size,), dtype=torch.int32, device=device)
    if batch:
        _ctx_split_fixup_kernel[(batch,)](
            ctx_len,
            rows,
            table,
            table.stride(0),
            pages_w,
            local,
            local_width,
            pages,
            offsets,
            owned.view(torch.uint8),
            seq_lens,
            kv_bounds,
            page_size,
            cp_size,
            cp_rank,
            BLOCK_Q=block_size,
            BLOCK_Q_POW2=triton.next_power_of_2(block_size),
            BLOCK_K=1024,
            num_warps=4,
        )
    return local, pages, offsets, owned, seq_lens, kv_bounds
