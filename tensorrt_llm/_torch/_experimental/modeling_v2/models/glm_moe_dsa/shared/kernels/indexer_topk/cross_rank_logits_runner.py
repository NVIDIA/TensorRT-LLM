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
"""Compile cache and launcher of the cross_rank_topk context-partitioned logits kernel.

The kernel (``fp8_paged_mqa_logits_cross_rank``) scores the 64-token pages of one
rank of the tensor-parallel group for every verify row of one request, with
the indexer_topk ``rows`` configuration (8 register-cached weights per row, fp32
accumulation, epilogue and output).
"""

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

from .fp8_paged_mqa_logits_cross_rank import FP8MQALogitsCPKernel

COMPUTE_BLOCK_KV = 128
# Two math warpgroups per CTA each take one 128-token tile per step.
SPLIT_KV = 2 * COMPUTE_BLOCK_KV
PAGE_TOKENS = 64
LOGITS_BUFFER_NAME = "cross_rank_topk_logits"
# indexer_topk default ("rows" kernel, 8 register-cached weights per verify row).
NUM_W_IN_REG = 8
# The kernel lets the dependent select/exchange op launch early (programmatic dependent launch).
TRIGGER_DEPENDENTS = True

_KERNEL_CACHE: Dict[Tuple, object] = {}
_NUM_SMS: Dict[int, int] = {}
# The kernel derives its schedule in-kernel; this placeholder fills the
# schedule argument of the shared launcher signature and is never read.
_UNUSED_SCHEDULE: Dict[int, torch.Tensor] = {}


def _num_sms(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _NUM_SMS:
        _NUM_SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _NUM_SMS[index]


def _unused_schedule(device: torch.device) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    t = _UNUSED_SCHEDULE.get(index)
    if t is None:
        t = torch.zeros(1, 2, dtype=torch.int32, device=device)
        _UNUSED_SCHEDULE[index] = t
    return t


def local_length(n: int, rank: int, world: int) -> int:
    """Tokens of ``rank`` among the global tokens ``[0, n)`` (page-interleaved)."""
    if n <= 0:
        return 0
    full_pages, tail = divmod(n, PAGE_TOKENS)
    length = (full_pages - rank + world - 1) // world * PAGE_TOKENS
    if tail and full_pages % world == rank:
        length += tail
    return length


def local_capacity(max_seq_len: int, world: int) -> int:
    """Local logits columns that cover any rank's share of ``max_seq_len`` tokens."""
    return max(local_length(max_seq_len, r, world) for r in range(world))


def local_to_global(local_cols: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Global token index of local logits columns of ``rank``."""
    return (local_cols // PAGE_TOKENS * world + rank) * PAGE_TOKENS + local_cols % PAGE_TOKENS


def _compile(
    phys_block_kv: int, num_heads: int, head_dim: int, next_n: int, num_sms: int, world: int
):
    n_cols = next_n * num_heads
    block_bytes = phys_block_kv * (head_dim + 4)
    sym_num_phys_blocks = cute.sym_int()
    sym_b = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks_per_seq = cute.sym_int()
    num_ctas = cute.sym_int()
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (sym_num_phys_blocks, block_bytes), stride=(cute.sym_int64(), 1)
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n_cols, head_dim, sym_b), stride_order=(1, 0, 2)
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (n_cols, sym_b), stride_order=(0, 1)
    )
    logits_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32, (cute.sym_int(), max_ctx), stride=(cute.sym_int64(), 1)
    )
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_b, max_blocks_per_seq), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_b,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = FP8MQALogitsCPKernel(
        block_kv=COMPUTE_BLOCK_KV,
        phys_block_kv=phys_block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_sms=num_sms,
        num_w_in_reg=NUM_W_IN_REG,
        world=world,
        trigger_dependents=TRIGGER_DEPENDENTS,
    )
    return cute.compile(
        kernel,
        kv_fake,
        q_fake,
        w_fake,
        logits_fake,
        bt_fake,
        cl_fake,
        sm_fake,
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(0),
        fake_stream,
        options="--enable-tvm-ffi",
    )


def ensure_compiled(
    device: torch.device, phys_block_kv: int, num_heads: int, head_dim: int, next_n: int, world: int
) -> None:
    """Compile the kernel for this shape now (host only, no device work).

    cross_rank_topk calls this before the group agrees to create its workspace, so the
    per-rank compile time is absorbed by that collective instead of making
    the first exchange wait for the slowest compiling rank.
    """
    key = (phys_block_kv, num_heads, head_dim, next_n, _num_sms(device), world)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _compile(*key)


def paged_mqa_logits_cp(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    kv_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_len: int,
    rank: int,
    world: int,
) -> torch.Tensor:
    """Score this rank's pages of one request for every verify row.

    Args:
        q: ``[1, next_n, H, D]`` float8_e4m3fn queries.
        kv_fused: ``[num_blocks, 64, 1, D + 4]`` uint8 indexer K page view.
        weights: ``[next_n, H]`` float32 epilogue weights.
        kv_lens: ``[1]`` int32 global indexer KV length.
        block_table: ``[1, max_blocks]`` int32 page table of the request.
        max_seq_len: Capture-stable indexer max sequence length.
        rank: This rank's index in the context-parallel group.
        world: Ranks in the group.

    Returns:
        ``[next_n, local_capacity(max_seq_len, world)]`` float32 view of a
        graph-stable arena; local column ``c`` holds global token
        ``local_to_global(c)``. Columns at or beyond a row's local length are
        unspecified.
    """
    batch, next_n, num_heads, head_dim = q.shape
    assert batch == 1
    phys_block_kv = kv_fused.shape[1]
    num_phys_blocks = kv_fused.shape[0]
    num_sms = _num_sms(q.device)
    width = local_capacity(max_seq_len, world)
    aligned = (width + SPLIT_KV - 1) // SPLIT_KV * SPLIT_KV
    logits = get_memory_buffers().get_buffer(
        [next_n, aligned],
        torch.float32,
        buffer_name=LOGITS_BUFFER_NAME,
        reserve_buffer=torch.cuda.is_current_stream_capturing(),
    )[:, :width]
    key = (phys_block_kv, num_heads, head_dim, next_n, num_sms, world)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        compiled = _compile(*key)
        _KERNEL_CACHE[key] = compiled
    q_3d = q.reshape(batch, next_n * num_heads, head_dim).permute(1, 2, 0)
    w_2d = weights.reshape(batch, next_n * num_heads).t()
    compiled(
        kv_fused.reshape(num_phys_blocks, -1),
        q_3d.view(torch.uint8),
        w_2d,
        logits,
        block_table,
        kv_lens,
        _unused_schedule(q.device),
        num_phys_blocks,
        batch,
        rank,
    )
    return logits
