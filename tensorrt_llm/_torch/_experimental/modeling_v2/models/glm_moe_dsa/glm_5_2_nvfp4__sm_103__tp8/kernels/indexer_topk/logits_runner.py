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
"""Compile cache and launcher of the indexer_topk all-rows FP8 paged-MQA logits kernels.

Mirrors the native ``CuteDSLPagedMQALogitsRunner`` (fake-tensor compile with
TVM FFI, persistent graph-stable logits arena) for the two indexer_topk kernels, with
fp32 accumulation, epilogue and output only (the native runner's
configuration for this model):

- ``rowsplit`` (:mod:`fp8_paged_mqa_logits_rowsplit`): one KV stream, both
  math warpgroups split the verify rows of each tile, two-stage TMEM ring.
  Needs an even ``next_n`` and ``2 * next_n * H <= 512``.
- ``rows`` (:mod:`fp8_paged_mqa_logits_rows`): the native two-group kernel
  at the full verify width with a bounded register weight cache.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

from .fp8_paged_mqa_logits_rows import FP8MQALogitsKernel
from .fp8_paged_mqa_logits_rowsplit import FP8MQALogitsRowSplitKernel

COMPUTE_BLOCK_KV = 128
# Two math warpgroups per CTA each take one 128-token tile per step.
SPLIT_KV = 2 * COMPUTE_BLOCK_KV
LOGITS_BUFFER_NAME = "indexer_topk_mqa_logits"


@dataclass(frozen=True)
class LogitsConfig:
    """Compile-time knobs of the indexer_topk logits kernel.

    Attributes:
        variant: ``"rowsplit"`` (default; falls back to ``"rows"`` when the
            shape does not allow it) or ``"rows"``.
        num_kv_stages: KV ring depth of the ``rowsplit`` kernel.
        num_q_stages: Query/weight ring depth of the ``rowsplit`` kernel.
        num_math_wg: Math warpgroups of the ``rowsplit`` kernel (2 or 3).
        debug: ``rowsplit`` diagnosis mode (non-zero values break results).
        num_w_in_reg: ``rows`` kernel: register-cached epilogue weights per
            verify row; None keeps the upstream heuristic.
        num_epi_subtiles: TMEM-load sub-tiles per verify row (1, 2, 4 or 8
            for 32 heads).
        max_kv_pipeline: Fill shared memory with KV stages.
        smem_subpartition_opt: Pad shared memory so weights/scales sit in the
            second sub-partition.
        early_tmem_copy: Issue the first TMEM load before the KV release.
    """

    variant: str = "rowsplit"
    num_kv_stages: int = 6
    num_q_stages: int = 2
    num_math_wg: int = 2
    debug: int = 0
    num_w_in_reg: Optional[int] = 8
    num_epi_subtiles: int = 1
    max_kv_pipeline: bool = False
    smem_subpartition_opt: bool = False
    early_tmem_copy: bool = False


# "rows" with 8 register-cached weights is the default variant; "rowsplit"
# is the alternative for shorter indexer contexts.
DEFAULT_CONFIG = LogitsConfig(variant="rows", num_w_in_reg=8)

_KERNEL_CACHE: Dict[Tuple, object] = {}
_NUM_SMS: Dict[int, int] = {}


def _num_sms(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _NUM_SMS:
        _NUM_SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _NUM_SMS[index]


def rowsplit_supported(next_n: int, num_heads: int, num_math_wg: int = 2) -> bool:
    """Whether the ``rowsplit`` kernel can hold this verify width."""
    n_cols = next_n * num_heads
    return next_n % num_math_wg == 0 and n_cols % 16 == 0 and n_cols <= 256


def _compile(
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    num_sms: int,
    config: LogitsConfig,
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
    if config.variant == "rowsplit" and rowsplit_supported(next_n, num_heads, config.num_math_wg):
        kernel = FP8MQALogitsRowSplitKernel(
            block_kv=COMPUTE_BLOCK_KV,
            phys_block_kv=phys_block_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            next_n=next_n,
            num_sms=num_sms,
            num_kv_stages=config.num_kv_stages,
            num_q_stages=config.num_q_stages,
            num_math_wg=config.num_math_wg,
            debug=config.debug,
        )
    else:
        kernel = FP8MQALogitsKernel(
            block_kv=COMPUTE_BLOCK_KV,
            phys_block_kv=phys_block_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            next_n=next_n,
            num_sms=num_sms,
            early_tmem_copy=config.early_tmem_copy,
            smem_subpartition_opt=config.smem_subpartition_opt,
            max_kv_pipeline=config.max_kv_pipeline,
            num_epi_subtiles=config.num_epi_subtiles,
            num_w_in_reg=config.num_w_in_reg,
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
        fake_stream,
        options="--enable-tvm-ffi",
    )


def paged_mqa_logits_rows(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    config: LogitsConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Score every verify row of every request in one launch.

    Args:
        q: ``[B, next_n, H, D]`` float8_e4m3fn queries, ``next_n * H <= 256``.
        kv_fused: ``[num_blocks, phys_block_kv, 1, D + 4]`` uint8 page view
            (FP8 values then fp32 scales per page); dim-0 may be strided.
        weights: ``[B * next_n, H]`` float32 epilogue weights.
        context_lens: ``[B]`` int32 KV lengths.
        block_table: ``[B, max_blocks]`` int32 page table.
        schedule_meta: ``[num_sms + 1, 2]`` int32 DeepGEMM schedule built from
            ``context_lens.view(-1, 1)`` with 64-token schedule blocks.
        max_context_len: Capture-stable logits width.
        config: Compile-time kernel knobs.

    Returns:
        ``[B * next_n, max_context_len]`` float32 view of a 256-column-aligned,
        graph-stable arena; columns at or beyond a row's KV length are
        unspecified.
    """
    batch, next_n, num_heads, head_dim = q.shape
    phys_block_kv = kv_fused.shape[1]
    num_phys_blocks = kv_fused.shape[0]
    num_sms = _num_sms(q.device)
    aligned = (max_context_len + SPLIT_KV - 1) // SPLIT_KV * SPLIT_KV
    logits = get_memory_buffers().get_buffer(
        [batch * next_n, aligned],
        torch.float32,
        buffer_name=LOGITS_BUFFER_NAME,
        reserve_buffer=torch.cuda.is_current_stream_capturing(),
    )[:, :max_context_len]
    key = (phys_block_kv, num_heads, head_dim, next_n, num_sms, config)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        compiled = _compile(phys_block_kv, num_heads, head_dim, next_n, num_sms, config)
        _KERNEL_CACHE[key] = compiled
    q_3d = q.reshape(batch, next_n * num_heads, head_dim).permute(1, 2, 0)
    w_2d = weights.reshape(batch, next_n * num_heads).t()
    compiled(
        kv_fused.reshape(num_phys_blocks, -1),
        q_3d.view(torch.uint8),
        w_2d,
        logits,
        block_table,
        context_lens,
        schedule_meta,
        num_phys_blocks,
        batch,
    )
    return logits
