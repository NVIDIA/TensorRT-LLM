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
"""TP exchange of the MTP draft argmax pairs.

Under plain tensor parallelism the one-model MTP draft LM head is vocab-sharded, so every
draft step picks the global argmax across the TP group from one (global argmax index, max
value) fp32 pair per rank: ``SpecWorkerBase._get_draft_tokens_from_gathered`` takes the argmax
of the gathered values. This module owns both producers of the gathered ``[rows, 2 * tp_size]``
rank-major pairs:

* ``gather_argmax_pairs_allgather``: the original one, ``torch.max`` pairs + an NCCL allgather.
* ``DraftArgmaxExchange``: ``local_argmax_pack`` (one Triton launch) writes each row's pair into
  this rank's two slots of a ``[rows, ROW_WIDTH]`` fp32 row, zeros elsewhere, and the generic
  ``AllReduce`` module SUMs the rows. Every slot has exactly one non-zero contributor, so the
  SUM equals the allgather exactly under any strategy (x + 0 == x in fp32); the strategy
  follows the model's ``allreduce_strategy``. Same tie-break (lowest index) and NaN rule (NaN
  wins, lowest NaN index) as ``torch.max``, bf16 widened bit-exactly in-kernel, so the packed
  pair is bit-identical to the original producer's. Two value-vs-bit corners of the SUM: a NaN
  max comes back canonicalised and a -0.0 max as +0.0; the consumer compares values, so the
  tokens are unaffected. Rows above ``max_rows`` take the original producer.

Lamport one-shot allreduce kernels clear the previous launch's payload with the current
launch's threads, so a tiny exchange sharing the model's workspace pays for clearing the model
payload; the tagged workspace isolates it. Today only the MNNVL registry is tagged; the custom
one-shot workspace is a follow-up.

Gate: ``TRTLLM_MTP_DRAFT_ARGMAX_ALLREDUCE`` (unset / "1" = on, "0" = original producer), read
once and cached so the choice is fixed before any CUDA-graph capture.
"""

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..distributed.ops import AllReduce, MNNVLAllReduce, allgather

DRAFT_ARGMAX_ALLREDUCE_ENV = "TRTLLM_MTP_DRAFT_ARGMAX_ALLREDUCE"

# fp32 slots per exchange row: one float4 per lane of a warp. Invariant:
# 2 * tp_size <= ROW_WIDTH for the widest group the kernels dispatch (64 ranks).
ROW_WIDTH = 128

# Tag of the exchange's own MNNVL workspace (see the module docstring).
WORKSPACE_TAG = "draft_argmax"


@functools.lru_cache(maxsize=1)
def draft_argmax_exchange_enabled() -> bool:
    """Process-constant gate: unset / "1" = DraftArgmaxExchange, "0" = original producer."""
    enabled = os.environ.get(DRAFT_ARGMAX_ALLREDUCE_ENV, "1").strip() != "0"
    logger.info(
        f"MTP draft argmax exchange through AllReduce {'ON' if enabled else 'OFF'} "
        f"({DRAFT_ARGMAX_ALLREDUCE_ENV}={os.environ.get(DRAFT_ARGMAX_ALLREDUCE_ENV)})"
    )
    return enabled


def local_argmax_pairs(logits: torch.Tensor, tp_rank: int) -> torch.Tensor:
    """``[rows, 2]`` fp32 ``[global argmax index, max value]`` of each row of this rank's
    ``[rows, vocab_shard]`` logits: ``torch.max`` plus the rank's vocab offset."""
    local_max_values, local_argmax = torch.max(logits, dim=-1, keepdim=True)
    max_index = local_argmax.type(torch.int32) + tp_rank * logits.shape[-1]
    return torch.stack([max_index.float(), local_max_values.float()], dim=-1).flatten(-2)


def gather_argmax_pairs_allgather(logits: torch.Tensor, mapping: Mapping) -> torch.Tensor:
    """The original producer: ``[rows, 2 * tp_size]`` rank-major pairs via an NCCL allgather."""
    return allgather(local_argmax_pairs(logits, mapping.tp_rank), mapping, dim=-1)


# int32 max: "no valid column seen yet" in the per-lane running argmax.
# Instantiated (not annotated) so @triton.jit kernels may read it.
_SENTINEL_INDEX = tl.constexpr(2147483647)


@triton.jit
def _argmax_first_nan_combine(v1, i1, v2, i2):
    # Total order of at::native GreaterOrNan: NaN beats everything, among
    # NaNs or equal values the lower index wins, otherwise the larger value.
    n1 = v1 != v1
    n2 = v2 != v2
    lower = i1 < i2
    take1 = (n1 & ((~n2) | lower)) | ((~n1) & (~n2) & ((v1 > v2) | ((v1 == v2) & lower)))
    return tl.where(take1, v1, v2), tl.where(take1, i1, i2)


@triton.jit(do_not_specialize=["n_cols", "stride_out", "idx_offset", "slot", "out_width"])
def _draft_local_argmax_pack_kernel(
    logits_ptr,
    out_ptr,
    n_cols,
    stride_row,
    stride_out,
    idx_offset,
    slot,
    out_width,
    BLOCK: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    BF16_BITS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = logits_ptr + row.to(tl.int64) * stride_row
    best_v = tl.full([BLOCK], float("-inf"), tl.float32)
    best_i = tl.full([BLOCK], _SENTINEL_INDEX, tl.int32)
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        m = cols < n_cols
        if BF16_BITS:
            # Widen by moving the bf16 bits into the fp32 high half, exactly
            # like at::BFloat16 -> float: keeps NaN payloads, which a cvt
            # would canonicalize.
            raw = tl.load(base + cols, mask=m, other=float("-inf"))
            bits = raw.to(tl.int16, bitcast=True).to(tl.int32) << 16
            v = bits.to(tl.float32, bitcast=True)
        else:
            v = tl.load(base + cols, mask=m, other=float("-inf")).to(tl.float32)
        # Chunks are visited in increasing column order, so within a lane
        # the new element always has the larger index: replace only on a
        # strictly better value (or NaN over non-NaN), or the first valid.
        nv = v != v
        nb = best_v != best_v
        take = m & ((best_i == _SENTINEL_INDEX) | (nv & (~nb)) | ((~nv) & (~nb) & (v > best_v)))
        best_v = tl.where(take, v, best_v)
        best_i = tl.where(take, cols, best_i)
    vmax, imax = tl.reduce((best_v, best_i), 0, _argmax_first_nan_combine)
    if BF16_BITS:
        # torch's bf16 max reduces in fp32 and rounds back through
        # c10::BFloat16, which maps every NaN to 0x7FFF: match its bits.
        vmax = tl.where(
            vmax != vmax, tl.full([], 0x7FFF0000, tl.int32).to(tl.float32, bitcast=True), vmax
        )
    idx_f = (imax + idx_offset).to(tl.float32)
    ocols = tl.arange(0, OUT_BLOCK)
    zeros = tl.zeros([OUT_BLOCK], tl.float32)
    packed = tl.where(ocols == slot, idx_f, tl.where(ocols == slot + 1, vmax, zeros))
    tl.store(out_ptr + row.to(tl.int64) * stride_out + ocols, packed, mask=ocols < out_width)


def local_argmax_pack(
    logits: torch.Tensor, idx_offset: int, out: torch.Tensor, slot: int = 0
) -> torch.Tensor:
    """Write (float(argmax + idx_offset), float(max)) of each logits row into
    ``out[row, slot:slot + 2]`` and zeros into the rest of ``out[row]``.

    Bit-identical to ``local_argmax_pairs`` placed at ``slot``. ``logits`` is 2D
    ``[rows, vocab_shard]`` with unit column stride; ``out`` is a contiguous fp32
    ``[rows, width]``. An empty ``logits`` is a no-op.
    """
    assert logits.dim() == 2 and logits.stride(-1) == 1
    assert out.dtype == torch.float32 and out.is_contiguous()
    rows, n_cols = logits.shape
    width = out.shape[1]
    assert out.shape[0] == rows and slot + 2 <= width
    block = min(triton.next_power_of_2(n_cols), 8192)
    num_warps = 16 if block >= 8192 else (8 if block >= 2048 else 4)
    _draft_local_argmax_pack_kernel[(rows,)](
        logits,
        out,
        n_cols,
        logits.stride(0),
        out.stride(0),
        idx_offset,
        slot,
        width,
        BLOCK=block,
        OUT_BLOCK=triton.next_power_of_2(width),
        BF16_BITS=logits.dtype == torch.bfloat16,
        num_warps=num_warps,
    )
    return out


class DraftArgmaxExchange(nn.Module):
    """Gathered ``[rows, 2 * tp_size]`` fp32 rank-major (index, value) pairs of one TP group.

    ``forward(logits)`` packs + SUM-allreduces up to ``max_rows`` rows and otherwise runs
    ``gather_argmax_pairs_allgather``; both feed ``_get_draft_tokens_from_gathered``. Build it
    at worker construction: the all-reduce workspaces must exist before any CUDA-graph capture,
    and the MNNVL workspace split is collective over the group. Only the staging buffer is
    allocated lazily (see ``__init__``): construction may run under ``MetaInitMode``.
    """

    def __init__(
        self, mapping: Mapping, strategy: AllReduceStrategy, max_rows: Optional[int] = None
    ):
        super().__init__()
        tp_size = mapping.tp_size
        assert 2 * tp_size <= ROW_WIDTH, f"tp_size {tp_size} exceeds {ROW_WIDTH // 2} pairs per row"
        self.mapping = mapping
        # Default: the most rows whose payload (rows x ROW_WIDTH fp32 from every
        # rank) the MNNVL heuristic still routes to the one-shot kernel.
        self.max_rows = (
            max_rows
            if max_rows is not None
            else MNNVLAllReduce.max_one_shot_tokens(ROW_WIDTH, tp_size, torch.float32)
        )
        self._slot = 2 * mapping.tp_rank
        self._idx_offset_factor = mapping.tp_rank  # idx_offset = tp_rank * vocab_per_rank
        self._out_cols = 2 * tp_size
        self.allreduce = AllReduce(
            mapping,
            strategy=strategy,
            dtype=torch.float32,
            workspace_tag=WORKSPACE_TAG,
            # Lamport buffer for the largest exchange forward runs on this path.
            mnnvl_initial_workspace_bytes=MNNVLAllReduce.get_required_workspace_size(
                self.max_rows, ROW_WIDTH, tp_size, torch.float32
            ),
        )
        # Pack-kernel target, fully overwritten on every exchange. Allocated on
        # the first forward, not here: the worker is built inside the model
        # constructor, which runs under MetaInitMode, and that mode places
        # every torch.empty on the meta device whatever device it names. The
        # model loader re-materialises only registered parameters and buffers,
        # so an eager allocation here would reach the pack kernel as a meta
        # tensor (data_ptr 0). The first forward is the eager warm-up, before
        # any CUDA-graph capture.
        self._staging: Optional[torch.Tensor] = None

    def _staging_rows(self, rows: int) -> torch.Tensor:
        if self._staging is None:
            self._staging = torch.empty(
                (self.max_rows, ROW_WIDTH), dtype=torch.float32, device="cuda"
            )
        return self._staging[:rows]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        rows = logits.shape[0]
        if not 0 < rows <= self.max_rows:
            return gather_argmax_pairs_allgather(logits, self.mapping)
        staging = self._staging_rows(rows)
        local_argmax_pack(logits, self._idx_offset_factor * logits.shape[1], staging, self._slot)
        return self.allreduce(staging)[:, : self._out_cols]
