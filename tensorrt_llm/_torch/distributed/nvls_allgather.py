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
"""Multicast all-gather of vocab-sharded BF16 logits into FP32 ``[rows, world * shard]``.

The gathered logits of a vocab-parallel LM head (``allgather(local, dim=-1).float()``) as one
kernel on the multicast workspace of :class:`NVLSAllReduce`: program ``(r, q)`` of every rank

1. arrives on its flag (one ``multimem.red`` that adds 1 on every rank) and waits until every
   rank has arrived, so every rank has finished the previous call, re-arming included;
2. stores the ``q``-th of ``world`` slices of its own shard of row ``r`` into the output buffer
   of every rank (``multimem.st``), at the shard's columns;
3. polls the ``q``-th slices of every rank's shard of row ``r`` in its local output buffer (what
   program ``(r, q)`` of every rank stored) until no BF16 lane holds the sentinel ``0xFFFF``,
   writes them widened to FP32 and re-arms them.

A program only waits for the same program of the other ranks, so the grid needs no
co-residency.

Each rank's shard reaches every rank once and the widening is exact, so the output equals the
NCCL all-gather + ``.float()`` bit for bit; the one exception is a logit whose BF16 bits are the
sentinel (a NaN), stored as the NaN ``0x7FFF``. The call counter per program lives on the device,
so the kernel replays inside CUDA graphs.
"""

import os
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm.mapping import Mapping

from .nvls_allreduce import NVLSAllReduce, _arrive, _has_sentinel, _no_sentinel, _wait

# Rows a workspace serves (rows x world x shard BF16 values); larger calls keep the NCCL path.
_DEFAULT_MAX_ROWS = int(os.environ.get("TRTLLM_VOCAB_GATHER_NVLS_MAX_ROWS", "128"))
# Smallest TP group served. GB300, 163840-entry vocab, us per gather (NCCL dim-0 all-gather + one
# widening kernel -> this kernel): 16 ranks 31.6 -> 11.3 (8 rows), 82.5 -> 39.8 (48), 118.8 -> 89.1
# (128); 8 ranks 22.3 -> 18.0 (8 rows), a tie at 48 and 96; 4 ranks slower below 16 rows.
_MIN_WORLD_SIZE = int(os.environ.get("TRTLLM_VOCAB_GATHER_NVLS_MIN_WORLD_SIZE", "16"))


@triton.jit
def _multimem_st_b32(addrs, value, mask):
    """Store one 32-bit word to the address on every rank of the multicast group."""
    return tl.inline_asm_elementwise(
        """
        {
        .reg .pred %p_st;
        setp.ne.s32 %p_st, $3, 0;
        mov.u32 $0, 0;
        @%p_st multimem.st.relaxed.sys.global.b32 [$1], $2;
        }
        """,
        "=r,l,r,r,~{memory}",
        args=[addrs, value, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["rank", "num_rows"])
def _nvls_vocab_gather_kernel(
    inp_ptr,
    out_ptr,
    out_buf_ptr,
    out_mc_addr,
    flag_uc_addr,
    flag_mc_addr,
    epoch_ptr,
    rank,
    num_rows,
    SHARD: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SLICE_BLOCK: tl.constexpr,
    FLAG_SPACING: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    q = tl.program_id(1)
    pid = row * WORLD_SIZE + q
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    epoch = tl.load(epoch_ptr + pid).to(tl.uint32)

    # 1. Every rank has finished the previous call of this program.
    flag_off = pid.to(tl.int64) * (FLAG_SPACING * 4)
    _arrive(flag_mc_addr + flag_off, False)
    _wait(flag_uc_addr + flag_off, (epoch + 1) * WORLD_SIZE)
    tl.debug_barrier()

    # 2. Multicast the q-th slice of this rank's shard of the row (32-bit words).
    SHARD_W: tl.constexpr = SHARD // 2
    SLICE_W: tl.constexpr = SHARD_W // WORLD_SIZE
    w = tl.arange(0, SLICE_BLOCK)
    wm = w < SLICE_W
    src = row.to(tl.int64) * SHARD_W + q * SLICE_W + w
    x = tl.load(inp_ptr + src, mask=wm, other=0).to(tl.uint32, bitcast=True)
    dst_w = (row.to(tl.int64) * WORLD_SIZE + rank) * SHARD_W + q * SLICE_W + w
    _multimem_st_b32(out_mc_addr + dst_w * 4, _no_sentinel(x), wm)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # 3. The q-th slices of every rank's shard of the row in the local buffer: wait, widen to
    #    FP32, re-arm.
    k = tl.arange(0, WORLD_SIZE)
    loc = (row.to(tl.int64) * WORLD_SIZE + k)[:, None] * SHARD_W + q * SLICE_W + w[None, :]
    lm = (k[:, None] < WORLD_SIZE) & wm[None, :]
    y = tl.load(out_buf_ptr + loc, mask=lm, other=0, volatile=True)
    pending = tl.max(tl.max((_has_sentinel(y) & lm).to(tl.int32), axis=1), axis=0)
    while pending != 0:
        y = tl.load(out_buf_ptr + loc, mask=lm, other=0, volatile=True)
        pending = tl.max(tl.max((_has_sentinel(y) & lm).to(tl.int32), axis=1), axis=0)
    lo = (y << 16).to(tl.float32, bitcast=True)
    hi = (y & -65536).to(tl.float32, bitcast=True)
    out_off = row.to(tl.int64) * (WORLD_SIZE * SHARD) + (
        k[:, None] * SHARD + q * (2 * SLICE_W) + 2 * w[None, :]
    )
    tl.store(out_ptr + out_off, lo, mask=lm)
    tl.store(out_ptr + out_off + 1, hi, mask=lm)
    tl.store(out_buf_ptr + loc, tl.full([WORLD_SIZE, SLICE_BLOCK], -1, tl.int32), mask=lm)
    tl.store(epoch_ptr + pid, (epoch + 1).to(tl.int32))


class NVLSVocabGather:
    """``allgather(local, dim=-1).float()`` of ``[rows, shard]`` BF16 logits over the TP group
    on a multicast workspace (one per mapping)."""

    _instances: Dict[Tuple[Mapping, int], "NVLSVocabGather"] = {}

    def __init__(self, mapping: Mapping, shard: int, max_rows: int = _DEFAULT_MAX_ROWS):
        from ..flashinfer_utils import get_env_enable_pdl

        self.mapping = mapping
        self.world_size = mapping.tp_size
        self.rank = mapping.tp_rank
        self.shard = shard
        self.max_rows = max_rows
        self.use_pdl = get_env_enable_pdl()
        nbytes = max_rows * self.world_size * shard * 2
        self._ws = NVLSAllReduce._get_workspace(mapping, f"vocab_gather_{shard}", nbytes)

    @classmethod
    def get(cls, mapping: Mapping, shard: int) -> Optional["NVLSVocabGather"]:
        """The gather of ``mapping``'s TP group for ``shard``-column shards, or None where the
        multicast workspace does not apply (created outside CUDA graph capture)."""
        key = (mapping, shard)
        inst = cls._instances.get(key)
        if inst is None:
            if (
                torch.cuda.is_current_stream_capturing()
                or mapping.tp_size < _MIN_WORLD_SIZE
                or not NVLSAllReduce.is_supported(mapping)
                or mapping.tp_size & (mapping.tp_size - 1)
                or shard % (2 * mapping.tp_size)
            ):
                return None
            inst = cls(mapping, shard)
            cls._instances[key] = inst
        return inst

    def can_gather(self, local: torch.Tensor) -> bool:
        return (
            local.dim() == 2
            and local.dtype == torch.bfloat16
            and local.is_cuda
            and local.is_contiguous()
            and local.shape[1] == self.shard
            and 0 < local.shape[0] <= self.max_rows
        )

    def __call__(self, local: torch.Tensor) -> torch.Tensor:
        rows = local.shape[0]
        out = torch.empty(
            (rows, self.world_size * self.shard), dtype=torch.float32, device=local.device
        )
        ws = self._ws
        shard_w = self.shard // 2
        _nvls_vocab_gather_kernel[(rows, self.world_size)](
            local.view(torch.int32),
            out,
            ws["out_buf"].view(torch.int32),
            ws["out_mc_addr"],
            ws["flag_uc_addr"],
            ws["flag_mc_addr"],
            ws["epoch"],
            self.rank,
            rows,
            SHARD=self.shard,
            WORLD_SIZE=self.world_size,
            SLICE_BLOCK=triton.next_power_of_2(shard_w // self.world_size),
            FLAG_SPACING=NVLSAllReduce.FLAG_SPACING,
            USE_PDL=self.use_pdl,
            num_warps=8,
            launch_pdl=self.use_pdl,
        )
        return out
