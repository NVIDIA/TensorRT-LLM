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
"""MTP draft-step glue trim (``TRTLLM_MTP_TAIL_TRIM``, on unless set to "0").

Numerics-preserving host/launch-level trims of the one-model MTP draft loop:

* ``local_argmax_pack``: one Triton launch replaces ``torch.max(dim=-1)`` (a
  single-CTA ``reduce_kernel`` over each rank's vocab shard) plus the int32
  cast / rank offset / two float casts / stack that pack each rank's (global
  argmax index, max value) pair, and writes the pair straight into the
  exchange buffer (the MNNVL mailbox row or the NCCL ``[rows, 2]`` input).
  Same tie-break (lowest index) and NaN rule (NaN wins, lowest NaN index) as
  ``at::native`` ``GreaterOrNan``; bf16 -> fp32 is exact, so the packed values
  are bit-identical.
* ``DraftArgmaxMailbox``: exchanges the packed pairs through a one-shot MNNVL
  SUM allreduce over a DEDICATED Lamport workspace instead of the NCCL
  allgather. Each rank writes its pair into its own slot of a zero row, so the
  fp32 sum is exact. The workspace is private because the one-shot kernel
  clears the dirty Lamport buffer left by the PREVIOUS call on the same
  workspace with the CURRENT launch's threads: a tiny payload must never
  follow a large one on a shared workspace, or its few threads end up
  clearing the model allreduce's payload.
"""

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger

from ..distributed.ops import (
    _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
    MNNVLAllReduce,
    _build_mnnvl_workspace,
    _get_mnnvl_workspace_comm,
    _launch_mnnvl_allreduce,
    allgather,
)

MTP_TAIL_TRIM_ENV = "TRTLLM_MTP_TAIL_TRIM"


@functools.lru_cache(maxsize=1)
def mtp_tail_trim_enabled() -> bool:
    """Process-constant gate: unset / "1" = trimmed path, "0" = original path.

    Cached so the choice is fixed before CUDA-graph capture.
    """
    enabled = os.environ.get(MTP_TAIL_TRIM_ENV, "1").strip() != "0"
    logger.info(
        f"MTP tail trim {'ON' if enabled else 'OFF'} "
        f"({MTP_TAIL_TRIM_ENV}={os.environ.get(MTP_TAIL_TRIM_ENV)})"
    )
    return enabled


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

    Bit-identical to ``SpecWorkerBase._get_local_max_and_combined`` (which is
    ``torch.max(logits, -1)`` + int32 offset + float casts + stack) placed at
    ``slot``. ``logits`` is 2D ``[rows, vocab_shard]`` with unit column stride;
    ``out`` is a contiguous fp32 ``[rows, width]``.
    """
    assert logits.dim() == 2 and logits.stride(-1) == 1
    assert out.dtype == torch.float32 and out.is_contiguous()
    rows, n_cols = logits.shape
    width = out.shape[1]
    assert out.shape[0] == rows and slot + 2 <= width
    if rows == 0:
        return out
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


# Plain SUM: no fusion epilogue.
_SUM_ONLY = AllReduceParams()


class DraftArgmaxMailbox:
    """Dedicated one-shot MNNVL SUM allreduce used as an allgather of the
    per-rank draft argmax pairs (see the module docstring)."""

    WIDTH = 128  # fp32 per row: one warp of float4 accesses in the kernel
    _instances: dict = {}
    # Communicators split for a mapping whose mailbox construction failed
    # (McastGPUBuffer only borrows them): kept for reuse instead of leaking one
    # world-wide split per attempt, like allreduce_mnnvl_pending_comms does
    # for the model workspace.
    _pending_comms: dict = {}

    def __init__(self, mapping):
        self.mapping = mapping
        tp = mapping.tp_size
        # Size the workspace so every exchange stays under the one-shot /
        # two-shot switch in allreduceOp.cpp: one Lamport buffer holds the
        # whole gathered payload (rows x WIDTH fp32 from each of the tp ranks).
        self.max_rows = _MNNVL_ONE_SHOT_THRESHOLD_BYTES // (self.WIDTH * 4 * tp)
        buffer_size_bytes = self.max_rows * self.WIDTH * 4 * tp
        # Same communicator flavour as the model's MNNVL workspace (an MPI
        # split under MPI, the TP process group under a non-MPI orchestrator).
        comm = self._pending_comms.get(mapping)
        if comm is None:
            comm = _get_mnnvl_workspace_comm(mapping)
            self._pending_comms[mapping] = comm
        # Converged across the group: every rank gets a workspace or raises.
        self._workspace = _build_mnnvl_workspace(comm, mapping, buffer_size_bytes)
        # Hand ownership of the communicator to the workspace.
        self._pending_comms.pop(mapping, None)
        # Persistent pack-kernel target; the kernel fully overwrites the rows
        # it is given on every exchange.
        self._staging = torch.empty(
            (self.max_rows, self.WIDTH),
            dtype=torch.float32,
            device=self._workspace["buffer_flags"].device,
        )

    @classmethod
    def get(cls, mapping) -> Optional["DraftArgmaxMailbox"]:
        """Mailbox for ``mapping`` or None (NCCL allgather) when unsupported.

        Built on the first eager call (CUDA-graph warm-up runs eagerly before
        capture); every TP rank reaches it in the same step, and the success
        is converged across the group so all ranks take the same path.
        Requires a pure-TP world (the communicator split is collective over
        the world) on an MNNVL fabric, the same condition under which the
        model's own allreduces use the MNNVL kernels: a group that spans nodes
        qualifies on its own, a single-node group only once the model has
        built its own MNNVL workspace for this mapping (i.e. MNNVL was
        requested explicitly).
        """
        if mapping in cls._instances:
            return cls._instances[mapping]
        if torch.cuda.is_current_stream_capturing():
            return None  # never allocate under capture; NCCL for this graph
        box = None
        try:
            tp = getattr(mapping, "tp_size", 1)
            model_uses_mnnvl = mapping in MNNVLAllReduce.allreduce_mnnvl_workspaces
            eligible = (
                tp > 1
                and 2 * tp <= cls.WIDTH
                and mapping.world_size == tp
                and not mapping.has_cp()
                and not mapping.enable_attention_dp
                and MNNVLAllReduce.is_mnnvl(
                    mapping, torch.float32, explicitly_requested=model_uses_mnnvl
                )
            )
            if eligible:
                box = cls(mapping)
                logger.info(
                    "MTP tail trim: draft argmax pairs exchanged through a "
                    f"dedicated MNNVL one-shot mailbox (tp={tp}, rows<="
                    f"{box.max_rows})"
                )
        except Exception as e:  # noqa: BLE001 - fall back to NCCL
            logger.warning(
                f"MTP tail trim: MNNVL draft-argmax mailbox "
                f"unavailable ({e}); using the NCCL allgather"
            )
            box = None
        cls._instances[mapping] = box
        return box

    def staging_rows(self, rows: int) -> torch.Tensor:
        """Pack-kernel target for an exchange of ``rows`` rows: a contiguous
        ``[rows, WIDTH]`` prefix of the persistent staging buffer."""
        return self._staging[:rows]

    def exchange(self, mailbox_rows: torch.Tensor) -> torch.Tensor:
        """SUM-allreduce of ``[rows, WIDTH]`` fp32 rows (one pair per rank)."""
        return _launch_mnnvl_allreduce(mailbox_rows, self._workspace, torch.float32, _SUM_ONLY)[0]


def gather_draft_argmax_pairs(logits: torch.Tensor, mapping) -> torch.Tensor:
    """Trimmed ``_get_local_max_and_combined`` + TP allgather along dim -1.

    Returns ``[rows, 2 * tp_size]`` fp32, rank-major (idx, val) pairs, exactly
    the tensor the original NCCL path produced (the one exception: a NaN max
    comes back with the canonical NaN payload from the mailbox's fp32 sum;
    the argmax that consumes it treats every NaN alike, so tokens match).
    """
    rows, vocab_per_rank = logits.shape
    tp, rank = mapping.tp_size, mapping.tp_rank
    idx_offset = rank * vocab_per_rank
    box = DraftArgmaxMailbox.get(mapping)
    if box is not None and rows <= box.max_rows:
        mailbox = box.staging_rows(rows)
        local_argmax_pack(logits, idx_offset, mailbox, slot=2 * rank)
        return box.exchange(mailbox)[:, : 2 * tp]
    combined = torch.empty((rows, 2), dtype=torch.float32, device=logits.device)
    local_argmax_pack(logits, idx_offset, combined, slot=0)
    return allgather(combined, mapping, dim=-1)
