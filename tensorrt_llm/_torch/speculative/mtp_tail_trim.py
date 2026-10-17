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
"""Pack local draft argmax pairs and control redundant DSA metadata updates."""

import functools
import os

import torch

from tensorrt_llm.mapping import Mapping

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton ships with the TRT-LLM image
    _HAS_TRITON = False

MTP_TAIL_TRIM_ENV = "TRTLLM_MTP_TAIL_TRIM"


@functools.lru_cache(maxsize=1)
def mtp_tail_trim_enabled() -> bool:
    """Process-constant gate: unset / "1" = trimmed path, "0" = original path.

    Cached so the choice is fixed before CUDA-graph capture.
    """
    return os.environ.get(MTP_TAIL_TRIM_ENV, "1").strip() != "0"


if _HAS_TRITON:

    @triton.jit
    def _argmax_first_nan_combine(v1, i1, v2, i2):
        # Total order of at::native GreaterOrNan: NaN beats everything, among
        # NaNs or equal values the lower index wins, otherwise the larger value.
        n1 = v1 != v1
        n2 = v2 != v2
        lower = i1 < i2
        take1 = (n1 & ((~n2) | lower)) | ((~n1) & (~n2) & ((v1 > v2) | ((v1 == v2) & lower)))
        return tl.where(take1, v1, v2), tl.where(take1, i1, i2)

    @triton.jit
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
        best_i = tl.full([BLOCK], 2147483647, tl.int32)
        for start in range(0, n_cols, BLOCK):
            cols = start + offs
            m = cols < n_cols
            if BF16_BITS:
                # bf16 passed as int16 bits: widen by shifting the bits into
                # the fp32 high half, exactly like at::BFloat16 -> float (keeps
                # NaN payloads, which a cvt would canonicalise). -128 = 0xFF80
                # = bf16 -inf.
                raw = tl.load(base + cols, mask=m, other=-128).to(tl.int32)
                v = (raw << 16).to(tl.float32, bitcast=True)
            else:
                v = tl.load(base + cols, mask=m, other=float("-inf")).to(tl.float32)
            # Chunks are visited in increasing column order, so within a lane
            # the new element always has the larger index: replace only on a
            # strictly better value (or NaN over non-NaN), or the first valid.
            nv = v != v
            nb = best_v != best_v
            take = m & ((best_i == 2147483647) | (nv & (~nb)) | ((~nv) & (~nb) & (v > best_v)))
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
    if not can_pack_draft_argmax(logits):
        raise ValueError("Expected contiguous-column CUDA FP16/BF16/FP32 logits")
    if out.dtype != torch.float32 or not out.is_contiguous() or out.device != logits.device:
        raise ValueError("Output must be contiguous fp32 on the logits device")
    rows, n_cols = logits.shape
    width = out.shape[1]
    if out.shape[0] != rows or slot < 0 or slot + 2 > width:
        raise ValueError("Output shape does not contain the requested argmax pair slot")
    if rows == 0:
        return out
    bf16_bits = logits.dtype == torch.bfloat16
    src = logits.view(torch.int16) if bf16_bits else logits
    block = min(triton.next_power_of_2(n_cols), 8192)
    num_warps = 16 if block >= 8192 else (8 if block >= 2048 else 4)
    _draft_local_argmax_pack_kernel[(rows,)](
        src,
        out,
        n_cols,
        logits.stride(0),
        out.stride(0),
        idx_offset,
        slot,
        width,
        BLOCK=block,
        OUT_BLOCK=triton.next_power_of_2(width),
        BF16_BITS=bf16_bits,
        num_warps=num_warps,
    )
    return out


def can_pack_draft_argmax(logits: torch.Tensor) -> bool:
    """Whether the fused pack supports these CUDA logits without conversion."""
    return (
        _HAS_TRITON
        and logits.is_cuda
        and logits.ndim == 2
        and logits.shape[1] > 0
        and logits.stride(-1) == 1
        and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def gather_draft_argmax_pairs(logits: torch.Tensor, mapping: Mapping) -> torch.Tensor:
    """Gather rank-major (index, value) pairs as fp32 ``[rows, 2 * tp_size]``."""
    from ..distributed.ops import allgather

    rows, vocab_per_rank = logits.shape
    combined = torch.empty((rows, 2), dtype=torch.float32, device=logits.device)
    local_argmax_pack(logits, mapping.tp_rank * vocab_per_rank, combined)
    return allgather(combined, mapping, dim=-1)
