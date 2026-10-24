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
"""Bandwidth-bound widening of LM-head logits to float32.

``LogitsProcessor`` widens the BF16 head output with ``Tensor.float()``. A
dtype-converting copy runs PyTorch's dynamic-casting ``unrolled_elementwise``
kernel, which reaches only a fraction of HBM bandwidth on a ``[rows, vocab]`` logits
tensor (K3 DEP16, 240 x 163840, twice per
speculative step). ``TLLM_FAST_LOGITS_CAST=1`` widens with a vectorized Triton
kernel instead. BF16/FP16 -> FP32 is exact, so the result is bit-identical.
"""

import os

import torch
import triton
import triton.language as tl

FAST_LOGITS_CAST_ENV = "TLLM_FAST_LOGITS_CAST"

# 2048 elements and 4 warps per program (larger programs are slower).
_BLOCK = 2048
_NUM_WARPS = 4


def fast_logits_cast_enabled() -> bool:
    """Whether ``TLLM_FAST_LOGITS_CAST=1`` selects the Triton widening kernel."""
    return os.environ.get(FAST_LOGITS_CAST_ENV, "0") == "1"


@triton.jit
def _widen_to_fp32_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    values = tl.load(src_ptr + offsets, mask=mask, eviction_policy="evict_first")
    tl.store(dst_ptr + offsets, values.to(tl.float32), mask=mask)


def widen_logits_to_fp32(logits: torch.Tensor) -> torch.Tensor:
    """``logits.float()``, bit-identical, through the vectorized kernel when it applies.

    Applies to a contiguous BF16/FP16 CUDA tensor; anything else (already
    float32, strided, CPU, empty) takes ``Tensor.float()``.
    """
    if (
        logits.dtype not in (torch.bfloat16, torch.float16)
        or not logits.is_cuda
        or not logits.is_contiguous()
        or logits.numel() == 0
    ):
        return logits.float()
    out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    numel = logits.numel()
    _widen_to_fp32_kernel[(triton.cdiv(numel, _BLOCK),)](
        logits, out, numel, BLOCK=_BLOCK, num_warps=_NUM_WARPS
    )
    return out
