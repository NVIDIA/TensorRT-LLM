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
"""Widen LM-head logits to float32 with a vectorized Triton kernel.

`LogitsProcessor` calls `widen_logits_to_fp32` when ``TLLM_FAST_LOGITS_CAST=1``.
"""

import os

import torch
import triton
import triton.language as tl

FAST_LOGITS_CAST_ENV = "TLLM_FAST_LOGITS_CAST"

# Elements and warps per program. With 256 x 163840 logits, 2048/4 takes 25.3 us
# (75% of the 252 MB byte floor), 4096/8 takes 28.0 us, and 8192/16 and larger
# are slower.
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
    """Return ``logits.float()``, through the vectorized kernel when it applies.

    The kernel applies to a contiguous BF16/FP16 CUDA tensor. Anything else
    (already float32, strided, CPU, empty) takes ``Tensor.float()``.

    ``Tensor.float()`` runs PyTorch's dynamic-casting ``unrolled_elementwise``
    kernel, which reaches about 1/6 of HBM bandwidth on ``[rows, vocab]`` logits
    (K3 DEP16, 240 x 163840: 120 us against an 18 us byte floor, twice per
    speculative step). BF16/FP16 -> FP32 is exact, so the kernel's result is
    bit-identical.
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
