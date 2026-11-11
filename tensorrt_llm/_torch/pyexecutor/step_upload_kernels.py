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
"""Small host-to-device uploads of a step written by one SM kernel instead of copy-engine copies.

A copy-engine copy hands the execution stream to the copy engine and back, which
delays the kernels queued behind it; for a few hundred bytes an SM kernel reading
pinned host memory over the host link is cheaper.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n0", "n1"])
def _small_copy_pair_kernel(dst0, src0, n0, dst1, src1, n1, BLOCK: tl.constexpr):
    # Program 0 copies n0 bytes src0 -> dst0, program 1 copies n1 bytes src1 -> dst1.
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        mask = offs < n0
        tl.store(dst0 + offs, tl.load(src0 + offs, mask=mask), mask=mask)
    else:
        mask = offs < n1
        tl.store(dst1 + offs, tl.load(src1 + offs, mask=mask), mask=mask)


_SMALL_COPY_MAX_BYTES = 4096


def copy_small_pinned_pair(
    dst0: torch.Tensor, src0: torch.Tensor, dst1: torch.Tensor, src1: torch.Tensor
) -> bool:
    """``dst0.copy_(src0, non_blocking=True); dst1.copy_(src1, non_blocking=True)`` as one
    kernel on the current stream, for small contiguous pinned host sources.

    The kernel reads the host buffers over the host link when it runs, as the copy engine
    reads them when the DMA runs. Unlike a ``copy_``, the kernel's read is not recorded
    with the pinned-memory allocator: the caller keeps the sources alive until the kernel
    has completed. Returns False (nothing launched) when a pair does not qualify.
    """
    pairs = ((dst0, src0), (dst1, src1))
    for dst, src in pairs:
        if not (
            dst.is_cuda
            and src.is_pinned()
            and dst.is_contiguous()
            and src.is_contiguous()
            and dst.dtype == src.dtype
            and dst.numel() == src.numel()
            and dst.numel() * dst.element_size() <= _SMALL_COPY_MAX_BYTES
        ):
            return False
    nbytes = [dst.numel() * dst.element_size() for dst, _ in pairs]
    block = triton.next_power_of_2(max(max(nbytes), 16))
    _small_copy_pair_kernel[(2,)](
        dst0.view(torch.uint8),
        src0.view(torch.uint8),
        nbytes[0],
        dst1.view(torch.uint8),
        src1.view(torch.uint8),
        nbytes[1],
        BLOCK=block,
    )
    return True
