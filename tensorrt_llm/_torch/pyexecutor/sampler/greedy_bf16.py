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
"""BF16 greedy logits (``TRTLLM_GREEDY_BF16_LOGITS=1``, default off).

``LogitsProcessor`` casts the BF16 ``lm_head`` output to FP32, a ``[B, vocab]``
casting copy every step, and the
greedy sampler then runs ``torch.argmax`` over the FP32 copy. BF16 -> FP32 is
exact, so the argmax over the BF16 logits is the same index. With the env set
the model engine keeps the logits in BF16 when every logits consumer of the
engine is the greedy argmax or casts on read (see
``PyTorchModelEngine._maybe_keep_bf16_greedy_logits``), and the sampler's fast
greedy path reads them with the two-pass Triton argmax below. Every other
sampling path, logits post processor and returned logits get an FP32 copy, as
before, so sampled tokens are unchanged.
"""

import os

import torch

try:
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]
except ImportError:  # pragma: no cover
    triton = None

GREEDY_BF16_LOGITS_ENV = "TRTLLM_GREEDY_BF16_LOGITS"

_INT32_MAX = 2147483647


def greedy_bf16_logits_enabled() -> bool:
    return os.environ.get(GREEDY_BF16_LOGITS_ENV, "0") == "1"


if triton is not None:

    @triton.jit
    def _argmax_partial_kernel(x_ptr, val_ptr, idx_ptr, V, row_stride, NCHUNK, CHUNK: tl.constexpr):
        # One CHUNK of one row: the first index of the max, where a NaN counts
        # as the max and the first NaN wins (torch.argmax's rule).
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        offs = chunk * CHUNK + tl.arange(0, CHUNK)
        mask = offs < V
        v = tl.load(
            x_ptr + row.to(tl.int64) * row_stride + offs, mask=mask, other=float("-inf")
        ).to(tl.float32)
        is_nan = (v != v) & mask
        nan_idx = tl.min(tl.where(is_nan, offs, 2147483647), axis=0)
        vv = tl.where(is_nan, float("-inf"), v)
        m = tl.max(vv, axis=0)
        first = tl.min(tl.where((vv == m) & mask, offs, 2147483647), axis=0)
        has_nan = nan_idx < 2147483647
        tl.store(val_ptr + row * NCHUNK + chunk, tl.where(has_nan, float("nan"), m))
        tl.store(idx_ptr + row * NCHUNK + chunk, tl.where(has_nan, nan_idx, first))

    @triton.jit
    def _argmax_final_kernel(val_ptr, idx_ptr, out_ptr, NCHUNK, NC_P2: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, NC_P2)
        mask = c < NCHUNK
        v = tl.load(val_ptr + row * NCHUNK + c, mask=mask, other=float("-inf"))
        i = tl.load(idx_ptr + row * NCHUNK + c, mask=mask, other=2147483647)
        is_nan = (v != v) & mask
        nan_idx = tl.min(tl.where(is_nan, i, 2147483647), axis=0)
        vv = tl.where(is_nan, float("-inf"), v)
        m = tl.max(vv, axis=0)
        first = tl.min(tl.where((vv == m) & mask, i, 2147483647), axis=0)
        res = tl.where(nan_idx < 2147483647, nan_idx, first)
        tl.store(out_ptr + row, res.to(tl.int64))


def bf16_argmax_supported(logits: torch.Tensor) -> bool:
    return (
        triton is not None
        and logits.is_cuda
        and logits.dtype == torch.bfloat16
        and logits.dim() == 2
        and logits.size(0) > 0
        and logits.stride(-1) == 1
        and logits.size(-1) < _INT32_MAX
    )


def bf16_argmax_last_dim(
    logits: torch.Tensor, chunk: int = 4096, num_warps: int = 4
) -> torch.Tensor:
    """``torch.argmax(logits, dim=-1)`` (int64) for 2-D BF16 logits with a unit
    last-dim stride (bf16_argmax_supported), in two passes at full bandwidth."""
    rows, vocab = logits.shape
    nchunk = triton.cdiv(vocab, chunk)
    vals = torch.empty((rows, nchunk), dtype=torch.float32, device=logits.device)
    idxs = torch.empty((rows, nchunk), dtype=torch.int32, device=logits.device)
    out = torch.empty((rows,), dtype=torch.int64, device=logits.device)
    _argmax_partial_kernel[(rows, nchunk)](
        logits, vals, idxs, vocab, logits.stride(0), nchunk, CHUNK=chunk, num_warps=num_warps
    )
    _argmax_final_kernel[(rows,)](
        vals, idxs, out, nchunk, NC_P2=triton.next_power_of_2(nchunk), num_warps=1
    )
    return out
