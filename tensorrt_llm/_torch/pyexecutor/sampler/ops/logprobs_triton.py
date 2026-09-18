# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split-row Triton kernels for the logprobs path of TorchSampler.

Why not Inductor: ``log_softmax`` / ``count_nonzero`` over a (rows, vocab) fp32
tensor with rows in the low tens and vocab ~250k is a tiny-outer / huge-inner
reduction. Inductor emits ONE CTA per row for it, and on sm100+ its split
reduction heuristic (``InductorChoices.reduction_split_factor``,
``no_split_threshold = 524288`` when ``props.major >= 10``) refuses to split
reductions shorter than 512k elements, so ``split_reductions=True`` is a no-op
for vocab 248320. Measured on a 212-SM sm107 part (rows 8..32, vocab 248320):
the compiled ``gather_log_softmax`` costs ~117 us and ``determine_sampled_rank``
43-74 us per call, independent of the row count; these kernels take 8-12 us and
4-5 us (results/_analysis_scratch/g4-sampler/results.md).

Both ops are two launches each: a pass that reduces each row split across
``NSPLIT`` program instances into a small partial buffer (no atomics, hence
deterministic), and a pass that combines the partials and writes the output.
fp32 accumulation throughout. The row count only enters the launch grid, so
there is no shape specialisation / recompilation across batch compositions;
the JIT compiles once per (dtype, alignment) signature on first use, which
``warmup_compile_fusions`` triggers at engine init.
"""

import torch
import triton
import triton.language as tl

_NEG_BIG = tl.constexpr(-3.0e38)  # finite "-inf": keeps exp(m - m_new) away from inf - inf
_NSPLIT = 16
_BLOCK = 2048
_NUM_WARPS = 8


@triton.jit
def _row_lse_partial_kernel(
    x_ptr,
    idx_ptr,
    part_ptr,
    V,
    x_stride,
    CHUNK,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    src = tl.load(idx_ptr + row).to(tl.int64)
    base = x_ptr + src * x_stride
    start = split * CHUNK
    m = tl.full([BLOCK], _NEG_BIG, tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for off in range(0, CHUNK, BLOCK):
        cols = start + off + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < V, other=_NEG_BIG).to(tl.float32)
        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new
    m_tot = tl.max(m, 0)
    s_tot = tl.sum(s * tl.exp(m - m_tot), 0)
    p = part_ptr + (row * NSPLIT + split) * 2
    tl.store(p, m_tot)
    tl.store(p + 1, s_tot)


@triton.jit
def _row_log_softmax_apply_kernel(
    x_ptr,
    idx_ptr,
    part_ptr,
    out_ptr,
    V,
    x_stride,
    CHUNK,
    NSPLIT: tl.constexpr,
    NSPLIT_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = tl.arange(0, NSPLIT_P2)
    pm = tl.load(part_ptr + (row * NSPLIT + offs) * 2, mask=offs < NSPLIT, other=_NEG_BIG)
    ps = tl.load(part_ptr + (row * NSPLIT + offs) * 2 + 1, mask=offs < NSPLIT, other=0.0)
    m_tot = tl.max(pm, 0)
    s_tot = tl.sum(ps * tl.exp(pm - m_tot), 0)
    lse = m_tot + tl.log(s_tot)
    src = tl.load(idx_ptr + row).to(tl.int64)
    base = x_ptr + src * x_stride
    obase = out_ptr + row.to(tl.int64) * V
    start = split * CHUNK
    for off in range(0, CHUNK, BLOCK):
        cols = start + off + tl.arange(0, BLOCK)
        msk = cols < V
        x = tl.load(base + cols, mask=msk, other=0.0).to(tl.float32)
        tl.store(obase + cols, x - lse, mask=msk)


@triton.jit
def _row_greater_count_partial_kernel(
    lp_ptr,
    thr_ptr,
    part_ptr,
    V,
    lp_stride,
    CHUNK,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    thr = tl.load(thr_ptr + row)
    base = lp_ptr + row.to(tl.int64) * lp_stride
    start = split * CHUNK
    acc = tl.zeros([BLOCK], tl.int32)
    for off in range(0, CHUNK, BLOCK):
        cols = start + off + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < V, other=_NEG_BIG)
        acc += (x > thr).to(tl.int32)
    tl.store(part_ptr + row * NSPLIT + split, tl.sum(acc, 0))


@triton.jit
def _row_sum_partials_kernel(part_ptr, out_ptr, NSPLIT: tl.constexpr, NSPLIT_P2: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, NSPLIT_P2)
    p = tl.load(part_ptr + row * NSPLIT + offs, mask=offs < NSPLIT, other=0)
    tl.store(out_ptr + row, tl.sum(p, 0))


def _chunk(vocab: int) -> int:
    return triton.cdiv(triton.cdiv(vocab, _NSPLIT), _BLOCK) * _BLOCK


def _p2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def gather_log_softmax(
    inputs_cuda: torch.Tensor,
    indices_cuda: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``log_softmax(inputs_cuda[indices_cuda], dim=-1)`` in fp32 (2 launches).

    ``inputs_cuda``: (num_rows, vocab), unit inner stride; ``indices_cuda``: 1-D
    integer row indices. Writes into ``out`` -- a (len(indices), vocab) fp32
    tensor with densely packed rows, e.g. a row slice of the sampler's logprobs
    buffer -- when given, otherwise into a fresh contiguous tensor. Returns the
    output tensor.
    """
    assert inputs_cuda.dim() == 2 and inputs_cuda.stride(1) == 1
    assert indices_cuda.dim() == 1
    rows = indices_cuda.shape[0]
    vocab = inputs_cuda.shape[1]
    if out is None:
        out = torch.empty((rows, vocab), dtype=torch.float32, device=inputs_cuda.device)
    else:
        assert out.dtype == torch.float32 and tuple(out.shape) == (rows, vocab)
        # The apply kernel addresses out as out_ptr + row * vocab + col.
        assert out.stride(1) == 1 and (rows <= 1 or out.stride(0) == vocab)
    if rows == 0:
        return out
    indices_cuda = indices_cuda.contiguous()
    part = torch.empty((rows, _NSPLIT, 2), dtype=torch.float32, device=inputs_cuda.device)
    grid = (rows, _NSPLIT)
    chunk = _chunk(vocab)
    _row_lse_partial_kernel[grid](
        inputs_cuda,
        indices_cuda,
        part,
        vocab,
        inputs_cuda.stride(0),
        chunk,
        NSPLIT=_NSPLIT,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    _row_log_softmax_apply_kernel[grid](
        inputs_cuda,
        indices_cuda,
        part,
        out,
        vocab,
        inputs_cuda.stride(0),
        chunk,
        NSPLIT=_NSPLIT,
        NSPLIT_P2=_p2(_NSPLIT),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


def determine_sampled_rank(
    group_logprobs_cuda: torch.Tensor, sampled_logprobs_cuda: torch.Tensor
) -> torch.Tensor:
    """``(group_logprobs_cuda > sampled_logprobs_cuda).count_nonzero(dim=-1)`` as int32 (2 launches).

    ``group_logprobs_cuda``: (rows, vocab) fp32 with unit inner stride;
    ``sampled_logprobs_cuda``: (rows, 1) or (rows,) fp32.
    """
    assert group_logprobs_cuda.dim() == 2 and group_logprobs_cuda.stride(1) == 1
    rows, vocab = group_logprobs_cuda.shape
    out = torch.empty((rows,), dtype=torch.int32, device=group_logprobs_cuda.device)
    if rows == 0:
        return out
    thr = sampled_logprobs_cuda.reshape(-1).to(group_logprobs_cuda.dtype).contiguous()
    assert thr.shape[0] == rows
    part = torch.empty((rows, _NSPLIT), dtype=torch.int32, device=group_logprobs_cuda.device)
    _row_greater_count_partial_kernel[(rows, _NSPLIT)](
        group_logprobs_cuda,
        thr,
        part,
        vocab,
        group_logprobs_cuda.stride(0),
        _chunk(vocab),
        NSPLIT=_NSPLIT,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    _row_sum_partials_kernel[(rows,)](
        part, out, NSPLIT=_NSPLIT, NSPLIT_P2=_p2(_NSPLIT), num_warps=1
    )
    return out
