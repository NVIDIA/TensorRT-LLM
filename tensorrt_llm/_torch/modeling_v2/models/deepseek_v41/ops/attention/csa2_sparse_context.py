# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 context attention on the engine's FP8 sparse context kernel (`trtllm::csa2_sparse_context`), as stock's
`forward_fused_context` runs it over the shared bank of `csa2_shared_stage`.

Each query's 64 heads attend to the bank rows `indices[t, :lens[t]]` with the sink in the softmax denominator. The bf16
query is quantized on load with `q_scale` (the staging scale), BMM1 / BMM2 take the stage's scales. The output,
`csa2_sparse_context_o_a_input`, is what the grouped O-A projection reads (stock's `output_fp8`): the attention rotated
back at each row's position (inverse RoPE of the 64-channel tail of every head) and quantized per row and 128 channels,
E4M3 `[groups, n, 64 / groups * 512]` with fp32 scales `[groups, 64 / groups * 4, n rounded up to 4]`; the separate
inverse-RoPE / quantize launch of the O-A input is gone.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_rope import csa2_rope_tail

HEADS, HEAD_DIM = 64, 512
_LOG2E = 1.4426950408889634


def csa2_sparse_context_supported(device) -> bool:
    """True when the engine's FP8 sparse context kernel serves this device (SM100 family). The op is part of the
    engine build this target requires, so it is not probed for (a missing op fails loudly at the call)."""
    return torch.cuda.get_device_capability(device)[0] == 10


def _attend(q, bank, indices, lens, q_scale, bmm1, bmm2, sink) -> torch.Tensor:
    """fp32 softmax attention over the dequantized bank rows; the softmax scale is `bmm1[0] / (q_dequant * kv_dequant)`
    with `q_dequant = 1 / q_scale` and `kv_dequant = bmm2` (the stage's BMM1 folds both dequant scales)."""
    # the live query rows: `indices` / `lens` may carry spare rows past them
    n, k = q.shape[0], indices.shape[1]
    indices = indices[:n]
    kv_dequant = bmm2.float().reshape(())
    q_scale = (
        torch.ones((), device=q.device) if q_scale is None else q_scale
    )  # an E4M3 query is at unit scale
    softmax = (bmm1[0].float() * q_scale.float().reshape(()) / kv_dequant).reshape(())
    rows = bank.float() * kv_dequant
    keep = torch.arange(k, device=q.device)[None, :] < lens[:n, None]
    kv = rows[indices.long().clamp(0, rows.shape[0] - 1)] * keep.unsqueeze(-1)  # [n, k, 512]
    logits = torch.einsum("nhd,nkd->nhk", q.float(), kv) * softmax
    logits = logits.masked_fill(~keep.unsqueeze(1), float("-inf"))
    full = torch.cat([logits, sink.float().view(1, -1, 1).expand(n, -1, 1)], dim=-1)
    p = torch.softmax(full, dim=-1)[..., :-1]
    return torch.einsum("nhk,nkd->nhd", p, kv)


def _valid_common(q, bank, indices, lens, q_scale, bmm1, bmm2, sink) -> None:
    # a bf16 query is quantized on load with `q_scale`; an E4M3 one (unit scale, stock's fused context query) has none
    assert q.dim() == 3 and q.shape[1:] == (HEADS, HEAD_DIM) and q.stride(2) == 1
    assert (q.dtype == torch.bfloat16) or (
        q.dtype == torch.float8_e4m3fn and q_scale is None and q.is_contiguous()
    )
    assert (
        bank.dtype == torch.float8_e4m3fn
        and bank.dim() == 2
        and bank.shape[1] == HEAD_DIM
        and bank.is_contiguous()
    )
    assert (
        indices.dtype == torch.int32 and indices.shape[0] >= q.shape[0] and indices.stride(1) == 1
    )
    assert lens.dtype == torch.int32 and lens.shape[0] >= q.shape[0]
    assert q_scale is None or q_scale.dtype == torch.float32
    assert bmm1.dtype == bmm2.dtype == torch.float32 and bmm1.numel() == 2 and bmm2.numel() == 1
    assert sink.dtype == torch.float32 and sink.numel() == HEADS and sink.is_contiguous()


class Csa2SparseContextOAInput(OpWrapper):
    """`(data E4M3 [groups, n, 64 / groups * 512], scale fp32 [groups, 64 / groups * 4, pad4(n)])`: each query's 64
    heads attend to the bank rows `indices[t, :lens[t]]`, with the sink in the softmax denominator; the result is
    rotated back at `pos` (int32 `[n]`; `cos_sin` the engine's fp32 `[P, 64]` table, all cos then all sin) and
    quantized per row and 128 channels, as the grouped block-128 O-A reads it.

    `q` bf16 `[n, 64, 512]` (rotated) is quantized on load with `q_scale` (or E4M3 at unit scale with `q_scale`
    None); `bmm1` `[2]` is `q_dequant * kv_dequant * softmax_scale` and its log2(e) multiple, `bmm2` `[1]` the bank's
    dequant scale (all three from `csa2_shared_stage`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep2: 8 groups of 8 heads, 300 queries of 192 keys",
            spec=dict(n=300, rows=2000, keys=192, groups=8),
        ),
        Cell(
            why="dep2: a 2051-query chunk (scale rows padded), 640 key slots",
            spec=dict(n=2051, rows=6000, keys=640, groups=8),
        ),
    )
    note = """
    The kernel's E4M3 Q, K and P (P x448: 6% relative RMS on their own) and the output's E4M3 rounding under a 1x128
    amax scale: gate 7% relative RMS of the dequantized output against the fp32 reference.
    """

    def __call__(
        self, q, bank, indices, lens, q_scale, bmm1, bmm2, sink, pos, cos_sin, groups: int
    ) -> tuple:
        n = q.shape[0]
        width = HEADS // groups * HEAD_DIM
        data = torch.empty(groups, n, width, dtype=torch.float8_e4m3fn, device=q.device)
        scale = torch.empty(
            groups, width // 128, (n + 3) // 4 * 4, dtype=torch.float32, device=q.device
        )
        torch.ops.trtllm.csa2_sparse_context(
            bank, q, q_scale, indices, lens, bmm1, bmm2, sink, data, scale, pos, cos_sin
        )
        return data, scale

    def reference(
        self, q, bank, indices, lens, q_scale, bmm1, bmm2, sink, pos, cos_sin, groups: int
    ) -> torch.Tensor:
        """fp32 `[n, 64 * 512]`: the attention rotated back (the scale-quantization is the test's band)."""
        att = _attend(q, bank, indices, lens, q_scale, bmm1, bmm2, sink)
        table = torch.stack([cos_sin[:, :32], cos_sin[:, 32:]], dim=-1)
        return csa2_rope_tail.reference(att, table, pos, 64, True).reshape(att.shape[0], -1)

    def is_valid(
        self, q, bank, indices, lens, q_scale, bmm1, bmm2, sink, pos, cos_sin, groups: int
    ) -> None:
        _valid_common(q, bank, indices, lens, q_scale, bmm1, bmm2, sink)
        assert HEADS % groups == 0 and pos.shape == (q.shape[0],) and pos.is_contiguous()
        assert pos.dtype in (torch.int32, torch.int64)
        assert (
            cos_sin.dtype == torch.float32 and cos_sin.is_contiguous() and cos_sin.shape[-1] == 64
        )


def o_a_input_dequant(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 `[n, groups * width]` of a `csa2_sparse_context_o_a_input` result (a test / reference helper)."""
    g, n, w = data.shape
    s = scale[:, :, :n].permute(0, 2, 1).repeat_interleave(128, dim=2)  # [g, n, w]
    return (data.float() * s).permute(1, 0, 2).reshape(n, g * w)


csa2_sparse_context_o_a_input = Csa2SparseContextOAInput()
