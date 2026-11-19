# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the attn_res + RMSNorm entries share: their shapes, cells, reference and guard."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell

#: Hidden size the kernel is compiled for (Kimi K3.1).
HIDDEN = 7168
#: Candidates per token (snapshots + the layer row) and rows per launch the kernel serves.
MIN_CANDIDATES, MAX_CANDIDATES, MAX_ROWS = 2, 12, 1024


def cells(add: bool) -> tuple[Cell, ...]:
    """Rows per rank of a generation server's decode / verify step, and the candidate counts of
    the model's attention-residual blocks (Kimi K3.1: block size 12 over 93 layers, so 2 in the
    first block up to 9 in the last; the kernel takes up to 12)."""

    def c(rows: int, candidates: int, why: str) -> Cell:
        return Cell(why=why, spec=dict(rows=rows, candidates=candidates, add=add))

    return (
        c(1, 2, "one decode row, the fewest candidates"),
        c(1, 12, "one decode row, the most candidates (8 CTAs per token)"),
        c(6, 5, "TEP16 low-latency verify, one request x 6 rows, middle block"),
        c(8, 7, "one DFlash2 verify step (1 + 7 rows), 8 CTAs per token"),
        c(8, 9, "TEP32 B1 verify (8 rows), last block"),
        c(17, 4, "first row count past the 8-CTA clusters (4 CTAs per token)"),
        c(24, 9, "TEP16 B4 verify (24 rows), last block"),
        c(72, 5, "TEP16 B12 verify (72 rows), middle block"),
        c(99, 9, "DEP16 mean rows per rank (14 requests x 7), last block"),
        c(112, 3, "attention TP verify, 14 sequences x 8 rows, two or three candidates"),
        c(192, 12, "attention TP generation server's row limit, deepest block"),
        c(224, 9, "DEP16 rows per rank at 32 requests x 7, last block"),
        c(256, 9, "attention DP generation server's row limit"),
        c(1024, 5, "the kernel's row limit"),
    )


def reference(
    layer: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> torch.Tensor:
    """Attention-residual mix of ``[block..., layer]`` then RMSNorm, FP32 math with the model's
    BF16 roundings: the mixture, the normalized row and its product with the weight."""
    v = torch.cat((block, layer.unsqueeze(0)), dim=0).float()  # [N, M, H]
    rsigma = torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + rms_eps)
    score_weight = rms_weight.float() * res_weight.float()
    logits = ((v * rsigma) * score_weight).sum(-1)  # [N, M]
    probs = logits.softmax(0)
    mixed = (probs.unsqueeze(-1) * v).sum(0).to(torch.bfloat16).float()
    normed = (mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + out_eps)).to(torch.bfloat16)
    return (normed.float() * out_weight.float()).to(torch.bfloat16)


def check_operands(
    layer: torch.Tensor, block: torch.Tensor, *rest: torch.Tensor, addends=()
) -> None:
    """The kernel reads raw pointers of a 7168-wide build and re-validates nothing: any other
    shape, dtype or layout is read as if it were that one."""
    assert layer.dim() == 2 and layer.shape[1] == HIDDEN, f"layer must be [M, {HIDDEN}]"
    m = layer.shape[0]
    assert 1 <= m <= MAX_ROWS, f"1 <= M <= {MAX_ROWS}"
    assert block.dim() == 3 and tuple(block.shape[1:]) == (m, HIDDEN), "block must be [N-1, M, H]"
    assert MIN_CANDIDATES <= block.shape[0] + 1 <= MAX_CANDIDATES, (
        f"{MIN_CANDIDATES} <= candidates <= {MAX_CANDIDATES}"
    )
    for t in (layer, block, *addends, *rest):
        assert t.dtype == torch.bfloat16 and t.is_contiguous(), (
            "every tensor must be contiguous BF16"
        )
        assert t.device == layer.device and t.data_ptr() % 16 == 0, (
            "every tensor must be 16-byte aligned on the layer's device"
        )
    for a in addends:
        assert a.shape == layer.shape, "the addend must be [M, H]"
    for w in rest:
        assert w.numel() == HIDDEN, f"each weight must have {HIDDEN} elements"
