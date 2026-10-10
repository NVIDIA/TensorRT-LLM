# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constants and per-shape buffers shared by the Wan fused FP8 ops."""

import math

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
HEAD_DIM = 128
LOG2E = 1.0 / math.log(2.0)
# sq = sk = 1 / sqrt(c), c = 1 / (sm_scale * log2 e): unit softmax scale.
QK_MUL = math.sqrt(HEAD_DIM**-0.5 * LOG2E)

_buffers = {}


def shape_buffers(batch: int, seq: int, device: torch.device) -> dict:
    """Per-shape device constants and scratch, reused across calls."""
    key = (batch, seq, str(device))
    if key not in _buffers:
        f32 = dict(device=device, dtype=torch.float32)
        _buffers[key] = dict(
            bmm1=torch.tensor([1.0 / LOG2E, 1.0], **f32),
            cu_seqlens=torch.arange(0, (batch + 1) * seq, seq, device=device, dtype=torch.int32),
            seqlens=torch.full((batch,), seq, device=device, dtype=torch.int32),
            qk_mul=torch.tensor([QK_MUL, QK_MUL], **f32),
            mul=torch.tensor([QK_MUL, QK_MUL, 1.0], **f32),
            scale_v=torch.ones(1, **f32),
            amax=torch.zeros(3, **f32),
            amax_v=torch.zeros(1, **f32),
        )
    return _buffers[key]


def rope_tables(cos: torch.Tensor, sin: torch.Tensor, tokens: int):
    cos2d = cos.reshape(-1, HEAD_DIM).float().contiguous()
    sin2d = sin.reshape(-1, HEAD_DIM).float().contiguous()
    # Fewer rows than tokens: one table shared by every batch entry.
    seq_per_batch = 0 if cos2d.shape[0] == tokens else cos2d.shape[0]
    return cos2d, sin2d, seq_per_batch
