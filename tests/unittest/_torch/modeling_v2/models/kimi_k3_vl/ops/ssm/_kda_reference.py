# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's KDA prefill constants and comparison helpers, for the ``ssm/kda_prefill`` entry test."""

import torch

HEAD_DIM = 128  # Kimi K3's KDA head_dim (K = V)
LOWER_BOUND = -5.0  # Kimi K3's gate_lower_bound
SCALE = HEAD_DIM**-0.5
CHUNK = 64


def cu_seqlens_of(seq_lens, dtype=torch.int64) -> torch.Tensor:
    """Kimi K3's layer passes the metadata's int64 ``query_start_loc_long``."""
    starts = [0]
    for n in seq_lens:
        starts.append(starts[-1] + n)
    return torch.tensor(starts, dtype=dtype, device="cuda")


def assert_scaled_close(got: torch.Tensor, ref: torch.Tensor, atol: float) -> None:
    """Within ``atol`` of ``ref``'s largest magnitude (chunked bf16 evaluation against an fp64 recurrence)."""
    scale = ref.abs().max().clamp_min(1e-6)
    torch.testing.assert_close(got.double() / scale, ref.double() / scale, rtol=0.0, atol=atol)
