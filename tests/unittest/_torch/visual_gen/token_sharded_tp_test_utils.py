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
"""Shared helpers for the token-sharded TP tests (CPU, collective and kernel files)."""

import torch

from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    TokenShardedTP,
    TokenShardPlan,
    swizzled_sf_numel,
)
from tensorrt_llm.math_utils import pad_up


def sf_offsets(rows: int, sf_cols: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Flat ``[rows * sf_cols]`` element offsets of the 128x4-swizzled SF layout.

    From ``get_sf_out_offset_128x4`` (quantization.cuh), independent of ``regroup_swizzled_sf``:
    ``[m // 128][k // 4][m % 32][(m % 128) // 32][k % 4]``.
    """
    num_k_tiles = pad_up(sf_cols, 4) // 4
    m = torch.arange(rows, device=device).view(-1, 1)
    k = torch.arange(sf_cols, device=device).view(1, -1)
    return (
        (m // 128) * (num_k_tiles * 512)
        + (k // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + (k % 4)
    ).reshape(-1)


def swizzle_ref(lin: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """Linear ``[rows, sf_cols]`` -> 1-D 128x4-swizzled buffer (pad entries = pad_value)."""
    rows, sf_cols = lin.shape
    out = torch.full(
        (swizzled_sf_numel(rows, sf_cols),), pad_value, dtype=lin.dtype, device=lin.device
    )
    out[sf_offsets(rows, sf_cols, lin.device)] = lin.reshape(-1)
    return out


def unswizzle_ref(buf: torch.Tensor, rows: int, sf_cols: int) -> torch.Tensor:
    """1-D swizzled buffer -> linear ``[rows, sf_cols]`` (on the buffer's device)."""
    return buf.reshape(-1)[sf_offsets(rows, sf_cols, buf.device)].view(rows, sf_cols)


def simulated_helper(plan: TokenShardPlan) -> TokenShardedTP:
    """A TokenShardedTP for one simulated rank's plan, without a process group.

    Only calls that run no collective work: ``begin`` for the plan's own shape, shard,
    local_view, per_sample_table, pad_row_input, the padding helpers and the collectives'
    shape checks.
    """
    ts = TokenShardedTP.__new__(TokenShardedTP)
    ts.group, ts.group_name = None, "simulated"
    ts.tp_size, ts.tp_rank = plan.tp_size, plan.tp_rank
    ts.row_align = plan.row_align
    ts._plans = {(plan.batch_size, plan.seq_len): plan}
    ts._plan = plan
    return ts


def padded_rows(t: torch.Tensor, plan: TokenShardPlan) -> torch.Tensor:
    """``[B, S, *rest]`` -> ``[B * S_pad, *rest]`` with zero pad rows (the padded stream)."""
    b, s = t.shape[:2]
    pad = t.new_zeros(b, plan.padded_seq_len - s, *t.shape[2:])
    return torch.cat([t, pad], dim=1).reshape(b * plan.padded_seq_len, *t.shape[2:])
