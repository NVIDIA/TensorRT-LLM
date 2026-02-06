# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Tuple

import torch
import triton

from .triton_rope_kernel import (
    rope_fwd_flattened_kernel,
    rope_fwd_interleaved_kernel,
    rope_fwd_kernel,
)


@torch.library.custom_op("auto_deploy::triton_rope_with_input_pos", mutates_args=())
def apply_rope_with_input_pos(
    x: torch.Tensor, freqs_cis: torch.Tensor, input_pos: torch.Tensor, layout: str
) -> torch.Tensor:
    """Embeds the input using RoPE (https://arxiv.org/abs/2104.09864).

    Supports 6 different layouts of ``x``.:
      - ``'bsnd'``:  (batch, seq_len, n_head, d_head)
      - ``'bnsd'``:  (batch, n_head, seq_len, d_head)
      - ``'sbnd'``:  (seq_len, batch, n_head, d_head)
      - ``'snbd'``:  (seq_len, n_head, batch, d_head)
      - ``'nbsd'``:  (n_head, batch, seq_len, d_head)
      - ``'nsbd'``:  (n_head, seq_len, batch, d_head)


    Args:
        x: key or query Tensor to be embedded
        freqs_cis: contains interleaved cos and sin frequencies.
        input_pos: Tensor of size `b` containing the input offsets.
        layout: string of layout above
    """
    assert set(layout) == {"b", "n", "s", "d"}, "invalid layout."
    assert layout[3] == "d"
    assert x.shape[3] % 2 == 0, "RoPE requires an even number as hidden size."

    y = torch.empty_like(x)

    batch_dim = layout.find("b")
    seq_dim = layout.find("s")
    nhead_dim = layout.find("n")
    if input_pos is None:
        input_pos = torch.tensor([0] * x.shape[batch_dim], device=x.device, dtype=torch.int32)
    N = x.shape[batch_dim]
    L = x.shape[seq_dim]
    H = x.shape[nhead_dim]
    D = x.shape[3]

    stride_n = x.stride(batch_dim)
    stride_l = x.stride(seq_dim)
    stride_h = x.stride(nhead_dim)
    stride_d = x.stride(3)

    BLOCK_SIZE_H = 32
    BLOCK_SIZE_L = min(triton.next_power_of_2(L), 32)
    grid = (
        N,
        triton.cdiv(H, BLOCK_SIZE_H),
        triton.cdiv(L, BLOCK_SIZE_L),
    )
    rope_fwd_kernel[grid](
        x,
        input_pos,
        freqs_cis,
        y,
        N,
        L,
        H,
        D,
        stride_n,
        stride_l,
        stride_h,
        stride_d,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    return y


@apply_rope_with_input_pos.register_fake
def apply_rope_with_input_pos_fake(x, freqs_cis, input_pos, layout):
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::triton_rope_on_flattened_inputs", mutates_args=())
def apply_rope_on_flattened_inputs(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    input_pos: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_start_indices: torch.Tensor,
) -> torch.Tensor:
    """Embeds the input using RoPE (https://arxiv.org/abs/2104.09864).

    Assumes input is flattened as [B*S, N, D]m

    Args:
        x: key or query Tensor to be embedded
        freqs_cis: contains interleaved cos and sin frequencies.
        input_pos: Tensor of size `B` containing the input offsets.
        seq_lens: Tensor of size `B` containing the length of sequences.
        seq_start_indices: Tensor of size `B` containing the start indices of sequences in the
            flattened representation.
    """
    y = torch.empty_like(x)

    B = len(input_pos)  # number of sequences
    assert seq_start_indices.shape[0] == seq_lens.shape[0]

    H = x.shape[1]
    D = x.shape[2]

    L = seq_lens.max().item()

    BLOCK_SIZE_H = 32
    BLOCK_SIZE_L = min(max(triton.next_power_of_2(L), 1), 32)
    grid = (
        B,
        triton.cdiv(H, BLOCK_SIZE_H),
        triton.cdiv(L, BLOCK_SIZE_L),
    )
    rope_fwd_flattened_kernel[grid](
        x,
        seq_lens,
        seq_start_indices,
        input_pos,
        freqs_cis,
        y,
        H,
        D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    return y


@apply_rope_on_flattened_inputs.register_fake
def apply_rope_on_flattened_inputs_fake(x, freqs_cis, input_pos, seq_lens, seq_start_indices):
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::triton_rope_on_interleaved_qk_inputs", mutates_args=())
def apply_rope_on_interleaved_qk_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RoPE for DeepSeek-style interleaved Q/K inputs.

    This kernel fuses:
    1. Position ID lookup for cos/sin from cache
    2. De-interleaving of Q/K
    3. RoPE application

    Args:
        q: Query tensor with interleaved layout [B, S, H_Q, D]
           Input: [q0_r, q0_i, q1_r, q1_i, ...]
        k: Key tensor with interleaved layout [B, S, H_K, D]
           Typically H_K=1 for MQA
        cos_cache: Cosine cache [max_seq_len, D]
        sin_cache: Sine cache [max_seq_len, D]
        position_ids: Position indices [B, S]

    Returns:
        Tuple of (q_rotated, k_rotated) with layout [B, S, H, D]
        Output: [y0, y1, ..., y_{D/2-1}, y_{D/2}, ..., y_{D-1}]
        where y_first = a*cos - b*sin, y_second = b*cos + a*sin
    """
    assert q.dim() == 4, f"Q must be 4D [B, S, H, D], got {q.dim()}D"
    assert k.dim() == 4, f"K must be 4D [B, S, H, D], got {k.dim()}D"
    assert q.shape[-1] % 2 == 0, "Head dimension must be even"
    assert q.shape[-1] == k.shape[-1], "Q and K must have same head dimension"
    assert cos_cache.shape == sin_cache.shape, "cos and sin cache must have same shape"

    B, S, H_Q, D = q.shape
    _, _, H_K, _ = k.shape
    assert k.shape[0] == B and k.shape[1] == S, "Q and K must have same batch and seq dims"
    assert position_ids.shape == (B, S), f"position_ids must be [B, S], got {position_ids.shape}"
    assert H_Q >= H_K, f"H_Q ({H_Q}) must be >= H_K ({H_K}) for grid sizing"

    # Allocate contiguous outputs
    # The kernel computes contiguous strides internally for output writes
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Block sizes
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_S = min(triton.next_power_of_2(S), 32)

    # Grid: (B, cdiv(H_Q, BLOCK_SIZE_H), cdiv(S, BLOCK_SIZE_S))
    # H_Q >= H_K is enforced above; K heads are masked within each block
    grid = (
        B,
        triton.cdiv(H_Q, BLOCK_SIZE_H),
        triton.cdiv(S, BLOCK_SIZE_S),
    )

    rope_fwd_interleaved_kernel[grid](
        q,
        k,
        cos_cache,
        sin_cache,
        position_ids,
        q_out,
        k_out,
        B,
        S,
        H_Q,
        H_K,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        position_ids.stride(0),
        position_ids.stride(1),
        cos_cache.stride(0),
        cos_cache.stride(1),
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
    )

    return q_out, k_out


@apply_rope_on_interleaved_qk_inputs.register_fake
def apply_rope_on_interleaved_qk_inputs_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k)
