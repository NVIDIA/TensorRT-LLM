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

"""Fused shared-expert output combine for Qwen3-Next style MoE blocks.

The model originally computes (per-token):

    y = F.sigmoid(gate_proj(x)) * shared_expert(x)
    final_hidden_states = final_hidden_states + y

which produces three separate pointwise kernels (sigmoid, broadcast-mul, add)
in a PyTorch trace. This module provides a single Triton kernel that fuses
all three into one pass, writing into an optional output buffer or updating
`final_hidden_states` in-place.
"""

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def _sigmoid_gate_mul_add_kernel(
    out_ptr,
    final_ptr,
    shared_ptr,
    gate_ptr,
    out_stride_row,
    final_stride_row,
    shared_stride_row,
    HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    col_block = tl.program_id(axis=1)

    # Per-row gate scalar, sigmoid applied in fp32 for precision.
    g = tl.load(gate_ptr + row).to(tl.float32)
    s = tl.sigmoid(g)

    out_base = row * out_stride_row
    final_base = row * final_stride_row
    shared_base = row * shared_stride_row
    offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN
    # `final` and `shared` are read once and never reused — evict from L2 early
    # to keep room for surrounding allreduce / attention tiles.
    f = tl.load(final_ptr + final_base + offsets, mask=mask, eviction_policy="evict_first").to(
        tl.float32
    )
    sh = tl.load(shared_ptr + shared_base + offsets, mask=mask, eviction_policy="evict_first").to(
        tl.float32
    )
    result = f + s * sh
    tl.store(out_ptr + out_base + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def fused_sigmoid_gate_mul_add(
    final_hidden_states: torch.Tensor,
    gate_logits: torch.Tensor,
    shared_expert_output: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute final + sigmoid(gate) * shared with one Triton kernel.

    Args:
        final_hidden_states: [num_tokens, hidden].
        gate_logits: [num_tokens, 1] or [num_tokens] per-token scalar (pre-sigmoid).
        shared_expert_output: [num_tokens, hidden], same dtype as final_hidden_states.
        output: Optional output buffer. If omitted, final_hidden_states is updated in-place.

    Returns:
        The output tensor.
    """
    assert final_hidden_states.dim() == 2, (
        f"final_hidden_states must be 2D, got {final_hidden_states.shape}"
    )
    assert shared_expert_output.shape == final_hidden_states.shape, (
        f"shape mismatch: final={final_hidden_states.shape} shared={shared_expert_output.shape}"
    )
    assert final_hidden_states.dtype == shared_expert_output.dtype, (
        f"dtype mismatch: final={final_hidden_states.dtype} shared={shared_expert_output.dtype}"
    )
    num_tokens = final_hidden_states.shape[0]
    assert gate_logits.numel() == num_tokens, (
        f"gate numel {gate_logits.numel()} must match num_tokens {num_tokens}"
    )
    if output is None:
        output = final_hidden_states
    else:
        assert output.shape == final_hidden_states.shape, (
            f"output shape mismatch: output={output.shape} final={final_hidden_states.shape}"
        )
        assert output.dtype == final_hidden_states.dtype, (
            f"output dtype mismatch: output={output.dtype} final={final_hidden_states.dtype}"
        )

    if num_tokens == 0:
        return output

    # The kernel only requires the last dimension to be contiguous (stride==1).
    # `allocate_output` may return symmetric-heap buffers whose row stride is
    # padded beyond `hidden`, so do NOT require fully-contiguous tensors.
    assert output.stride(-1) == 1, (
        f"output last-dim must be contiguous, got stride={output.stride()}"
    )
    if final_hidden_states.stride(-1) != 1:
        final_hidden_states = final_hidden_states.contiguous()
    if shared_expert_output.stride(-1) != 1:
        shared_expert_output = shared_expert_output.contiguous()
    # Make gate dense before reshape — reshape may fail on non-contig tensors.
    gate_flat = gate_logits.contiguous().reshape(-1)

    num_tokens, hidden = final_hidden_states.shape
    # Prefer a single program per row when hidden fits in one block (avoids
    # redundant per-block gate loads and reduces launch grid). Cap at 8192 so
    # we don't hit Triton's per-program register pressure for huge hiddens.
    MAX_BLOCK = 8192
    BLOCK_SIZE = min(triton.next_power_of_2(hidden), MAX_BLOCK)
    BLOCK_SIZE = max(BLOCK_SIZE, 16)  # Triton requires power-of-2 >= 16
    num_col_blocks = triton.cdiv(hidden, BLOCK_SIZE)

    # Memory-bound pointwise — more warps improves coalesced bandwidth.
    if BLOCK_SIZE >= 4096:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 4
    else:
        num_warps = 2

    _sigmoid_gate_mul_add_kernel[(num_tokens, num_col_blocks)](
        output,
        final_hidden_states,
        shared_expert_output,
        gate_flat,
        output.stride(0),
        final_hidden_states.stride(0),
        shared_expert_output.stride(0),
        HIDDEN=hidden,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=3,
    )
    return output
