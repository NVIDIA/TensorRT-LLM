# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked KDA (Kimi Delta Attention) prefill, reading and writing each sequence's recurrent state in place."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops  # noqa: F401 — registers the op


def kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    varlen_is_aligned: Optional[bool] = None,
    single_sequence_length: Optional[int] = None,
) -> torch.Tensor:
    """Run the gated delta rule over each sequence from `state_pool[state_indices[b]]`; return `o`.

    The final state of sequence b is written back to `state_pool[state_indices[b]]` (fp32, V-first
    `[slots, H, V, K]`). Returns `o` shaped like `v`, a view of the runner's scratch for this batch
    shape: the next call with the same shape overwrites it.
    """
    return torch.ops.trtllm.kda_prefill(
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        state_indices,
        scale,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        A_log,
        dt_bias,
        varlen_is_aligned,
        single_sequence_length,
    )
