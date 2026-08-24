# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for indexed KDA prefill parity tests."""

import torch

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch
from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import reset_recurrent_state_rows

KDAInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
GateParams = tuple[torch.Tensor, torch.Tensor]


def run_indexed_prefill(
    dispatch: KDAKernelDispatch,
    gate_params: GateParams,
    inputs: KDAInputs,
    cu_seqlens: torch.Tensor | None,
    *,
    initial_state: torch.Tensor | None = None,
    state_pool: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    lower_bound: float = -5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run optimized prefill against a pool and return its selected rows."""
    assert dispatch.prefill_kernel_path == "optimized"
    q, k, v, g, beta = inputs
    a_log, dt_bias = gate_params
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    pool_was_provided = state_pool is not None

    if state_indices is None:
        state_indices = torch.arange(num_sequences, dtype=torch.int32, device=q.device)
    state_indices_long = state_indices.long()
    if state_pool is None:
        state_pool = torch.full(
            (num_sequences, q.shape[-2], v.shape[-1], k.shape[-1]),
            float("nan"),
            dtype=torch.float32,
            device=q.device,
        )
    if initial_state is not None:
        state_pool.index_copy_(0, state_indices_long, initial_state)
    if has_initial_states is None:
        has_initial_states = torch.full(
            (num_sequences,),
            initial_state is not None or pool_was_provided,
            dtype=torch.bool,
            device=q.device,
        )

    assert dispatch.can_use_indexed_prefill(
        state_pool=state_pool,
        state_indices=state_indices,
        has_initial_states=has_initial_states,
        cu_seqlens=cu_seqlens,
        num_sequences=num_sequences,
        num_tokens=q.shape[1],
    )
    reset_recurrent_state_rows(state_pool, state_indices, has_initial_states)
    output, final_state = dispatch.prefill_chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=a_log,
        dt_bias=dt_bias,
        scale=k.shape[-1] ** -0.5,
        initial_state=None,
        safe_gate=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        state_pool=state_pool,
        state_indices=state_indices,
    )
    assert final_state is None
    selected_state = state_pool.index_select(0, state_indices_long)
    return output, selected_state, state_pool


def run_fla_prefill(
    dispatch: KDAKernelDispatch,
    gate_params: GateParams,
    inputs: KDAInputs,
    cu_seqlens: torch.Tensor | None,
    *,
    initial_state: torch.Tensor | None = None,
    lower_bound: float = -5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the independent FLA state path."""
    assert dispatch.prefill_kernel_path == "fla"
    q, k, v, g, beta = inputs
    a_log, dt_bias = gate_params
    output, final_state = dispatch.prefill_chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=a_log,
        dt_bias=dt_bias,
        scale=k.shape[-1] ** -0.5,
        initial_state=initial_state.clone() if initial_state is not None else None,
        safe_gate=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    assert final_state is not None
    return output, final_state


def assert_kda_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Apply the KDA numerical parity thresholds."""
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    assert cosine > 0.999 and relative_l2 < 3e-2, (
        f"{name}: cosine={cosine:.6f}, relative_l2={relative_l2:.3e}"
    )
