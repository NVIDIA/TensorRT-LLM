"""Triton-backed custom ops for Bamba SSM transforms.

These ops mirror the signatures of the original torch reference ops defined in
`tensorrt_llm._torch.auto_deploy.models.patches.bamba` but dispatch to fast Triton
implementations from the Mamba Triton modules (`ssd_combined` and `selective_state_update`).
"""

from typing import List, Tuple

import torch

from tensorrt_llm._torch.modules.mamba.selective_state_update import \
    selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import \
    mamba_chunk_scan_combined


@torch.library.custom_op("auto_deploy::ssm_transform_triton", mutates_args={})
def ssm_transform_triton(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_size: int,
    # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    time_step_limit: List[float],
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    n_groups: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of uncached SSM transform.

    Shapes:
      - hidden_states: [batch_size, seq_len, num_heads * head_dim]
      - dt: [batch_size, seq_len, num_heads]
      - A: [num_heads]
      - B, C: [batch_size, seq_len, n_groups * ssm_state_size]
      - D: [num_heads]
    Returns:
      - y: [batch_size, seq_len, num_heads * head_dim]
      - final_states: [batch_size, num_heads, head_dim, ssm_state_size]
    """
    # Prepare tensors
    x = hidden_states.reshape(batch_size, seq_len, num_heads, head_dim)
    B4 = B.reshape(batch_size, seq_len, n_groups, ssm_state_size)
    C4 = C.reshape(batch_size, seq_len, n_groups, ssm_state_size)

    # Call combined Triton kernel
    out, final_states = mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B4,
        C4,
        chunk_size=chunk_size,
        D=D,  # accepts [num_heads] or [num_heads, head_dim]
        z=None,
        dt_bias=dt_bias,
        seq_idx=None,
        dt_softplus=True,
        dt_limit=(time_step_limit[0], time_step_limit[1]),
        return_final_states=True,
    )

    y = out.reshape(batch_size, seq_len, num_heads * head_dim)
    # Ensure dtype matches reference op expectations
    y = y.to(hidden_states.dtype)
    final_states = final_states.to(hidden_states.dtype)
    return y, final_states


@ssm_transform_triton.register_fake
def _(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_size: int,
    time_step_limit: List[float],
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    n_groups: int,
    chunk_size: int,
):
    y_shape = (batch_size, seq_len, num_heads * head_dim)
    state_shape = (batch_size, num_heads, head_dim, ssm_state_size)
    return hidden_states.new_empty(y_shape), hidden_states.new_empty(
        state_shape)


@torch.library.custom_op("auto_deploy::ssm_transform_cached_triton",
                         mutates_args={})
def ssm_transform_cached_triton(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_size: int,
    # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    time_step_limit: List[float],
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    n_groups: int,
    chunk_size: int,
    cached_ssm_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of cached SSM transform (single-token update).

    Shapes:
      - hidden_states: [batch_size, num_heads * head_dim]
      - dt: [batch_size, 1, num_heads]
      - A: [num_heads]
      - B, C: [batch_size, n_groups * ssm_state_size]
      - D: [num_heads]
      - cached_ssm_state: [batch_size, num_heads, head_dim, ssm_state_size]
    Returns:
      - y: [batch_size, 1, num_heads * head_dim]
      - new_state: [batch_size, num_heads, head_dim, ssm_state_size]
    """
    # Expand per-head per-dim parameters
    dt_hp = dt[:, 0, :][:, :, None].expand(batch_size, num_heads, head_dim)
    dt_bias_hp = dt_bias[..., None].expand(num_heads, head_dim)
    A_full = A[..., None, None].expand(num_heads, head_dim, ssm_state_size)
    D_full = D[..., None].expand(num_heads, head_dim)

    B_grouped = B.reshape(batch_size, n_groups, ssm_state_size)
    C_grouped = C.reshape(batch_size, n_groups, ssm_state_size)

    x = hidden_states.reshape(batch_size, num_heads, head_dim)
    # compute new state; avoid mutating input cache
    new_state = cached_ssm_state.clone()
    y_hp = selective_state_update(
        new_state,
        x,
        dt_hp,
        A_full,
        B_grouped,
        C_grouped,
        D=D_full,
        z=None,
        dt_bias=dt_bias_hp,
        dt_softplus=True,
    )
    y = y_hp.reshape(batch_size, -1)[:, None, ...]
    # Ensure dtype matches reference op expectations
    y = y.to(hidden_states.dtype)
    new_state = new_state.to(hidden_states.dtype)
    return y, new_state


@ssm_transform_cached_triton.register_fake
def _(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_size: int,
    time_step_limit: List[float],
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    n_groups: int,
    chunk_size: int,
    cached_ssm_state: torch.Tensor,
):
    y_shape = (batch_size, 1, num_heads * head_dim)
    state_shape = (batch_size, num_heads, head_dim, ssm_state_size)
    return hidden_states.new_empty(y_shape), hidden_states.new_empty(
        state_shape)
