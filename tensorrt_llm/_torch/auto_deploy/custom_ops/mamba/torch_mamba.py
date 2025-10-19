"""Custom op collection for uncached mamba mixer (linear attention)."""

from typing import List, Tuple

import torch
from torch import nn


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] ->
        # [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def _segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0
    )
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def _torch_ssm_prefill(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    time_step_limit: List[
        float
    ],  # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # retrieve some shape information
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    n_groups, ssm_state_size = B.shape[2:]

    # !! This is the uncached code path from the original implementation.
    # begin ssd naive implementation without einsums
    dt = nn.functional.softplus(dt + dt_bias)
    dt = torch.clamp(dt, time_step_limit[0], time_step_limit[1])
    hidden_states = hidden_states.reshape(batch_size, seq_len, num_heads, head_dim).float()
    B = B.reshape(batch_size, seq_len, n_groups, ssm_state_size).float()
    C = C.reshape(batch_size, seq_len, n_groups, ssm_state_size).float()
    B = B.repeat_interleave(num_heads // n_groups, dim=2, output_size=num_heads)
    C = C.repeat_interleave(num_heads // n_groups, dim=2, output_size=num_heads)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    D_residual = D[..., None] * _pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt

    # Rearrange into blocks/chunks
    hidden_states, A, B, C = [
        _reshape_into_chunks(t, pad_size, chunk_size) for t in (hidden_states, A, B, C)
    ]

    # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    # This is the analog of a causal mask
    L = torch.exp(_segment_sum(A))

    # Contraction of C and B to get G (attention-weights like)
    G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
    G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

    # Compute M, equivalent to applying attention mask to weights
    M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_intermediate.sum(dim=-1)

    # Compute Y_diag (apply to values)
    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(_segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, ssm_state = new_states[:, :-1], new_states[:, -1]
    # states = new_states[:, :-1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    y = Y_diag + Y_off
    # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
    y = y.reshape(batch_size, -1, num_heads, head_dim)

    y = y + D_residual
    # Cutting off padded chunks
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    y = y.reshape(batch_size, seq_len, num_heads, head_dim)

    return y, ssm_state


@torch.library.custom_op("auto_deploy::torch_ssm", mutates_args={})
def _torch_ssm(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    time_step_limit: List[
        float
    ],  # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    chunk_size: int,
) -> torch.Tensor:
    y, _ = _torch_ssm_prefill(hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size)
    return y


@_torch_ssm.register_fake
def _torch_ssm_meta(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    time_step_limit: List[float],
    chunk_size: int,
) -> torch.Tensor:
    return torch.empty_like(hidden_states, dtype=torch.float32)
