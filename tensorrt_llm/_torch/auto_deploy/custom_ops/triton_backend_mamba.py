from typing import List

import torch

# Triton kernels
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

from .attention_interface import AttentionRegistry, MHACallable
from .torch_backend_mamba import TorchBackendSSM


@torch.library.custom_op("auto_deploy::triton_cached_ssm_transform", mutates_args={})
def _triton_cached_ssm_transform(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # METADATA
    seq_len: torch.Tensor,  # [num_seq]
    seq_start: torch.Tensor,  # [num_seq]
    slot_idx: torch.Tensor,  # [num_seq]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
) -> torch.Tensor:
    """Flattened cached SSM transform op that respects slot-indexed state caches.

    Split mixed batches into prefill (seq_len>1) and decode (seq_len==1):
    - Prefill: run one varlen combined scan over concatenated prefill tokens and update final states per slot.
    - Decode: batch single-token updates with selective_state_update and update states per slot.
    """
    b, s = hidden_states.shape[:2]
    num_seq = seq_len.shape[0]

    # Flatten tokens for indexing/scatter
    bs = b * s
    device = hidden_states.device
    hs_flat = hidden_states.reshape(bs, *hidden_states.shape[2:])  # [bs, H, D]
    B_flat = B.reshape(bs, *B.shape[2:])  # [bs, G, N]
    C_flat = C.reshape(bs, *C.shape[2:])  # [bs, G, N]
    dt_flat = dt.reshape(bs, dt.shape[2])  # [bs, H]

    y = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    y_flat = y.view(bs, *y.shape[2:])

    num_heads = hidden_states.shape[2]
    head_dim = hidden_states.shape[3]
    ssm_state_size = B.shape[3]

    if s == 1:
        num_prefill = 0
        num_decode = num_seq
    else:
        prefill_mask = seq_len > 1
        num_prefill = int(prefill_mask.sum().item())
        num_decode = num_seq - num_prefill

    # Prefill: concatenate tokens at the front and run combined scan
    if num_prefill > 0:
        seq_len_prefill = seq_len[:num_prefill].to(torch.int32)
        total_prefill_tokens = int(seq_len_prefill.sum().item())
        prefill_idx = torch.arange(total_prefill_tokens, device=device, dtype=torch.long)

        hs_prefill = hs_flat.index_select(0, prefill_idx).unsqueeze(0)  # [1, S_p, H, D]
        B_prefill = B_flat.index_select(0, prefill_idx).unsqueeze(0)  # [1, S_p, G, N]
        C_prefill = C_flat.index_select(0, prefill_idx).unsqueeze(0)  # [1, S_p, G, N]
        dt_prefill = dt_flat.index_select(0, prefill_idx).unsqueeze(0)  # [1, S_p, H]

        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(seq_len_prefill, dim=0),
            ],
            dim=0,
        )
        seq_ids = torch.arange(num_prefill, device=device, dtype=torch.int32)
        seq_idx_prefill = torch.repeat_interleave(seq_ids, seq_len_prefill).view(1, -1)

        y_prefill, varlen_states = mamba_chunk_scan_combined(
            hs_prefill,
            dt_prefill,
            A,
            B_prefill,
            C_prefill,
            chunk_size=chunk_size,
            D=D,
            z=None,
            dt_bias=dt_bias,
            initial_states=None,
            seq_idx=seq_idx_prefill,
            chunk_indices=None,
            chunk_offsets=None,
            cu_seqlens=cu_seqlens,
            dt_softplus=True,
            dt_limit=(time_step_limit[0], time_step_limit[1]),
            return_final_states=False,
            return_varlen_states=True,
        )

        y_flat.index_copy_(0, prefill_idx, y_prefill[0].to(y_flat.dtype))
        ssm_state_cache.index_copy_(
            0, slot_idx[:num_prefill].to(torch.long), varlen_states.to(ssm_state_cache.dtype)
        )

    # Decode: batch single-token updates via selective_state_update
    if num_decode > 0:
        decode_idx = seq_start[num_prefill:].to(torch.long)
        slot_idx_decode = slot_idx[num_prefill:].to(torch.long)

        x_decode = hs_flat.index_select(0, decode_idx)  # [nd, H, D]
        B_decode = B_flat.index_select(0, decode_idx)  # [nd, G, N]
        C_decode = C_flat.index_select(0, decode_idx)  # [nd, G, N]
        dt_decode = dt_flat.index_select(0, decode_idx)  # [nd, H]

        dt_hp = dt_decode[:, :, None].expand(-1, num_heads, head_dim)
        dt_bias_hp = dt_bias[..., None].expand(num_heads, head_dim)
        dt_pre = torch.nn.functional.softplus(dt_hp + dt_bias_hp.to(dtype=dt_hp.dtype))
        dt_pre = torch.clamp(dt_pre, time_step_limit[0], time_step_limit[1])
        A_full = A[..., None, None].expand(num_heads, head_dim, ssm_state_size)
        D_full = D[..., None].expand(num_heads, head_dim)

        dt_bias_zero = torch.zeros_like(dt_bias_hp)
        y_dec = selective_state_update(
            ssm_state_cache,
            x_decode,
            dt_pre,
            A_full,
            B_decode,
            C_decode,
            D=D_full,
            z=None,
            dt_bias=dt_bias_zero,
            dt_softplus=False,
            state_batch_indices=slot_idx_decode,
        )  # [nd, H, D]

        y_flat.index_copy_(0, decode_idx, y_dec.to(y_flat.dtype))

    return y


@_triton_cached_ssm_transform.register_fake
def _triton_cached_ssm_transform_fake(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # METADATA
    seq_len: torch.Tensor,  # [num_seq]
    seq_start: torch.Tensor,  # [num_seq]
    slot_idx: torch.Tensor,  # [num_seq]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
):
    # Return a correctly-shaped tensor for tracing with fake tensors
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


## Note: we reuse the existing metadata custom op and its registered fake from torch backend.


@AttentionRegistry.register("triton_ssm")
class TritonBackendSSM(TorchBackendSSM):
    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_cached_ssm_transform
