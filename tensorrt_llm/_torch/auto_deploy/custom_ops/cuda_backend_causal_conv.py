"""CUDA-backed cached causal conv1d custom ops and attention descriptor.

This mirrors `torch_backend_causal_conv.py` but reuses existing TRT-LLM CUDA
operators for performance:
- Prefill uses `torch.ops.trtllm.causal_conv1d_fwd`
- Decode uses `torch.ops.trtllm.causal_conv1d_update`

The flattened cached op integrates with the auto_deploy attention interface
and updates a slot-indexed convolution state cache internally.
"""

from typing import Optional

import torch
from torch.fx import Node

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from .attention_interface import AttentionRegistry, CacheConfig, CacheHandlerDict, MHACallable
from .torch_backend_causal_conv import TorchBackendCausalConv


def _build_conv_state_from_sequence(input_bt_c: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Builds a convolution state of fixed window `kernel_size` from a sequence.

    input_bt_c: [B, T, C]
    Returns: [B, C, K]
    """
    # [B, T, C] -> [B, C, T]
    input_b_c_t = input_bt_c.transpose(1, 2)
    seq_len = input_b_c_t.shape[-1]
    if seq_len >= kernel_size:
        return input_b_c_t[..., -kernel_size:]
    pad_amount = kernel_size - seq_len
    # F.pad last dim (time) with (pad_left, pad_right)
    return torch.nn.functional.pad(input_b_c_t, (pad_amount, 0))


# ---------------------------------------------------------------
# Metadata + flattened cached op that integrates with the AD i/f
# ---------------------------------------------------------------


@torch.library.custom_op("auto_deploy::cuda_cached_causal_conv1d", mutates_args={})
def _cuda_cached_causal_conv1d(
    # INPUTS (dense but may be flattened across sequences)
    input: torch.Tensor,  # [b, s, c_in]
    weight: torch.Tensor,  # [c_out, c_in/groups, k] but we expect depthwise use: [c_in, k]
    bias: Optional[torch.Tensor],
    # METADATA
    seq_len: torch.Tensor,  # [num_seq]
    seq_start: torch.Tensor,  # [num_seq]
    slot_idx: torch.Tensor,  # [num_seq]
    # CACHES
    conv_state_cache: torch.Tensor,  # [max_batch_size, c_in, k-1]
    # CONSTANTS
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
) -> torch.Tensor:
    """Flattened cached causal conv that respects slot-indexed state caches (CUDA backend).

    Supports two layouts from the attention interface:
    - Generate-only: input is [b, 1, c_in]. We'll gather caches using slot_idx[:b].
    - Flattened context/mixed: input is [1, total_s, c_in] and seq_len/seq_start
      describe per-sequence segments. We'll process each segment and scatter final states to caches.
    """
    b, s = input.shape[:2]
    num_seq = seq_len.shape[0]

    # Split by lengths: assume prefills first, decodes after
    if s == 1:
        num_prefill = 0
        num_decode = num_seq
    else:
        prefill_mask = seq_len > 1
        num_prefill = int(prefill_mask.sum().item())
        num_decode = num_seq - num_prefill

    # Flatten tokens
    bs = b * s
    inp_flat = input.reshape(bs, *input.shape[2:])  # [total_s, C_in]
    y = torch.empty(b, s, weight.shape[0], device=input.device, dtype=input.dtype)
    y_flat = y.view(bs, *y.shape[2:])

    # Prepare weight as [dim, width] (depthwise)
    if weight.ndim == 3:
        assert weight.shape[-2] == 1
        w2d = weight.squeeze(-2)
    else:
        w2d = weight

    total_prefill_tokens = 0

    # PREFILL: concatenate all prefill tokens and run one varlen forward
    if num_prefill > 0:
        seq_len_prefill = seq_len[:num_prefill].to(torch.int32)
        total_prefill_tokens = int(seq_len_prefill.sum().item())

        # x_varlen: (dim, cu_seq_len)
        x_varlen = inp_flat[:total_prefill_tokens].transpose(0, 1).contiguous()

        # Metadata
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=input.device),
                torch.cumsum(seq_len_prefill, dim=0, dtype=torch.int32),
            ],
            dim=0,
        ).contiguous()
        cache_indices = slot_idx[:num_prefill].to(torch.int32).contiguous()
        has_initial_state = torch.zeros(num_prefill, dtype=torch.bool, device=input.device)

        # Run varlen conv; updates conv_state_cache in-place per cache_indices
        y_varlen = causal_conv1d_fn(
            x_varlen,
            w2d,
            bias,
            query_start_loc=cu_seqlens,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            conv_states=conv_state_cache,
            activation=None,
            pad_slot_id=PAD_SLOT_ID,
        )  # (dim, total_prefill_tokens)

        # Scatter outputs back to y
        y_prefill = y_varlen.transpose(0, 1)  # [total_prefill_tokens, C_out]
        y_flat[:total_prefill_tokens].copy_(y_prefill.to(y_flat.dtype))

    # DECODE: batch update for single-token sequences
    if num_decode > 0:
        # Use true start offsets for decode tokens (tail after prefills)
        decode_idx = seq_start[num_prefill:].to(torch.long)
        x_decode = inp_flat.index_select(0, decode_idx)  # [num_decode, C_in]

        y_dec = causal_conv1d_update(
            x_decode,  # [batch, dim]
            conv_state_cache,
            w2d,
            bias,
            activation=None,
            cache_seqlens=None,
            conv_state_indices=slot_idx[num_prefill:].to(torch.int32),
            pad_slot_id=PAD_SLOT_ID,
        )

        if y_dec.dim() == 3:
            y_dec = y_dec.squeeze(-1)
        y_flat.index_copy_(0, decode_idx, y_dec.to(y_flat.dtype))

    # Custom op must not return an alias of any input; return a fresh tensor
    return y.contiguous().clone()


@_cuda_cached_causal_conv1d.register_fake
def _cuda_cached_causal_conv1d_fake(
    # INPUTS
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    # METADATA
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    slot_idx: torch.Tensor,
    # CACHES
    conv_state_cache: torch.Tensor,
    # CONSTANTS
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
):
    return torch.empty(
        input.shape[0], input.shape[1], weight.shape[0], device=input.device, dtype=input.dtype
    )


@AttentionRegistry.register("cuda_causal_conv")
class CudaBackendCausalConv(TorchBackendCausalConv):
    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.cuda_cached_causal_conv1d

    @classmethod
    def get_cache_handlers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheHandlerDict:
        handlers = super().get_cache_handlers(source_attn_node, cache_config)
        for handler in handlers.values():
            handler.kernel_size = max(1, handler.kernel_size - 1)
        return handlers
