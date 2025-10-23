"""CUDA-backed cached causal conv1d custom ops and attention descriptor.

This mirrors `torch_backend_causal_conv.py` but reuses existing TRT-LLM CUDA
operators for performance:
- Prefill uses `torch.ops.trtllm.causal_conv1d_fwd`
- Decode uses `torch.ops.trtllm.causal_conv1d_update`

The flattened cached op integrates with the auto_deploy attention interface
and updates a slot-indexed convolution state cache internally.
"""

from typing import List, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BufferInitializerDict,
    CacheConfig,
    CacheInitializerDict,
    Constant,
    MHACallable,
    PrepareMetadataCallable,
    SequenceInfo,
)


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
@torch.library.custom_op("auto_deploy::cuda_causal_conv_prepare_metadata", mutates_args=())
def cuda_causal_conv_prepare_metadata(
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    slot_idx: torch.Tensor,
    page_size: int,
) -> List[torch.Tensor]:
    """Prepare metadata for cached causal conv (CUDA backend).

    Returns a tuple of (seq_len_sanitized, seq_start, slot_idx_sanitized).
    """
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(position_ids, seq_len)
    num_seq = len(seq_len_sanitized)

    seq_start = torch.zeros_like(seq_len_sanitized)
    if num_seq > 1:
        seq_start[1:] = torch.cumsum(seq_len_sanitized[:-1], 0)

    slot_idx_sanitized = slot_idx[:num_seq].clone().to(torch.long)
    # This is only used during prefill to determine if we should use the initial states from the cache.
    use_initial_states = input_pos > 0
    return (seq_len_sanitized, seq_start, slot_idx_sanitized, use_initial_states)


@cuda_causal_conv_prepare_metadata.register_fake
def cuda_causal_conv_prepare_metadata_fake(
    position_ids, seq_len, input_pos, cache_loc, pages_per_seq, slot_idx, page_size
):
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(position_ids, seq_len)
    num_seq = len(seq_len_sanitized)
    return (
        torch.empty_like(seq_len_sanitized),
        torch.empty_like(seq_len_sanitized),
        torch.empty(num_seq, dtype=torch.long, device=slot_idx.device),
        torch.empty(num_seq, dtype=torch.bool, device=slot_idx.device),
    )


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
    use_initial_states: torch.Tensor,  # [num_seq]
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
        has_initial_state = use_initial_states[:num_prefill].to(torch.bool)

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
    use_initial_states: torch.Tensor,  # [num_seq]
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
class CudaBackendCausalConv(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        # Hidden states follow [b, s, c]
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # torch_causal_conv1d signature has 3 relevant tensor arguments
        # TODO: bias can be optional!! How to handle None bias here?
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_causal_conv1d

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.cuda_cached_causal_conv1d

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        # Returns (seq_len, seq_start, slot_idx, use_initial_states)
        return torch.ops.auto_deploy.cuda_causal_conv_prepare_metadata, 4

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        inp_fake: torch.Tensor = source_attn_node.args[0].meta["val"]
        w_fake: torch.Tensor = source_attn_node.args[1].meta["val"]

        in_channels = inp_fake.shape[-1]
        kernel_size = w_fake.shape[-1]

        def _get_conv_cache(si: SequenceInfo):
            return torch.empty(
                si.max_batch_size,
                in_channels,
                max(1, kernel_size - 1),
                device=si.device,
                dtype=cache_config.dtype or inp_fake.dtype,
            )

        return {"conv_state_cache": _get_conv_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        return {}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        stride, padding, dilation, groups, padding_mode = extract_op_args(
            source_attn_node, "stride", "padding", "dilation", "groups", "padding_mode"
        )
        return [stride, padding, dilation, groups, padding_mode]
