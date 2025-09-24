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

from ..utils.node_utils import extract_op_args
from .attention_interface import (
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


def _cuda_causal_conv1d_prefill(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefill path using TRT-LLM forward kernel; returns (y, conv_state[K-1])."""
    assert padding_mode == "zeros", "padding_mode must be zeros"
    # Shapes: convert input to [B, C, T]
    x_b_c_t = input.transpose(1, 2).contiguous()
    k = weight.shape[-1]
    # Weight to [C, K]
    w2d = weight.squeeze(1) if weight.ndim == 3 else weight
    w2d = w2d.contiguous()
    # Initialize state [B, C, K-1] to zeros
    conv_state = torch.zeros(
        x_b_c_t.shape[0], x_b_c_t.shape[1], k - 1, device=x_b_c_t.device, dtype=x_b_c_t.dtype
    )
    # Run TRT forward (in-place on x_b_c_t and conv_state)
    torch.ops.trtllm.causal_conv1d_fwd(
        x_b_c_t, w2d, bias, conv_state, None, None, None, False, PAD_SLOT_ID
    )
    y = x_b_c_t.transpose(1, 2)
    return y, conv_state


def _cuda_causal_conv1d_decode(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
    conv_state_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode path using TRT-LLM update kernel for last-step output and cache update.

    Returns (y, updated_conv_state) where y: [B, 1, C_out] and updated state: [B, C_in, K].
    """
    assert padding_mode == "zeros", "padding_mode must be zeros"
    # For cached decode we currently support stride=1 and dilation=1 (standard causal conv)
    assert stride == 1, "cached causal conv1d currently supports stride == 1 only"
    assert dilation == 1, "cached causal conv1d currently supports dilation == 1 only"

    batch_size, seq_len, _ = input.shape
    assert seq_len == 1, "decode path expects seq_len == 1"

    kernel_size = weight.shape[-1]
    # TRT update expects state len >= K-1
    assert conv_state_cache.shape[-1] >= kernel_size - 1, (
        "conv_state_cache's last dim must be >= kernel_size - 1"
    )

    # TRT-LLM update kernel expects depthwise form: weight [dim, width], groups == dim
    in_channels = input.shape[-1]
    assert groups == in_channels, (
        "cuda cached causal conv decode currently supports depthwise conv with groups == in_channels"
    )
    # Convert weight to [dim, width]
    if weight.ndim == 3:
        # Expect [C_out, C_in/groups, K] with C_in/groups == 1
        assert weight.shape[-2] == 1 and weight.shape[0] == in_channels, (
            "expected depthwise weight with shape [C, 1, K] matching input channels"
        )
        weight_2d = weight.squeeze(-2)
    elif weight.ndim == 2:
        weight_2d = weight
        assert weight_2d.shape[0] == in_channels, (
            "weight rows must match input channels for depthwise conv"
        )
    else:
        raise AssertionError("unsupported weight rank for causal conv update; expected 2D or 3D")

    # Prepare buffers for TRT-LLM update kernel call.
    # TRT-LLM update kernel signature (Python):
    # torch.ops.trtllm.causal_conv1d_update(x, conv_state, weight, bias,
    #                                       activation_val, cache_seqlens,
    #                                       conv_state_indices, pad_slot_id)
    # We set activation to None and other optional args to None to get a plain linear conv.
    # Convert input to [B, C, T]
    x_b_c_t = input.transpose(1, 2).contiguous()
    updated_cache = conv_state_cache.clone()
    torch.ops.trtllm.causal_conv1d_update(
        x_b_c_t, updated_cache, weight_2d, bias, False, None, None, PAD_SLOT_ID
    )
    y = x_b_c_t.transpose(1, 2)
    return y, updated_cache


# ---------------------------------------------------------------
# Metadata + flattened cached op that integrates with the AD i/f
# ---------------------------------------------------------------


@torch.library.custom_op("auto_deploy::cuda_causal_conv_prepare_metadata", mutates_args=())
def cuda_causal_conv_prepare_metadata(
    input_ids: torch.Tensor,
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
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(input_ids, seq_len)
    num_seq = len(seq_len_sanitized)

    seq_start = torch.zeros_like(seq_len_sanitized)
    if num_seq > 1:
        seq_start[1:] = torch.cumsum(seq_len_sanitized[:-1], 0)

    slot_idx_sanitized = slot_idx[:num_seq].clone().to(torch.long)

    return (seq_len_sanitized, seq_start, slot_idx_sanitized)


@cuda_causal_conv_prepare_metadata.register_fake
def cuda_causal_conv_prepare_metadata_fake(
    input_ids, position_ids, seq_len, input_pos, cache_loc, pages_per_seq, slot_idx, page_size
):
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(input_ids, seq_len)
    num_seq = len(seq_len_sanitized)
    return (
        torch.empty_like(seq_len_sanitized),
        torch.empty_like(seq_len_sanitized),
        torch.empty(num_seq, dtype=torch.long, device=slot_idx.device),
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

    if s == 1:
        # Generate-only batch
        slot_idx_long = slot_idx.to(torch.long)
        cache_batch = conv_state_cache.index_select(0, slot_idx_long)

        y, updated_state = _cuda_causal_conv1d_decode(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
            cache_batch,
        )

        conv_state_cache.index_copy_(0, slot_idx_long, updated_state.to(conv_state_cache.dtype))
        # Custom op must not return an alias of any input; return a fresh tensor
        return y.to(input.dtype).contiguous().clone()

    # Context/mixed phase (flattened sequences)
    bs = b * s
    flat_idx = torch.arange(bs, device=input.device, dtype=torch.long)

    inp_flat = input.reshape(bs, *input.shape[2:])
    y = torch.empty(b, s, weight.shape[0], device=input.device, dtype=input.dtype)
    y_flat = y.view(bs, *y.shape[2:])

    for i in range(num_seq):
        length_i = seq_len[i]
        if length_i.eq(0):
            continue
        start_i = seq_start[i]
        end_i = start_i + length_i

        mask_i = (flat_idx >= start_i.to(torch.long)) & (flat_idx < end_i.to(torch.long))
        idx_i = torch.nonzero(mask_i, as_tuple=False).squeeze(-1)

        inp_seq = inp_flat.index_select(0, idx_i).unsqueeze(0)

        y_seq, conv_state_i = _cuda_causal_conv1d_prefill(
            inp_seq,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
        )

        y_flat.index_copy_(0, idx_i, y_seq[0].to(y_flat.dtype))

        slot_i = slot_idx[i].to(torch.long).unsqueeze(0)
        conv_state_cache.index_copy_(0, slot_i, conv_state_i.to(conv_state_cache.dtype))

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
        # Returns (seq_len, seq_start, slot_idx)
        return torch.ops.auto_deploy.cuda_causal_conv_prepare_metadata, 3

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
