"""Custom op collection for cached causal conv1d in pure PyTorch.

This mirrors the structure used by the cached Mamba/SSM ops:
- clean functional interface identical to the uncached op plus cache argument at the end
- prefill vs decode handled internally
- cache read/write handled internally

In addition, this file provides an AttentionDescriptor-compatible flattened cached op
and a metadata op to integrate with the auto_deploy attention interface.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    CausalConvResourceHandler,
    Constant,
    MHACallable,
    ResourceHandlerDict,
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
    return F.pad(input_b_c_t, (pad_amount, 0))


def _torch_causal_conv1d_prefill(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefill path: compute full conv and produce initial cache state.

    Returns (y, conv_state) where y: [B, T, C_out] and conv_state: [B, C_in, K].
    """
    assert padding_mode == "zeros", "padding_mode must be zeros"

    batch_size, seq_len, _ = input.shape
    # Reuse the uncached op for the actual convolution
    y = torch.ops.auto_deploy.torch_causal_conv1d(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        padding_mode,
    )

    kernel_size = weight.shape[-1]
    conv_state = _build_conv_state_from_sequence(input, kernel_size)
    return y, conv_state


def _torch_causal_conv1d_decode(
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
    """Decode path: update cache with the latest token and compute the last output.

    Returns (y, updated_conv_state) where y: [B, 1, C_out] and updated state: [B, C_in, K].
    """
    assert padding_mode == "zeros", "padding_mode must be zeros"
    # For cached decode we currently support stride=1 and dilation=1 (standard causal conv)
    # This mirrors the original decode implementation assumptions for Bamba.
    assert stride == 1, "cached causal conv1d currently supports stride == 1 only"
    assert dilation == 1, "cached causal conv1d currently supports dilation == 1 only"

    batch_size, seq_len, _ = input.shape
    assert seq_len == 1, "decode path expects seq_len == 1"

    kernel_size = weight.shape[-1]
    assert conv_state_cache.shape[-1] == kernel_size, (
        "conv_state_cache's last dim must equal kernel_size"
    )

    # Update cache in-place: roll left and place the new element at the end
    updated_cache = conv_state_cache.roll(shifts=-1, dims=-1)
    # [B, T=1, C] -> [B, C]
    new_sample_bc = input.transpose(1, 2)[..., 0]
    updated_cache[:, :, -1] = new_sample_bc.to(updated_cache.dtype).to(updated_cache.device)

    # Compute output for the current step using the cache window.
    # Convert cache window [B, C_in, K] -> short sequence [B, K, C_in]
    window_seq = updated_cache.transpose(1, 2).to(device=weight.device, dtype=weight.dtype)
    # Reuse the uncached op with padding=0, stride=1, dilation=1 so output length == 1
    y = torch.ops.auto_deploy.torch_causal_conv1d(
        window_seq,
        weight,
        bias,
        1,  # stride
        0,  # padding
        1,  # dilation
        groups,
        padding_mode,
    )

    return y, updated_cache


# ---------------------------------------------------------------
# Metadata + flattened cached op that integrates with the AD i/f
# ---------------------------------------------------------------


# TODO(https://github.com/NVIDIA/TensorRT-LLM/issues/8170): update torch
# reference implementation to support chunked prefill.
# Returns (seq_len, seq_start, slot_idx)
@torch.library.custom_op("auto_deploy::torch_cached_causal_conv1d", mutates_args={})
def _torch_cached_causal_conv1d(
    # INPUTS (dense but may be flattened across sequences)
    input: torch.Tensor,  # [b, s, c_in]
    weight: torch.Tensor,  # [c_out, c_in/groups, k]
    bias: Optional[torch.Tensor],
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    conv_state_cache: torch.Tensor,  # [max_batch_size, c_in, k]
    # CONSTANTS
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
) -> torch.Tensor:
    """Flattened cached causal conv that respects slot-indexed state caches.

    Supports two layouts from the attention interface:
    - Generate-only: input is [b, 1, c_in]. We'll gather caches using slot_idx[:b].
    - Flattened context/mixed: input is [1, total_s, c_in] and seq_len/seq_start
      describe per-sequence segments. We'll process each segment and scatter final states to caches.
    """
    b, s = input.shape[:2]
    num_seq = seq_len.shape[0]

    # get cleaned up metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    seq_len = seq_len[:num_seq]
    seq_start = cu_seqlen[:num_seq]
    slot_idx = slot_idx[:num_seq].to(torch.long)
    use_initial_states = use_initial_states[:num_seq]

    if s == 1:
        # Generate-only batch
        slot_idx_long = slot_idx.to(torch.long)
        cache_batch = conv_state_cache.index_select(0, slot_idx_long)

        y, updated_state = _torch_causal_conv1d_decode(
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
        return y.to(input.dtype)

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

        y_seq, conv_state_i = _torch_causal_conv1d_prefill(
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

    return y


@_torch_cached_causal_conv1d.register_fake
def _torch_cached_causal_conv1d_fake(
    # INPUTS (dense but may be flattened across sequences)
    input: torch.Tensor,  # [b, s, c_in]
    weight: torch.Tensor,  # [c_out, c_in/groups, k]
    bias: Optional[torch.Tensor],
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    conv_state_cache: torch.Tensor,  # [max_batch_size, c_in, k]
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


@AttentionRegistry.register("torch_causal_conv")
class TorchBackendCausalConv(AttentionDescriptor):
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
        return torch.ops.auto_deploy.torch_cached_causal_conv1d.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        inp_fake: torch.Tensor = source_attn_node.args[0].meta["val"]
        w_fake: torch.Tensor = source_attn_node.args[1].meta["val"]

        in_channels = inp_fake.shape[-1]
        kernel_size = w_fake.shape[-1]

        # NOTE: torch backend stores kernel_size elements in state (full conv window).
        # CausalConvResourceHandler.state_shape = (conv_dim, d_conv - 1), so d_conv = kernel_size + 1.
        return {
            "conv_state_cache": CausalConvResourceHandler(
                conv_dim=in_channels,
                d_conv=kernel_size + 1,  # state_shape[-1] = d_conv - 1 = kernel_size
                dtype=cls.resolve_cache_dtype("auto", inp_fake.dtype),
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        stride, padding, dilation, groups, padding_mode = extract_op_args(
            source_attn_node, "stride", "padding", "dilation", "groups", "padding_mode"
        )
        # None is for activation parameter, which may not exist in the source node (added by fusion later)
        return [stride, padding, dilation, groups, padding_mode, None]
