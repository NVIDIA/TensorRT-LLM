"""Cached attention op for delta rule using the fla kernel library.

Delta Rule is based on this paper: https://arxiv.org/abs/2406.06484

Kernels are based on this repo: https://github.com/fla-org/flash-linear-attention
"""

from typing import List, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

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
from .delta_rule.chunk import chunk_delta_rule_fwd
from .delta_rule.fused_recurrent import fused_recurrent_delta_rule_fwd


@torch.library.custom_op("auto_deploy::fla_delta_prepare_metadata", mutates_args=())
def fla_delta_prepare_metadata(
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    slot_idx: torch.Tensor,
    page_size: int,
    chunk_size: int,
) -> List[torch.Tensor]:
    """Prepare metadata for cached chunked delta rule.

    Returns a tuple of (cu_seq_lens, slot_idx_sanitized, use_initial_states, batch_info_tensor).
    """
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(position_ids, seq_len)
    num_seq = len(seq_len_sanitized)
    cu_seqlens = torch.zeros(num_seq + 2, dtype=torch.int32, device=seq_len_sanitized.device)

    slot_idx_sanitized = slot_idx[:num_seq].clone().to(torch.long)
    use_initial_states = input_pos[:num_seq] > 0

    _, s = position_ids.shape[:2]
    if s > 1:
        prefill_mask = seq_len_sanitized > 1
        num_prefill = int(prefill_mask.sum().item())
        num_prefill_tokens = int(seq_len_sanitized[:num_prefill].sum().item())
        num_decode = num_seq - num_prefill

        # compute cu_seq_lens for the prefill sequences first
        cu_seqlens[1 : num_prefill + 1] = torch.cumsum(seq_len_sanitized[:num_prefill], 0)
    else:
        num_prefill = 0
        num_prefill_tokens = 0
        num_decode = num_seq

    # decode is just arange...
    cu_seqlens[num_prefill + 1 :] = torch.arange(
        num_decode + 1, device=cu_seqlens.device, dtype=cu_seqlens.dtype
    )
    batch_info_tensor = torch.tensor(
        [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32
    )

    return cu_seqlens, slot_idx_sanitized, use_initial_states, batch_info_tensor


@fla_delta_prepare_metadata.register_fake
def fla_delta_prepare_metadata_fake(
    position_ids,
    seq_len,
    input_pos,
    cache_loc,
    pages_per_seq,
    slot_idx,
    page_size,
    chunk_size,
):
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(position_ids, seq_len)
    num_seq = len(seq_len_sanitized)
    cu_seq_lens = torch.empty(num_seq + 2, dtype=torch.int32, device=seq_len_sanitized.device)
    return (
        cu_seq_lens,
        torch.empty(num_seq, dtype=torch.long, device=slot_idx.device),
        torch.empty(num_seq, dtype=torch.bool, device=slot_idx.device),
        torch.empty(3, dtype=torch.int32),  # host tensor
    )


@torch.library.custom_op("auto_deploy::fla_cached_delta_rule", mutates_args=())
def fla_cached_delta_rule(
    # INPUTS (dense but may be flattened across sequences)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    # METADATA
    cu_seqlens: torch.Tensor,  # [num_seq + 1]
    slot_idx: torch.Tensor,  # [num_seq]
    use_initial_states: torch.Tensor,  # [num_seq]
    batch_info_tensor: torch.Tensor,  # [3]
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, H, K, V]
    # CONSTANTS
    scale: float,
) -> torch.Tensor:
    b, s, num_heads, _ = q.shape

    # flatten it
    q_flat = q.view(1, b * s, num_heads, -1)
    k_flat = k.view(1, b * s, num_heads, -1)
    v_flat = v.view(1, b * s, num_heads, -1)
    beta_flat = beta.view(1, b * s, num_heads)

    # pre-allocate output
    y = torch.empty_like(v, memory_format=torch.contiguous_format)
    y_flat = y.view(1, b * s, num_heads, -1)

    num_prefill, num_prefill_tokens, num_decode = batch_info_tensor.tolist()

    if num_prefill > 0:
        initial_states = None
        if torch.any(use_initial_states[:num_prefill]):
            initial_states = torch.where(
                use_initial_states[:num_prefill, None, None, None],
                delta_cache[slot_idx[:num_prefill]],
                0,
            )

        y_prefill, _, final_state = chunk_delta_rule_fwd(
            q=q_flat[:, :num_prefill_tokens],
            k=k_flat[:, :num_prefill_tokens],
            v=v_flat[:, :num_prefill_tokens],
            beta=beta_flat[:, :num_prefill_tokens],
            scale=scale,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens[: num_prefill + 1],
        )

        y_flat[:, :num_prefill_tokens] = y_prefill.to(y_flat.dtype)
        delta_cache.index_copy_(0, slot_idx[:num_prefill], final_state.to(delta_cache.dtype))

        del y_prefill, initial_states, final_state

    if num_decode > 0:
        # NOTE: avoiding state clone here and adopting the kernel to handle
        # indexed initial states would give a boost
        y_decode, _, final_state = fused_recurrent_delta_rule_fwd(
            q=q_flat[:, num_prefill_tokens:],
            k=k_flat[:, num_prefill_tokens:],
            v=v_flat[:, num_prefill_tokens:],
            beta=beta_flat[:, num_prefill_tokens:],
            scale=scale,
            initial_state=delta_cache[slot_idx[num_prefill:]].clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens[num_prefill + 1 :],
        )

        y_flat[:, num_prefill_tokens:] = y_decode.to(y_flat.dtype)
        delta_cache.index_copy_(0, slot_idx[num_prefill:], final_state.to(delta_cache.dtype))

        del y_decode, final_state

    return y


@fla_cached_delta_rule.register_fake
def fla_cached_delta_rule_fake(
    # INPUTS (dense but may be flattened across sequences)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    # METADATA
    cu_seqlens: torch.Tensor,  # [num_seq + 1]
    slot_idx: torch.Tensor,  # [num_seq]
    use_initial_states: torch.Tensor,  # [num_seq]
    batch_info_tensor: torch.Tensor,  # [3]
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, H, K, V]
    # CONSTANTS
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(v)


@AttentionRegistry.register("fla_delta")
class FlaDeltaBackend(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        # TODO: we should refine our notion of "is_paged" --> seems counterintuitive for ssm nows
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # q, k, v, beta
        return 4

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.fla_delta_rule

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.fla_cached_delta_rule

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        # Returns (cu_seq_lens, slot_idx, use_initial_states, batch_info_tensor)
        return torch.ops.auto_deploy.fla_delta_prepare_metadata, 4

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        key_node = source_attn_node.args[1]
        value_node = source_attn_node.args[2]
        num_heads = key_node.meta["val"].shape[-2]
        key_dim = key_node.meta["val"].shape[-1]
        value_dim = value_node.meta["val"].shape[-1]
        key_dtype = key_node.meta["val"].dtype

        def _get_delta_cache(si: SequenceInfo):
            return torch.empty(
                si.max_batch_size,
                num_heads,
                key_dim,
                value_dim,
                device=si.device,
                dtype=cache_config.delta_dtype or key_dtype,
            )

        return {"delta_cache": _get_delta_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        return {}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        scale = extract_op_args(source_attn_node, "scale")[0]
        if scale is None:
            key_node = source_attn_node.args[1]
            key_dim = key_node.meta["val"].shape[-1]
            scale = key_dim**-0.5
        return [scale]
