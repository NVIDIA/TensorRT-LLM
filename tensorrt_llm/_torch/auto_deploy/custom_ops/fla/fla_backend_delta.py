"""Cached attention op for delta rule using the fla kernel library.

Delta Rule is based on this paper: https://arxiv.org/abs/2406.06484

Kernels are based on this repo: https://github.com/fla-org/flash-linear-attention
"""

from typing import List

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    StateResourceHandler,
)
from .delta_rule.chunk import chunk_delta_rule_fwd
from .delta_rule.fused_recurrent import fused_recurrent_delta_rule_fwd


@torch.library.custom_op("auto_deploy::fla_cached_delta_rule", mutates_args=())
def fla_cached_delta_rule(
    # INPUTS (dense but may be flattened across sequences)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, H, K, V]
    # CONSTANTS
    scale: float,
) -> torch.Tensor:
    b, s, num_heads, _ = q.shape

    # flatten it
    q_flat = q.view(b * s, num_heads, -1)
    k_flat = k.view(b * s, num_heads, -1)
    v_flat = v.view(b * s, num_heads, -1)
    beta_flat = beta.view(b * s, num_heads)

    # pre-allocate output
    y = torch.empty_like(v, memory_format=torch.contiguous_format)
    y_flat = y.view(b * s, num_heads, -1)

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    # clean up metadata
    cu_seqlen_prefill = cu_seqlen[: num_prefill + 1]
    slot_idx = slot_idx[:num_seq].to(torch.long)
    use_initial_states = use_initial_states[:num_seq]

    if num_prefill > 0:
        initial_states = None
        if torch.any(use_initial_states[:num_prefill]):
            initial_states = torch.where(
                use_initial_states[:num_prefill, None, None, None],
                delta_cache[slot_idx[:num_prefill]],
                0,
            )

        y_prefill, _, final_state = chunk_delta_rule_fwd(
            q=q_flat[None, :num_prefill_tokens],
            k=k_flat[None, :num_prefill_tokens],
            v=v_flat[None, :num_prefill_tokens],
            beta=beta_flat[None, :num_prefill_tokens],
            scale=scale,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlen_prefill,
        )

        y_flat[None, :num_prefill_tokens] = y_prefill.to(y_flat.dtype)
        delta_cache.index_copy_(0, slot_idx[:num_prefill], final_state.to(delta_cache.dtype))

        del y_prefill, initial_states, final_state

    if num_decode > 0:
        # NOTE: avoiding state clone here and adopting the kernel to handle
        # indexed initial states would give a boost
        y_decode, _, final_state = fused_recurrent_delta_rule_fwd(
            q=q_flat[num_prefill_tokens:, None],
            k=k_flat[num_prefill_tokens:, None],
            v=v_flat[num_prefill_tokens:, None],
            beta=beta_flat[num_prefill_tokens:, None],
            scale=scale,
            initial_state=delta_cache[slot_idx[num_prefill:]].clone(),
            output_final_state=True,
        )

        y_flat[num_prefill_tokens:, None] = y_decode.to(y_flat.dtype)
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
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, H, K, V]
    # CONSTANTS
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(v)


@AttentionRegistry.register("fla_delta")
class FlaDeltaBackend(AttentionDescriptor):
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
        return torch.ops.auto_deploy.fla_cached_delta_rule.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        key_node = source_attn_node.args[1]
        value_node = source_attn_node.args[2]
        num_heads = key_node.meta["val"].shape[-2]
        key_dim = key_node.meta["val"].shape[-1]
        value_dim = value_node.meta["val"].shape[-1]
        key_dtype = key_node.meta["val"].dtype

        return {
            "delta_cache": StateResourceHandler(
                num_heads,
                key_dim,
                value_dim,
                # NOTE: not configurable at the moment, using auto to match the key dtype
                dtype=cls.resolve_cache_dtype("auto", key_dtype),
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        scale = extract_op_args(source_attn_node, "scale")[0]
        if scale is None:
            key_node = source_attn_node.args[1]
            key_dim = key_node.meta["val"].shape[-1]
            scale = key_dim**-0.5
        return [scale]
