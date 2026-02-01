"""Custom ops for MHA/XQA attention."""

import math
from typing import List, Optional

import torch
import triton
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from ....llmapi.llm_args import KvCacheConfig
from ..utils.logger import ad_logger
from ..utils.node_utils import extract_op_args
from .attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    UnpagedResourceHandler,
)
from .triton_kernels.attention_with_kv_cache import (
    attention_kv_stage2,
    context_attention_kv_flattened,
    gqa_attention_kv_stage1,
    update_kv_cache,
)


def _decode_attention(
    q: torch.Tensor,  # [num_decode, num_heads, qk_head_dim]
    k: torch.Tensor,  # [num_decode, num_kv_heads, qk_head_dim]
    v: torch.Tensor,  # [num_decode, num_kv_heads, v_head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_idx: torch.Tensor,  # [num_decode]
    input_pos: torch.Tensor,  # [num_decode]
    scale: float,
    out: torch.Tensor,  # [num_decode, num_heads, v_head_dim]
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
) -> None:
    """Handle decode phase - single token generation attention."""
    num_decode = q.shape[0]
    n_heads, q_d_head = q.shape[-2:]
    max_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]
    device = q.device

    SEQ_BLOCK_SIZE = 64
    num_blocks = (max_seq_len + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE
    stage1_output_values = torch.empty(
        num_decode, n_heads, num_blocks, v_d_head, device=device, dtype=torch.float32
    )
    stage1_output_logsumexp = torch.empty(
        num_decode, n_heads, num_blocks, device=device, dtype=torch.float32
    ) - float("inf")

    update_kv_cache[(num_decode, n_kv_heads, 1)](
        k,
        v,
        None,
        None,
        k_cache,
        v_cache,
        input_pos,
        slot_idx,
        max_seq_len,
        n_kv_heads,
        q_d_head,
        v_d_head,
        1,
        GENERATE_ONLY=True,
    )

    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))
    gqa_attention_kv_stage1[
        (
            num_decode,
            n_kv_heads,
            num_blocks,
        )
    ](
        q,
        k_cache,
        v_cache,
        slot_idx,
        input_pos,
        stage1_output_values,
        stage1_output_logsumexp,
        num_blocks,
        scale,
        max_seq_len,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK_SIZE,
        HEAD_BLOCK_SIZE,
        sliding_window if sliding_window is not None else -1,
    )
    has_sinks = sinks is not None

    attention_kv_stage2[(num_decode, n_heads, 1)](
        stage1_output_values,
        stage1_output_logsumexp,
        out,
        input_pos,
        num_blocks,
        n_heads,
        v_d_head,
        SEQ_BLOCK_SIZE,
        has_sinks,
        sinks,
    )


def _prefill_attention(
    q: torch.Tensor,  # [num_prefill_tokens, num_heads, qk_head_dim]
    k: torch.Tensor,  # [num_prefill_tokens, num_kv_heads, qk_head_dim]
    v: torch.Tensor,  # [num_prefill_tokens, num_kv_heads, v_head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    input_pos: torch.Tensor,  # [num_prefill]
    slot_idx: torch.Tensor,  # [num_prefill]
    seq_len: torch.Tensor,  # [num_prefill]
    seq_start: torch.Tensor,  # [num_prefill]
    scale: float,
    out: torch.Tensor,  # [num_prefill_tokens, num_heads, v_head_dim]
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
) -> None:
    """Handle prefill phase - context attention with variable sequence lengths."""
    # NOTE: num_prefill_tokens == sum(seq_len)
    num_prefill_tokens, n_heads, q_d_head = q.shape
    max_cache_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]
    num_prefill = len(input_pos)
    SEQ_BLOCK = 32

    update_kv_cache[(num_prefill, n_kv_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)](
        k,
        v,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        input_pos,
        slot_idx,
        max_cache_seq_len,
        n_kv_heads,
        q_d_head,
        v_d_head,
        32,
        GENERATE_ONLY=False,
    )

    grid = (num_prefill, n_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    has_sinks = sinks is not None

    context_attention_kv_flattened[grid](
        q,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        input_pos,
        slot_idx,
        out,
        scale,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK,
        max_cache_seq_len,
        sliding_window if sliding_window is not None else -1,
        has_sinks,
        sinks,
    )


@torch.library.custom_op("auto_deploy::triton_attention_flattened_mha_with_cache", mutates_args=())
def flattened_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    # <none>
    # CONSTANTS
    scale: Optional[float],
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Flattened MHA with cache that takes q, k, v in BSND layout.

    NOTE: this op can also handle seq_len==0, which might be useful for CUDAGRAPH.
    """
    # Extract batch info from batch_info_host
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    # Get cache and head dimensions
    num_kv_heads, qk_head_dim = k_cache.shape[-2:]
    v_head_dim = v_cache.shape[-1]
    b, s = q.shape[:2]

    # Determine num_heads from input shape
    num_heads = q.shape[2] // qk_head_dim if q.ndim == 3 else q.shape[2]

    # Define output shape (preserve original input format)
    output_shape = (b, s, num_heads * v_head_dim) if q.ndim == 3 else (b, s, num_heads, v_head_dim)

    # Flatten Q, K, V to [total_tokens, heads, head_dim]
    bs = b * s
    q_flat = q.contiguous().view(bs, num_heads, qk_head_dim)
    k_flat = k.contiguous().view(bs, num_kv_heads, qk_head_dim)
    v_flat = v.contiguous().view(bs, num_kv_heads, v_head_dim)

    # Compute scale if not provided
    scale = 1.0 / math.sqrt(qk_head_dim) if scale is None else scale

    # Preallocate output tensor
    y = q_flat.new_empty(bs, num_heads, v_head_dim)

    # PREFILL: process context tokens with variable sequence lengths
    if num_prefill > 0:
        _prefill_attention(
            q_flat[:num_prefill_tokens],
            k_flat[:num_prefill_tokens],
            v_flat[:num_prefill_tokens],
            k_cache,
            v_cache,
            input_pos[:num_prefill],
            slot_idx[:num_prefill],
            seq_len[:num_prefill],
            cu_seqlen[:num_prefill],
            scale,
            y[:num_prefill_tokens],
            sinks,
            sliding_window,
        )

    # DECODE: process single-token generation
    if num_decode > 0:
        _decode_attention(
            q_flat[num_prefill_tokens:num_total_tokens],
            k_flat[num_prefill_tokens:num_total_tokens],
            v_flat[num_prefill_tokens:num_total_tokens],
            k_cache,
            v_cache,
            slot_idx[num_prefill:num_seq],
            input_pos[num_prefill:num_seq],
            scale,
            y[num_prefill_tokens:num_total_tokens],
            sinks,
            sliding_window,
        )

    return y.view(*output_shape)


@flattened_mha_with_cache.register_fake
def flattened_mha_fake(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    # <none>
    # CONSTANTS
    scale: Optional[float],
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
):
    return q.new_empty(*q.shape[:-1], v.shape[-1]).contiguous()


@AttentionRegistry.register("triton")
class TritonAttention(AttentionDescriptor):
    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the source op and the cached attention op."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        # source op is [bsnd] layout already
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        v_fake: FakeTensor = source_attn_node.args[2].meta["val"]
        num_kv_heads = k_fake.shape[2]
        k_head_dim = k_fake.shape[3]
        v_head_dim = v_fake.shape[3]

        return {
            "k_cache": UnpagedResourceHandler(
                num_kv_heads,
                k_head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype),
            ),
            "v_cache": UnpagedResourceHandler(
                num_kv_heads,
                v_head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, v_fake.dtype),
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        # Sanity check: layout == "bsnd"
        # Prefer kwargs; fall back to the final positional arg if it's a string.
        layout = source_attn_node.kwargs.get("layout", None)
        if (
            layout is None
            and len(source_attn_node.args) > 0
            and isinstance(source_attn_node.args[-1], str)
        ):
            layout = source_attn_node.args[-1]
        if layout != "bsnd":
            raise RuntimeError(
                f"Expected torch_attention layout='bsnd' but got {layout!r} "
                f"for node: {source_attn_node.format_node()}"
            )

        # retrieve head_dim from k_fake
        attn_mask, dropout_p, is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )
        if attn_mask is not None or dropout_p != 0.0 or not is_causal:
            ad_logger.debug(
                "Unsupported attention arguments for "
                f"{source_attn_node=}: {attn_mask=}, {dropout_p=}, {is_causal=}"
            )

        # Get scale from args or kwargs
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        # do a sanity check on the scale if it is not None, we only support the default scale
        # of 1/sqrt(head_dim) and so we should do an approximate check for that one
        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=} is not a float. Using default scale instead.")
            scale = None
        # Get sinks and sliding_window from args or kwargs
        sinks = extract_op_args(source_attn_node, "sinks")[0]
        sliding_window = extract_op_args(source_attn_node, "sliding_window")[0]
        return [
            scale,  # softmax scale
            sinks,
            sliding_window,
        ]
