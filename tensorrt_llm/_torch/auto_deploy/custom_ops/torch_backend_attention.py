"""Torch backend attention using pure PyTorch reference implementations."""

import math
from typing import List, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from ..utils.logger import ad_logger
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
from .torch_attention import repeat_kv, update_kv_cache


def _apply_logit_softcapping(attn_scores: torch.Tensor, logit_cap: Optional[float]) -> torch.Tensor:
    """Apply logit softcapping using the formula: logit_cap * tanh(logits / logit_cap)"""
    if logit_cap is not None and logit_cap > 0.0:
        return logit_cap * torch.tanh(attn_scores / logit_cap)
    return attn_scores


def _torch_generate_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    input_pos: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    logit_cap: Optional[float] = None,
    sliding_window_size: Optional[int] = None,
    sinks: Optional[torch.Tensor] = None,
):
    """Generate-only attention (single token per sequence) using manual computation with existing update_kv_cache."""
    b, s, n_heads, head_dim = q.shape  # q has shape (b, 1, n_heads, head_dim) in generate phase
    assert s == 1, f"Expected sequence length 1 for generate phase, got {s}"
    n_kv_heads = k.shape[2]  # k has shape (b, 1, n_kv_heads, head_dim)

    # Update KV cache for single token
    for i in range(b):
        cache_idx = cache_loc[i].item()
        pos = input_pos[i].item()
        k_cache[cache_idx, pos] = k[i, 0]  # Remove sequence dim
        v_cache[cache_idx, pos] = v[i, 0]  # Remove sequence dim

    # Compute attention for each sequence using manual computation
    for i in range(b):
        cache_idx = cache_loc[i].item()
        pos = input_pos[i].item()

        # Get query, key, value for this sequence
        q_i = q[i, 0]  # [n_heads, head_dim]

        # Apply sliding window: limit the range of keys/values we attend to
        if sliding_window_size is not None and sliding_window_size > 0:
            # Sliding window: attend to [max(0, pos - sliding_window_size + 1), pos]
            start_pos = max(0, pos - sliding_window_size + 1)
            k_i = k_cache[cache_idx, start_pos : pos + 1]  # [window_len, n_kv_heads, head_dim]
            v_i = v_cache[cache_idx, start_pos : pos + 1]  # [window_len, n_kv_heads, v_head_dim]
        else:
            # No sliding window: attend to all previous tokens [0, pos]
            k_i = k_cache[cache_idx, : pos + 1]  # [seq_len, n_kv_heads, head_dim]
            v_i = v_cache[cache_idx, : pos + 1]  # [seq_len, n_kv_heads, v_head_dim]

        # Transpose for attention: [n_heads, 1, head_dim] and [n_kv_heads, seq_len, head_dim]
        q_i = q_i.unsqueeze(1)  # [n_heads, 1, head_dim]
        k_i = k_i.transpose(0, 1)  # [n_kv_heads, seq_len, head_dim]
        v_i = v_i.transpose(0, 1)  # [n_kv_heads, seq_len, v_head_dim]

        # Handle GQA using existing repeat_kv function if needed
        if n_heads != n_kv_heads:
            n_rep = n_heads // n_kv_heads
            # Reshape to [batch, num_kv_heads, seq_len, head_dim] for repeat_kv
            # k_i is currently [n_kv_heads, seq_len, head_dim]
            k_i_batch = k_i.unsqueeze(0)  # [1, n_kv_heads, seq_len, head_dim]
            v_i_batch = v_i.unsqueeze(0)  # [1, n_kv_heads, seq_len, v_head_dim]
            k_i_expanded = repeat_kv(k_i_batch, n_rep)  # [1, n_heads, seq_len, head_dim]
            v_i_expanded = repeat_kv(v_i_batch, n_rep)  # [1, n_heads, seq_len, v_head_dim]
            k_i = k_i_expanded[0]  # [n_heads, seq_len, head_dim]
            v_i = v_i_expanded[0]  # [n_heads, seq_len, v_head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale  # [n_heads, 1, seq_len]

        # Apply logit softcapping if enabled
        attn_scores = _apply_logit_softcapping(attn_scores, logit_cap)

        # Apply sinks if provided (following the model file pattern)
        if sinks is not None:
            # Concatenate sinks to attention scores
            sinks = sinks.reshape(-1, 1, 1)
            attn_weights = torch.cat([attn_scores, sinks], dim=-1)
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            # Use only the non-sink portion for computing output (ignore sinks)
            attn_out = torch.matmul(
                attn_weights[..., : -sinks.size(-1)], v_i
            )  # [n_heads, 1, v_head_dim]
        else:
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(attn_weights, v_i)  # [n_heads, 1, v_head_dim]

        # Store result: remove sequence dimension
        out[i] = attn_out.squeeze(1)  # [n_heads, v_head_dim]


def _torch_context_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    logit_cap: Optional[float] = None,
    sliding_window_size: Optional[int] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    """Context attention (multiple tokens, potentially multiple sequences) using existing torch functions."""
    # Update KV cache first using existing function
    update_kv_cache(k, v, k_cache, v_cache, seq_len, input_pos, cache_loc, seq_start)

    # Compute attention for each sequence
    attn_outputs = []
    for idx in range(seq_len.shape[0]):
        seq_len_i = seq_len[idx].item()
        input_pos_i = input_pos[idx].item()
        cache_loc_i = cache_loc[idx].item()
        seq_start_i = seq_start[idx].item()

        # Skip sequences with zero length
        if seq_len_i == 0:
            continue

        # Get query for this sequence
        q_seq = q[seq_start_i : seq_start_i + seq_len_i]  # [seq_len_i, n_heads, head_dim]

        # Get keys and values from cache
        kv_seq_len = input_pos_i + seq_len_i
        k_seq = k_cache[cache_loc_i, :kv_seq_len]  # [kv_seq_len, n_kv_heads, head_dim]
        v_seq = v_cache[cache_loc_i, :kv_seq_len]  # [kv_seq_len, n_kv_heads, head_dim]

        # Manual attention computation (shared path for both softcapping and non-softcapping)
        n_heads = q_seq.shape[1]
        n_kv_heads = k_seq.shape[1]

        # Transpose to [batch, num_heads, seq_len, head_dim] format
        q_seq_t = q_seq.transpose(0, 1).unsqueeze(0)  # [1, n_heads, seq_len_i, head_dim]
        k_seq_t = k_seq.transpose(0, 1).unsqueeze(0)  # [1, n_kv_heads, kv_seq_len, head_dim]
        v_seq_t = v_seq.transpose(0, 1).unsqueeze(0)  # [1, n_kv_heads, kv_seq_len, head_dim]

        # Handle GQA by repeating KV if needed
        if n_heads != n_kv_heads:
            n_rep = n_heads // n_kv_heads
            k_seq_t = repeat_kv(k_seq_t, n_rep)  # [1, n_heads, kv_seq_len, head_dim]
            v_seq_t = repeat_kv(v_seq_t, n_rep)  # [1, n_heads, kv_seq_len, head_dim]

        # Compute attention scores: Q @ K^T
        attn_scores = (
            torch.matmul(q_seq_t, k_seq_t.transpose(-2, -1)) * scale
        )  # [1, n_heads, seq_len_i, kv_seq_len]

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len_i, kv_seq_len, device=q.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len_i + 1,
        )

        # Apply sliding window mask if specified
        if sliding_window_size is not None and sliding_window_size > 0:
            # Create sliding window mask: each query position i can only attend to keys in [i-window_size+1, i]
            # For context phase, we need to account for the offset between query and key positions

            # Query positions are [input_pos_i, input_pos_i + seq_len_i)
            # Key positions are [0, input_pos_i + seq_len_i)
            query_positions = torch.arange(
                input_pos_i, input_pos_i + seq_len_i, device=q.device
            )  # [seq_len_i]
            key_positions = torch.arange(0, kv_seq_len, device=q.device)  # [kv_seq_len]

            # Create position difference matrix: query_pos - key_pos
            pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(
                0
            )  # [seq_len_i, kv_seq_len]

            # Sliding window mask: allow attention only if 0 <= pos_diff < sliding_window_size
            sliding_window_mask = pos_diff >= sliding_window_size

            # Combine causal and sliding window masks
            combined_mask = causal_mask | sliding_window_mask
        else:
            combined_mask = causal_mask

        attn_scores.masked_fill_(combined_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply logit softcapping if enabled
        attn_scores = _apply_logit_softcapping(attn_scores, logit_cap)

        # Apply sinks if provided (following the model file pattern)
        if sinks is not None:
            # Concatenate sinks to attention scores
            new_sinks = sinks.reshape(1, -1, 1, 1).expand(
                attn_scores.shape[0], -1, attn_scores.shape[2], 1
            )
            attn_weights = torch.cat([attn_scores, new_sinks], dim=-1)
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            # Use only the non-sink portion for computing output (ignore sinks)
            attn_out = torch.matmul(
                attn_weights[..., : -new_sinks.size(-1)], v_seq_t
            )  # [1, n_heads, seq_len_i, v_head_dim]
        else:
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(attn_weights, v_seq_t)  # [1, n_heads, seq_len_i, v_head_dim]

        # Remove batch dimension and transpose back to [seq_len_i, n_heads, v_head_dim]
        attn_out = attn_out[0].transpose(0, 1)

        attn_outputs.append(attn_out)

    # Concatenate all outputs
    if len(attn_outputs) == 0:
        # No sequences to process - this shouldn't happen but handle gracefully
        out.zero_()
    elif len(attn_outputs) == 1:
        # Single sequence
        out.copy_(attn_outputs[0])
    else:
        # Multiple sequences or context phase
        out.copy_(torch.cat(attn_outputs, dim=0))


@torch.library.custom_op("auto_deploy::torch_cached_attention_with_cache", mutates_args=())
def torch_backend_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    # <none>
    # CONSTANTS
    scale: Optional[float],
    sinks: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    logit_cap: Optional[float] = None,
) -> torch.Tensor:
    """Torch backend MHA with cache that takes q, k, v in BSND layout."""
    # Get dimensions
    num_kv_heads, qk_head_dim = k_cache.shape[-2:]
    v_head_dim = v_cache.shape[-1]
    b, s = q.shape[:2]

    # check for num_heads
    num_heads = q.shape[2] // qk_head_dim if q.ndim == 3 else q.shape[2]

    # Define output shape
    output_shape = (b, s, num_heads * v_head_dim) if q.ndim == 3 else (b, s, num_heads, v_head_dim)

    # Reshape to standard layout
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)

    q = q.contiguous().view(*bs_view, num_heads, qk_head_dim)
    k = k.contiguous().view(*bs_view, num_kv_heads, qk_head_dim)
    v = v.contiguous().view(*bs_view, num_kv_heads, v_head_dim)

    scale = 1.0 / math.sqrt(qk_head_dim) if scale is None else scale

    # Create output tensor
    y = q.new_empty(*bs_view, num_heads, v_head_dim).contiguous()

    # Compute attention
    if s == 1:
        # Generate-only phase
        _torch_generate_mha(
            q,
            k,
            v,
            k_cache,
            v_cache,
            cache_loc,
            input_pos,
            scale,
            y,
            logit_cap,
            sliding_window_size,
            sinks,
        )
    else:
        # Context phase
        _torch_context_mha(
            q,
            k,
            v,
            input_pos,
            cache_loc,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
            scale,
            y,
            logit_cap,
            sliding_window_size,
            sinks,
        )

    return y.view(*output_shape)


@torch_backend_mha_with_cache.register_fake
def torch_backend_mha_with_cache_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: Optional[float],
    sinks: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    logit_cap: Optional[float] = None,
):
    return q.new_empty(*q.shape[:-1], v.shape[-1]).contiguous()


@torch.library.custom_op("auto_deploy::torch_cached_attention_prepare_metadata", mutates_args=())
def torch_backend_prepare_metadata(
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    slot_idx: torch.Tensor,
    page_size: int,
) -> List[torch.Tensor]:
    """Prepare metadata for torch backend attention (similar to triton backend)."""
    num_seq = SequenceInfo._get_sanitized_num_sequences(position_ids, seq_len)
    seq_start = torch.zeros_like(seq_len[:num_seq])
    seq_start[1:] = torch.cumsum(seq_len[: num_seq - 1], 0)
    return (
        seq_len[:num_seq].clone(),
        input_pos[:num_seq].clone(),
        cache_loc[:num_seq].clone(),
        seq_start,
    )


@torch_backend_prepare_metadata.register_fake
def torch_backend_prepare_metadata_fake(
    position_ids, seq_len, input_pos, cache_loc, pages_per_seq, slot_idx, page_size
):
    num_seq = SequenceInfo._get_sanitized_num_sequences(position_ids, seq_len)
    return (
        torch.empty_like(seq_len[:num_seq]),
        torch.empty_like(input_pos[:num_seq]),
        torch.empty_like(cache_loc[:num_seq]),
        torch.empty_like(seq_len[:num_seq]),
    )


@AttentionRegistry.register("torch")
class TorchBackendAttention(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""
        return False

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
        return torch.ops.auto_deploy.torch_cached_attention_with_cache

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        return torch.ops.auto_deploy.torch_cached_attention_prepare_metadata, 4

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        # source op is [bsnd] layout already
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        v_fake: FakeTensor = source_attn_node.args[2].meta["val"]
        num_kv_heads = k_fake.shape[2]
        k_head_dim = k_fake.shape[3]
        v_head_dim = v_fake.shape[3]

        def _get_k_cache(si: SequenceInfo):
            assert not si.is_paged, "Paged cache not supported for torch backend"
            return torch.empty(
                si.num_pages,
                si.page_size,
                num_kv_heads,
                k_head_dim,
                device=si.device,
                dtype=cache_config.dtype or k_fake.dtype,
            )

        def _get_v_cache(si: SequenceInfo):
            assert not si.is_paged, "Paged cache not supported for torch backend"
            return torch.empty(
                si.num_pages,
                si.page_size,
                num_kv_heads,
                v_head_dim,
                device=si.device,
                dtype=cache_config.dtype or v_fake.dtype,
            )

        return {"k_cache": _get_k_cache, "v_cache": _get_v_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        return {}

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

        # Check other arguments
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

        # Validate scale
        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        # Get sinks, sliding_window, and logit_cap from args or kwargs
        sinks = extract_op_args(source_attn_node, "sinks")[0]
        sliding_window = extract_op_args(source_attn_node, "sliding_window")[0]
        logit_cap = extract_op_args(source_attn_node, "logit_cap")[0]

        return [
            scale,  # softmax scale
            sinks,  # sinks parameter
            sliding_window,  # sliding window parameter
            logit_cap,  # logit cap parameter
        ]
