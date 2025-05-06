from typing import Optional, Type

import torch

from ...models.modeling_utils import QuantConfig
from . import IS_FLASHINFER_AVAILABLE
from .interface import AttentionBackend, MLAParams, PositionalEmbeddingParams
from .trtllm import TrtllmAttention
from .vanilla import VanillaAttention


def get_attention_backend(backend_name: str) -> Type[AttentionBackend]:
    if backend_name == "VANILLA":
        return VanillaAttention
    elif backend_name == "TRTLLM":
        return TrtllmAttention
    elif backend_name == "FLASHINFER" and IS_FLASHINFER_AVAILABLE:
        from .flashinfer import FlashInferAttention

        return FlashInferAttention
    elif backend_name == "FLASHINFER_STAR_ATTENTION" and IS_FLASHINFER_AVAILABLE:
        from .star_flashinfer import StarAttention

        return StarAttention

    return TrtllmAttention


def create_attention(
    backend_name: str,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    pos_embd_params: Optional[PositionalEmbeddingParams] = None,
    quant_config: Optional[QuantConfig] = None,
    q_scaling: Optional[float] = None,
    is_mla_enable: bool = False,
    q_lora_rank: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    qk_nope_head_dim: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    predicted_tokens_per_seq: Optional[int] = 1,
    skip_create_weights_in_init: bool = False,
):

    attn_cls = get_attention_backend(backend_name)

    if is_mla_enable:
        assert attn_cls.support_mla(
        ), f"MLA is not supported for {backend_name} backend"
        assert (q_lora_rank > 0 and kv_lora_rank > 0 and qk_rope_head_dim > 0
                and qk_nope_head_dim > 0 and v_head_dim > 0)
        mla_params = MLAParams(
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
        )
    else:
        mla_params = None

    return attn_cls(
        layer_idx,
        num_heads,
        head_dim,
        num_kv_heads,
        quant_config=quant_config,
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        mla_params=mla_params,
        skip_create_weights_in_init=skip_create_weights_in_init,
    )


def create_context_chunk_mask(
    total_prompt_len: int,
    context_len: int,
    attention_chunk_size: int = 8192,
) -> torch.Tensor:
    """
    Creates a chunked causal attention mask specifically for a context chunk.

    This mask defines attention patterns for the last `context_len` query tokens
    within a sequence of `total_prompt_len`. Attention is allowed only to
    key tokens within the same `attention_chunk_size` block and where the
    key index is less than or equal to the query index (causal).

    Args:
        total_prompt_len: The total sequence length currently being considered
                          (e.g., length of keys/values in the KV cache).
        context_len: The number of query tokens in the current chunk (usually >= 1).
                     These are the last `context_len` tokens.
        attention_chunk_size: The size of the blocks for chunked attention.
        device: The torch device to create the mask on.

    Returns:
        A boolean tensor of shape (context_len, total_prompt_len)
        where True indicates attention is allowed from the query token (row)
        to the key token (column).
    """

    # 1. Define the indices for the query tokens (the current context chunk)
    # These are the last `context_len` indices of the total length.
    start_of_chunk = total_prompt_len - context_len
    # Rows correspond to query tokens in the current chunk
    q_indices = torch.arange(start_of_chunk,
                             total_prompt_len,
                             device=torch.device("cuda"),
                             dtype=torch.int32)  # Shape: [context_len]

    # 2. Define the indices for all key tokens (the entire sequence processed so far)
    # Columns correspond to key tokens
    k_indices = torch.arange(total_prompt_len,
                             device=torch.device("cuda"),
                             dtype=torch.int32)  # Shape: [total_prompt_len]

    # 3. Expand indices to compute pairwise relationships
    q_indices_expanded = q_indices.unsqueeze(1)  # Shape: [context_len, 1]
    k_indices_expanded = k_indices.unsqueeze(0)  # Shape: [1, total_prompt_len]

    # 4. Calculate block assignments for queries and keys
    q_block = q_indices_expanded // attention_chunk_size
    k_block = k_indices_expanded // attention_chunk_size

    # 5. Create the mask based on chunked causal attention rules
    # Rule 1: Key must be in the same block as the query (block_pos == 0)
    same_block_mask = (q_block == k_block)
    # Rule 2: Key index must be less than or equal to query index (token_pos <= 0 relative to original func, or q >= k)
    causal_mask = (q_indices_expanded >= k_indices_expanded)

    # Combine rules: must be causal AND within the same block
    # Resulting mask shape: [context_len, total_prompt_len]
    combined_mask = same_block_mask & causal_mask
    assert combined_mask.shape == (context_len, total_prompt_len)

    return combined_mask
