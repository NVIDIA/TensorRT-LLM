"""Generic VLM attention mask generation ops.

This module provides custom ops for generating attention masks for VLM models.
The ops are model-agnostic - they dispatch to model-specific generators
registered in VlmMaskGeneratorRegistry.

Key features:
- Generic dispatcher op that routes to model-specific mask generators
- Model-specific mask creation logic is isolated in registered generators
- sliding_window parameter (backend may use native sliding window instead)

The masks are generated in the flattened format expected by FlashInfer's
BatchPrefillWithPagedKVCacheWrapper: a 1D boolean tensor of shape
[sum(q_len[i] * k_len[i]) for context sequences].
"""

from typing import List

import torch
from torch import Tensor

from .vlm_mask_registry import VlmMaskGeneratorRegistry

# =============================================================================
# Generic dispatcher op - routes to model-specific generators
# =============================================================================


@torch.library.custom_op("auto_deploy::create_attention_mask", mutates_args=())
def create_attention_mask(
    token_info: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
    sliding_window: int,
    model_type: str,
) -> Tensor:
    """Generate attention mask for VLM models.

    This is the generic VLM mask dispatcher. It routes to model-specific
    mask generators registered in VlmMaskGeneratorRegistry.

    Args:
        token_info: Model-specific token information tensor [total_tokens].
            Interpretation depends on the model (e.g., token_type_ids for Gemma3
            where 1 = image token).
        qo_indptr: Tensor [num_contexts + 1] from attention metadata.
            Defines the boundaries of each context sequence in the flattened stream.
        seq_len: Tensor [num_seqs] with sequence lengths.
            Used to distinguish context (seq_len > 1) from generation (seq_len == 1).
        sliding_window: Sliding window size. -1 or 0 = no sliding window.
            Backend may ignore this if it handles sliding window natively.
        model_type: Model type string for registry lookup (e.g., "gemma3").

    Returns:
        custom_mask: Flattened bool mask for attention layers.
    """
    # Dispatch to model-specific generator
    generator = VlmMaskGeneratorRegistry.get(model_type)
    if generator is None:
        raise ValueError(
            f"No model-specific generator found for model type: {model_type}. \
        Registered model types: {VlmMaskGeneratorRegistry.registered_model_types()}."
        )

    return generator(token_info, qo_indptr, seq_len, sliding_window)


@create_attention_mask.register_fake
def _create_attention_mask_fake(
    token_info: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
    sliding_window: int,
    model_type: str,
) -> Tensor:
    """Fake implementation for tracing - returns tensor with correct dtype.

    Note: The exact size depends on runtime values (num_contexts, seq_lens),
    so we return a conservatively sized tensor for tracing purposes.
    """
    device = token_info.device

    # Count context sequences
    num_contexts = (seq_len > 1).sum()

    # Upper bound estimate: sum of squares of context sequence lengths
    # In practice, this is an overestimate but safe for tracing
    if num_contexts > 0:
        total_ctx_tokens = qo_indptr[num_contexts]
        # Conservative upper bound: total_tokens^2
        max_size = total_ctx_tokens * total_ctx_tokens
    else:
        max_size = 0

    return torch.empty((max_size,), dtype=torch.bool, device=device)


# =============================================================================
# Gemma3-specific mask generation
# =============================================================================


def _get_context_mask_with_bidir_images(image_token_mask: Tensor) -> Tensor:
    """Generate attention mask for a single context sequence (Gemma3 style).

    Args:
        image_token_mask: Boolean tensor of shape [seq_len] where True = image token.

    Returns:
        Boolean mask of shape [seq_len, seq_len] where True = attention allowed.
        The mask is causal (lower triangular) with bidirectional override for
        image-image token pairs.
    """
    seq_len = image_token_mask.shape[0]
    device = image_token_mask.device

    # Base causal mask: lower triangular (query can attend to key if key_pos <= query_pos)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Image-image bidirectional: if both query and key are image tokens, allow attention
    is_image_q = image_token_mask.unsqueeze(1)  # [seq_len, 1]
    is_image_k = image_token_mask.unsqueeze(0)  # [1, seq_len]
    bidir_image = is_image_q & is_image_k  # [seq_len, seq_len]

    # Override causal restriction for image-image pairs
    mask = mask | bidir_image

    return mask


def _gemma3_mask_impl(
    token_info: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
    sliding_window: int,
) -> Tensor:
    """Gemma3-specific mask generation implementation.

    Creates causal mask with bidirectional attention for image tokens.
    Sliding window is ignored - FlashInfer handles it via window_left.
    """
    device = token_info.device

    # Identify context sequences (seq_len > 1)
    num_contexts = (seq_len > 1).sum().item()

    if num_contexts == 0:
        return torch.empty(0, dtype=torch.bool, device=device)

    masks: List[Tensor] = []
    qo_indptr_ctx = qo_indptr[: num_contexts + 1]

    for i in range(num_contexts):
        start = qo_indptr_ctx[i].item()
        end = qo_indptr_ctx[i + 1].item()

        # Extract image token mask for this sequence
        token_info_i = token_info[start:end]

        # Generate Gemma3-style mask (causal + bidirectional for images)
        mask_i = _get_context_mask_with_bidir_images(token_info_i)

        masks.append(mask_i.flatten())

    return torch.cat(masks).contiguous()


@VlmMaskGeneratorRegistry.register("gemma3_text")
def generate_gemma3_vlm_mask(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
    sliding_window: int,
) -> Tensor:
    """Generate attention mask for Gemma3 VLM.

    For Gemma3:
    - token_info is boolean where True = image token
    - Image tokens get bidirectional attention to each other
    - Text tokens have standard causal attention
    - sliding_window is handled by FlashInfer's window_left (ignored here)

    Args:
        image_token_mask: Boolean tensor [total_tokens] where True = image token.
        qo_indptr: Tensor [num_contexts + 1] from attention metadata.
        seq_len: Tensor [num_seqs] with sequence lengths.
        sliding_window: Sliding window size (ignored by FlashInfer backend).

    Returns:
        custom_mask: Flattened bool mask for attention layers.
    """
    return _gemma3_mask_impl(image_token_mask, qo_indptr, seq_len, sliding_window)
