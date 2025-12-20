"""FlashInfer custom mask generation for VLM models (e.g., Gemma3).

This module provides a custom op for generating FlashInfer-compatible attention masks
for VLM models. The masks support:
- Causal attention for text tokens
- Bidirectional attention for image tokens (image tokens attend to each other)

Note: Sliding window constraints are handled separately by FlashInfer's window_left
parameter, not baked into the mask. This allows the same mask to be used for both
full attention and sliding attention layers.

The masks are generated in the flattened format expected by FlashInfer's
BatchPrefillWithPagedKVCacheWrapper, where the mask is a 1D boolean tensor
of shape [sum(q_len[i] * k_len[i]) for context sequences].
"""

from typing import List

import torch
from torch import Tensor

from .vlm_mask_registry import VlmMaskGeneratorRegistry


def _get_context_mask(image_token_mask: Tensor) -> Tensor:
    """Generate attention mask for a single context sequence.

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
    # True = allowed
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Image-image bidirectional: if both query and key are image tokens, allow attention
    # regardless of position
    is_image_q = image_token_mask.unsqueeze(1)  # [seq_len, 1]
    is_image_k = image_token_mask.unsqueeze(0)  # [1, seq_len]
    bidir_image = is_image_q & is_image_k  # [seq_len, seq_len]

    # Override causal restriction for image-image pairs
    mask = mask | bidir_image

    return mask


def _flashinfer_vlm_mask_gen_impl(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
) -> Tensor:
    """Implementation of VLM mask generation for FlashInfer.

    Args:
        image_token_mask: Boolean tensor [total_tokens] where True = image token.
        qo_indptr: Tensor [num_contexts + 1] with cumulative token counts per context seq.
        seq_len: Tensor [num_seqs] with sequence lengths (used to identify context seqs).

    Returns:
        Flattened bool mask for attention (causal + image-image bidirectional).
        Sliding window is handled separately by FlashInfer's window_left parameter.
    """
    device = image_token_mask.device

    # Identify context sequences (seq_len > 1)
    # Generation sequences have seq_len == 1 and don't need custom masks
    num_contexts = (seq_len > 1).sum().item()

    if num_contexts == 0:
        # No context requests → return empty mask
        return torch.empty(0, dtype=torch.bool, device=device)

    masks: List[Tensor] = []

    # Process only context sequences
    qo_indptr_ctx = qo_indptr[: num_contexts + 1]

    for i in range(num_contexts):
        start = qo_indptr_ctx[i].item()
        end = qo_indptr_ctx[i + 1].item()

        # Extract image mask for this sequence
        img_mask_i = image_token_mask[start:end]

        # Generate mask for this sequence (causal + bidirectional for images)
        mask_i = _get_context_mask(img_mask_i)

        # Flatten and append
        masks.append(mask_i.flatten())

    # Concatenate all sequence masks into single flattened vector
    custom_mask = torch.cat(masks).contiguous()

    return custom_mask


# Register the custom op
@torch.library.custom_op("auto_deploy::flashinfer_vlm_mask_gen", mutates_args=())
def flashinfer_vlm_mask_gen(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
) -> Tensor:
    """Generate FlashInfer custom mask for VLM models.

    This custom op generates a flattened boolean mask for FlashInfer that provides:
    - Causal attention (query attends to earlier keys)
    - Bidirectional attention for image tokens (image tokens attend to each other)

    Sliding window constraints are NOT baked into this mask - they are handled
    separately by FlashInfer's window_left parameter per attention layer.

    The mask is in FlashInfer's expected format: a 1D boolean tensor where
    True = attention allowed, with shape [sum(q_len[i] * k_len[i])] over
    context sequences only.

    Args:
        image_token_mask: Boolean tensor [total_tokens] where True = image token.
            Must be aligned with the flattened token stream.
        qo_indptr: Tensor [num_contexts + 1] from flashinfer prepare_metadata.
            Defines the boundaries of each context sequence in the flattened stream.
        seq_len: Tensor [num_seqs] with sequence lengths.
            Used to distinguish context (seq_len > 1) from generation (seq_len == 1).

    Returns:
        custom_mask: Flattened bool mask for attention layers.
    """
    return _flashinfer_vlm_mask_gen_impl(image_token_mask, qo_indptr, seq_len)


@flashinfer_vlm_mask_gen.register_fake
def _flashinfer_vlm_mask_gen_fake(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
) -> Tensor:
    """Fake implementation for tracing - returns tensor with correct dtype.

    Note: The exact size depends on runtime values (num_contexts, seq_lens),
    so we return a conservatively sized tensor for tracing purposes.
    """
    device = image_token_mask.device

    # Count context sequences
    num_contexts = (seq_len > 1).sum()

    # Upper bound estimate: sum of squares of context sequence lengths
    # In practice, this is an overestimate but safe for tracing
    # We use qo_indptr[-1] as total context tokens, then square it
    if num_contexts > 0:
        total_ctx_tokens = qo_indptr[num_contexts]
        # Conservative upper bound: total_tokens^2
        # In reality it's sum(len_i^2), which is <= (sum len_i)^2
        max_size = total_ctx_tokens * total_ctx_tokens
    else:
        max_size = 0

    return torch.empty((max_size,), dtype=torch.bool, device=device)


# Register Gemma3 mask generator with the VLM mask registry
@VlmMaskGeneratorRegistry.register("gemma3")
def generate_gemma3_vlm_mask(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
) -> Tensor:
    """Generate FlashInfer custom mask for Gemma3 VLM.

    This is the registry entry point for Gemma3 VLM mask generation.
    It delegates to the torch custom op for the actual mask computation.

    Args:
        image_token_mask: Boolean tensor [total_tokens] where True = image token.
        qo_indptr: Tensor [num_contexts + 1] from flashinfer prepare_metadata.
        seq_len: Tensor [num_seqs] with sequence lengths.

    Returns:
        custom_mask: Flattened bool mask for attention layers.
    """
    return torch.ops.auto_deploy.flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)
