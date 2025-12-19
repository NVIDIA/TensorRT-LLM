"""FlashInfer custom mask generation for Gemma3 VLM.

This module provides a custom op for generating FlashInfer-compatible attention masks
for Gemma3 VLM. The masks support:
- Causal attention for text tokens
- Bidirectional attention for image tokens (image tokens attend to each other)
- Optional sliding window constraint for sliding attention layers

The masks are generated in the flattened format expected by FlashInfer's
BatchPrefillWithPagedKVCacheWrapper, where the mask is a 1D boolean tensor
of shape [sum(q_len[i] * k_len[i]) for context sequences].
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .vlm_mask_registry import VlmMaskGeneratorRegistry


def _get_context_mask(
    image_token_mask: Tensor,
    sliding_window: Optional[int],
) -> Tensor:
    """Generate attention mask for a single context sequence.

    Args:
        image_token_mask: Boolean tensor of shape [seq_len] where True = image token.
        sliding_window: If not None, apply sliding window constraint.

    Returns:
        Boolean mask of shape [seq_len, seq_len] where True = attention allowed.
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

    # Apply sliding window constraint if specified
    if sliding_window is not None and sliding_window > 0:
        positions = torch.arange(seq_len, device=device)
        # distance[q, k] = q_pos - k_pos
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]

        # Block if key is too far back (distance >= sliding_window)
        # But still allow if both are image tokens (bidir_image overrides)
        outside_window = distance >= sliding_window
        mask = mask & ~(outside_window & ~bidir_image)

    return mask


def _flashinfer_gemma3_mask_gen_impl(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
    sliding_window: int,
) -> Tuple[Tensor, Tensor]:
    """Implementation of Gemma3 mask generation for FlashInfer.

    Args:
        image_token_mask: Boolean tensor [total_tokens] where True = image token.
        qo_indptr: Tensor [num_contexts + 1] with cumulative token counts per context seq.
        seq_len: Tensor [num_seqs] with sequence lengths (used to identify context seqs).
        sliding_window: Sliding window size from Gemma3 config.

    Returns:
        Tuple of:
            - custom_mask_full: Flattened bool mask for full attention layers
            - custom_mask_sliding: Flattened bool mask for sliding attention layers
    """
    device = image_token_mask.device

    # Identify context sequences (seq_len > 1)
    # Generation sequences have seq_len == 1 and don't need custom masks
    num_contexts = (seq_len > 1).sum().item()

    if num_contexts == 0:
        # No context requests → return empty masks
        return (
            torch.empty(0, dtype=torch.bool, device=device),
            torch.empty(0, dtype=torch.bool, device=device),
        )

    full_masks: List[Tensor] = []
    sliding_masks: List[Tensor] = []

    # Process only context sequences
    qo_indptr_ctx = qo_indptr[: num_contexts + 1]

    for i in range(num_contexts):
        start = qo_indptr_ctx[i].item()
        end = qo_indptr_ctx[i + 1].item()

        # Extract image mask for this sequence
        img_mask_i = image_token_mask[start:end]

        # Generate masks for this sequence
        full_mask_i = _get_context_mask(img_mask_i, sliding_window=None)
        sliding_mask_i = _get_context_mask(img_mask_i, sliding_window=sliding_window)

        # Flatten and append
        full_masks.append(full_mask_i.flatten())
        sliding_masks.append(sliding_mask_i.flatten())

    # Concatenate all sequence masks into single flattened vectors
    custom_mask_full = torch.cat(full_masks).contiguous()
    custom_mask_sliding = torch.cat(sliding_masks).contiguous()

    return custom_mask_full, custom_mask_sliding


# Register the custom op
@torch.library.custom_op("auto_deploy::flashinfer_gemma3_mask_gen", mutates_args=())
def flashinfer_gemma3_mask_gen(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
    sliding_window: int,
) -> Tuple[Tensor, Tensor]:
    """Generate FlashInfer custom masks for Gemma3 VLM.

    This custom op generates two flattened boolean masks for FlashInfer:
    - One for full attention layers (causal + image-image bidirectional)
    - One for sliding attention layers (same + sliding window constraint)

    The masks are in FlashInfer's expected format: a 1D boolean tensor where
    True = attention allowed, with shape [sum(q_len[i] * k_len[i])] over
    context sequences only.

    Args:
        image_token_mask: Boolean tensor [total_tokens] where True = image token.
            Must be aligned with the flattened token stream.
        qo_indptr: Tensor [num_contexts + 1] from flashinfer prepare_metadata.
            Defines the boundaries of each context sequence in the flattened stream.
        seq_len: Tensor [num_seqs] with sequence lengths.
            Used to distinguish context (seq_len > 1) from generation (seq_len == 1).
        sliding_window: Sliding window size from Gemma3 config.

    Returns:
        Tuple of:
            - custom_mask_full: Flattened bool mask for full attention layers
            - custom_mask_sliding: Flattened bool mask for sliding attention layers
    """
    return _flashinfer_gemma3_mask_gen_impl(image_token_mask, qo_indptr, seq_len, sliding_window)


@flashinfer_gemma3_mask_gen.register_fake
def _flashinfer_gemma3_mask_gen_fake(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
    sliding_window: int,
) -> Tuple[Tensor, Tensor]:
    """Fake implementation for tracing - returns tensors with correct dtype.

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

    return (
        torch.empty((max_size,), dtype=torch.bool, device=device),
        torch.empty((max_size,), dtype=torch.bool, device=device),
    )


# Register Gemma3 mask generator with the VLM mask registry
@VlmMaskGeneratorRegistry.register("gemma3")
def generate_gemma3_vlm_masks(
    image_token_mask: Tensor,
    qo_indptr: Tensor,
    seq_len: Tensor,
    sliding_window: int,
) -> Tuple[Tensor, Tensor]:
    """Generate FlashInfer custom masks for Gemma3 VLM.

    This is the registry entry point for Gemma3 VLM mask generation.
    It delegates to the torch custom op for the actual mask computation.

    Args:
        image_token_mask: Boolean tensor [total_tokens] where True = image token.
        qo_indptr: Tensor [num_contexts + 1] from flashinfer prepare_metadata.
        seq_len: Tensor [num_seqs] with sequence lengths.
        sliding_window: Sliding window size from Gemma3 config.

    Returns:
        Tuple of (custom_mask_full, custom_mask_sliding).
    """
    return torch.ops.auto_deploy.flashinfer_gemma3_mask_gen(
        image_token_mask, qo_indptr, seq_len, sliding_window
    )
