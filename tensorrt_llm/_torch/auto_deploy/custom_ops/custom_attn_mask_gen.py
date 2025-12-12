"""Custom op for attention mask generation.

This module provides a custom op for creating attention masks during prefill.
The mask supports both full attention (causal + bidirectional for images) and
sliding window attention patterns.
"""

from typing import Optional

import torch
from torch import Tensor


def _create_attention_mask_impl(
    token_type_ids: Optional[Tensor],
    cache_position: Tensor,
    mask_type: str,
    sliding_window: int,
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    """Implementation of attention mask creation.

    Args:
        token_type_ids: Tensor of shape (batch_size, seq_len) indicating token types.
            Value 1 indicates image tokens, 0 indicates text tokens.
            If None, assumes all text tokens (standard causal mask).
        cache_position: Tensor of shape (seq_len,) with position indices.
        mask_type: Either "full_attention" or "sliding_attention".
        sliding_window: Size of sliding window for sliding attention layers.
        dtype: Data type for the output mask.
        batch_size: Batch size (used when token_type_ids is None).

    Returns:
        Attention mask tensor of shape (batch_size, 1, seq_len, seq_len).
        Uses 0.0 for allowed attention and -inf for blocked attention.
    """
    seq_len = cache_position.shape[0]
    device = cache_position.device

    # If token_type_ids is None, create all-zeros (all text tokens)
    if token_type_ids is None:
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    else:
        batch_size = token_type_ids.shape[0]

    # Create base causal mask (lower triangular)
    # Shape: (seq_len, seq_len)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
    )  # True = blocked

    # For image tokens (token_type_ids == 1), allow bidirectional attention within image
    # Image tokens can attend to each other regardless of position
    is_image = token_type_ids == 1  # (batch_size, seq_len)

    # Create bidirectional mask for image tokens
    # Query is image AND Key is image -> allow (override causal)
    # Shape: (batch_size, seq_len, seq_len)
    is_image_query = is_image.unsqueeze(2)  # (batch_size, seq_len, 1)
    is_image_key = is_image.unsqueeze(1)  # (batch_size, 1, seq_len)
    bidirectional_image = is_image_query & is_image_key  # (batch_size, seq_len, seq_len)

    # Expand causal mask to batch
    causal_mask_batch = causal_mask.unsqueeze(0).expand(
        batch_size, -1, -1
    )  # (batch_size, seq_len, seq_len)

    # Override causal mask for image-to-image attention
    # blocked = causal AND NOT (both are images)
    blocked = causal_mask_batch & ~bidirectional_image

    if mask_type == "sliding_attention":
        # Add sliding window constraint
        # Positions outside the window are also blocked
        # Window: each position can only attend to positions within [pos - sliding_window + 1, pos]
        positions = cache_position.view(1, -1)  # (1, seq_len)
        distance = positions - positions.T  # (seq_len, seq_len): query_pos - key_pos

        # Block if key is too far back (distance > sliding_window - 1)
        # or if key is ahead (handled by causal mask)
        outside_window = distance >= sliding_window  # (seq_len, seq_len)
        outside_window = outside_window.unsqueeze(0).expand(batch_size, -1, -1)

        # For sliding attention with images:
        # - Text follows sliding window + causal
        # - Images within sliding window can still be bidirectional
        blocked = blocked | (outside_window & ~bidirectional_image)

    # Convert to attention mask format: 0.0 for allowed, -inf for blocked
    mask = torch.where(
        blocked, torch.finfo(dtype).min, torch.tensor(0.0, device=device, dtype=dtype)
    )

    # Add head dimension: (batch_size, 1, seq_len, seq_len)
    mask = mask.unsqueeze(1)

    return mask.to(dtype)


# Register the custom op
@torch.library.custom_op("autodeploy::custom_attn_mask_gen_op", mutates_args=())
def custom_attn_mask_gen_op(
    token_type_ids: Optional[Tensor],
    cache_position: Tensor,
    mask_type: str,
    sliding_window: int,
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    """Create attention mask for prefill.

    This custom op generates attention masks that support:
    - Causal attention for text tokens
    - Bidirectional attention for image tokens
    - Optional sliding window constraint

    Args:
        token_type_ids: Token type indicators (1=image, 0=text). Can be None.
        cache_position: Current position indices.
        mask_type: "full_attention" or "sliding_attention".
        sliding_window: Window size for sliding attention.
        dtype: Output dtype.
        batch_size: Batch size (used when token_type_ids is None).

    Returns:
        Attention mask of shape (batch, 1, seq_len, seq_len).
    """
    return _create_attention_mask_impl(
        token_type_ids, cache_position, mask_type, sliding_window, dtype, batch_size
    )


@custom_attn_mask_gen_op.register_fake
def _(
    token_type_ids: Optional[Tensor],
    cache_position: Tensor,
    mask_type: str,
    sliding_window: int,
    dtype: torch.dtype,
    batch_size: int,
) -> Tensor:
    """Fake implementation for tracing - returns tensor with correct shape."""
    seq_len = cache_position.shape[0]
    # Use batch_size from token_type_ids if available, otherwise use the explicit param
    if token_type_ids is not None:
        batch_size = token_type_ids.shape[0]
    return torch.empty((batch_size, 1, seq_len, seq_len), dtype=dtype, device=cache_position.device)
