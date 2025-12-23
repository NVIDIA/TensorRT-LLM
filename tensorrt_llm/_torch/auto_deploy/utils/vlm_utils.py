"""Utilities for Vision-Language Model (VLM) support in AutoDeploy.

This module provides helper functions for handling VLM-specific inputs like
image token masks, which are used for custom attention masking in models
like Gemma3 that require bidirectional attention for image tokens.
"""

from typing import Dict, Optional

import torch


def get_image_token_mask(
    named_args: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Get a boolean mask indicating image tokens from model inputs.

    Args:
        named_args: The named arguments dict from SequenceInfo, containing
            flattened input tensors.

    Returns:
        Boolean tensor of shape [total_tokens] where True = image token,
        or None if no image tokens present.

    Note:
        Uses `token_type_ids` where 0 = text token, 1 = image token.
    """
    token_type_ids = named_args.get("token_type_ids")
    if token_type_ids is not None:
        return token_type_ids == 1
    return None
