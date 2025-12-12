"""Utilities for Vision-Language Model (VLM) support in AutoDeploy.

This module provides helper functions for handling VLM-specific inputs like
image token masks, which are used for custom attention masking in models
like Gemma3 that require bidirectional attention for image tokens.
"""

from typing import Dict, Optional

import torch


def get_image_token_mask(
    model_type: str,
    named_args: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Get a boolean mask indicating image tokens from model inputs.

    Different VLMs use different field names to indicate image vs text tokens.
    This utility abstracts that difference so the runtime can work with any VLM.

    Args:
        model_type: Model identifier (e.g., "gemma3", "qwen2_vl", "llava").
            This should match the model type string used in the factory.
        named_args: The named arguments dict from SequenceInfo, containing
            flattened input tensors.

    Returns:
        Boolean tensor of shape [total_tokens] where True = image token,
        or None if no image tokens present or not a supported VLM.

    Supported models and their token type fields:
        - gemma3: Uses `token_type_ids` (0=text, 1=image)
        - qwen2_vl, qwen2.5_vl, qwen3_vl: Uses `mm_token_type_ids` (0=text, 1=image)
        - llava, llava_next: Uses `mm_token_type_ids` (0=text, 1=image)
    """
    if model_type == "gemma3":
        token_type_ids = named_args.get("token_type_ids")
        if token_type_ids is not None:
            # In Gemma3: 0 = text token, 1 = image token
            return token_type_ids == 1
        return None

    elif model_type in ("qwen2_vl", "qwen2.5_vl", "qwen3_vl"):
        mm_token_type_ids = named_args.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            return mm_token_type_ids == 1
        return None

    elif model_type in ("llava", "llava_next"):
        mm_token_type_ids = named_args.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            return mm_token_type_ids == 1
        return None

    # Model type not recognized or doesn't need image token masking
    return None


def has_image_tokens(
    model_type: str,
    named_args: Dict[str, torch.Tensor],
) -> bool:
    """Check if the current batch contains any image tokens.

    This is useful for determining whether to apply VLM-specific constraints
    like disabling chunked prefill or KV cache reuse.

    Args:
        model_type: Model identifier.
        named_args: The named arguments dict from SequenceInfo.

    Returns:
        True if there are image tokens in the batch, False otherwise.
    """
    image_mask = get_image_token_mask(model_type, named_args)
    if image_mask is None:
        return False
    return image_mask.any().item()
