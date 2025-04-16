from typing import Tuple

import flashinfer
import torch


@torch.library.custom_op("rope::flashinfer", mutates_args=())
def apply_rope_with_input_pos_flashinfer(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings (RoPE) to query and key tensors using the FlashInfer kernel.
    This updated version expects precomputed positional IDs and a fused cosine-sine cache.

    Inputs:
    - q, k (torch.Tensor):
        Tensors of shape [batch, seq_len, n_head, head_dim] (or a 3D variant)
        in half precision. Note: head_dim must be a multiple of 64.
    - position_ids (torch.Tensor):
        Precomputed tensor of positional indices; it is shared across calls in the graph.
    - cos_sin_cache (torch.Tensor):
        Precomputed fused tensor created by concatenating the first half of the cosine and sine
        components derived from the inv_freq.
    - is_neox (bool):
        Flag to indicate whether to invoke the FlashInfer kernel in Neox mode.

    Returns:
    A tuple of:
      - Rotated query tensor of the same shape and half precision as input.
      - Rotated key tensor of the same shape and half precision as input.
    """
    q_shape = q.shape
    k_shape = k.shape
    batch_size, seq_len = q_shape[:2]

    head_dim = cos_sin_cache.shape[-1]

    q_flat = q.view(batch_size * seq_len, -1)
    k_flat = k.view(batch_size * seq_len, -1)

    position_ids = position_ids.to(q.device)

    print("cos_sin_cache.shape", cos_sin_cache.shape)

    query_rotated_flash, key_rotated_flash = flashinfer.rope.apply_rope_with_cos_sin_cache(
        position_ids, q_flat, k_flat, head_dim, cos_sin_cache, is_neox=is_neox
    )
    query_rotated_flash = query_rotated_flash.view(q_shape)
    key_rotated_flash = key_rotated_flash.view(k_shape)
    return query_rotated_flash, key_rotated_flash


@apply_rope_with_input_pos_flashinfer.register_fake
def apply_rope_with_input_pos_flashinfer_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q, k
