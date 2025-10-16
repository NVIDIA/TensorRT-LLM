from typing import Tuple

import flashinfer
import torch


@torch.library.custom_op("auto_deploy::flashinfer_rope", mutates_args=())
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
        Precomputed tensor of positional indices indicating idx in cos_sin_cache for each token;
        Shape [batch, seq_len] or [batch * seq_len]
    - cos_sin_cache (torch.Tensor):
        Precomputed fused tensor created by concatenating the first half of the cosine and sine
        components derived from the inv_freq. Shape [max_seq_len, head_dim]. Must be float32.
    - is_neox (bool):
        Flag to indicate whether to invoke the FlashInfer kernel in Neox mode.

    Returns:
    A tuple of:
      - Rotated query tensor of the same shape and half precision as input.
      - Rotated key tensor of the same shape and half precision as input.
    """
    q_shape = q.shape
    k_shape = k.shape
    head_dim = cos_sin_cache.shape[-1]

    position_ids = position_ids.view(-1).to(q.device).int()  # flashinfer requires int
    num_nnz = position_ids.shape[0]

    q_flat = q.view(num_nnz, -1)
    k_flat = k.view(num_nnz, -1)

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
    return torch.empty_like(q), torch.empty_like(k)
