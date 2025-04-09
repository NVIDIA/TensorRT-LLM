from typing import Tuple

import flashinfer
import torch


@torch.library.custom_op("rope::flashinfer", mutates_args=())
def apply_rope_with_input_pos_flashinfer(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings (RoPE) to query and key tensors using the FlashInfer kernel.

    Inputs:
    - q, k (torch.Tensor):
        Tensors of shape [batch, seq_len, n_head, head_dim] or [batch, seq_len, n_head*head_dim]
        in half precision (torch.float16 or torch.bfloat16). Note: head_dim must be a multiple of 64.
    - cos, sin (torch.Tensor):
        Tensors of shape [seq_len, head_dim]. For 3D Tensors, the first dimension is discarded.
        Only the first half of the last dimension (head_dim//2 values) is used.
        They are concatenated (cos[..., :head_dim//2] with sin[..., :head_dim//2]) to form the
        non-interleaved cos-sin cache required by the FlashInfer kernel.
    - is_neox (bool):
        Flag to indicate whether to invoke the FlashInfer kernel in Neox mode.

    Internal Processing:
    - Constructs positional indices by repeating [0, seq_len) for each batch.
    - Invokes the FlashInfer kernel with the provided `is_neox` flag.

    Returns:
    A tuple of:
        - Rotated query tensor of shape same as input in half precision.
        - Rotated key tensor of shape same as input in half precision.
    """
    cos = cos[0] if cos.dim() == 3 else cos
    sin = sin[0] if sin.dim() == 3 else sin

    q_shape = q.shape
    batch_size, seq_len = q_shape[:2]
    head_dim = cos.shape[-1]
    device = q.device

    q_flat = q.view(batch_size * seq_len, -1)
    k_flat = k.view(batch_size * seq_len, -1)

    positions = torch.arange(seq_len, device=device).view(1, -1).expand(batch_size, -1).flatten()
    cos_sin_cache = torch.cat([cos[..., : head_dim // 2], sin[..., : head_dim // 2]], dim=-1)

    query_rotated_flash, key_rotated_flash = flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions, q_flat, k_flat, head_dim, cos_sin_cache, is_neox=is_neox
    )
    query_rotated_flash = query_rotated_flash.view(q_shape)
    key_rotated_flash = key_rotated_flash.view(q_shape)
    return query_rotated_flash, key_rotated_flash


@apply_rope_with_input_pos_flashinfer.register_fake
def apply_rope_with_input_pos_flashinfer_fake(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q, k
