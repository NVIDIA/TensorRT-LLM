from typing import Tuple

import flashinfer
import torch


@torch.library.custom_op("rope::flashinfer", mutates_args=())
def apply_rope_with_input_pos_flashinfer(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings (RoPE) to query and key tensors using the FlashInfer kernel.

    Inputs:
    - q, k (torch.Tensor):
        4D tensors of shape [batch, n_head, seq_len, head_dim] in half precision (torch.float16 or torch.bfloat16).
        Note: head_dim must be a multiple of 64.
    - cos, sin (torch.Tensor):
        Tensors of shape [seq_len, head_dim]. Only the first half of the last dimension (head_dim//2 values)
        is used. They are concatenated (cos[..., :head_dim//2] with sin[..., :head_dim//2]) to form the
        non-interleaved cos-sin cache required by the FlashInfer kernel.

    Internal Processing:
    - Transposes q and k to shape [batch, seq_len, n_head, head_dim] and flattens them to
        [batch * seq_len, n_head * head_dim].
    - Constructs positional indices by repeating [0, seq_len) for each batch.
    - Invokes the FlashInfer kernel in "Neox" mode using the flattened tensors, head_dim, and cos-sin cache.

    Returns:
    A tuple of:
        - Rotated query tensor of shape [batch, n_head, seq_len, head_dim] (converted to torch.float).
        - Rotated key tensor of shape [batch, n_head, seq_len, head_dim] (converted to torch.float).
    """

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    batch, seq_len, n_head, head_dim = q.shape
    device = q.device

    q_flat = q.view(batch * seq_len, n_head * head_dim)
    k_flat = k.view(batch * seq_len, n_head * head_dim)

    positions = torch.cat([torch.arange(seq_len, device=device) for _ in range(batch)])

    D = cos.shape[-1] // 2
    cos_sin_cache = torch.cat([cos[..., :D], sin[..., :D]], dim=-1)

    query_rotated_flash, key_rotated_flash = flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions, q_flat, k_flat, head_dim, cos_sin_cache, is_neox=True
    )

    query_rotated_flash = query_rotated_flash.view(batch, seq_len, n_head, head_dim).transpose(1, 2)
    key_rotated_flash = key_rotated_flash.view(batch, seq_len, n_head, head_dim).transpose(1, 2)

    return query_rotated_flash.to(torch.float), key_rotated_flash.to(torch.float)


@apply_rope_with_input_pos_flashinfer.register_fake
def apply_rope_with_input_pos_flashinfer_fake(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q, k
