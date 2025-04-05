from typing import Tuple

import flashinfer
import torch


@torch.library.custom_op("rope::flashinfer", mutates_args=())
def apply_rope_with_input_pos_flashinfer(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
