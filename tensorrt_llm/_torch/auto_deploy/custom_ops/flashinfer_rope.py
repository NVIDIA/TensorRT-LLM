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
        2D tensors of shape [batch * seq_len, n_head * head_dim] in half precision (torch.float16 or torch.bfloat16).
        Note: head_dim must be a multiple of 64.
    - cos, sin (torch.Tensor):
        Tensors of shape [seq_len, head_dim]. Only the first half of the last dimension (head_dim//2 values)
        is used. They are concatenated (cos[..., :head_dim//2] with sin[..., :head_dim//2]) to form the
        non-interleaved cos-sin cache required by the FlashInfer kernel.

    Internal Processing:
    - Constructs positional indices by repeating [0, seq_len) for each batch.
    - Invokes the FlashInfer kernel in "Neox" mode.

    Returns:
    A tuple of:
        - Rotated query tensor of shape [batch * seq_len, n_head * head_dim](converted to torch.float).
        - Rotated key tensor of shape [batch * seq_len, n_head * head_dim] (converted to torch.float).
    """

    seq_len = cos.shape[0]
    batch = q.shape[0] // seq_len
    head_dim = cos.shape[1]
    device = q.device

    positions = torch.cat([torch.arange(seq_len, device=device) for _ in range(batch)])

    D = cos.shape[-1] // 2
    cos_sin_cache = torch.cat([cos[..., :D], sin[..., :D]], dim=-1)

    query_rotated_flash, key_rotated_flash = flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions, q, k, head_dim, cos_sin_cache, is_neox=True
    )

    return query_rotated_flash.to(torch.float), key_rotated_flash.to(torch.float)


@apply_rope_with_input_pos_flashinfer.register_fake
def apply_rope_with_input_pos_flashinfer_fake(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q, k
