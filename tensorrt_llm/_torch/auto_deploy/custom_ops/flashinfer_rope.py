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
        components derived from the inv_freq. Shape [seq_len, head_dim]. Must be float32.
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
    cos_sin_cache = cos_sin_cache.view(batch_size * seq_len, -1).float()
    # position_ids = torch.arange(batch_size * seq_len, device=q.device)

    query_rotated_flash, key_rotated_flash = flashinfer.rope.apply_rope_with_cos_sin_cache(
        position_ids, q_flat, k_flat, head_dim, cos_sin_cache, is_neox=is_neox
    )
    # query_rotated_flash, key_rotated_flash = apply_rope_with_cos_sin_cache_torch(
    #     position_ids, q_flat, k_flat, head_dim, cos_sin_cache, is_neox=is_neox
    # )
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


def apply_rope_with_cos_sin_cache_torch(
    positions: torch.Tensor,  # (nnz,)
    query: torch.Tensor,  # (nnz, num_q_heads * head_size)
    key: torch.Tensor,
    head_size: int,  # (nnz, num_k_heads * head_size)
    cos_sin_cache: torch.Tensor,  # (max_seq_len, 2*rotary_dim)
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch version of FlashInfer's apply_rope_with_cos_sin_cache.
    """
    nnz = query.shape[0]
    # figure out how many heads:
    num_qh = query.shape[1] // head_size
    num_kh = key.shape[1] // head_size

    # reshape into (nnz, heads, head_size)
    q = query.view(nnz, num_qh, head_size)
    k = key.view(nnz, num_kh, head_size)

    # look up cos & sin for each position
    # cache has shape (max_seq_len, 2*rotary_dim); first half = cos, second = sin
    rotary_dim2 = cos_sin_cache.shape[-1]
    half = rotary_dim2 // 2
    # cos_sin_cache[p] → length (2*half); we split it out to two tensors of shape (half,)
    cos_all = cos_sin_cache[:, :half]  # (max_seq_len, half)
    sin_all = cos_sin_cache[:, half:]  # (max_seq_len, half)

    # for each element in our nnz batch, pick its cos/sin
    positions = positions.to(torch.long)
    cos = cos_all[positions]  # (nnz, half)
    sin = sin_all[positions]  # (nnz, half)

    # now broadcast into shape (nnz, heads, half)
    cos = cos.view(nnz, 1, half)
    sin = sin.view(nnz, 1, half)

    def rotate(t: torch.Tensor) -> torch.Tensor:
        """
        Rotate the first 2*half dims of `t` according to cos/sin,
        and copy through any remaining dims untouched.
        """
        if not is_neox:
            # interleaved format: dims [0,1], [2,3], … each pair is (even,odd)
            t_even = t[..., 0::2]  # (..., half)
            t_odd = t[..., 1::2]  # (..., half)
            # apply complex rotation: (x + i y) → (x⋅cos – y⋅sin) + i(x⋅sin + y⋅cos)
            r_even = t_even * cos - t_odd * sin
            r_odd = t_even * sin + t_odd * cos
            # weave them back together
            out = torch.empty_like(t)
            out[..., 0::2] = r_even
            out[..., 1::2] = r_odd
            return out

        else:
            # Neox‐style: first `half` dims are the “real” part, next `half` dims are the “imag” part
            real = t[..., :half]  # (..., half)
            imag = t[..., half : 2 * half]  # (..., half)
            r_real = real * cos - imag * sin
            r_imag = real * sin + imag * cos
            # concatenate rotated + any trailing dims unchanged
            trailing = t[..., 2 * half :]
            return torch.cat([r_real, r_imag, trailing], dim=-1)

    q_rope = rotate(q)  # (nnz, num_qh, head_size)
    k_rope = rotate(k)  # (nnz, num_kh, head_size)

    # restore original flat shape
    return q_rope.view_as(query).type_as(query), k_rope.view_as(key).type_as(key)
