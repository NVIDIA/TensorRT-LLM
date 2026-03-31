# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    position_ids = position_ids.view(-1).to(device=q.device, dtype=torch.int32)
    num_nnz = position_ids.shape[0]

    if q.is_contiguous() and k.is_contiguous():
        # Standard path: flatten to 2D and use the public FlashInfer API.
        q_flat = q.view(num_nnz, -1)
        k_flat = k.view(num_nnz, -1)
        q_rope, k_rope = flashinfer.rope.apply_rope_with_cos_sin_cache(
            position_ids, q_flat, k_flat, head_dim, cos_sin_cache, is_neox=is_neox
        )
    else:
        # Strided path (e.g. MLA split_with_sizes outputs where the head dim
        # is non-contiguous): flatten batch+seq dims only, keep (H, D) as-is
        # so we never need to materialise a contiguous copy.
        q_3d = q.flatten(0, -3)  # [B, S, H, D] -> [nnz, H, D]
        k_3d = k.flatten(0, -3)
        q_rope = torch.empty_like(q_3d)
        k_rope = torch.empty_like(k_3d)
        flashinfer.rope._apply_rope_pos_ids_cos_sin_cache(
            q=q_3d,
            k=k_3d,
            q_rope=q_rope,
            k_rope=k_rope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=position_ids,
            interleave=(not is_neox),
        )

    return q_rope.view(q_shape), k_rope.view(k_shape)


@apply_rope_with_input_pos_flashinfer.register_fake
def apply_rope_with_input_pos_flashinfer_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k)
