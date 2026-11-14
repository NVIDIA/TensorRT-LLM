# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def slots_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(tensor.shape[0], -1)


def replay_pools_supported(
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    state_tok: torch.Tensor,
    pending: torch.Tensor,
    heads: int,
    num_spec: int,
) -> bool:
    channels = heads * 128
    columns = 3 + num_spec
    if cs_q.dtype not in (torch.float32, torch.bfloat16):
        return False
    for cache in (cs_q, cs_k, cs_v):
        if (
            cache.ndim != 3
            or cache.dtype != cs_q.dtype
            or tuple(cache.shape[1:]) != (channels, columns)
            or cache.stride()[1:] != (1, channels)
            or cache.stride(0) < channels * columns
            or cache.stride(0) % 4 != 0
            or cache.data_ptr() % (4 * cache.element_size()) != 0
        ):
            return False
    return (
        state_tok.ndim == 5
        and tuple(state_tok.shape[1:]) == (3, num_spec, heads, 128)
        and state_tok.dtype == torch.float32
        and state_tok.stride()[1:] == (num_spec * channels, channels, 128, 1)
        and state_tok.stride(0) >= 3 * num_spec * channels
        and state_tok.stride(0) % 4 == 0
        and state_tok.data_ptr() % 16 == 0
        and pending.ndim == 1
        and pending.dtype == torch.int32
        and pending.stride(0) > 0
    )
