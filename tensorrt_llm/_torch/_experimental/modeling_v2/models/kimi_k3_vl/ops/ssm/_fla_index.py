# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# and https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/index.py
"""FLA's varlen chunk indices for the KDA prefill: ``(sequence, chunk)`` of every chunk."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.
    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.
    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.
    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    cache_entries = (
                        cache_entries[:i] + cache_entries[i + 1 :] + [(args, kwargs, last_result)]
                    )
                    return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
