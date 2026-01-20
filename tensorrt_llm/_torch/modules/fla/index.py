# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/index.py
# -*- coding: utf-8 -*-

import torch
import triton

from tensorrt_llm._torch.modules.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor,
                          chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([
        torch.arange(n)
        for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    ])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor,
                          chunk_size: int) -> torch.LongTensor:
    return torch.cat([
        cu_seqlens.new_tensor([0]),
        triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    ]).cumsum(-1)
