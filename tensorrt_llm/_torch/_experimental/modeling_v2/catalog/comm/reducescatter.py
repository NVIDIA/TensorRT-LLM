# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sum every rank's copy of a tensor and hand each rank back its own rows."""

from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def reducescatter(
    input: torch.Tensor,
    sizes: Optional[List[int]],
    group: List[int],
) -> torch.Tensor:
    """Sum `input` across every rank in `group`, then keep this rank's slice.

    Every rank passes a tensor of the same full shape; the elementwise sum is
    split along dim 0 in ascending rank order and rank `i` of the group gets
    slice `i`. `sizes` gives each rank's slice height in that order, or is
    `None` when the split is even. The result is a freshly allocated tensor on
    every rank; `input` is left untouched.
    """
    # Pure-metadata guards, each for a violation measured on this machine to
    # end in silent corruption, a wedge, or a process kill rather than an error.
    assert input.dim() >= 1, (
        "input must have at least one dimension; a 0-d tensor segfaults inside "
        "ReducescatterOp::run_list"
    )
    assert input.is_contiguous(), (
        "input must be contiguous; the op reads it as packed memory and a "
        "strided view is silently reduced over the wrong elements"
    )
    assert input.dtype is not torch.float8_e4m3fn, (
        "float8_e4m3fn is accepted by the op but summed as raw unsigned bytes, "
        "not as floats, so the result is meaningless; reduce in bf16/fp16/fp32 "
        "and quantize afterwards"
    )
    if sizes is None:
        assert input.shape[0] % len(group) == 0, (
            f"input.shape[0]={input.shape[0]} must be divisible by "
            f"len(group)={len(group)} when sizes is None; the op takes "
            "shape[0] // len(group) rows per rank and silently drops the remainder"
        )
    else:
        assert len(sizes) == len(group), (
            f"len(sizes)={len(sizes)} must equal len(group)={len(group)}; a short "
            "list raises on the highest rank and wedges the others, a long one "
            "reduces more rows than the input holds and returns garbage"
        )
        assert sum(sizes) == input.shape[0], (
            f"sum(sizes)={sum(sizes)} must equal input.shape[0]={input.shape[0]}; "
            "the op sizes the reduction from `sizes` alone, so a short split "
            "silently drops the trailing rows and a long one reads past the input"
        )
    return torch.ops.trtllm.reducescatter(input, sizes, group)
