# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concatenate every rank's rows of a tensor into the group-wide full set."""

from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def allgather(
    input: torch.Tensor,
    sizes: Optional[List[int]],
    group: List[int],
) -> torch.Tensor:
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """
    # Pure-metadata guards, each for a violation measured on this machine to
    # end in silent corruption or a process kill rather than an error.
    assert input.dim() >= 1, (
        "input must have at least one dimension; a 0-d tensor segfaults inside "
        "AllgatherOp::run_list"
    )
    assert input.is_contiguous(), (
        "input must be contiguous; the op reads it as packed memory and a "
        "strided view is silently gathered from the wrong elements"
    )
    assert sizes is None or len(sizes) == len(group), (
        f"len(sizes)={len(sizes) if sizes is not None else None} must equal "
        f"len(group)={len(group)}; the op sizes its output from `sizes` alone, "
        "so a short list silently drops the trailing ranks and a long one "
        "appends uninitialized rows"
    )
    return torch.ops.trtllm.allgather(input, sizes, group)
