# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variant-independent CUDA Graph helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
        CUDAGraphRunner,
        EncoderCUDAGraphRunner,
    )

__all__ = ["cuda_graph_capture", "cuda_graph_disabled", "filter_cuda_graph_batch_sizes"]


def filter_cuda_graph_batch_sizes(
    cuda_graph_batch_sizes: list[int],
    max_batch_size: int,
    max_num_tokens: int,
    tokens_per_request: int,
    enable_padding: bool,
) -> list[int]:
    """Drop graph batch sizes that exceed the request or token budget."""
    max_cuda_graph_batch_size = min(
        max_batch_size,
        max_num_tokens // tokens_per_request,
    )
    if max_cuda_graph_batch_size < 1:
        return []

    result: list[int] = []
    for index, batch_size in enumerate(cuda_graph_batch_sizes):
        if batch_size <= max_cuda_graph_batch_size:
            result.append(batch_size)
            continue
        if enable_padding and (index == 0 or result[index - 1] != max_cuda_graph_batch_size):
            logger.warning(
                "CUDA graph padding is enabled, but one of the given CUDA "
                f"graph batch sizes ({batch_size}) is larger than the "
                f"executor's max batch size ({max_cuda_graph_batch_size}). "
                f"We will pad batches to {max_cuda_graph_batch_size}."
            )
            result.append(max_cuda_graph_batch_size)
        break
    return result


@contextmanager
def cuda_graph_disabled(runner: CUDAGraphRunner | EncoderCUDAGraphRunner) -> Iterator[None]:
    """Temporarily disable graphs, restoring the backend's previous state."""
    enabled = runner.enabled
    runner.enabled = False
    try:
        yield
    finally:
        runner.enabled = enabled


@contextmanager
def cuda_graph_capture(runner: CUDAGraphRunner | EncoderCUDAGraphRunner) -> Iterator[None]:
    """Allow startup capture; the caller controls warmup and capture phase ordering."""
    warmup_only = runner.is_warmup_only
    with runner.allow_capture():
        try:
            yield
        finally:
            runner.is_warmup_only = warmup_only
