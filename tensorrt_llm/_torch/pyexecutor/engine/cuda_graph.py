# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variant-independent CUDA Graph helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tensorrt_llm.llmapi.llm_args import BaseCudaGraphConfig, CudaGraphConfig
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
        CUDAGraphRunner,
        EncoderCUDAGraphRunner,
    )

__all__ = [
    "CudaGraphPlan",
    "cuda_graph_capture",
    "cuda_graph_disabled",
    "filter_cuda_graph_batch_sizes",
    "filter_cuda_graph_num_tokens",
    "filter_cuda_graph_seq_lens",
    "filter_prefill_capture_num_tokens",
    "plan_cuda_graphs",
    "resolve_cuda_graph_batch_sizes",
]


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


def filter_cuda_graph_num_tokens(
    values: list[int],
    max_num_tokens: int,
    enable_padding: bool,
) -> list[int]:
    """Limit graph token buckets to the token budget, clamping when padding is enabled."""
    result: list[int] = []
    for index, value in enumerate(values):
        if value <= max_num_tokens:
            result.append(value)
            continue
        if enable_padding and (index == 0 or result[index - 1] != max_num_tokens):
            logger.warning(
                "CUDA graph padding is enabled, but one of the given CUDA "
                f"graph num_tokens ({value}) is larger than max_num_tokens "
                f"({max_num_tokens}). We will pad to {max_num_tokens}."
            )
            result.append(max_num_tokens)
        break
    return result


def filter_cuda_graph_seq_lens(
    values: list[int],
    max_seq_len: int,
    enable_padding: bool,
) -> list[int]:
    """Limit graph sequence-length buckets, clamping when padding is enabled."""
    result: list[int] = []
    for index, value in enumerate(values):
        if value <= max_seq_len:
            result.append(value)
            continue
        if enable_padding and (index == 0 or result[index - 1] != max_seq_len):
            logger.warning(
                "CUDA graph padding is enabled, but one of the given CUDA "
                f"graph seq_lens ({value}) is larger than max_seq_len "
                f"({max_seq_len}). We will pad to {max_seq_len}."
            )
            result.append(max_seq_len)
        break
    return result


def filter_prefill_capture_num_tokens(
    candidate_num_tokens: list[int],
    max_num_tokens: int,
    max_batch_size: int,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Cap prefill (piecewise or breakable) CUDA graph capture candidates at the engine's reachable
    `num_tokens` ceiling `max_batch_size * (max_seq_len - 1)`
    clamping user-requested sizes above it down to the ceiling.

    Each in-flight request must leave room for at least one decode token,
    so the ceiling is the largest forward-pass `num_tokens` the warmup
    builder can construct. Candidates above the ceiling cannot be
    recorded; clamping them down to the ceiling preserves the user's
    intent (a requested 128 becomes 127 when only 127 is recordable)
    without inventing capture sizes the user never asked
    for. Appending sizes beyond the user's list is harmful: runtime
    padding rounds iterations up to the nearest captured size, so a far
    appended ceiling (e.g. 65536 over a list topping at 13914) would
    make every iteration in the gap execute the full ceiling shape.

    Returns `(kept, unrecordable)` where `kept` is sorted ascending and
    deduped, with above-ceiling candidates clamped to the ceiling.
    `unrecordable` is the sorted unique set of input entries above the
    ceiling but within `max_num_tokens` (the clamped ones, reported so
    the caller's warning fires).
    """
    max_capturable_num_tokens = max(0, max_batch_size * (max_seq_len - 1))
    piecewise_capacity_limit = min(max_num_tokens, max_capturable_num_tokens)
    if piecewise_capacity_limit > 0:
        kept = sorted(
            {
                min(i, piecewise_capacity_limit)
                for i in candidate_num_tokens
                if 0 < i <= max_num_tokens
            }
        )
    else:
        kept = []
    unrecordable = sorted(
        {i for i in candidate_num_tokens if max_capturable_num_tokens < i <= max_num_tokens}
    )
    return kept, unrecordable


@dataclass(frozen=True)
class CudaGraphPlan:
    """The decode and prefill CUDA graph sizes an engine captures."""

    padding_enabled: bool
    batch_sizes: list[int]
    max_batch_size: int
    prefill_num_tokens: list[int]


def plan_cuda_graphs(
    cuda_graph_config: BaseCudaGraphConfig | None,
    prefill_capture_num_tokens: list[int] | None,
    *,
    max_batch_size: int,
    max_num_tokens: int,
    max_seq_len: int,
    decode_tokens_per_request: int,
) -> CudaGraphPlan:
    """Size the decode and prefill CUDA graphs from the config and the clamped capacity."""
    cuda_graph_batch_sizes = (
        cuda_graph_config.batch_sizes
        if cuda_graph_config
        else CudaGraphConfig.model_fields["batch_sizes"].default
    )
    cuda_graph_padding_enabled = (
        cuda_graph_config.enable_padding
        if cuda_graph_config
        else CudaGraphConfig.model_fields["enable_padding"].default
    )

    batch_sizes = (
        filter_cuda_graph_batch_sizes(
            cuda_graph_batch_sizes,
            max_batch_size,
            max_num_tokens,
            decode_tokens_per_request,
            cuda_graph_padding_enabled,
        )
        if cuda_graph_batch_sizes
        else []
    )

    prefill_cuda_graph_num_tokens = prefill_capture_num_tokens
    if prefill_cuda_graph_num_tokens is None:
        prefill_cuda_graph_num_tokens = cuda_graph_batch_sizes or []

    prefill_num_tokens, unrecordable = filter_prefill_capture_num_tokens(
        prefill_cuda_graph_num_tokens,
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if unrecordable:
        logger.warning(
            f"Skipping prefill CUDA graph capture for num_tokens="
            f"{unrecordable}: exceeds reachable ceiling "
            f"max_batch_size*(max_seq_len-1)="
            f"{max(0, max_batch_size * (max_seq_len - 1))}. "
            f"Clamping them to the ceiling; raise max_seq_len for larger graphs."
        )

    return CudaGraphPlan(
        padding_enabled=cuda_graph_padding_enabled,
        batch_sizes=batch_sizes,
        max_batch_size=batch_sizes[-1] if batch_sizes else 0,
        prefill_num_tokens=prefill_num_tokens,
    )


def resolve_cuda_graph_batch_sizes(
    batch_sizes: tuple[int, ...], max_batch_size: int, *, pad_to_limit: bool
) -> tuple[int, ...]:
    """Resolve scheduling batch sizes, optionally including the requested limit."""
    bounded_sizes = [size for size in batch_sizes if size <= max_batch_size]
    if (
        pad_to_limit
        and any(size > max_batch_size for size in batch_sizes)
        and max_batch_size > 0
        and (not bounded_sizes or bounded_sizes[-1] != max_batch_size)
    ):
        bounded_sizes.append(max_batch_size)
    return tuple(bounded_sizes)


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
