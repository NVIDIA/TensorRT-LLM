# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from torch._dynamo.eval_frame import OptimizedModule

from .embedding import LMHead
from .linear import TensorParallelMode


@dataclass(frozen=True)
class LogitsMemoryFootprint:
    """LM-head peak bytes per output row and whether only handoffs run."""

    bytes_per_row: int
    context_only: bool = False


@runtime_checkable
class LogitsMemoryProcessor(Protocol):
    def logits_memory_footprint(
        self, lm_head: LMHead, model_dtype: torch.dtype, is_last_pp_rank: bool
    ) -> LogitsMemoryFootprint: ...


@runtime_checkable
class LogitsMemoryProvider(Protocol):
    def get_logits_memory_footprint(self) -> LogitsMemoryFootprint | None: ...


def logits_memory_footprint_of(model: nn.Module) -> LogitsMemoryFootprint | None:
    """Read a model's explicit footprint, including through torch.compile."""
    if isinstance(model, OptimizedModule):
        model = model._orig_mod
    if isinstance(model, LogitsMemoryProvider):
        return model.get_logits_memory_footprint()
    return None


class LogitsMemoryWrapper:
    """Opt in to footprint delegation for models wrapping a text decoder."""

    llm: nn.Module

    def get_logits_memory_footprint(self) -> LogitsMemoryFootprint | None:
        return logits_memory_footprint_of(self.llm)


def lm_head_memory_footprint(
    lm_head: LMHead, model_dtype: torch.dtype, is_last_pp_rank: bool
) -> LogitsMemoryFootprint:
    """Bound the head, TP gather and FP32 cast, not later sampler copies."""
    output_dtype = (
        lm_head.output_dtype or lm_head.dtype or model_dtype if is_last_pp_rank else model_dtype
    )
    vocab_size = lm_head.vocab_size_padded
    output_bytes = vocab_size * output_dtype.itemsize
    peak_bytes = output_bytes
    if output_dtype != torch.float32:
        peak_bytes += vocab_size * torch.float32.itemsize
    if (
        is_last_pp_rank
        and lm_head.tp_mode == TensorParallelMode.COLUMN
        and lm_head.gather_output
        and lm_head.tp_size > 1
    ):
        peak_bytes = max(peak_bytes, 2 * output_bytes + output_bytes // lm_head.tp_size)
    return LogitsMemoryFootprint(bytes_per_row=peak_bytes)
