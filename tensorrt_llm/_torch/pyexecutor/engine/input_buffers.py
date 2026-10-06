# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner-owned storage for token input preparation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InputBuffers:
    """Reusable input storage allocated and retained by a runner.

    Token, position and gather buffers are one-dimensional integer tensors;
    draft tokens use the configured speculative capacity. Cache indirection is
    local attention staging with shape [batch capacity, beam width, sequence
    length], distinct from the sampler's per-call input.

    Tensor contents remain writable, but storage must remain stable while
    metadata or captured graphs use it. The runner synchronizes input reuse
    and releases graphs before replacing or disposing of their storage.
    """

    input_ids_cuda: torch.Tensor
    position_ids_cuda: torch.Tensor
    cache_indirection: torch.Tensor | None = None
    gather_ids_cuda: torch.Tensor | None = None
    draft_tokens_cuda: torch.Tensor | None = None

    @classmethod
    def allocate(
        cls,
        *,
        max_num_tokens: int,
        max_batch_size: int,
        max_beam_width: int,
        max_seq_len: int,
        use_cache_indirection: bool,
        max_num_draft_tokens: int | None = None,
    ) -> "InputBuffers":
        """Allocate from resolved capacities before warmup or graph capture.

        ``max_num_draft_tokens=None`` disables speculative storage. Zero still
        allocates a gather buffer and an empty draft buffer for speculative
        models whose configured draft capacity is zero.
        """
        return cls(
            input_ids_cuda=torch.empty(max_num_tokens, dtype=torch.int, device="cuda"),
            position_ids_cuda=torch.empty(max_num_tokens, dtype=torch.int, device="cuda"),
            cache_indirection=(
                torch.zeros(
                    (max_batch_size, max_beam_width, max_seq_len),
                    dtype=torch.int32,
                    device="cuda",
                )
                if use_cache_indirection and max_beam_width > 1
                else None
            ),
            gather_ids_cuda=(
                torch.empty(max_num_tokens, dtype=torch.int, device="cuda")
                if max_num_draft_tokens is not None
                else None
            ),
            draft_tokens_cuda=(
                torch.empty(max_num_draft_tokens, dtype=torch.int, device="cuda")
                if max_num_draft_tokens is not None
                else None
            ),
        )
