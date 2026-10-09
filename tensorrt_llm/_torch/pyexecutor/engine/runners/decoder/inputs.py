# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder input preparation state."""

from dataclasses import dataclass

import torch

from tensorrt_llm._utils import prefer_pinned

from .config import DecoderRunnerConfig


@dataclass(slots=True)
class DecoderInputBuffers:
    """Persistent buffers read by decoder execution.

    CUDA graphs record these addresses, so they are allocated once and only
    ever sliced or written in place.
    """

    draft_tokens_cuda: torch.Tensor | None = None
    gather_ids_cuda: torch.Tensor | None = None
    num_accepted_draft_tokens_cuda: torch.Tensor | None = None
    previous_pos_indices_cuda: torch.Tensor | None = None
    previous_pos_id_offsets_cuda: torch.Tensor | None = None
    previous_kv_lens_offsets_cuda: torch.Tensor | None = None
    previous_batch_indices_cuda: torch.Tensor | None = None
    input_ids_cuda: torch.Tensor | None = None
    position_ids_cuda: torch.Tensor | None = None
    # Host counter of the steady-state generation prepare cache.
    steady_gen_positions_pinned: torch.Tensor | None = None
    mrope_position_ids_cuda: torch.Tensor | None = None
    cache_indirection_attention: torch.Tensor | None = None

    def allocate(self, config: DecoderRunnerConfig, use_beam_search: bool) -> None:
        if config.is_spec_decode:
            max_num_draft_tokens = config.max_draft_loop_tokens * config.max_batch_size
            self.draft_tokens_cuda = torch.empty(
                (max_num_draft_tokens,), dtype=torch.int, device="cuda"
            )
            self.gather_ids_cuda = torch.empty(
                (config.max_num_tokens,), dtype=torch.int, device="cuda"
            )
            self.num_accepted_draft_tokens_cuda = torch.empty(
                (config.max_batch_size,), dtype=torch.int, device="cuda"
            )
            self.previous_pos_indices_cuda = torch.empty(
                (config.max_num_tokens,), dtype=torch.int, device="cuda"
            )
            self.previous_pos_id_offsets_cuda = torch.zeros(
                (config.max_num_tokens,), dtype=torch.int, device="cuda"
            )
            self.previous_kv_lens_offsets_cuda = torch.zeros(
                (config.max_batch_size,), dtype=torch.int, device="cuda"
            )
        self.previous_batch_indices_cuda = torch.empty(
            (config.max_num_tokens,), dtype=torch.int, device="cuda"
        )
        self.input_ids_cuda = torch.empty((config.max_num_tokens,), dtype=torch.int, device="cuda")
        self.position_ids_cuda = torch.empty(
            (config.max_num_tokens,), dtype=torch.int, device="cuda"
        )
        self.steady_gen_positions_pinned = torch.empty(
            (config.max_num_tokens,), dtype=torch.int, pin_memory=prefer_pinned()
        )
        if config.use_mrope:
            self.mrope_position_ids_cuda = torch.empty(
                (3, 1, config.max_num_tokens), dtype=torch.int, device="cuda"
            )
        if use_beam_search:
            self.cache_indirection_attention = torch.zeros(
                (
                    config.max_batch_size,
                    config.max_beam_width,
                    config.max_seq_len,
                ),
                device="cuda",
                dtype=torch.int32,
            )
