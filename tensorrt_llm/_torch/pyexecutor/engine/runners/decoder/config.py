# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved settings for scheduled decoder execution."""

from dataclasses import dataclass

import torch

from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    DecodingBaseConfig,
    PrefillCudaGraphBackend,
)

from ..interface import RunnerConfig


@dataclass(frozen=True)
class DecoderRunnerConfig(RunnerConfig):
    """Capacities and graph/compile settings resolved by the engine."""

    dtype: torch.dtype
    enable_attention_dp: bool
    disable_overlap_scheduler: bool
    is_encode_only: bool
    is_spec_decode: bool
    spec_config: DecodingBaseConfig | None
    max_draft_len: int
    max_total_draft_tokens: int
    max_draft_loop_tokens: int
    original_max_draft_len: int
    original_max_total_draft_tokens: int
    spec_dec_max_total_draft_tokens: int
    num_seq_slots: int | None
    cuda_graph_config: CudaGraphConfig | None
    cuda_graph_batch_sizes: list[int]
    cuda_graph_padding_enabled: bool
    max_cuda_graph_batch_size: int
    prefill_cuda_graph_backend: PrefillCudaGraphBackend
    prefill_cuda_graph_num_tokens: list[int]
    enable_in_graph_sampling: bool
    torch_compile_enabled: bool
    torch_compile_piecewise_cuda_graph: bool
    torch_compile_prefill_only: bool
    use_mrope: bool
    is_multimodal: bool
    mm_encoder_cache_enabled: bool
    enable_autotuner: bool
    cuda_graph_specialize_lora: bool
