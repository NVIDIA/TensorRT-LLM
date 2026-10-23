# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-reduction causal MHA for deterministic prefill and generation."""

import os
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.kernels.deterministic_attention import deterministic_attention
from tensorrt_llm.bindings import DataType
from tensorrt_llm.functional import AttentionMaskType

from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from .interface import FmhaPhase

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class DeterministicFmha(FlashInferTrtllmGenFmha):
    """Reuse paged-KV preprocessing with a fixed FP32 per-query reduction."""

    _supports_deterministic = True

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        enabled = (
            os.getenv("FORCE_DETERMINISTIC") == "1"
            or os.getenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC") == "1"
        )
        return (
            enabled
            and not attn.is_mla_enable
            and attn.head_dim in (64, 128, 256)
            and super()._is_available(attn)
        )

    def _is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        if (
            q.dtype != torch.bfloat16
            or self._get_kv_cache_dtype(metadata) != DataType.BF16
            or metadata.is_cross
            or metadata.beam_width != 1
            or metadata.tokens_per_block not in (32, 64)
            or self.attn.num_heads // self.attn.num_kv_heads > self.MAX_HEADS_RATIO_GENERATION
            or metadata.use_spec_decoding
            or forward_args.attention_sinks is not None
            or AttentionMaskType(forward_args.mask_type) != AttentionMaskType.causal
            or self.attn.attention_chunk_size
            or forward_args.attention_window_size is None
            or forward_args.attention_window_size < metadata.max_seq_len
        ):
            return False
        return super()._is_supported(q, k, v, metadata, forward_args, phase=phase)

    def _run_attention(self, phase: FmhaPhase, **kwargs) -> None:
        query = kwargs["query"]
        fixed_q = kwargs.get("q_len_per_req") or 0
        seq_lens = kwargs["seq_lens"]
        if phase == FmhaPhase.CONTEXT:
            # PhasedFmha keeps the generation tail in the context length view.
            # The preprocessed Q, page tables and output contain context rows
            # only; launching for the tail would read/write past those rows.
            seq_lens = seq_lens[: kwargs["batch_size"]]
        deterministic_attention(
            query=query,
            key_cache=kwargs["kv_cache"][0],
            value_cache=kwargs["kv_cache"][1],
            block_tables=kwargs["block_tables"],
            seq_lens=seq_lens,
            cu_query_lens=kwargs.get("cum_seq_lens_q"),
            fixed_query_len=fixed_q,
            max_query_len=kwargs.get("max_q_len") or fixed_q,
            scale=kwargs["bmm1_scale"],
            shared_page_indices=kwargs["uses_shared_paged_kv_idx"],
            output=kwargs["out"],
        )
