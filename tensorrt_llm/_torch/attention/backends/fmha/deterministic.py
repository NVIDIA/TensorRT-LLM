# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-reduction causal MHA for deterministic prefill and generation."""

import math
import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.kernels.deterministic_attention import (
    deterministic_attention,
    split_workspace_bytes,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from .interface import FmhaPhase

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class DeterministicFmha(FlashInferTrtllmGenFmha):
    """Reuse paged-KV preprocessing with a fixed per-query reduction."""

    _supports_deterministic = True
    # Decode runs its key segments in parallel when the attention workspace can
    # hold their partial results. Larger requests keep the serial layout, which
    # gives the same bits.
    _MAX_SPLIT_WORKSPACE_BYTES = 256 << 20

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        enabled = (
            os.getenv("FORCE_DETERMINISTIC") == "1"
            or os.getenv("FORCE_ATTENTION_KERNEL_DETERMINISTIC") == "1"
        )
        if not enabled:
            return False
        if attn.is_mla_enable:
            reason = "MLA attention"
        elif attn.head_dim not in (64, 128, 256):
            reason = f"head dimension {attn.head_dim}"
        elif not super()._is_available(attn):
            reason = "the TRTLLM-Gen FMHA prerequisites are missing"
        else:
            return True
        logger.warning_once(
            "Deterministic attention is requested, but the deterministic FMHA library "
            f"cannot serve this layer: {reason}. Attention falls back to a kernel whose "
            "reduction order depends on the batch.",
            key=("deterministic_fmha_unavailable", reason),
        )
        return False

    def _unsupported_reason(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
    ) -> str | None:
        heads_ratio = self.attn.num_heads // self.attn.num_kv_heads
        if q.dtype != torch.bfloat16:
            return f"query dtype {q.dtype}"
        if self._get_kv_cache_dtype(metadata) != DataType.BF16:
            return f"KV cache dtype {self._get_kv_cache_dtype(metadata)}"
        if metadata.is_cross:
            return "cross attention"
        if metadata.beam_width != 1:
            return f"beam width {metadata.beam_width}"
        if metadata.tokens_per_block not in (32, 64):
            return f"KV page size {metadata.tokens_per_block}"
        if heads_ratio > self.MAX_HEADS_RATIO_GENERATION:
            return f"{heads_ratio} query heads per KV head"
        if metadata.use_spec_decoding:
            return "speculative decoding"
        if forward_args.attention_sinks is not None:
            return "attention sinks"
        if AttentionMaskType(forward_args.mask_type) != AttentionMaskType.causal:
            return f"mask type {AttentionMaskType(forward_args.mask_type).name}"
        if self.attn.attention_chunk_size:
            return "chunked attention"
        if (
            forward_args.attention_window_size is None
            or forward_args.attention_window_size < metadata.max_seq_len
        ):
            return "a sliding attention window"
        return None

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
        reason = self._unsupported_reason(q, metadata, forward_args)
        if reason is None and not super()._is_supported(
            q, k, v, metadata, forward_args, phase=phase
        ):
            reason = "the TRTLLM-Gen paged-KV preprocessing does not support this batch"
        if reason is None:
            return True
        logger.warning_once(
            "Deterministic attention is requested, but the deterministic FMHA library "
            f"does not support this batch: {reason}. Attention falls back to a kernel "
            "whose reduction order depends on the batch.",
            key=("deterministic_fmha_unsupported", reason),
        )
        return False

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        workspace: torch.Tensor,
    ) -> None:
        super().prepare_workspace(q, k, v, metadata, forward_args, workspace)
        required = split_workspace_bytes(
            metadata.max_num_sequences or metadata.max_num_requests,
            self.attn.num_heads,
            self.attn.num_kv_heads,
            self.attn.head_dim,
            metadata.max_seq_len,
        )
        current = workspace.numel() * workspace.element_size()
        capturing = metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing()
        if current < required <= self._MAX_SPLIT_WORKSPACE_BYTES and not capturing:
            workspace.resize_((math.ceil(required / workspace.element_size()),))

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
            workspace=kwargs.get("workspace_buffer") if phase == FmhaPhase.GENERATION else None,
        )
