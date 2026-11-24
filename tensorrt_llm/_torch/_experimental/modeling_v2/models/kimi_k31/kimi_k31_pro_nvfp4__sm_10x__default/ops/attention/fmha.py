# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-target sparse MLA fallback admission."""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    CustomAttentionMask,
)


class TargetFallbackFmha(FallbackFmha):
    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        del k, v
        # A context phase served by the thop op with use_paged_context_fmha
        # enabled would silently corrupt the cached prefix wherever the fused
        # context kernel is absent. Raise rather than return False: no
        # library follows this one, and the error names the cause and the
        # remedy instead of the generic no-library message. Generation-only
        # batches never run the context path, so they pass; whether a batch
        # has a context phase is part of the FMHA cache key
        # (``context_batch_size``), so the admitted result cannot be reused
        # for a context batch.
        if metadata.num_contexts > 0:
            # Extra page sizes are supported by TRTLLM-Gen, not the thop
            # paged-context kernel. Refuse before the kernel can read KV.
            extra_page_sizes = getattr(
                metadata.kv_cache_manager, "trtllm_gen_extra_tokens_per_block", ()
            )
            if (
                extra_page_sizes
                and phase != FmhaPhase.GENERATION
                and metadata.use_paged_context_fmha
                and metadata.tokens_per_block in extra_page_sizes
            ):
                raise RuntimeError(
                    f"FallbackFmha cannot serve paged prefill with "
                    f"{metadata.tokens_per_block}-token KV pages. "
                    "Use a supported TRTLLM-Gen configuration or disable "
                    "chunked prefill, KV reuse and speculative decoding."
                )
            self._validate_paged_context_fmha(metadata)
        # A verify group may straddle a page boundary onto two CP ranks, so
        # its KV ownership is per-token. The fused thop path cannot express
        # that: its spec-dec mask and the per-sequence helix_is_inactive_rank
        # gate both assume the new KV entries are the trailing slots of one
        # rank's kv_len. Reject rather than run it silently wrong; being last
        # in the library list, this makes dispatch raise. Token-sparse MLA is
        # exempt: it attends only to its per-token Top-K rows, which the DSA
        # indexer bounds by ``helix_kv_bounds``, and the DSA metadata presents
        # one row per token as long as its kept entries
        # (``DSAtrtllmAttentionMetadata.token_major_gen_view``).
        if (
            metadata.helix_position_offsets is not None
            and metadata._helix_spec_tokens_valid
            and metadata.num_generations > 0
            and forward_args.sparse_runtime_params.sparse_attn_indices is None
            # Count generation tokens only: in a mixed batch ``q`` also holds
            # the context tokens, which would otherwise trip this on a batch
            # that has exactly one query token per generation sequence.
            and q.shape[0] - metadata.num_ctx_tokens > metadata.num_generations
        ):
            return False
        if q is not None and q.dtype == torch.float8_e4m3fn:
            return False
        if forward_args.attention_mask == CustomAttentionMask.CUSTOM:
            return False
        if not forward_args.update_kv_cache and not metadata.is_cross:
            return False
        return True


if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
