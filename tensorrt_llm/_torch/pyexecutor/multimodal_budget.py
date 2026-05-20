# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal encoder budget — first-class object for encoder-side limits.

Encapsulates the (currently compute-only) budget for one encoder forward
pass: how many attention tokens fit, how many items per batch. Used by
``_create_dummy_mm_context_request`` and
``PyTorchModelEngine._set_up_multimodal_encoder_attn_metadata`` to size
encoder workspaces during KV-cache profiling.

Future-extensible without breaking call sites: when an encoder-cache
layer lands, add a separate :class:`CacheBudget` and combine the two at
profile time. The current ``MultimodalEncoderBudget`` intentionally
exposes only compute-side limits to keep the two concepts orthogonal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Tuple

if TYPE_CHECKING:
    from ...inputs import Modality
    from ...inputs.registry import BaseMultimodalInputProcessor


@dataclass(frozen=True)
class MultimodalEncoderBudget:
    """Per-encoder-forward compute budget.

    Attributes:
        max_tokens_per_step: Encoder attention tokens (pre-merger) per
            single encoder forward. Matches
            ``AttentionMetadata.max_num_tokens`` on the encoder side and
            ``BaseLlmArgs.encoder_max_num_tokens``.
        max_items_per_step: Max distinct items (images/videos/audios)
            per encoder forward. Matches
            ``AttentionMetadata.max_num_requests`` on the encoder side
            and ``BaseLlmArgs.encoder_max_batch_size``.
    """

    max_tokens_per_step: int
    max_items_per_step: int

    @classmethod
    def from_llm_args(cls, llm_args) -> "MultimodalEncoderBudget":
        """Build from a ``BaseLlmArgs`` instance.

        Reads ``get_encoder_runtime_sizes()`` so the same fallback
        semantics (encoder-specific knob → LLM-side knob) are applied in
        one place.
        """
        items, tokens = llm_args.get_encoder_runtime_sizes()
        return cls(max_tokens_per_step=int(tokens), max_items_per_step=int(items))

    def llm_visible_tokens(self, spatial_merge_unit: int) -> int:
        """Encoder-token budget expressed in LLM-visible (post-merger) units.

        Used by ``_create_dummy_mm_context_request`` to clamp the
        per-request dummy prompt length so the resulting placeholders
        translate back into exactly ``max_tokens_per_step`` encoder
        attention tokens.
        """
        merge = max(int(spatial_merge_unit), 1)
        return max(self.max_tokens_per_step // merge, 1)

    def iter_modality_dummies(
        self,
        processor: "BaseMultimodalInputProcessor",
    ) -> Iterator[Tuple["Modality", Dict[str, int]]]:
        """Yield one ``(modality, size_kwargs)`` per supported modality.

        Each pair represents a worst-case single-item dummy for that
        modality under :attr:`max_tokens_per_step`. Consumers (typically
        ``_create_dummy_mm_context_request``) construct one dummy request
        per pair so multi-encoder models (e.g. Nemotron Nano VL2 with a
        separate audio encoder) have *every* encoder warmed up — not
        just the worst-case modality.

        Skips modalities whose :meth:`get_size_with_most_features`
        raises ``NotImplementedError``; subclasses opt in per modality.
        """
        modalities = getattr(processor, "supported_modalities", ())
        for modality in modalities:
            try:
                size = processor.get_size_with_most_features(
                    modality, max_tokens=self.max_tokens_per_step
                )
            except NotImplementedError:
                continue
            yield modality, size
