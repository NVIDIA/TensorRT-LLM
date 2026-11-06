# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only CED recovery contract shared by admission and input preparation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_request import LlmRequest


@dataclass
class EncoderCheckpoint:
    """Encoder prefix restored by normal KVCache resume without recomputation."""

    request_id: int
    cache_identity: int
    global_end: int


@dataclass
class EncoderReplay:
    """Issued after a stable Global claim; consumed only after model success."""

    request_id: int
    cache_identity: int
    global_end: int
    start: int
    consumed: bool = False

    @property
    def num_tokens(self) -> int:
        return 0 if self.consumed else self.global_end - self.start


def encoder_replay_tokens(request: "LlmRequest") -> int:
    plan = getattr(request, "py_ced_replay", None)
    return plan.num_tokens if isinstance(plan, EncoderReplay) else 0


def uses_private_decoder_cache(model, cache_manager) -> bool:
    """Whether the model's replay boundary uses private Decoder working pages."""
    has_private_swa_suffix = getattr(cache_manager, "has_private_swa_suffix", None)
    return (
        getattr(model, "ced_kv_precompute", False) is True
        and getattr(model, "decoder_replay_split", None) is not None
        and has_private_swa_suffix is not None
        and has_private_swa_suffix(model.decoder_replay_split)
    )


def requires_full_decoder_prefill(request: "LlmRequest") -> bool:
    """Requests whose consumers need every context row."""
    return bool(
        getattr(request, "py_return_context_logits", False)
        or getattr(request, "py_additional_outputs", None)
        or getattr(request, "py_multimodal_data", None) is not None
    )
