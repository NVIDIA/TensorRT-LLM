# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

from tensorrt_llm.bindings import executor as tllme


class DisaggScheduleStyle(IntEnum):
    CONTEXT_FIRST = 0
    GENERATION_FIRST = 1


@dataclass(slots=True, kw_only=True)
class DisaggregatedParams:
    """Disaggregated serving parameters.

    Args:
        request_type (str): The type of request ("context_only" | "generation_only" | "context_and_generation")
        first_gen_tokens (List[int]): The first tokens of the generation request
        ctx_request_id (int): The context request id
        opaque_state(bytes): Any additional state needing to be exchanged between context and gen instances
        draft_tokens (List[int]): The draft tokens of the generation request
        disagg_request_id (int): The disaggregated request id, if set, both context and generation requests will use it
         as underlying request id.
        first_gen_log_probs (List): The logprobs for first_gen_tokens, produced during prefill.
         Each entry is a list (one per beam) of either ``TokenLogprobs`` (``list[dict[int, Logprob]]``,
         default format) or ``SimpleTokenLogprobs`` (``list[float]``, simple format).
        first_gen_logits (List): The generation logits for first_gen_tokens, produced during prefill.
         Each entry is a torch.Tensor of shape [num_tokens, vocab_size] (one per beam/sequence).
        ctx_usage (Dict[str, Any]): The context usage payload to preserve exact
         usage accounting on the generation server.
        remote_tail_start (int): Prompt-token boundary transferred by a remote-tail
         context worker. The generation worker replays the suffix starting here.
        multimodal (Dict[str, Any]): Versioned multimodal handoff. ``items``
         contains encoded item handles, actual output lengths and model metadata.
         Optional ``request_layout`` contains finalized prompt tokens and layout;
         optional ``decode_inputs`` contains model-required generation state.
    """

    request_type: Optional[str] = None
    # P-D Disaggregated Params
    first_gen_tokens: Optional[List[int]] = None
    first_gen_log_probs: Optional[List] = None
    first_gen_logits: Optional[List] = None
    ctx_request_id: Optional[int] = None
    opaque_state: Optional[bytes] = None
    draft_tokens: Optional[List[int]] = None
    # If disagg_request_id is set, both context and generation requests will use it as underlying request id.
    disagg_request_id: Optional[int] = None
    ctx_dp_rank: Optional[int] = None
    ctx_info_endpoint: Optional[str] = None
    schedule_style: Optional[DisaggScheduleStyle] = None
    ctx_usage: Optional[Dict[str, Any]] = None
    remote_tail_start: Optional[int] = None
    # Existing HTTP P/D span fields. Canonical E/P data uses request_layout.
    multimodal_positions: Optional[List[int]] = None
    multimodal_lengths: Optional[List[int]] = None

    # Encoded item data and optional finalized request layout. Live Params,
    # raw media, local cache references and CUDA events are never serialized.
    multimodal: Optional[Dict[str, Any]] = None

    @property
    def has_multimodal_embeddings(self) -> bool:
        """Whether the handoff contains encoder outputs to import for prefill."""
        return self.multimodal is not None and bool(self.multimodal["items"])

    def get_multimodal_embedding_handles(self) -> list[dict[str, Any]]:
        """Read primary item handles in their declared order."""
        if self.multimodal is None:
            return []
        return [item["embeddings"]["multimodal_embedding"] for item in self.multimodal["items"]]

    def get_context_phase_params(self) -> tllme.ContextPhaseParams:
        # Prefer disagg_request_id over ctx_request_id
        request_id = (
            self.disagg_request_id if self.disagg_request_id is not None else self.ctx_request_id
        )
        # `first_gen_tokens` is now required by bindings and cannot be None.
        first_gen_tokens = self.first_gen_tokens if self.first_gen_tokens is not None else []
        return tllme.ContextPhaseParams(
            first_gen_tokens,
            request_id,
            self.opaque_state,
            self.draft_tokens,
            self.ctx_dp_rank,
            self.ctx_info_endpoint,
        )

    def get_request_type(self) -> tllme.RequestType:
        if self.request_type == "context_only":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_ONLY
        elif self.request_type == "generation_only":
            return tllme.RequestType.REQUEST_TYPE_GENERATION_ONLY
        elif self.request_type == "context_and_generation":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        else:
            raise ValueError(
                f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                "context_and_generation"
            )

    def __post_init__(self):
        if (self.multimodal_positions is None) != (self.multimodal_lengths is None):
            raise ValueError(
                "multimodal_positions and multimodal_lengths must be provided together"
            )
        if self.multimodal_positions is not None:
            if len(self.multimodal_positions) != len(self.multimodal_lengths):
                raise ValueError(
                    "multimodal_positions and multimodal_lengths must have the same length"
                )
            if any(position < 0 for position in self.multimodal_positions):
                raise ValueError("multimodal_positions must be non-negative")
            if any(length <= 0 for length in self.multimodal_lengths):
                raise ValueError("multimodal_lengths must be positive")
        if self.multimodal is not None:
            if self.multimodal.get("schema_version") != 1:
                raise ValueError("Unsupported multimodal handoff schema_version; expected 1")
            if not isinstance(self.multimodal.get("items"), list):
                raise ValueError("Multimodal handoff items must be a list")
        if self.request_type is not None:
            self.request_type = self.request_type.lower()
            if self.request_type not in [
                "context_only",
                "generation_only",
                "context_and_generation",
            ]:
                raise ValueError(
                    f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                    "context_and_generation"
                )
