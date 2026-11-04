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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py

from typing import Any, List, Literal, Optional, Union

import xgrammar
from pydantic import (BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt,
                      field_validator, model_serializer, model_validator)
from typing_extensions import TypeAlias

from tensorrt_llm.kv_hints import KvHintsEnvelope
from tensorrt_llm.kv_hints.protocol import admit_kv_hints
from tensorrt_llm.llmapi import DisaggScheduleStyle

# Beam-search stopping mode as exposed over HTTP, mirroring the HuggingFace
# Transformers interface. The engine encodes it as an integer, but that encoding
# is an implementation detail and is kept out of the public schema.
EarlyStopping: TypeAlias = Union[bool, Literal["never"]]


def ensure_request_chat_template_allowed(request: Any,
                                         allow_request_chat_template: bool):
    if (getattr(request, "chat_template", None) is not None
            and not allow_request_chat_template):
        raise ValueError(
            "chat_template cannot be supplied per request unless request-level "
            "chat templates are enabled at server startup.")


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields & allow to initialize by both alias and field name
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("kv_hints", mode="before", check_fields=False)
    @classmethod
    def admit_hint_envelope(cls, value: Any) -> Optional[KvHintsEnvelope]:
        """Apply optional-hint admission before nested schema validation.

        Args:
            value: The decoded hint field from a trusted ingress.

        Returns:
            Supported metadata or None for ignored hints, including malformed
            envelopes. Validation of other request fields remains unchanged.
        """
        return admit_kv_hints(value)


class SpeculativeDecodingStats(OpenAIBaseModel):
    """Per-request speculative-decoding acceptance for one generated sequence.

    Opt-in: emitted only when the server sets per_request_spec_decode_stats.
    Absent entirely when the request drafted nothing. PyTorch backend only.

    ``mean acceptance length`` is deliberately not a field here: it is already
    reported per choice as ``avg_decoded_tokens_per_iter``, and duplicating it
    would let the two drift. Consumers derive it as
    ``1 + total_accepted_draft_tokens / num_spec_steps``.

    Three identities hold on every emitted record, and a consumer may rely on
    them:

    * ``sum(acceptance_histogram) == num_spec_steps``
    * ``sum(j * acceptance_histogram[j]) == total_accepted_draft_tokens``
    * ``total_accepted_draft_tokens <= total_draft_tokens``
    """

    acceptance_rate: float = Field(
        description="Accepted draft tokens divided by proposed draft tokens.")
    total_accepted_draft_tokens: int = Field(
        description="Draft tokens accepted across the request, excluding the "
        "always-accepted bonus token.")
    total_draft_tokens: int = Field(
        description="Draft tokens proposed across the request. For tree "
        "drafting this counts paths, not tree nodes, mirroring the "
        "getMaxDraftPathLen clamp in updateNumTokensPerIteration.")
    num_spec_steps: int = Field(
        description="Verify steps performed for the request. Equals the sum of "
        "acceptance_histogram.")
    acceptance_histogram: List[int] = Field(
        description="Dense histogram indexed by accepted-draft count: entry j "
        "is the number of verify steps that accepted exactly j draft tokens. "
        "Tree-agnostic -- it records output lengths per step and encodes no "
        "parent/child structure.")
    num_spec_tokens: Optional[int] = Field(
        default=None,
        description="Maximum draft length per step, when the run has a fixed "
        "bound. None under draft_len_schedule, where the bound varies by batch "
        "size.")


class _OmitsAbsentSpecDecodeStats(OpenAIBaseModel):
    """Drops ``speculative_decoding`` from serialized output when it is absent.

    Per-request spec-decode stats are off by default, so without this every
    response on the paths that serialize with a plain ``model_dump()`` -- the
    non-streaming chat and completions responses, and the completions stream
    (``exclude_unset=False``) -- would gain ``"speculative_decoding": null`` for
    every user, whether or not they enabled ``per_request_spec_decode_stats``.

    Scoped to this one field on purpose. Blanket ``exclude_none`` would also
    strip unrelated optional fields that clients may rely on being present, and
    the dump calls are spread across the serving layer rather than funnelled
    through one place where an ``exclude=`` argument could be applied.
    """

    @model_serializer(mode="wrap")
    def _omit_absent_spec_decode_stats(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict) and data.get("speculative_decoding",
                                               ...) is None:
            data.pop("speculative_decoding", None)
        return data


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = False


class PromptTokensDetails(OpenAIBaseModel):
    cached_tokens: int = 0


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokensDetails] = None


class ResponseFormat(OpenAIBaseModel):
    type: Literal["text", "json", "json_schema", "json_object", "regex", "ebnf",
                  "structural_tag"]
    schema: Optional[dict] = None
    json_schema: Optional[dict] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    format: Optional[xgrammar.structural_tag.Format] = None


class DisaggregatedParams(OpenAIBaseModel):
    request_type: str
    first_gen_tokens: Optional[List[int]] = None
    first_gen_log_probs: Optional[List] = None
    first_gen_logits: Optional[List] = None
    ctx_request_id: Optional[int] = None
    encoded_opaque_state: Optional[str] = None
    draft_tokens: Optional[List[int]] = None
    disagg_request_id: Optional[int] = None
    ctx_dp_rank: Optional[int] = None
    ctx_info_endpoint: Optional[str] = None
    schedule_style: Optional[DisaggScheduleStyle] = None
    ctx_usage: Optional[UsageInfo] = None
    remote_tail_start: Optional[NonNegativeInt] = Field(
        default=None,
        description=
        "Prompt-token boundary transferred by a remote-tail context worker.")
    multimodal_positions: Optional[List[NonNegativeInt]] = Field(
        default=None,
        description="Prompt-token offset of each multimodal item.")
    multimodal_lengths: Optional[List[PositiveInt]] = Field(
        default=None, description="Prompt-token count of each multimodal item.")
    # TODO(TRTLLM-12407): Multimodal E/PD over trtllm-serve needs these protocol fields too:
    # encoder embedding handles, multimodal hashes, and optional mRoPE handles.
    # Add them here and in to_disaggregated_params()/to_llm_disaggregated_params()
    # before routing MM encoder -> context -> generation through OpenAI protocol.
    # Orchestrator -> context-worker instruction: return prompt_token_ids as a
    # base64 int32 buffer (prompt_token_ids_b64) instead of a JSON int array.
    return_prompt_token_ids_b64: bool = False
    # Context worker -> generation worker: the reasoning mode the context
    # worker read off the prompt it rendered. The generation worker only sees
    # prompt_token_ids, so it cannot resolve this for itself.
    resolved_thinking: Optional[bool] = None
    # Orchestrator -> generation worker: seconds since arrival at the orchestrator.
    orig_elapsed_s: Optional[float] = None

    @model_validator(mode="after")
    def validate_multimodal_spans(self) -> "DisaggregatedParams":
        if (self.multimodal_positions is None) != (self.multimodal_lengths
                                                   is None):
            raise ValueError(
                "multimodal_positions and multimodal_lengths must be provided together"
            )
        if (self.multimodal_positions is not None and len(
                self.multimodal_positions) != len(self.multimodal_lengths)):
            raise ValueError(
                "multimodal_positions and multimodal_lengths must have the same length"
            )
        return self


class ConversationParams(OpenAIBaseModel):
    model_config = ConfigDict(extra="forbid",
                              populate_by_name=True,
                              validate_assignment=True)

    conversation_id: str = Field(
        description=("Stable multi-turn conversation id used for routing"), )

    # Forwarded as a header; excluded from the request body for older workers.
    subagent_affinity_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description=("Server-private parent-session id for sub-agent routing "
                     "affinity; forwarded to workers as a header, never in the "
                     "body."))

    @field_validator("conversation_id", mode="before")
    @classmethod
    def validate_conversation_id(cls, value: Any) -> str:
        if value is None:
            raise ValueError("conversation_id must be non-empty")
        conversation_id = str(value).strip()
        if not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        return conversation_id


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    # HTTP status by default; a machine-readable string for conditions that
    # define one (e.g. "context_length_exceeded", matching OpenAI).
    code: Union[int, str]
