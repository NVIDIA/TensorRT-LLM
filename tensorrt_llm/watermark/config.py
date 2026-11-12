# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Engine-level configuration for text watermarking."""

from typing import Literal, Optional

from pydantic import Field, field_validator

from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.logger import logger

# Context widths above this make the mark fragile under small edits. The limit
# is advisory; MAX_CONTEXT_WIDTH is what the device buffers are sized for.
RECOMMENDED_MAX_CONTEXT_WIDTH = 16
MAX_CONTEXT_WIDTH = 64


class WatermarkConfig(StrictBaseModel):
    """Configuration of the statistical text watermark applied while sampling.

    The watermark is keyed: the same ``key`` and sampling settings are needed
    to detect it later. ``key`` is a secret and is excluded from ``repr``.
    """

    algorithm: Literal["gumbel", "gumbel_dual_key"] = Field(
        default="gumbel",
        description="Watermarking algorithm. 'gumbel' replaces the random draw of "
        "the final token selection with a keyed Gumbel-max draw, which leaves the "
        "output distribution unchanged on average over keys. 'gumbel_dual_key' is the "
        "same for ordinary decoding and additionally supports speculative decoding with "
        "rejection sampling: draft tokens are drawn under the key, and the tokens "
        "that replace rejected drafts (or follow the last accepted one) under a second "
        "key derived from it, which keeps the output distribution exact.",
    )
    key: int = Field(
        description="Secret key, an integer in [0, 2**64). Detection needs the same value.",
        ge=0,
        lt=2**64,
        repr=False,
        # The key is a secret, so it must never enter usage telemetry.
        json_schema_extra={"telemetry": {"exclude": True}},
    )
    prf: Literal["philox"] = Field(
        default="philox",
        description="Pseudo-random function that turns (key, context) into per-token "
        "random numbers. Fast and parallel, but not a cryptographic PRF: it does not "
        "protect against key recovery or forgery.",
    )
    context_width: int = Field(
        default=4,
        ge=1,
        le=MAX_CONTEXT_WIDTH,
        description="Number of preceding tokens that seed each sampling decision. "
        "Larger values make the mark less robust to edits.",
    )
    deduplicate_contexts: Literal["none", "single_turn", "all"] = Field(
        default="single_turn",
        description="How repeated contexts are treated. A repeated context would "
        "repeat its token and be counted as independent evidence, so it falls back "
        "to ordinary sampling. 'none' disables this; 'single_turn' deduplicates "
        "within the tokens this request generates; 'all' also searches the prompt (for "
        "instance earlier turns of a conversation) and leaves the first context_width "
        "generated tokens unwatermarked, so short answers do not share a random "
        "context across requests.",
    )
    deduplicate_contexts_max_history: Optional[int] = Field(
        default=8192,
        ge=1,
        description="How many generated positions back the repeated-context search "
        "reaches. None means the whole generation.",
    )
    allow_target_only_watermarking: bool = Field(
        default=False,
        description="Only relevant to rejection-sampling speculative decoding. "
        "Accepted draft tokens are then not watermarked, so the signal weakens in "
        "proportion to the acceptance rate.",
    )
    allow_request_opt_out: bool = Field(
        default=True,
        description="Whether a request may disable watermarking with "
        "SamplingParams.watermarking=False. Set to False to enforce the watermark "
        "on every request that can carry one.",
    )

    @field_validator("context_width")
    @classmethod
    def _warn_wide_context(cls, value: int) -> int:
        if value > RECOMMENDED_MAX_CONTEXT_WIDTH:
            logger.warning(
                f"watermark context_width={value} exceeds "
                f"{RECOMMENDED_MAX_CONTEXT_WIDTH}; the mark becomes fragile under edits."
            )
        return value
