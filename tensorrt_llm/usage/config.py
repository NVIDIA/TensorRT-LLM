# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Telemetry configuration types.

Canonical location for TelemetryConfig and UsageContext. These are defined
here (in the usage package) rather than in llm_args.py so that the
dependency arrow points correctly: llm_args imports from usage, not
vice versa.

Imported by tensorrt_llm.llmapi.llm_args for use in BaseLlmArgs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _StrictUsageBaseModel(BaseModel):
    """Strict usage-local base. Same extra=forbid contract as llmapi StrictBaseModel."""

    # Keep usage import light. Do not import llmapi.utils here: it pulls torch
    # and HF deps. Needed contract stays same: extra fields forbidden.
    model_config = ConfigDict(extra="forbid")


class UsageContext(str, Enum):
    """Identifies how TRT-LLM was invoked for telemetry tracking."""

    UNKNOWN = "unknown"
    LLM_CLASS = "llm_class"
    CLI_SERVE = "cli_serve"
    CLI_BENCH = "cli_bench"
    CLI_EVAL = "cli_eval"


@dataclass(frozen=True)
class TelemetryField:
    """Field-local opt-in metadata for LLM API config telemetry capture."""

    kind: Literal["value", "categorical"] = "value"
    converter: Optional[Literal["allowlist"]] = None
    allowed_values: Optional[tuple[Any, ...]] = None

    @classmethod
    def categorical(cls, *allowed_values: Any) -> "TelemetryField":
        """Build a categorical allowlist field from the recognized values.

        Shorthand for the common bare-string allowlist case: marks the field
        categorical and pins capture to the explicit allowed values via the
        allowlist converter.
        """
        return cls(
            kind="categorical",
            converter="allowlist",
            allowed_values=tuple(allowed_values),
        )

    def as_json_schema_extra(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind}
        if self.converter is not None:
            data["converter"] = self.converter
        if self.allowed_values is not None:
            data["allowed_values"] = list(self.allowed_values)
        return data


class TelemetryConfig(_StrictUsageBaseModel):
    """Telemetry configuration for usage data collection.

    Controls opt-out behavior and tracks which entry point invoked TRT-LLM.
    """

    disabled: bool = Field(
        default=False,
        description="Disable anonymous usage telemetry collection. "
        "Can also be set via TRTLLM_NO_USAGE_STATS=1, TELEMETRY_DISABLED=true, "
        "DO_NOT_TRACK=1, or file ~/.config/trtllm/do_not_track.",
    )
    usage_context: UsageContext = Field(
        default=UsageContext.UNKNOWN,
        description="Identifies how TRT-LLM was invoked (CLI command vs Python API). "
        "Set automatically by CLI commands; defaults to UNKNOWN (promoted to "
        "LLM_CLASS by BaseLLM.__init__ for direct Python API usage).",
    )
