# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration shared by text watermark generation and detection."""

from typing import Literal

from pydantic import Field

from tensorrt_llm.llmapi.utils import StrictBaseModel


class WatermarkConfig(StrictBaseModel):
    """Configure a context-keyed watermark for ordinary token sampling.

    The generation and detection configurations must match, including the
    tokenizer used to produce the token IDs. Keep the key private.
    """

    algorithm: Literal["gumbel", "dual_key_gumbel"] = Field(
        default="gumbel", description="Text watermark sampling algorithm."
    )
    key: int = Field(
        ge=0,
        le=2**64 - 1,
        strict=True,
        repr=False,
        description="Secret unsigned 64-bit key shared by generation and detection.",
        json_schema_extra={"telemetry": {"exclude": True}},
    )
    context_width: int = Field(
        default=4,
        ge=1,
        le=16,
        strict=True,
        description="Number of preceding token IDs used to derive each watermark seed.",
    )
    deduplicate_contexts_max_history: int = Field(
        default=8192,
        ge=1,
        strict=True,
        description="Maximum number of recent contexts excluded from repeated watermarking.",
    )
    alpha: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Mixing parameter for dual-key Gumbel watermark sampling.",
    )
