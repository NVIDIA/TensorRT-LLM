# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tuner's system prompt: the worker, ending as a turn of the tune campaign ends."""

from __future__ import annotations

from ...operators.coverage import MODELING_V2_PACKAGE
from ..decisions import names as decision_names
from ._render import render_template
from .worker import worker_prompt

_ENDING = render_template(
    "tuner_ending.md.j2",
    MODELING_V2_PACKAGE=MODELING_V2_PACKAGE,
    decisions=" | ".join(decision_names("tuner")),
)

SYSTEM_PROMPT = worker_prompt(_ENDING)

__all__ = ["SYSTEM_PROMPT"]
