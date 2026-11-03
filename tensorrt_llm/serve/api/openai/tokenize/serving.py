# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /_internal/tokenize``: token ids of a plain-text prompt."""

from typing import Any

from tensorrt_llm.serve._base.serving import BaseServing


class TokenizeServing(BaseServing):
    """``POST /_internal/tokenize``: token ids of a plain-text prompt."""

    def __init__(self, *, model: str, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
