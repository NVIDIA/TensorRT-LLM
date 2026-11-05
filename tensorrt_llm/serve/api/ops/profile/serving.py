# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /start_profile`` and ``POST /stop_profile``."""

from typing import Any

from tensorrt_llm.serve._base.serving import BaseServing


class ProfileServing(BaseServing):
    """Starts and stops the backend engine's runtime profiler."""

    def __init__(self, *, model: str, generator: Any) -> None:
        self.model = model
        self.generator = generator
