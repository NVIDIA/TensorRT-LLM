# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The RL control endpoints: ``/release_memory``, ``/resume_memory``, ``/update_weights``."""

from typing import Any, Optional

from tensorrt_llm.serve._base.serving import BaseServing


class RLServing(BaseServing):
    """Collective RPCs of an ``AsyncLLM`` behind the HMAC-signed RL control header."""

    def __init__(self, *, model: str, generator: Any,
                 rl_control_api_key: Optional[str]) -> None:
        self.model = model
        self.generator = generator
        self._rl_control_api_key = rl_control_api_key
