# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime control: ``/release_memory``, ``/resume_memory`` and ``/memory_status``."""

from typing import Any, Optional

from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.runtime_control_auth import RuntimeControlReplayCache


class RuntimeControlServing(BaseServing):
    """Memory checkpointing of the engine behind signed, replay-protected requests."""

    def __init__(
            self, *, model: str, generator: Any,
            runtime_control_api_key: Optional[str],
            runtime_control_replay_cache: RuntimeControlReplayCache) -> None:
        self.model = model
        self.generator = generator
        self._runtime_control_api_key = runtime_control_api_key
        self._runtime_control_replay_cache = runtime_control_replay_cache
