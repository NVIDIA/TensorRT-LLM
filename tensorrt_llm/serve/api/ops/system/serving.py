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
"""/health, /version and /v1/models for every server role."""

from __future__ import annotations

from typing import Callable

from tensorrt_llm.serve.openai_protocol import ModelCard, ModelList
from tensorrt_llm.version import __version__ as VERSION


class SystemServing:
    """Answers the system endpoints from values the composition root injects.

    The health probe stays on the composition root (it reads executor state and
    may raise SIGINT); this class only calls it.
    """

    def __init__(self, *, model: str, health_probe: Callable[[], bool]) -> None:
        self._model = model
        self._health_probe = health_probe

    def is_healthy(self) -> bool:
        return self._health_probe()

    def version(self) -> dict[str, str]:
        return {"version": VERSION}

    def models(self) -> ModelList:
        # Built per request: ModelCard.created is the request time, as today.
        return ModelList(data=[ModelCard(id=self._model)])
