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
"""Serving classes of the Anthropic Messages dialect.

Messages, count_tokens and the Message Batches, served from values the
composition root injects; no class here reads a server attribute.
"""

from typing import Callable, Optional

from tensorrt_llm.serve.api.anthropic.batches import (AnthropicBatchStore,
                                                      BatchItemRunner)
from tensorrt_llm.serve.render import RenderResources


class AnthropicCountTokensServing:
    """Counts the input tokens of a Messages request over the rendered prompt."""

    def __init__(self, *, render_resources: Callable[[],
                                                     RenderResources]) -> None:
        # Called per request: the resources are read at call time, as the
        # chat route reads them.
        self._render_resources = render_resources


class AnthropicBatchesServing:
    """The Message Batches routes of the aggregated server."""

    def __init__(self, *, runner: BatchItemRunner) -> None:
        # The batch methods create the store on first use, with this runner.
        self._run_batched_message = runner
        self._anthropic_batch_store: Optional[AnthropicBatchStore] = None
