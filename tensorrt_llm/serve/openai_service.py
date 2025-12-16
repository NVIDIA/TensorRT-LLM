# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# yapf: disable
from abc import ABC, abstractmethod

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest
from tensorrt_llm.serve.responses_utils import UCompletionResponseOrGenerator

# yapf: enable


class OpenAIService(ABC):
    @abstractmethod
    async def openai_completion(self, request: CompletionRequest) -> UCompletionResponseOrGenerator:
        """Return either a completion response or an async completion response generator.

        When request is streaming, the generator will be used to stream the response.
        When request is not streaming, the response will be returned directly.
        """
        ...

    @abstractmethod
    async def openai_chat_completion(
        self, request: ChatCompletionRequest
    ) -> UCompletionResponseOrGenerator:
        """Similar to openai_completion, but for chat completion protocol."""
        ...

    @abstractmethod
    async def is_ready(self) -> bool:
        """Check if the service is ready to accept requests."""
        ...

    @abstractmethod
    async def setup(self) -> None:
        """Setup the service."""
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """Teardown the service."""
        ...
