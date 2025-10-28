from abc import ABC, abstractmethod
from typing import Union

from tensorrt_llm.serve.openai_client import CompletionResponseGenerator
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)


class OpenAIService(ABC):
    @abstractmethod
    async def openai_completion(
        self, request: CompletionRequest
    ) -> Union[CompletionResponse, CompletionResponseGenerator]:
        """
        Return a tuple of (completion response, async completion response generator)
        When request is streaming, the generator will be used to stream the response.
        When request is not streaming, the generator will be ignore and the response will be returned directly.
        """
        ...

    @abstractmethod
    async def openai_chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, CompletionResponseGenerator]:
        """Similar to openai_completion, but for chat completion protocol."""
        ...

    @abstractmethod
    async def is_ready(self) -> bool: ...

    @abstractmethod
    async def setup(self) -> None: ...

    @abstractmethod
    async def teardown(self) -> None: ...
