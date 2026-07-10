"""Unit tests for the ``TokenizeRequest`` validator.

``TokenizeRequest.check_prompt_or_messages`` is the only custom logic in the
``/_internal/tokenize`` request schema: the caller must supply exactly one of
``prompt`` or ``messages``. Neither and both are rejected.
"""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import TokenizeRequest


def test_neither_prompt_nor_messages_rejected():
    with pytest.raises(ValidationError, match="Either 'prompt' or 'messages'"):
        TokenizeRequest()


def test_both_prompt_and_messages_rejected():
    with pytest.raises(ValidationError, match="Only one of 'prompt' or 'messages'"):
        TokenizeRequest(prompt="hello", messages=[{"role": "user", "content": "hello"}])
