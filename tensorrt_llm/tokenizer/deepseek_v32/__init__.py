"""DeepSeek V3.2 tokenizer and encoding utilities.

This is a temporary workaround for DeepSeek-V3.2 model as HF does not support it yet.
TODO: Remove this once HF supports DeepSeek-V3.2
"""

from .encoding import encode_messages, parse_message_from_completion_text
from .tokenizer import DeepseekV32Tokenizer

__all__ = [
    "DeepseekV32Tokenizer",
    "encode_messages",
    "parse_message_from_completion_text",
]
