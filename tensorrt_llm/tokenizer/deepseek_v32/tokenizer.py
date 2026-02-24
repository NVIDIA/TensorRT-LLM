"""DeepSeek V3.2 tokenizer implementation.

This is a temporary workaround for DeepSeek-V3.2 model as HF does not support it yet.
TODO: Remove this once HF supports DeepSeek-V3.2
"""

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..tokenizer import TransformersTokenizer
from .encoding import encode_messages


class DeepseekV32Tokenizer(TransformersTokenizer):
    """DeepSeek V3.2 tokenizer with custom chat template."""

    def __init__(self, tokenizer):
        # tokenizer should be the HF tokenizer
        self.tokenizer = tokenizer
        self._all_special_tokens_set = set(self.tokenizer.all_special_tokens)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "DeepseekV32Tokenizer":
        # Load HF tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
        return DeepseekV32Tokenizer(hf_tokenizer)

    def apply_chat_template(self, messages, tools=None, **kwargs):
        thinking = kwargs.get("thinking", False)
        thinking_mode = "thinking" if thinking else "chat"
        messages = messages.copy()
        drop_thinking = True
        if tools is not None and len(tools) > 0:
            messages.insert(0, {"role": "system"})
            messages[0]["tools"] = tools
            drop_thinking = False
        encode_config = dict(thinking_mode=thinking_mode, drop_thinking=drop_thinking)
        prompt_str = encode_messages(messages, **encode_config)  # type: ignore
        return prompt_str

    @property
    def all_special_tokens(self) -> list[str]:
        return self.tokenizer.all_special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self.tokenizer.all_special_ids

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def is_fast(self) -> bool:
        return self.tokenizer.is_fast

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def max_token_id(self) -> int:
        return self.tokenizer.max_token_id

    @property
    def truncation_side(self) -> str:
        return self.tokenizer.truncation_side

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        # </think> is an added token in DeepseekV32 tokenizer
        return self.vocab_size + len(self.get_added_vocab())

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> Any:
        return self.tokenizer(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_added_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_added_vocab()

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self.tokenizer.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False, **kwargs) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
