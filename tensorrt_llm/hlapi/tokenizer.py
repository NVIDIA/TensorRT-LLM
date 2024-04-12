from pathlib import Path
from typing import Any, List, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

TokenIdsTy = List[int]


class TokenizerBase(PreTrainedTokenizerBase):
    ''' This is a protocol for the tokenizer. Users can implement their own tokenizer by inheriting this class.  '''


class TransformersTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                                  **kwargs)
        return TransformersTokenizer(tokenizer)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str, *args, **kwargs) -> Any:
        return self.tokenizer(text, *args, **kwargs)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, text: str, *args, **kwargs) -> TokenIdsTy:
        return self.tokenizer.encode(text, *args, **kwargs)

    def decode(self, token_ids: TokenIdsTy, *args, **kwargs) -> str:
        return self.tokenizer.decode(token_ids, *args, **kwargs)

    def batch_encode_plus(self, texts: List[str], *args, **kwargs) -> dict:
        return self.tokenizer.batch_encode_plus(texts, *args, **kwargs)


def tokenizer_factory(
        obj: Union[str, Path, TokenizerBase, PreTrainedTokenizerBase, None],
        **kwargs) -> Union[TokenizerBase, PreTrainedTokenizerBase, None]:
    if obj is None:
        return None
    if isinstance(obj, (str, Path)):
        default_kwargs = {
            'legacy': False,
            'padding_side': 'left',
            'truncation_side': 'left',
            'trust_remote_code': True,
            'use_fast': True,
        }
        default_kwargs.update(kwargs)
        return AutoTokenizer.from_pretrained(obj, **kwargs)

    return obj
