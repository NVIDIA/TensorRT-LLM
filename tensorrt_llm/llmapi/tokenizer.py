from pathlib import Path
from typing import Any, List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TokenizerBase(PreTrainedTokenizerBase):
    ''' This is a protocol for the tokenizer. Users can implement their own tokenizer by inheriting this class.  '''


class TransformersTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

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

    def encode(self, text: str, *args, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, *args, **kwargs)

    def decode(self, token_ids: List[int], *args, **kwargs) -> str:
        return self.tokenizer.decode(token_ids, *args, **kwargs)

    def batch_encode_plus(self, texts: List[str], *args, **kwargs) -> dict:
        return self.tokenizer.batch_encode_plus(texts, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tokenizer})"

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                                  **kwargs)
        return cls(tokenizer)


def tokenizer_factory(obj: Optional[Union[str, Path, PreTrainedTokenizerBase,
                                          TokenizerBase]] = None,
                      **kwargs) -> Optional[TokenizerBase]:
    if obj is None:
        return None
    elif isinstance(obj, (str, Path)):
        default_kwargs = {
            'legacy': False,
            'padding_side': 'left',
            'truncation_side': 'left',
            'trust_remote_code': True,
            'use_fast': True,
        }
        default_kwargs.update(kwargs)
        return TransformersTokenizer.from_pretrained(obj, **default_kwargs)
    elif isinstance(obj, PreTrainedTokenizerBase):
        return TransformersTokenizer(obj)
    elif isinstance(obj, TokenizerBase):
        return obj
    else:
        raise TypeError(f"Unrecognized tokenizer {obj}")
