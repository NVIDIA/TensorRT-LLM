from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TokenizerBase(PreTrainedTokenizerBase):
    ''' This is a protocol for the tokenizer. Users can implement their own tokenizer by inheriting this class.  '''


class TransformersTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._all_special_tokens_set = set(self.tokenizer.all_special_tokens)

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

    def clean_up_tokenization(self, out_string: str) -> str:
        return self.tokenizer.clean_up_tokenization(out_string)

    @property
    def clean_up_tokenization_spaces(self):
        return self.tokenizer.clean_up_tokenization_spaces

    @property
    def is_fast(self) -> bool:
        return self.tokenizer.is_fast

    def get_added_vocab(self) -> Dict[str, int]:
        # Assumed to be O(1) complexity
        return self.tokenizer.get_added_vocab()

    def convert_ids_to_tokens(
            self,
            ids: Union[int, List[int]],
            skip_special_tokens: bool = False) -> Union[str, List[str]]:
        return self.tokenizer.convert_ids_to_tokens(
            ids, skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_string(
            self,
            tokens: List[str],
            skip_special_tokens: bool = False,
            spaces_between_special_tokens: bool = True) -> str:
        # Adapted from
        # https://github.com/vllm-project/vllm/blob/v0.6.3/vllm/transformers_utils/detokenizer.py#L172
        if self.is_fast or not self.get_added_vocab():
            return self.tokenizer.convert_tokens_to_string(tokens)

        sub_texts: List[str] = []
        current_sub_text: List[str] = []
        for token in tokens:
            if skip_special_tokens and token in self._all_special_tokens_set:
                continue
            if token in self.get_added_vocab():
                if current_sub_text:
                    sub_text = self.tokenizer.convert_tokens_to_string(
                        current_sub_text)
                    sub_texts.append(sub_text)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_text = self.tokenizer.convert_tokens_to_string(current_sub_text)
            sub_texts.append(sub_text)
        if spaces_between_special_tokens:
            return " ".join(sub_texts)
        else:
            return "".join(sub_texts)

    def decode_incrementally(
            self,
            token_ids: List[int],
            prev_text: Optional[str] = None,
            states: Optional[dict] = None,
            *,
            flush: bool = False,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            spaces_between_special_tokens: bool = True) -> Tuple[str, dict]:
        """Incremental detokenization, typically used for streaming generation.

        Args:
            token_ids (List[int]): The incremental token ids.
            prev_text (str): The previous decoded text. None if it's the first iteration.
            states (dict): A dict that saves previous states for incremental detokenization. None if it's the first iteration.
            flush (bool): Force flushing the pending tokens to decoded text.
            skip_special_tokens (bool): Whether to remove special tokens in the decoding.
            clean_up_tokenization_spaces (bool): Whether to clean up tokenization spaces.
            spaces_between_special_tokens (bool): Whether to add spaces between special tokens.

        Returns:
            text, states (Tuple[str, dict]): text is the current decoded text, states is the current incremental detokenization states.
            They should be passed to next incremental detokenization iteration, if any.
        """
        # Adapted from
        # https://github.com/vllm-project/vllm/blob/v0.6.3/vllm/transformers_utils/detokenizer.py#L238
        if prev_text is None:
            prev_text = ""

        if states is None:
            states = {}
        last_new_tokens = states.pop('last_new_tokens', [])
        pending_tokens = states.pop('pending_tokens', [])

        if len(last_new_tokens) > 0:
            last_new_text = self.convert_tokens_to_string(
                last_new_tokens,
                skip_special_tokens=skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens)
        else:
            last_new_text = ""

        new_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens)
        pending_tokens.extend(new_tokens)

        curr_new_text = self.convert_tokens_to_string(
            last_new_tokens + pending_tokens,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens)
        if not flush and (len(curr_new_text.rstrip()) <= len(
                last_new_text.rstrip()) or curr_new_text.endswith("�")):
            return prev_text, {
                'last_new_tokens': last_new_tokens,
                'pending_tokens': pending_tokens
            }

        # Remove the part of last_new_text
        curr_new_text = curr_new_text[len(last_new_text):]
        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = self.clean_up_tokenization_spaces
        if clean_up_tokenization_spaces:
            curr_new_text = self.clean_up_tokenization(curr_new_text)
        return prev_text + curr_new_text, {'last_new_tokens': pending_tokens}


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
    elif isinstance(obj, TokenizerBase):
        return obj
    elif isinstance(obj, PreTrainedTokenizerBase):
        return TransformersTokenizer(obj)
    else:
        raise TypeError(f"Unrecognized tokenizer {obj}")
