import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import (AutoTokenizer, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast)

from .._utils import nvtx_range_debug
from ..logger import logger

TLLM_INCREMENTAL_DETOKENIZATION_BACKEND = os.environ.get(
    "TLLM_INCREMENTAL_DETOKENIZATION_BACKEND", "HF")
TLLM_STREAM_INTERVAL_THRESHOLD = int(
    os.environ.get("TLLM_STREAM_INTERVAL_THRESHOLD", "24"))
try:
    from tokenizers.decoders import DecodeStream  # noqa
except ImportError:
    logger.warning(
        f"HF incremental detokenization is unsupported by tokenizer<0.21.0; fallback to TRTLLM incremental detokenization."
    )
    TLLM_INCREMENTAL_DETOKENIZATION_BACKEND = "TRTLLM"


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

    @property
    def name_or_path(self) -> str:
        return self.tokenizer.name_or_path

    def encode(self, text: str, *args, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, *args, **kwargs)

    def decode(self, token_ids: List[int], *args, **kwargs) -> str:
        return self.tokenizer.decode(token_ids, *args, **kwargs)

    def batch_encode_plus(self, texts: List[str], *args, **kwargs) -> dict:
        return self.tokenizer.batch_encode_plus(texts, *args, **kwargs)

    def get_chat_template(self,
                          chat_template: Optional[str] = None,
                          tools: Optional[List[Dict]] = None) -> str:
        return self.tokenizer.get_chat_template(chat_template, tools)

    def apply_chat_template(
            self, conversation: Union[List[Dict[str, str]],
                                      List[List[Dict[str, str]]]], *args,
            **kwargs) -> Union[str, List[int], List[str], List[List[int]]]:
        return self.tokenizer.apply_chat_template(conversation, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tokenizer})"

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                                  **kwargs)
        return cls(tokenizer)

    def save_pretrained(self, pretrained_model_dir: str, **kwargs):
        self.tokenizer.save_pretrained(pretrained_model_dir, **kwargs)

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

    @nvtx_range_debug("decode_incrementally")
    def decode_incrementally(
            self,
            token_ids: List[int],
            prev_text: Optional[str] = None,
            states: Optional[dict] = None,
            *,
            flush: bool = False,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: Optional[bool] = None,
            spaces_between_special_tokens: bool = True,
            stream_interval: int = 1) -> Tuple[str, dict]:
        """Incremental detokenization, typically used for streaming generation.

        Args:
            token_ids (List[int]): The incremental token ids.
            prev_text (str): The previous decoded text. None if it's the first iteration.
            states (dict): A dict that saves previous states for incremental detokenization. None if it's the first iteration.
            flush (bool): Force flushing the pending tokens to decoded text.
            skip_special_tokens (bool): Whether to remove special tokens in the decoding.
            clean_up_tokenization_spaces (bool): Whether to clean up tokenization spaces.
            spaces_between_special_tokens (bool): Whether to add spaces between special tokens.
            stream_interval (int): The iteration interval to create responses under the streaming mode.

        Returns:
            text, states (Tuple[str, dict]): text is the current decoded text, states is the current incremental detokenization states.
            They should be passed to next incremental detokenization iteration, if any.
        """
        # HF incremental detokenization implementation is faster than TRTLLM when stream_interval is smaller.
        if (TLLM_INCREMENTAL_DETOKENIZATION_BACKEND == "TRTLLM"
                or stream_interval >= TLLM_STREAM_INTERVAL_THRESHOLD
                or spaces_between_special_tokens is False
                or not hasattr(self.tokenizer, "_tokenizer")):
            return self.trtllm_decode_incrementally(
                token_ids,
                prev_text,
                states,
                flush=flush,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                spaces_between_special_tokens=spaces_between_special_tokens)
        else:
            return self.hf_decode_incrementally(
                token_ids,
                prev_text,
                states,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    def trtllm_decode_incrementally(
            self,
            token_ids: List[int],
            prev_text: Optional[str] = None,
            states: Optional[dict] = None,
            *,
            flush: bool = False,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: Optional[bool] = None,
            spaces_between_special_tokens: bool = True) -> Tuple[str, dict]:
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

    def hf_decode_incrementally(
        self,
        token_ids: List[int],
        prev_text: Optional[str] = None,
        states: Optional[dict] = None,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None
    ) -> Tuple[str, dict]:
        if states is None:
            states = {
                'decode_stream':
                DecodeStream(skip_special_tokens=skip_special_tokens)
            }

        decode_stream = states.get('decode_stream')
        results = [
            result for tid in token_ids
            if (result := decode_stream.step(self.tokenizer._tokenizer, tid)
                ) is not None
        ]
        curr_new_text = "".join(results)
        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = self.clean_up_tokenization_spaces
        if clean_up_tokenization_spaces:
            curr_new_text = self.clean_up_tokenization(curr_new_text)

        if prev_text is None:
            return curr_new_text, states
        else:
            return prev_text + curr_new_text, states


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


def _xgrammar_tokenizer_info(tokenizer):
    # Reference: https://github.com/mlc-ai/xgrammar/blob/b9a16de54e1e0eff58da14c65750414cceaf1a6f/python/xgrammar/tokenizer_info.py#L133
    if isinstance(tokenizer, TokenizerBase):
        tokenizer = tokenizer.tokenizer

    stop_token_ids = [tokenizer.eos_token_id]

    try:
        encoded_vocab = tokenizer.get_vocab()
        encoded_vocab = [
            token
            for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])
        ]
    except AttributeError as e:
        msg = (
            f"Cannot get the vocabulary of the tokenizer {type(tokenizer)}. The tokenizer "
            "should have a get_vocab method.")
        raise ValueError(msg) from e

    if isinstance(tokenizer, PreTrainedTokenizerFast):
        backend_str = tokenizer.backend_tokenizer.to_str()
        return {
            "encoded_vocab": encoded_vocab,
            "tokenizer_str": backend_str,
            "stop_token_ids": stop_token_ids
        }
    elif ("vocab_file" in tokenizer.vocab_files_names
          and "tiktoken" in tokenizer.vocab_files_names["vocab_file"]):
        return {
            "encoded_vocab": encoded_vocab,
            "stop_token_ids": stop_token_ids
        }
    else:
        raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")


def _llguidance_tokenizer_info(tokenizer):
    tokenizer_info = _xgrammar_tokenizer_info(tokenizer)
    if tokenizer_info.get("tokenizer_str") is None:
        raise ValueError("missing tokenizer_str")
    return tokenizer_info


def load_hf_tokenizer(model_dir: str,
                      trust_remote_code: bool = True,
                      use_fast: bool = True,
                      **kwargs) -> Optional[TransformersTokenizer]:
    ''' Load a tokenizer from a Hugging Face model directory.

    Args:
        model_dir (str): The model directory.
        trust_remote_code (bool): Whether to trust the remote code.
        use_fast (bool): Whether to use the fast tokenizer.

    Returns:
        A TransformersTokenizer object if the tokenizer is loaded successfully.
    '''

    try:
        return TransformersTokenizer.from_pretrained(
            model_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            **kwargs)

    except Exception as e:
        logger.warning(
            f"Failed to load hf tokenizer from {model_dir}, encounter error: {e}"
        )
        return None
