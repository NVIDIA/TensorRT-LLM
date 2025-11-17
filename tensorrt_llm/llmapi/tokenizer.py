import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.base import (SpecialTokenPolicy,
                                                   SpecialTokens)
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from mistral_common.tokens.tokenizers.sentencepiece import \
    SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from transformers import (AutoTokenizer, BatchFeature, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast)
from transformers.tokenization_mistral_common import \
    MistralCommonTokenizer as TransformersMistralTokenizer

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


# Adapted from:
# https://github.com/vllm-project/vllm/blob/8e67b2557aae7204c697d7a5c61e00754da465be/vllm/transformers_utils/tokenizers/mistral.py#L166
class MistralTokenizer(TransformersTokenizer):

    def __init__(self, tokenizer: "TransformersMistralTokenizer"):

        self.transformers_tokenizer = tokenizer
        self.mistral = tokenizer.tokenizer
        self.instruct = self.mistral.instruct_tokenizer
        self.tokenizer = self.instruct.tokenizer

        _mistral_version_str = str(self.tokenizer.version.value)
        self.version: int = int(_mistral_version_str.split("v")[-1])

        self.is_tekken = isinstance(self.tokenizer, Tekkenizer)
        self.is_spm = isinstance(self.tokenizer, SentencePieceTokenizer)
        if not (self.is_tekken or self.is_spm):
            raise TypeError(f"Unsupported tokenizer: {type(self.tokenizer)}")

        # Reverse order to ensure that the lowest token id is kept.
        self._vocab_dict = {
            self.convert_ids_to_tokens([i], skip_special_tokens=False)[0]: i
            for i in range(self.transformers_tokenizer.vocab_size - 1, -1, -1)
        }
        # Sort the dict for convenience
        self._vocab_dict = dict(
            sorted(self._vocab_dict.items(), key=lambda x: x[1]))

        # Cache special tokens for faster access.
        self._special_token_ids = self._get_special_token_ids()
        self._special_token_ids_set = set(self._special_token_ids)
        self._special_tokens = self._get_special_tokens(self._special_token_ids)
        self._special_tokens_set = set(self._special_tokens)

        # Vocab sorted by token id.
        self._vocab = self.tokenizer._vocab
        self._max_token_id = self.transformers_tokenizer.vocab_size - 1

        self._all_special_tokens_set = set(self.all_special_tokens)

    def _get_special_tokens(self, all_special_ids: list[int]) -> list[str]:
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        return [
            self.tokenizer.decode([i],
                                  special_token_policy=SpecialTokenPolicy.KEEP)
            for i in all_special_ids
        ]

    # the following attributes are set to fit vLLM's design and are used
    # by the structured output backends.
    @property
    def all_special_tokens_extended(self) -> list[str]:
        return self.all_special_tokens

    @property
    def all_special_tokens(self) -> list[str]:
        return self._special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self._special_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        if Path(pretrained_model_dir).is_file():
            tokenizer = TransformersMistralTokenizer(
                tokenizer_path=pretrained_model_dir)
        else:
            tokenizer = TransformersMistralTokenizer.from_pretrained(
                pretrained_model_dir)
        return cls(tokenizer)

    def _get_special_token_ids(self) -> list[int]:
        from mistral_common.tokens.tokenizers.sentencepiece import \
            SentencePieceTokenizer
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer

        if self.is_tekken:
            assert isinstance(self.tokenizer, Tekkenizer), type(self.tokenizer)
            special_ids = {
                t["rank"]
                for t in self.tokenizer._all_special_tokens
            }
        elif self.is_spm:
            assert isinstance(self.tokenizer,
                              SentencePieceTokenizer), type(self.tokenizer)
            special_ids = self.tokenizer._control_tokens
        else:
            raise ValueError(f"Unknown tokenizer type: {type(self.tokenizer)}")
        return sorted(special_ids)

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> str:
        return self.transformers_tokenizer.pad_token

    @property
    def pad_token_id(self) -> int:
        return self.transformers_tokenizer.pad_token_id

    def __call__(self, text: str, *args, **kwargs) -> Any:
        return self.transformers_tokenizer(text=text, *args, **kwargs)

    @property
    def name_or_path(self) -> str:
        raise NotImplementedError

    def batch_encode_plus(self, texts: List[str], *args, **kwargs) -> dict:
        raise NotImplementedError

    def get_chat_template(self,
                          chat_template: Optional[str] = None,
                          tools: Optional[List[Dict]] = None) -> str:
        raise NotImplementedError

    def clean_up_tokenization(self, out_string: str) -> str:
        #self.transformers_tokenizer.clean_up_tokenization(out_string)
        raise NotImplementedError

    @property
    def is_fast(self) -> bool:
        return True

    def get_added_vocab(self) -> Dict[str, int]:
        # Mistral tokenizers have no added vocabulary
        return {}

    def _tekken_token_to_id(self, tokenizer: "Tekkenizer",
                            t: str | bytes) -> int:

        assert isinstance(tokenizer, Tekkenizer), type(tokenizer)

        t_bytes = t.encode("utf-8") if not isinstance(t, bytes) else t
        shift = tokenizer.num_special_tokens
        try:
            return shift + tokenizer._tekken_token2id_nospecial[t_bytes]
        except KeyError:
            t_str = t_bytes.decode("utf-8")
            if t_str in tokenizer._special_tokens_reverse_vocab:
                return tokenizer._special_tokens_reverse_vocab[t_str]
            logger.warning(
                "Failed to convert token %s to id, replacing with <unk>",
                t_bytes)
            return tokenizer.unk_id

    def _is_special_token_id(self, token_id: int) -> bool:
        return token_id in self._special_token_ids_set

    def convert_tokens_to_string(
            self,
            tokens: list[str],
            skip_special_tokens: bool = False,
            spaces_between_special_tokens: bool = True) -> str:

        to_decode_special_tokens = {SpecialTokens.tool_calls}
        if self.is_tekken:
            assert isinstance(self.tokenizer, Tekkenizer), type(self.tokenizer)
            tokens = [
                t for t in tokens if (t in to_decode_special_tokens
                                      or t not in self._special_tokens_set)
            ]

            if any(isinstance(t, bytes) for t in tokens):
                # we need to encode and decode all tokens again
                ids = [
                    self._tekken_token_to_id(self.tokenizer, t) for t in tokens
                ]
                # We filtered unwanted special tokens before
                # so we can decode the rest.
                decoded = self.tokenizer.decode(ids, SpecialTokenPolicy.KEEP)
            else:
                decoded = "".join(tokens)
        else:
            # make sure certain special tokens like Tool calls are
            # not decoded
            assert isinstance(self.tokenizer,
                              SentencePieceTokenizer), type(self.tokenizer)

            regular_tokens: list[str] = []
            decoded_list: list[str] = []
            decoded = ""

            for token in tokens:
                if token in to_decode_special_tokens:
                    if regular_tokens:
                        decoded_list.append(
                            self.tokenizer.decode(regular_tokens,
                                                  SpecialTokenPolicy.IGNORE))
                        regular_tokens = []
                    decoded_list.append(token)
                else:
                    regular_tokens.append(token)

            if regular_tokens:
                decoded_list.append(
                    self.tokenizer.decode(regular_tokens,
                                          SpecialTokenPolicy.IGNORE))
            decoded = "".join(decoded_list)

        return decoded

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> List[int]:
        if add_special_tokens is not None:
            return self.transformers_tokenizer.encode(
                text,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            )
        else:
            encoded = self.tokenizer.encode(text, bos=True, eos=False)

            if truncation is not False and max_length is not None:
                return encoded[:max_length]
            else:
                return encoded

    def decode(self,
               token_ids: list[int] | int,
               skip_special_tokens: bool = True,
               *args,
               **kwargs) -> str:
        return self.transformers_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(
        self,
        ids: Union[int, List[int]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        from mistral_common.tokens.tokenizers.base import (SpecialTokenPolicy,
                                                           SpecialTokens)
        from mistral_common.tokens.tokenizers.instruct import \
            InstructTokenizerV13

        if not skip_special_tokens:
            return [self.tokenizer.id_to_piece(token_id) for token_id in ids]

        non_skip_special_tokens_ids = {
            self.tokenizer.get_control_token(SpecialTokens.tool_calls),
        }
        if isinstance(self.instruct, InstructTokenizerV13):
            if self.instruct.BEGIN_THINK:
                non_skip_special_tokens_ids.add(self.instruct.BEGIN_THINK)
            if self.instruct.END_THINK:
                non_skip_special_tokens_ids.add(self.instruct.END_THINK)

        ids_kept = [
            i for i in ids if i in non_skip_special_tokens_ids
            or not self._is_special_token_id(i)
        ]

        # We filtered unwanted special tokens so we can decode the rest.
        tokens = [self.tokenizer.id_to_piece(token_id) for token_id in ids_kept]

        if any("�" in t for t in tokens) and self.is_tekken:
            # if a decoded token contains the replacement character, then the
            # token has an incomplete UTF-8 character so we must use bytes
            # See: https://github.com/vllm-project/vllm/pull/8640
            #      https://github.com/vllm-project/vllm/pull/9625
            # if underlying tokenizer is sentencepiece, we just add "�".
            # We filtered unwanted special tokens so we can decode the rest.
            tokens = [
                self.tokenizer.id_to_byte_piece(token_id,
                                                SpecialTokenPolicy.KEEP)
                if token_id not in self._special_token_ids_set else
                self.tokenizer.decode([token_id], SpecialTokenPolicy.KEEP)
                for token_id in ids_kept
            ]

        return tokens

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_dict)

    @property
    def clean_up_tokenization_spaces(self):
        return False

    def hf_decode_incrementally(
        self,
        token_ids: List[int],
        prev_text: Optional[str] = None,
        states: Optional[dict] = None,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None
    ) -> Tuple[str, dict]:
        raise NotImplementedError

    def apply_chat_template(
            self, conversation: Union[List[Dict[str, str]],
                                      List[List[Dict[str, str]]]], *args,
            **kwargs) -> Union[str, List[int], List[str], List[List[int]]]:
        # FIXME
        new_conversation = []
        # only keep the str contents since the tokenizer does not support data of other types
        for idx in range(len(conversation)):
            new_conversation.append({})
            for key in conversation[idx]:
                if isinstance(conversation[idx][key], str):
                    new_conversation[idx][key] = conversation[idx][key]
        return self.transformers_tokenizer.apply_chat_template(
            new_conversation, *args, **kwargs)


class PixtralProcessorAdapter:
    """
    Provide a HF-compatible interface for
    `mistral_common.tokens.tokenizers.multimodal.ImageEncoder`.
    """

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        super().__init__()

        self.tokenizer = tokenizer

    @property
    def image_processor(self) -> ImageEncoder:
        image_encoder = self.tokenizer.instruct.mm_encoder
        assert isinstance(image_encoder, ImageEncoder)
        return image_encoder

    @property
    def image_break_id(self) -> int:
        return self.image_processor.special_ids.img_break

    @property
    def image_token_id(self) -> int:
        return self.image_processor.special_ids.img

    @property
    def image_end_id(self) -> int:
        return self.image_processor.special_ids.img_end

    @property
    def image_size(self) -> int:
        return self.image_processor.mm_config.max_image_size

    @property
    def patch_size(self) -> int:
        return self.image_processor.mm_config.image_patch_size

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:  # Mapping[str, NestedTensors]:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if not images:
            input_ids = self.tokenizer(text).input_ids

            return {"input_ids": torch.tensor(input_ids)}

        # Allow dummy text, which is used for profiling as well as token inputs
        if any(len(t) > 0 for t in text):
            raise ValueError(
                "You've passed text inputs instead of token inputs. "
                "Make sure to process your input via `mistral_common`'s "
                "tokenizer or pass a chat completion request. "
                "For more info, see: "
                "https://github.com/vllm-project/vllm/issues/8411.")

        images_processed = list[torch.Tensor]()
        images_tokens = list[torch.Tensor]()

        for image in images:
            image_inputs = self.image_processor(ImageChunk(image=image))
            image_processed = torch.tensor(image_inputs.image)
            image_tokens = torch.tensor(image_inputs.tokens)

            images_processed.append(image_processed)
            images_tokens.append(image_tokens)

        return BatchFeature({
            "input_ids":
            torch.cat(images_tokens)[None].expand(len(text), -1),
            "images":
            images_processed,
        })


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
