# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from mistral_common.protocol.instruct.chunk import ImageChunk, ImageURLChunk
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens
from mistral_common.tokens.tokenizers.image import ImageEncoder, ImageEncoding, image_from_chunk
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from PIL import Image
from transformers.tokenization_mistral_common import (
    MistralCommonTokenizer as TransformersMistralTokenizer,
)

from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    import llguidance
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as MistralCommonTokenizer


class _ImageGridEncoder(ImageEncoder):
    """mistral-common's image encoder reduced to each image's token grid.

    The grid of ``[IMG]``, ``[IMG_BREAK]`` and ``[IMG_END]`` tokens depends on the image size
    alone, so a chat encodes to the same ids as with the full encoder, and no pixel is resized
    or normalized.
    """

    def __call__(self, content: Union[ImageChunk, ImageURLChunk]) -> ImageEncoding:
        width_tokens, height_tokens = self._image_to_num_tokens(image_from_chunk(content))
        tokens = (
            [self.special_ids.img] * width_tokens + [self.special_ids.img_break]
        ) * height_tokens
        tokens[-1] = self.special_ids.img_end
        return ImageEncoding(tokens=tokens, image=np.empty((0, 0, 0), dtype=np.float32))


# The ``chat_template_kwargs`` keys mistral-common's chat encoder reads. It has no Jinja
# template, so other keys have nothing to apply to.
_CHAT_KWARGS = frozenset(("continue_final_message", "reasoning_effort", "response_format"))

# The URL ``render_chat`` gives each image part of a chat; mistral-common's request then
# takes a blank image of the loaded image's size there.
_IMAGE_SLOT = "trtllm-loaded-image"


def _response_format(response_format: Any) -> Optional[Any]:
    """An OpenAI ``response_format`` as mistral-common's, or None if it takes none.

    mistral-common takes ``json_object`` and a named ``json_schema`` from 1.13.0 on, which
    needs Python 3.11, and states them in the prompt when the tokenizer file declares a
    response-format model setting. Every response format constrains decoding either way.
    """
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    named = response_format.get("json_schema")
    if kind not in ("json_object", "json_schema") or (
        kind == "json_schema"
        and (not isinstance(named, dict) or named.get("schema") is None or not named.get("name"))
    ):
        return None
    from mistral_common.protocol.instruct import request as mistral_request

    if not hasattr(mistral_request, "JsonSchema"):
        logger.warning_once(
            "mistral-common takes response formats from 1.13.0 on (Python "
            "3.11 or later); the prompt leaves the response format out, and it "
            "only constrains decoding.",
            key="mistral_common_response_format_unrendered",
        )
        return None
    ResponseFormat = mistral_request.ResponseFormat
    ResponseFormats = mistral_request.ResponseFormats
    JsonSchema = mistral_request.JsonSchema
    if kind == "json_object":
        return ResponseFormat(type=ResponseFormats.json_object)
    return ResponseFormat(
        type=ResponseFormats.json_schema,
        json_schema=JsonSchema(
            name=named["name"], schema=named["schema"], strict=bool(named.get("strict"))
        ),
    )


def _tools(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    """OpenAI tool dicts in the form mistral-common validates: no null ``strict``."""
    if tools is None:
        return None
    normalized = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("strict", False) is None:
            function = {key: value for key, value in function.items() if key != "strict"}
            tool = {**tool, "function": function}
        normalized.append(tool)
    return normalized


def _check_generation_prompt(
    messages: list[dict[str, Any]], add_generation_prompt: bool, continue_final_message: bool
) -> None:
    """Reject generation-prompt flags that contradict the final message."""
    from mistral_common.exceptions import InvalidMessageStructureException

    if add_generation_prompt and continue_final_message:
        raise InvalidMessageStructureException(
            "Cannot set both `add_generation_prompt` and `continue_final_message` to True."
        )
    last_role = messages[-1].get("role") if messages else None
    if add_generation_prompt and last_role == "assistant":
        raise InvalidMessageStructureException(
            "Cannot set `add_generation_prompt` to True when the last message "
            "is from the assistant. Consider using `continue_final_message` "
            "instead."
        )
    if continue_final_message and last_role != "assistant":
        raise InvalidMessageStructureException(
            "Cannot set `continue_final_message` to True when "
            "the last message is not from the assistant."
        )


# Adapted from:
# https://github.com/vllm-project/vllm/blob/8e67b2557aae7204c697d7a5c61e00754da465be/vllm/transformers_utils/tokenizers/mistral.py#L166
class MistralTokenizer(TransformersTokenizer):
    # Tekken's vocabulary is bytes, with separate control tokens, and has no Hugging Face
    # tokenizer JSON: guided decoding takes it from xgrammar_tokenizer_info and
    # llguidance_tokenizer.
    describes_grammar_vocabulary = True

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

        # Reverse order to ensure that the lowest token id is kept. The vocab keys each id
        # by its piece as written (convert_ids_to_tokens may return bytes for streaming).
        self._vocab_dict = {
            self.tokenizer.id_to_piece(i): i
            for i in range(self.transformers_tokenizer.vocab_size - 1, -1, -1)
        }
        # Sort the dict for convenience
        self._vocab_dict = dict(sorted(self._vocab_dict.items(), key=lambda x: x[1]))

        # Cache special tokens for faster access.
        self._special_token_ids = self._get_special_token_ids()
        self._special_token_ids_set = set(self._special_token_ids)
        self._special_tokens = self._get_special_tokens(self._special_token_ids)
        self._special_tokens_set = set(self._special_tokens)

        # Vocab sorted by token id.
        self._vocab = self.tokenizer._vocab
        self._max_token_id = self.transformers_tokenizer.vocab_size - 1

        self._all_special_tokens_set = set(self.all_special_tokens)

        self._encoding_identity: Optional[dict] = None
        self._grid_only_mistral: Optional["MistralCommonTokenizer"] = None

    def __reduce__(self):
        # TransformersTokenizer.__reduce__ would rebuild a TransformersTokenizer
        # around ``self.tokenizer``, which here is the raw mistral-common
        # tokenizer. Rebuild this class from the transformers wrapper instead;
        # mistral-common pickles the tokenizer by the path of its file.
        return (self.__class__, (self.transformers_tokenizer,))

    @property
    def encoding_identity(self) -> Optional[dict]:
        """What turns text, images and chats into ids, for the rendering fingerprint.

        mistral-common reads all of it from the tokenizer file (vocabulary, pattern,
        control tokens, image and model settings) and renders chats with its own code,
        so the identity is the file's digest and mistral-common's version. ``None``
        when the file is not known. The file is hashed on first use.
        """
        if self._encoding_identity is None:
            path = getattr(self.tokenizer, "file_path", None)
            if path is None or not Path(path).is_file():
                return None
            self._encoding_identity = {
                "kind": "mistral_common",
                "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "mistral_common": package_version("mistral_common"),
            }
        return self._encoding_identity

    @property
    def grid_only_mistral(self) -> "MistralCommonTokenizer":
        """``self.mistral`` with an image encoder that makes each image's token grid only.

        It encodes a chat to the same ids as ``self.mistral``, and an image needs only its
        size: no pixel is resized or normalized. Chat rendering uses it; the input
        processor encodes the pixels. Built on first use.
        """
        if self._grid_only_mistral is None:
            image_encoder = getattr(self.instruct, "image_encoder", None)
            if image_encoder is None:
                self._grid_only_mistral = self.mistral
            else:
                instruct = copy.copy(self.instruct)
                instruct.image_encoder = _ImageGridEncoder(
                    image_encoder.image_config, image_encoder.special_ids
                )
                # The constructor, not copy.copy: mistral-common pickles (and so copies) a
                # tokenizer by reloading its file.
                self._grid_only_mistral = type(self.mistral)(
                    instruct,
                    self.mistral._chat_completion_request_validator,
                    self.mistral._instruct_request_normalizer,
                )
        return self._grid_only_mistral

    def _get_special_tokens(self, all_special_ids: list[int]) -> list[str]:
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        return [
            self.tokenizer.decode([i], special_token_policy=SpecialTokenPolicy.KEEP)
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
            tokenizer = TransformersMistralTokenizer(tokenizer_path=pretrained_model_dir)
        else:
            tokenizer = TransformersMistralTokenizer.from_pretrained(pretrained_model_dir)
        return cls(tokenizer)

    def xgrammar_tokenizer_info(self) -> dict:
        """The vocabulary xgrammar matches grammars against, for a Tekken tokenizer.

        Each token is its raw bytes, which need not be valid UTF-8. Control tokens and
        the tokens xgrammar reserves (those starting with a NUL byte) are empty strings,
        which xgrammar treats as special: a grammar never selects them, and the
        end-of-sequence id is the only stop token.

        Returns:
            The ``encoded_vocab`` and ``stop_token_ids`` of a guided decoding config.
        """
        if not self.is_tekken:
            raise ValueError("Guided decoding supports Tekken Mistral tokenizers only.")
        num_special = self.tokenizer.num_special_tokens
        encoded_vocab: list[str | bytes] = []
        for token_id in range(self.tokenizer.n_words):
            piece = b"" if token_id < num_special else self.tokenizer.id_to_byte_piece(token_id)
            encoded_vocab.append("" if piece.startswith(b"\x00") or not piece else piece)
        return {"encoded_vocab": encoded_vocab, "stop_token_ids": [self.eos_token_id]}

    def llguidance_tokenizer(self, n_vocab: int) -> "llguidance.LLTokenizer":
        """The llguidance tokenizer of a Tekken vocabulary, as mistral-common builds it.

        Ordinary tokens are their raw bytes; control tokens are special, so a grammar
        selects one only by reference (``<[9]>`` for ``[TOOL_CALLS]``, as
        mistral-common's grammar factory writes them), never as text. Text is
        tokenized with Tekken itself.

        Args:
            n_vocab: The vocabulary size the token bitmask covers (the model's padded
                vocabulary size); at least the Tekken vocabulary size.

        Returns:
            The tokenizer an llguidance matcher factory compiles grammars against.
        """
        if not self.is_tekken:
            raise ValueError("Guided decoding supports Tekken Mistral tokenizers only.")
        import llguidance
        from mistral_common.guidance.tokenizer import MistralLLGTokenizer

        wrapper = llguidance.TokenizerWrapper(MistralLLGTokenizer(self.tokenizer))
        # llguidance sizes the vocabulary of a wrapped tokenizer from its token list, not
        # n_vocab: pad it with empty tokens, which a grammar never selects.
        wrapper.tokens = wrapper.tokens + [b""] * (n_vocab - len(wrapper.tokens))
        return llguidance.LLTokenizer(wrapper, n_vocab=n_vocab, eos_token=self.eos_token_id)

    def _get_special_token_ids(self) -> list[int]:
        from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer

        if self.is_tekken:
            assert isinstance(self.tokenizer, Tekkenizer), type(self.tokenizer)
            special_ids = {t["rank"] for t in self.tokenizer._all_special_tokens}
        elif self.is_spm:
            assert isinstance(self.tokenizer, SentencePieceTokenizer), type(self.tokenizer)
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

    def __call__(self, text: str, *args, **kwargs) -> any:
        return self.transformers_tokenizer(text=text, *args, **kwargs)

    @property
    def name_or_path(self) -> str:
        return self.transformers_tokenizer.name_or_path

    def batch_encode_plus(self, texts: list[str], *args, **kwargs) -> dict:
        raise NotImplementedError

    def get_chat_template(
        self, chat_template: str | None = None, tools: list[dict] | None = None
    ) -> str:
        raise NotImplementedError

    def clean_up_tokenization(self, out_string: str) -> str:
        raise NotImplementedError

    @property
    def is_fast(self) -> bool:
        return True

    def get_added_vocab(self) -> dict[str, int]:
        # Mistral tokenizers have no added vocabulary
        return {}

    def _tekken_token_to_id(self, tokenizer: "Tekkenizer", t: str | bytes) -> int:
        assert isinstance(tokenizer, Tekkenizer), type(tokenizer)

        t_bytes = t.encode("utf-8") if not isinstance(t, bytes) else t
        shift = tokenizer.num_special_tokens
        try:
            return shift + tokenizer._tekken_token2id_nospecial[t_bytes]
        except KeyError:
            t_str = t_bytes.decode("utf-8")
            if t_str in tokenizer._special_tokens_reverse_vocab:
                return tokenizer._special_tokens_reverse_vocab[t_str]
            logger.warning("Failed to convert token %s to id, replacing with <unk>", t_bytes)
            return tokenizer.unk_id

    def _is_special_token_id(self, token_id: int) -> bool:
        return token_id in self._special_token_ids_set

    def convert_tokens_to_string(
        self,
        tokens: list[str],
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        to_decode_special_tokens = {SpecialTokens.tool_calls}
        if self.is_tekken:
            assert isinstance(self.tokenizer, Tekkenizer), type(self.tokenizer)
            # Without skip_special_tokens every control token stays, as decode() keeps
            # them, so the streamed text equals decode()'s.
            if skip_special_tokens:
                tokens = [
                    t
                    for t in tokens
                    if (t in to_decode_special_tokens or t not in self._special_tokens_set)
                ]

            if any(isinstance(t, bytes) for t in tokens):
                # we need to encode and decode all tokens again
                ids = [self._tekken_token_to_id(self.tokenizer, t) for t in tokens]
                # We filtered unwanted special tokens before
                # so we can decode the rest.
                decoded = self.tokenizer.decode(ids, SpecialTokenPolicy.KEEP)
            else:
                decoded = "".join(tokens)
        else:
            # make sure certain special tokens like Tool calls are
            # not decoded
            assert isinstance(self.tokenizer, SentencePieceTokenizer), type(self.tokenizer)

            regular_tokens: list[str] = []
            decoded_list: list[str] = []
            decoded = ""

            for token in tokens:
                if token in to_decode_special_tokens:
                    if regular_tokens:
                        decoded_list.append(
                            self.tokenizer.decode(regular_tokens, SpecialTokenPolicy.IGNORE)
                        )
                        regular_tokens = []
                    decoded_list.append(token)
                else:
                    regular_tokens.append(token)

            if regular_tokens:
                decoded_list.append(
                    self.tokenizer.decode(regular_tokens, SpecialTokenPolicy.IGNORE)
                )
            decoded = "".join(decoded_list)

        return decoded

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
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

    def decode(
        self, token_ids: list[int] | int, skip_special_tokens: bool = True, *args, **kwargs
    ) -> str:
        return self.transformers_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens
        from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13

        if skip_special_tokens:
            non_skip_special_tokens_ids = {
                self.tokenizer.get_control_token(SpecialTokens.tool_calls),
            }
            if isinstance(self.instruct, InstructTokenizerV13):
                if self.instruct.BEGIN_THINK:
                    non_skip_special_tokens_ids.add(self.instruct.BEGIN_THINK)
                if self.instruct.END_THINK:
                    non_skip_special_tokens_ids.add(self.instruct.END_THINK)

            ids_kept = [
                i
                for i in ids
                if i in non_skip_special_tokens_ids or not self._is_special_token_id(i)
            ]
        else:
            ids_kept = list(ids)

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
                self.tokenizer.id_to_byte_piece(token_id, SpecialTokenPolicy.KEEP)
                if token_id not in self._special_token_ids_set
                else self.tokenizer.decode([token_id], SpecialTokenPolicy.KEEP)
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
        token_ids: list[int],
        prev_text: str | None = None,
        states: dict | None = None,
        *,
        flush: bool = False,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        mm_data: Optional[dict[str, Any]] = None,
        *,
        add_generation_prompt: bool = True,
        tools: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[int]:
        """Token ids of an OpenAI-format chat, encoded by mistral-common.

        Each content part of ``messages`` that is not text stands for the next of the
        loaded images in ``mm_data``. mistral-common makes an image's token grid from its
        size alone, so the part is encoded as a blank image of that size with
        ``grid_only_mistral``; the input processor encodes the pixels.

        Args:
            messages: The chat as OpenAI messages, each image part in its place. A
                ``developer`` message is a system message, as mistral-common reads it from
                1.13.1 on.
            mm_data: The loaded multimodal data; images only.
            add_generation_prompt: Whether the chat asks for the next assistant turn.
                mistral-common always opens it after a final user or tool message, so
                this only selects which chats are valid.
            tools: OpenAI tool definitions.
            documents: Not read by mistral-common; a warning names them.
            chat_template: Not read by mistral-common (the server's default template,
                for one); a warning names it.
            chat_template_kwargs: mistral-common reads ``continue_final_message``,
                ``reasoning_effort`` and ``response_format`` (an OpenAI response
                format); a warning names the others.

        Returns:
            The chat's token ids.

        Raises:
            MistralCommonException: mistral-common rejects the chat as an invalid request,
                for example a reasoning effort the tokenizer file does not declare, a
                message it cannot convert, or generation-prompt flags that contradict the
                final message.
            ValueError: ``mm_data`` holds other media than images, or not one image per
                image part.
        """
        from mistral_common.exceptions import InvalidRequestException
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        from tensorrt_llm._torch.models.modeling_mistral import MistralCommonImageProcessor

        mm_data = mm_data or {}
        if set(mm_data) - {"image"}:
            raise ValueError("Mistral-native checkpoints currently support image inputs only")
        images = MistralCommonImageProcessor.image_list(mm_data.get("image"))
        slotted = []
        num_slots = 0
        for message in messages:
            if message.get("role") == "developer":
                message = {**message, "role": "system"}
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") != "text":
                        part = {"type": "image_url", "image_url": {"url": _IMAGE_SLOT}}
                        num_slots += 1
                    parts.append(part)
                message = {**message, "content": parts}
            slotted.append(message)
        if num_slots != len(images):
            # mistral-common sizes each image's tokens from the decoded image, so the images
            # must be loaded before the chat is rendered.
            raise ValueError(
                f"The chat has {num_slots} images, but {len(images)} were loaded for "
                "rendering; this route does not take images for Mistral-native checkpoints."
            )
        image_sizes = [MistralCommonImageProcessor.pil_image_size(image) for image in images]

        requested = chat_template_kwargs or {}
        continue_final_message = bool(requested.get("continue_final_message", False))
        _check_generation_prompt(messages, add_generation_prompt, continue_final_message)
        ignored = sorted(set(requested) - _CHAT_KWARGS)
        if documents is not None:
            ignored.append("documents")
        if chat_template is not None:
            ignored.append("chat_template")
        if ignored:
            message = (
                f"Mistral-native checkpoints render chats with mistral-common, which "
                f"does not use {ignored}; ignoring them."
            )
            logger.warning_once(message, key=message)
        response_format = _response_format(requested.get("response_format"))
        # mistral-common before 1.13.0 has no response formats.
        request_kwargs = {} if response_format is None else {"response_format": response_format}
        try:
            request = ChatCompletionRequest.from_openai(
                messages=slotted,
                tools=_tools(tools),
                continue_final_message=continue_final_message,
                reasoning_effort=requested.get("reasoning_effort"),
                **request_kwargs,
            )
        except ValueError as error:
            # mistral-common validates the messages as it converts them (a pydantic
            # ValidationError is a ValueError); they make an invalid request.
            raise InvalidRequestException(str(error)) from error
        filled = 0
        for message in request.messages:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue
            chunks = []
            for chunk in content:
                if isinstance(chunk, ImageURLChunk) and chunk.get_url() == _IMAGE_SLOT:
                    chunk = ImageChunk(image=Image.new("1", image_sizes[filled]))
                    filled += 1
                chunks.append(chunk)
            message.content = chunks
        if filled != len(image_sizes):
            raise ValueError(f"The chat has {filled} image slots for {len(image_sizes)} images.")
        return self.grid_only_mistral.encode_chat_completion(request).tokens

    def apply_chat_template(
        self, conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]], *args, **kwargs
    ) -> Union[str, list[int], list[str], list[list[int]]]:
        return self.transformers_tokenizer.apply_chat_template(conversation, *args, **kwargs)
