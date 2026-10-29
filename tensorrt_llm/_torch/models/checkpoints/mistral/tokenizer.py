# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Union

from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from pydantic import ValidationError
from transformers.tokenization_mistral_common import (
    MistralCommonTokenizer as TransformersMistralTokenizer,
)

from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    import llguidance
    from mistral_common.guidance.grammar_factory import GrammarFactory


# Adapted from:
# https://github.com/vllm-project/vllm/blob/8e67b2557aae7204c697d7a5c61e00754da465be/vllm/transformers_utils/tokenizers/mistral.py#L166
class MistralTokenizer(TransformersTokenizer):
    def __init__(self, tokenizer: "TransformersMistralTokenizer"):
        self._compat_tokenizer_dir = None
        self._compat_tokenizer_path = None
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

    def __reduce__(self):
        if self._compat_tokenizer_path is not None:
            # Workers rebuild their own temporary copy from the original checkpoint.
            return (self.__class__.from_pretrained, (self._compat_tokenizer_path,))
        # TransformersTokenizer.__reduce__ would rebuild a TransformersTokenizer
        # around ``self.tokenizer``, which here is the raw mistral-common
        # tokenizer. Rebuild this class from the transformers wrapper instead;
        # mistral-common pickles the tokenizer by the path of its file.
        return (self.__class__, (self.transformers_tokenizer,))

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
        try:
            if Path(pretrained_model_dir).is_file():
                tokenizer = TransformersMistralTokenizer(tokenizer_path=pretrained_model_dir)
            else:
                tokenizer = TransformersMistralTokenizer.from_pretrained(pretrained_model_dir)
        except ValidationError as error:
            # A mistral-common without JSON-schema model settings rejects a tekken.json
            # whose model_settings_builder declares json_schema; load it without that entry.
            # TODO: Remove this fallback and its tests: requirements.txt needs mistral-common
            # 1.13.0, which reads model_settings_builder.json_schema.
            errors = error.errors()
            if error.title == "ModelSettingsBuilder" and any(
                err["loc"][:1] == ("reasoning_effort",) for err in errors
            ):
                raise ValueError(
                    f"The tekken.json of {pretrained_model_dir} declares reasoning efforts the "
                    "installed mistral-common does not know. Install mistral-common 1.13.0 or "
                    "later from PyPI."
                ) from error
            if (
                error.title != "ModelSettingsBuilder"
                or len(errors) != 1
                or (errors[0]["loc"] != ("json_schema",) or errors[0]["type"] != "extra_forbidden")
            ):
                raise
            tokenizer_path = Path(pretrained_model_dir)
            if tokenizer_path.is_dir():
                tokenizer_path /= "tekken.json"
            with tokenizer_path.open(encoding="utf-8") as source:
                metadata = json.load(source)
            settings = metadata.get("model_settings_builder", {})
            if settings.get("json_schema") != {
                "type": "json_schema",
                "accepts_none": False,
                "default": None,
            }:
                raise
            del settings["json_schema"]
            # mistral-common keeps the file path for serialization and save_pretrained.
            temporary_dir = TemporaryDirectory(prefix="trtllm-tekken-")
            compatible_path = Path(temporary_dir.name) / tokenizer_path.name
            compatible_path.write_text(json.dumps(metadata), encoding="utf-8")
            tokenizer = TransformersMistralTokenizer(tokenizer_path=str(compatible_path))
            tokenizer.name_or_path = str(pretrained_model_dir)
            result = cls(tokenizer)
            result._compat_tokenizer_dir = temporary_dir
            result._compat_tokenizer_path = str(pretrained_model_dir)
            logger.warning(
                "Ignoring unsupported model_settings_builder.json_schema metadata for "
                "text tokenization; JSON-schema model settings remain unsupported by "
                "the installed mistral-common."
            )
            return result
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

        return llguidance.LLTokenizer(
            llguidance.TokenizerWrapper(MistralLLGTokenizer(self.tokenizer)),
            n_vocab=n_vocab,
            eos_token=self.eos_token_id,
        )

    def llguidance_json_schema_lark(self, json_schema: str) -> str:
        """The Lark grammar of a JSON response, as mistral-common's grammar factory writes it.

        With the ``[THINK]`` and ``[/THINK]`` control tokens in the vocabulary, the
        response may open with one reasoning section, ``[THINK]`` text ``[/THINK]``,
        before the JSON: the model reasons there when the request turns reasoning on.
        Otherwise the response is the JSON alone.

        Args:
            json_schema: The JSON schema of the response, serialized.

        Returns:
            The Lark grammar an llguidance matcher compiles against
            `llguidance_tokenizer`, which references the control tokens by id.
        """
        factory, template = self._grammar_factory
        return factory.get_lark_for_json_schema(template, json.loads(json_schema))

    def llguidance_tool_grammar(
        self, tools: list[dict], tool_choice, parallel_tool_calls=None
    ) -> str:
        """Render native tool calls, including their reasoning and control tokens.

        A named tool choice is resolved as mistral-common's
        `GrammarFactory.get_grammar_for_chat_request` resolves it: the grammar
        offers that tool alone, requires a call, and pins the call to its name.
        """
        from mistral_common.protocol.instruct.tool_calls import (
            NamedToolChoice,
            Tool,
            ToolChoiceEnum,
        )

        native_tools = [Tool.from_openai(tool) for tool in tools]
        if isinstance(tool_choice, dict):
            name = NamedToolChoice.model_validate(tool_choice).function.name
            native_tools = [tool for tool in native_tools if tool.function.name == name]
            if not native_tools:
                raise ValueError(f"Named tool {name!r} is not declared in tools.")
            mode, force_tool_names = ToolChoiceEnum.required, True
        else:
            mode, force_tool_names = ToolChoiceEnum(tool_choice), False
        factory, template = self._grammar_factory
        return factory.get_lark_from_jinja(
            template=template,
            mode=mode,
            tools=native_tools,
            json_schema=None,
            parallel_tool_calls=parallel_tool_calls,
            json_only=False,
            force_tool_names=force_tool_names,
        )

    @functools.cached_property
    def _grammar_factory(self) -> tuple["GrammarFactory", str]:
        from mistral_common.guidance.grammar_factory import GrammarFactory

        if not GrammarFactory.is_supported(self.mistral):
            raise ValueError(
                f"mistral-common renders no grammars for this tokenizer (version v{self.version})."
            )
        factory = GrammarFactory(self.mistral)
        return factory, factory.select_jinja_template()

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
            # Kept special tokens stay, as decode() keeps them: streamed text then
            # carries the control tokens tool and reasoning parsers read.
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

    def apply_chat_template(
        self, conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]], *args, **kwargs
    ) -> Union[str, list[int], list[str], list[list[int]]]:
        return self.transformers_tokenizer.apply_chat_template(conversation, *args, **kwargs)
