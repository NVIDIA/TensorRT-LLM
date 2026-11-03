# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import llguidance
import llguidance.torch
import torch
import xgrammar

from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

from ...bindings.executor import GuidedDecodingParams


def _drop_negative_lower_bounds(node: Any) -> Any:
    """Copy of a JSON (schema) document without negative lower bounds.

    xgrammar's range regex for a negative ``minimum`` / ``exclusiveMinimum``
    admits a bare "-" and rejects in-range negatives; without the bound the
    grammar still enforces the type.
    """
    if isinstance(node, dict):
        return {
            key: _drop_negative_lower_bounds(value)
            for key, value in node.items()
            if not (key in ("minimum", "exclusiveMinimum") and isinstance(
                value, (int,
                        float)) and not isinstance(value, bool) and value < 0)
        }
    if isinstance(node, list):
        return [_drop_negative_lower_bounds(value) for value in node]
    return node


def _without_negative_lower_bounds(document: str) -> str:
    return json.dumps(_drop_negative_lower_bounds(json.loads(document)))


class GrammarMatcher(ABC):

    @abstractmethod
    def accept_token(self, token_id: int) -> bool:
        pass

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        pass

    @abstractmethod
    def fill_next_token_bitmask(self, next_token_bitmask: torch.Tensor,
                                index: int) -> None:
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass


class GrammarMatcherFactory(ABC):

    @abstractmethod
    def create(self,
               guided_decoding_params: GuidedDecodingParams,
               stop_token_ids: Optional[List[int]] = None) -> GrammarMatcher:
        """Build a matcher for one request.

        ``stop_token_ids`` are the request's own terminators; a backend that
        supports it admits them once the grammar is complete instead of only
        the tokenizer's EOS.
        """


class XGrammarMatcher(GrammarMatcher):

    def __init__(self, matcher: xgrammar.GrammarMatcher):
        super().__init__()
        self._matcher = matcher

    def accept_token(self, token_id: int) -> bool:
        return self._matcher.accept_token(token_id)

    def rollback(self, num_tokens: int) -> None:
        self._matcher.rollback(num_tokens)

    def fill_next_token_bitmask(self, next_token_bitmask: torch.Tensor,
                                index: int) -> None:
        self._matcher.fill_next_token_bitmask(next_token_bitmask, index)

    def is_terminated(self) -> bool:
        return self._matcher.is_terminated()


class XGrammarMatcherFactory(GrammarMatcherFactory):

    def __init__(self,
                 guided_decoding_config: GuidedDecodingConfig,
                 vocab_size_padded: int,
                 max_num_draft_tokens: int = 0):
        super().__init__()
        vocab_type = xgrammar.VocabType.RAW
        add_prefix_space = False
        if guided_decoding_config.tokenizer_str is not None:
            metadata = xgrammar.TokenizerInfo._detect_metadata_from_hf(
                guided_decoding_config.tokenizer_str)
            vocab_type = metadata["vocab_type"]
            add_prefix_space = metadata["add_prefix_space"]
        elif guided_decoding_config.vocab_type is not None:
            vocab_type = xgrammar.VocabType[
                guided_decoding_config.vocab_type.upper()]

        tokenizer_info = xgrammar.TokenizerInfo(
            guided_decoding_config.encoded_vocab,
            vocab_type=vocab_type,
            vocab_size=vocab_size_padded,
            stop_token_ids=guided_decoding_config.stop_token_ids,
            add_prefix_space=add_prefix_space)

        # Default cache limit is 1GB.
        cache_limit_gb = float(os.getenv("XGRAMMAR_CACHE_LIMIT_GB", "1"))
        cache_limit_bytes = int(cache_limit_gb * 1024 * 1024 * 1024)
        self._xgrammar_compiler = xgrammar.GrammarCompiler(
            tokenizer_info,
            cache_enabled=True,
            cache_limit_bytes=cache_limit_bytes,
        )
        self.max_num_draft_tokens = max_num_draft_tokens

    def create(self,
               guided_decoding_params: GuidedDecodingParams,
               stop_token_ids: Optional[List[int]] = None) -> XGrammarMatcher:
        guide_type = guided_decoding_params.guide_type
        guide = guided_decoding_params.guide
        match guide_type:
            case GuidedDecodingParams.GuideType.JSON:
                compiled_grammar = self._xgrammar_compiler.compile_builtin_json_grammar(
                )
            case GuidedDecodingParams.GuideType.JSON_SCHEMA:
                compiled_grammar = self._xgrammar_compiler.compile_json_schema(
                    _without_negative_lower_bounds(guide))
            case GuidedDecodingParams.GuideType.REGEX:
                compiled_grammar = self._xgrammar_compiler.compile_regex(guide)
            case GuidedDecodingParams.GuideType.EBNF_GRAMMAR:
                compiled_grammar = self._xgrammar_compiler.compile_grammar(
                    guide)
            case GuidedDecodingParams.GuideType.STRUCTURAL_TAG:
                # The tag's embedded json_schema formats need the same care.
                compiled_grammar = self._xgrammar_compiler.compile_structural_tag(
                    _without_negative_lower_bounds(guide))
            case _:
                raise ValueError(f"Unsupported guide type: {guide_type}.")

        matcher = xgrammar.GrammarMatcher(
            compiled_grammar,
            override_stop_tokens=stop_token_ids or None,
            max_rollback_tokens=self.max_num_draft_tokens)
        return XGrammarMatcher(matcher)


class LLGuidanceMatcher(GrammarMatcher):

    def __init__(self, matcher: llguidance.LLMatcher, eos_token: int):
        super().__init__()
        self._matcher = matcher
        self._eos_token = eos_token
        self._is_terminated = False

    def accept_token(self, token_id: int) -> bool:
        if self._matcher.is_stopped():
            # Accept EOS token only if the matcher is stopped.
            if token_id == self._eos_token:
                self._is_terminated = True
                return True
            else:
                return False

        num_accepted = self._matcher.try_consume_tokens([token_id])
        self._check_err()
        return num_accepted > 0

    def rollback(self, num_tokens: int) -> None:
        if num_tokens == 0:
            return
        if self._is_terminated:
            self._is_terminated = False
            num_tokens -= 1
        self._matcher.rollback(num_tokens)
        self._check_err()

    def fill_next_token_bitmask(self, next_token_bitmask: torch.Tensor,
                                index: int) -> None:
        llguidance.torch.fill_next_token_bitmask(self._matcher,
                                                 next_token_bitmask, index)
        self._check_err()

    def is_terminated(self) -> bool:
        return self._is_terminated

    def _check_err(self) -> None:
        if self._matcher.is_error():
            raise ValueError(
                f"LLGuidance matcher error: {self._matcher.get_error()}")


class LLGuidanceMatcherFactory(GrammarMatcherFactory):

    def __init__(self, guided_decoding_config: GuidedDecodingConfig,
                 vocab_size_padded: int):
        super().__init__()
        tokenizer_str = guided_decoding_config.tokenizer_str
        stop_token_ids = guided_decoding_config.stop_token_ids

        if tokenizer_str is None:
            raise ValueError("tokenizer_str is required")

        eos_token = None
        if stop_token_ids is not None:
            if len(stop_token_ids) != 1:
                raise ValueError("expected stop_token_ids size to be 1")
            eos_token = stop_token_ids[0]

        self._tokenizer = llguidance.LLTokenizer(tokenizer_str,
                                                 n_vocab=vocab_size_padded,
                                                 eos_token=eos_token)

    def create(self,
               guided_decoding_params: GuidedDecodingParams,
               stop_token_ids: Optional[List[int]] = None) -> LLGuidanceMatcher:
        # LLGuidance terminates on the single EOS token of its config.
        del stop_token_ids
        guide_type = guided_decoding_params.guide_type
        guide = guided_decoding_params.guide

        grammar = None
        match guide_type:
            case GuidedDecodingParams.GuideType.JSON:
                grammar = llguidance.LLMatcher.grammar_from_json_schema(
                    '{"type": "object"}')
            case GuidedDecodingParams.GuideType.JSON_SCHEMA:
                grammar = llguidance.LLMatcher.grammar_from_json_schema(guide)
            case GuidedDecodingParams.GuideType.REGEX:
                grammar = llguidance.LLMatcher.grammar_from_regex(guide)
            case GuidedDecodingParams.GuideType.EBNF_GRAMMAR:
                # Note: LLGuidance expects Lark grammar format, not standard EBNF.
                # When using LLGuidance backend with EBNF_GRAMMAR type, users must
                # provide Lark-formatted grammar instead of standard EBNF.
                grammar = llguidance.LLMatcher.grammar_from_lark(guide)
            case _:
                raise ValueError(f"Unsupported guide type: {guide_type}.")

        matcher = llguidance.LLMatcher(self._tokenizer, grammar)
        if matcher.is_error():
            raise ValueError(f"LLGuidance matcher error: {matcher.get_error()}")

        return LLGuidanceMatcher(matcher, self._tokenizer.eos_token)
