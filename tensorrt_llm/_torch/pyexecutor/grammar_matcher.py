import json
from abc import ABC, abstractmethod

import llguidance
import llguidance.torch
import torch
import xgrammar

from ...bindings.executor import GuidedDecodingConfig, GuidedDecodingParams


class GrammarMatcher(ABC):

    @abstractmethod
    def accept_token(self, token_id: int) -> bool:
        pass

    @abstractmethod
    def fill_next_token_bitmask(self, next_token_bitmask: torch.Tensor,
                                index: int) -> None:
        pass


class GrammarMatcherFactory(ABC):

    @abstractmethod
    def create(self,
               guided_decoding_params: GuidedDecodingParams) -> GrammarMatcher:
        pass


class XGrammarMatcher(GrammarMatcher):

    def __init__(self, matcher: xgrammar.GrammarMatcher):
        super().__init__()
        self._matcher = matcher

    def accept_token(self, token_id: int) -> bool:
        return self._matcher.accept_token(token_id)

    def fill_next_token_bitmask(self, next_token_bitmask: torch.Tensor,
                                index: int) -> None:
        self._matcher.fill_next_token_bitmask(next_token_bitmask, index)


class XGrammarMatcherFactory(GrammarMatcherFactory):

    def __init__(self, guided_decoding_config: GuidedDecodingConfig,
                 vocab_size_padded: int):
        super().__init__()
        if guided_decoding_config.tokenizer_str is not None:
            metadata = xgrammar.TokenizerInfo._detect_metadata_from_hf(
                guided_decoding_config.tokenizer_str)
            tokenizer_info = xgrammar.TokenizerInfo(
                guided_decoding_config.encoded_vocab,
                vocab_type=metadata["vocab_type"],
                vocab_size=vocab_size_padded,
                stop_token_ids=guided_decoding_config.stop_token_ids,
                add_prefix_space=metadata["add_prefix_space"])
        else:
            tokenizer_info = xgrammar.TokenizerInfo(
                guided_decoding_config.encoded_vocab,
                xgrammar.VocabType.RAW,
                vocab_size=vocab_size_padded,
                stop_token_ids=guided_decoding_config.stop_token_ids)
        self._xgrammar_compiler = xgrammar.GrammarCompiler(tokenizer_info)

    def create(self,
               guided_decoding_params: GuidedDecodingParams) -> XGrammarMatcher:
        guide_type = guided_decoding_params.guide_type
        guide = guided_decoding_params.guide
        match guide_type:
            case GuidedDecodingParams.GuideType.JSON:
                compiled_grammar = self._xgrammar_compiler.compile_builtin_json_grammar(
                )
            case GuidedDecodingParams.GuideType.JSON_SCHEMA:
                compiled_grammar = self._xgrammar_compiler.compile_json_schema(
                    guide)
            case GuidedDecodingParams.GuideType.REGEX:
                grammar = xgrammar.Grammar.from_regex(guide)
                compiled_grammar = self._xgrammar_compiler.compile_grammar(
                    grammar)
            case GuidedDecodingParams.GuideType.EBNF_GRAMMAR:
                grammar = xgrammar.Grammar.from_ebnf(guide)
                compiled_grammar = self._xgrammar_compiler.compile_grammar(
                    grammar)
            case GuidedDecodingParams.GuideType.STRUCTURAL_TAG:
                structural_tag_parameters = json.loads(guide)
                structures = structural_tag_parameters["structures"]
                structures = [
                    xgrammar.StructuralTagItem(begin=s["begin"],
                                               schema=json.dumps(s["schema"]),
                                               end=s["end"]) for s in structures
                ]
                triggers = structural_tag_parameters["triggers"]
                compiled_grammar = self._xgrammar_compiler.compile_structural_tag(
                    structures, triggers)
            case _:
                raise ValueError(f"Unrecognized guide type: {guide_type}.")

        matcher = xgrammar.GrammarMatcher(compiled_grammar)
        return XGrammarMatcher(matcher)


class LLGuidanceMatcher(GrammarMatcher):

    def __init__(self, matcher: llguidance.LLMatcher):
        super().__init__()
        self._matcher = matcher

    def accept_token(self, token_id: int) -> bool:
        result = self._matcher.consume_token(token_id)
        self._check_err()
        return result

    def fill_next_token_bitmask(self, next_token_bitmask: torch.Tensor,
                                index: int) -> None:
        llguidance.torch.fill_next_token_bitmask(self._matcher,
                                                 next_token_bitmask, index)
        self._check_err()

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

    def create(
            self,
            guided_decoding_params: GuidedDecodingParams) -> LLGuidanceMatcher:
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
                raise ValueError(f"Unrecognized guide type: {guide_type}.")

        matcher = llguidance.LLMatcher(self._tokenizer, grammar)
        if matcher.is_error():
            raise ValueError(f"LLGuidance matcher error: {matcher.get_error()}")

        return LLGuidanceMatcher(matcher)
