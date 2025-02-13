import json
import os
from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple, Union

import torch
from pydantic import BaseModel

from tensorrt_llm.bindings import executor as tllme
from tensorrt_llm.logger import logger


@dataclass(slots=True, kw_only=True)
class GuidedDecodingParams:
    """
    Guided decoding parameters for text generation. Only one of the fields could be effective.

    Args:
        json (str, BaseModel, dict, optional): The generated text is amenable to json format with additional user-specified restrictions, namely schema. Defaults to None.
        regex (str, optional): The generated text is amenable to the user-specified regular expression. Defaults to None.
        grammar (str, optional): The generated text is amenable to the user-specified extended Backus-Naur form (EBNF) grammar. Defaults to None.
        json_object (bool): If True, the generated text is amenable to json format. Defaults to False.
    """
    json: Optional[Union[str, BaseModel, dict]] = None
    regex: Optional[str] = None
    grammar: Optional[str] = None
    json_object: bool = False

    @property
    def num_guides(self):
        num_guides = 0
        for field in fields(self):
            num_guides += bool(getattr(self, field.name))
        return num_guides

    def _validate(self):
        if (num_guides := self.num_guides) > 1:
            raise ValueError(
                f"Only one guide can be used for a request, but got {num_guides}."
            )


@dataclass(slots=True, kw_only=True)
class AdditionalModelOutput:
    """
    An additional output to gather from the model.

    Args:
        name (str): The name of the additional output to gather from the model.
        gather_context (bool): A value indicating whether or not to gather the additional output from the context too. Defaults to False.
    """
    name: str
    gather_context: bool


@dataclass(slots=True, kw_only=True)
class SamplingParams:
    """
    Sampling parameters for text generation.

    Args:
        end_id (int, optional): The end token id. Defaults to None.
        pad_id (int, optional): The pad token id. Defaults to None.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 32.
        max_new_tokens (int, optional): The maximum number of tokens to generate. This argument is being deprecated; please use max_tokens instead. Defaults to None.
        bad (str, List[str], optional): A string or a list of strings that redirect the generation when they are generated, so that the bad strings are excluded from the returned output. Defaults to None.
        bad_token_ids (List[int], optional): A list of token ids that redirect the generation when they are generated, so that the bad ids are excluded from the returned output. Defaults to None.
        stop (str, List[str], optional): A string or a list of strings that stop the generation when they are generated. The returned output will not contain the stop strings unless include_stop_str_in_output is True. Defaults to None.
        stop_token_ids (List[int], optional): A list of token ids that stop the generation when they are generated. Defaults to None.
        include_stop_str_in_output (bool): Whether to include the stop strings in output text. Defaults to False.
        embedding_bias (torch.Tensor, optional): The embedding bias tensor. Expected type is kFP32 and shape is [vocab_size]. Defaults to None.
        external_draft_tokens_config (ExternalDraftTokensConfig, optional): The speculative decoding configuration. Defaults to None.
        logits_post_processor_name (str, optional): The logits postprocessor name. Must correspond to one of the logits postprocessor name provided to the ExecutorConfig. Defaults to None.

        n (int): Number of sequences to generate. Defaults to 1.
        best_of (int, optional): Number of sequences to consider for best output. Defaults to None.
        use_beam_search (bool): Whether to use beam search. Defaults to False.

        beam_width (int): The beam width. Setting 1 disables beam search. This parameter will be deprecated from the LLM API in a future release. Please use n/best_of/use_beam_search instead. Defaults to 1.
        num_return_sequences (int, optional): The number of sequences to return. If set to None, it defaults to the value of `beam_width`. The default is None. This parameter will be deprecated from the LLM API in a future release. Please use n/best_of/use_beam_search instead. Defaults to None.

        top_k (int): Controls number of logits to sample from. Default is 0 (all logits).
        top_p (float): Controls the top-P probability to sample from. Default is 0.f
        top_p_min (float): Controls decay in the top-P algorithm. topPMin is lower-bound. Default is 1.e-6.
        top_p_reset_ids (int): Controls decay in the top-P algorithm. Indicates where to reset the decay. Default is 1.
        top_p_decay (float): Controls decay in the top-P algorithm. The decay value. Default is 1.f
        seed (int): Controls the random seed used by the random number generator in sampling
        random_seed (int): Controls the random seed used by the random number generator in sampling. This argument is being deprecated; please use seed instead.
        temperature (float): Controls the modulation of logits when sampling new tokens. It can have values > 0.f. Default is 1.0f
        min_tokens (int): Lower bound on the number of tokens to generate. Values < 1 have no effect. Default is 1.
        min_length (int): Lower bound on the number of tokens to generate. Values < 1 have no effect. Default is 1. This argument is being deprecated; please use min_tokens instead.
        beam_search_diversity_rate (float): Controls the diversity in beam search.
        repetition_penalty (float): Used to penalize tokens based on how often they appear in the sequence. It can have any value > 0.f. Values < 1.f encourages repetition, values > 1.f discourages it. Default is 1.f
        presence_penalty (float): Used to penalize tokens already present in the sequence (irrespective of the number of appearances). It can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f
        frequency_penalty (float): Used to penalize tokens already present in the sequence (dependent on the number of appearances). It can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f
        length_penalty (float): Controls how to penalize longer sequences in beam search. Default is 0.f
        early_stopping (int): Controls whether the generation process finishes once beamWidth sentences are generated (ends with end_token)
        no_repeat_ngram_size (int): Controls how many repeat ngram size are acceptable. Default is 1 << 30.
        min_p (float): scale the most likely token to determine the minimum token probability. Default is 0.0.

        return_log_probs (bool): Controls if Result should contain log probabilities. Default is false.
        return_context_logits (bool): Controls if Result should contain the context logits. Default is false.
        return_generation_logits (bool): Controls if Result should contain the generation logits. Default is false.
        exclude_input_from_output (bool): Controls if output tokens in Result should include the input tokens. Default is true.
        return_encoder_output (bool): Controls if Result should contain encoder output hidden states (for encoder-only and encoder-decoder models). Default is false.
        return_perf_metrics (bool): Controls if Result should contain the performance metrics for this request. Default is false.
        additional_model_outputs (list[AdditionalModelOutput], optional): The additional outputs to gather from the model.

        lookahead_config (LookaheadDecodingConfig , optional): Lookahead decoding config. Defaults to None.
        guided_decoding (GuidedDecodingParams, optional): Guided decoding params. Defaults to None.

        ignore_eos (bool): Whether to ignore the EOS token and continue generating tokens after the EOS token is generated. Defaults to False.
        detokenize (bool): Whether to detokenize the output. Defaults to True.
        add_special_tokens (bool): Whether to add special tokens to the prompt. Defaults to True.
        truncate_prompt_tokens (int, optional): If set to an integer k, will use only the last k tokens from the prompt (i.e., left truncation). Defaults to None.
        skip_special_tokens (bool): Whether to skip special tokens in the output. Defaults to True.
        spaces_between_special_tokens (bool): Whether to add spaces between special tokens in the output. Defaults to True.
    """
    # [TO DEVELOPER] This class provides an interface to LLMAPI users.
    # Internally, it manages and dispatches fields to Python bindings of C++ objects, currently including:
    # (1) all fields of tllme.SamplingConfig;
    # (2) all fields of tllme.OutputConfig;
    # (3) some fields of tllme.Request.
    # If you changed the implementation of C++ objects and corresponding Python bindings, please update:
    # (1) the fields and corresponding docstring of this class, and
    # (2) the expected_fields defined in _get_xxx_config methods.

    end_id: Optional[int] = None
    pad_id: Optional[int] = None
    max_tokens: int = 32
    max_new_tokens: Optional[int] = None

    bad: Optional[Union[str, List[str]]] = None
    bad_token_ids: Optional[List[int]] = None
    _bad_word_ids: Optional[List[List[int]]] = field(default=None,
                                                     init=False,
                                                     repr=False)
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: bool = False
    _stop_word_ids: Optional[List[List[int]]] = field(default=None,
                                                      init=False,
                                                      repr=False)

    embedding_bias: Optional[torch.Tensor] = None
    external_draft_tokens_config: Optional[
        tllme.ExternalDraftTokensConfig] = None
    logits_post_processor_name: Optional[str] = None

    n: int = 1
    best_of: Optional[int] = None
    use_beam_search: bool = False

    # Keep the below fields in sync with tllme.SamplingConfig or maintin the mapping table.
    beam_width: int = 1
    num_return_sequences: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    top_p_min: Optional[float] = None
    top_p_reset_ids: Optional[int] = None
    top_p_decay: Optional[float] = None
    seed: Optional[int] = None
    random_seed: Optional[int] = None
    temperature: Optional[float] = None
    min_tokens: Optional[int] = None
    min_length: Optional[int] = None
    beam_search_diversity_rate: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None
    min_p: Optional[float] = None

    # Keep the below fields in sync with tllme.OutputConfig
    return_log_probs: bool = False
    return_context_logits: bool = False
    return_generation_logits: bool = False
    exclude_input_from_output: bool = True
    return_encoder_output: bool = False
    return_perf_metrics: bool = False
    additional_model_outputs: Optional[list[AdditionalModelOutput]] = None

    # Lookahead decoding config
    lookahead_config: Optional[tllme.LookaheadDecodingConfig] = None

    # Guided decoding params
    guided_decoding: Optional[GuidedDecodingParams] = None

    # Tokenizer-related configs
    ignore_eos: bool = False
    detokenize: bool = True
    add_special_tokens: bool = True
    truncate_prompt_tokens: Optional[int] = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True

    BATCHED_POST_PROCESSOR_NAME = tllme.Request.BATCHED_POST_PROCESSOR_NAME

    def __post_init__(self):
        if self.pad_id is None:
            self.pad_id = self.end_id

        # Handle the compatibility between OpenAI and HF style-parameters.
        hf_style = self.beam_width > 1 or self.num_return_sequences
        openai_style = self.n > 1 or self.best_of or self.use_beam_search

        if hf_style and openai_style:
            ambiguous_params = {
                'beam_width': self.beam_width,
                'num_return_sequences': self.num_return_sequences,
                'n': self.n,
                'best_of': self.best_of,
                'use_beam_search': self.use_beam_search,
            }
            raise ValueError(
                'Got ambiguous parameters. Please specify either Hugging Face '
                'style parameters (beam_width or num_return_sequences) or '
                'OpenAI style parameters (n, best_of, or use_beam_search), '
                f'but not both: {ambiguous_params}. It is recommended to use '
                'OpenAI style parameters (n, best_of, use_beam_search).')

        if hf_style:
            logger.warning(
                "Please use 'n' and 'best_of' for the LLM API. The use of "
                "'beam_width' and 'num_return_sequences' will be deprecated "
                "in a future release.")
            self.n = self.beam_width
            self.best_of = self.num_return_sequences
            self.use_beam_search = self.beam_width > 1

        self.best_of = self.best_of or self.n

        if (not self.use_beam_search and self.n < self.best_of
                and not self.return_log_probs):
            logger.info(
                f"Enable 'return_log_probs' to trim the {self.n}-best among "
                f"{self.best_of} outputs under sampling decoding.")
            self.return_log_probs = True

        self._validate()

    def _validate(self):
        ''' Verify the sampling parameters.

        This function verifies the sampling parameters in the LLM API, which
        may have stricter requirements than the Executor class of C++ runtime.
        For instance, while the greedy decoding with n > 1 is capable in the
        Executor class of C++ runtime, the LLM API disallows such combination.
        '''
        if self.best_of is not None:
            if self.best_of > 1 and self.best_of < self.n:
                raise ValueError(
                    f'In beam search, beam_width ({self.beam_width}) must be '
                    f'greater than or equal to num_return_sequences '
                    f'({self.num_return_sequences}).')

            if (self.best_of > 1 and self.greedy_decoding and
                    not os.environ.get('TLLM_ALLOW_N_GREEDY_DECODING', None)):
                raise ValueError(
                    f'Greedy decoding in the LLM API does not allow multiple '
                    f'returns. Please set to best_of=1, got best_of={self.best_of}. '
                    f'Please set to best_of=1 or set an environment variable '
                    f'TLLM_ALLOW_N_GREEDY_DECODING=1 to allow best_of > 1 '
                    f'under the greedy decoding.')

        if self.truncate_prompt_tokens is not None and self.truncate_prompt_tokens < 1:
            raise ValueError(
                f"truncate_prompt_tokens must be >= 1, got {self.truncate_prompt_tokens}"
            )

        if self.guided_decoding is not None:
            self.guided_decoding._validate()

    @property
    def greedy_decoding(self) -> bool:
        return (not self.use_beam_search
                and (self.top_k is None or self.top_k == 1)
                and (self.top_p is None or self.top_p == 0.0))

    def setup(self,
              tokenizer,
              add_special_tokens: bool = False) -> 'SamplingParams':
        if self.end_id is None:
            self.end_id = tokenizer.eos_token_id
            self.pad_id = tokenizer.pad_token_id
            if self.pad_id is None:
                self.pad_id = self.end_id

        if self.bad is not None:
            strs = [self.bad] if isinstance(self.bad, str) else self.bad
            self._bad_word_ids = [
                tokenizer.encode(s, add_special_tokens=add_special_tokens)
                for s in strs
            ]

        if self.stop is not None:
            strs = [self.stop] if isinstance(self.stop, str) else self.stop
            self._stop_word_ids = [
                tokenizer.encode(s, add_special_tokens=add_special_tokens)
                for s in strs
            ]

        return self

    def _get_bad_words(self) -> List[List[int]]:
        words = []
        if self.bad_token_ids is not None:
            words = [[i] for i in self.bad_token_ids]

        if self.bad is None:
            return words
        else:
            if self._bad_word_ids is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.bad ({self.bad}) is not processed by tokenizer, "
                    "please call the setup method.")
            return words + self._bad_word_ids

    def _get_stop_words(self) -> List[List[int]]:
        words = []
        if self.stop_token_ids is not None:
            words = [[i] for i in self.stop_token_ids]

        if self.stop is None:
            return words
        else:
            if self._stop_word_ids is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.stop ({self.stop}) is not processed by tokenizer, "
                    "please call the setup method.")
            return words + self._stop_word_ids

    def _get_stop_reasons_and_words(
            self) -> List[Tuple[Union[str, int], List[int]]]:
        stop_reasons = []
        if self.stop_token_ids is not None:
            stop_reasons.extend(self.stop_token_ids)
        if self.stop is not None:
            if isinstance(self.stop, str):
                stop_reasons.append(self.stop)
            else:
                stop_reasons.extend(self.stop)
        stop_words = self._get_stop_words()
        if len(stop_reasons) != len(stop_words):
            raise RuntimeError(
                f"The number of {self.__class__.__name__}.stop_token_ids ({self.stop_token_ids}) "
                f"and {self.__class__.__name__}.stop ({self.stop}) are inconsistent with the "
                f"processed stop_words ({stop_words}).")
        return list(zip(stop_reasons, stop_words))

    def _get_sampling_config(self) -> tllme.SamplingConfig:
        expected_fields = {
            "beam_width", "top_k", "top_p", "top_p_min", "top_p_reset_ids",
            "top_p_decay", "seed", "random_seed", "temperature", "min_tokens",
            "min_length", "beam_search_diversity_rate", "repetition_penalty",
            "presence_penalty", "frequency_penalty", "length_penalty",
            "early_stopping", "no_repeat_ngram_size", "num_return_sequences",
            "min_p"
        }
        found_fields = {
            f
            for f in dir(tllme.SamplingConfig) if not f.startswith('__')
        }

        if found_fields != expected_fields:
            raise RuntimeError(
                "Found fields in `tllme.SamplingConfig` different than expected; "
                f"if `tllme.SamplingConfig` is changed, please update {self.__class__.__name__} accordingly. "
                "See [TO DEVELOPER] comments for detailed instructions.")

        # A map from the SamplingConfig fields of the LLM API to their
        # corresponding field names of the Executor of TRT-LLM C++ runtime.
        # In sampling, there is no parameter that directly matches 'best_of',
        # so outputs must be trimmed during postprocessing.
        #               |     LLM API     |    TRT-LLM Executor    |
        # --------------|-----------------|------------------------|
        # | Beam search | use_beam_search | beam_width > 1         |
        # | Beam search | n               | num_return_sequences   |
        # | Beam search | best_of         | beam_width             |
        # |-------------|-----------------|------------------------|
        # | Sampling    | use_beam_search | beam_width == 1        |
        # | Sampling    | n               | num_return_sequences   |
        # | Sampling    | best_of         | no corresponding param |
        unmatched_params = [
            'num_return_sequences', 'beam_width', 'n', 'best_of',
            'use_beam_search'
        ]
        llmapi_to_rt_param_map = {
            f: getattr(self, f)
            for f in expected_fields if f not in unmatched_params
        }
        if self.use_beam_search:
            llmapi_to_rt_param_map['num_return_sequences'] = self.n
            llmapi_to_rt_param_map['beam_width'] = self.best_of
        else:
            llmapi_to_rt_param_map['num_return_sequences'] = self.best_of
            llmapi_to_rt_param_map['beam_width'] = 1

        return tllme.SamplingConfig(**llmapi_to_rt_param_map)

    def _get_output_config(self) -> tllme.OutputConfig:
        expected_fields = [
            "return_log_probs", "return_context_logits",
            "return_generation_logits", "exclude_input_from_output",
            "return_encoder_output", "return_perf_metrics",
            "additional_model_outputs"
        ]
        found_fields = [
            f for f in dir(tllme.OutputConfig) if not f.startswith('__')
        ]
        if set(found_fields) != set(expected_fields):
            raise RuntimeError(
                "Found fields in `tllme.OutputConfig` different than expected; "
                f"if `tllme.OutputConfig` is changed, please update {self.__class__.__name__} accordingly. "
                "See [TO DEVELOPER] comments for detailed instructions.")
        return tllme.OutputConfig(
            **{f: getattr(self, f)
               for f in expected_fields})

    def _get_guided_decoding_params(self) -> tllme.GuidedDecodingParams:
        if self.guided_decoding is None or self.guided_decoding.num_guides == 0:
            return None

        if self.guided_decoding.json_object:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.JSON)
        elif self.guided_decoding.json is not None:
            json_schema = self.guided_decoding.json
            if isinstance(json, BaseModel):
                json_schema = json_schema.model_json_schema()
            if isinstance(json_schema, dict):
                json_schema = json.dumps(json_schema)
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.JSON_SCHEMA, json_schema)
        elif self.guided_decoding.regex is not None:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.REGEX,
                self.guided_decoding.regex)
        else:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.EBNF_GRAMMAR,
                self.guided_decoding.grammar)
