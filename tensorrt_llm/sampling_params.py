import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
from pydantic import BaseModel

from tensorrt_llm.bindings import executor as tllme
from tensorrt_llm.logger import logger


@dataclass(slots=True, kw_only=True)
class GuidedDecodingParams:
    """Guided decoding parameters for text generation. Only one of the fields could be effective.

    Args:
        json (str, pydantic.main.BaseModel, dict, optional): The generated text is amenable to json format with additional user-specified restrictions, namely schema. Defaults to None.
        regex (str, optional): The generated text is amenable to the user-specified regular expression. Defaults to None.
        grammar (str, optional): The generated text is amenable to the user-specified extended Backus-Naur form (EBNF) grammar. Defaults to None.
        json_object (bool): If True, the generated text is amenable to json format. Defaults to False.
        structural_tag (str, optional): The generated text is amenable to the user-specified structural tag. Structural tag is supported by xgrammar backend only. Defaults to None.
    """  # noqa: E501

    json: Optional[Union[str, BaseModel, dict]] = None
    regex: Optional[str] = None
    grammar: Optional[str] = None
    json_object: bool = False
    structural_tag: Optional[str] = None

    def _validate(self):
        num_guides = 0
        for _field in fields(self):
            num_guides += bool(getattr(self, _field.name))
        if num_guides > 1:
            raise ValueError(f"Only one guide can be used for a request, but got {num_guides}.")


class LogprobParams(NamedTuple):
    prompt_logprobs: Optional[int] = None
    logprobs: Optional[int] = None
    # Drop the logits once the logprobs are computed
    drop_context_logits: bool = False
    # Drop the geneation_logits once the logprobs are computed
    drop_generation_logits: bool = False


class LogitsProcessor(ABC):
    """Base class for logits processor.

    The recommended way to create a customized logits processor:
        * Subclass this class and implement the processing logics in the __call__ method.
        * Create an instance and pass to SamplingParams.
    Alternatively, you can create any callable with the same signature with the __call__ method.
    """

    @abstractmethod
    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        """Logits processing callback. The callback is expected to inplace modify the logits.

        Args:
            req_id (int): Request id.
            logits (torch.Tensor): Logits tensor to be modified.
            token_ids (List[List[int]]): Token ids produced by the request so far.
                The shape is beam_width * sequence_length.
            stream_ptr (int, optional): The operation stream used by the logits tensor.
                Not required for PyTorch backend.
            client_id (int, optional): An optional client id.
        """
        pass  # noqa


class BatchedLogitsProcessor(ABC):
    """Base class for batched logits processor.

    The recommended way to create a customized batched logits processor:
        * Subclass this class and implement the processing logics in the __call__ method.
        * Create an instance and pass to LLM.
    Alternatively, you can create any callable with the same signature with the __call__ method.
    """

    @abstractmethod
    def __call__(
        self,
        req_ids: List[int],
        logits: List[torch.Tensor],
        token_ids: List[List[List[int]]],
        stream_ptr: int,
        client_ids: List[Optional[int]],
    ) -> None:
        """Batched logits processing callback. The callback is expected to inplace modify the logits.

        Args:
            req_ids (List[int]): A batch of request ids.
            logits (List[torch.Tensor]): A batch of the logits tensors.
            token_ids (List[List[List[int]]]): A batch of the token ids produced by the requests so far.
                The shape is batch * beam_width * sequence_length.
            stream_ptr (int): The operation stream used by the logits tensors.
            client_ids (List[Optional[int]]): A batch of optional client ids.
        """
        pass  # noqa


@dataclass(slots=True, kw_only=True)
class SamplingParams:
    """Sampling parameters for text generation.

    Usage Examples:

        use_beam_search is False:
            - best_of is None: (top-p/top-k) sampling n responses and return n generations
            - best_of is not None: (top-p/top-k) sampling best_of responses and return n generations (best_of >= n must hold)
        use_beam_search is True:
            - best_of is None: beam search with beam width of n, return n generations
            - best_of is not None: beam search with beam width of best_of, return n generations (best_of >= n must hold)

    Args:
        end_id (int, optional): The end token id. Defaults to None.
        pad_id (int, optional): The pad token id. Defaults to None.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 32.
        bad (str, List[str], optional): A string or a list of strings that redirect the generation when they are generated, so that the bad strings are excluded from the returned output. Defaults to None.
        bad_token_ids (List[int], optional): A list of token ids that redirect the generation when they are generated, so that the bad ids are excluded from the returned output. Defaults to None.
        stop (str, List[str], optional): A string or a list of strings that stop the generation when they are generated. The returned output will not contain the stop strings unless include_stop_str_in_output is True. Defaults to None.
        stop_token_ids (List[int], optional): A list of token ids that stop the generation when they are generated. Defaults to None.
        include_stop_str_in_output (bool): Whether to include the stop strings in output text. Defaults to False.
        embedding_bias (torch.Tensor, optional): The embedding bias tensor. Expected type is kFP32 and shape is [vocab_size]. Defaults to None.
        logits_processor (tensorrt_llm.sampling_params.LogitsProcessor, List[tensorrt_llm.sampling_params.LogitsProcessor], optional): The logits postprocessor callback(s). Defaults to None.
            If a list, each processor is applied in order during generation (supported in PyTorch backend only).
        apply_batched_logits_processor (bool): Whether to apply batched logits postprocessor callback. Defaults to False.
            The BatchedLogitsProcessor class is recommended for callback creation. The callback must be provided when initializing LLM.

        n (int): Number of sequences to generate. Defaults to 1.
        best_of (int, optional): Number of sequences to consider for best output. Defaults to None.
        use_beam_search (bool): Whether to use beam search. Defaults to False.

        top_k (int, optional): Controls number of logits to sample from. Can assume non-negative values, where 0 means 'all logits'. Defaults to None.
            The value None is treated as "not specified" in the following.
            If neither temperature, top_p, nor top_k are specified, sampling is greedy.
            If temperature > 0 and/or top_p < 1 are specified, sampling will proceed accordingly and top_k will default to top_k = 0.
            Setting top_k = 1 results in greedy sampling.
        top_p (float, optional): Controls the top-P probability to sample from. Can have values between 0 and 1. Defaults to None.
            The value None is treated as "not specified" in the following.
            If neither temperature, top_p, nor top_k are specified, sampling is greedy.
            If temperature > 0 and/or top_k > 1 are specified, sampling will proceed accordingly and top_p will default to top_p = 1.
            Setting top_p = 0 should result in greedy sampling, but is currently disallowed in the backend.
        top_p_min (float, optional): Controls decay in the top-P algorithm. topPMin is lower-bound. None means using C++ runtime default 1.e-6. Defaults to None.
        top_p_reset_ids (int, optional): Controls decay in the top-P algorithm. Indicates where to reset the decay. None means using C++ runtime default 1. Defaults to None.
        top_p_decay (float, optional): Controls decay in the top-P algorithm. The decay value. None means using C++ runtime default 1.f. Defaults to None.
        seed (int, optional): Controls the random seed used by the random number generator in sampling. None means using C++ runtime default 0. Defaults to None.
        temperature (float, optional): Controls the modulation of logits when sampling new tokens. It can have values >= 0.f. Defaults to None.
            The value None is treated as "not specified" in the following.
            If neither temperature, top_p, nor top_k are specified, sampling is greedy.
            If top_p < 1 and/or top_k > 1 are specified, sampling will proceed accordingly and temperature will default to temperature = 1.
            Setting temperature = 0 results in greedy sampling.
        min_tokens (int, optional): Lower bound on the number of tokens to generate. Values < 1 have no effect. None means using C++ runtime default 1. Defaults to None.
        beam_search_diversity_rate (float, optional): Used to penalize tokens based on how often they appear in the sequence. It can have any value > 0.f. Values < 1.f encourages repetition, values > 1.f discourages it. None means using C++ runtime default 1.f. Defaults to None.
        repetition_penalty (float, optional): Used to penalize tokens based on how often they appear in the sequence. It can have any value > 0.f. Values < 1.f encourages repetition, values > 1.f discourages it. None means using C++ runtime default 1.f. Defaults to None.
        presence_penalty (float, optional): Used to penalize tokens already present in the sequence (irrespective of the number of appearances). It can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. None means using C++ runtime default 0.f. Defaults to None.
        frequency_penalty (float, optional): Used to penalize tokens already present in the sequence (dependent on the number of appearances). It can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. None means using C++ runtime default 0.f. Defaults to None.
        length_penalty (float, optional): Controls how to penalize longer sequences in beam search. None means using C++ runtime default 0.f. Defaults to None.
        early_stopping (int, optional): Controls whether the generation process finishes once beamWidth sentences are generated (ends with end_token).  None means using C++ runtime default 1. Defaults to None.
        no_repeat_ngram_size (int, optional): Controls how many repeat ngram size are acceptable. None means using C++ runtime default 1 << 30. Defaults to None.
        min_p (float, optional): scale the most likely token to determine the minimum token probability. None means using C++ runtime default 0.0. Defaults to None.
        beam_width_array (List[int], optional): The array of beam width using in Variable-Beam-Width-Search. Defaults to None.

        logprobs (int, optional): Number of log probabilities to return per output token. Defaults to None.
        prompt_logprobs (int, optional): Number of log probabilities to return per prompt token. Defaults to None.
        return_context_logits (bool): Controls if Result should contain the context logits. Defaults to False.
        return_generation_logits (bool): Controls if Result should contain the generation logits. Defaults to False.
        exclude_input_from_output (bool): Controls if output tokens in Result should include the input tokens. Defaults to True.
        return_encoder_output (bool): Controls if Result should contain encoder output hidden states (for encoder-only and encoder-decoder models). Defaults to False.
        return_perf_metrics (bool): Controls if Result should contain the performance metrics for this request. Defaults to False.
        additional_model_outputs (List[str], optional): The additional outputs to gather from the model. Defaults to None.

        lookahead_config (tensorrt_llm.bindings.executor.LookaheadDecodingConfig , optional): Lookahead decoding config. Defaults to None.
        guided_decoding (tensorrt_llm.sampling_params.GuidedDecodingParams, optional): Guided decoding params. Defaults to None.

        ignore_eos (bool): Whether to ignore the EOS token and continue generating tokens after the EOS token is generated. Defaults to False.
        detokenize (bool): Whether to detokenize the output. Defaults to True.
        add_special_tokens (bool): Whether to add special tokens to the prompt. Defaults to True.
        truncate_prompt_tokens (int, optional): If set to an integer k, will use only the last k tokens from the prompt (i.e., left truncation). Defaults to None.
        skip_special_tokens (bool): Whether to skip special tokens in the output. Defaults to True.
        spaces_between_special_tokens (bool): Whether to add spaces between special tokens in the output. Defaults to True.
    """  # noqa: E501

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
    bad: Optional[Union[str, List[str]]] = None
    bad_token_ids: Optional[List[int]] = None
    _bad_word_ids: Optional[List[List[int]]] = field(default=None, init=False, repr=False)
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: bool = False
    _stop_word_ids: Optional[List[List[int]]] = field(default=None, init=False, repr=False)

    embedding_bias: Optional[torch.Tensor] = None
    logits_processor: Optional[Union[LogitsProcessor, List[LogitsProcessor]]] = None
    apply_batched_logits_processor: bool = False

    n: int = 1
    best_of: Optional[int] = None
    use_beam_search: bool = False

    # Keep the below fields in sync with tllme.SamplingConfig or maintin the mapping table.
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    top_p_min: Optional[float] = None
    top_p_reset_ids: Optional[int] = None
    top_p_decay: Optional[float] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    min_tokens: Optional[int] = None
    beam_search_diversity_rate: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None
    min_p: Optional[float] = None
    beam_width_array: Optional[List[int]] = None

    # Keep the below fields in sync with tllme.OutputConfig
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    return_context_logits: bool = False
    return_generation_logits: bool = False
    exclude_input_from_output: bool = True
    return_encoder_output: bool = False
    return_perf_metrics: bool = False
    additional_model_outputs: Optional[List[str]] = None

    # Used in logprobs calculation in TRT flow to drop logits early if user did not explicitly request them.
    # Can be deprecated after migration to PyTorch backend.
    _context_logits_auto_enabled: bool = False
    _generation_logits_auto_enabled: bool = False

    # TODO: deprecate this after trtllm-serve migrate to use TopK logprobs
    _return_log_probs: bool = False

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
    # Currently, _stream_interval is only used to pass llm.args.stream_interval to tokenizer.
    # TODO: make this a per-request parameter.
    _stream_interval: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.pad_id is None:
            self.pad_id = self.end_id

        self.best_of = self.best_of or self.n

        if self.embedding_bias is not None:
            if isinstance(self.embedding_bias, torch.Tensor):
                self.embedding_bias = self.embedding_bias.detach().clone()
            else:
                self.embedding_bias = torch.tensor(self.embedding_bias, dtype=torch.float32)

        self._validate()

    def _validate(self):
        """Verify the sampling parameters.

        This function verifies the sampling parameters in the LLM API, which
        may have stricter requirements than the Executor class of C++ runtime.
        For instance, while the greedy decoding with n > 1 is capable in the
        Executor class of C++ runtime, the LLM API disallows such combination.
        """
        if self.top_p is not None and (self.top_p < 0 or self.top_p > 1):
            raise ValueError(f"require 0 <= top_p <= 1, got top_p={self.top_p}")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError(f"require top_k >= 0, got top_k={self.top_k}")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError(f"require temperature >= 0, got temperature={self.temperature}")

        if self.best_of is not None and self.best_of < self.n:
            raise ValueError(f"best_of ({self.best_of}) cannot be less than n ({self.n})")

        if (
            self.best_of is not None
            and self.best_of > 1
            and self._greedy_decoding
            and not os.environ.get("TLLM_ALLOW_N_GREEDY_DECODING", None)
        ):
            raise ValueError(
                f"Greedy decoding in the LLM API does not allow multiple "
                f"returns. Please set to best_of=1, got best_of={self.best_of}. "
                f"Please set to best_of=1 or set an environment variable "
                f"TLLM_ALLOW_N_GREEDY_DECODING=1 to allow best_of > 1 "
                f"under the greedy decoding."
            )

        if self.truncate_prompt_tokens is not None and self.truncate_prompt_tokens < 1:
            raise ValueError(
                f"truncate_prompt_tokens must be >= 1, got {self.truncate_prompt_tokens}"
            )

        if self.guided_decoding is not None:
            self.guided_decoding._validate()

        # correct types as users might pass in logprob=True for Top-1 logprobs
        self.logprobs = self.logprobs and int(self.logprobs)
        self.prompt_logprobs = self.prompt_logprobs and int(self.prompt_logprobs)

    # NB: Static, because downstream code only holds instances of
    #     bindings.SamplingConfig (not SamplingParams).
    @staticmethod
    def params_imply_greedy_decoding(
        *, temperature: Optional[float], top_p: Optional[float], top_k: Optional[int]
    ):
        return (
            (temperature is None and top_p is None and top_k is None)
            or top_k == 1
            or top_p == 0.0
            or temperature == 0
        )

    @property
    def _greedy_decoding(self) -> bool:
        return not self.use_beam_search and self.params_imply_greedy_decoding(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

    @property
    def _need_return_context_logits(self) -> bool:
        return self.return_context_logits and not self._context_logits_auto_enabled

    @property
    def _need_return_generation_logits(self) -> bool:
        return self.return_generation_logits and not self._generation_logits_auto_enabled

    def _setup(
        self, tokenizer, hf_model_config, generation_config, add_special_tokens: bool = False
    ) -> "SamplingParams":
        if self.end_id is None:
            self.end_id = tokenizer.eos_token_id
            self.pad_id = tokenizer.pad_token_id
            # kimi_k2 model uses the eos_token_id in generation config
            if (
                hf_model_config is not None
                and hf_model_config.model_type == "kimi_k2"
                and generation_config is not None
                and isinstance(generation_config.eos_token_id, int)
            ):
                self.end_id = generation_config.eos_token_id

            if self.pad_id is None:
                self.pad_id = self.end_id

        def _encode(tokenizer, text, add_special_tokens):
            try:
                return tokenizer.encode(text, add_special_tokens=add_special_tokens)
            except TypeError:
                # For tiktokenizer, the encode method does not have add_special_tokens argument
                return tokenizer.encode(text)

        if self.bad is not None:
            strs = [self.bad] if isinstance(self.bad, str) else self.bad
            self._bad_word_ids = [_encode(tokenizer, s, add_special_tokens) for s in strs]

        if self.stop is not None:
            strs = [self.stop] if isinstance(self.stop, str) else self.stop
            self._stop_word_ids = [_encode(tokenizer, s, add_special_tokens) for s in strs]

        return self

    def _get_bad_words(self) -> List[List[int]]:
        words = []
        if self.bad_token_ids:
            words = [[i] for i in self.bad_token_ids]

        if self.bad is None:
            return words
        else:
            if self._bad_word_ids is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.bad ({self.bad}) is not processed by tokenizer, "
                    "please call the setup method."
                )
            return words + self._bad_word_ids

    def _get_stop_words(self) -> List[List[int]]:
        words = []
        if self.stop_token_ids:
            words = [[i] for i in self.stop_token_ids]

        if self.stop is None:
            return words
        else:
            if self._stop_word_ids is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.stop ({self.stop}) is not processed by tokenizer, "
                    "please call the setup method."
                )
            return words + self._stop_word_ids

    def _get_stop_reasons_and_words(self) -> List[Tuple[Union[str, int], List[List[int]]]]:
        stop_reasons = []
        if self.stop_token_ids is not None:
            stop_reasons.extend(self.stop_token_ids)
        if self.stop is not None:
            if isinstance(self.stop, str):
                stop_reasons.append(self.stop)
            else:
                stop_reasons.extend(self.stop)
        stop_words = self._get_stop_words()
        return list(zip(stop_reasons, stop_words))

    def _get_sampling_config(self) -> tllme.SamplingConfig:
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
        fields = {f for f in dir(tllme.SamplingConfig) if not f.startswith("__")}
        unmatched_params = [
            "num_return_sequences",
            "beam_width",
            "n",
            "best_of",
            "use_beam_search",
        ]
        llmapi_to_rt_param_map = {f: getattr(self, f) for f in fields if f not in unmatched_params}
        if self.use_beam_search:
            llmapi_to_rt_param_map["num_return_sequences"] = self.n
            llmapi_to_rt_param_map["beam_width"] = self.best_of
        else:
            llmapi_to_rt_param_map["num_return_sequences"] = self.best_of
            llmapi_to_rt_param_map["beam_width"] = 1

        return tllme.SamplingConfig(**llmapi_to_rt_param_map)

    def _get_output_config(self, is_pytorch_backend: bool = False) -> tllme.OutputConfig:
        sampling_param_fields = set(dir(SamplingParams))
        fields = [
            f
            for f in dir(tllme.OutputConfig)
            if not f.startswith("__") and f in sampling_param_fields
        ]

        config_kwargs = {f: getattr(self, f) for f in fields}

        if is_pytorch_backend:
            config_kwargs["return_log_probs"] = bool(self.logprobs)
            if self.prompt_logprobs and not self.return_context_logits:
                logger.info(
                    "Since prompt_logprobs is requested but return_context_logits is False, "
                    "internally enabling context logits for prompt logprobs computation. "
                    "context logits will be dropped after computation as the user didn't explicitly request them."
                )
                # TODO: Find a more elegant way to do this.
                # NOTE: This is an internal hack, so we can entirely avoid introducing
                # `prompt_logprobs` into the executor bindings and further into
                # model engine / sampler.
                # This is because, prompt_logprobs is a derived quantity from
                # context logits, and the capability to post-compute it
                # already exists in the worker. (see _get_logprobs in worker.py)
                config_kwargs["return_context_logits"] = True
        else:
            config_kwargs["return_log_probs"] = self._return_log_probs

        if config_kwargs.get("additional_model_outputs") is not None:
            config_kwargs["additional_model_outputs"] = [
                tllme.AdditionalModelOutput(name=output_name, gather_context=False)
                for output_name in config_kwargs["additional_model_outputs"]
            ]

        return tllme.OutputConfig(**config_kwargs)

    def _get_guided_decoding_params(self) -> tllme.GuidedDecodingParams:
        if self.guided_decoding is None:
            return None

        if self.guided_decoding.json_object:
            return tllme.GuidedDecodingParams(tllme.GuidedDecodingParams.GuideType.JSON)
        elif self.guided_decoding.json is not None:
            json_schema = self.guided_decoding.json
            if isinstance(json_schema, BaseModel):
                json_schema = json_schema.model_json_schema()
            if isinstance(json_schema, dict):
                json_schema = json.dumps(json_schema)
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.JSON_SCHEMA, json_schema
            )
        elif self.guided_decoding.regex is not None:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.REGEX, self.guided_decoding.regex
            )
        elif self.guided_decoding.grammar is not None:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.EBNF_GRAMMAR, self.guided_decoding.grammar
            )
        elif self.guided_decoding.structural_tag is not None:
            return tllme.GuidedDecodingParams(
                tllme.GuidedDecodingParams.GuideType.STRUCTURAL_TAG,
                self.guided_decoding.structural_tag,
            )
        else:
            return None
