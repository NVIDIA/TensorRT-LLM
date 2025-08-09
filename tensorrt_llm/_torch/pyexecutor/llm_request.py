from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

import tensorrt_llm.bindings
from tensorrt_llm.bindings import executor as tllm_executor
from tensorrt_llm.executor.result import TokenLogprobs

SamplingConfig = tensorrt_llm.bindings.SamplingConfig
'''
CONTEXT_INIT: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.CONTEXT_INIT: 2>
ENCODER_INIT: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.ENCODER_INIT: 1>
GENERATION_COMPLETE: typing.ClassVar[
    LlmRequestState]  # value = <LlmRequestState.GENERATION_COMPLETE: 5>
GENERATION_IN_PROGRESS: typing.ClassVar[
    LlmRequestState]  # value = <LlmRequestState.GENERATION_IN_PROGRESS: 3>
GENERATION_TO_COMPLETE: typing.ClassVar[
    LlmRequestState]  # value = <LlmRequestState.GENERATION_TO_COMPLETE: 4>
UNKNOWN: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.UNKNOWN: 0>
'''
LlmRequestState = tensorrt_llm.bindings.LlmRequestState
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType

ExecutorRequest = tllm_executor.Request
ExecutorResponse = tllm_executor.Response
ExecutorSamplingConfig = tllm_executor.SamplingConfig
FinishReason = tllm_executor.FinishReason

REQUEST_TYPE_MAPPING = {
    tllm_executor.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION:
    LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
    tllm_executor.RequestType.REQUEST_TYPE_CONTEXT_ONLY:
    LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    tllm_executor.RequestType.REQUEST_TYPE_GENERATION_ONLY:
    LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
}


class LogitsStorage:

    def __init__(self,
                 seq_length: int,
                 use_device_memory=True,
                 should_exclude_last=False):
        if should_exclude_last:
            # Exclude last logits is used when overlap scheduler is used, that generates one extra token,
            # so we should make sure there's memory for that extra +1.
            seq_length += 1
        self.seq_length = seq_length
        self.use_device_memory = use_device_memory
        self._should_exclude_last = should_exclude_last
        self._logits_indices = []

        # Lazily initialized by _init() upon first append()
        self._storage: torch.Tensor | None = None
        self.beam_width = -1
        self.vocab_size = -1

    def _init(self, logits: torch.Tensor):
        _, self.beam_width, self.vocab_size = logits.shape

        if self.use_device_memory:
            self._storage = torch.empty(
                (self.seq_length, self.beam_width, self.vocab_size),
                dtype=logits.dtype,
                device='cuda',
                requires_grad=False)
        else:
            self._storage = torch.empty(
                (self.seq_length, self.beam_width, self.vocab_size),
                dtype=logits.dtype,
                device='cpu',
                pin_memory=True,
                requires_grad=False)

    def append(self, logits: torch.Tensor):
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        assert logits.ndim == 3, f"Bad logits shape, expect [num_tokens, beam_width, vocab_size], got {logits.shape}"

        if self.beam_width == -1:
            self._init(logits)

        assert logits.size(1) == self.beam_width, "Beam width mismatch"

        position = 0 if not self._logits_indices else self._logits_indices[-1][1]
        new_position = logits.size(0) + position
        if new_position > self.seq_length:
            raise ValueError(
                f"LogitsStorage overflow. This storage can only hold {self.seq_length} logits "
                f"({position} already filled) but trying to append {logits.size(0)} more logits"
            )

        self._storage[position:new_position].copy_(logits, non_blocking=True)
        self._logits_indices.append((position, new_position))

    def get(self, all_logits: bool) -> torch.Tensor | None:
        """Returns the used logits storage if there are any, otherwise, returns None.
        When all_logits is True then all set logits are returned, otherwise, only the last logits are returned."""
        try:
            last = -2 if self._should_exclude_last else -1
            start = 0 if all_logits else self._logits_indices[last][0]
            end = self._logits_indices[last][1]
            return self._storage[start:end]
        except IndexError:
            return None

    def set_exclude_last(self, should_exclude_last: bool) -> None:
        self._should_exclude_last = should_exclude_last


class LogProbStorage:
    beam_width: int = -1
    log_probs: list[TokenLogprobs]
    cum_log_probs: list[float]

    def _init(self, first_input: list[TokenLogprobs]):
        self.beam_width = len(first_input)
        self.log_probs = [[] for _ in range(self.beam_width)]
        self.cum_log_probs = [0 for _ in range(self.beam_width)]

    def append(self,
               new_probs: list[TokenLogprobs],
               cum_log_probs: Optional[list[float]] = None):
        """
        new_probs: [beam_width, num_tokens]
        cum_log_probs: [beam_width]
        """
        if self.beam_width == -1:
            self._init(new_probs)

        assert len(new_probs) == self.beam_width, "Beam width mismatch"
        for beam_idx, probs in enumerate(new_probs):
            self.log_probs[beam_idx].extend(probs)
            if cum_log_probs is not None:
                self.cum_log_probs[beam_idx] = cum_log_probs[beam_idx]
            else:
                self.cum_log_probs[beam_idx] += sum(
                    next(iter(prob.values())).logprob for prob in probs)

    def set_log_probs(self, log_probs: list[TokenLogprobs],
                      cum_log_probs: list[float]):
        """
        Reset the storage and refill it with new values
        log_probs: [beam_width, num_tokens]
        cum_log_probs: [beam_width]
        """
        # reinitialize the storage to clear the lists
        self._init(log_probs)
        # append the new values
        self.append(log_probs, cum_log_probs)


class PyResult:
    """PyResult reimplements some features of `bindings.executor.Result` in Python"""

    def __init__(self,
                 prompt_len: int,
                 max_new_tokens: int,
                 use_device_memory=True,
                 streaming=False,
                 return_log_probs: bool = False,
                 return_context_logits: bool = False,
                 return_generation_logits: bool = False,
                 exclude_last_generation_logits: bool = False):
        self._streaming = streaming
        self._context_logits = LogitsStorage(
            prompt_len, use_device_memory) if return_context_logits else None
        self._generation_logits = LogitsStorage(
            max_new_tokens, use_device_memory, exclude_last_generation_logits
        ) if return_generation_logits else None
        self._log_probs = LogProbStorage() if return_log_probs else None

    def append_context_logits(self, context_logits: torch.Tensor):
        if self._context_logits:
            self._context_logits.append(context_logits)

    def append_generation_logits(self, generation_logits: torch.Tensor):
        if self._generation_logits:
            self._generation_logits.append(generation_logits)

    def append_log_probs(self,
                         log_probs: list[TokenLogprobs],
                         cum_log_probs: Optional[list[float]] = None):
        if self._log_probs:
            self._log_probs.append(log_probs, cum_log_probs)

    def set_log_probs(self, log_probs: list[TokenLogprobs],
                      cum_log_probs: list[float]):
        """
        Set log_probs and cum_log_probs to the new values
        log_probs: [beam_width, num_tokens]
        cum_log_probs: [beam_width]
        """
        if self._log_probs:
            self._log_probs.set_log_probs(log_probs, cum_log_probs)

    @property
    def context_logits(self) -> torch.Tensor | None:
        if self._context_logits is None or (storage := self._context_logits.get(
                all_logits=True)) is None:
            return None
        return storage[:, 0]  # remove beam_width axis for context

    @property
    def generation_logits(self) -> torch.Tensor | None:
        # Internal storage: [seq_length, beam_width, vocab_size]
        # API expect: [beam_width, seq_length, vocab_size]
        if not self._generation_logits:
            return None

        storage = self._generation_logits.get(all_logits=not self._streaming)
        if storage is None:
            return None
        return storage.transpose(0, 1)

    @property
    def log_probs(self) -> list[TokenLogprobs] | None:
        return self._log_probs and self._log_probs.log_probs

    @property
    def cum_log_probs(self) -> list[float] | None:
        return self._log_probs and self._log_probs.cum_log_probs


class LlmResult:
    """LlmResult wraps `bindings.executor.Result` but detour some features to Python implementation"""
    py_result_properties = frozenset(
        ('context_logits', 'generation_logits', 'log_probs', 'cum_log_probs'))

    def __init__(self,
                 result: Union[bytes, tensorrt_llm.bindings.executor.Result],
                 py_result: PyResult,
                 is_final: bool = False):
        self._result = result
        self._py_result = py_result
        self.is_final = is_final

    def __getattr__(self, item):
        if item in self.py_result_properties:
            return getattr(self._py_result, item)
        if item == 'is_final':
            return object.__getattribute__(self, 'is_final')
        result = object.__getattribute__(self, '_result')
        return getattr(result, item)

    def deserialize(self):
        self._result = tensorrt_llm.bindings.executor.deserialize_result(
            self._result)

    def get_result(self):
        if tmp_res := tensorrt_llm.bindings.executor.deserialize_result(
                self._result):
            return tmp_res
        return None


@dataclass
class LlmResponse:
    request_id: int
    error_msg: Optional[str] = None
    result: Optional[LlmResult] = None
    client_id: Optional[int] = None

    def has_error(self):
        return self.error_msg is not None


class LlmRequest(tensorrt_llm.bindings.internal.batch_manager.LlmRequest):
    """LlmRequest wraps `bindings.internal.batch_manager.LlmRequest`
    but detour some features to Python implementation"""

    def __init__(
            self,
            *args,
            # Detour handling of some parameters
            client_id: int = None,
            return_log_probs: bool = False,
            return_context_logits: bool = False,
            return_generation_logits: bool = False,
            return_logits_device_memory: bool = True,
            exclude_last_generation_logits: bool = False,
            return_perf_metrics: bool = False,
            stop_words_list: list[list[int]] | None = None,
            llm_request: Optional[
                tensorrt_llm.bindings.internal.batch_manager.LlmRequest] = None,
            is_draft: bool = False,
            seq_slot: Optional[int] = None,
            target_seq_slot: Optional[int] = None,
            **kwargs):

        self.py_logits_post_processors = kwargs.pop("py_logits_post_processors",
                                                    None)
        self.py_lora_path: str | None = kwargs.pop("py_lora_path", None)
        # Multimodal data
        self.py_multimodal_data = kwargs.pop("py_multimodal_data", None)
        if llm_request is not None:
            super().__init__(llm_request)
        else:
            super().__init__(
                *args,
                client_id=client_id,
                return_log_probs=return_log_probs,
                return_context_logits=False,
                return_generation_logits=False,
                return_perf_metrics=return_perf_metrics,
                stop_words_list=torch.tensor(stop_words_list, dtype=torch.int32)
                if stop_words_list else None,
                **kwargs)
        self.py_client_id = client_id
        self.py_request_id = self.request_id
        self.py_llm_request_type = self.llm_request_type
        self.py_end_id = self.end_id
        self.py_prompt_len = self.prompt_len
        self.py_orig_prompt_len = self.orig_prompt_len
        self.py_max_new_tokens = self.max_new_tokens
        self.py_batch_idx = None
        self.py_draft_pages_allocated = 0
        self.py_rewind_len = 0
        self.py_draft_tokens = [] if self.draft_tokens is None else self.draft_tokens
        self.py_last_context_chunk = (None, None)
        self.py_draft_logits = None
        self.py_target_probs = None
        self.py_last_draft_tokens = None
        self.py_num_accepted_draft_tokens = 0
        self.py_decoding_iter = 0
        self.is_attention_dp_dummy = False
        self.is_cuda_graph_dummy = False
        self.py_lora_task_layer_module_configs = None

        self.py_return_log_probs = return_log_probs
        self.py_return_context_logits = return_context_logits
        self.py_return_generation_logits = return_generation_logits
        self.py_return_logits_device_memory = return_logits_device_memory
        self.py_is_draft = is_draft
        # The request's sequence slot ID, an index between 0 (inclusive) and max_batch_size (exclusive).
        self.py_seq_slot = seq_slot
        # If the request is a draft request, target_seq_slot is the sequence slot ID of its target request.
        self.py_target_seq_slot = target_seq_slot

        # TODO: remove this when use DynamicDecodeOp in pytorch flow.
        # currently, keep py_stop_words_list as python list, rather than tensor.
        self.py_stop_words_list = stop_words_list

        self.py_result = PyResult(self.py_prompt_len, self.py_max_new_tokens,
                                  return_logits_device_memory, self.streaming,
                                  return_log_probs, return_context_logits,
                                  return_generation_logits,
                                  exclude_last_generation_logits)
        self.child_requests = []

        self._py_embedding_bias_1d = None
        if hasattr(self, 'embedding_bias') and self.embedding_bias is not None:
            # Pre-squeeze to 1D if needed (remove batch dimension)
            if self.embedding_bias.dim() > 1:
                self._py_embedding_bias_1d = self.embedding_bias.squeeze(0)
            else:
                self._py_embedding_bias_1d = self.embedding_bias

    def is_generation_only_request(self):
        return self.py_llm_request_type == LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY

    def create_response(
            self,
            use_fast_logits=False,
            mpi_world_rank=0) -> tensorrt_llm.bindings.executor.Response | None:
        result, is_final = super().create_serialized_result(
            use_fast_logits, mpi_world_rank)
        return LlmResponse(
            request_id=self.py_request_id
            if self.is_child else self.parent_request_id,
            result=LlmResult(result, self.py_result, is_final),
            client_id=self.py_client_id) if len(result) > 0 else None

    @property
    def is_dummy(self):
        return self.is_attention_dp_dummy or self.is_cuda_graph_dummy or self.is_dummy_request

    def finish_by(self, reason: FinishReason, beam: int) -> None:
        """CPP finish by reason does not support beam_width > 1"""
        self.state = LlmRequestState.GENERATION_COMPLETE
        self.set_finished_reason(reason, beam)

    def create_child_request(self, child_id):
        child = super().create_child_request(child_id)
        py_request = LlmRequest(llm_request=child)

        # Copy all py_* attributes from parent to child
        for attr_name, attr_value in self.__dict__.items():
            if attr_name.startswith('py_'):
                attr_value = getattr(self, attr_name)
                setattr(py_request, attr_name, deepcopy(attr_value))
            elif attr_name in ['is_attention_dp_dummy', 'is_cuda_graph_dummy']:
                setattr(py_request, attr_name, attr_value)

        # Rewrite specific attributes that should use child_request values.
        py_request.py_request_id = child.request_id
        py_request.py_batch_idx = None
        py_request.py_seq_slot = None

        py_request.child_requests = []

        assert py_request.is_child
        assert py_request.request_id == child.request_id
        assert py_request.parent_request_id == self.request_id
        assert py_request.sampling_config.random_seed != self.sampling_config.random_seed

        self.child_requests.append(py_request)


def convert_wordlist(word_list) -> List[List[int]]:
    """Converts a wordlist from format:

    [[word_0 token_0, word_0 token_1, ...], [word_1 token_0, ...], ...]]

    into the TRTLLM word list format. The TRTLLM format expects a list of tokens,
    and an inclusive prefix sum. A word_list (either bad_words or stop_words) is
    a list that encodes the list of words that have to be banned / trigger the stop from generated sequences.
    Its shape is [2, badWordsLength], as explained below, or [batchSize, 2, badWordsLength]
    when there is a different list for each sequence in the batch.

    The badWordsList and stopWordsList tensors have the same shape [2, length]. Let's consider an example with three words to describe the
    representation of those lists.  The first word contains tokens [5, 7, 3], the
    second one contains [9, 2] and the third one is composed of tokens [6, 2, 4, 1]. In total, there are 9 tokens. That's the length. The shape of the tensor
    is [2, 9].  The first row of the tensor must contain the 9 token IDs and the
    second row must store the inclusive prefix-sum of the word lengths as shown on the following diagram:

        0           3       5              9
        |           |       |              |
        V           V       V              V
    [  5,  7,  3,  9,  2,  6,  2,  4,  1]
    [  3,  5,  9, -1, -1, -1, -1, -1, -1]
    """
    if not word_list:
        return []
    tokens = []
    offsets = []
    current_offset = 0
    for word_tokens in word_list:
        tokens.extend(word_tokens)
        current_offset += len(word_tokens)
        offsets.append(current_offset)
    if len(tokens) > len(offsets):
        offsets.extend([-1] * (len(tokens) - len(offsets)))
    return [tokens, offsets]


def executor_request_to_llm_request(
        req_id: int,
        executor_request: ExecutorRequest,
        child_req_ids: List[int],
        exclude_last_generation_logits: bool,
        input_token_ids: Optional[List] = None) -> LlmRequest:
    executor_sampling_config = executor_request.sampling_config
    sampling_config = SamplingConfig(executor_sampling_config)

    input_tokens = input_token_ids if input_token_ids is not None else executor_request.input_token_ids

    llm_request_type = REQUEST_TYPE_MAPPING[executor_request.request_type]
    stop_words_list = convert_wordlist(
        executor_request.stop_words) if executor_request.stop_words else None

    # Extract multimodal fields from executor request
    multimodal_hashes = None
    multimodal_positions = None
    multimodal_lengths = None
    if executor_request.multimodal_input is not None:
        multimodal_hashes = executor_request.multimodal_input.multimodal_hashes
        multimodal_positions = executor_request.multimodal_input.multimodal_positions
        multimodal_lengths = executor_request.multimodal_input.multimodal_lengths

    # Extract mrope fields
    mrope_rotary_cos_sin = None
    mrope_position_deltas = None
    if executor_request.mrope_config is not None:
        mrope_rotary_cos_sin = executor_request.mrope_config.mrope_rotary_cos_sin
        mrope_position_deltas = executor_request.mrope_config.mrope_position_deltas

    llm_request = LlmRequest(
        request_id=req_id,
        max_new_tokens=executor_request.max_tokens,
        input_tokens=input_tokens,
        sampling_config=sampling_config,
        is_streaming=executor_request.streaming,
        end_id=executor_request.end_id,
        pad_id=executor_request.pad_id,
        embedding_bias=executor_request.embedding_bias,
        bad_words_list=torch.tensor(
            convert_wordlist(executor_request.bad_words), dtype=torch.int32)
        if executor_request.bad_words else None,
        stop_words_list=stop_words_list,
        prompt_embedding_table=None if executor_request.prompt_tuning_config
        is None else executor_request.prompt_tuning_config.embedding_table,
        prompt_vocab_size=None if executor_request.prompt_tuning_config is None
        else executor_request.prompt_tuning_config.embedding_table.shape[0],
        multimodal_hashes=multimodal_hashes,
        multimodal_positions=multimodal_positions,
        multimodal_lengths=multimodal_lengths,
        multimodal_embedding=executor_request.multimodal_embedding,
        lora_task_id=executor_request.lora_config.task_id
        if executor_request.lora_config is not None else None,
        lora_weights=executor_request.lora_config.weights
        if executor_request.lora_config is not None else None,
        lora_config=executor_request.lora_config.config
        if executor_request.lora_config is not None else None,
        py_lora_path=getattr(executor_request, "py_lora_path", None),
        mrope_rotary_cos_sin=mrope_rotary_cos_sin,
        mrope_position_deltas=mrope_position_deltas,
        lookahead_config=None,
        return_log_probs=executor_request.output_config.return_log_probs,
        return_context_logits=executor_request.output_config.
        return_context_logits,
        return_perf_metrics=executor_request.output_config.return_perf_metrics,
        return_generation_logits=executor_request.output_config.
        return_generation_logits,
        exclude_last_generation_logits=exclude_last_generation_logits,
        draft_tokens=getattr(executor_request, "draft_tokens", None),
        draft_logits=None,
        exclude_input_from_output=executor_request.output_config.
        exclude_input_from_output,
        logits_post_processor=None,
        apply_logits_post_processor_batched=False,
        guided_decoding_params=executor_request.guided_decoding_params,
        py_logits_post_processors=getattr(executor_request,
                                          "py_logits_post_processors", None),
        encoder_input_tokens=None,
        return_encoder_output=False,
        client_id=executor_request.client_id
        if executor_request.client_id is not None else req_id,
        priority=0.5,
        llm_request_type=llm_request_type,
        context_phase_params=executor_request.context_phase_params,
        py_multimodal_data=getattr(executor_request, "py_multimodal_data",
                                   None))
    if child_req_ids:
        for child_id in child_req_ids:
            llm_request.create_child_request(child_id)

    return llm_request


def get_draft_token_length(request: LlmRequest) -> int:
    """Get the length of draft tokens for a given request.

    Args:
        request: The LlmRequest to get draft token length for

    Returns:
        The number of draft tokens, or 0 if no draft tokens exist
    """
    if request.py_draft_tokens is not None:
        return len(request.py_draft_tokens)
    return 0
