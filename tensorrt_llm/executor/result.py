import asyncio
import json
import weakref
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import (TYPE_CHECKING, Any, Callable, List, Literal, NamedTuple,
                    Optional, TypeAlias, Union)
from weakref import WeakMethod

import torch
import torch.nn.functional as F

from .._utils import nvtx_range_debug
from ..bindings import executor as tllm
from ..disaggregated_params import DisaggregatedParams
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import AsyncQueue
from ..sampling_params import LogprobParams, SamplingParams
from .utils import ErrorResponse, has_event_loop, is_llm_response

from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from .executor import GenerationExecutor
    from .postproc_worker import PostprocParams, PostprocWorker
    from .request import GenerationRequest

__all__ = [
    "CompletionOutput",
    "GenerationResultBase",
    "DetokenizedGenerationResultBase",
    "GenerationResult",
    "IterationResult",
]


@dataclass(slots=True)
class Logprob:
    """Holds logprob and vocab rank for a token."""
    logprob: float
    rank: Optional[int] = None


# List of token_id_to_Logprob dict for prompt or generation texts
TokenLogprobs: TypeAlias = list[dict[int, Logprob]]


class LogProbsResult(NamedTuple):
    """Optional log probability outputs computed post runtime."""
    prompt: Optional[TokenLogprobs] = None
    generation: Optional[TokenLogprobs] = None


class ResponseWrapper:
    """Wrapper of runtime response with optional outputs computed post runtime.
    """

    def __init__(self,
                 response: Union["PostprocWorker.Output", tllm.Response],
                 logprobs: Optional[LogProbsResult] = None):
        self._response = response
        self.logprobs = logprobs

    @property
    def _is_llm_response(self):
        response = object.__getattribute__(self, '_response')
        return isinstance(response, tllm.Response)

    def __getattr__(self, name):
        response = object.__getattribute__(self, '_response')
        return getattr(response, name)


@dataclass(slots=True)
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index (int): The index of the output in the request.
        text (str): The generated output text. Defaults to "".
        token_ids (List[int], optional): The token ids of the generated output text. Defaults to [].
        cumulative_logprob (float, optional): The cumulative log probability of the generated output text. Defaults to None.
        logprobs (TokenLogprobs, optional): The log probabilities of the top probability words at each position if the logprobs are requested. Defaults to None.
        prompt_logprobs (TokenLogprobs, optional): The log probabilities per prompt token. Defaults to None.
        finish_reason (Literal['stop', 'length', 'timeout', 'cancelled'], optional): The reason why the sequence is finished. Defaults to None.
        stop_reason (int, str, optional): The stop string or token id that caused the completion to stop, None if the completion finished for some other reason. Defaults to None.
        generation_logits (torch.Tensor, optional): The logits on the generated output token ids. Defaults to None.
        disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, optional): Parameters needed for disaggregated serving. Includes the type of request, the first generated tokens, the context request id and the any additional state needing to be transferred from context and generation instances. Defaults to None.
        request_perf_metrics (tensorrt_llm.bindings.executor.RequestPerfMetrics, optional): Performance metrics for the request. Defaults to None.

    Attributes:
        length (int): The number of generated tokens.
        token_ids_diff (List[int]): Newly generated token ids.
        logprobs_diff (List[float]): Logprobs of newly generated tokens.
        text_diff (str): Newly generated tokens.
    """
    index: int
    text: str = ""
    token_ids: Optional[List[int]] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[TokenLogprobs] = field(default_factory=list)
    prompt_logprobs: Optional[TokenLogprobs] = field(default_factory=list)
    finish_reason: Optional[Literal['stop', 'length', 'timeout',
                                    'cancelled']] = None
    stop_reason: Optional[Union[int, str]] = None
    generation_logits: Optional[torch.Tensor] = None
    disaggregated_params: Optional[DisaggregatedParams] = None
    request_perf_metrics: Optional[tllm.RequestPerfMetrics] = None

    # hidden fields for tracking the diffs
    _last_text_len: int = field(default=0, init=False, repr=False)
    _last_token_ids_len: int = field(default=0, init=False, repr=False)
    _last_logprobs_len: int = field(default=0, init=False, repr=False)
    _incremental_states: Optional[dict] = field(default=None,
                                                init=False,
                                                repr=False)
    # the result of result_handler passed to postprocess workers
    _postprocess_result: Any = None

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def text_diff(self) -> str:
        return self.text[self._last_text_len:]

    @property
    def token_ids_diff(self) -> List[int]:
        return self.token_ids[self._last_token_ids_len:]

    @property
    def logprobs_diff(self) -> List[float]:
        return self.logprobs[self._last_logprobs_len:]


class GenerationResultBase:
    ''' This holds the core logic of the GenerationResult class. '''

    def __init__(self,
                 id: int,
                 sampling_params: SamplingParams,
                 background_error_handler: Optional[Callable] = None,
                 postproc_params: "Optional[PostprocParams]" = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None):
        self.id = id
        self.sampling_params = sampling_params
        self.postproc_params = postproc_params
        self.disaggregated_params = None
        self.decoding_iter = 0
        self._done = False
        self.tokenizer = tokenizer


        if has_event_loop():
            self.aqueue = AsyncQueue()
            self.queue = self.aqueue.sync_q
        else:
            self.queue = Queue()
            self.aqueue = None

        # In Sampling mode, the Executor runtime will return best_of sequences
        # in total, which the LLM API will select the n-best sequences among
        # them based on their cumulative log probabilities.
        self._outputs: List[CompletionOutput] = [
            CompletionOutput(i) for i in range(self.sampling_params.best_of)
        ]
        self._context_logits: Optional[torch.Tensor] = None

        self._background_error_handler = None
        if background_error_handler is not None:
            if not isinstance(background_error_handler, WeakMethod):
                self._background_error_handler = WeakMethod(
                    background_error_handler)
            else:
                self._background_error_handler = background_error_handler

        # This is used for avoid duplicate transmission the sampling_params for a
        # request. SamplingParams is necessary for creating dummy
        # GenerationResultBase instances on postprocess worker processes.
        self._params_transmitted = False

    @property
    def outputs(self) -> List[CompletionOutput]:
        sampling_param = self.sampling_params
        if (sampling_param.use_beam_search
                or sampling_param.n == sampling_param.best_of):
            return self._outputs[:sampling_param.n]
        # Pick the top-n outputs, sorted by cumulative log probs.
        sorted_outputs = sorted(
            self._outputs,
            key=lambda x:
            (x.cumulative_logprob
             if x.cumulative_logprob is not None else float('-inf')),
            reverse=True)
        # Reindex the sequence.
        for i, sorted_out in enumerate(sorted_outputs):
            sorted_out.index = i
        return sorted_outputs[:sampling_param.n]

    @property
    def context_logits(self) -> Optional[torch.Tensor]:
        return self._context_logits

    def _check_text_stop_criteria(self, output, stop_reason: str, stop_ids: list) -> bool:
        """Check if the stop text is found in newly generated tokens."""
        now_token_ids_len = len(output.token_ids)
        new_generated_token_ids = output.token_ids[output._last_token_ids_len:now_token_ids_len]
        
        for idx in range(len(new_generated_token_ids)):
            if self.tokenizer is None:
                continue
            new_generated_text = self.tokenizer.decode(
                new_generated_token_ids[idx],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            if stop_reason in new_generated_text:
                output.stop_reason = stop_reason
                if not self.sampling_params.include_stop_str_in_output:
                    output.token_ids = output.token_ids[:output._last_token_ids_len + idx]
                else:
                    output.token_ids = output.token_ids[:output._last_token_ids_len + idx] + stop_ids
                return True
        return False

    def _handle_sequence(self,
                         finish_reasons,
                         response_tensors,
                         sequence_index,
                         logprobs_result=None):
        """ Handle a single sequence in the response. """

        seq_idx = sequence_index
        src_idx = sequence_index if self.sampling_params.use_beam_search else 0

        output = self._outputs[seq_idx]
        output.disaggregated_params = self.disaggregated_params
        output._last_token_ids_len = len(output.token_ids)
        if self.sampling_params.use_beam_search:
            # Beam search enforces returning all generated tokens
            output.token_ids = response_tensors.output_token_ids[src_idx]
        else:
            output.token_ids.extend(response_tensors.output_token_ids[src_idx])

        if response_tensors.cum_log_probs is not None:
            output.cumulative_logprob = response_tensors.cum_log_probs[src_idx]

        if logprobs_result:
            output.prompt_logprobs = logprobs_result.prompt
            output.logprobs = logprobs_result.generation

        if response_tensors.log_probs is not None:
            output._last_logprobs_len = len(output.logprobs)
            output.logprobs = response_tensors.log_probs[src_idx]
            # overcome some WAR in the cpp executor
            if finish_reasons[src_idx] != tllm.FinishReason.CANCELLED:
                assert len(output.logprobs) == output.length
        if response_tensors.generation_logits is not None:
            output.generation_logits = response_tensors.generation_logits[
                src_idx, :output.length]

        # when sampling_params.n > 1 and is cancelled, make sure all the outputs
        # be marked as cancelled.
        if finish_reasons and finish_reasons[
                src_idx] == tllm.FinishReason.CANCELLED:
            output.finish_reason = 'cancelled'

        if response_tensors.request_perf_metrics is not None:
            output.request_perf_metrics = response_tensors.request_perf_metrics

        if self._done:
            if finish_reasons[src_idx] == tllm.FinishReason.END_ID:
                output.finish_reason = 'stop'
            elif finish_reasons[src_idx] == tllm.FinishReason.STOP_WORDS:
                output.finish_reason = 'stop'
                for stop_reason, stop_ids in self.sampling_params._get_stop_reasons_and_words(
                ):
                    if isinstance(stop_reason, str):
                        if self._check_text_stop_criteria(output, stop_reason, stop_ids):
                            break
                    else:
                        if output.token_ids[-len(stop_ids):] == stop_ids:
                            output.stop_reason = stop_reason
                            if not self.sampling_params.include_stop_str_in_output:
                                output.token_ids = output.token_ids[:-len(stop_ids)]
                            break
            elif finish_reasons[src_idx] == tllm.FinishReason.LENGTH:
                output.finish_reason = 'length'
            elif finish_reasons[src_idx] == tllm.FinishReason.TIMED_OUT:
                output.finish_reason = 'timeout'
            # For disaggregated serving, finish reason might be NOT_FINISHED which is ok
            elif finish_reasons[
                    src_idx] == tllm.FinishReason.NOT_FINISHED and self.disaggregated_params is not None and self.disaggregated_params.request_type == "context_only":
                output.finish_reason = 'not_finished'
            elif finish_reasons[src_idx] == tllm.FinishReason.CANCELLED:
                pass
            else:
                raise ValueError(
                    f"Unknown finish reason: {finish_reasons[src_idx]}")

    @nvtx_range_debug("handle_response",
                      color="red",
                      category="GenerationResultBase")
    def _handle_response(self,
                         response: Union["PostprocWorker.Output", tllm.Response,
                                         ResponseWrapper, ErrorResponse]):
        if isinstance(response, ResponseWrapper):
            logprobs_result = response.logprobs
            response = response._response
        else:
            logprobs_result = None

        if isinstance(response, PostprocWorker.Output):
            self._done = response.is_final
            if isinstance(response.res, CompletionOutput):
                # in streaming mode
                self._outputs[0] = response.res
            else:
                self._outputs[0]._postprocess_result = response.res

            if response.error:
                if self._background_error_handler is not None and (
                        handler := self._background_error_handler()):
                    handler(response.error)
        elif is_llm_response(response):
            if response.has_error():
                if self._background_error_handler is not None and (
                        handler := self._background_error_handler()):
                    handler(response.error_msg)

            response_result = response.result
            if hasattr(response_result, "_result"):
                response_result.deserialize()

            self._done = response_result.is_final
            context_phase_params = response_result.context_phase_params
            self.decoding_iter = response_result.decoding_iter
            if context_phase_params is not None:
                self.disaggregated_params = DisaggregatedParams(
                    request_type="context_only",
                    first_gen_tokens=context_phase_params.first_gen_tokens,
                    ctx_request_id=context_phase_params.req_id,
                    opaque_state=context_phase_params.opaque_state,
                    draft_tokens=context_phase_params.draft_tokens)

            finish_reasons = response_result.finish_reasons
            # output_token_ids = (beams, tokens)
            if self.sampling_params.use_beam_search:
                for beam_idx, _ in enumerate(response_result.output_token_ids):
                    self._handle_sequence(finish_reasons, response_result,
                                          beam_idx, logprobs_result)
            else:
                self._handle_sequence(finish_reasons, response_result,
                                      response_result.sequence_index,
                                      logprobs_result)

            if response_result.context_logits is not None:
                self._context_logits = response_result.context_logits

            # Processing background errors here ASAF during generation.
            if self._background_error_handler and (
                    handler := self._background_error_handler()):
                handler()
        elif isinstance(response, ErrorResponse):
            if self._background_error_handler is not None and (
                    handler := self._background_error_handler()):
                handler(response.error_msg)
        else:
            raise ValueError(f"Unknown response type: {response}")


class DetokenizedGenerationResultBase(GenerationResultBase):
    ''' The base class for the generation result with detokenization support. '''
    # import once and avoid cyclic import
    from .postproc_worker import PostprocWorker

    def __init__(self,
                 id: int,
                 sampling_params: SamplingParams,
                 tokenizer: Optional[Callable] = None,
                 streaming: bool = False,
                 background_error_handler: Optional[Callable] = None,
                 postproc_params: Optional["PostprocParams"] = None):
        super().__init__(
            id,
            sampling_params,
            background_error_handler=background_error_handler,
            postproc_params=postproc_params,
        )
        self.tokenizer = tokenizer
        self._streaming = streaming

    def _handle_response(self, response: "GenerationExecutor.Response"):
        GenerationResultBase._handle_response(self, response)

        # The postprocess has been performed, return directly
        if isinstance(response, PostprocWorker.Output):
            return

        kwargs = {
            'skip_special_tokens':
            self.sampling_params.skip_special_tokens,
            'spaces_between_special_tokens':
            self.sampling_params.spaces_between_special_tokens
        }
        if self.sampling_params.detokenize and self.tokenizer is not None:
            for beam_output in self.outputs:
                beam_output._last_text_len = len(beam_output.text)
                if hasattr(
                        self.tokenizer, 'decode_incrementally'
                ) and self._streaming and not self.sampling_params.use_beam_search:
                    beam_output.text, beam_output._incremental_states = self.tokenizer.decode_incrementally(
                        beam_output.token_ids_diff,
                        prev_text=beam_output.text,
                        states=beam_output._incremental_states,
                        flush=self._done,
                        stream_interval=self.sampling_params._stream_interval,
                        **kwargs)
                else:
                    beam_output.text = self.tokenizer.decode(
                        beam_output.token_ids, **kwargs)


# alias
PostprocWorker = DetokenizedGenerationResultBase.PostprocWorker


class GenerationResult(GenerationResultBase):
    '''
    The result of a generation request. It can be used to wait for the completion of the request.

    Args:
        generation_request (GenerationRequest): The generation request object.
        background_error_handler (Callable, optional): The error handler to process the errors from the background threads/processes. Defaults to None.
        executor (GenerationExecutor, optional): The executor that created this result. Defaults to None.
    '''

    def __init__(
        self,
        generation_request: "GenerationRequest",
        background_error_handler: Optional[Callable] = None,
        executor: Optional["GenerationExecutor"] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
        logprob_params: Optional[LogprobParams] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None
    ) -> None:
        super().__init__(
            generation_request.id,
            generation_request.sampling_params,
            background_error_handler,
            postproc_params=generation_request.postproc_params,
            tokenizer=tokenizer
        )
        self._generation_request = generation_request
        self._streaming = generation_request.streaming
        self.disaggregated_params = disaggregated_params
        # minimal sampling params needed for logprob calculation
        self._logprob_params = logprob_params

        # for aborting the request
        self._executor: Optional[weakref.ReferenceType[
            "GenerationExecutor"]] = weakref.ref(executor) if executor else None
        self._aborted = False

    @property
    def request_id(self) -> int:
        return self._generation_request.id

    @property
    def prompt_token_ids(self) -> List[int]:
        return self._generation_request.prompt_token_ids

    def abort(self) -> None:
        """Abort the generation request.
        """
        assert self._executor is not None, "The executor is not set for this result."
        self._executor().abort_request(self.request_id)
        self._aborted = True

    def aborted(self) -> bool:
        """Return whether the generation request is aborted.

        Returns:
            bool: whether the generation request is aborted.
        """
        return self._aborted

    @property
    def finished(self) -> bool:
        return self._done

    def clear_logprob_params(self) -> None:
        # Remove temporary attribute used in executor
        # for a cleaner external-facing output.
        if hasattr(self, "_logprob_params"):
            del self._logprob_params

    def _result_step(self, timeout: Optional[float] = None):
        response = self.queue.get(timeout=timeout)
        self._handle_response(response)

    async def _aresult_step(self):
        assert self.aqueue is not None, "The asyncio event loop was not present during initialization, so async operations are not available."
        response = await self.aqueue.get()
        global_tracer().log_instant("result_step.get")
        self._handle_response(response)

    def result(self, timeout: Optional[float] = None) -> "GenerationResult":
        """Wait for the completion of the request, and return the result.

        Args:
            timeout (float, optional): Timeout. Defaults to None.

        Returns:
            tensorrt_llm.executor.result.GenerationResult: generation result.
        """
        while not self._done:
            self._result_step(timeout)
        return self

    async def aresult(self) -> "GenerationResult":
        """Wait for the completion of the request, and return the result.

        Returns:
            tensorrt_llm.executor.result.GenerationResult: generation result.
        """
        while not self._done:
            await self._aresult_step()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration

        self._result_step()
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        await self._aresult_step()
        return self

    def _exception(self, timeout: Optional[float] = None):
        try:
            self.result(timeout)
        except RuntimeError as e:
            return e

    def _repr_fields(self):
        return [
            'request_id', 'prompt_token_ids', 'outputs', 'finished',
            "context_logits"
        ]

    def __repr__(self) -> str:
        repr = []
        for field in self._repr_fields():
            value = getattr(self, field)
            if isinstance(value, str):
                repr.append(f"{field}={value!r}")
            else:
                repr.append(f"{field}={value}")
        repr = ", ".join(repr)
        repr = f"{self.__class__.__name__}({repr})"
        return repr

    def __hash__(self):
        return hash(self.request_id)


class IterationResult:
    """
    Runtime results for all available iterations.
    """

    def __init__(self):
        self._done = False
        self._timeout = 2

        if has_event_loop():
            self.aqueue = AsyncQueue()
            self.queue = self.aqueue.sync_q
        else:
            self.queue = Queue()
            self.aqueue = None

    def set_timeout(self, timeout: float):
        self._timeout = timeout

    def mark_undone(self):
        # should be called when new prompts are submitted
        self._done = False

    def get_results(self) -> List[dict]:
        """
        Return all runtime results in the queue.
        """
        results = []
        while not self._done:
            try:
                data = self.queue.get(timeout=self._timeout)
                results.append(json.loads(data))
            except Empty:
                self._done = True
        return results

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        assert self.aqueue is not None, "The asyncio event loop was not present during initialization, so async operations are not available."

        try:
            data = await self.aqueue.get(timeout=self._timeout)
            return json.loads(data)
        except asyncio.TimeoutError:
            self._done = True
            raise StopAsyncIteration


def compute_logprobs(
    k_prompt_logprobs: int,
    k_logprobs: int,
    context_logits: Optional[torch.Tensor],
    generation_logits: Optional[torch.Tensor],
    output_token_ids: Optional[list[int]],
) -> LogProbsResult:
    """
    Compute top-K logprobs and ranks for each token position.

    Returns:
        LogProbsResult, a NamedTuple containing:
            - prompt: Optional[List[Dict[token_id, Logprob]]] logprobs for prompt tokens.
            - generation: Optional[List[Dict[token_id, Logprob]]] logprobs for generated tokens.
    """

    def _topk_logprobs(logits: torch.Tensor, top_k: int,
                       tokens: Optional[list[int]]) -> TokenLogprobs:
        if logits.dim() == 3:
            # reshape from [1, T, V] to [T, V]
            logits = logits.squeeze(0)

        if tokens is not None and logits.size(0) > len(tokens):
            # WAR for nvbug 5324291 where TRT backend might return more logits
            # than output tokens.
            logits = logits[:len(tokens)]

        logprobs = F.log_softmax(logits.to("cuda", dtype=torch.float32), dim=-1)
        topk_vals, topk_indices = torch.topk(logprobs, k=top_k, dim=-1)

        results: TokenLogprobs = []
        # for each token position
        for t in range(logprobs.size(0)):
            token_dict = {
                idx.item(): Logprob(logprob=val.item(), rank=r + 1)
                for r, (val,
                        idx) in enumerate(zip(topk_vals[t], topk_indices[t]))
            }

            # If we have the sampled token list and it's not in top-k, add it
            if tokens is not None:
                token_id = tokens[t]
                if token_id not in token_dict:
                    token_logprob = logprobs[t, token_id].item()
                    rank = (logprobs[t] > token_logprob).sum().item() + 1
                    token_dict[token_id] = Logprob(logprob=token_logprob,
                                                   rank=rank)

            results.append(token_dict)
        return results

    prompt_logprobs = _topk_logprobs(
        context_logits, k_prompt_logprobs,
        None) if k_prompt_logprobs and context_logits is not None else None
    generation_logprobs = _topk_logprobs(
        generation_logits, k_logprobs, output_token_ids
    ) if k_logprobs and generation_logits is not None else None

    return LogProbsResult(prompt=prompt_logprobs,
                          generation=generation_logprobs)
