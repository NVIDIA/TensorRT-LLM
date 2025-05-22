import asyncio
import json
import weakref
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Union
from weakref import WeakMethod

import torch

from ..bindings import executor as tllm
from ..disaggregated_params import DisaggregatedParams
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import AsyncQueue, nvtx_range
from ..sampling_params import SamplingParams
from .utils import ErrorResponse, has_event_loop

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
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index (int): The index of the output in the request.
        text (str): The generated output text. Defaults to "".
        token_ids (List[int], optional): The token ids of the generated output text. Defaults to None.
        cumulative_logprob (float, optional): The cumulative log probability of the generated output text. Defaults to None.
        logprobs (List[float], optional): The log probabilities of the top probability words at each position if the logprobs are requested. Defaults to None.
        finish_reason (Literal['stop', 'length', 'timeout', 'cancelled'], optional): The reason why the sequence is finished. Defaults to None.
        stop_reason (int, str, optional): The stop string or token id that caused the completion to stop, None if the completion finished for some other reason. Defaults to None.
        generation_logits (torch.Tensor, optional): The logits on the generated output token ids. Defaults to None.
        disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, optional): Parameters needed for disaggregated serving. Includes the type of request, the first generated tokens, the context request id and the any additional state needing to be transferred from context and generation instances. Defaults to None.

    Attributes:
        length (int): The number of generated tokens.
        token_ids_diff (List[int]): Newly generated token ids.
        logprobs_diff (List[float]): Logprobs of newly generated tokens.
        text_diff (str): Newly generated tokens.
    """
    index: int
    text: str = ""
    token_ids: Optional[List[int]] = None
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[Literal['stop', 'length', 'timeout',
                                    'cancelled']] = None
    stop_reason: Optional[Union[int, str]] = None
    generation_logits: Optional[torch.Tensor] = None
    disaggregated_params: Optional[DisaggregatedParams] = None

    # hidden fields for tracking the diffs
    _last_text_len: int = field(default=0, init=False, repr=False)
    _last_token_ids_len: int = field(default=0, init=False, repr=False)
    _last_logprobs_len: int = field(default=0, init=False, repr=False)
    _incremental_states: Optional[dict] = field(default=None,
                                                init=False,
                                                repr=False)
    # the result of result_handler passed to postprocess workers
    _postprocess_result: Any = None

    def __post_init__(self):
        if self.token_ids is None:
            self.token_ids = []
        if self.logprobs is None:
            self.logprobs = []

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
                 postproc_params: "Optional[PostprocParams]" = None):
        self.id = id
        self.sampling_params = sampling_params
        self.postproc_params = postproc_params
        self.disaggregated_params = None
        self.decoding_iter = 0
        self._done = False

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

    def _handle_sequence(self, finish_reasons, response_tensors,
                         sequence_index):
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

        # In PD, the first generation response will return 2 tokens
        # Skip output the first generated token in generation response
        # TODO: We should have a better way to handle this when enable
        # beam search with PD.
        # Remove, breaks specdec
        # if not self.sampling_params.use_beam_search and \
        #     len(response_tensors.output_token_ids[src_idx]) == 2:
        #     output._last_token_ids_len = 1

        if response_tensors.cum_log_probs is not None:
            output.cumulative_logprob = response_tensors.cum_log_probs[src_idx]
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

        if self._done:
            if finish_reasons[src_idx] == tllm.FinishReason.END_ID:
                output.finish_reason = 'stop'
            elif finish_reasons[src_idx] == tllm.FinishReason.STOP_WORDS:
                output.finish_reason = 'stop'
                for stop_reason, stop_ids in self.sampling_params._get_stop_reasons_and_words(
                ):
                    if output.token_ids[-len(stop_ids):] == stop_ids:
                        output.stop_reason = stop_reason
                        if not self.sampling_params.include_stop_str_in_output:
                            output.token_ids = output.token_ids[:-len(stop_ids)]
                        break
            elif finish_reasons[src_idx] == tllm.FinishReason.LENGTH:
                output.finish_reason = 'length'
            elif finish_reasons[src_idx] == tllm.FinishReason.TIMED_OUT:
                output.finish_reason = 'timeout'
            elif finish_reasons[src_idx] == tllm.FinishReason.CANCELLED:
                pass
            else:
                raise ValueError(
                    f"Unknown finish reason: {finish_reasons[src_idx]}")

    @nvtx_range("handle_response", color="red", category="GenerationResultBase")
    def _handle_response(self, response: Union["PostprocWorker.Output",
                                               tllm.Response, ErrorResponse]):

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
        elif isinstance(response, tllm.Response):
            if response.has_error():
                if self._background_error_handler is not None and (
                        handler := self._background_error_handler()):
                    handler(response.error_msg)

            response_result = response.result
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
                                          beam_idx)
            else:
                self._handle_sequence(finish_reasons, response_result,
                                      response_result.sequence_index)

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

    @nvtx_range("handle_response",
                color="red",
                category="DetokenizedGenerationResultBase")
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
                if hasattr(self.tokenizer, 'decode_incrementally'):
                    if self._streaming and not self.sampling_params.use_beam_search:
                        beam_output.text, beam_output._incremental_states = self.tokenizer.decode_incrementally(
                            beam_output.token_ids_diff,
                            prev_text=beam_output.text,
                            states=beam_output._incremental_states,
                            flush=self._done,
                            **kwargs)
                    else:
                        beam_output.text, _ = self.tokenizer.decode_incrementally(
                            beam_output.token_ids, flush=self._done, **kwargs)
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

    def __init__(self,
                 generation_request: "GenerationRequest",
                 background_error_handler: Optional[Callable] = None,
                 executor: Optional["GenerationExecutor"] = None) -> None:
        super().__init__(
            generation_request.id,
            generation_request.sampling_params,
            background_error_handler,
            postproc_params=generation_request.postproc_params,
        )
        self._generation_request = generation_request
        self._streaming = generation_request.streaming

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
