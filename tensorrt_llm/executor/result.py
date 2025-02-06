from dataclasses import dataclass, field
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Union
from weakref import WeakMethod

import torch

from ..bindings import executor as tllm
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import AsyncQueue
from ..sampling_params import SamplingParams
from .utils import has_event_loop

if TYPE_CHECKING:
    from .executor import GenerationExecutor
    from .postproc_worker import PostprocWorker
    from .request import GenerationRequest

__all__ = [
    "GenerationResultBase",
    "DetokenizedGenerationResultBase",
    "GenerationResult",
]


@dataclass(slots=True)
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index (int): The index of the output in the request.
        text (str): The generated output text. Defaults to "".
        token_ids (List[int]): The token ids of the generated output text. Defaults to [].
        cumulative_logprob (float, optional): The cumulative log probability of the generated output text. Defaults to None.
        logprobs (List[float]): The log probabilities of the top probability words at each position if the logprobs are requested. Defaults to [].
        finish_reason (Literal['stop', 'length'], optional): The reason why the sequence is finished. Defaults to None.
        stop_reason (int, str, optional): The stop string or token id that caused the completion to stop, None if the completion finished for some other reason. Defaults to None.
        generation_logits (torch.Tensor, optional): The logits on the generated output token ids. Defaults to None.

    Properties:
        length (int): The number of generated tokens.
        token_ids_diff (List[int]): Newly generated token ids.
        logprobs_diff (List[float]): Logprobs of newly generated tokens.
        text_diff (str): Newly generated tokens.
    """
    index: int
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: List[float] = field(default_factory=list)
    finish_reason: Optional[Literal['stop', 'length']] = None
    stop_reason: Optional[Union[int, str]] = None
    generation_logits: Optional[torch.Tensor] = None

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
    def length(self):
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
                 background_error_handler: Optional[Callable] = None):
        self.id = id
        self.sampling_params = sampling_params
        self._done = False
        self._cancelled = False

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
        self.context_logits: Optional[torch.Tensor] = None

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
        self._postproc_sampling_params_transmitted = False

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

    def handle_sequence(self, response: "GenerationExecutor.Response",
                        sequence_index: int):
        """ Handle a single sequence in the response. """

        tensors = response.tensors
        assert tensors is not None

        beam_search = self.sampling_params.use_beam_search
        seq_idx = sequence_index
        src_idx = sequence_index if beam_search else 0

        output = self._outputs[seq_idx]

        output._last_token_ids_len = len(output.token_ids)
        output.token_ids.extend(tensors.output_token_ids[src_idx])
        if tensors.cum_log_probs is not None:
            output.cumulative_logprob = tensors.cum_log_probs[src_idx]
        if tensors.log_probs is not None:
            output._last_logprobs_len = len(output.logprobs)
            output.logprobs = tensors.log_probs[src_idx]
            assert len(output.logprobs) == output.length
        if tensors.generation_logits is not None:
            output.generation_logits = tensors.generation_logits[
                src_idx, :output.length]

        if self._done:
            if response.finish_reasons[src_idx] == tllm.FinishReason.END_ID:
                output.finish_reason = 'stop'
            elif response.finish_reasons[
                    src_idx] == tllm.FinishReason.STOP_WORDS:
                output.finish_reason = 'stop'
                for stop_reason, stop_ids in self.sampling_params._get_stop_reasons_and_words(
                ):
                    if output.token_ids[-len(stop_ids):] == stop_ids:
                        output.stop_reason = stop_reason
                        if not self.sampling_params.include_stop_str_in_output:
                            output.token_ids = output.token_ids[:-len(stop_ids)]
                        break
            elif response.finish_reasons[src_idx] == tllm.FinishReason.LENGTH:
                output.finish_reason = 'length'

    def handle_response(self, response: Union["GenerationExecutor.Response",
                                              "PostprocWorker.Output"]):

        is_postprocess_res = isinstance(response, PostprocWorker.Output)
        if is_postprocess_res:
            self._done = response.is_final
            if isinstance(response.res, CompletionOutput):
                # in streaming mode
                self._outputs[0] = response.res
            else:
                self._outputs[0]._postprocess_result = response.res

        self._done = response.is_final

        if response.error:
            if self._background_error_handler is not None and (
                    handler := self._background_error_handler()):
                handler(response.error)

        if is_postprocess_res: return

        tensors = response.tensors

        # output_token_ids = (beams, tokens)
        if self.sampling_params.use_beam_search:
            for beam_idx, _ in enumerate(tensors.output_token_ids):
                self.handle_sequence(response, beam_idx)
        else:
            self.handle_sequence(response, response.sequence_index)

        if tensors.context_logits is not None:
            self.context_logits = tensors.context_logits

        # Processing background errors here ASAF during generation.
        if self._background_error_handler and (
                handler := self._background_error_handler()):
            handler()

    @property
    def done(self) -> bool:
        return self._done


class DetokenizedGenerationResultBase(GenerationResultBase):
    ''' The base class for the generation result with detokenization support. '''
    # import once and avoid cyclic import
    from .postproc_worker import PostprocWorker

    def __init__(self,
                 id: int,
                 sampling_params: SamplingParams,
                 tokenizer: Optional[Callable] = None,
                 streaming: bool = False,
                 background_error_handler: Optional[Callable] = None):
        super().__init__(id, sampling_params, background_error_handler)
        self.tokenizer = tokenizer
        self._streaming = streaming

    def handle_response(self, response: "GenerationExecutor.Response"):
        GenerationResultBase.handle_response(self, response)

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
    '''

    def __init__(self,
                 generation_request: "GenerationRequest",
                 background_error_handler: Optional[Callable] = None) -> None:
        super().__init__(generation_request.id,
                         generation_request.sampling_params,
                         background_error_handler)
        self._generation_request = generation_request

    @property
    def request_id(self) -> int:
        return self._generation_request.id

    @property
    def prompt_token_ids(self) -> List[int]:
        return self._generation_request.prompt_token_ids

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def streaming(self):
        return self._generation_request.streaming

    @property
    def generation_request(self) -> "GenerationRequest":
        return self._generation_request

    def result_step(self, timeout: Optional[float] = None):
        response = self.queue.get(timeout=timeout)
        self.handle_response(response)

    async def aresult_step(self):
        assert self.aqueue is not None, "The asyncio event loop was not present during initialization, so async operations are not available."
        response = await self.aqueue.get()
        global_tracer().log_instant("result_step.get")
        self.handle_response(response)

    def result(self, timeout: Optional[float] = None) -> "GenerationResult":
        while not self._done:
            self.result_step(timeout)
        return self

    async def aresult(self) -> "GenerationResult":
        while not self._done:
            await self.aresult_step()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration

        self.result_step()
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        await self.aresult_step()
        return self

    def running(self) -> bool:
        return not self._done

    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self):
        raise NotImplementedError

    def exception(self, timeout: Optional[float] = None):
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
